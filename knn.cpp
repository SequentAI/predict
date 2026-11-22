#include <algorithm>
#include "predictor.cpp"
#include "view.cpp"

class kNN : public Predictor {
  public:
    // [credit_deps] makes the dependencies of each neighbour gain relevance
    kNN(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, bool credit_deps);

    void learn(long from, long to);
    bool forget(long from);

  protected:
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;

  private:
    Tfidf tfidf;
    bool credit_deps;
};

kNN::kNN(const LVecVec& deps, const LVecVec& syms,
  const LVecVec& sym_ths, long sym_num, bool credit_deps)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num),
  credit_deps(credit_deps) {
}

void kNN::learn(long from, long to) {
  tfidf.resize(sym_ths.size());
  for (long i = from; i < to; ++i)
    tfidf.add(syms[i]);
}

bool kNN::forget(long from) {
  for (long i = from; i < (long)syms.size(); ++i)
    tfidf.remove(syms[i]);
  return true;
}

inline double age(long k) {
  return 500000000 - 100 * k;
}

// The k-NN ranking: [ans] holds (i, 0) for each theorem i, [neighbours] the
// similarity to the query of each theorem that may be predicted;
// [deps_of](i) gives the dependencies of theorem i, and [visible](i) whether
// it may be predicted.
template <class DepsOf, class Visible>
void credit_neighbours(LDPairVec& ans, LDPairVec neighbours, long no_adv,
  bool credit_deps, DepsOf deps_of, Visible visible) {
  // neighbours[0, sorted) are the best neighbours in order; the loop below
  // usually needs few of them, so they are sorted on demand
  const long n_neighbours = neighbours.size();
  long sorted = 0;
  for (auto& a : ans) a.second = 0;


  // In typical k-NN implementation the number 'k' is fixed. Unfortunately for
  // premise selection, using a wrong number may have quite bad implications,
  // namely the neighbours may all be very similar and have very similiar dependencies.
  // This may mean that we will not have enough non-zero entries to fill no-adv predictions.
  // This is why we use an adaptive 'k'. We count how many premises we recommended,
  // and we continue until this number is enough.
  // Since we want to sort the list of recommended answers only once, we want to make
  // sure that items recommended by a certain 'k' are before those coming for the increased
  // k. In order to do this, we use an "age" that is added to non-zero predictions
  // and decreases when k increases.

  long no_recommends = 0;
  for (long k = 0; k < n_neighbours && no_recommends < no_adv; ++k) {
    if (k == sorted) {
      sorted = min(n_neighbours, max(2 * sorted, no_adv));
      partial_sort(neighbours.begin() + k, neighbours.begin() + sorted,
        neighbours.end(), sortfun);
    }
    long nn = neighbours[k].first;
    double o = neighbours[k].second; // distance, we do not use sqrt
    if (ans[nn].second <= 0) {
      no_recommends++;
      ans[nn].second = age(k) + o;
    }
    else
      ans[nn].second += o;

    // dependencies of the neighbor also gain some relevance, depending on the
    // number of dependencies of the current theorem
    if (!credit_deps) continue;
    const LVec& ds = deps_of(nn);
    if (ds.empty()) continue;
    double ol = 2.7 * o / ds.size();
    for (const auto& d : ds) {
      if (visible(d)) { // in case of future dependencies, do not predict them
        if (ans[d].second <= 0) {
          no_recommends++;
          ans[d].second = age(k) + ol;
        }
        else
          ans[d].second += ol;
      }
    }
  }
}

LDPairVec kNN::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec ans = LDPairVec(maxth, make_pair(0, 0));

  // initialise theorem importance matrix
  for (long i = 0; i < maxth; ++i) {
    ans[i].first = i; ans[i].second = 0;
  }

  // for each symbol, increase the importance of the theorems which contain
  // the symbol by a given symbol weight
  for (const auto& sym : csyms) {
    const LVec& ths = sym_ths[sym];
    double weight = tfidf.get(sym);
    for (const auto& th : ths) {
      if (visible(th, maxth)) ans[th].second += pow(weight, 6);
    }
  }

  LDPairVec neighbours(ans);
  keep_visible(neighbours, maxth);
  credit_neighbours(ans, move(neighbours), no_adv, credit_deps,
    [&](long i) -> const LVec& { return deps[i]; },
    [&](long i) { return visible(i, maxth); });

  sort_prediction(ans, keep_visible(ans, no_adv));
  return ans;
}

// kNN's ranking of the theorems of a view, computed as kNN computes it in a
// process started on them: with the view's IDF, crediting a neighbour's
// dependencies that are theorems of the view.
class kNNView : public ViewPredictor {
  public:
    kNNView(const View& view, long sym_num, bool credit_deps);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;

  private:
    const View& view;
    bool credit_deps;
    Tfidf tfidf;  // by rank
    LVec ids;     // View::id_table, if credit_deps
};

kNNView::kNNView(const View& view, long sym_num, bool credit_deps)
: view(view), credit_deps(credit_deps), tfidf(view.features()),
  ids(credit_deps ? view.id_table() : LVec()) {
  const LVec table = view.rank_table(sym_num);
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(ranks_in(s, table));
}

LDPairVec kNNView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  const auto visible = [&](long i) { return !in_scope || (*in_scope)[i]; };
  const long n = view.size();
  LDPairVec ans(n);
  for (long i = 0; i < n; ++i) ans[i] = make_pair(i, 0.0);

  // as in kNN::predict, a feature's weight added in query order
  for (const auto& fw : query) {
    const double weight = tfidf.get(view.rank_of(fw.first));
    for (size_t v = 0; v < view.versions.size(); ++v) {
      const auto ths = view.versions[v]->theorems_of(fw.first);
      for (auto th = ths.first; th != ths.second; ++th) {
        const long i = view.first[v] + *th;
        if (visible(i)) ans[i].second += pow(weight, 6);
      }
    }
  }

  LDPairVec neighbours(ans);
  keep_in_scope(neighbours, in_scope, n);
  credit_neighbours(ans, move(neighbours), no_adv, credit_deps,
    [&](long i) { return view.deps_of(i, ids); }, visible);

  sort_prediction(ans, keep_in_scope(ans, in_scope, no_adv));
  return ans;
}
