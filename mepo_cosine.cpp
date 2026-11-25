#include <algorithm>
#include "predictor.cpp"
#include "view.cpp"

inline double sq(const double& x) { return x * x; }

// [key](f) is the number of feature f of syms2 in syms1 and tfidf
template <class Syms, class Key>
double cosine(const LDMap &syms1, const Syms &syms2,
              const Tfidf &tfidf, Key key){
  double sig = 0, sig1 = 0, sig2 = 0;
  for (auto s : syms1)
    sig1 += sq(tfidf.get(s.first)) * sq(s.second);
  for (auto f : syms2) {
    const long s = key(f);
    const double tw = tfidf.get(s);
    sig2 += sq(tw);
    auto got = syms1.find(s);
    if (got != syms1.end())
      sig += sq(tw) * got->second;
  }
  if (sig1 == 0 || sig2 == 0) return 0;
  return (sq(sig) / (sig1 * sig2));
  // for jaccard use:
  // return (sig / (sig1 + sig2 + sig));

}

double euclid(const LDMap &syms1, const LVec &syms2, const Tfidf &tfidf) {
  double sig = 0;

  for (auto s : syms2)
    if (syms1.find(s) == syms1.end())
      sig += sq(tfidf.get(s));
  for (const auto& s : syms1)
    if (find(syms2.begin(), syms2.end(), s.first) == syms2.end())
      sig += sq(tfidf.get(s.first)) * sq(s.second);

  return (1 / (1 + sig));
}



// theorems ranked in each round of MePoCosine
const long mepo_cosine_incr = 100;

class MePoCosine : public Predictor {
  public:
    MePoCosine(const Rows& deps, const Rows& syms, const Rows& sym_ths,
      long sym_num);

    void learn(long from, long to);

  protected:
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;

  private:
    Tfidf tfidf;
};

MePoCosine::MePoCosine(const Rows& deps, const Rows& syms,
  const Rows& sym_ths, long sym_num)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num) {
}

void MePoCosine::learn(long from, long to) {
  for (long i = from; i < to; ++i)
    tfidf.add(syms[i]);
}


// MePo's iterative cosine ranking of the theorems in [ans], (i, 0) for each
// theorem i that may be predicted: [syms_of](i) gives the features of theorem
// i, and [key](f) the number of feature f in the query [csyms] and [tfidf].
template <class SymsOf, class Key>
LDPairVec mepo_cosine_rank(LDPairVec ans, const LVec& csyms, long no_adv,
  const Tfidf& tfidf, SymsOf syms_of, Key key) {
  const long n = ans.size();

  LDMap nsyms;
  for (auto s : csyms)
    nsyms[s] = 1.0;

  for (long sofar = 0; sofar < no_adv; sofar += mepo_cosine_incr) {
    for (long i = sofar; i < n; ++i)
      ans[i].second = cosine(nsyms, syms_of(ans[i].first), tfidf, key);

    long until = min({sofar + mepo_cosine_incr, no_adv, n});

    partial_sort(ans.begin() + sofar, ans.begin() + until, ans.end(), sortfun);

    if (until >= n) return ans;

    for (long i = sofar; i < until; i++)
      for (auto f : syms_of(ans[i].first)) {
        const long s = key(f);
        nsyms[s] = max(nsyms[s], ans[i].second);
      }
  }

  return ans;
}

LDPairVec MePoCosine::predict(const LVec& csyms, long maxth, long no_adv) const {
  // theorem importance matrix
  LDPairVec ans;
  for (long i = 0; i < maxth; ++i)
    if (visible(i, maxth)) ans.push_back(make_pair(i, 0.0));
  return mepo_cosine_rank(move(ans), csyms, no_adv, tfidf,
    [&](long i) { return syms[i]; }, [](long f) { return f; });
}

// MePoCosine's ranking of the theorems of a view, computed as MePoCosine
// computes it in a process started on them: with the view's IDF, numbering
// features by their rank in the view.
class MePoCosineView : public ViewPredictor {
  public:
    MePoCosineView(const View& view, long sym_num);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;

  private:
    const View& view;
    LVec ranks;   // View::rank_table
    Tfidf tfidf;  // by rank
};

MePoCosineView::MePoCosineView(const View& view, long sym_num)
: view(view), ranks(view.rank_table(sym_num)), tfidf(view.features()) {
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(ranks_in(s, ranks));
}

LDPairVec MePoCosineView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  LDPairVec ans;
  for (long i = 0; i < view.size(); ++i)
    if (!in_scope || (*in_scope)[i]) ans.push_back(make_pair(i, 0.0));
  LVec csyms;
  for (const auto& fw : query) csyms.push_back(view.rank_of(fw.first));
  return mepo_cosine_rank(move(ans), csyms, no_adv, tfidf,
    [&](long i) -> const LVec& {
      const auto at = view.locate(i);
      return at.first->syms[at.second];
    },
    [&](long f) { return ranks[f]; });
}
