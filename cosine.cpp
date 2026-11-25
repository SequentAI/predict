#include <algorithm>
#include "predictor.cpp"
#include "view.cpp"

// Ranks theorems by the cosine between the tf-idf vectors of the query and of
// the theorem: a query feature f with weight w has coordinate w idf_f, and a
// theorem's feature has coordinate idf_f. Theorems sharing no feature with
// the query score 0. Unless [dense], the dot products are summed over the
// theorems containing each query feature (sym_ths) rather than over all
// theorems; both add the same terms in the same (feature) order, so they
// compute the same scores.
class Cosine : public Predictor {
  public:
    // [sorted_syms] gives the rows of [syms] sorted for some first theorems
    // (Corpus::sorted_syms); Cosine sorts the others.
    Cosine(const Rows& deps, const Rows& syms, const Rows& sym_ths,
      const Rows& sorted_syms, long sym_num, bool dense);

    void learn(long from, long to);
    bool forget(long from);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    // syms[i] in increasing feature order
    Row sorted(long i) const;
    // the squared norm of theorem i's vector
    double norm2(long i) const;

    Tfidf tfidf;
    bool dense;
    const Rows& sorted_syms;
    // sorted(i) for i from own_from on, beyond sorted_syms
    long own_from = 0;
    vector<vector<int32_t> > own_sorted;
};

Cosine::Cosine(const Rows& deps, const Rows& syms, const Rows& sym_ths,
  const Rows& sorted_syms, long sym_num, bool dense)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num), dense(dense),
  sorted_syms(sorted_syms) {}

Row Cosine::sorted(long i) const {
  if (i < own_from) return sorted_syms[i];
  const auto& s = own_sorted[i - own_from];
  return Row(s.data(), s.size());
}

double Cosine::norm2(long i) const {
  double n = 0;
  for (const auto f : sorted(i)) {
    const double idf = tfidf.get(f);
    n += idf * idf;
  }
  return n;
}

void Cosine::learn(long from, long to) {
  tfidf.resize(sym_ths.size());
  // the corpus may have dropped its sorted rows since
  if (own_from != sorted_syms.size()) {
    own_from = sorted_syms.size();
    own_sorted.clear();
  }
  for (long i = own_from + own_sorted.size(); i < syms.size(); ++i) {
    const Row s = syms[i];
    own_sorted.emplace_back(s.begin(), s.end());
    sort(own_sorted.back().begin(), own_sorted.back().end());
  }
  if (from == 0 && to == (long)syms.size()) {
    // every theorem: a feature's frequency is its number of theorems, read
    // without the theorems' rows
    vector<long> f(sym_ths.size());
    for (long s = 0; s < (long)f.size(); ++s) f[s] = sym_ths[s].size();
    tfidf.start(to, f);
    return;
  }
  for (long i = from; i < to; ++i)
    tfidf.add(syms[i]);
}

bool Cosine::forget(long from) {
  for (long i = from; i < (long)syms.size(); ++i)
    tfidf.remove(syms[i]);
  own_sorted.resize(max(0L, from - own_from));
  return true;
}

LDPairVec Cosine::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto f : csyms) query.push_back(make_pair(f, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec Cosine::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  // query features in increasing order with their terms w idf^2 in the dot
  // product. Scaling every weight alike leaves the cosine unchanged, so the
  // weights are divided by the largest one of a feature with nonzero IDF (the
  // others' coordinates are 0), to keep w^2 from overflowing or underflowing.
  LDPairVec terms(query);
  sort(terms.begin(), terms.end());
  double wmax = 0;
  for (const auto& ft : terms)
    if (tfidf.get(ft.first) != 0)
      wmax = max(wmax, ft.second);
  double qnorm2 = 0;
  for (auto& ft : terms) {
    const double idf = tfidf.get(ft.first);
    const double w = idf != 0 && wmax > 0 ? ft.second / wmax : 0;
    qnorm2 += w * w * (idf * idf);
    ft.second = w * (idf * idf);
  }

  LDPairVec ans(maxth);
  for (long i = 0; i < maxth; ++i)
    ans[i] = make_pair(i, 0.0);

  if (dense) {
    for (long i = 0; i < maxth; ++i) {
      auto q = terms.begin();
      for (const auto f : sorted(i)) {
        while (q != terms.end() && q->first < f) ++q;
        if (q != terms.end() && q->first == f) ans[i].second += q->second;
      }
    }
  }
  else
    for (const auto& ft : terms)
      for (const auto i : sym_ths[ft.first])
        if (visible(i, maxth)) ans[i].second += ft.second;

  // a theorem's norm only where it is needed: idf changes with every
  // learned theorem, so every norm changes
  for (long i = 0; i < maxth; ++i) {
    const double n2 = ans[i].second != 0 && qnorm2 != 0 ? norm2(i) : 0;
    if (n2 != 0)
      ans[i].second /= sqrt(qnorm2 * n2);
    else
      ans[i].second = 0;
  }

  sort_prediction(ans, keep_visible(ans, no_adv));
  return ans;
}

// Cosine's ranking of the theorems of a view, computed as Cosine computes it
// in a process started on them: with the view's IDF, numbering features and
// adding them up by their rank in the view.
class CosineView : public ViewPredictor {
  public:
    CosineView(const View& view, bool dense, long sym_num);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;

  private:
    // the squared norm of the vector of a theorem with features of ranks
    // [ranks]
    double norm2(LVec ranks) const;

    const View& view;
    bool dense;
    Tfidf tfidf;  // by rank
    // norm2 of each theorem of the view
    vector<double> norms2;
};

CosineView::CosineView(const View& view, bool dense, long sym_num)
: view(view), dense(dense), tfidf(view.features()) {
  const LVec table = view.rank_table(sym_num);
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(ranks_in(s, table));
  if (!dense)
    for (const auto v : view.versions)
      for (const auto& s : v->syms) norms2.push_back(norm2(ranks_in(s, table)));
}

double CosineView::norm2(LVec ranks) const {
  sort(ranks.begin(), ranks.end());
  double n = 0;
  for (const auto r : ranks) {
    const double idf = tfidf.get(r);
    n += idf * idf;
  }
  return n;
}

LDPairVec CosineView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  // as in Cosine::predict_weighted, with features in rank order
  struct Term {
    long f, r;
    double w;
  };
  vector<Term> terms;
  for (const auto& fw : query)
    terms.push_back(Term{fw.first, view.rank_of(fw.first), fw.second});
  sort(terms.begin(), terms.end(),
    [](const Term& a, const Term& b) { return a.r < b.r; });
  double wmax = 0;
  for (const auto& t : terms)
    if (tfidf.get(t.r) != 0)
      wmax = max(wmax, t.w);
  double qnorm2 = 0;
  for (auto& t : terms) {
    const double idf = tfidf.get(t.r);
    const double w = idf != 0 && wmax > 0 ? t.w / wmax : 0;
    qnorm2 += w * w * (idf * idf);
    t.w = w * (idf * idf);
  }

  const long n = view.size();
  LDPairVec ans(n);
  for (long i = 0; i < n; ++i)
    ans[i] = make_pair(i, 0.0);

  if (dense) {
    for (long i = 0; i < n; ++i) {
      const auto at = view.locate(i);
      const LVec& syms = at.first->syms[at.second];
      // once for each time the theorem has the feature, as Cosine does
      for (const auto& t : terms)
        for (const auto f : syms)
          if (f == t.f) ans[i].second += t.w;
    }
  }
  else
    for (const auto& t : terms)
      for (size_t v = 0; v < view.versions.size(); ++v) {
        const auto ths = view.versions[v]->theorems_of(t.f);
        for (auto th = ths.first; th != ths.second; ++th) {
          const long i = view.first[v] + *th;
          if (!in_scope || (*in_scope)[i]) ans[i].second += t.w;
        }
      }

  for (long i = 0; i < n; ++i) {
    double n2 = 0;
    if (!dense) n2 = norms2[i];
    else {
      const auto at = view.locate(i);
      LVec ranks;
      for (const auto f : at.first->syms[at.second])
        ranks.push_back(view.rank_of(f));
      n2 = norm2(ranks);
    }
    if (ans[i].second != 0 && qnorm2 != 0 && n2 != 0)
      ans[i].second /= sqrt(qnorm2 * n2);
    else
      ans[i].second = 0;
  }

  if (in_scope)
    ans.erase(remove_if(ans.begin(), ans.end(),
      [&](const pair<long, double>& p) { return !(*in_scope)[p.first]; }),
      ans.end());
  sort_prediction(ans, min(no_adv, (long)ans.size()));
  return ans;
}
