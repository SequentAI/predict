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
    Cosine(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, bool dense);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    // the squared norm of theorem i's vector
    double norm2(long i) const;

    Tfidf tfidf;
    bool dense;
    // syms[i] in increasing feature order
    LVecVec sorted_syms;
    // norm2 of the learned theorems
    vector<double> norms2;
};

Cosine::Cosine(const LVecVec& deps, const LVecVec& syms,
  const LVecVec& sym_ths, long sym_num, bool dense)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num), dense(dense),
  sorted_syms(syms) {
  for (auto& s : sorted_syms) sort(s.begin(), s.end());
}

double Cosine::norm2(long i) const {
  double n = 0;
  for (const auto f : sorted_syms[i]) {
    const double idf = tfidf.get(f);
    n += idf * idf;
  }
  return n;
}

void Cosine::learn(long from, long to) {
  for (long i = from; i < to; ++i)
    tfidf.add(syms[i]);

  // idf changes with every learned theorem, so every norm changes
  if (!dense) {
    norms2.resize(to);
    for (long i = 0; i < to; ++i)
      norms2[i] = norm2(i);
  }
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
      for (const auto f : sorted_syms[i]) {
        while (q != terms.end() && q->first < f) ++q;
        if (q != terms.end() && q->first == f) ans[i].second += q->second;
      }
    }
  }
  else
    for (const auto& ft : terms)
      for (const auto i : sym_ths[ft.first])
        if (visible(i, maxth)) ans[i].second += ft.second;

  for (long i = 0; i < maxth; ++i) {
    const double n2 = dense ? norm2(i) : norms2[i];
    if (ans[i].second != 0 && qnorm2 != 0 && n2 != 0)
      ans[i].second /= sqrt(qnorm2 * n2);
    else
      ans[i].second = 0;
  }

  sort_prediction(ans, keep_visible(ans, no_adv));
  return ans;
}

// Cosine's ranking of the theorems of a view, computed as Cosine computes it
// in a process started on them: with the view's IDF, adding up features in
// the view's order.
class CosineView : public ViewPredictor {
  public:
    CosineView(const View& view, bool dense);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;

  private:
    // the squared norm of the vector of a theorem with features [syms]
    double norm2(const LVec& syms) const;

    const View& view;
    bool dense;
    Tfidf tfidf;
    // norm2 of each theorem of the view
    vector<double> norms2;
};

CosineView::CosineView(const View& view, bool dense)
: view(view), dense(dense), tfidf(view.rank.size()) {
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(s);
  if (!dense)
    for (const auto v : view.versions)
      for (const auto& s : v->syms) norms2.push_back(norm2(s));
}

double CosineView::norm2(const LVec& syms) const {
  LVec in_order(syms);
  sort(in_order.begin(), in_order.end(),
    [&](long f, long g) { return view.rank[f] < view.rank[g]; });
  double n = 0;
  for (const auto f : in_order) {
    const double idf = tfidf.get(f);
    n += idf * idf;
  }
  return n;
}

LDPairVec CosineView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  // as in Cosine::predict_weighted, with features in the view's order
  LDPairVec terms(query);
  sort(terms.begin(), terms.end(),
    [&](const pair<long, double>& a, const pair<long, double>& b) {
      return view.rank[a.first] < view.rank[b.first]; });
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

  const long n = view.size();
  LDPairVec ans(n);
  for (long i = 0; i < n; ++i)
    ans[i] = make_pair(i, 0.0);

  if (dense) {
    for (long i = 0; i < n; ++i) {
      const auto at = view.locate(i);
      const LVec& syms = at.first->syms[at.second];
      for (const auto& ft : terms)
        if (find(syms.begin(), syms.end(), ft.first) != syms.end())
          ans[i].second += ft.second;
    }
  }
  else
    for (const auto& ft : terms)
      for (size_t v = 0; v < view.versions.size(); ++v) {
        const auto ths = view.versions[v]->theorems_of(ft.first);
        for (auto t = ths.first; t != ths.second; ++t) {
          const long i = view.first[v] + *t;
          if (!in_scope || (*in_scope)[i]) ans[i].second += ft.second;
        }
      }

  for (long i = 0; i < n; ++i) {
    double n2 = 0;
    if (!dense) n2 = norms2[i];
    else {
      const auto at = view.locate(i);
      n2 = norm2(at.first->syms[at.second]);
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
