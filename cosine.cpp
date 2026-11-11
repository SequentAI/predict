#include <algorithm>
#include "predictor.cpp"

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
  // weights are divided by the largest to keep w^2 from overflowing or
  // underflowing.
  LDPairVec terms(query);
  sort(terms.begin(), terms.end());
  double wmax = 0;
  for (const auto& ft : terms)
    wmax = max(wmax, ft.second);
  double qnorm2 = 0;
  for (auto& ft : terms) {
    const double idf = tfidf.get(ft.first);
    const double w = wmax > 0 ? ft.second / wmax : 0;
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
        if (i < maxth) ans[i].second += ft.second;

  for (long i = 0; i < maxth; ++i) {
    const double n2 = dense ? norm2(i) : norms2[i];
    if (ans[i].second != 0 && qnorm2 != 0 && n2 != 0)
      ans[i].second /= sqrt(qnorm2 * n2);
    else
      ans[i].second = 0;
  }

  sort_prediction(ans, no_adv);
  return ans;
}
