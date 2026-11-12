#include "knn.cpp"
#include "mepo.cpp"
#include "nbayes.cpp"

// The rank fusions of Isabelle's Sledgehammer (src/HOL/Tools/Sledgehammer/
// sledgehammer_mash.ML, lines 164-200 at commit 34ca456d of the GitHub mirror
// isabelle-prover/mirror-isabelle: scaled_avg, normalize_scores, mesh_facts,
// steep_weight_of_fact). MaSh fuses knn with naive Bayes, and MeSh fuses MePo
// with MaSh.

// The weight of the fact at rank r (from 0) of a ranking
inline double steep_weight(long r) {
  return pow(0.62, log2(r + 1.0));
}

// An input ranking of a fusion and its weight
struct Leg {
  double weight;
  LVec ranking;
};

// The theorem ids of the first n of p's predictions for query
LVec first_ids(const Predictor& p, const LDPairVec& query, long maxth,
  long n) {
  n = min(n, maxth);
  const LDPairVec ans = p.predict_weighted(query, maxth, n);
  LVec ids;
  for (long j = 0; j < n; ++j) ids.push_back(ans[j].first);
  return ids;
}

// mesh_facts: the candidates are the first max_facts theorems of each leg.
// In a leg, the theorem at rank r weighs steep_weight(r), divided by the mean
// weight of the leg's first max_facts theorems; a candidate scores the sum,
// over the legs it is in, of its weight times the leg's. Isabelle averages
// the sum over the legs and rounds it to a multiple of 1e-8, which can only
// make nearly equal scores equal; here, equal scores come in <seq> order.
LDPairVec mesh_facts(const vector<Leg>& legs, long maxth, long max_facts) {
  LDPairVec ans(maxth);
  for (long i = 0; i < maxth; ++i) ans[i] = make_pair(i, 0.0);
  vector<char> candidate(maxth, 0);
  for (const auto& leg : legs) {
    const long n = leg.ranking.size();
    const long head = min(n, max_facts);
    if (head == 0) continue;
    double mean = 0;
    for (long r = 0; r < head; ++r) mean += steep_weight(r) / head;
    for (long r = 0; r < n; ++r) {
      const long i = leg.ranking[r];
      ans[i].second += leg.weight * steep_weight(r) / mean;
      if (r < max_facts) candidate[i] = 1;
    }
  }
  // the others rank after the candidates
  for (long i = 0; i < maxth; ++i)
    if (!candidate[i]) ans[i].second = 0;
  sort_prediction(ans, min(max_facts, maxth));
  return ans;
}

// knn and naive Bayes, each cut to the first no_adv theorems, weighted 0.5
// each (MaSh_NB_kNN in Isabelle's query_internal)
class MaSh : public Predictor {
  public:
    MaSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, long max_deps, bool credit_deps);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    kNN knn;
    NaiveBayes nbayes;
};

MaSh::MaSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
  long sym_num, long max_deps, bool credit_deps)
: Predictor(deps, syms, sym_ths, sym_num),
  knn(deps, syms, sym_ths, sym_num, credit_deps),
  nbayes(deps, syms, sym_ths, sym_num, max_deps, false) {}

void MaSh::learn(long from, long to) {
  knn.learn(from, to);
  nbayes.learn(from, to);
}

LDPairVec MaSh::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto s : csyms) query.push_back(make_pair(s, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec MaSh::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  return mesh_facts({{0.5, first_ids(nbayes, query, maxth, no_adv)},
                     {0.5, first_ids(knn, query, maxth, no_adv)}},
    maxth, no_adv);
}

// MePo and MaSh, weighted 0.5 each (Isabelle's relevant_facts). For M
// predictions, MePo is cut to its first M theorems and MaSh to 2M + 25; with
// [over], both are cut to 51M/50.
class MeSh : public Predictor {
  public:
    MeSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, const vector<unsigned char>& statures, long max_deps,
      bool credit_deps, bool over);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    MePo mepo;
    MaSh mash;
    bool over;
};

MeSh::MeSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
  long sym_num, const vector<unsigned char>& statures, long max_deps,
  bool credit_deps, bool over)
: Predictor(deps, syms, sym_ths, sym_num),
  mepo(deps, syms, sym_ths, sym_num, statures),
  mash(deps, syms, sym_ths, sym_num, max_deps, credit_deps), over(over) {}

void MeSh::learn(long from, long to) {
  mepo.learn(from, to);
  mash.learn(from, to);
}

LDPairVec MeSh::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto s : csyms) query.push_back(make_pair(s, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec MeSh::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  const long n_mepo = over ? no_adv * 51 / 50 : no_adv;
  const long n_mash = over ? no_adv * 51 / 50 : 2 * no_adv + 25;
  return mesh_facts({{0.5, first_ids(mepo, query, maxth, n_mepo)},
                     {0.5, first_ids(mash, query, maxth, n_mash)}},
    maxth, no_adv);
}
