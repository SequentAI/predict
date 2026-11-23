#ifndef MESH_CPP
#define MESH_CPP

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

// The theorem ids of the first n predictions of ans
LVec first_ids(const LDPairVec& ans, long n) {
  LVec ids;
  for (long j = 0; j < min(n, (long)ans.size()); ++j)
    ids.push_back(ans[j].first);
  return ids;
}

// The theorem ids of the first n of p's predictions for query
LVec first_ids(const Predictor& p, const LDPairVec& query, long maxth,
  long n) {
  n = min(n, maxth);
  return first_ids(p.predict_weighted(query, maxth, n), n);
}

// mesh_facts: the candidates are the first max_facts theorems of each leg.
// In a leg, the theorem at rank r weighs steep_weight(r), divided by the mean
// weight of the leg's first max_facts theorems; a candidate scores the sum,
// over the legs it is in, of its weight times the leg's. Isabelle averages
// the sum over the legs and rounds it to a multiple of 1e-8, which can only
// make nearly equal scores equal; here, equal scores come in <seq> order, or,
// sorted by sort_fused with [first], in the order the legs first rank them.
// Unsorted: the caller drops the theorems out of scope, then sorts.
LDPairVec mesh_facts(const vector<Leg>& legs, long maxth, long max_facts,
  vector<long>* first = nullptr) {
  LDPairVec ans(maxth);
  for (long i = 0; i < maxth; ++i) ans[i] = make_pair(i, 0.0);
  vector<char> candidate(maxth, 0);
  if (first) first->assign(maxth, maxth);
  long seen = 0;
  for (const auto& leg : legs) {
    const long n = leg.ranking.size();
    const long head = min(n, max_facts);
    if (first)
      for (const auto i : leg.ranking)
        if ((*first)[i] == maxth) (*first)[i] = seen++;
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
  return ans;
}

// Sorts the first n of [ans] as sort_prediction does, but equal scores in
// the order [first] gives (mesh_facts), then in <seq> order.
void sort_fused(LDPairVec& ans, const vector<long>& first, long n) {
  partial_sort(ans.begin(), ans.begin() + n, ans.end(),
    [&](const pair<long, double>& a, const pair<long, double>& b) {
      if (a.second != b.second) return a.second > b.second;
      if (first[a.first] != first[b.first])
        return first[a.first] < first[b.first];
      return a.first < b.first;
    });
}

// knn and naive Bayes, each cut to the first no_adv theorems, weighted 0.5
// each (MaSh_NB_kNN in Isabelle's query_internal). With [deps_after] (-P),
// MaSh fuses naive Bayes and knn's neighbours alone instead, each cut to
// 2 no_adv + 25 and fused at that depth, then follows each theorem at once by
// its dependencies (with [credit_deps]), each theorem at its first place, as
// knn places a neighbour's (HOL4's add_thmdep), cut to no_adv.
class MaSh : public Predictor {
  public:
    MaSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, long max_deps, bool credit_deps, bool deps_after = false);

    void learn(long from, long to);
    bool forget(long from) { return knn.forget(from) && nbayes.forget(from); }
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;
    void set_scope(const vector<char>* scope);
    // its knn, which ranks as a -p knn process; MaSh fuses knn's credited
    // ranking (kNN::predict_credited)
    const Predictor& knn_part() const { return knn; }

  private:
    // the -P ranking (deps_after)
    LDPairVec predict_deps_after(const LDPairVec& query, long maxth,
      long no_adv) const;

    kNN knn;
    NaiveBayes nbayes;
    bool credit_deps, deps_after;
};

MaSh::MaSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
  long sym_num, long max_deps, bool credit_deps, bool deps_after)
: Predictor(deps, syms, sym_ths, sym_num),
  knn(deps, syms, sym_ths, sym_num, credit_deps),
  nbayes(deps, syms, sym_ths, sym_num, max_deps, false),
  credit_deps(credit_deps), deps_after(deps_after) {}

void MaSh::learn(long from, long to) {
  knn.learn(from, to);
  nbayes.learn(from, to);
}

void MaSh::set_scope(const vector<char>* scope) {
  Predictor::set_scope(scope);
  knn.set_scope(scope);
  nbayes.set_scope(scope);
}

LDPairVec MaSh::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto s : csyms) query.push_back(make_pair(s, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec MaSh::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  if (deps_after) return predict_deps_after(query, maxth, no_adv);
  const long n = min(no_adv, maxth);
  LDPairVec ans = mesh_facts(
    {{0.5, first_ids(nbayes, query, maxth, no_adv)},
     {0.5, first_ids(knn.predict_credited(query, maxth, n), n)}}, maxth,
    no_adv);
  sort_prediction(ans, keep_visible(ans, no_adv));
  return ans;
}

LDPairVec MaSh::predict_deps_after(const LDPairVec& query, long maxth,
  long no_adv) const {
  const long m = min(2 * no_adv + 25, maxth);
  vector<long> first;
  LDPairVec fused = mesh_facts(
    {{0.5, first_ids(nbayes, query, maxth, m)},
     {0.5, first_ids(knn.predict_neighbours(query, maxth, m), m)}}, maxth, m,
    &first);
  sort_fused(fused, first, keep_visible(fused, maxth));
  // a theorem scores the fused score of the one that placed it
  vector<char> placed(maxth, 0);
  LDPairVec ans;
  const auto place = [&](long i, double score) {
    if (placed[i]) return;
    placed[i] = 1;
    ans.push_back(make_pair(i, score));
  };
  for (const auto& f : fused) {
    if ((long)ans.size() >= no_adv) break;
    place(f.first, f.second);
    if (!credit_deps) continue;
    for (const auto d : deps[f.first])
      if (visible(d, maxth)) place(d, f.second);
  }
  ans.resize(min(no_adv, (long)ans.size()));
  return ans;
}

// The view's numbers of the first n of p's predictions for query, of the
// view's [size] theorems, as first_ids
LVec first_ids(const ViewPredictor& p, const LDPairVec& query,
  const vector<char>* in_scope, long size, long n) {
  n = min(n, size);
  return first_ids(p.predict_weighted(query, in_scope, n), n);
}

// MaSh's ranking of the theorems of a view, from its parts' views, as MaSh
// ranks in a process started on them.
class MaShView : public ViewPredictor {
  public:
    MaShView(const View& view, long sym_num, long max_deps, bool credit_deps);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;
    const ViewPredictor& knn_part() const { return knn; }

  private:
    const View& view;
    kNNView knn;
    NaiveBayesView nbayes;
};

MaShView::MaShView(const View& view, long sym_num, long max_deps,
  bool credit_deps)
: view(view), knn(view, sym_num, credit_deps),
  nbayes(view, sym_num, max_deps, false) {}

LDPairVec MaShView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  const long n = view.size();
  const long m = min(no_adv, n);
  LDPairVec ans = mesh_facts(
    {{0.5, first_ids(nbayes, query, in_scope, n, no_adv)},
     {0.5, first_ids(knn.predict_credited(query, in_scope, m), m)}}, n,
    no_adv);
  sort_prediction(ans, keep_in_scope(ans, in_scope, no_adv));
  return ans;
}

// MePo and MaSh, weighted 0.5 each (Isabelle's relevant_facts). For M
// predictions, MePo is cut to its first M theorems and MaSh to 2M + 25; with
// [over], both are cut to 51M/50. [mepo](n) and [mash](n) give the first n
// theorems of each. Unsorted, as mesh_facts.
template <class MePoFirst, class MaShFirst>
LDPairVec mesh_fusion(long maxth, long no_adv, bool over, MePoFirst mepo,
  MaShFirst mash) {
  const long n_mepo = over ? no_adv * 51 / 50 : no_adv;
  const long n_mash = over ? no_adv * 51 / 50 : 2 * no_adv + 25;
  return mesh_facts({{0.5, mepo(n_mepo)}, {0.5, mash(n_mash)}}, maxth,
    no_adv);
}

// mesh_fusion's ranking of the theorems of the process. With [split] (-P),
// MePo ranks the typed symbols, numbered in [split], and MaSh, which is then
// MaSh with deps_after, the features; predict_split fuses them as mesh_fusion
// does, but with MaSh cut to the same length as MePo (MaSh itself goes
// 2 no_adv + 25 deep), and equal scores in the order the legs first rank
// them.
class MeSh : public Predictor {
  public:
    MeSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, const vector<unsigned char>& statures, long max_deps,
      bool credit_deps, bool over, const TypedTable* split = nullptr,
      const TypeTable* types = nullptr, const TypedSyms* typed = nullptr,
      const TypedSyms* freq_only = nullptr);

    void learn(long from, long to);
    bool forget(long from) { return mepo.forget(from) && mash.forget(from); }
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;
    LDPairVec predict_split(const LDPairVec& query,
      const vector<TypedSym>& typed, long maxth, long no_adv) const;
    void set_scope(const vector<char>* scope);
    // the part [method] names, which ranks as a -p [method] process: knn,
    // mash or mepo; null for another
    const Predictor* part(const string& method) const;

  private:
    MePo mepo;
    MaSh mash;
    bool over;
};

MeSh::MeSh(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
  long sym_num, const vector<unsigned char>& statures, long max_deps,
  bool credit_deps, bool over, const TypedTable* split, const TypeTable* types,
  const TypedSyms* typed, const TypedSyms* freq_only)
: Predictor(deps, syms, sym_ths, sym_num),
  mepo(deps, split ? split->syms : syms, split ? split->sym_ths : sym_ths,
    split ? (long)split->no_sym.size() : sym_num, statures,
    split ? types : nullptr, split ? typed : nullptr,
    split ? freq_only : nullptr),
  mash(deps, syms, sym_ths, sym_num, max_deps, credit_deps, split != nullptr),
  over(over) {}

void MeSh::learn(long from, long to) {
  mepo.learn(from, to);
  mash.learn(from, to);
}

void MeSh::set_scope(const vector<char>* scope) {
  Predictor::set_scope(scope);
  mepo.set_scope(scope);
  mash.set_scope(scope);
}

const Predictor* MeSh::part(const string& method) const {
  if (method == "knn") return &mash.knn_part();
  if (method == "mash") return &mash;
  if (method == "mepo") return &mepo;
  return nullptr;
}

LDPairVec MeSh::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto s : csyms) query.push_back(make_pair(s, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec MeSh::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  LDPairVec ans = mesh_fusion(maxth, no_adv, over,
    [&](long n) { return first_ids(mepo, query, maxth, n); },
    [&](long n) { return first_ids(mash, query, maxth, n); });
  sort_prediction(ans, keep_visible(ans, no_adv));
  return ans;
}

LDPairVec MeSh::predict_split(const LDPairVec& query,
  const vector<TypedSym>& typed, long maxth, long no_adv) const {
  const long n = min(over ? no_adv * 51 / 50 : no_adv, maxth);
  vector<long> first;
  LDPairVec ans = mesh_facts(
    {{0.5, first_ids(mepo.predict_split(query, typed, maxth, n), n)},
     {0.5, first_ids(mash, query, maxth, n)}}, maxth, no_adv, &first);
  sort_fused(ans, first, keep_visible(ans, no_adv));
  return ans;
}

// MeSh's ranking of the theorems of a view, from its parts' views, as MeSh
// ranks in a process started on them.
class MeShView : public ViewPredictor {
  public:
    MeShView(const View& view, long sym_num, long max_deps, bool credit_deps,
      bool over);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;
    // as MeSh::part
    const ViewPredictor* part(const string& method) const;

  private:
    const View& view;
    MePoView mepo;
    MaShView mash;
    bool over;
};

MeShView::MeShView(const View& view, long sym_num, long max_deps,
  bool credit_deps, bool over)
: view(view), mepo(view, sym_num, nullptr),
  mash(view, sym_num, max_deps, credit_deps), over(over) {}

const ViewPredictor* MeShView::part(const string& method) const {
  if (method == "knn") return &mash.knn_part();
  if (method == "mash") return &mash;
  if (method == "mepo") return &mepo;
  return nullptr;
}

LDPairVec MeShView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  const long n = view.size();
  LDPairVec ans = mesh_fusion(n, no_adv, over,
    [&](long m) { return first_ids(mepo, query, in_scope, n, m); },
    [&](long m) { return first_ids(mash, query, in_scope, n, m); });
  sort_prediction(ans, keep_in_scope(ans, in_scope, no_adv));
  return ans;
}

#endif
