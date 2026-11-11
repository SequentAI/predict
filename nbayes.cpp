#include <algorithm>
#include <map>
#include <set>
#include "predictor.cpp"

typedef long feature_t;
typedef long sample_t;

// Naive Bayes in the form of Isabelle's MaSh. With t = tfreq[i] and
// s_f = sfreq[i][f], S the features in sfreq[i] and G the query's, theorem
// i scores
//
//   30 ln t + sum_{f in S & G} w_f idf_f ln(5 s_f / t)
//           + sum_{f in S \ G} 0.2 idf_f ln(1 + (1 - s_f) / t)
//           - sum_{f in G \ S} 18 w_f idf_f.
//
// score() computes this directly, over every theorem and every feature in S.
// Unless [dense], predictions instead use the equal form
//
//   B_i + C + sum_{f in S & G} (w_f a_{i,f} - b_{i,f}),
//
// where b_{i,f} = 0.2 idf_f ln(1 + (1 - s_f) / t), a_{i,f} = idf_f
// (ln(5 s_f / t) + 18), B_i = 30 ln t + sum_{f in S} b_{i,f} and
// C = -sum_{f in G} 18 w_f idf_f. B_i, and (i, a_{i,f}, b_{i,f}) in a
// posting list per feature f, are computed once after learning, so a query
// reads only the postings of its own features.
//
// The two forms round differently, so where the sparse one puts scores within
// a margin far wider than either's rounding error of each other, score()
// settles their order: predictions are those score() gives, ties included.
class NaiveBayes : public Predictor {
  public:
    // A theorem's dependencies learn its features only for the first
    // [max_deps] of them (all if 0). [dense] predicts with score().
    NaiveBayes(const LVecVec& deps, const LVecVec& syms,
      const LVecVec& sym_ths, long sym_num, long max_deps, bool dense);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    double score(sample_t i, map<feature_t, double> symh) const;
    void settle(LDPairVec& ans, const LDPairVec& query, long no_adv,
      double c) const;

    void learn(const LVec& csyms, sample_t i, const LVec& cdeps);
    void add_sample_freqs(const LVec& csyms, sample_t i, long weight);
    void index();

    // tfreq[t] = number of theorems having t as dependency
    vector<long> tfreq;
    // sfreq[t][f] = number of theorems having f and having t as dependency
    vector<unordered_map <feature_t, long> > sfreq;

    Tfidf tfidf;
    long sym_num, max_deps;
    bool dense;

    struct Posting {
      sample_t i;
      double a, b;
    };
    // base[i] = B_i
    vector<double> base;
    // postings[post_start[f], post_start[f + 1]) are those of feature f, in
    // increasing order of i
    vector<Posting> postings;
    vector<size_t> post_start;
};

NaiveBayes::NaiveBayes(const LVecVec& deps, const LVecVec& syms,
  const LVecVec& sym_ths, long sym_num, long max_deps, bool dense)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num), sym_num(sym_num),
  max_deps(max_deps), dense(dense) {
  tfreq.resize(deps.size());
  sfreq.resize(deps.size(), unordered_map <feature_t, long> (100));
}

void NaiveBayes::learn(sample_t from, sample_t to) {
  for (sample_t i = from; i < to; ++i)
    tfidf.add(syms[i]);

  for (sample_t i = from; i < to; i++)
    learn(syms[i], i, deps[i]);

  // idf changes with every learned theorem, so every entry changes
  if (!dense)
    index();
}

void NaiveBayes::index() {
  const sample_t n = tfreq.size();

  post_start.assign(sym_num + 1, 0);
  for (sample_t i = 0; i < n; ++i)
    for (const auto& sv : sfreq[i])
      ++post_start[sv.first + 1];
  for (long f = 0; f < sym_num; ++f)
    post_start[f + 1] += post_start[f];
  postings.resize(post_start[sym_num]);

  vector<size_t> next(post_start.begin(), post_start.end() - 1);
  base.resize(n);
  vector<pair<feature_t, long> > sv;
  for (sample_t i = 0; i < n; ++i) {
    const long t = tfreq[i];
    base[i] = 30 * log(t);
    // in feature order, so that theorems with equal statistics get equal
    // baselines
    sv.assign(sfreq[i].begin(), sfreq[i].end());
    sort(sv.begin(), sv.end());
    for (const auto& fs : sv) {
      const double idf = tfidf.get(fs.first);
      const double s = fs.second;
      const double b = idf * 0.2 * log(1 + (1 - s) / t);
      base[i] += b;
      postings[next[fs.first]++] = {i, idf * (log(5 * s / t) + 18), b};
    }
  }
}

LDPairVec NaiveBayes::predict(const LVec &csyms, sample_t maxth, long no_adv) const {
  LDPairVec query;
  for (const auto f : csyms) query.push_back(make_pair(f, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec NaiveBayes::predict_weighted(const LDPairVec &query, sample_t maxth,
  long no_adv) const {
  LDPairVec ans = LDPairVec(maxth, make_pair(0, 0));

  if (dense) {
    // query features and their weights
    map<feature_t, double> symh(query.begin(), query.end());

    for(long i = 0; i < maxth; ++i) {
      ans[i].first  = i;
      ans[i].second = score(i, symh);
    }

    sort_prediction(ans, no_adv);
    return ans;
  }

  for (long i = 0; i < maxth; ++i)
    ans[i] = make_pair(i, base[i]);

  // in feature order, so that theorems with equal statistics get equal scores
  LDPairVec sorted(query);
  sort(sorted.begin(), sorted.end());
  double c = 0;
  for (const auto& fw : sorted) {
    const double w = fw.second;
    c += w * tfidf.get(fw.first) * 18;
    for (size_t p = post_start[fw.first]; p < post_start[fw.first + 1]; ++p) {
      const Posting& post = postings[p];
      if (post.i >= maxth)
        break;
      ans[post.i].second += w * post.a - post.b;
    }
  }

  settle(ans, query, no_adv, c);
  return ans;
}

// Given the sparse scores without C in [ans], puts the best [no_adv] first,
// in score()'s order. Scores within [margin] of their neighbours may be in
// the other order under score(), so each run of such scores is scored again
// with score() and sorted; a run reaching the cutoff is followed into the
// rest of [ans].
void NaiveBayes::settle(LDPairVec& ans, const LDPairVec& query, long no_adv,
  double c) const {
  if (no_adv == 0)
    return;
  sort_prediction(ans, no_adv);
  const double cutoff = ans[no_adv - 1].second;
  const double margin =
    1e-9 * (fabs(ans[0].second) + fabs(cutoff) + fabs(c) + 1);
  // the rest within the margin of the cutoff, sorted, after the best
  const auto end = partition(ans.begin() + no_adv, ans.end(),
    [&](const pair<long, double>& p) { return p.second >= cutoff - margin; });
  sort(ans.begin() + no_adv, end, sortfun);

  const map<feature_t, double> symh(query.begin(), query.end());
  auto run = ans.begin();
  while (run < ans.begin() + no_adv) {
    auto next = run + 1;
    while (next < end && (next - 1)->second - next->second <= margin)
      ++next;
    if (next - run == 1)
      run->second -= c;
    else {
      for (auto p = run; p < next; ++p)
        p->second = score(p->first, symh);
      sort(run, next, sortfun);
    }
    run = next;
  }
}

double NaiveBayes::score(sample_t i, map<feature_t, double> symh) const {
  // number of times current theorem was used as dependency
  const long n      = tfreq[i];
  const auto sfreqh = sfreq[i];

  double s = 30 * log(n);

  for (const auto& sv : sfreqh) {
    // sv.first ranges over all features of theorems depending on i
    // sv.second is the number of times sv.first appears among theorems
    // depending on i
    double sfreqv = sv.second;

    // if sv.first exists in query features
    const auto q = symh.find(sv.first);
    if (q != symh.end()) {
      s += q->second * tfidf.get(sv.first) * log (5 * sfreqv / n);
      symh.erase(q);
    }
    else
      s += tfidf.get(sv.first) * 0.2 * log (1 + (1 - sfreqv) / n);
  }

  // for all query features that did not appear in features of dependencies
  // of current theorem
  for (const auto& f : symh) s -= f.second * tfidf.get(f.first) * 18;

  return s;
}

void add_sym(unordered_map <feature_t, long> &m, feature_t sym, long w) {
  auto itr = m.find(sym);
  if (itr == m.end()) m[sym]  = w;
  else         (itr->second) += w;
}

void NaiveBayes::add_sample_freqs(const LVec& csyms, sample_t i, long w) {
  tfreq[i] += w;
  for (const auto s : csyms) add_sym(sfreq[i], s, w);
}

void NaiveBayes::learn(const LVec& csyms, sample_t i, const LVec& cdeps) {
  add_sample_freqs(csyms, i, 1000);
  const size_t n = max_deps > 0 ? min(cdeps.size(), (size_t)max_deps)
                                : cdeps.size();
  for (size_t k = 0; k < n; ++k) add_sample_freqs(csyms, cdeps[k], 1);
}
