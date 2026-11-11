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
//   (30 ln t + B_i - sum_{f in S & G} b_{i,f}) + sum_{f in S & G} w_f a_{i,f}
//     + C,
//
// where b_{i,f} = 0.2 idf_f ln(1 + (1 - s_f) / t), a_{i,f} = idf_f
// (ln(5 s_f / t) + 18), B_i = sum_{f in S} b_{i,f} and C = -sum_{f in G} 18
// w_f idf_f. B_i, and (i, a_{i,f}, b_{i,f}) in a posting list per feature f,
// are computed once after learning, so a query reads only the postings of its
// own features. The b terms are summed in fixed point, which is exact, so
// that subtracting them from B_i leaves exactly the sum over S \ G.
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

    // b terms in units of 2^-64: far finer than a double's resolution at
    // the size of a score, and ample for the sum of every b term
    typedef __int128 Exact;
    static Exact exact(double x) { return (Exact)ldexp(x, 64); }
    static double inexact(Exact x) {
      // the high and low 64 bits, faster than converting the whole
      return (double)(long long)(x >> 64)
        + ldexp((double)(unsigned long long)x, -64);
    }

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
      double a;
      Exact b;
    };
    // base[i] = B_i; log_t30[i] = 30 ln t, and base_score[i] = 30 ln t + B_i,
    // the score without C of a theorem sharing no feature with the query
    vector<Exact> base;
    vector<double> log_t30, base_score;
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
  base.assign(n, 0);
  log_t30.resize(n);
  base_score.resize(n);
  for (sample_t i = 0; i < n; ++i) {
    const double t = tfreq[i];
    for (const auto& fs : sfreq[i]) {
      const double idf = tfidf.get(fs.first);
      const double s = fs.second;
      const Exact b = exact(idf * 0.2 * log(1 + (1 - s) / t));
      base[i] += b;
      postings[next[fs.first]++] = {i, idf * (log(5 * s / t) + 18), b};
    }
    log_t30[i] = 30 * log(t);
    base_score[i] = log_t30[i] + inexact(base[i]);
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
    ans[i] = make_pair(i, base_score[i]);

  // for the theorems sharing features with the query, the sums of b_{i,f}
  // and of w_f a_{i,f} over the shared features
  vector<Exact> b(maxth);
  vector<double> a(maxth);
  vector<char> hit(maxth);
  vector<sample_t> hits;
  double c = 0;
  for (const auto& fw : query) {
    const double w = fw.second;
    c += w * tfidf.get(fw.first) * 18;
    for (size_t p = post_start[fw.first]; p < post_start[fw.first + 1]; ++p) {
      const Posting& post = postings[p];
      if (post.i >= maxth)
        break;
      if (!hit[post.i]) {
        hit[post.i] = 1;
        hits.push_back(post.i);
      }
      b[post.i] += post.b;
      a[post.i] += w * post.a;
    }
  }
  for (const auto i : hits)
    ans[i].second = (log_t30[i] + inexact(base[i] - b[i])) + a[i];

  // C is the same for every theorem; adding it after ranking keeps it from
  // rounding apart scores that differ only in their last bits
  sort_prediction(ans, no_adv);
  for (long j = 0; j < no_adv; ++j)
    ans[j].second -= c;
  return ans;
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
