#include <algorithm>
#include <map>
#include <set>
#include "predictor.cpp"
#include "view.cpp"

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

    sort_prediction(ans, keep_visible(ans, no_adv));
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
      if (visible(post.i, maxth))
        ans[post.i].second += w * post.a - post.b;
    }
  }

  settle(ans, query, keep_visible(ans, no_adv), c);
  return ans;
}

// Given the sparse scores without C in [ans], puts the best [no_adv] first,
// in the order of [score](i), score()'s score of theorem i. Scores within
// [margin] of their neighbours may be in the other order under score(), so
// each run of such scores is scored again with score() and sorted; a run
// reaching the cutoff is followed into the rest of [ans].
template <class Score>
void settle_nbayes(LDPairVec& ans, long no_adv, double c, Score score) {
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

  auto run = ans.begin();
  while (run < ans.begin() + no_adv) {
    auto next = run + 1;
    while (next < end && (next - 1)->second - next->second <= margin)
      ++next;
    if (next - run == 1)
      run->second -= c;
    else {
      for (auto p = run; p < next; ++p)
        p->second = score(p->first);
      sort(run, next, sortfun);
    }
    run = next;
  }
}

void NaiveBayes::settle(LDPairVec& ans, const LDPairVec& query, long no_adv,
  double c) const {
  const map<feature_t, double> symh(query.begin(), query.end());
  settle_nbayes(ans, no_adv, c, [&](long i) { return score(i, symh); });
}

// The score of a theorem used [n] times as a dependency, with feature counts
// [sfreqh] (its sfreq), for the query features and weights [symh]; adds the
// counts up in sfreqh's order.
double score_nbayes(long n, const unordered_map<feature_t, long>& sfreqh,
  map<feature_t, double> symh, const Tfidf& tfidf) {
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

double NaiveBayes::score(sample_t i, map<feature_t, double> symh) const {
  return score_nbayes(tfreq[i], sfreq[i], move(symh), tfidf);
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

// NaiveBayes's ranking of the theorems of a view, computed as NaiveBayes
// computes it in a process started on them: with the view's IDF and the
// counts the view's theorems give, numbering features by their rank in the
// view. Unless [dense], a theorem's counts are not stored: they are its own
// features and those of the theorems learning it as a dependency, which
// score() adds up again as the process does.
class NaiveBayesView : public ViewPredictor {
  public:
    // as NaiveBayes's [max_deps] and [dense]
    NaiveBayesView(const View& view, long sym_num, long max_deps, bool dense);

    LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const;

  private:
    // theorem i's sfreq, as the process builds it, in the same order, so
    // that it is iterated in the same order
    unordered_map<feature_t, long> sfreq(long i) const;
    double score(long i, map<feature_t, double> symh) const;
    // adds the features of theorem j to [m] with weight [w], as add_sym does
    void add(unordered_map<feature_t, long>& m, long j, long w) const;
    // theorem i's counts (sfreq) in rank order; [acc] is zero for each
    // feature, as it is left
    vector<pair<feature_t, long> > counts(long i, LVec& acc) const;
    // NaiveBayes::index, for the view
    void index();

    const View& view;
    LVec ranks;   // View::rank_table
    Tfidf tfidf;  // by rank
    bool dense;
    // The theorems learning the features of theorem i, with weight 1 each,
    // are users[user_start[i]], ..., users[user_start[i + 1] - 1], in the
    // order the process learns them (once for each time).
    LVec user_start, users;

    // with [dense], each theorem's sfreq, which every query reads;
    // otherwise, as NaiveBayes's, by rank
    vector<unordered_map<feature_t, long> > sfreqs;
    vector<double> base;
    struct Posting {
      long i;
      double a, b;
    };
    vector<Posting> postings;
    vector<size_t> post_start;
};

NaiveBayesView::NaiveBayesView(const View& view, long sym_num,
  long max_deps, bool dense)
: view(view), ranks(view.rank_table(sym_num)), tfidf(view.features()),
  dense(dense) {
  const long n = view.size();
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(ranks_in(s, ranks));

  // theorem i learns its first max_deps dependencies (NaiveBayes::learn)
  const LVec ids = view.id_table();
  LVecVec learned(n);
  for (long i = 0; i < n; ++i) {
    learned[i] = view.deps_of(i, ids);
    if (max_deps > 0 && (long)learned[i].size() > max_deps)
      learned[i].resize(max_deps);
  }
  user_start.assign(n + 1, 0);
  for (const auto& ds : learned)
    for (const auto d : ds) ++user_start[d + 1];
  for (long i = 0; i < n; ++i) user_start[i + 1] += user_start[i];
  users.resize(user_start[n]);
  LVec next(user_start.begin(), user_start.end() - 1);
  for (long i = 0; i < n; ++i)
    for (const auto d : learned[i]) users[next[d]++] = i;

  if (dense)
    for (long i = 0; i < n; ++i) sfreqs.push_back(sfreq(i));
  else
    index();
}

void NaiveBayesView::add(unordered_map<feature_t, long>& m, long j,
  long w) const {
  const auto at = view.locate(j);
  for (const auto f : at.first->syms[at.second]) add_sym(m, ranks[f], w);
}

unordered_map<feature_t, long> NaiveBayesView::sfreq(long i) const {
  // NaiveBayes's sfreq starts from a copy of this
  static const unordered_map<feature_t, long> empty(100);
  unordered_map<feature_t, long> m(empty);
  // theorem i learns itself before the theorems after it
  bool self = false;
  for (long u = user_start[i]; u < user_start[i + 1]; ++u) {
    if (!self && users[u] >= i) {
      add(m, i, 1000);
      self = true;
    }
    add(m, users[u], 1);
  }
  if (!self) add(m, i, 1000);
  return m;
}

double NaiveBayesView::score(long i, map<feature_t, double> symh) const {
  const long t = 1000 + user_start[i + 1] - user_start[i];
  if (dense)
    return score_nbayes(t, sfreqs[i], move(symh), tfidf);
  return score_nbayes(t, sfreq(i), move(symh), tfidf);
}

vector<pair<feature_t, long> > NaiveBayesView::counts(long i, LVec& acc)
  const {
  LVec touched;
  const auto add = [&](long j, long w) {
    const auto at = view.locate(j);
    for (const auto f : at.first->syms[at.second]) {
      const long r = ranks[f];
      if (acc[r] == 0) touched.push_back(r);
      acc[r] += w;
    }
  };
  add(i, 1000);
  for (long u = user_start[i]; u < user_start[i + 1]; ++u) add(users[u], 1);
  sort(touched.begin(), touched.end());
  vector<pair<feature_t, long> > out;
  for (const auto r : touched) {
    out.push_back(make_pair(r, acc[r]));
    acc[r] = 0;
  }
  return out;
}

void NaiveBayesView::index() {
  const long n = view.size(), features = view.features();
  LVec acc(features, 0);

  post_start.assign(features + 1, 0);
  for (long i = 0; i < n; ++i)
    for (const auto& fs : counts(i, acc)) ++post_start[fs.first + 1];
  for (long f = 0; f < features; ++f)
    post_start[f + 1] += post_start[f];
  postings.resize(post_start[features]);

  vector<size_t> next(post_start.begin(), post_start.end() - 1);
  base.resize(n);
  for (long i = 0; i < n; ++i) {
    const long t = 1000 + user_start[i + 1] - user_start[i];
    base[i] = 30 * log(t);
    // in rank order, as NaiveBayes::index adds them up
    for (const auto& fs : counts(i, acc)) {
      const double idf = tfidf.get(fs.first);
      const double s = fs.second;
      const double b = idf * 0.2 * log(1 + (1 - s) / t);
      base[i] += b;
      postings[next[fs.first]++] = {i, idf * (log(5 * s / t) + 18), b};
    }
  }
}

LDPairVec NaiveBayesView::predict_weighted(const LDPairVec& query,
  const vector<char>* in_scope, long no_adv) const {
  // the query by rank, as NaiveBayes::predict_weighted takes it
  LDPairVec ranked;
  for (const auto& fw : query)
    ranked.push_back(make_pair(view.rank_of(fw.first), fw.second));
  const map<feature_t, double> symh(ranked.begin(), ranked.end());
  const long n = view.size();

  if (dense) {
    LDPairVec ans;
    for (long i = 0; i < n; ++i)
      if (!in_scope || (*in_scope)[i])
        ans.push_back(make_pair(i, score(i, symh)));
    sort_prediction(ans, min(no_adv, (long)ans.size()));
    return ans;
  }

  LDPairVec ans(n);
  for (long i = 0; i < n; ++i)
    ans[i] = make_pair(i, base[i]);

  // in rank order, as NaiveBayes::predict_weighted adds features up
  sort(ranked.begin(), ranked.end());
  double c = 0;
  for (const auto& fw : ranked) {
    const double w = fw.second;
    c += w * tfidf.get(fw.first) * 18;
    for (size_t p = post_start[fw.first]; p < post_start[fw.first + 1]; ++p) {
      const Posting& post = postings[p];
      if (!in_scope || (*in_scope)[post.i])
        ans[post.i].second += w * post.a - post.b;
    }
  }

  settle_nbayes(ans, keep_in_scope(ans, in_scope, no_adv), c,
    [&](long i) { return score(i, symh); });
  return ans;
}
