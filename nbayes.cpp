#include <algorithm>
#include <map>
#include <set>
#include "predictor.cpp"
#include "view.cpp"

typedef long feature_t;
typedef long sample_t;
// (feature, count) pairs
typedef vector<pair<feature_t, long> > Counts;

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
    bool forget(long from);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

  private:
    double score(sample_t i, map<feature_t, double> symh) const;
    void settle(LDPairVec& ans, const LDPairVec& query, long no_adv,
      double c) const;

    // the number of the first dependencies of theorem u that learn its
    // features
    size_t learned_deps(sample_t u) const;
    // adds [w] times the features of theorem u to [delta], a change of
    // counts
    void gain(Counts& delta, sample_t u, long w) const {
      for (const auto f : syms[u]) delta.push_back(make_pair(f, w));
    }
    // adds [delta], the changes of some theorems' counts, to them, merging
    // each theorem's once; with [dense], computes the sfreqs of the learned
    // ones again instead
    void apply(map<sample_t, Counts>& delta);
    // tfreq[i] of each theorem i, and counts[i] unless [dense], from the
    // users of the theorems [0, to) learned
    void count_all(sample_t to);
    // Theorem i's sfreq: its counts in the order an unordered_map adding up
    // its own features and those of its users, in learn order, iterates
    // them, which score() sums in.
    unordered_map<feature_t, long> sfreq(sample_t i) const;
    void index();

    // the theorems [0, learned) are learned
    sample_t learned = 0;
    // tfreq[t] = number of theorems having t as dependency, 1000 for t
    // itself
    vector<long> tfreq;
    // unless [dense], counts[t] = the (f, number of theorems having f and
    // having t as dependency, 1000 for t itself) in increasing order of f,
    // for each f with a nonzero number
    vector<Counts> counts;
    // users[t] = the theorems learning t as a dependency, in learn order
    // (once for each time)
    LVecVec users;
    // with [dense], each learned theorem's sfreq, which every query reads
    vector<unordered_map<feature_t, long> > sfreqs;

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
  max_deps(max_deps), dense(dense) {}

// Adds the weight of each (feature, weight) pair of [delta] to the counts
// [c], dropping those that reach 0.
void add_counts(Counts& c, Counts delta) {
  sort(delta.begin(), delta.end());
  Counts out;
  out.reserve(c.size() + delta.size());
  auto p = c.begin();
  for (size_t k = 0; k < delta.size(); ) {
    const feature_t f = delta[k].first;
    long s = 0;
    for (; k < delta.size() && delta[k].first == f; ++k) s += delta[k].second;
    for (; p != c.end() && p->first < f; ++p) out.push_back(*p);
    if (p != c.end() && p->first == f) s += (p++)->second;
    if (s != 0) out.push_back(make_pair(f, s));
  }
  out.insert(out.end(), p, c.end());
  c.swap(out);
}

void NaiveBayes::apply(map<sample_t, Counts>& delta) {
  for (auto& td : delta)
    if (!dense)
      add_counts(counts[td.first], move(td.second));
    else if (td.first < learned)
      sfreqs[td.first] = sfreq(td.first);
}

size_t NaiveBayes::learned_deps(sample_t u) const {
  return max_deps > 0 ? min(deps[u].size(), (size_t)max_deps)
                      : deps[u].size();
}

void NaiveBayes::count_all(sample_t to) {
  const sample_t n = syms.size();
  for (sample_t u = 0; u < to; ++u)
    for (size_t k = 0; k < learned_deps(u); ++k)
      users[deps[u][k]].push_back(u);

  LVec acc(sym_num, 0), touched;
  const auto add = [&](sample_t j, long w) {
    for (const auto f : syms[j]) {
      if (acc[f] == 0) touched.push_back(f);
      acc[f] += w;
    }
  };
  for (sample_t i = 0; i < n; ++i) {
    tfreq[i] = (i < to ? 1000 : 0) + users[i].size();
    if (dense) continue;
    if (i < to) add(i, 1000);
    for (const auto u : users[i]) add(u, 1);
    sort(touched.begin(), touched.end());
    for (const auto f : touched) {
      counts[i].push_back(make_pair(f, acc[f]));
      acc[f] = 0;
    }
    touched.clear();
  }
}

void NaiveBayes::learn(sample_t from, sample_t to) {
  sym_num = sym_ths.size();
  tfidf.resize(sym_num);
  const sample_t n = syms.size();
  tfreq.resize(n);
  counts.resize(n);
  users.resize(n);
  for (sample_t i = from; i < to; ++i)
    tfidf.add(syms[i]);

  const bool first = learned == 0;
  map<sample_t, Counts> delta;
  if (!first)
    for (sample_t u = from; u < to; ++u) {
      tfreq[u] += 1000;
      gain(delta[u], u, 1000);
      for (size_t k = 0; k < learned_deps(u); ++k) {
        const sample_t d = deps[u][k];
        users[d].push_back(u);
        ++tfreq[d];
        gain(delta[d], u, 1);
      }
    }
  learned = to;
  if (dense) sfreqs.resize(to);
  if (first) {
    count_all(to);
    if (dense)
      for (sample_t i = 0; i < to; ++i) sfreqs[i] = sfreq(i);
  }
  else
    apply(delta);

  // idf changes with every learned theorem, so every entry changes
  if (!dense)
    index();
}

bool NaiveBayes::forget(sample_t from) {
  map<sample_t, Counts> delta;
  for (sample_t u = learned - 1; u >= from; --u) {
    for (size_t k = learned_deps(u); k-- > 0; ) {
      const sample_t d = deps[u][k];
      if (d >= from) continue;
      users[d].pop_back();
      --tfreq[d];
      gain(delta[d], u, -1);
    }
    tfidf.remove(syms[u]);
  }
  tfreq.resize(from);
  counts.resize(from);
  users.resize(from);
  learned = from;
  if (dense) sfreqs.resize(from);
  apply(delta);
  return true;
}

void NaiveBayes::index() {
  const sample_t n = tfreq.size();

  post_start.assign(sym_num + 1, 0);
  for (sample_t i = 0; i < n; ++i)
    for (const auto& fs : counts[i])
      ++post_start[fs.first + 1];
  for (long f = 0; f < sym_num; ++f)
    post_start[f + 1] += post_start[f];
  postings.resize(post_start[sym_num]);

  vector<size_t> next(post_start.begin(), post_start.end() - 1);
  base.resize(n);
  for (sample_t i = 0; i < n; ++i) {
    const long t = tfreq[i];
    base[i] = 30 * log(t);
    // in feature order, so that theorems with equal statistics get equal
    // baselines
    for (const auto& fs : counts[i]) {
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
  if (dense)
    return score_nbayes(tfreq[i], sfreqs[i], move(symh), tfidf);
  // only near ties need it
  return score_nbayes(tfreq[i], sfreq(i), move(symh), tfidf);
}

void add_sym(unordered_map <feature_t, long> &m, feature_t sym, long w) {
  auto itr = m.find(sym);
  if (itr == m.end()) m[sym]  = w;
  else         (itr->second) += w;
}

// Theorem i's sfreq as NaiveBayes::learn added it up: from an empty map,
// [add](m, j, w) adding the features of theorem j to m with weight w, for
// each user u in [users, users_end) (in learn order), and for i itself with
// weight 1000 if [self], i learning itself before the theorems after it.
template <class Add>
unordered_map<feature_t, long> replay_sfreq(long i, const long* users,
  const long* users_end, bool self, Add add) {
  // a copy of an empty map, as each theorem's was before it learned
  static const unordered_map<feature_t, long> empty(100);
  unordered_map<feature_t, long> m(empty);
  for (; users != users_end; ++users) {
    if (self && *users >= i) {
      add(m, i, 1000);
      self = false;
    }
    add(m, *users, 1);
  }
  if (self) add(m, i, 1000);
  return m;
}

unordered_map<feature_t, long> NaiveBayes::sfreq(sample_t i) const {
  return replay_sfreq(i, users[i].data(), users[i].data() + users[i].size(),
    i < learned, [&](unordered_map<feature_t, long>& m, long j, long w) {
      for (const auto f : syms[j]) add_sym(m, f, w);
    });
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
    // Theorem i's sfreq, as the process builds it, in the same order, so
    // that it is iterated in the same order. Features are ranked by [table]
    // (View::rank_table), or if it is null by View::rank_of.
    unordered_map<feature_t, long> sfreq(long i, const LVec* table) const;
    double score(long i, map<feature_t, double> symh) const;
    // adds the features of theorem j to [m] with weight [w], as add_sym
    // does, ranked as sfreq ranks them
    void add(unordered_map<feature_t, long>& m, long j, long w,
      const LVec* table) const;
    // theorem i's counts (sfreq) in rank order, ranked by [table]; [acc] is
    // zero for each feature, as it is left
    vector<pair<feature_t, long> > counts(long i, LVec& acc,
      const LVec& table) const;
    // NaiveBayes::index, for the view
    void index(const LVec& table);

    const View& view;
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
: view(view), tfidf(view.features()), dense(dense) {
  const long n = view.size();
  const LVec table = view.rank_table(sym_num);
  for (const auto v : view.versions)
    for (const auto& s : v->syms) tfidf.add(ranks_in(s, table));

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
    for (long i = 0; i < n; ++i) sfreqs.push_back(sfreq(i, &table));
  else
    index(table);
}

void NaiveBayesView::add(unordered_map<feature_t, long>& m, long j,
  long w, const LVec* table) const {
  const auto at = view.locate(j);
  for (const auto f : at.first->syms[at.second])
    add_sym(m, table ? (*table)[f] : view.rank_of(f), w);
}

unordered_map<feature_t, long> NaiveBayesView::sfreq(long i,
  const LVec* table) const {
  return replay_sfreq(i, users.data() + user_start[i],
    users.data() + user_start[i + 1], true,
    [&](unordered_map<feature_t, long>& m, long j, long w) {
      add(m, j, w, table);
    });
}

double NaiveBayesView::score(long i, map<feature_t, double> symh) const {
  const long t = 1000 + user_start[i + 1] - user_start[i];
  if (dense)
    return score_nbayes(t, sfreqs[i], move(symh), tfidf);
  // only near ties need it, so features are ranked without a table
  return score_nbayes(t, sfreq(i, nullptr), move(symh), tfidf);
}

vector<pair<feature_t, long> > NaiveBayesView::counts(long i, LVec& acc,
  const LVec& table) const {
  LVec touched;
  const auto add = [&](long j, long w) {
    const auto at = view.locate(j);
    for (const auto f : at.first->syms[at.second]) {
      const long r = table[f];
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

void NaiveBayesView::index(const LVec& table) {
  const long n = view.size(), features = view.features();
  LVec acc(features, 0);

  post_start.assign(features + 1, 0);
  for (long i = 0; i < n; ++i)
    for (const auto& fs : counts(i, acc, table)) ++post_start[fs.first + 1];
  for (long f = 0; f < features; ++f)
    post_start[f + 1] += post_start[f];
  postings.resize(post_start[features]);

  vector<size_t> next(post_start.begin(), post_start.end() - 1);
  base.resize(n);
  for (long i = 0; i < n; ++i) {
    const long t = 1000 + user_start[i + 1] - user_start[i];
    base[i] = 30 * log(t);
    // in rank order, as NaiveBayes::index adds them up
    for (const auto& fs : counts(i, acc, table)) {
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
