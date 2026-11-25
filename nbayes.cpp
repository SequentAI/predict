#include <algorithm>
#include <map>
#include <set>
#include "predictor.cpp"
#include "snapshot.cpp"
#include "view.cpp"

typedef long feature_t;
typedef long sample_t;
// (feature, count) pairs
typedef vector<pair<feature_t, long> > Counts;

// ln k for each k up to the largest asked for, from which every sparse score
// computes its logarithms, so that a process started on a snapshot and one
// started on text compute equal postings
class LnTable {
  public:
    // makes ln k available for each k <= [top]
    void cover(long top) {
      while ((long)v.size() <= top) v.push_back(log(double(v.size())));
    }
    double operator()(long k) const { return v[k]; }

  private:
    vector<double> v;
};

const double ln5 = log(5.0);

// a_{i,f} of a count [s] of a theorem used [t] times, for a feature of idf
// [idf] (NaiveBayes): idf (ln 5 + ln s - ln t + 18)
inline double nb_a(double idf, long s, long t, const LnTable& ln) {
  return idf * (ln5 + ln(s) - ln(t) + 18);
}

// b_{i,f} / idf_f for a count [s] of a theorem used [t] times:
// 0.2 (ln(t + 1 - s) - ln t), as s <= t
inline double nb_c(long s, long t, const LnTable& ln) {
  return 0.2 * (ln(t + 1 - s) - ln(t));
}

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
// (ln(5 s_f / t) + 18) (nb_a, nb_c), B_i = 30 ln t + sum_{f in S} b_{i,f} and
// C = -sum_{f in G} 18 w_f idf_f. B_i, and (i, a_{i,f}, b_{i,f}) in a
// posting list per feature f, are computed once after learning, so a query
// reads only the postings of its own features.
//
// The two forms round differently, so where the sparse one puts scores within
// a margin far wider than either's rounding error of each other, score()
// settles their order: predictions are those score() gives, ties included.
//
// Started on a snapshot, the sparse form learns only the theorems after it:
// the snapshot holds the counts of its own theorems (LearnedCounts), and
// their B_i and its postings are computed from them, B_i adjusted to the idf
// the later theorems change. As the two compute B_i in another order, -S
// scores may differ in the last bits from a process started on text.
class NaiveBayes : public Predictor {
  public:
    // A theorem's dependencies learn its features only for the first
    // [max_deps] of them (all if 0). [dense] predicts with score(). Unless
    // [dense], [mapped] (if not null) is what a snapshot of the corpus's
    // first theorems holds, which this learns from while they are unchanged
    // and it was learned with [max_deps].
    NaiveBayes(const Rows& deps, const Rows& syms, const Rows& sym_ths,
      long sym_num, long max_deps, bool dense,
      const LearnedCounts* mapped = nullptr);

    // Puts what it learned of the whole corpus, learned with learn_all(), in
    // the sections of a snapshot (LearnedCounts).
    void put_counts(vector<string>& s) const;

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
    // counts, unless [dense], which keeps no counts
    void gain(Counts& delta, sample_t u, long w) const {
      if (dense) return;
      for (const auto f : syms[u]) delta.push_back(make_pair(f, w));
    }
    // adds [delta], the changes of some theorems' counts, to them, merging
    // each theorem's once; with [dense], computes the sfreqs of the learned
    // ones again instead
    void apply(map<sample_t, Counts>& delta);
    // tfreq[i] of each theorem i, and counts[i] unless [dense], from the
    // users of the theorems [0, to) learned
    void count_all(sample_t to);
    // Appends theorem i's counts to [out] in increasing order of feature:
    // 1000 for each of its features if [self], 1 for each feature of each
    // of its users; [acc] (zero at every feature) and [fs] (empty) are
    // scratch, left as they were.
    void count(sample_t i, bool self, LVec& acc, LVec& fs, Counts& out) const;
    // Theorem i's sfreq: its counts in the order an unordered_map adding up
    // its own features and those of its users, in learn order, iterates
    // them, which score() sums in.
    unordered_map<feature_t, long> sfreq(sample_t i) const;
    void index();

    // Learns the theorems [from, to) after the snapshot's.
    void learn_after(sample_t from, sample_t to);
    // index(), for a process started on a snapshot: B_i of each snapshot
    // theorem that no later theorem learns as a dependency adjusted to the
    // idf, and B_i and postings computed for the others.
    void index_after();

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
    Rows users;
    // with [dense], each learned theorem's sfreq, which every query reads
    vector<unordered_map<feature_t, long> > sfreqs;

    Tfidf tfidf;
    long sym_num, max_deps;
    bool dense;
    const LearnedCounts* mapped;
    // the theorems [0, anchor) are the snapshot's, learned from [mapped]; 0
    // if it learned them all
    sample_t anchor = 0;
    // those of the snapshot's theorems that a later one learns as a
    // dependency; every one's postings are [mapped]'s, a touched one's
    // counts raised by its [later_counts]
    vector<char> touched;
    // each touched theorem's counts from its later users, in increasing order
    // of feature
    unordered_map<sample_t, Counts> later_counts;
    // the counts the snapshot gives some of its theorems, kept once a later
    // theorem learned them as a dependency, in increasing order of feature
    unordered_map<sample_t, vector<pair<int32_t, int32_t> > > anchor_counts;
    LnTable ln;

    struct Posting {
      sample_t i;
      double a, b;
    };
    // base[i] = B_i
    vector<double> base;
    // postings[post_start[f], post_start[f + 1]) are those of feature f, in
    // increasing order of i (with an anchor, only those of the theorems after
    // it and the counts a touched one has only from its later users)
    vector<Posting> postings;
    vector<size_t> post_start;
};

NaiveBayes::NaiveBayes(const Rows& deps, const Rows& syms,
  const Rows& sym_ths, long sym_num, long max_deps, bool dense,
  const LearnedCounts* mapped)
: Predictor(deps, syms, sym_ths, sym_num), tfidf(sym_num), sym_num(sym_num),
  max_deps(max_deps), dense(dense), mapped(mapped) {}

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
      users.push(deps[u][k], u);

  LVec acc(sym_num, 0), touched;
  for (sample_t i = 0; i < n; ++i) {
    tfreq[i] = (i < to ? 1000 : 0) + users[i].size();
    if (!dense) count(i, i < to, acc, touched, counts[i]);
  }
}

void NaiveBayes::count(sample_t i, bool self, LVec& acc, LVec& fs,
  Counts& out) const {
  const auto add = [&](sample_t j, long w) {
    for (const auto f : syms[j]) {
      if (acc[f] == 0) fs.push_back(f);
      acc[f] += w;
    }
  };
  if (self) add(i, 1000);
  for (const auto u : users[i]) add(u, 1);
  sort(fs.begin(), fs.end());
  for (const auto f : fs) {
    out.push_back(make_pair(f, acc[f]));
    acc[f] = 0;
  }
  fs.clear();
}

void NaiveBayes::learn(sample_t from, sample_t to) {
  if (learned == 0 && !dense && mapped && mapped->theorems > 0 &&
      mapped->max_deps == max_deps && to >= mapped->theorems) {
    anchor = mapped->theorems;
    sym_num = sym_ths.size();
    tfidf.resize(sym_num);
    vector<long> freq(sym_num, 0);
    for (long f = 0; f < mapped->features; ++f)
      freq[f] = mapped->freq_off[f + 1] - mapped->freq_off[f];
    tfidf.start(anchor, freq);
    users = mapped->users;
    tfreq.resize(anchor);
    for (sample_t i = 0; i < anchor; ++i) tfreq[i] = 1000 + users[i].size();
    learned = anchor;
    from = anchor;
  }
  if (anchor > 0) {
    learn_after(from, to);
    return;
  }
  sym_num = sym_ths.size();
  tfidf.resize(sym_num);
  const sample_t n = syms.size();
  tfreq.resize(n);
  if (!dense) counts.resize(n);
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
        users.push(d, u);
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
  if (from < anchor) return false;
  map<sample_t, Counts> delta;
  for (sample_t u = learned - 1; u >= from; --u) {
    for (size_t k = learned_deps(u); k-- > 0; ) {
      const sample_t d = deps[u][k];
      if (d >= from) continue;
      users.pop_back(d);
      --tfreq[d];
      gain(delta[d], u, -1);
    }
    tfidf.remove(syms[u]);
  }
  tfreq.resize(from);
  users.resize(from);
  learned = from;
  if (anchor > 0) return true;
  if (!dense) counts.resize(from);
  if (dense) sfreqs.resize(from);
  apply(delta);
  return true;
}

void NaiveBayes::learn_after(sample_t from, sample_t to) {
  sym_num = sym_ths.size();
  tfidf.resize(sym_num);
  const sample_t n = syms.size();
  tfreq.resize(n);
  users.resize(n);
  for (sample_t u = from; u < to; ++u) {
    tfidf.add(syms[u]);
    tfreq[u] += 1000;
    for (size_t k = 0; k < learned_deps(u); ++k) {
      const sample_t d = deps[u][k];
      users.push(d, u);
      ++tfreq[d];
    }
  }
  learned = to;
  index_after();
}

void NaiveBayes::index_after() {
  const sample_t n = tfreq.size();
  ln.cover(*max_element(tfreq.begin(), tfreq.end()));
  touched.assign(anchor, 0);
  // the theorems whose counts are computed here, in increasing order
  vector<sample_t> own;
  for (sample_t u = anchor; u < learned; ++u)
    for (size_t k = 0; k < learned_deps(u); ++k) {
      const sample_t d = deps[u][k];
      if (d < anchor && !touched[d]) {
        touched[d] = 1;
        own.push_back(d);
      }
    }
  sort(own.begin(), own.end());
  const size_t touched_n = own.size();
  for (sample_t i = anchor; i < n; ++i) own.push_back(i);

  // the snapshot's B_i at the idf of its theorems, plus the change of each
  // feature's idf times its sum of nb_c: the same for every feature the
  // later theorems lack
  base.resize(n);
  const double same = tfidf.log_theorems() - log(anchor);
  for (sample_t i = 0; i < anchor; ++i)
    if (!touched[i])
      base[i] = mapped->bases[2 * i] + same * mapped->bases[2 * i + 1];
  vector<long> later;
  for (sample_t u = anchor; u < learned; ++u)
    for (const auto f : syms[u])
      if (f < mapped->features) later.push_back(f);
  sort(later.begin(), later.end());
  later.erase(unique(later.begin(), later.end()), later.end());
  for (const auto f : later) {
    const double then = log(anchor) -
      log(mapped->freq_off[f + 1] - mapped->freq_off[f]);
    const double change = tfidf.get(f) - then - same;
    for_postings(*mapped, f, [&](sample_t i, long s) {
      if (!touched[i]) base[i] += change * nb_c(s, tfreq[i], ln);
      return true;
    });
  }

  // the snapshot's counts of the touched theorems not kept yet, in one pass
  // over its postings
  vector<char> wanted(anchor, 0);
  bool any = false;
  for (size_t k = 0; k < touched_n; ++k)
    if (!anchor_counts.count(own[k])) {
      wanted[own[k]] = 1;
      any = true;
    }
  for (long f = 0; any && f < mapped->features; ++f)
    for_postings(*mapped, f, [&](sample_t i, long s) {
      if (wanted[i]) anchor_counts[i].push_back(make_pair(f, s));
      return true;
    });
  for (size_t k = 0; k < touched_n; ++k)
    if (wanted[own[k]]) anchor_counts[own[k]].shrink_to_fit();

  // the counts, B_i and postings of the others, as count_all and index()
  // compute them; a touched theorem's postings are the snapshot's but for the
  // features only its later users have
  later_counts.clear();
  for (size_t k = 0; k < touched_n; ++k) {
    Counts gained;
    for (const auto u : users[own[k]])
      if (u >= anchor) gain(gained, u, 1);
    add_counts(later_counts[own[k]], move(gained));
  }
  vector<Counts> cs(own.size() - touched_n);
  LVec acc(sym_num, 0), fs;
  for (size_t k = touched_n; k < own.size(); ++k)
    count(own[k], own[k] < learned, acc, fs, cs[k - touched_n]);
  // calls fn(f, s, new) for each count s of feature f of own[k], in
  // increasing order of f; new if the snapshot has no count of f for it
  const auto for_counts = [&](size_t k, auto fn) {
    if (k >= touched_n) {
      for (const auto& c : cs[k - touched_n]) fn(c.first, c.second, true);
      return;
    }
    const auto& a = anchor_counts[own[k]];
    const Counts& d = later_counts[own[k]];
    auto p = a.begin();
    for (const auto& c : d) {
      for (; p != a.end() && p->first < c.first; ++p)
        fn(p->first, p->second, false);
      if (p != a.end() && p->first == c.first)
        fn(c.first, (p++)->second + c.second, false);
      else
        fn(c.first, c.second, true);
    }
    for (; p != a.end(); ++p) fn(p->first, p->second, false);
  };
  post_start.assign(sym_num + 1, 0);
  for (size_t k = 0; k < own.size(); ++k)
    for_counts(k, [&](long f, long, bool fresh) {
      if (fresh) ++post_start[f + 1];
    });
  for (long f = 0; f < sym_num; ++f)
    post_start[f + 1] += post_start[f];
  postings.resize(post_start[sym_num]);
  vector<size_t> next(post_start.begin(), post_start.end() - 1);
  for (size_t k = 0; k < own.size(); ++k) {
    const sample_t i = own[k];
    const long t = tfreq[i];
    base[i] = 30 * log(t);
    for_counts(k, [&](long f, long c, bool fresh) {
      const double idf = tfidf.get(f);
      const double b = idf * nb_c(c, t, ln);
      base[i] += b;
      if (fresh) postings[next[f]++] = {i, nb_a(idf, c, t, ln), b};
    });
  }
}

void NaiveBayes::put_counts(vector<string>& s) const {
  put_rows(users, s[USERS_OFFSETS], s[USERS]);
  const sample_t n = tfreq.size();
  vector<uint64_t> start(sym_num + 1, 0);
  for (sample_t i = 0; i < n; ++i)
    for (const auto& c : counts[i]) ++start[c.first + 1];
  for (long f = 0; f < sym_num; ++f) start[f + 1] += start[f];
  vector<int32_t> pairs(2 * start[sym_num]);
  vector<uint64_t> next(start.begin(), start.end() - 1);
  for (sample_t i = 0; i < n; ++i) {
    double sum = 0;
    for (const auto& c : counts[i]) {
      const uint64_t k = next[c.first]++;
      pairs[2 * k] = i;
      pairs[2 * k + 1] = c.second;
      sum += nb_c(c.second, tfreq[i], ln);
    }
    put(s[BASES], base[i]);
    put(s[BASES], sum);
  }
  SnapshotWriter w;
  for (long f = 0; f < sym_num; ++f) {
    put(s[POSTINGS_OFFSETS], uint64_t(w.buf.size()));
    long prev = -1;
    for (uint64_t k = start[f]; k < start[f + 1]; ++k) {
      w.number(pairs[2 * k] - prev - 1);
      w.number(pairs[2 * k + 1]);
      prev = pairs[2 * k];
    }
  }
  put(s[POSTINGS_OFFSETS], uint64_t(w.buf.size()));
  s[POSTINGS] = move(w.buf);
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
  if (n > 0) ln.cover(*max_element(tfreq.begin(), tfreq.end()));
  for (sample_t i = 0; i < n; ++i) {
    const long t = tfreq[i];
    base[i] = 30 * log(t);
    // in feature order, so that theorems with equal statistics get equal
    // baselines
    for (const auto& fs : counts[i]) {
      const double idf = tfidf.get(fs.first);
      const double b = idf * nb_c(fs.second, t, ln);
      base[i] += b;
      postings[next[fs.first]++] = {i, nb_a(idf, fs.second, t, ln), b};
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
    const double idf = tfidf.get(fw.first);
    c += w * idf * 18;
    // the snapshot's postings, as index() computes them
    if (anchor > 0 && fw.first < mapped->features)
      for_postings(*mapped, fw.first, [&](sample_t i, long s) {
        if (i >= maxth) return false;
        if (!visible(i, maxth)) return true;
        if (touched[i]) {
          // the count its later users add
          const Counts& d = later_counts.at(i);
          const auto got = lower_bound(d.begin(), d.end(),
            make_pair(fw.first, 0L));
          if (got != d.end() && got->first == fw.first) s += got->second;
        }
        const long t = tfreq[i];
        ans[i].second += w * nb_a(idf, s, t, ln) - idf * nb_c(s, t, ln);
        return true;
      });
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
template <class Users, class Add>
unordered_map<feature_t, long> replay_sfreq(long i, Users users,
  Users users_end, bool self, Add add) {
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
  const Row us = users[i];
  return replay_sfreq(i, us.begin(), us.end(), i < learned, [&](unordered_map<feature_t, long>& m, long j, long w) {
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
  LnTable ln;
  for (long i = 0; i < n; ++i) {
    const long t = 1000 + user_start[i + 1] - user_start[i];
    ln.cover(t);
    base[i] = 30 * log(t);
    // in rank order, as NaiveBayes::index adds them up
    for (const auto& fs : counts(i, acc, table)) {
      const double idf = tfidf.get(fs.first);
      const double b = idf * nb_c(fs.second, t, ln);
      base[i] += b;
      postings[next[fs.first]++] = {i, nb_a(idf, fs.second, t, ln), b};
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
