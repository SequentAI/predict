#ifndef VERSIONS_CPP
#define VERSIONS_CPP

#include <numeric>
#include <set>
#include "scope.cpp"
#include "view.cpp"

// Numbers keys (features by hash, theorem ids) for the stored versions. A
// number is held by the versions having its key; once none does, the key is
// forgotten and its number given to the next new key, so that the tables
// stay as large as the most keys held at once. A version holds each number
// once.
template <class K>
struct Numbering {
  unordered_map<K, long> no;
  vector<K> key;
  LVec holders;
  LVec free;

  // the number of [k], numbering it if it is new; a new number must be held
  // before the next is released
  long number(const K& k) {
    auto got = no.find(k);
    if (got != no.end()) return got->second;
    long n = key.size();
    if (free.empty()) {
      key.push_back(k);
      holders.push_back(0);
    }
    else {
      n = free.back();
      free.pop_back();
      key[n] = k;
    }
    no[k] = n;
    return n;
  }

  void hold(long n) { ++holders[n]; }

  void release(long n) {
    if (--holders[n] > 0) return;
    no.erase(key[n]);
    K none{};
    swap(key[n], none);  // frees a string key's storage
    free.push_back(n);
  }

  long size() const { return key.size(); }
};

// The versions of theories a process stores and its views of them.
// Versions number features and theorem ids apart from the process's
// theorems, so that storing one changes nothing for the default view.
struct Views {
  map<pair<string, string>, Version> versions;  // by theory and id
  map<string, View> views;
  Numbering<uint64_t> features;  // by feature_hash
  Numbering<string> ids;
  long serial = 0;

  // the number of theorems of every stored version
  long theorems() const {
    long n = 0;
    for (const auto& v : versions) n += v.second.ids.size();
    return n;
  }

  // the ids a version names, each once: its theorems' and their
  // dependencies'
  static LVec named_ids(const Version& v) {
    LVec out(v.ids);
    for (const auto& ds : v.deps) out.insert(out.end(), ds.begin(), ds.end());
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    return out;
  }

  // drops the versions no view uses, and the numbers only they held
  void drop_unused() {
    set<const Version*> used;
    for (const auto& v : views)
      for (const auto& c : v.second.chosen) used.insert(c.second);
    for (auto v = versions.begin(); v != versions.end(); ) {
      if (used.count(&v->second)) {
        ++v;
        continue;
      }
      for (const auto f : v->second.appearance) features.release(f);
      for (const auto i : named_ids(v->second)) ids.release(i);
      v = versions.erase(v);
    }
  }
};

/* Reads the dependencies of each theorem of a version, numbered in [th_no],
   from lines of <deps>'s format, every line of a theorem of [th_no] (from
   [seqf]). A dependency need not be of the version's theory, so it is kept
   as an id, which a view resolves. */
vector<SVec> read_version_deps(const string& fname, const SLMap& th_no,
                               const string& seqf) {
  vector<SVec> deps(th_no.size());
  ifstream ic(fname);
  if (!ic)
    exit_error("Cannot read dependencies file `" + fname + "'!");
  string line;
  while (getline(ic, line)) {
    size_t pos = 0;
    const string thn = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");
    ++pos;
    auto got = th_no.find(thn);
    if (got == th_no.end())
      exit_error("Theorem `" + thn + "' of `" + fname + "' is not in `" +
                 seqf + "'!");
    while (pos < line.size()) {
      if (line[pos] == ' ') {
        ++pos;
        continue;
      }
      deps[got->second].push_back(read_id(line, pos, " "));
    }
  }
  if (ic.bad())
    exit_error("Cannot read dependencies file `" + fname + "'!");
  return deps;
}

/* Reads a version of [theory] from the files `<syms> <deps> <seq>
   <theories>` (command_files), in which every theorem must be of [theory],
   numbering its features and ids in [views]. */
Version read_version(Views& views, const string& theory,
                     const vector<string>& files) {
  const string &symsf = files[0], &depsf = files[1], &seqf = files[2],
    &theoriesf = files[3];
  SLMap th_no;
  SVec no_th;
  read_order(seqf, th_no, no_th);
  const long n = no_th.size();
  Theories theories;
  theories.resize(n);
  read_theories(theoriesf, th_no, theories);
  const long t = theories.find(theory);
  for (long i = 0; i < n; ++i)
    if (t == -1 || theories.of[i] != t)
      exit_error("Theorem `" + no_th[i] + "' is not of theory `" + theory +
                 "' in `" + theoriesf + "'!");
  // read_theories skips lines of theorems not in <seq> and keeps a
  // theorem's last line, so every line is checked here
  {
    ifstream ic(theoriesf);
    string line;
    while (getline(ic, line)) {
      size_t pos = 0;
      const string thn = read_id(line, pos, ":");
      if (!th_no.count(thn))
        exit_error("Theorem `" + thn + "' of `" + theoriesf +
                   "' is not in `" + seqf + "'!");
      if (line.substr(pos + 1, line.find(' ', pos) - pos - 1) != theory)
        exit_error("Theorem `" + thn + "' is not of theory `" + theory +
                   "' in `" + theoriesf + "'!");
    }
    if (ic.bad())
      exit_error("Cannot read theories file `" + theoriesf + "'!");
  }

  const vector<SVec> deps = read_version_deps(depsf, th_no, seqf);
  // features numbered in the order the file first names them
  LVecVec syms(n), sym_ths;
  long sym_num = 0;
  HLMap sym_no;
  SVec no_sym;
  read_syms(symsf, syms, sym_ths, sym_num, th_no, sym_no, no_sym);

  Version v;
  v.theory = theory;
  v.priv = theories.priv;
  for (long k = 0; k < sym_num; ++k)
    v.appearance.push_back(views.features.number(feature_hash(no_sym[k])));
  v.syms.resize(n);
  v.deps.resize(n);
  for (long i = 0; i < n; ++i) {
    v.ids.push_back(views.ids.number(no_th[i]));
    for (const auto k : syms[i]) v.syms[i].push_back(v.appearance[k]);
    for (const auto& d : deps[i]) v.deps[i].push_back(views.ids.number(d));
  }
  for (const auto f : v.appearance) views.features.hold(f);
  for (const auto i : Views::named_ids(v)) views.ids.hold(i);
  LVec by_feature(sym_num);
  iota(by_feature.begin(), by_feature.end(), 0);
  sort(by_feature.begin(), by_feature.end(),
    [&](long a, long b) { return v.appearance[a] < v.appearance[b]; });
  v.start.push_back(0);
  for (const auto k : by_feature) {
    v.feats.push_back(v.appearance[k]);
    v.ths.insert(v.ths.end(), sym_ths[k].begin(), sym_ths[k].end());
    v.start.push_back(v.ths.size());
  }
  return v;
}

// in_scope[i] says whether theorem i of [view] may be predicted under
// [scope], as scope_mask says for the theorems of a process.
vector<char> view_scope_mask(const Scope& scope, const View& view) {
  vector<char> in_scope(view.size(), 1);
  const unordered_set<string> excluded(scope.exclude.begin(),
    scope.exclude.end());
  for (size_t v = 0; v < view.versions.size(); ++v) {
    const Version& version = *view.versions[v];
    const bool own = version.theory == scope.theory;
    const bool out = excluded.count(version.theory);
    for (long j = 0; j < (long)version.ids.size(); ++j)
      if (out || (version.priv[j] && !own) ||
          (own && scope.before >= 0 && j >= scope.before))
        in_scope[view.first[v] + j] = 0;
  }
  return in_scope;
}

#endif
