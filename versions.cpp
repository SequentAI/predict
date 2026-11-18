#ifndef VERSIONS_CPP
#define VERSIONS_CPP

#include <numeric>
#include <set>
#include "scope.cpp"
#include "view.cpp"

// The versions of theories a process stores and its views of them.
struct Views {
  map<pair<string, string>, Version> versions;  // by theory and id
  map<string, View> views;
  SLMap id_no;   // the number of each theorem id versions name
  SVec no_id;
  long serial = 0;

  long id_number(const string& id) {
    auto got = id_no.find(id);
    if (got != id_no.end()) return got->second;
    id_no[id] = no_id.size();
    no_id.push_back(id);
    return no_id.size() - 1;
  }

  // the number of theorems of every stored version
  long theorems() const {
    long n = 0;
    for (const auto& v : versions) n += v.second.ids.size();
    return n;
  }

  // drops the versions no view uses
  void drop_unused() {
    set<const Version*> used;
    for (const auto& v : views)
      for (const auto& c : v.second.chosen) used.insert(c.second);
    for (auto v = versions.begin(); v != versions.end(); )
      if (used.count(&v->second)) ++v;
      else v = versions.erase(v);
  }
};

/* Reads the dependencies of [version], whose theorems are numbered in
   [th_no], from lines of <deps>'s format. A dependency need not be of the
   version's theory, so it is kept as an id, which a view resolves. */
void read_version_deps(const string& fname, const SLMap& th_no,
                       Views& views, Version& version) {
  ifstream ic(fname);
  string line;
  while (getline(ic, line)) {
    size_t pos = 0;
    const string thn = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");
    ++pos;
    auto got = th_no.find(thn);
    if (got == th_no.end()) {
      cerr << "dep item missing in seq: " << thn << endl;
      continue;
    }
    while (pos < line.size()) {
      if (line[pos] == ' ') {
        ++pos;
        continue;
      }
      version.deps[got->second].push_back(
        views.id_number(read_id(line, pos, " ")));
    }
  }
}

/* Reads a version of [theory] from the files `<syms> <deps> <seq>
   <theories>` (command_files), in which every theorem must be of [theory].
   New features are added to [corpus]'s tables, with no theorem of its own. */
Version read_version(Views& views, Corpus& corpus, const string& theory,
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

  Version v;
  v.theory = theory;
  v.priv = theories.priv;
  v.syms.resize(n);
  v.deps.resize(n);
  read_version_deps(depsf, th_no, views, v);

  // features numbered in the order the file first names them
  LVecVec syms(n), sym_ths;
  long sym_num = 0;
  HLMap sym_no;
  SVec no_sym;
  read_syms(symsf, syms, sym_ths, sym_num, th_no, sym_no, no_sym);

  for (long k = 0; k < sym_num; ++k) {
    const uint64_t h = feature_hash(no_sym[k]);
    auto got = corpus.sym_no.find(h);
    if (got == corpus.sym_no.end()) {
      got = corpus.sym_no.emplace(h, corpus.no_sym.size()).first;
      corpus.no_sym.push_back(no_sym[k]);
      corpus.sym_ths.push_back(LVec());
    }
    v.appearance.push_back(got->second);
  }
  for (long i = 0; i < n; ++i) {
    v.ids.push_back(views.id_number(no_th[i]));
    for (const auto k : syms[i]) v.syms[i].push_back(v.appearance[k]);
  }
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
