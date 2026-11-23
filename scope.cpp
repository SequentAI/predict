#ifndef SCOPE_CPP
#define SCOPE_CPP

#include <functional>
#include <unordered_set>
#include "mesh.cpp"
#include "snapshot.cpp"

// The theory of each theorem, numbered in [no], -1 if it has none, and
// whether it is private to its theory.
struct Theories {
  SLMap no;
  vector<long> of;
  vector<char> priv;

  // gives the theorems past the last one with a slot no theory
  void resize(size_t n) {
    of.resize(n, -1);
    priv.resize(n, 0);
  }

  long number(const string& theory) {
    auto got = no.find(theory);
    if (got != no.end()) return got->second;
    const long t = no.size();
    no[theory] = t;
    return t;
  }

  // the number of [theory], -1 if no theorem was given it
  long find(const string& theory) const {
    auto got = no.find(theory);
    return got == no.end() ? -1 : got->second;
  }
};

// A theory name is nonempty and has no space, comma or `]`, which separate
// it from others in a scope.
bool valid_theory(const string& theory) {
  return !theory.empty() && theory.find_first_of(" ,]") == string::npos;
}

/* Reads lines of the format
   thmid:theory
   thmid:theory private
   into [theories], indexed by theorem. Ids are written as in <deps>;
   theorems without a line keep what they had. */
void read_theories(const string& fname, const SLMap& th_no,
                   Theories& theories) {
  ifstream ic(fname);
  if (!ic)
    exit_error("Cannot read theories file `" + fname + "'!");
  string line;
  while (getline(ic, line)) {
    size_t pos = 0;
    const string thn = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");
    ++pos;
    auto got = th_no.find(thn);
    if (got == th_no.end()) {
      cerr << "theory item missing in seq: " << thn << endl;
      continue;
    }
    const size_t space = line.find(' ', pos);
    const string theory = line.substr(pos, space - pos);
    const string flag = space == string::npos ? "" : line.substr(space + 1);
    if (!valid_theory(theory) || (flag != "" && flag != "private"))
      exit_error("Expected `id:theory' or `id:theory private', not `" +
                 line + "'!");
    theories.of[got->second] = theories.number(theory);
    theories.priv[got->second] = flag == "private";
  }
}

// The scope of a query: the theorems of [view] (the process's if empty),
// [k] predictions (-1 for -n's), none of [exclude]'s theorems, nor private
// ones unless of [theory], nor [theory]'s past its first [before] (-1 for
// none past); [ancestors]' theorems are boosted. [method] names the part of a
// -p mesh process that ranks the query (empty for the process's method).
// [limits] says whether it has a key other than view and method, [scoped]
// whether the query has a scope at all: brackets with a key other than
// method, or none.
struct Scope {
  string view;
  string method;
  long k = -1;
  string theory;
  long before = -1;
  vector<string> exclude, ancestors;
  bool limits = false;
  bool scoped = false;
};

long scope_number(const string& key, const string& value,
                  const string& line) {
  char *rest;
  errno = 0;
  const long n = strtol(value.c_str(), &rest, 10);
  if (value.empty() || *rest != '\0' || errno != 0 || n < 0 ||
      (key == "k" && n == 0))
    exit_error("Bad " + key + " `" + value + "' in query `" + line + "'!");
  return n;
}

vector<string> scope_theories(const string& key, const string& value,
                              const string& line) {
  vector<string> out;
  size_t pos = 0;
  while (true) {
    const size_t comma = value.find(',', pos);
    out.push_back(value.substr(pos, comma - pos));
    if (!valid_theory(out.back()))
      exit_error("Bad " + key + " `" + value + "' in query `" + line + "'!");
    if (comma == string::npos) return out;
    pos = comma + 1;
  }
}

/* Parses the scope that may begin a query line,
   [view=V method=M k=8 theory=T before=3 exclude=A,B ancestors=C,D]
   each key at most once, in any order, into [scope]; returns the position
   after it, 0 if the line has none. */
size_t parse_scope(const string& line, Scope& scope) {
  if (line.empty() || line[0] != '[') return 0;
  const size_t end = line.find(']');
  if (end == string::npos)
    exit_error("Expected `]' in query `" + line + "'!");
  unordered_set<string> seen;
  size_t pos = 1;
  while (pos < end) {
    if (line[pos] == ' ') {
      ++pos;
      continue;
    }
    const size_t stop = min(line.find(' ', pos), end);
    const string item = line.substr(pos, stop - pos);
    pos = stop;
    const size_t eq = item.find('=');
    const string key = item.substr(0, eq);
    const string value = eq == string::npos ? "" : item.substr(eq + 1);
    if (eq == string::npos || !seen.insert(key).second)
      exit_error("Bad scope item `" + item + "' in query `" + line + "'!");
    scope.limits = scope.limits || (key != "view" && key != "method");
    if (key == "view") {
      if (!valid_theory(value))
        exit_error("Bad view `" + value + "' in query `" + line + "'!");
      scope.view = value;
    }
    else if (key == "method") {
      if (value.empty())
        exit_error("Bad method `' in query `" + line + "'!");
      scope.method = value;
    }
    else if (key == "k") scope.k = scope_number(key, value, line);
    else if (key == "before") scope.before = scope_number(key, value, line);
    else if (key == "theory") {
      if (!valid_theory(value))
        exit_error("Bad theory `" + value + "' in query `" + line + "'!");
      scope.theory = value;
    }
    else if (key == "exclude")
      scope.exclude = scope_theories(key, value, line);
    else if (key == "ancestors")
      scope.ancestors = scope_theories(key, value, line);
    else
      exit_error("Unknown scope key `" + key + "' in query `" + line + "'!");
  }
  if (scope.before >= 0 && scope.theory.empty())
    exit_error("before needs theory in query `" + line + "'!");
  // method= alone, as no brackets, makes no scope
  scope.scoped = !(seen.size() == 1 && seen.count("method"));
  return end + 1;
}

// in_scope[i] says whether theorem i may be predicted under [scope].
vector<char> scope_mask(const Scope& scope, const Theories& theories) {
  const long n = theories.of.size();
  const long own = scope.theory.empty() ? -1 : theories.find(scope.theory);
  unordered_set<long> excluded;
  for (const auto& t : scope.exclude) excluded.insert(theories.find(t));
  vector<char> in_scope(n, 1);
  long seen_own = 0;
  for (long i = 0; i < n; ++i) {
    const long t = theories.of[i];
    if (t == -1) continue;
    if (excluded.count(t) || (theories.priv[i] && t != own))
      in_scope[i] = 0;
    if (t == own && scope.before >= 0 && seen_own++ >= scope.before)
      in_scope[i] = 0;
  }
  return in_scope;
}

// Re-ranks [ans] by the weight of each rank r (from 0), steep_weight(r),
// times [boost] for a theorem i with ancestor(i) set; equal weights keep
// their order.
void boost_ancestors(LDPairVec& ans, const function<bool(long)>& ancestor,
                     double boost) {
  vector<pair<double, long> > weighted;
  for (long r = 0; r < (long)ans.size(); ++r)
    weighted.push_back(make_pair(steep_weight(r) *
      (ancestor(ans[r].first) ? boost : 1.0), r));
  stable_sort(weighted.begin(), weighted.end(),
    [](const pair<double, long>& a, const pair<double, long>& b) {
      return a.first > b.first; });
  LDPairVec out;
  for (const auto& w : weighted) out.push_back(ans[w.second]);
  ans = out;
}

/* Removes the theorems of theory [t] from [corpus], [statures] and
   [theories], keeping the others in their order. Their features and types
   stay known, possibly with no theorem having them. */
void remove_theory(Corpus& corpus, vector<unsigned char>& statures,
                   Theories& theories, long t) {
  const long n = corpus.no_th.size();
  vector<long> to(n, -1);
  long m = 0;
  for (long i = 0; i < n; ++i)
    if (theories.of[i] != t) to[i] = m++;
  for (long i = 0; i < n; ++i) {
    if (to[i] == -1) continue;
    corpus.no_th[to[i]] = corpus.no_th[i];
    corpus.deps[to[i]] = corpus.deps[i];
    corpus.syms[to[i]] = corpus.syms[i];
    if (corpus.types) {
      corpus.typed[to[i]] = corpus.typed[i];
      corpus.freq_only[to[i]] = corpus.freq_only[i];
    }
    statures[to[i]] = statures[i];
    theories.of[to[i]] = theories.of[i];
    theories.priv[to[i]] = theories.priv[i];
  }
  corpus.no_th.resize(m);
  corpus.deps.resize(m);
  corpus.syms.resize(m);
  if (corpus.types) {
    corpus.typed.resize(m);
    corpus.freq_only.resize(m);
  }
  statures.resize(m);
  theories.of.resize(m);
  theories.priv.resize(m);
  auto renumber = [&](LVec& ths) {
    LVec out;
    for (const auto i : ths)
      if (to[i] != -1) out.push_back(to[i]);
    ths = out;
  };
  for (auto& ds : corpus.deps) renumber(ds);
  for (auto& ths : corpus.sym_ths) renumber(ths);
  corpus.th_no.clear();
  for (long i = 0; i < m; ++i) corpus.th_no[corpus.no_th[i]] = i;
}

/* The first theorem of theory [t] (-1 for none), or the number of theorems
   if it has none, if the corpus is untyped, its theorems are the last ones,
   no other theorem depends on one of them, and the features beyond the first
   [kept_syms] that its theorems were the first to have come after all the
   others and belong to no other theorem, so that removing them leaves the
   others, and the features' numbers, as a process given only the others
   has them; -1 otherwise. */
long last_theory_start(const Corpus& corpus, const Theories& theories,
                       long t, long kept_syms) {
  long from = theories.of.size();
  if (corpus.types) return -1;
  if (t == -1) return from;
  while (from > 0 && theories.of[from - 1] == t) --from;
  for (long i = 0; i < from; ++i) {
    if (theories.of[i] == t) return -1;
    for (const auto d : corpus.deps[i])
      if (d >= from) return -1;
  }
  // sym_ths lists a feature's theorems in the order <syms> gave them, so
  // its first theorem is the one that numbered it
  const long sym_num = corpus.sym_ths.size();
  long first_new = sym_num;
  for (long i = from; i < (long)corpus.syms.size(); ++i)
    for (const auto s : corpus.syms[i])
      if (s >= kept_syms && corpus.sym_ths[s][0] >= from)
        first_new = min(first_new, s);
  for (long s = first_new; s < sym_num; ++s)
    for (const auto i : corpus.sym_ths[s])
      if (i < from) return -1;
  return from;
}

/* Removes the theorems [from, n) from the untyped [corpus], [statures] and
   [theories], as last_theory_start allows, then the features at the end
   that no theorem has, down to [kept_syms] features. */
void drop_last_theory(Corpus& corpus, vector<unsigned char>& statures,
                      Theories& theories, long from, long kept_syms) {
  const long n = corpus.no_th.size();
  LVec dropped;
  for (long i = from; i < n; ++i) {
    dropped.insert(dropped.end(), corpus.syms[i].begin(),
      corpus.syms[i].end());
    corpus.th_no.erase(corpus.no_th[i]);
  }
  sort(dropped.begin(), dropped.end());
  dropped.erase(unique(dropped.begin(), dropped.end()), dropped.end());
  // not always at the end of the list: <syms> may give a feature to a
  // theorem before [from] after one at or past it
  for (const auto s : dropped) {
    LVec& ths = corpus.sym_ths[s];
    ths.erase(remove_if(ths.begin(), ths.end(),
      [&](long i) { return i >= from; }), ths.end());
  }
  corpus.no_th.resize(from);
  corpus.deps.resize(from);
  corpus.syms.resize(from);
  statures.resize(from);
  theories.of.resize(from);
  theories.priv.resize(from);
  while ((long)corpus.no_sym.size() > kept_syms &&
         corpus.sym_ths.back().empty()) {
    corpus.sym_no.erase(feature_hash(corpus.no_sym.back()));
    corpus.no_sym.pop_back();
    corpus.sym_ths.pop_back();
  }
}

#endif
