#ifndef FORMAT_CPP
#define FORMAT_CPP

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

typedef unordered_map<string, long> SLMap;
typedef vector<string> SVec;
typedef vector<long> LVec;
typedef vector<vector<long> > LVecVec;
typedef vector<pair<long, double> > LDPairVec;
typedef unordered_map<long, double> LDMap;
typedef unordered_map<uint64_t, long> HLMap;

/* Features are looked up by a 64-bit hash of their text (FNV-1a), so that a
   snapshot need not hold the text; features with equal hashes are one. */
uint64_t feature_hash(const string &f) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (const unsigned char c : f) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

void exit_error(string message) {
  cerr << message << endl;
  exit(1);
}

/* Given a file name, fills the given empty [th_no], [no_th], and sets th_num */
void read_order(const string &fname, SLMap &th_no, SVec &no_th) {
  ifstream ic(fname);
  string line;
  while (getline (ic, line)) {
    if (th_no.find (line) == th_no.end ()) {
      th_no[line] = no_th.size();
      no_th.push_back(line);
    }
    else
      exit_error("Duplicate `" + line + 
                 "' detected upon reading file `" + fname + ".");
  }
}

/* Reads a theorem id starting at [pos], up to the first character of [stops]
   or the end of [line], and leaves [pos] there. In <syms> and <deps> an id
   writes `\` as `\\` and `:` as `\:`. */
string read_id(const string &line, size_t &pos, const string &stops) {
  string id;
  while (pos < line.size() && stops.find(line[pos]) == string::npos) {
    char c = line[pos++];
    if (c == '\\') {
      if (pos == line.size() || (line[pos] != '\\' && line[pos] != ':'))
        exit_error("Bad escape in theorem id in line `" + line + "'!");
      c = line[pos++];
    }
    id.push_back(c);
  }
  return id;
}

// Exits: a line of [fname] is about theorem [th], which its command doesn't
// add.
void not_added(const string &fname, const string &th) {
  exit_error("A line of `" + fname + "' is about theorem `" + th +
             "', which the command does not add!");
}

/* Given a file name and trans, fills deps in an allocated vector of empty vectors.
   A line of a theorem before [first] is an error. */
void read_deps(const string &fname, SLMap &th_no, LVecVec &deps,
               long first = 0) {
  ifstream ic(fname);
  string line;
  while (getline (ic, line)) {
    size_t pos = 0;
    const string thn = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");
    ++pos;
    if (th_no.find (thn) == th_no.end ())
      cerr << "dep item missing in seq: " << thn << endl;
    else {
      long th = th_no[thn];
      if (th < first) not_added(fname, thn);
      while (pos < line.size()) {
        if (line[pos] == ' ') {
          ++pos;
          continue;
        }
        const string t = read_id(line, pos, " ");
        if (th_no.find (t) == th_no.end ())
          cerr << "dependency missing in seq: " << t << endl;
        else
          // a dependency may come later in <seq> than its dependent
          deps[th].push_back(th_no[t]);
      }
    }
  }
}

// Reads a quoted string whose opening quote [begin] has passed, decoding
// `\"`, `\\` and `\n`; leaves [begin] after the closing quote.
string read_quoted(string::iterator &begin, string::iterator end) {
  string result;
  while (true) {
    if (begin == end)
      exit_error("Features stop with double quote expected!");
    char c = *begin++;
    if (c == '"')
      return result;
    if (c == '\\') {
      if (begin == end)
        exit_error("Features stop with double quote expected!");
      c = *begin++;
      if (c == 'n')
        c = '\n';
      else if (c != '"' && c != '\\')
        exit_error(string("Unknown escape `\\") + c + "' in feature!");
    }
    result.push_back(c);
  }
}

// The inverse of read_quoted, without the quotes.
string escape_quoted(const string &s) {
  string result;
  for (const char c : s)
    if (c == '"' || c == '\\') {
      result.push_back('\\');
      result.push_back(c);
    }
    else if (c == '\n')
      result += "\\n";
    else
      result.push_back(c);
  return result;
}

// A type pattern as written: a type variable `_`, or a type constructor,
// by name, applied to patterns.
struct PatternText {
  bool var;
  string name;
  vector<PatternText> args;
};

// A feature as written, with its optional parts: a weight (queries), an
// occurrence count (<syms>), and a type: a kind, an order and patterns.
// A frequency-only token (`+n` in <syms>) counts its occurrences in
// frequencies without being a symbol of the theorem.
struct Token {
  string feature;
  double weight = 1.0;
  long count = 1;
  bool freq_only = false;
  // whether it has a type, or an occurrence count `*n`
  bool typed = false;
  char kind = 'c';
  long order = 1;
  vector<PatternText> patterns;
};

// Reads a decimal integer from 0 to 10^9 at [begin], leaving [begin] after
// it. The bound keeps the sums of counts over a corpus from overflowing.
long read_count(string::iterator &begin, string::iterator end,
  const string &what) {
  const long max_count = 1000000000;
  const auto start = begin;
  while (begin != end && isdigit((unsigned char)*begin)) ++begin;
  const string digits(start, begin);
  char *rest;
  errno = 0;
  const long n = strtol(digits.c_str(), &rest, 10);
  if (digits.empty() || *rest != '\0' || errno != 0 || n > max_count)
    exit_error("Bad " + what + " `" + digits + "' (at most 10^9)!");
  return n;
}

// Reads a type pattern at [begin]: `_`, or a quoted constructor name
// optionally followed by `(` patterns separated by `, ` `)`.
PatternText read_pattern(string::iterator &begin, string::iterator end) {
  if (begin != end && *begin == '_') {
    ++begin;
    return PatternText{true, "", {}};
  }
  if (begin == end || *begin != '"')
    exit_error("Type pattern expected: `_' or a quoted type constructor!");
  ++begin;
  PatternText p{false, read_quoted(begin, end), {}};
  if (begin == end || *begin != '(')
    return p;
  ++begin;
  while (true) {
    p.args.push_back(read_pattern(begin, end));
    if (begin != end && *begin == ')') {
      ++begin;
      return p;
    }
    if (end - begin < 2 || *begin != ',' || *(begin + 1) != ' ')
      exit_error("Comma and space or `)' expected after a type argument!");
    begin += 2;
  }
}

// Reads a type annotation whose `<` has passed: a kind (c, v, a or t), a
// space, an order, then a space before each pattern, and `>`.
void read_type(string::iterator &begin, string::iterator end, Token &t) {
  if (begin == end || string("cvat").find(*begin) == string::npos)
    exit_error("Type kind c, v, a or t expected after `<'!");
  t.typed = true;
  t.kind = *begin++;
  if (begin == end || *begin != ' ')
    exit_error("Space expected after the type kind!");
  ++begin;
  t.order = read_count(begin, end, "type order");
  while (begin != end && *begin == ' ') {
    ++begin;
    t.patterns.push_back(read_pattern(begin, end));
  }
  if (begin == end || *begin != '>')
    exit_error("`>' expected at the end of a type!");
  ++begin;
}

// parses strings of the format:
// "string1", "string2", ..., "stringn"
// where a string writes `"` as `\"`, `\` as `\\` and a newline as `\n`.
// An empty range is an empty list. A string may be followed by a type
// `<...>`, then, if [weighted], by `@w`, a finite non-negative weight (1.0
// if absent), and otherwise by `*n`, an occurrence count (1 if absent), or
// by `+n`, a frequency-only occurrence count (at least 1).
vector<Token> parse_string_list(string::iterator begin,
  string::iterator end, bool weighted) {
  vector<Token> result;

  while (begin != end) {
    if (*begin != '"')
      exit_error("Feature start with double quote expected!");

    ++begin;
    Token t;
    t.feature = read_quoted(begin, end);

    if (begin != end && *begin == '<') {
      ++begin;
      read_type(begin, end, t);
    }
    if (begin != end && *begin == '*') {
      if (weighted)
        exit_error("Occurrence counts are not allowed in queries!");
      ++begin;
      t.typed = true;
      t.count = read_count(begin, end, "occurrence count");
      // a constant's or variable's weight divides by its frequency's log
      if (t.count == 0 && t.kind != 'a' && t.kind != 't')
        exit_error("Occurrence count 0 of the constant or variable `" +
                   t.feature + "'!");
    }
    else if (begin != end && *begin == '+') {
      if (weighted)
        exit_error("Frequency-only counts are not allowed in queries!");
      ++begin;
      t.freq_only = true;
      t.count = read_count(begin, end, "frequency-only count");
      if (t.count == 0)
        exit_error("Frequency-only count 0 of `" + t.feature + "'!");
    }
    if (begin != end && *begin == '@') {
      if (!weighted)
        exit_error("Feature weights are allowed only in queries!");
      const auto stop = find(begin + 1, end, ',');
      const string ws(begin + 1, stop);
      char *rest;
      errno = 0;
      t.weight = strtod(ws.c_str(), &rest);
      if (ws.empty() || *rest != '\0' || errno != 0 || !isfinite(t.weight) ||
          t.weight < 0)
        exit_error("Bad feature weight `" + ws + "'!");
      begin = stop;
    }
    result.push_back(t);

    if (begin == end)
      break;
    const auto comma = begin;
    const auto space = comma + 1;
    if (space == end)
      exit_error("Space expected after comma!");
    else if (*comma != ',' || *space != ' ')
      exit_error("Comma and space expected after feature!");
    begin = space + 1;
    if (begin == end)
      exit_error("Feature start with double quote expected!");
  }

  return result;
}

// The type patterns read so far, numbered so that equal patterns, and equal
// (order, patterns) pairs, get one number each: comparing numbers compares
// them. All type variables are one pattern.
class TypeTable {
  public:
    // a pattern (constructor number, or -1 for a variable; argument
    // patterns), or an (order, patterns) pair
    typedef pair<long, LVec> Key;

    // the number of a pattern, numbering it if it is new
    long pattern(const PatternText &p) {
      LVec args;
      for (const auto &a : p.args) args.push_back(pattern(a));
      long ctor = -1;
      if (!p.var) {
        auto got = ctor_no.find(p.name);
        if (got == ctor_no.end())
          got = ctor_no.emplace(p.name, ctor_no.size()).first;
        ctor = got->second;
      }
      return number(nodes, node_no, make_pair(ctor, args));
    }

    // the number of an (order, patterns) pair, numbering it if it is new
    long ptype(long order, const vector<PatternText> &patterns) {
      LVec ps;
      for (const auto &p : patterns) ps.push_back(pattern(p));
      return number(ptypes, ptype_no, make_pair(order, ps));
    }

    long order(long pt) const { return ptypes[pt].first; }

    // Whether the patterns of ptype [p] generalize those of [q]: the type
    // patterns of Isabelle's sledgehammer_mepo.ML (match_ptype), where a
    // shorter list of instances is matched by its prefix.
    bool match_ptype(long p, long q) const {
      return match_patterns(ptypes[p].second, ptypes[q].second);
    }

    // The tables, for a snapshot: the constructor names, patterns and
    // (order, patterns) pairs, each by number.
    SVec ctors() const {
      SVec out(ctor_no.size());
      for (const auto& c : ctor_no) out[c.second] = c.first;
      return out;
    }
    const vector<Key>& patterns() const { return nodes; }
    const vector<Key>& ptype_keys() const { return ptypes; }

    // Fills an empty table with the tables ctors, patterns and ptype_keys
    // gave; returns false if they are not such tables: a number out of range
    // (a pattern's arguments come before it), or an entry twice.
    bool restore(const SVec& cs, const vector<Key>& ps,
      const vector<Key>& pts) {
      for (size_t i = 0; i < cs.size(); ++i)
        if (!ctor_no.emplace(cs[i], i).second) return false;
      for (size_t i = 0; i < ps.size(); ++i) {
        if (ps[i].first < -1 || ps[i].first >= (long)cs.size()) return false;
        for (const auto a : ps[i].second)
          if (a < 0 || a >= (long)i) return false;
        if (number(nodes, node_no, ps[i]) != (long)i) return false;
      }
      for (size_t i = 0; i < pts.size(); ++i) {
        for (const auto p : pts[i].second)
          if (p < 0 || p >= (long)ps.size()) return false;
        if (number(ptypes, ptype_no, pts[i]) != (long)i) return false;
      }
      return true;
    }

  private:
    static long number(vector<Key> &keys, map<Key, long> &numbers,
      const Key &k) {
      auto got = numbers.find(k);
      if (got != numbers.end()) return got->second;
      numbers[k] = keys.size();
      keys.push_back(k);
      return keys.size() - 1;
    }

    bool match_pattern(long p, long q) const {
      if (nodes[p].first < 0) return true;
      if (nodes[q].first < 0) return false;
      return nodes[p].first == nodes[q].first &&
        match_patterns(nodes[p].second, nodes[q].second);
    }

    bool match_patterns(const LVec &ps, const LVec &qs) const {
      for (size_t i = 0; ; ++i) {
        if (i == qs.size()) return true;
        if (i == ps.size()) return false;
        if (!match_pattern(ps[i], qs[i])) return false;
      }
    }

    SLMap ctor_no;
    // a pattern: (constructor number, or -1 for a variable; arguments)
    vector<Key> nodes;
    map<Key, long> node_no;
    // (order, patterns)
    vector<Key> ptypes;
    map<Key, long> ptype_no;
};

// A symbol of a theorem or query with its type: the number of its
// (order, patterns) in the TypeTable, its kind, and how often the theorem
// has it (in a query, its weight).
struct TypedSym {
  long sym;
  long ptype;
  char kind;
  long count;
  double weight;
};
typedef vector<vector<TypedSym> > TypedSyms;

// The typed symbol of [t]: a feature without a type is a constant of order
// 1 with no type patterns.
TypedSym typed_sym(long sym, const Token &t, TypeTable &types) {
  return TypedSym{sym, types.ptype(t.order, t.patterns), t.kind, t.count,
    t.weight};
}

// Adds [ts] to a theorem's typed symbols [ths], adding up the counts of a
// symbol repeated with an equal type.
void add_typed(vector<TypedSym> &ths, const TypedSym &ts) {
  auto same = find_if(ths.begin(), ths.end(), [&](const TypedSym& o) {
    return o.sym == ts.sym && o.ptype == ts.ptype; });
  if (same == ths.end()) ths.push_back(ts);
  else same->count += ts.count;
}

// The typed symbols of a -P corpus whose untyped features are kept apart
// (-p knn, mash or mesh): numbered on their own, as a corpus of the typed
// tokens alone numbers them, with the theorems that have each.
struct TypedTable {
  HLMap sym_no;
  SVec no_sym;
  LVecVec syms, sym_ths;
};

// parses strings of the format:
// thmid:"feature1", "feature2", ..., "featuren"
// A feature repeated in a theorem's lines, on one line or several, counts
// once. With [types], also reads each theorem's typed symbols into [typed]:
// a feature repeated with an equal type counts once, its occurrence counts
// added up; and its frequency-only tokens into [freq_only] alike,
// registering a new feature with no theorems. Without [types],
// frequency-only tokens are ignored. With [split] (and [types]), a token
// with a type or a count is only a typed symbol, its feature numbered in
// [split], and the others are only features. A line of a theorem before
// [first] is an error.
void read_syms(const string &fname, LVecVec &syms, LVecVec &sym_ths,
               long &sym_num, const SLMap &th_no,
               HLMap &sym_no, SVec& no_sym,
               TypeTable *types = nullptr, TypedSyms *typed = nullptr,
               TypedSyms *freq_only = nullptr, long first = 0,
               TypedTable *split = nullptr) {
  ifstream ic(fname);
  string line;
  while (getline(ic, line)) {
    size_t pos = 0;
    const string th_str = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");

    auto th_got = th_no.find(th_str);
    if (th_got == th_no.end())
      exit_error("Theorem `" + th_str + "' not found!");
    long th = th_got->second;
    if (th < first) not_added(fname, th_str);

    // the theorem's features from its earlier lines count as seen
    unordered_set<long> seen(syms[th].begin(), syms[th].end());
    unordered_set<long> seen_typed;
    if (split)
      seen_typed.insert(split->syms[th].begin(), split->syms[th].end());
    for (const auto& t : parse_string_list(line.begin() + pos + 1, line.end(), false)) {
      const string& f = t.feature;
      const uint64_t fh = feature_hash(f);
      if (split && (t.typed || t.freq_only)) {
        auto got = split->sym_no.find(fh);
        long s;
        if (got == split->sym_no.end()) {
          s = split->no_sym.size();
          split->sym_no[fh] = s;
          split->no_sym.push_back(f);
          split->sym_ths.push_back(LVec());
        }
        else s = got->second;
        if (t.freq_only) {
          add_typed((*freq_only)[th], typed_sym(s, t, *types));
          continue;
        }
        if (seen_typed.insert(s).second) {
          split->syms[th].push_back(s);
          split->sym_ths[s].push_back(th);
        }
        add_typed((*typed)[th], typed_sym(s, t, *types));
        continue;
      }
      auto ftr_got = sym_no.find(fh);
      if (t.freq_only) {
        if (!types || split) continue;
        long s;
        if (ftr_got == sym_no.end()) {
          s = sym_num;
          sym_ths.push_back(vector<long>());
          sym_no[fh] = sym_num++;
          no_sym.push_back(f);
        }
        else s = ftr_got->second;
        add_typed((*freq_only)[th], typed_sym(s, t, *types));
        continue;
      }
      long s;
      if (ftr_got == sym_no.end()) {
        s = sym_num;
        syms[th].push_back(sym_num);
        sym_ths.push_back(vector<long>(1,th));
        seen.insert(sym_num);

        // register new feature
        sym_no[fh] = sym_num++;
        no_sym.push_back(f);
      }
      else {
        s = ftr_got->second;
        if (seen.insert(s).second) {
          syms[th].push_back(s);
          sym_ths[s].push_back(th);
        }
      }
      if (types && !split) add_typed((*typed)[th], typed_sym(s, t, *types));
    }
  }
}

// Parses a query line into (feature, weight) pairs, dropping unknown
// features. A feature repeated in the query counts once, with its largest
// weight.
LDPairVec feature_list(const vector<Token>& tokens, const HLMap& ftr_no) {
  LDPairVec result;
  unordered_map<long, size_t> index;
  for (const auto& t : tokens) {
    auto got = ftr_no.find(feature_hash(t.feature));
    if (got == ftr_no.end())
      continue;
    auto seen = index.find(got->second);
    if (seen == index.end()) {
      index[got->second] = result.size();
      result.push_back(make_pair(got->second, t.weight));
    }
    else
      result[seen->second].second = max(result[seen->second].second, t.weight);
  }

  return result;
}

LDPairVec parse_feature_list(string::iterator begin, string::iterator end,
  const HLMap& ftr_no) {
  return feature_list(parse_string_list(begin, end, true), ftr_no);
}

// Parses a query line into typed symbols, dropping unknown features. A
// feature repeated with an equal type counts once, with its largest weight.
vector<TypedSym> typed_query(const vector<Token>& tokens, const HLMap& ftr_no,
  TypeTable& types) {
  vector<TypedSym> result;
  for (const auto& t : tokens) {
    auto got = ftr_no.find(feature_hash(t.feature));
    if (got == ftr_no.end())
      continue;
    const TypedSym ts = typed_sym(got->second, t, types);
    auto same = find_if(result.begin(), result.end(), [&](const TypedSym& o) {
      return o.sym == ts.sym && o.ptype == ts.ptype; });
    if (same == result.end()) result.push_back(ts);
    else same->weight = max(same->weight, ts.weight);
  }
  return result;
}

vector<TypedSym> parse_typed_query(string::iterator begin,
  string::iterator end, const HLMap& ftr_no, TypeTable& types) {
  return typed_query(parse_string_list(begin, end, true), ftr_no, types);
}

// Parses a query line of a corpus whose typed symbols are numbered in
// [split]: its tokens with a type as parse_typed_query into [typed], and the
// others as parse_feature_list into [untyped].
void parse_split_query(string::iterator begin, string::iterator end,
  const HLMap& ftr_no, const TypedTable& split, TypeTable& types,
  LDPairVec& untyped, vector<TypedSym>& typed) {
  vector<Token> plain, with_type;
  for (auto& t : parse_string_list(begin, end, true))
    (t.typed ? with_type : plain).push_back(move(t));
  untyped = feature_list(plain, ftr_no);
  typed = typed_query(with_type, split.sym_no, types);
}

// Fact statures, as bit flags: a fact may be a simp rule, local to the goal's
// theory, both or neither.
const unsigned char STATURE_SIMP = 1, STATURE_LOCAL = 2;

/* Reads the statures of theorems from lines of the format
   thmid:flag1 flag2 ...
   where a flag is `simp` or `local`, into [statures], indexed by theorem.
   Theorems without a line have no flag. */
void read_statures(const string &fname, const SLMap &th_no,
                   vector<unsigned char> &statures) {
  ifstream ic(fname);
  if (!ic)
    exit_error("Cannot read statures file `" + fname + "'!");
  string line;
  while (getline(ic, line)) {
    size_t pos = 0;
    const string thn = read_id(line, pos, ":");
    if (pos == line.size())
      exit_error("Expected colon in line `" + line + "'!");
    ++pos;
    auto got = th_no.find(thn);
    if (got == th_no.end()) {
      cerr << "stature item missing in seq: " << thn << endl;
      continue;
    }
    while (pos < line.size()) {
      if (line[pos] == ' ') {
        ++pos;
        continue;
      }
      const size_t stop = min(line.find(' ', pos), line.size());
      const string flag = line.substr(pos, stop - pos);
      if (flag == "simp")
        statures[got->second] |= STATURE_SIMP;
      else if (flag == "local")
        statures[got->second] |= STATURE_LOCAL;
      else
        exit_error("Unknown stature `" + flag + "' in line `" + line + "'!");
      pos = stop;
    }
  }
}

void read_eval(const string &fname, const unordered_map<string, long> &th_no,
               unordered_set<long> &eval) {
  ifstream ic(fname);
  string line;
  while (getline(ic, line)) {
    auto get = th_no.find (line);
    if (get == th_no.end())
      exit_error("Could not find theorem `" + line +
                 "' in file `" + fname + "'.");
    else
      eval.insert(get->second);
  }
}

#endif

