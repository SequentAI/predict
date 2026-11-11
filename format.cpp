#ifndef FORMAT_CPP
#define FORMAT_CPP

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

/* Given a file name and trans, fills deps in an allocated vector of empty vectors */
void read_deps(const string &fname, SLMap &th_no, LVecVec &deps) {
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

// parses strings of the format:
// "string1", "string2", ..., "stringn"
// where a string writes `"` as `\"`, `\` as `\\` and a newline as `\n`.
// An empty range is an empty list. If [weighted], a string may be followed by
// `@w`, a finite non-negative weight; strings without one weigh 1.0.
vector<pair<string, double> > parse_string_list(string::iterator begin,
  string::iterator end, bool weighted) {
  vector<pair<string, double> > result;

  while (begin != end) {
    if (*begin != '"')
      exit_error("Feature start with double quote expected!");

    string feature;
    ++begin;
    while (true) {
      if (begin == end)
        exit_error("Features stop with double quote expected!");
      char c = *begin++;
      if (c == '"')
        break;
      if (c == '\\') {
        if (begin == end)
          exit_error("Features stop with double quote expected!");
        c = *begin++;
        if (c == 'n')
          c = '\n';
        else if (c != '"' && c != '\\')
          exit_error(string("Unknown escape `\\") + c + "' in feature!");
      }
      feature.push_back(c);
    }

    double weight = 1.0;
    if (begin != end && *begin == '@') {
      if (!weighted)
        exit_error("Feature weights are allowed only in queries!");
      const auto stop = find(begin + 1, end, ',');
      const string ws(begin + 1, stop);
      char *rest;
      errno = 0;
      weight = strtod(ws.c_str(), &rest);
      if (ws.empty() || *rest != '\0' || errno != 0 || !isfinite(weight) ||
          weight < 0)
        exit_error("Bad feature weight `" + ws + "'!");
      begin = stop;
    }
    result.push_back(make_pair(feature, weight));

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

// parses strings of the format:
// thmid:"feature1", "feature2", ..., "featuren"
// A feature repeated on a line counts once.
void read_syms(const string &fname, LVecVec &syms, LVecVec &sym_ths,
               long &sym_num, const SLMap &th_no,
               SLMap &sym_no, SVec& no_sym) {
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

    unordered_set<long> seen;
    for (const auto& fw : parse_string_list(line.begin() + pos + 1, line.end(), false)) {
      const string& f = fw.first;
      auto ftr_got = sym_no.find(f);
      if (ftr_got == sym_no.end()) {
        syms[th].push_back(sym_num);
        sym_ths.push_back(vector<long>(1,th));
        seen.insert(sym_num);

        // register new feature
        sym_no[f] = sym_num++;
        no_sym.push_back(f);
      }
      else if (seen.insert(ftr_got->second).second) {
        syms[th].push_back(ftr_got->second);
        sym_ths[ftr_got->second].push_back(th);
      }
    }
  }
}

// Parses a query line into (feature, weight) pairs, dropping unknown
// features. A feature repeated in the query counts once, with its largest
// weight.
LDPairVec parse_feature_list(string::iterator begin, string::iterator end,
  const SLMap& ftr_no) {
  LDPairVec result;
  unordered_map<long, size_t> index;
  for (const auto& fw : parse_string_list(begin, end, true)) {
    auto got = ftr_no.find(fw.first);
    if (got == ftr_no.end())
      continue;
    auto seen = index.find(got->second);
    if (seen == index.end()) {
      index[got->second] = result.size();
      result.push_back(make_pair(got->second, fw.second));
    }
    else
      result[seen->second].second = max(result[seen->second].second, fw.second);
  }

  return result;
}

void read_eval(const string &fname, unordered_map<string, long> &th_no,
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

