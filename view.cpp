#ifndef VIEW_CPP
#define VIEW_CPP

#include <map>
#include <memory>
#include "format.cpp"

// A version of a theory, stored by `!version`: its theorems as read from its
// files. Features are numbered as in the rest of the process, and theorem ids
// (also those of dependencies, which may be of other theories) in a table of
// their own.
struct Version {
  string theory;
  long serial = 0;      // versions are ordered by when they were stored
  LVec ids;             // each theorem's id, in <seq> order
  vector<char> priv;    // whether each theorem is private
  LVecVec syms;         // each theorem's features, in <syms> order
  LVecVec deps;         // each theorem's dependencies
  LVec appearance;      // the features in the order <syms> first names them
  // The theorems having feature feats[k] are ths[start[k]], ...,
  // ths[start[k + 1] - 1], in the order <syms> names them; feats is
  // increasing.
  LVec feats, start, ths;

  // the range of the theorems having feature f
  pair<const long*, const long*> theorems_of(long f) const {
    const long k = lower_bound(feats.begin(), feats.end(), f) - feats.begin();
    if (k == (long)feats.size() || feats[k] != f)
      return make_pair(nullptr, nullptr);
    return make_pair(ths.data() + start[k], ths.data() + start[k + 1]);
  }
};

// What a selector learned from the theorems of one view.
class ViewPredictor {
  public:
    virtual ~ViewPredictor() {}
    // Ranks the view's theorems for a query of (feature, weight) pairs, each
    // a feature some theorem of the view has, as Predictor::predict_weighted
    // ranks the theorems of the process; with [in_scope], only the theorems
    // i with in_scope[i] set.
    virtual LDPairVec predict_weighted(const LDPairVec& query,
      const vector<char>* in_scope, long no_adv) const = 0;
};

// A view, named by `!view`: one version for each of some theories. Its
// theorems are those of its versions, in the order the versions were stored,
// each version's in <seq> order: the corpus of a process started on them.
struct View {
  map<string, const Version*> chosen;  // the version of each theory
  vector<const Version*> versions;     // in the order they were stored
  // the number in the view of each version's first theorem, then the number
  // of theorems
  LVec first;
  // (f, r) for each feature f of the view's theorems, by f, where r is the
  // number a process started on the view's theorems would give f: features
  // numbered from 0 in the order their <syms> first name them. Selectors
  // number the view's statistics by rank and add up a theorem's features in
  // rank order, as that process would, so that the sums round alike.
  vector<pair<long, long> > ranks;
  unique_ptr<ViewPredictor> predictor;

  long size() const { return first.back(); }

  // the version of theorem i of the view, and i's number in it
  pair<const Version*, long> locate(long i) const {
    const long v =
      upper_bound(first.begin(), first.end(), i) - first.begin() - 1;
    return make_pair(versions[v], i - first[v]);
  }

  // the number of features of the view's theorems
  long features() const { return ranks.size(); }

  // the rank of feature f, -1 if no theorem of the view has it
  long rank_of(long f) const {
    auto got = lower_bound(ranks.begin(), ranks.end(), make_pair(f, -1L));
    return got != ranks.end() && got->first == f ? got->second : -1;
  }

  // the rank of each feature numbered below sym_num, -1 for features no
  // theorem of the view has: a table for learning, kept no longer
  LVec rank_table(long sym_num) const {
    LVec table(sym_num, -1);
    for (const auto& fr : ranks) table[fr.first] = fr.second;
    return table;
  }

  // the number in the view of each theorem id numbered up to the largest of
  // the view's theorems, -1 for the ids of no theorem of the view
  LVec id_table() const {
    long top = -1;
    for (const auto v : versions)
      for (const auto id : v->ids) top = max(top, id);
    LVec table(top + 1, -1);
    for (size_t v = 0; v < versions.size(); ++v)
      for (size_t j = 0; j < versions[v]->ids.size(); ++j)
        table[versions[v]->ids[j]] = first[v] + j;
    return table;
  }

  // the dependencies of theorem i that are theorems of the view, numbered
  // in it, in order, as a process started on the view's theorems reads them;
  // [table] is id_table()
  LVec deps_of(long i, const LVec& table) const {
    const auto at = locate(i);
    LVec out;
    for (const auto d : at.first->deps[at.second])
      if (d < (long)table.size() && table[d] >= 0) out.push_back(table[d]);
    return out;
  }

  // Sets versions, first and ranks from chosen, for the sym_num features of
  // the versions.
  void arrange(long sym_num) {
    versions.clear();
    for (const auto& c : chosen) versions.push_back(c.second);
    sort(versions.begin(), versions.end(),
      [](const Version* a, const Version* b) { return a->serial < b->serial; });
    first.assign(1, 0);
    LVec table(sym_num, -1);
    long next = 0;
    for (const auto v : versions) {
      first.push_back(first.back() + v->ids.size());
      for (const auto f : v->appearance)
        if (table[f] < 0) table[f] = next++;
    }
    // a fresh vector, so that it is sized to this view, not a past one
    vector<pair<long,long>> compact;
    for (long f = 0; f < sym_num; ++f)
      if (table[f] >= 0) compact.push_back(make_pair(f, table[f]));
    ranks = move(compact);
  }
};

// Drops the theorems not in scope from [ans], if [in_scope] is set; returns
// no_adv, or the number left if fewer (as Predictor::keep_visible).
inline long keep_in_scope(LDPairVec& ans, const vector<char>* in_scope,
  long no_adv) {
  if (in_scope)
    ans.erase(remove_if(ans.begin(), ans.end(),
      [&](const pair<long, double>& p) { return !(*in_scope)[p.first]; }),
      ans.end());
  return min(no_adv, (long)ans.size());
}

// the ranks of features [syms] in [table] (View::rank_table)
inline LVec ranks_in(const LVec& syms, const LVec& table) {
  LVec out;
  for (const auto f : syms) out.push_back(table[f]);
  return out;
}

#endif
