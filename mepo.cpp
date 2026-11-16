#include <algorithm>
#include "predictor.cpp"

// The relevance filter of Meng and Paulson ("Lightweight relevance filtering
// for machine-generated resolution problems", J. Applied Logic 7(1), 2009) as
// Isabelle's Sledgehammer runs it (src/HOL/Tools/Sledgehammer/
// sledgehammer_mepo.ML, by Jia Meng and Jasmin Blanchette; Blanchette et al.,
// "Hammering towards QED", J. Formalized Reasoning 9(1), 2016), over untyped
// symbols: every feature is a symbol, matched by its full name (a constant
// written `Thy$Name` matches only `Thy$Name`, at any type).
//
// A fact's score is rel / (rel + irrel), where rel sums the weights of its
// symbols in the relevant set and irrel those of its other symbols, less a
// bonus for the fact's stature. The relevant set starts as the goal's
// symbols. Each iteration accepts the facts scoring at least the threshold,
// at most ceil(11.5 ^ (remaining / max_facts)) of them below 0.99999; their
// symbols join the relevant set, and every accepted fact moves the threshold
// toward 0.85. Facts are ranked in the order they are accepted.
//
// Untyped, every symbol is matched by its whole name, and the parts of
// Sledgehammer that need types or terms are left out: the 1.05 ^ (order - 1)
// factor on a higher-order symbol's irrelevant weight (every symbol weighs
// as order 1), the 1.5 multiplier on free variables, and lambda and theory
// pseudo-constants. Typed (predict_typed), symbols carry their kind, order
// and type patterns, and all of these are in. Left out either way: the
// special treatment of set constants, and the facts Sledgehammer inserts
// afterwards (ext, set axioms).
class MePo : public Predictor {
  public:
    // [types], [typed], the theorems' typed symbols, and [freq_only],
    // their occurrences counted only in frequencies, are given for
    // predict_typed and null otherwise.
    MePo(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, const vector<unsigned char>& statures,
      const TypeTable* types = nullptr, const TypedSyms* typed = nullptr,
      const TypedSyms* freq_only = nullptr);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    // A query feature weighing less than 1 occurs only in the goal's
    // assumptions; one weighing 1 or more occurs in its conclusion.
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;
    // MePo over typed symbols, with the query's weights as in
    // predict_weighted.
    LDPairVec predict_typed(const vector<TypedSym>& query, long maxth,
      long no_adv) const;

  private:
    // the weight of a symbol in the relevant set
    double rel_weight(long s) const;
    // the weight of a symbol outside the relevant set
    double irrel_weight(long s) const;
    // the score of fact i against the relevant set [rel]; [chained] marks
    // the symbols that occur only in the goal's assumptions
    double fact_weight(long i, const vector<char>& rel,
      const vector<char>& chained) const;

    // The typed weights and score: [rel] and [chained] hold the ptypes of
    // each symbol in the relevant set and in the goal's assumptions only.
    double typed_rel_weight(const TypedSym& s, long freq) const;
    double typed_irrel_weight(const TypedSym& s, long freq,
      const LVecVec& chained) const;
    double typed_fact_weight(long i, const LVecVec& rel,
      const LVecVec& chained) const;
    double stature_bonus(long i) const;

    // The iterations of the filter, for at most no_adv facts: [weigh](i)
    // scores fact i against the relevant set, and [accept](i, fresh) adds
    // fact i's symbols to it, appending those it changed to [fresh].
    template <class Weigh, class Accept>
    LDPairVec filter(long maxth, long no_adv, Weigh weigh,
      Accept accept) const;

    vector<unsigned char> statures;
    // the number of learned facts containing each symbol
    vector<long> freq;

    const TypeTable* types;
    const TypedSyms* typed;
    const TypedSyms* freq_only;
    // the learned occurrences of each symbol, by ptype
    vector<vector<pair<long, long> > > type_counts;
    // for each typed symbol of each fact, the occurrences of its symbol at
    // a ptype it generalizes, and at a ptype that generalizes it
    vector<vector<pair<long, long> > > typed_freq;
};

// sledgehammer_mepo.ML's default_relevance_fudge and Sledgehammer's default
// fact_thresholds
namespace mepo_fudge {
  const double local_const_multiplier = 1.5;
  const double worse_irrel_freq = 100.0;
  const double higher_order_irrel_weight = 1.05;
  const double abs_rel_weight = 0.5;
  const double abs_irrel_weight = 2.0;
  const double theory_const_rel_weight = 0.5;
  const double theory_const_irrel_weight = 0.25;
  const double chained_const_irrel_weight = 0.25;
  const double simp_bonus = 0.15;
  const double local_bonus = 0.55;
  const double max_imperfect = 11.5;
  const double max_imperfect_exp = 1.0;
  const double threshold_divisor = 2.0;
  const double ridiculous_threshold = 0.1;
  const double threshold0 = 0.45;
  const double threshold1 = 0.85;
  const double perfect = 0.99999;
  // facts scoring below hopeless are dropped at iteration hopeless_iter
  const double hopeless = 0.001;
  const long hopeless_iter = 5;
}

MePo::MePo(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
  long sym_num, const vector<unsigned char>& statures,
  const TypeTable* types, const TypedSyms* typed, const TypedSyms* freq_only)
: Predictor(deps, syms, sym_ths, sym_num), statures(statures),
  freq(sym_num, 0), types(types), typed(typed), freq_only(freq_only),
  type_counts(typed ? sym_num : 0) {}

void MePo::learn(long from, long to) {
  for (long i = from; i < to; ++i)
    for (const auto s : syms[i])
      ++freq[s];
  if (!typed) return;

  // the symbols' occurrences, including those only counted in frequencies
  // (sledgehammer_mepo.ML counts every constant occurrence of a fact, the
  // quantifiers that are not its symbols among them)
  auto count = [&](const TypedSym& ts) {
    auto& counts = type_counts[ts.sym];
    auto same = find_if(counts.begin(), counts.end(),
      [&](const pair<long, long>& c) { return c.first == ts.ptype; });
    if (same == counts.end()) counts.push_back(make_pair(ts.ptype, ts.count));
    else same->second += ts.count;
  };
  for (long i = from; i < to; ++i) {
    for (const auto& ts : (*typed)[i]) count(ts);
    if (freq_only)
      for (const auto& ts : (*freq_only)[i]) count(ts);
  }

  // A symbol's frequency at a ptype sums its occurrences at the ptypes the
  // ptype generalizes (as a relevant symbol) or that generalize it (as an
  // irrelevant one): sledgehammer_mepo.ML's pconst_freq.
  map<pair<long, long>, pair<long, long> > memo;
  typed_freq.assign(typed->size(), {});
  for (size_t i = 0; i < typed->size(); ++i)
    for (const auto& ts : (*typed)[i]) {
      const auto key = make_pair(ts.sym, ts.ptype);
      auto got = memo.find(key);
      if (got == memo.end()) {
        long below = 0, above = 0;
        for (const auto& c : type_counts[ts.sym]) {
          if (types->match_ptype(ts.ptype, c.first)) below += c.second;
          if (types->match_ptype(c.first, ts.ptype)) above += c.second;
        }
        got = memo.emplace(key, make_pair(below, above)).first;
      }
      typed_freq[i].push_back(got->second);
    }
}

// Rare symbols weigh more: 1 + 2 / ln(freq + 1).
double MePo::rel_weight(long s) const {
  return 1.0 + 2.0 / log((double)freq[s] + 1.0);
}

// Very rare and very common symbols weigh less than those of frequency
// around 100: rare ones cannot pull in many facts, and common ones say
// little. The weight rises as ln(freq + 1) / ln 100 up to frequency 99, is 1
// at 100, and falls as the relevant weight does beyond.
double MePo::irrel_weight(long s) const {
  using namespace mepo_fudge;
  const long k = (long)ceil(worse_irrel_freq);
  if (freq[s] < k)
    return log((double)freq[s] + 1.0) / log(worse_irrel_freq);
  const double relk = 1.0 + 2.0 / log((double)k + 1.0);
  return rel_weight(s) / relk;
}

double MePo::fact_weight(long i, const vector<char>& rel,
  const vector<char>& chained) const {
  using namespace mepo_fudge;
  double relw = 0, irrelw = 0;
  bool any_rel = false;
  for (const auto s : syms[i])
    if (rel[s]) {
      any_rel = true;
      relw += rel_weight(s);
    }
    else
      irrelw += irrel_weight(s) * (chained[s] ? chained_const_irrel_weight : 1.0);
  if (!any_rel) return 0;
  // a simp rule's bonus takes precedence over a local fact's
  if (statures[i] & STATURE_SIMP) irrelw -= simp_bonus;
  else if (statures[i] & STATURE_LOCAL) irrelw -= local_bonus;
  const double res = relw / (relw + irrelw);
  return isfinite(res) ? res : 0;
}

LDPairVec MePo::predict(const LVec& csyms, long maxth, long no_adv) const {
  LDPairVec query;
  for (const auto s : csyms) query.push_back(make_pair(s, 1.0));
  return predict_weighted(query, maxth, no_adv);
}

LDPairVec MePo::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  const long nsyms = freq.size();

  // the relevant set starts as the conclusion's symbols, or the assumptions'
  // if the conclusion has none
  vector<char> rel(nsyms, 0), chained(nsyms, 0);
  bool concl = false;
  for (const auto& sw : query)
    if (sw.second < 1) chained[sw.first] = 1;
    else rel[sw.first] = concl = true;
  if (!concl) rel = chained;

  return filter(maxth, no_adv,
    [&](long i) { return fact_weight(i, rel, chained); },
    [&](long i, vector<long>& fresh) {
      for (const auto s : syms[i])
        if (!rel[s]) {
          rel[s] = 1;
          fresh.push_back(s);
        }
    });
}

double MePo::stature_bonus(long i) const {
  using namespace mepo_fudge;
  // a simp rule's bonus takes precedence over a local fact's
  if (statures[i] & STATURE_SIMP) return simp_bonus;
  if (statures[i] & STATURE_LOCAL) return local_bonus;
  return 0;
}

// sledgehammer_mepo.ML's rel_pconst_weight: [freq] counts the occurrences
// at the ptypes this one generalizes.
double MePo::typed_rel_weight(const TypedSym& s, long freq) const {
  using namespace mepo_fudge;
  if (s.kind == 'a') return abs_rel_weight;
  if (s.kind == 't') return theory_const_rel_weight;
  return (s.kind == 'v' ? local_const_multiplier : 1.0) *
    (1.0 + 2.0 / log((double)freq + 1.0));
}

// x ^ n as sledgehammer_mepo.ML's pow_int computes it, without its recursion;
// once r is 0 or inf it stays so
static double pow_int(double x, long n) {
  double r = 1.0;
  for (; n > 0 && isfinite(r) && r != 0; --n) r *= x;
  for (; n < 0 && isfinite(r) && r != 0; ++n) r /= x;
  return r;
}

// sledgehammer_mepo.ML's irrel_pconst_weight: [freq] counts the occurrences
// at the ptypes that generalize this one. A symbol whose ptype generalizes
// one of the goal's assumptions only weighs a quarter.
double MePo::typed_irrel_weight(const TypedSym& s, long freq,
  const LVecVec& chained) const {
  using namespace mepo_fudge;
  if (s.kind == 'a') return abs_irrel_weight;
  if (s.kind == 't') return theory_const_irrel_weight;
  const long k = (long)ceil(worse_irrel_freq);
  const double w =
    (freq < k ? log((double)freq + 1.0) / log(worse_irrel_freq)
              : (1.0 + 2.0 / log((double)freq + 1.0)) /
                (1.0 + 2.0 / log((double)k + 1.0))) *
    pow_int(higher_order_irrel_weight, types->order(s.ptype) - 1);
  double res = (s.kind == 'v' ? local_const_multiplier : 1.0) * w;
  for (const auto c : chained[s.sym])
    if (types->match_ptype(s.ptype, c)) return res * chained_const_irrel_weight;
  return res;
}

// sledgehammer_mepo.ML's fact_weight: a symbol is relevant if its ptype
// generalizes one of its symbol's in the relevant set, counts on neither
// side if one of those generalizes its ptype, and is irrelevant otherwise.
// An irrelevant symbol that a relevant one of the fact generalizes also
// counts on neither side. A fact whose symbols on either side are all
// abstractions and theories scores 0.
double MePo::typed_fact_weight(long i, const LVecVec& rel,
  const LVecVec& chained) const {
  const auto& ts = (*typed)[i];
  vector<size_t> relevant, irrelevant;
  bool odd = true;
  for (size_t j = 0; j < ts.size(); ++j) {
    const auto& r = rel[ts[j].sym];
    if (any_of(r.begin(), r.end(),
          [&](long p) { return types->match_ptype(ts[j].ptype, p); }))
      relevant.push_back(j);
    else if (none_of(r.begin(), r.end(),
          [&](long p) { return types->match_ptype(p, ts[j].ptype); }))
      irrelevant.push_back(j);
    else
      continue;
    odd = odd && (ts[j].kind == 'a' || ts[j].kind == 't');
  }
  if (relevant.empty() || odd) return 0;

  double relw = 0, irrelw = -stature_bonus(i);
  for (const auto j : relevant)
    relw += typed_rel_weight(ts[j], typed_freq[i][j].first);
  for (const auto j : irrelevant) {
    if (any_of(relevant.begin(), relevant.end(), [&](size_t k) {
          return ts[k].sym == ts[j].sym &&
            types->match_ptype(ts[k].ptype, ts[j].ptype); }))
      continue;
    irrelw += typed_irrel_weight(ts[j], typed_freq[i][j].second, chained);
  }
  const double res = relw / (relw + irrelw);
  return isfinite(res) ? res : 0;
}

LDPairVec MePo::predict_typed(const vector<TypedSym>& query, long maxth,
  long no_adv) const {
  const long nsyms = freq.size();
  auto add = [](LVec& ptypes, long p) {
    if (find(ptypes.begin(), ptypes.end(), p) != ptypes.end()) return false;
    ptypes.push_back(p);
    return true;
  };

  // as in predict_weighted; and if the query has no symbol, the relevant
  // set starts as the symbols of the local facts (Sledgehammer's
  // if_empty_replace_with_scope)
  LVecVec rel(nsyms), chained(nsyms);
  bool concl = false;
  for (const auto& q : query) {
    if (q.weight < 1) add(chained[q.sym], q.ptype);
    else {
      add(rel[q.sym], q.ptype);
      concl = true;
    }
  }
  if (!concl) rel = chained;
  if (query.empty()) {
    rel.assign(nsyms, {});
    for (long i = 0; i < maxth; ++i)
      if (visible(i, maxth) && (statures[i] & STATURE_LOCAL))
        for (const auto& ts : (*typed)[i]) add(rel[ts.sym], ts.ptype);
  }

  return filter(maxth, no_adv,
    [&](long i) { return typed_fact_weight(i, rel, chained); },
    [&](long i, vector<long>& fresh) {
      for (const auto& ts : (*typed)[i])
        if (add(rel[ts.sym], ts.ptype) &&
            find(fresh.begin(), fresh.end(), ts.sym) == fresh.end())
          fresh.push_back(ts.sym);
    });
}

template <class Weigh, class Accept>
LDPairVec MePo::filter(long maxth, long no_adv, Weigh weigh,
  Accept accept) const {
  using namespace mepo_fudge;
  const long max_facts = no_adv;

  enum { HOPEFUL, HOPELESS, ACCEPTED, DROPPED };
  vector<char> status(maxth, HOPEFUL);
  vector<double> weight(maxth, 0);
  vector<char> cached(maxth, 0);
  // facts without symbols, or out of scope, are never considered
  vector<long> hopeful;
  for (long i = 0; i < maxth; ++i)
    if (syms[i].empty() || !visible(i, maxth)) status[i] = DROPPED;
    else hopeful.push_back(i);

  LVec accepted;
  long remaining = max_facts;
  double threshold = threshold0;
  const double decay = pow((1 - threshold1) / (1 - threshold0),
    1.0 / (max_facts + 1));

  for (long iter = 0; remaining > 0; ++iter) {
    if (iter == hopeless_iter)
      for (long i = 0; i < maxth; ++i)
        if (status[i] == HOPELESS && weight[i] < hopeless) status[i] = DROPPED;

    LDPairVec candidates;
    for (const auto i : hopeful) {
      if (!cached[i]) {
        weight[i] = weigh(i);
        cached[i] = 1;
      }
      if (weight[i] >= threshold) candidates.push_back(make_pair(i, weight[i]));
      else status[i] = HOPELESS;
    }

    if (candidates.empty()) {
      // nothing passed the first threshold: halve it and start again, with
      // every fact hopeful
      if (iter == 0 && threshold >= ridiculous_threshold) {
        threshold /= threshold_divisor;
        iter = -1;
        for (const auto i : hopeful) status[i] = HOPEFUL;
        continue;
      }
      break;
    }

    // all perfect candidates, then at most max_imperfect imperfect ones, but
    // no more than remaining in all
    sort(candidates.begin(), candidates.end(), sortfun);
    const long nperfect = find_if(candidates.begin(), candidates.end(),
      [](const pair<long, double>& c) { return c.second <= perfect; }) -
      candidates.begin();
    const long imperfect = (long)ceil(pow(max_imperfect,
      pow((double)remaining / max_facts, max_imperfect_exp)));
    const long naccept = min({remaining, (long)candidates.size(),
      nperfect + imperfect});

    // the accepted facts' symbols join the relevant set; facts containing a
    // new relevant symbol are scored again and are hopeful again
    vector<long> fresh;
    for (long c = 0; c < naccept; ++c) {
      const long i = candidates[c].first;
      status[i] = ACCEPTED;
      accepted.push_back(i);
      accept(i, fresh);
    }
    hopeful.clear();
    for (long c = naccept; c < (long)candidates.size(); ++c)
      hopeful.push_back(candidates[c].first);
    for (const auto s : fresh)
      for (const auto i : sym_ths[s])
        if (visible(i, maxth)) {
          cached[i] = 0;
          if (status[i] == HOPELESS) {
            status[i] = HOPEFUL;
            hopeful.push_back(i);
          }
        }

    threshold = 1 - (1 - threshold) * pow(decay, (double)naccept);
    remaining -= naccept;
  }

  // the accepted facts in their order, then the others by their score
  // against the final relevant set
  LDPairVec ans;
  for (const auto i : accepted) ans.push_back(make_pair(i, weight[i]));
  const long nacc = ans.size();
  for (long i = 0; i < maxth; ++i)
    if (status[i] != ACCEPTED && visible(i, maxth))
      ans.push_back(make_pair(i,
        syms[i].empty() ? 0.0 : weigh(i)));
  partial_sort(ans.begin() + nacc,
    ans.begin() + max(nacc, min(no_adv, (long)ans.size())), ans.end(),
    sortfun);
  return ans;
}
