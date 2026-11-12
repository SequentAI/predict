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
// Left out, since they need types or terms: the 1.05 ^ (order - 1) factor on
// a higher-order symbol's irrelevant weight (every symbol weighs as order 1),
// the 1.5 multiplier on free variables, lambda and theory pseudo-constants,
// the special treatment of set constants, and the facts Sledgehammer inserts
// afterwards (ext, set axioms).
class MePo : public Predictor {
  public:
    MePo(const LVecVec& deps, const LVecVec& syms, const LVecVec& sym_ths,
      long sym_num, const vector<unsigned char>& statures);

    void learn(long from, long to);
    LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const;
    // A query feature weighing less than 1 occurs only in the goal's
    // assumptions; one weighing 1 or more occurs in its conclusion.
    LDPairVec predict_weighted(const LDPairVec& query, long maxth,
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

    vector<unsigned char> statures;
    // the number of learned facts containing each symbol
    vector<long> freq;
};

// sledgehammer_mepo.ML's default_relevance_fudge and Sledgehammer's default
// fact_thresholds
namespace mepo_fudge {
  const double worse_irrel_freq = 100.0;
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
  long sym_num, const vector<unsigned char>& statures)
: Predictor(deps, syms, sym_ths, sym_num), statures(statures),
  freq(sym_num, 0) {}

void MePo::learn(long from, long to) {
  for (long i = from; i < to; ++i)
    for (const auto s : syms[i])
      ++freq[s];
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
  using namespace mepo_fudge;
  const long max_facts = no_adv;
  const long nsyms = freq.size();

  // the relevant set starts as the conclusion's symbols, or the assumptions'
  // if the conclusion has none
  vector<char> rel(nsyms, 0), chained(nsyms, 0);
  bool concl = false;
  for (const auto& sw : query)
    if (sw.second < 1) chained[sw.first] = 1;
    else rel[sw.first] = concl = true;
  if (!concl) rel = chained;

  enum { HOPEFUL, HOPELESS, ACCEPTED, DROPPED };
  vector<char> status(maxth, HOPEFUL);
  vector<double> weight(maxth, 0);
  vector<char> cached(maxth, 0);
  // facts without symbols are never considered
  vector<long> hopeful;
  for (long i = 0; i < maxth; ++i)
    if (syms[i].empty()) status[i] = DROPPED;
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
        weight[i] = fact_weight(i, rel, chained);
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
      for (const auto s : syms[i])
        if (!rel[s]) {
          rel[s] = 1;
          fresh.push_back(s);
        }
    }
    hopeful.clear();
    for (long c = naccept; c < (long)candidates.size(); ++c)
      hopeful.push_back(candidates[c].first);
    for (const auto s : fresh)
      for (const auto i : sym_ths[s])
        if (i < maxth) {
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
    if (status[i] != ACCEPTED)
      ans.push_back(make_pair(i,
        syms[i].empty() ? 0.0 : fact_weight(i, rel, chained)));
  partial_sort(ans.begin() + nacc,
    ans.begin() + max(nacc, min(no_adv, maxth)), ans.end(), sortfun);
  return ans;
}
