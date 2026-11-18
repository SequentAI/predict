#ifndef PREDICTOR_CPP
#define PREDICTOR_CPP

#include <cmath>
#include "format.cpp"
#include "tfidf.cpp"

// A predictor refers to the corpus tables it is given, without copying them:
// they must outlive it, and a change to them needs a new predictor.
class Predictor {
  public:
    Predictor(const LVecVec& deps, const LVecVec& syms,
      const LVecVec& sym_ths, long sym_num);
    virtual ~Predictor() {}

    virtual void learn(long from, long to) = 0;  // learn [from, ..., to)
    void learn_all();

    LDPairVec predict(long i, long maxth, long no_adv);
    virtual LDPairVec predict(const LVec& csyms, long maxth, long no_adv) const = 0;
    // Predicts for a query of (feature, weight) pairs. Predictors that do not
    // use weights treat every weight as 1.0.
    virtual LDPairVec predict_weighted(const LDPairVec& query, long maxth,
      long no_adv) const;

    void set_tables(const SVec& no_th, const SVec& no_sym, const SLMap& th_no,
      const HLMap& sym_no);

    virtual void import_data(string location) {}
    virtual void export_data(string location) const {}

    // Limits the following predictions to the theorems i with scope[i] set,
    // or lifts the limit if [scope] is null. Theorems out of scope are
    // treated as theorems past maxth are: never predicted, never a neighbour
    // or a source of credit, while the learned statistics stay as they are.
    virtual void set_scope(const vector<char>* scope) { this->scope = scope; }

  protected:
    // whether a prediction for maxth may include theorem i
    bool visible(long i, long maxth) const {
      return i < maxth && (!scope || (*scope)[i]);
    }
    // Drops the theorems out of scope from [ans]; returns no_adv, or the
    // number left if fewer.
    long keep_visible(LDPairVec& ans, long no_adv) const;

    const vector<char>* scope = nullptr;

    const LVecVec &deps,     // dependencies of each theorem
                  &syms,     // syms[t] holds the symbols of a theorem t
                  &sym_ths;  // sym_ths[s] holds the theorems which contain s

    const SVec *no_th = nullptr,
               *no_sym = nullptr;
    const SLMap* th_no = nullptr;
    const HLMap* sym_no = nullptr;

  private:
    void print_answer(long no_adv);
};

Predictor::Predictor(const LVecVec& deps, const LVecVec& syms,
  const LVecVec& sym_ths, long sym_num)
  : deps(deps), syms(syms), sym_ths(sym_ths) {}

void Predictor::learn_all() {
  learn(0, syms.size());
}

LDPairVec Predictor::predict_weighted(const LDPairVec& query, long maxth,
  long no_adv) const {
  LVec csyms;
  for (const auto& q : query) csyms.push_back(q.first);
  return predict(csyms, maxth, no_adv);
}

LDPairVec Predictor::predict(long i, long maxth, long n_predictions) {
  return predict(syms[i], maxth, n_predictions);
}

long Predictor::keep_visible(LDPairVec& ans, long no_adv) const {
  if (scope)
    ans.erase(remove_if(ans.begin(), ans.end(),
      [&](const pair<long, double>& p) { return !(*scope)[p.first]; }),
      ans.end());
  return min(no_adv, (long)ans.size());
}

void Predictor::set_tables(const SVec& no_th, const SVec& no_sym,
  const SLMap& th_no, const HLMap& sym_no) {
  this->no_th  = &no_th;
  this->no_sym = &no_sym;
  this->th_no  = &th_no;
  this->sym_no = &sym_no;
}

// higher scores first; equal scores in theorem order, so that rankings do
// not depend on the sort implementation
inline bool sortfun (pair<long,double> i, pair<long,double> j) {
  return (j.second < i.second) ||
         (j.second == i.second && i.first < j.first);
}

// sort the theorem-probability table such that the first n_predictions
// elements of it contain the best predictions
void sort_prediction(LDPairVec& prediction, long n_predictions) {
  partial_sort(prediction.begin(), prediction.begin() + n_predictions,
    prediction.end(), sortfun);
}

#endif
