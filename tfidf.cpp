#ifndef TFIDF_CPP
#define TFIDF_CPP

template <typename T>
T sum_of_vector(const std::vector<T>& v) {
  return accumulate(v.begin(), v.end(), 0);
}

// term frequency - inverse document frequency
class Tfidf {
  public:
    Tfidf(long sym_num)
    : thn(0), thv(0), freq(vector<long>  (sym_num, 0)),
                         a(vector<double>(sym_num, 0)) {};

    double get(long s) const {return thv - a[s];}

    // Alternate version, slightly weaker than the above according to the
    // PxTP 2013 paper evaluations.
    //double get(long s) const {return (1.0 / (1.0 + a[s]));}

    double get_sum() const { return (a.size() * thv) - sum_of_vector(a); }

    void add(const LVec &syms) {
      for (const auto& sym : syms) {
        freq[sym]++;
        a[sym] = log(freq[sym]);
      }

      thv = log(++thn);
    }

    // undoes add(syms), leaving what a Tfidf never given them holds
    void remove(const LVec &syms) {
      for (const auto& sym : syms) {
        freq[sym]--;
        a[sym] = freq[sym] ? log(freq[sym]) : 0;
      }
      thv = --thn ? log(thn) : 0;
    }

    // to [sym_num] features; features dropped have no theorems
    void resize(long sym_num) {
      freq.resize(sym_num, 0);
      a.resize(sym_num, 0);
    }

    void add_many(const LVecVec &syms) {
      for (const auto& sx : syms)
        for (const auto& sy : sx)
          freq[sy]++;

      for (unsigned i = 0; i < a.size(); ++i)
        a[i] = log(freq[i]);

      thn += syms.size();
      thv = log(thn);
    }

  private:
    long thn;           // number of theorems
    double thv;         // logarithmic number of theorems
    vector<long> freq;  // feature frequency
    vector<double> a;   // logarithmic feature frequency
};

#endif
