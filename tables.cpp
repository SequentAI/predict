#ifndef TABLES_CPP
#define TABLES_CPP

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std;

// The corpus tables. A process started on a snapshot (-i) reads the
// snapshot's entries from its mapped file and holds only those added after
// them; a process started on text holds every entry.

// A row of a Rows table: a slice of a snapshot's row, then the numbers
// appended to it after (for a row added after the snapshot, only the latter).
class Row {
  public:
    class iterator {
      public:
        typedef forward_iterator_tag iterator_category;
        typedef int32_t value_type;
        typedef ptrdiff_t difference_type;
        typedef const int32_t* pointer;
        typedef const int32_t& reference;

        iterator() {}
        iterator(const int32_t* a, size_t na, const int32_t* b, size_t k)
        : a(a), b(b), na(na), k(k) {}
        reference operator*() const { return k < na ? a[k] : b[k - na]; }
        iterator& operator++() { ++k; return *this; }
        iterator operator++(int) { iterator old(*this); ++k; return old; }
        bool operator==(const iterator& o) const { return k == o.k; }
        bool operator!=(const iterator& o) const { return k != o.k; }

      private:
        const int32_t *a = nullptr, *b = nullptr;
        size_t na = 0, k = 0;
    };

    Row(const int32_t* a, size_t na, const int32_t* b = nullptr, size_t nb = 0)
    : a(a), b(b), na(na), nb(nb) {}

    size_t size() const { return na + nb; }
    bool empty() const { return na + nb == 0; }
    int32_t operator[](size_t k) const { return k < na ? a[k] : b[k - na]; }
    iterator begin() const { return iterator(a, na, b, 0); }
    iterator end() const { return iterator(a, na, b, na + nb); }

  private:
    const int32_t *a, *b;
    size_t na, nb;
};

// A table of rows of numbers (theorems or features). The first [base] rows
// are a snapshot's, row r being data[off[r], off[r + 1]) of its mapped file,
// each followed by the numbers appended to it later; the rows after them are
// held here.
class Rows {
  public:
    // Takes the [n] rows given by [off] and [data] as the first rows of an
    // empty table.
    void map(const uint64_t* off, const int32_t* data, long n) {
      base_off = off;
      base_data = data;
      base = n;
    }

    long size() const { return base + tail.size(); }

    Row operator[](long r) const {
      if (r >= base) {
        const auto& t = tail[r - base];
        return Row(t.data(), t.size());
      }
      const int32_t* a = base_data + base_off[r];
      const size_t na = base_off[r + 1] - base_off[r];
      if (ext.empty()) return Row(a, na);
      const auto got = ext.find(r);
      if (got == ext.end()) return Row(a, na);
      return Row(a, na, got->second.data(), got->second.size());
    }

    // appends [x] to row r
    void push(long r, long x) {
      if (r >= base) tail[r - base].push_back(x);
      else ext[r].push_back(x);
    }

    // removes the last number appended to row r
    void pop_back(long r) {
      if (r >= base) {
        tail[r - base].pop_back();
        return;
      }
      const auto got = ext.find(r);
      got->second.pop_back();
      if (got->second.empty()) ext.erase(got);
    }

    // to [n] rows, the new ones empty
    void resize(long n) {
      if (n < base) own();
      tail.resize(n - base);
    }

    // removes the numbers from [x] on from row r
    void erase_from(long r, long x) {
      const auto drop = [&](vector<int32_t>& v) {
        v.erase(remove_if(v.begin(), v.end(),
          [&](int32_t y) { return y >= x; }), v.end());
      };
      if (r >= base) {
        drop(tail[r - base]);
        return;
      }
      if (any_of(base_data + base_off[r], base_data + base_off[r + 1],
                 [&](int32_t y) { return y >= x; })) {
        drop(own()[r]);
        return;
      }
      const auto got = ext.find(r);
      if (got == ext.end()) return;
      drop(got->second);
      if (got->second.empty()) ext.erase(got);
    }

    // whether a snapshot's row was appended to
    bool extended() const { return !ext.empty(); }

    // Every row, held here, copying the snapshot's: for changes other than
    // appending.
    vector<vector<int32_t> >& own() {
      if (base == 0) return tail;
      vector<vector<int32_t> > all(size());
      for (long r = 0; r < size(); ++r) {
        const Row row = (*this)[r];
        all[r].assign(row.begin(), row.end());
      }
      tail = move(all);
      ext.clear();
      base = 0;
      return tail;
    }

  private:
    const uint64_t* base_off = nullptr;
    const int32_t* base_data = nullptr;
    long base = 0;
    unordered_map<long, vector<int32_t> > ext;
    vector<vector<int32_t> > tail;
};

// The theorem names, numbered in order: a snapshot's, read from its mapped
// file and looked up by binary search in their order by name, then those
// added after them.
class Names {
  public:
    // Takes the [n] names, name i being blob[off[i], off[i + 1]), with their
    // numbers in increasing order of name [by_name], as the first of an empty
    // table.
    void map(const uint64_t* off, const char* blob, const int32_t* by_name,
      long n) {
      base_off = off;
      base_blob = blob;
      base_by_name = by_name;
      base = n;
    }

    long size() const { return base + tail.size(); }

    string_view operator[](long i) const {
      if (i >= base) return tail[i - base];
      return string_view(base_blob + base_off[i],
        base_off[i + 1] - base_off[i]);
    }

    // the number of name [s], -1 if none
    long find(const string& s) const {
      const auto got = tail_no.find(s);
      if (got != tail_no.end()) return got->second;
      const int32_t* end = base_by_name + base;
      const int32_t* at = lower_bound(base_by_name, end, string_view(s),
        [&](int32_t i, string_view v) { return (*this)[i] < v; });
      return at != end && (*this)[*at] == s ? *at : -1;
    }

    // numbers [s], which must be new
    void push_back(const string& s) {
      tail_no[s] = size();
      tail.push_back(s);
    }

    // to the first [n] names
    void resize(long n) {
      if (n < base) own();
      for (long i = n; i < size(); ++i) tail_no.erase(tail[i - base]);
      tail.resize(n - base);
    }

  private:
    // holds every name here
    void own() {
      vector<string> all;
      for (long i = 0; i < size(); ++i) all.push_back(string((*this)[i]));
      base = 0;
      tail.clear();
      tail_no.clear();
      for (const auto& s : all) push_back(s);
    }

    const uint64_t* base_off = nullptr;
    const char* base_blob = nullptr;
    const int32_t* base_by_name = nullptr;
    long base = 0;
    vector<string> tail;
    unordered_map<string, long> tail_no;
};

// The features, known by the hashes of their texts and numbered in order: a
// snapshot's, read from its mapped file and looked up by binary search in
// their order by hash, then those added after them, with their texts.
class Features {
  public:
    // Takes [n] features, by their increasing [hashes], with their numbers
    // [ids], as the first of an empty table.
    void map(const uint64_t* hashes, const int32_t* ids, long n) {
      base_hashes = hashes;
      base_ids = ids;
      base = n;
    }

    long size() const { return base + tail.size(); }

    // the number of the feature with hash [h], -1 if none
    long find(uint64_t h) const {
      const uint64_t* end = base_hashes + base;
      const uint64_t* at = lower_bound(base_hashes, end, h);
      if (at != end && *at == h) return base_ids[at - base_hashes];
      const auto got = tail_no.find(h);
      return got == tail_no.end() ? -1 : got->second;
    }

    // numbers the feature with hash [h] and text [text], which must be new
    long add(uint64_t h, const string& text) {
      const long i = size();
      tail_no[h] = i;
      tail.push_back(make_pair(h, text));
      return i;
    }

    // drops the last feature, which must have been added after the snapshot
    void pop_back() {
      tail_no.erase(tail.back().first);
      tail.pop_back();
    }

    // the text of feature i: empty for a snapshot's, which holds no texts
    const string& text(long i) const {
      static const string none;
      return i < base ? none : tail[i - base].second;
    }

    // (hash, number) of every feature, by hash
    vector<pair<uint64_t, int32_t> > by_hash() const {
      vector<pair<uint64_t, int32_t> > out;
      for (long k = 0; k < base; ++k)
        out.push_back(make_pair(base_hashes[k], base_ids[k]));
      for (const auto& t : tail_no) out.push_back(make_pair(t.first, t.second));
      sort(out.begin(), out.end());
      return out;
    }

  private:
    const uint64_t* base_hashes = nullptr;
    const int32_t* base_ids = nullptr;
    long base = 0;
    vector<pair<uint64_t, string> > tail;
    unordered_map<uint64_t, long> tail_no;
};

// The number of feature hash [h] in [index], -1 if none: for the functions
// that take the corpus's features or a view's (Numbering's map).
inline long find_feature(const Features& index, uint64_t h) {
  return index.find(h);
}
inline long find_feature(const unordered_map<uint64_t, long>& index,
  uint64_t h) {
  const auto got = index.find(h);
  return got == index.end() ? -1 : got->second;
}

// What naive Bayes learned of a snapshot's [theorems] theorems (0 if none)
// and [features] features, with -d [max_deps], as the snapshot holds it:
// each theorem's [users] in learn order; each feature f's (theorem, count)
// pairs in increasing order of theorem, in the bytes [post_off][f] to
// [post_off][f + 1] of [postings], each the LEB128 varints of its theorem's
// difference from the previous one's (from -1) minus 1 and its count
// (for_postings); for each theorem i,
// [bases][2 i] its B_i and [bases][2 i + 1] the sum over its counts s of
// 0.2 ln(1 + (1 - s) / t) (NaiveBayes); and each feature's number of
// theorems, [freq_off][f + 1] - [freq_off][f].
struct LearnedCounts {
  long theorems = 0, features = 0, max_deps = 0;
  Rows users;
  const uint64_t *post_off = nullptr, *freq_off = nullptr;
  const unsigned char* postings = nullptr;
  const double* bases = nullptr;
};

// Reads the LEB128 varint at [p], moving [p] past it.
inline uint64_t read_varint(const unsigned char*& p) {
  uint64_t n = 0;
  for (int shift = 0; ; shift += 7) {
    const unsigned char b = *p++;
    n |= uint64_t(b & 0x7f) << shift;
    if (!(b & 0x80)) return n;
  }
}

// Calls [fn](i, s) for each (theorem i, count s) pair of feature f in [c], in
// increasing order of i, until it returns false.
template <class F>
void for_postings(const LearnedCounts& c, long f, F fn) {
  const unsigned char* p = c.postings + c.post_off[f];
  const unsigned char* end = c.postings + c.post_off[f + 1];
  long i = -1;
  while (p < end) {
    i += read_varint(p) + 1;
    const long s = read_varint(p);
    if (!fn(i, s)) return;
  }
}

#endif
