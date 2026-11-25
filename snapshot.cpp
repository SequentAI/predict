#ifndef SNAPSHOT_CPP
#define SNAPSHOT_CPP

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "format.cpp"

// A snapshot is a corpus as predict holds it once it has read <seq>, <deps>
// and <syms>: the theorem names, the features' hashes, and each theorem's
// dependencies and features and each feature's theorems as numbers. A
// process started on one maps the file and reads its tables where they lie,
// so that it starts without parsing the text or copying the tables. A
// snapshot is read only by a predict built from the commit that wrote it.
//
// Layout: the magic line; the version (a string) and 1 if it keeps typed
// symbols apart (-P with knn, mash or mesh) else 0, each a number; zeros up
// to a multiple of 8 bytes; then the header, of 8-byte numbers: the format
// (3), the numbers of theorems, features and typed symbols, the -d naive Bayes
// learned its counts with (LearnedCounts), and the offset
// and length in bytes of each section (Section, in order; offset 0 for an
// empty one). Each section starts at a multiple of 4096 bytes, and the file
// ends with the last. A table of rows is two sections: the 8-byte offset of
// each row's first element in the second, then the number of elements, and
// the rows' 4-byte elements. The sections are the theorem names (offsets and
// texts), the theorem numbers in increasing order of name, the feature hashes
// in increasing order and each one's number, and the tables of each theorem's
// dependencies and features (in the order <syms> gives them), each theorem's
// features in increasing order (sharing the offsets of the former), each
// feature's theorems, and naive Bayes' counts: each theorem's users (a table),
// each feature's (theorem, count) pairs (the 8-byte offset of each feature's
// first byte, then the number of bytes, and the pairs as LearnedCounts holds
// them), and each theorem's two 8-byte doubles. With typed symbols, then the
// typed symbols' hashes and numbers and their tables as the features', and
// the type table and each
// theorem's typed symbols and frequency-only occurrences: the constructor
// names (a list of strings), its patterns and its (order, patterns) pairs
// (each their number, then each a list: the constructor's number plus 1, or 0
// for a variable, then the argument patterns; the order, then the patterns),
// then per theorem its typed symbols and then its frequency-only occurrences
// (each their number, then each a list: symbol, (order, patterns), kind,
// count). Numbers in the header and sections are little-endian, unsigned
// except the 4-byte ones, signed; the other numbers are LEB128 varints, a
// string is its length and its bytes, a list of strings is its length and its
// elements, and a list of numbers is its length and each element's difference
// from the previous one (from 0 for the first), zigzag-encoded.

const string snapshot_magic = "predict snapshot\n";
const uint64_t snapshot_format = 3;
const uint64_t snapshot_page = 4096;

enum Section {
  NAME_OFFSETS, NAMES, NAME_ORDER, FEATURE_HASHES, FEATURE_IDS,
  DEPS_OFFSETS, DEPS, SYMS_OFFSETS, SYMS, SORTED_SYMS, SYM_THS_OFFSETS,
  SYM_THS, USERS_OFFSETS, USERS, POSTINGS_OFFSETS, POSTINGS, BASES,
  TYPED_HASHES, TYPED_IDS, TYPED_SYMS_OFFSETS, TYPED_SYMS,
  TYPED_THS_OFFSETS, TYPED_THS, TYPES, SECTIONS
};

// The theorems predict holds: with [types] (-P), [typed] and [freq_only]
// give each theorem's typed symbols and frequency-only occurrences, as
// read_syms reads them; without, they are empty. With [split], the typed
// symbols are numbered apart from the features. A snapshot holds it all
// only with [split]; -P without it does not work with snapshots.
// [sorted_syms] holds the rows of [syms] sorted, for some first theorems:
// those of the snapshot the process started on, while they are unchanged
// (Cosine reads it); none without one. [counts] holds what naive Bayes learned
// of those theorems, while they and their dependencies are unchanged.
struct Corpus {
  Names &names;
  Features &features;
  Rows &deps, &syms, &sym_ths, &sorted_syms;
  TypeTable *types;
  TypedSyms &typed, &freq_only;
  TypedTable *split;
  LearnedCounts &counts;
};

bool little_endian() {
  const uint16_t one = 1;
  unsigned char first;
  memcpy(&first, &one, 1);
  return first == 1;
}

class SnapshotWriter {
  public:
    void number(uint64_t n) {
      for (; n >= 0x80; n >>= 7) buf.push_back(char(n | 0x80));
      buf.push_back(char(n));
    }
    void str(const string &s) { number(s.size()); buf += s; }
    void list(const LVec &l) {
      number(l.size());
      long prev = 0;
      for (const auto x : l) {
        const long d = x - prev;
        number(d < 0 ? ~(uint64_t(d) << 1) : uint64_t(d) << 1);
        prev = x;
      }
    }
    void keys(const vector<TypeTable::Key> &ks, long shift) {
      number(ks.size());
      for (const auto& k : ks) {
        LVec l(1, k.first + shift);
        l.insert(l.end(), k.second.begin(), k.second.end());
        list(l);
      }
    }
    void typed(const TypedSyms &tss) {
      for (const auto& ts : tss) {
        number(ts.size());
        for (const auto& t : ts) list(LVec{t.sym, t.ptype, t.kind, t.count});
      }
    }
    string buf;
};

// appends the bytes of [x] to [s]: little-endian, as the host is
template <class T>
void put(string &s, T x) {
  s.append(reinterpret_cast<const char*>(&x), sizeof x);
}

// The sections of the table [rows]: its offsets and its elements, each row
// sorted if [sorted].
void put_rows(const Rows &rows, string &offsets, string &elements,
  bool sorted = false) {
  uint64_t n = 0;
  vector<int32_t> row;
  for (long r = 0; r < rows.size(); ++r) {
    put(offsets, n);
    const Row got = rows[r];
    row.assign(got.begin(), got.end());
    if (sorted) sort(row.begin(), row.end());
    for (const auto x : row) put(elements, x);
    n += row.size();
  }
  put(offsets, n);
}

// The sections of the feature index [f]: its hashes, then their numbers.
void put_features(const Features &f, string &hashes, string &ids) {
  for (const auto& h : f.by_hash()) {
    put(hashes, h.first);
    put(ids, h.second);
  }
}

// Writes [c] to [fname] as a snapshot of predict [version], with naive Bayes'
// sections [s] (NaiveBayes::put_counts, learned with -d [max_deps]).
void write_snapshot(const string &fname, const string &version,
  const Corpus &c, long max_deps, vector<string> s) {
  if (!little_endian())
    exit_error("Cannot write snapshot `" + fname + "' on a big-endian host!");
  const long n = c.names.size();
  vector<int32_t> order(n);
  for (long i = 0; i < n; ++i) order[i] = i;
  sort(order.begin(), order.end(),
    [&](int32_t a, int32_t b) { return c.names[a] < c.names[b]; });
  uint64_t at = 0;
  for (long i = 0; i < n; ++i) {
    put(s[NAME_OFFSETS], at);
    s[NAMES] += c.names[i];
    at += c.names[i].size();
    put(s[NAME_ORDER], order[i]);
  }
  put(s[NAME_OFFSETS], at);
  put_features(c.features, s[FEATURE_HASHES], s[FEATURE_IDS]);
  put_rows(c.deps, s[DEPS_OFFSETS], s[DEPS]);
  put_rows(c.syms, s[SYMS_OFFSETS], s[SYMS]);
  string unused;
  put_rows(c.syms, unused, s[SORTED_SYMS], true);
  put_rows(c.sym_ths, s[SYM_THS_OFFSETS], s[SYM_THS]);
  if (c.split) {
    const TypedTable& t = *c.split;
    put_features(t.features, s[TYPED_HASHES], s[TYPED_IDS]);
    put_rows(t.syms, s[TYPED_SYMS_OFFSETS], s[TYPED_SYMS]);
    put_rows(t.sym_ths, s[TYPED_THS_OFFSETS], s[TYPED_THS]);
    SnapshotWriter w;
    const SVec ctors = c.types->ctors();
    w.number(ctors.size());
    for (const auto& name : ctors) w.str(name);
    w.keys(c.types->patterns(), 1);
    w.keys(c.types->ptype_keys(), 0);
    w.typed(c.typed);
    w.typed(c.freq_only);
    s[TYPES] = move(w.buf);
  }

  SnapshotWriter w;
  w.buf = snapshot_magic;
  w.str(version);
  w.number(c.split ? 1 : 0);
  string& buf = w.buf;
  buf.resize((buf.size() + 7) / 8 * 8, '\0');
  const size_t header = buf.size();
  buf.resize(header + 8 * (5 + 2 * SECTIONS), '\0');
  vector<uint64_t> h{snapshot_format, uint64_t(n), uint64_t(c.features.size()),
    uint64_t(c.split ? c.split->features.size() : 0), uint64_t(max_deps)};
  for (const auto& sec : s) {
    if (sec.empty()) {
      h.push_back(0);
      h.push_back(0);
      continue;
    }
    buf.resize((buf.size() + snapshot_page - 1) / snapshot_page *
      snapshot_page, '\0');
    h.push_back(buf.size());
    h.push_back(sec.size());
    buf += sec;
  }
  for (size_t k = 0; k < h.size(); ++k)
    memcpy(&buf[header + 8 * k], &h[k], 8);

  // A process started on [fname] maps it: write a new file, created here
  // (never one found at its path), and rename it over [fname], so that the
  // mapping keeps the old one.
  const string tmp = fname + ".tmp" + to_string(getpid());
  const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
  bool ok = fd >= 0;
  for (size_t at = 0; ok && at < buf.size(); ) {
    const ssize_t k = write(fd, buf.data() + at, buf.size() - at);
    ok = k > 0;
    at += ok ? k : 0;
  }
  if (fd >= 0) ok = close(fd) == 0 && ok;
  if (!ok || rename(tmp.c_str(), fname.c_str()) != 0) {
    if (fd >= 0) remove(tmp.c_str());
    exit_error("Cannot write snapshot `" + fname + "'!");
  }
}

// Reads the varint parts of a snapshot from a buffer; each method returns
// false when the buffer ends before what it reads.
class SnapshotReader {
  public:
    SnapshotReader(string_view buf) : buf(buf), pos(0) {}
    bool starts_with(const string &s) {
      if (buf.compare(0, s.size(), s) != 0) return false;
      pos = s.size();
      return true;
    }
    bool number(uint64_t &n) {
      n = 0;
      for (int shift = 0; pos < buf.size() && shift < 64; shift += 7) {
        const unsigned char b = buf[pos++];
        n |= uint64_t(b & 0x7f) << shift;
        if (!(b & 0x80)) return true;
      }
      return false;
    }
    bool str(string &s) {
      uint64_t n;
      if (!number(n) || left() < n) return false;
      s.assign(buf.substr(pos, n));
      pos += n;
      return true;
    }
    // Reads a list of numbers into [l], clearing [in_range] if one is not
    // below [bound].
    bool list(LVec &l, uint64_t bound, bool &in_range) {
      uint64_t n;
      // each element takes at least a byte
      if (!number(n) || left() < n) return false;
      l.resize(n);
      uint64_t prev = 0;
      for (auto& x : l) {
        uint64_t z;
        if (!number(z)) return false;
        prev += z & 1 ? ~(z >> 1) : z >> 1;
        in_range = in_range && prev < bound;
        x = prev;
      }
      return true;
    }
    size_t left() const { return buf.size() - pos; }
    size_t at() const { return pos; }
    bool keys(vector<TypeTable::Key> &ks, long shift) {
      uint64_t n;
      bool in_range = true;
      if (!number(n) || left() < n) return false;
      ks.resize(n);
      for (auto& k : ks) {
        if (!list(k.second, UINT64_MAX, in_range) || k.second.empty() ||
            k.second[0] < LONG_MIN + shift)
          return false;
        k.first = k.second[0] - shift;
        k.second.erase(k.second.begin());
      }
      return true;
    }
    // Reads [tss]' lists of typed symbols, clearing [in_range] if one is not
    // as read_syms reads them: its symbol below [syms], its (order,
    // patterns) below [ptypes], its kind c, v, a or t, and its count at most
    // max_count and at least 1 ([freq_only], or kinds c and v).
    bool typed(TypedSyms &tss, uint64_t syms, uint64_t ptypes, bool freq_only,
               bool &in_range) {
      for (auto& ts : tss) {
        uint64_t n;
        if (!number(n) || left() < n) return false;
        ts.resize(n);
        for (auto& t : ts) {
          LVec l;
          bool ok = true;
          if (!list(l, UINT64_MAX, ok) || l.size() != 4) return false;
          const bool zero_ok = !freq_only && (l[2] == 'a' || l[2] == 't');
          in_range = in_range && l[0] >= 0 && l[0] < (long)syms &&
            l[1] >= 0 && l[1] < (long)ptypes &&
            (l[2] == 'c' || l[2] == 'v' || l[2] == 'a' || l[2] == 't') &&
            l[3] >= (zero_ok ? 0 : 1) && l[3] <= max_count;
          t = TypedSym{l[0], l[1], char(l[2]), l[3], 1.0};
        }
      }
      return true;
    }

  private:
    string_view buf;
    size_t pos;
};

// The sections of a mapped snapshot, checked to lie in it.
class SnapshotSections {
  public:
    SnapshotSections(const char* file, const uint64_t* table)
    : file(file), table(table) {}
    uint64_t bytes(Section s) const { return table[2 * s + 1]; }
    template <class T>
    const T* at(Section s) const {
      return reinterpret_cast<const T*>(file + table[2 * s]);
    }

  private:
    const char* file;
    const uint64_t* table;
};

// Checks that section [off] holds the offsets of [n] rows of [size]
// elements in all: why not, or "".
string check_offsets(const SnapshotSections &s, Section off, uint64_t n,
  uint64_t size) {
  if (s.bytes(off) != 8 * (n + 1)) return "its section table is malformed";
  const uint64_t* o = s.at<uint64_t>(off);
  if (o[0] != 0 || o[n] != size) return "a table's offsets are malformed";
  for (uint64_t r = 0; r < n; ++r)
    if (o[r] > o[r + 1]) return "a table's offsets are malformed";
  return "";
}

// Checks the table whose sections [off] and [data] give [n] rows of numbers
// below [bound]: why it is malformed, or "".
string check_rows(const SnapshotSections &s, Section off, Section data,
  uint64_t n, uint64_t bound) {
  if (s.bytes(data) % 4 != 0) return "its section table is malformed";
  const string why = check_offsets(s, off, n, s.bytes(data) / 4);
  if (!why.empty()) return why;
  const uint64_t* o = s.at<uint64_t>(off);
  const int32_t* d = s.at<int32_t>(data);
  for (uint64_t k = 0; k < o[n]; ++k)
    if (d[k] < 0 || uint64_t(d[k]) >= bound) return "a number is out of range";
  return "";
}

// Checks the [n] hashes and numbers of the sections [hashes] and [ids]: why
// they are not n distinct hashes in increasing order, numbered 0 to n - 1, or
// "". [what] names them.
string check_index(const SnapshotSections &s, Section hashes, Section ids,
  uint64_t n, const string &what) {
  if (s.bytes(hashes) != 8 * n || s.bytes(ids) != 4 * n)
    return "its section table is malformed";
  const uint64_t* h = s.at<uint64_t>(hashes);
  for (uint64_t k = 1; k < n; ++k) {
    if (h[k - 1] == h[k]) return what + " hash occurs twice";
    if (h[k - 1] > h[k])
      return "its " + what.substr(2) + " index is not sorted";
  }
  const int32_t* id = s.at<int32_t>(ids);
  vector<char> seen(n, 0);
  for (uint64_t k = 0; k < n; ++k) {
    if (id[k] < 0 || uint64_t(id[k]) >= n) return "a number is out of range";
    if (seen[id[k]]++) return what + " number occurs twice";
  }
  return "";
}

// Checks naive Bayes' postings and bases for [n] theorems and [nsyms]
// features: each feature's pairs whole varints, of a theorem below n, in
// increasing order, and a positive count, and two doubles a theorem. Returns
// why they are malformed, or "".
string check_counts(const SnapshotSections &s, uint64_t n, uint64_t nsyms) {
  if (s.bytes(BASES) != 16 * n) return "its section table is malformed";
  const string why = check_offsets(s, POSTINGS_OFFSETS, nsyms,
    s.bytes(POSTINGS));
  if (!why.empty()) return why;
  const char* malformed = "its naive Bayes counts are malformed";
  const double* bases = s.at<double>(BASES);
  for (uint64_t k = 0; k < 2 * n; ++k)
    if (!isfinite(bases[k])) return malformed;
  // a count of theorem i is at most its use count 1000 + |users of i|,
  // which bounds the logarithms naive Bayes looks up
  const uint64_t* users = s.at<uint64_t>(USERS_OFFSETS);
  const uint64_t* o = s.at<uint64_t>(POSTINGS_OFFSETS);
  const char* p = s.at<char>(POSTINGS);
  for (uint64_t f = 0; f < nsyms; ++f) {
    SnapshotReader r(string_view(p + o[f], o[f + 1] - o[f]));
    // the theorems before the next pair's
    uint64_t before = 0, d, count;
    while (r.left() > 0) {
      if (!r.number(d) || !r.number(count) || d >= n - before || count < 1 ||
          count > 1000 + users[before + d + 1] - users[before + d])
        return malformed;
      before += d + 1;
    }
  }
  return "";
}

// Fills the empty tables of [c] from the snapshot [fname] if predict
// [version] wrote it, mapping the file for as long as the process runs, and
// with [counts] its naive Bayes counts too (which other methods never read,
// so they are neither checked nor mapped). Returns why the snapshot cannot be
// used, or "" once it is read.
string read_snapshot(const string &fname, const string &version, Corpus &c,
  bool counts) {
  const int fd = open(fname.c_str(), O_RDONLY);
  if (fd < 0) return "cannot open it";
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    return "cannot read it";
  }
  const size_t size = st.st_size;
  const char* file = "";
  if (size > 0) {
    void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
      close(fd);
      return string("cannot map it: ") + strerror(errno);
    }
    file = static_cast<const char*>(m);
  }
  close(fd);

  SnapshotReader r(string_view(file, size));
  if (!r.starts_with(snapshot_magic)) return "not a snapshot";
  string v;
  if (!r.str(v)) return "truncated";
  if (v != version)
    return "written by predict " + v + ", this is predict " + version;
  uint64_t split;
  if (!r.number(split)) return "truncated";
  if (split != (c.split ? 1 : 0))
    return split ? "written with -P, read without" :
      "written without -P, read with";
  if (!little_endian()) return "this host is big-endian";

  const size_t header = (r.at() + 7) / 8 * 8;
  if (size < header || size - header < 8 * (5 + 2 * SECTIONS))
    return "truncated";
  const uint64_t* h = reinterpret_cast<const uint64_t*>(file + header);
  if (h[0] != snapshot_format)
    return "of format " + to_string(h[0]) + ", not " +
      to_string(snapshot_format);
  // numbers are 4 bytes in the tables
  const uint64_t n = h[1], nsyms = h[2], ntyped = h[3], max_deps = h[4];
  if (n > INT32_MAX || nsyms > INT32_MAX || ntyped > INT32_MAX ||
      max_deps > uint64_t(LONG_MAX) || (!c.split && ntyped != 0))
    return "its header is malformed";
  const uint64_t* table = h + 5;
  uint64_t end = header + 8 * (5 + 2 * SECTIONS);
  for (int k = 0; k < SECTIONS; ++k) {
    const uint64_t off = table[2 * k], len = table[2 * k + 1];
    if (len == 0 && off == 0) continue;
    if (off % snapshot_page != 0 || off < end)
      return "its section table is malformed";
    if (off > size || len > size - off) return "truncated";
    end = off + len;
  }
  if (end != size) return "longer than its contents";

  const SnapshotSections s(file, table);
  if (s.bytes(NAME_ORDER) != 4 * n) return "its section table is malformed";
  string why = check_offsets(s, NAME_OFFSETS, n, s.bytes(NAMES));
  if (!why.empty()) return why;
  c.names.map(s.at<uint64_t>(NAME_OFFSETS), s.at<char>(NAMES),
    s.at<int32_t>(NAME_ORDER), n);
  const int32_t* order = s.at<int32_t>(NAME_ORDER);
  for (uint64_t k = 0; k < n; ++k) {
    if (order[k] < 0 || uint64_t(order[k]) >= n)
      return "a number is out of range";
    if (k == 0) continue;
    const string_view a = c.names[order[k - 1]], b = c.names[order[k]];
    if (a == b) return "theorem `" + string(b) + "' occurs twice";
    if (a > b) return "its theorem index is not sorted";
  }

  why = check_index(s, FEATURE_HASHES, FEATURE_IDS, nsyms, "a feature");
  if (why.empty()) why = check_rows(s, DEPS_OFFSETS, DEPS, n, n);
  if (why.empty()) why = check_rows(s, SYMS_OFFSETS, SYMS, n, nsyms);
  if (why.empty()) why = check_rows(s, SYM_THS_OFFSETS, SYM_THS, nsyms, n);
  if (why.empty() && s.bytes(SORTED_SYMS) != s.bytes(SYMS))
    why = "its section table is malformed";
  if (why.empty() && counts) why = check_rows(s, USERS_OFFSETS, USERS, n, n);
  if (why.empty() && counts) why = check_counts(s, n, nsyms);
  if (!why.empty()) return why;
  const uint64_t* syms_off = s.at<uint64_t>(SYMS_OFFSETS);
  const int32_t* sorted = s.at<int32_t>(SORTED_SYMS);
  for (uint64_t i = 0; i < n; ++i)
    for (uint64_t k = syms_off[i]; k < syms_off[i + 1]; ++k)
      if (sorted[k] < 0 || uint64_t(sorted[k]) >= nsyms ||
          (k > syms_off[i] && sorted[k - 1] >= sorted[k]))
        return "its sorted features are malformed";
  // naive Bayes counts a theorem's features from its SYMS row, so a
  // repeated one would count above the theorem's use count: the row must
  // hold the sorted row's features, each once; and it takes a feature's
  // frequency from its SYM_THS row's length, which must be the number of
  // theorems having it
  if (counts) {
    const int32_t* syms = s.at<int32_t>(SYMS);
    // the last row each feature was seen in, and in how many rows
    vector<int32_t> seen(nsyms, -1);
    vector<uint64_t> rows(nsyms, 0);
    for (uint64_t i = 0; i < n; ++i)
      for (uint64_t k = syms_off[i]; k < syms_off[i + 1]; ++k) {
        if (seen[syms[k]] == int32_t(i) ||
            !binary_search(sorted + syms_off[i], sorted + syms_off[i + 1],
              syms[k]))
          return "its features differ from its sorted features";
        seen[syms[k]] = i;
        ++rows[syms[k]];
      }
    const uint64_t* ths_off = s.at<uint64_t>(SYM_THS_OFFSETS);
    for (uint64_t f = 0; f < nsyms; ++f)
      if (ths_off[f + 1] - ths_off[f] != rows[f])
        return "its feature frequencies differ from its features";
  }
  c.features.map(s.at<uint64_t>(FEATURE_HASHES), s.at<int32_t>(FEATURE_IDS),
    nsyms);
  c.deps.map(s.at<uint64_t>(DEPS_OFFSETS), s.at<int32_t>(DEPS), n);
  c.syms.map(syms_off, s.at<int32_t>(SYMS), n);
  c.sorted_syms.map(syms_off, sorted, n);
  c.sym_ths.map(s.at<uint64_t>(SYM_THS_OFFSETS), s.at<int32_t>(SYM_THS),
    nsyms);
  if (counts) {
    LearnedCounts& lc = c.counts;
    lc.theorems = n;
    lc.features = nsyms;
    lc.max_deps = max_deps;
    lc.users.map(s.at<uint64_t>(USERS_OFFSETS), s.at<int32_t>(USERS), n);
    lc.post_off = s.at<uint64_t>(POSTINGS_OFFSETS);
    lc.postings = s.at<unsigned char>(POSTINGS);
    lc.bases = s.at<double>(BASES);
    lc.freq_off = s.at<uint64_t>(SYM_THS_OFFSETS);
  }

  if (!c.split) {
    for (int k = TYPED_HASHES; k < SECTIONS; ++k)
      if (s.bytes(Section(k)) != 0) return "its section table is malformed";
    return "";
  }
  why = check_index(s, TYPED_HASHES, TYPED_IDS, ntyped, "a typed symbol's");
  if (why.empty())
    why = check_rows(s, TYPED_SYMS_OFFSETS, TYPED_SYMS, n, ntyped);
  if (why.empty())
    why = check_rows(s, TYPED_THS_OFFSETS, TYPED_THS, ntyped, n);
  if (!why.empty()) return why;
  TypedTable& t = *c.split;
  t.features.map(s.at<uint64_t>(TYPED_HASHES), s.at<int32_t>(TYPED_IDS),
    ntyped);
  t.syms.map(s.at<uint64_t>(TYPED_SYMS_OFFSETS), s.at<int32_t>(TYPED_SYMS), n);
  t.sym_ths.map(s.at<uint64_t>(TYPED_THS_OFFSETS), s.at<int32_t>(TYPED_THS),
    ntyped);

  SnapshotReader types(string_view(s.at<char>(TYPES), s.bytes(TYPES)));
  uint64_t nctors;
  if (!types.number(nctors) || types.left() < nctors)
    return "its type table is malformed";
  SVec ctors(nctors);
  for (auto& name : ctors)
    if (!types.str(name)) return "its type table is malformed";
  vector<TypeTable::Key> patterns, ptypes;
  if (!types.keys(patterns, 1) || !types.keys(ptypes, 0))
    return "its type table is malformed";
  for (const auto& pt : ptypes)
    if (pt.first < 0 || pt.first > max_count)
      return "its type table is malformed";
  if (!c.types->restore(ctors, patterns, ptypes))
    return "its type table is malformed";
  c.typed.resize(n);
  c.freq_only.resize(n);
  bool in_range = true;
  if (!types.typed(c.typed, ntyped, ptypes.size(), false, in_range) ||
      !types.typed(c.freq_only, ntyped, ptypes.size(), true, in_range) ||
      types.left() != 0)
    return "its type table is malformed";
  if (!in_range) return "a number is out of range";
  return "";
}

#endif
