#ifndef SNAPSHOT_CPP
#define SNAPSHOT_CPP

#include <cstdint>
#include "format.cpp"

// A snapshot is a corpus as predict holds it once it has read <seq>, <deps>
// and <syms>: the theorem names, the features' hashes, and each theorem's
// dependencies and features and each feature's theorems as numbers. Starting
// from one skips parsing the text, which takes most of predict's start-up
// time. A snapshot is read only by a predict built from the commit that wrote
// it.
//
// Layout: the magic line, the version (a string), 1 if it keeps typed symbols
// apart (-P with knn, mash or mesh) else 0, the theorem names (a list of
// strings), the feature hashes (their number, then 8 bytes each, least
// significant first), then per theorem its dependencies, per theorem its
// features, per feature its theorems (each a list of numbers). With typed
// symbols, then the type table: its constructor names (a list of strings),
// its patterns and its (order, patterns) pairs (each their number, then each
// a list: the constructor's number plus 1, or 0 for a variable, then the
// argument patterns; the order, then the patterns); the typed symbols'
// hashes, per theorem its typed symbols, per typed symbol its theorems; per
// theorem its typed symbols and then its frequency-only occurrences (each
// their number, then each a list: symbol, (order, patterns), kind, count).
// A number is a
// LEB128 varint; a string is its length and its bytes; a list of strings is
// its length and its elements; a list of numbers is its length and each
// element's difference from the previous one (from 0 for the first),
// zigzag-encoded.

const string snapshot_magic = "predict snapshot\n";

// The theorems predict holds: with [types] (-P), [typed] and [freq_only]
// give each theorem's typed symbols and frequency-only occurrences, as
// read_syms reads them; without, they are empty. With [split], the typed
// symbols are numbered apart from the features. A snapshot holds it all
// only with [split]; -P without it does not work with snapshots.
struct Corpus {
  SVec &no_th, &no_sym;
  SLMap &th_no;
  HLMap &sym_no;
  LVecVec &deps, &syms, &sym_ths;
  TypeTable *types;
  TypedSyms &typed, &freq_only;
  TypedTable *split;
};

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
    void hash(uint64_t h) {
      for (int i = 0; i < 8; ++i, h >>= 8) buf.push_back(char(h & 0xff));
    }
    void hashes(const HLMap &no, size_t n) {
      vector<uint64_t> hs(n);
      for (const auto& h : no) hs[h.second] = h.first;
      number(n);
      for (const auto h : hs) hash(h);
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

// Writes [c] to [fname] as a snapshot of predict [version].
void write_snapshot(const string &fname, const string &version,
  const Corpus &c) {
  SnapshotWriter w;
  w.buf = snapshot_magic;
  w.str(version);
  w.number(c.split ? 1 : 0);
  w.number(c.no_th.size());
  for (const auto& s : c.no_th) w.str(s);
  w.hashes(c.sym_no, c.no_sym.size());
  for (const auto& l : c.deps) w.list(l);
  for (const auto& l : c.syms) w.list(l);
  for (const auto& l : c.sym_ths) w.list(l);
  if (c.split) {
    const SVec ctors = c.types->ctors();
    w.number(ctors.size());
    for (const auto& s : ctors) w.str(s);
    w.keys(c.types->patterns(), 1);
    w.keys(c.types->ptype_keys(), 0);
    w.hashes(c.split->sym_no, c.split->no_sym.size());
    for (const auto& l : c.split->syms) w.list(l);
    for (const auto& l : c.split->sym_ths) w.list(l);
    w.typed(c.typed);
    w.typed(c.freq_only);
  }

  ofstream out(fname, ios::binary);
  out.write(w.buf.data(), w.buf.size());
  out.close();
  if (!out)
    exit_error("Cannot write snapshot `" + fname + "'!");
}

// Reads a snapshot from a buffer; each method returns false when the buffer
// ends before what it reads.
class SnapshotReader {
  public:
    SnapshotReader(const string &buf) : buf(buf), pos(0) {}
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
      s.assign(buf, pos, n);
      pos += n;
      return true;
    }
    bool hash(uint64_t &h) {
      if (left() < 8) return false;
      h = 0;
      for (int i = 7; i >= 0; --i) h = h << 8 | (unsigned char)buf[pos + i];
      pos += 8;
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
    // Reads [n] hashes into [no], numbering them in order; clears [unique]
    // if one occurs twice.
    bool hashes(HLMap &no, uint64_t &n, bool &unique) {
      if (!number(n) || left() / 8 < n) return false;
      for (uint64_t i = 0; i < n; ++i) {
        uint64_t h = 0;
        hash(h);
        unique = no.emplace(h, i).second && unique;
      }
      return true;
    }
    bool keys(vector<TypeTable::Key> &ks, long shift) {
      uint64_t n;
      bool in_range = true;
      if (!number(n) || left() < n) return false;
      ks.resize(n);
      for (auto& k : ks) {
        if (!list(k.second, UINT64_MAX, in_range) || k.second.empty())
          return false;
        k.first = k.second[0] - shift;
        k.second.erase(k.second.begin());
      }
      return true;
    }
    // Reads [tss]' lists of typed symbols, clearing [in_range] if one's
    // symbol is not below [syms] or its (order, patterns) not below [ptypes].
    bool typed(TypedSyms &tss, uint64_t syms, uint64_t ptypes,
               bool &in_range) {
      for (auto& ts : tss) {
        uint64_t n;
        if (!number(n) || left() < n) return false;
        ts.resize(n);
        for (auto& t : ts) {
          LVec l;
          bool ok = true;
          if (!list(l, UINT64_MAX, ok) || l.size() != 4) return false;
          in_range = in_range && l[0] >= 0 && l[0] < (long)syms &&
            l[1] >= 0 && l[1] < (long)ptypes && l[2] >= 0 && l[2] < 256 &&
            l[3] >= 0;
          t = TypedSym{l[0], l[1], char(l[2]), l[3], 1.0};
        }
      }
      return true;
    }

  private:
    const string &buf;
    size_t pos;
};

// Fills the empty tables of [c] from the snapshot [fname] if predict
// [version] wrote it. The snapshot's features get empty names, since it holds
// only their hashes. Returns why the snapshot cannot be used, or "" once it is
// read.
string read_snapshot(const string &fname, const string &version, Corpus &c) {
  ifstream in(fname, ios::binary | ios::ate);
  if (!in) return "cannot open it";
  string buf(in.tellg(), '\0');
  in.seekg(0);
  if (!in.read(&buf[0], buf.size())) return "cannot read it";

  SnapshotReader r(buf);
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

  uint64_t n;
  // each name takes at least a byte
  if (!r.number(n) || r.left() < n) return "truncated";
  c.no_th.resize(n);
  for (auto& s : c.no_th)
    if (!r.str(s)) return "truncated";
  bool unique = true;
  if (!r.hashes(c.sym_no, n, unique)) return "truncated";
  if (!unique) return "a feature hash occurs twice";
  c.no_sym.resize(n);

  c.deps.resize(c.no_th.size());
  c.syms.resize(c.no_th.size());
  c.sym_ths.resize(c.no_sym.size());
  bool in_range = true;
  for (auto& l : c.deps)
    if (!r.list(l, c.no_th.size(), in_range)) return "truncated";
  for (auto& l : c.syms)
    if (!r.list(l, c.no_sym.size(), in_range)) return "truncated";
  for (auto& l : c.sym_ths)
    if (!r.list(l, c.no_th.size(), in_range)) return "truncated";
  if (c.split) {
    if (!r.number(n) || r.left() < n) return "truncated";
    SVec ctors(n);
    for (auto& s : ctors)
      if (!r.str(s)) return "truncated";
    vector<TypeTable::Key> patterns, ptypes;
    if (!r.keys(patterns, 1) || !r.keys(ptypes, 0)) return "truncated";
    if (!c.types->restore(ctors, patterns, ptypes))
      return "its type table is malformed";
    TypedTable& t = *c.split;
    if (!r.hashes(t.sym_no, n, unique)) return "truncated";
    if (!unique) return "a typed symbol's hash occurs twice";
    t.no_sym.resize(n);
    t.syms.resize(c.no_th.size());
    t.sym_ths.resize(n);
    for (auto& l : t.syms)
      if (!r.list(l, n, in_range)) return "truncated";
    for (auto& l : t.sym_ths)
      if (!r.list(l, c.no_th.size(), in_range)) return "truncated";
    c.typed.resize(c.no_th.size());
    c.freq_only.resize(c.no_th.size());
    if (!r.typed(c.typed, n, ptypes.size(), in_range) ||
        !r.typed(c.freq_only, n, ptypes.size(), in_range))
      return "truncated";
  }
  if (r.left() != 0) return "longer than its contents";
  if (!in_range) return "a number is out of range";

  for (size_t i = 0; i < c.no_th.size(); ++i)
    if (!c.th_no.emplace(c.no_th[i], i).second)
      return "theorem `" + c.no_th[i] + "' occurs twice";
  return "";
}

#endif
