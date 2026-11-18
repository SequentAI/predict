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
// Layout: the magic line, the version (a string), the theorem names (a list of
// strings), the feature hashes (their number, then 8 bytes each, least
// significant first), then per theorem its dependencies, per theorem its
// features, per feature its theorems (each a list of numbers). A number is a
// LEB128 varint; a string is its length and its bytes; a list of strings is
// its length and its elements; a list of numbers is its length and each
// element's difference from the previous one (from 0 for the first),
// zigzag-encoded.

const string snapshot_magic = "predict snapshot\n";

// The theorems predict holds. A snapshot holds all but the typed symbols:
// with [types] (-P), [typed] and [freq_only] give each theorem's typed
// symbols and frequency-only occurrences, as read_syms reads them; without,
// they are empty.
struct Corpus {
  SVec &no_th, &no_sym;
  SLMap &th_no;
  HLMap &sym_no;
  LVecVec &deps, &syms, &sym_ths;
  TypeTable *types;
  TypedSyms &typed, &freq_only;
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
    string buf;
};

// Writes [c] to [fname] as a snapshot of predict [version].
void write_snapshot(const string &fname, const string &version,
  const Corpus &c) {
  vector<uint64_t> hashes(c.no_sym.size());
  for (const auto& hs : c.sym_no) hashes[hs.second] = hs.first;

  SnapshotWriter w;
  w.buf = snapshot_magic;
  w.str(version);
  w.number(c.no_th.size());
  for (const auto& s : c.no_th) w.str(s);
  w.number(hashes.size());
  for (const auto h : hashes) w.hash(h);
  for (const auto& l : c.deps) w.list(l);
  for (const auto& l : c.syms) w.list(l);
  for (const auto& l : c.sym_ths) w.list(l);

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

  uint64_t n;
  // each name takes at least a byte
  if (!r.number(n) || r.left() < n) return "truncated";
  c.no_th.resize(n);
  for (auto& s : c.no_th)
    if (!r.str(s)) return "truncated";
  if (!r.number(n) || r.left() / 8 < n) return "truncated";
  c.no_sym.resize(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t h = 0;
    r.hash(h);
    if (!c.sym_no.emplace(h, i).second) return "a feature hash occurs twice";
  }

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
  if (r.left() != 0) return "longer than its contents";
  if (!in_range) return "a number is out of range";

  for (size_t i = 0; i < c.no_th.size(); ++i)
    if (!c.th_no.emplace(c.no_th[i], i).second)
      return "theorem `" + c.no_th[i] + "' occurs twice";
  return "";
}

#endif
