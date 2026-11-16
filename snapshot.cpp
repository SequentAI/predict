#ifndef SNAPSHOT_CPP
#define SNAPSHOT_CPP

#include <cstdint>
#include <cstring>
#include "format.cpp"

// A snapshot is a corpus as predict holds it once it has read <seq>, <deps>
// and <syms>: the theorem and feature names, and each theorem's dependencies
// and features and each feature's theorems as numbers. Starting from one skips
// parsing the text, which takes most of predict's start-up time. Numbers are
// 32-bit, in the byte order of the machine that wrote the snapshot; a snapshot
// is read only by the version of predict that wrote it.
//
// Layout: the magic line, the version (a string), the number 1 (byte order),
// the theorem names, the feature names, then per theorem its dependencies,
// per theorem its features, per feature its theorems. A string is its length
// and its bytes; a list of names or numbers is its length and its elements.

const string snapshot_magic = "predict snapshot\n";

// The tables a snapshot holds.
struct Corpus {
  SVec &no_th, &no_sym;
  SLMap &th_no, &sym_no;
  LVecVec &deps, &syms, &sym_ths;
};

class SnapshotWriter {
  public:
    void number(uint64_t n) {
      if (n > UINT32_MAX)
        exit_error("Cannot write a snapshot: " + to_string(n) +
                   " exceeds 32 bits!");
      const uint32_t x = n;
      buf.append(reinterpret_cast<const char*>(&x), sizeof x);
    }
    void str(const string &s) { number(s.size()); buf += s; }
    void names(const SVec &v) {
      number(v.size());
      for (const auto& s : v) str(s);
    }
    void lists(const LVecVec &v) {
      for (const auto& l : v) {
        number(l.size());
        for (const auto x : l) number(x);
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
  w.number(1);
  w.names(c.no_th);
  w.names(c.no_sym);
  w.lists(c.deps);
  w.lists(c.syms);
  w.lists(c.sym_ths);
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
    bool number(uint32_t &x) {
      if (buf.size() - pos < sizeof x) return false;
      memcpy(&x, buf.data() + pos, sizeof x);
      pos += sizeof x;
      return true;
    }
    bool str(string &s) {
      uint32_t n;
      if (!number(n) || buf.size() - pos < n) return false;
      s.assign(buf, pos, n);
      pos += n;
      return true;
    }
    bool names(SVec &v) {
      uint32_t n;
      if (!number(n) || (buf.size() - pos) / sizeof n < n) return false;
      v.resize(n);
      for (auto& s : v)
        if (!str(s)) return false;
      return true;
    }
    // Reads [v.size()] lists of numbers, clearing [in_range] if one is not
    // below [bound].
    bool lists(LVecVec &v, size_t bound, bool &in_range) {
      for (auto& l : v) {
        uint32_t n;
        if (!number(n) || (buf.size() - pos) / sizeof n < n) return false;
        l.resize(n);
        for (auto& x : l) {
          uint32_t y = 0;
          number(y);
          in_range = in_range && y < bound;
          x = y;
        }
      }
      return true;
    }
    bool at_end() const { return pos == buf.size(); }

  private:
    const string &buf;
    size_t pos;
};

// Fills the empty tables of [c] from the snapshot [fname] if predict
// [version] wrote it. Returns why the snapshot cannot be used, or "" once it
// is read.
string read_snapshot(const string &fname, const string &version, Corpus &c) {
  ifstream in(fname, ios::binary | ios::ate);
  if (!in) return "cannot open it";
  string buf(in.tellg(), '\0');
  in.seekg(0);
  if (!in.read(&buf[0], buf.size())) return "cannot read it";

  SnapshotReader r(buf);
  if (!r.starts_with(snapshot_magic)) return "not a snapshot";
  string v;
  uint32_t one;
  if (!r.str(v) || !r.number(one)) return "truncated";
  if (v != version)
    return "written by predict " + v + ", this is predict " + version;
  if (one != 1) return "written on a machine of another byte order";

  if (!r.names(c.no_th) || !r.names(c.no_sym)) return "truncated";
  c.deps.resize(c.no_th.size());
  c.syms.resize(c.no_th.size());
  c.sym_ths.resize(c.no_sym.size());
  bool in_range = true;
  if (!r.lists(c.deps, c.no_th.size(), in_range) ||
      !r.lists(c.syms, c.no_sym.size(), in_range) ||
      !r.lists(c.sym_ths, c.no_th.size(), in_range))
    return "truncated";
  if (!r.at_end()) return "longer than its contents";
  if (!in_range) return "a number is out of range";

  for (size_t i = 0; i < c.no_th.size(); ++i)
    if (!c.th_no.emplace(c.no_th[i], i).second)
      return "theorem `" + c.no_th[i] + "' occurs twice";
  for (size_t i = 0; i < c.no_sym.size(); ++i)
    if (!c.sym_no.emplace(c.no_sym[i], i).second)
      return "feature `" + c.no_sym[i] + "' occurs twice";
  return "";
}

#endif
