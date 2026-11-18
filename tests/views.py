#!/usr/bin/env python3
"""Measures the memory and the view changes of predict's views.

Usage: tests/views.py <predict> <prefix> <method> [<predict options>...]

On the corpus tests/gen.py wrote at <prefix>, cuts the theorems, in <seq>
order, into theories T0, T1, ... of 150 each (the last shorter), each stored
as version a. Theories T(3v), T(3v+1) and T(3v+2), for view v in 0..7, also
get version b: the same theorem ids and dependencies, each theorem with the
features of the theorem 80,000 places later in <seq> (wrapping around). View
v has every theory at a but its three at b. predict starts on empty files and
-L of an empty file, view v0 is created and its peak memory is read ("one
view"); then views v1 to v7 are created, and the peak is read again ("eight
views"). Last, five changes of v0 switch its three theories between a and b,
and the median time of the changes is printed. Since predict drops the
versions no view uses, each view change is preceded by `!version` commands
storing the versions it uses that predict no longer holds; they are not
timed.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time

THEORY_SIZE = 150
SHIFT = 80000
VIEWS = 8


def read_lines(path: str) -> dict[str, str]:
    """Maps each theorem id to the rest of its line after the colon."""
    with open(path) as f:
        return dict(line.rstrip("\n").split(":", 1) for line in f)


class Predict:
    """A predict process in interactive mode."""

    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)

    def send(self, line: str) -> str:
        """Sends a line; returns predict's answer."""
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            sys.exit(f"predict exited after `{line}'")
        return out

    def peak_mb(self) -> float:
        with open(f"/proc/{self.proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
        raise RuntimeError("no VmHWM in /proc/<pid>/status")

    def close(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.close()
        self.proc.wait()


class Versions:
    """The versions' files and the view files, in a directory."""

    def __init__(self, work: str, prefix: str) -> None:
        self.work = work
        with open(prefix + ".seq") as f:
            seq = [line.rstrip("\n") for line in f]
        self.syms = read_lines(prefix + ".syms")
        self.deps = read_lines(prefix + ".deps")
        n = len(seq)
        self.theories = [seq[i:i + THEORY_SIZE]
                         for i in range(0, n, THEORY_SIZE)]
        self.shifted = {t: seq[(i + SHIFT) % n] for i, t in enumerate(seq)}
        self.stores = {(t, v): self.write(t, v)
                       for t in range(len(self.theories)) for v in "ab"
                       if v == "a" or t < 3 * VIEWS}

    def write(self, theory: int, version: str) -> str:
        """Writes a version's files; returns the command storing it."""
        base = os.path.join(self.work, f"T{theory}{version}")
        ids = self.theories[theory]
        with open(base + ".seq", "w") as f:
            f.writelines(t + "\n" for t in ids)
        with open(base + ".syms", "w") as f:
            f.writelines(
                f"{t}:{self.syms[self.shifted[t] if version == 'b' else t]}\n"
                for t in ids)
        with open(base + ".deps", "w") as f:
            f.writelines(f"{t}:{self.deps.get(t, '')}\n" for t in ids)
        with open(base + ".thy", "w") as f:
            f.writelines(f"{t}:T{theory}\n" for t in ids)
        return (f"!version T{theory} {version} {base}.syms {base}.deps "
                f"{base}.seq {base}.thy")

    def view_file(self, name: str, lines: list[str]) -> str:
        path = os.path.join(self.work, name)
        with open(path, "w") as f:
            f.writelines(line + "\n" for line in lines)
        return path


class Session:
    """A predict process with views, and the versions it holds."""

    def __init__(self, p: Predict, vs: Versions) -> None:
        self.p = p
        self.vs = vs
        self.views: dict[str, dict[int, str]] = {}
        self.stored: set[tuple[int, str]] = set()

    def view(self, name: str, changes: dict[int, str]) -> float:
        """Changes a view, storing its versions first; returns the time of
        the !view command."""
        new = {**self.views.get(name, {}), **changes}
        for key in sorted(set(new.items()) - self.stored):
            self.p.send(self.vs.stores[key])
            self.stored.add(key)
        path = self.vs.view_file(name, [f"T{t} {v}" for t, v in changes.items()])
        start = time.perf_counter()
        self.p.send(f"!view {name} {path}")
        elapsed = time.perf_counter() - start
        self.views[name] = new
        self.stored = {k for v in self.views.values() for k in v.items()}
        return elapsed


def own(v: int) -> range:
    """The theories view v has at version b."""
    return range(3 * v, 3 * v + 3)


def main() -> None:
    predict, prefix, method = sys.argv[1:4]
    options = sys.argv[4:]
    with tempfile.TemporaryDirectory() as work:
        vs = Versions(work, prefix)
        empty = vs.view_file("empty", [])
        p = Predict([predict, empty, empty, empty, "-L", empty, "-p", method,
                     *options])
        s = Session(p, vs)

        def create(v: int) -> None:
            s.view(f"v{v}", {t: "b" if t in own(v) else "a"
                             for t in range(len(vs.theories))})

        create(0)
        one = p.peak_mb()
        for v in range(1, VIEWS):
            create(v)
        eight = p.peak_mb()

        times = [s.view("v0", {t: "a" if i % 2 == 0 else "b" for t in own(0)})
                 for i in range(5)]
        p.close()

    print(f"peak memory, one view: {one:.0f} MB")
    print(f"peak memory, eight views: {eight:.0f} MB")
    print(f"view change, median of 5: {statistics.median(times) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
