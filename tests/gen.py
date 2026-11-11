#!/usr/bin/env python3
"""Writes a seeded synthetic corpus for predict.

Usage: tests/gen.py <prefix> <theorems> <features per theorem> <vocabulary>
                    <seed> [<queries>]

Writes <prefix>.seq, .syms and .deps, and two query files: <prefix>.queries
(plain) and <prefix>.wqueries (some features weighted), each with <queries>
lines (default 20). Features are drawn from a Zipf distribution over the
vocabulary. Some theorems repeat an earlier theorem's features, so that equal
scores occur; a few have more than 20 dependencies, and a few depend on later
theorems.
"""

import bisect
import itertools
import random
import sys


def main():
    prefix = sys.argv[1]
    n, mean, vocab, seed = (int(a) for a in sys.argv[2:6])
    queries = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    r = random.Random(seed)
    cum = list(itertools.accumulate(1.0 / (i + 1) for i in range(vocab)))

    def features(k):
        return sorted({bisect.bisect_left(cum, r.random() * cum[-1])
                       for _ in range(k)})

    def size():
        return max(1, int(r.gauss(mean, mean / 3)))

    def fmt(fs):
        return ", ".join(f'"f{f}"' for f in fs)

    syms = []
    with open(prefix + ".seq", "w") as seq, \
            open(prefix + ".syms", "w") as sy, \
            open(prefix + ".deps", "w") as de:
        for t in range(n):
            if t > 0 and r.random() < 0.05:
                fs = syms[r.randrange(t)]
            else:
                fs = features(size())
            syms.append(fs)
            seq.write(f"t{t}\n")
            sy.write(f"t{t}:{fmt(fs)}\n")
            x = r.random()
            if x < 0.02:
                ds = r.sample(range(t), min(t, r.randint(25, 40)))
            elif x < 0.03 and t + 1 < n:
                ds = [r.randrange(t + 1, n)]
            else:
                ds = r.sample(range(max(0, t - 2000), t), min(t, r.randint(0, 8)))
            de.write(f"t{t}:" + " ".join(f"t{d}" for d in ds) + "\n")

    with open(prefix + ".queries", "w") as q, \
            open(prefix + ".wqueries", "w") as wq:
        for _ in range(queries):
            q.write(fmt(features(size())) + "\n")
            wq.write(", ".join(
                f'"f{f}"' + r.choice(["", "", "@0", "@0.25", "@0.5", "@2", "@3.7"])
                for f in features(size())) + "\n")


if __name__ == "__main__":
    main()
