#!/usr/bin/env python3
"""Times predict's learning and its answers to queries.

Usage: tests/bench.py <predict> <prefix> <method> [<predict options>...]

Starts predict on <prefix>.syms, .deps and .seq with -p <method>, then sends
the lines of <prefix>.queries one at a time, each after the answer to the
previous one. Prints the load time (from start until predict reports that
learning is done) and the median, minimum and maximum time from sending a
query to reading its answer.
"""

import statistics
import subprocess
import sys
import time


def main():
    binary, prefix, method = sys.argv[1:4]
    with open(prefix + ".queries") as f:
        queries = f.readlines()
    start = time.perf_counter()
    p = subprocess.Popen(
        [binary, prefix + ".syms", prefix + ".deps", prefix + ".seq",
         "-p", method] + sys.argv[4:],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    for line in p.stderr:
        if line.startswith("Learning done"):
            break
    load = time.perf_counter() - start
    times = []
    for q in queries:
        t = time.perf_counter()
        p.stdin.write(q)
        p.stdin.flush()
        if not p.stdout.readline():
            sys.exit("predict stopped")
        times.append(time.perf_counter() - t)
    p.stdin.close()
    p.wait()
    print(f"{method} load {load:.3f} s, per query median "
          f"{1000 * statistics.median(times):.2f} ms (min "
          f"{1000 * min(times):.2f}, max {1000 * max(times):.2f}), "
          f"{len(times)} queries")


if __name__ == "__main__":
    main()
