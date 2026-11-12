# predict

`predict` is a premise selector for hammers: given the features of a goal, it
ranks the theorems of a library by how likely they are to help prove it. It
offers k-nearest neighbours (`knn`), naive Bayes (`nbayes`), TF-IDF cosine
similarity (`cosine`), the Meng–Paulson relevance filter (`mepo`), an
iterative TF-IDF cosine filter (`mepo-cosine`), random forests (`rforest`), and
two rank fusions: `mash`, of `knn` and `nbayes`, and `mesh`, of `mepo` and
`mash`.

## Building and testing

```sh
make                 # builds ./predict with g++ -O3 -std=c++17
make test            # runs tests/run.sh against ./predict and against a
                     # build with address and bounds checking
./predict --version
```

`tests/golden/` holds the rankings of `knn`, `nbayes`, `mepo`, `mepo-cosine`,
`cosine`, `mash` and `mesh` (with their default options) on the test corpus in `tests/data/`; `make test` compares against
them, and `make golden` rewrites them after an intended ranking change. The tests also check
that `nbayes` and `cosine` rank exactly as `nbayes-dense` and `cosine-dense`,
which compute the same scores, up to rounding, by scanning every theorem, on
the test corpus and on a synthetic corpus that `tests/gen.py` writes (the
tests need `python3`). The version string in
`main.cpp` is raised by every change to rankings, options or formats, so a
build from a source archive still says what it is.

## Usage

```
predict <syms> <deps> <seq> [-p knn|mepo|mepo-cosine|nbayes|cosine|mash|mesh|rforest] [-n N] [-S] [-e <eval>] [-F <statures>]
```

`predict` learns every theorem of `<seq>`, then reads one query per line on
standard input and prints, for each, one line with the `N` best theorem ids
(default 1000), each followed by a space. With `-S`, each id is followed by
its score in parentheses: `id (score) `. Run `predict` without arguments
for all options.

With `-e <eval>`, a file of theorem ids, one per line, `predict` instead
predicts for each of those theorems from its own features, having learned
only the theorems before it in `<seq>`, and prints `id:` before its line.

## Input formats

All files are text with one record per line.

`<seq>` lists every theorem id once, in order. The order matters only for
`-e`. An id may contain any character but a newline; since output separates
ids by spaces, ids should not contain spaces.

`<syms>` gives each theorem's features:

```
id:"feature1", "feature2", ..., "featuren"
```

- In `<syms>` and `<deps>`, an id writes `\` as `\\` and `:` as `\:`.
- A feature is double-quoted. Inside the quotes, `"` is written `\"`, `\`
  is written `\\`, and a newline is written `\n`; any other backslash is an
  error.
- Features are separated by a comma and one space.
- The list may be empty (`id:`). A feature repeated on a line counts once.
  Theorems without a line have no features.

`<deps>` gives each theorem's dependencies, the theorems its proof uses:

```
id:dep1 dep2 ... depn
```

Dependencies are separated by spaces and may come anywhere in `<seq>`,
before or after the theorem. Ids missing from `<seq>` are reported on
standard error and skipped.

A query line is a feature list in the format of `<syms>` without the `id:`,
where each feature may be followed by `@weight`, a finite non-negative
number:

```
"feature1"@0.5, "feature2", "feature3"@2
```

A feature without a weight weighs 1.0. `nbayes` scales each query feature's
contribution to a score by its weight, `cosine` uses the weight as the
feature's term frequency, and `mepo` reads a weight below 1 as marking a
symbol that occurs only in the goal's assumptions; the other methods ignore
weights.
Features no theorem has are ignored, a repeated feature counts once with its
largest weight, and an empty line is a query with no features.

`<statures>`, given with `-F` and read only by `mepo`, flags theorems:

```
id:flag1 flag2 ...
```

A flag is `simp` (the theorem is a simplification rule) or `local` (it is
local to the goal's theory, for instance proved earlier in the same file).
Ids are written as in `<deps>`. Theorems without a line, and every theorem
when `-F` is not given, have no flag.

## Methods

`nbayes` is the naive Bayes of Isabelle's MaSh. A theorem's statistics are
how often it was used as a dependency, and how often each feature occurred in
the theorems that used it; a theorem counts as using itself 1000 times. Each
theorem lends its features to only the first 20 dependencies listed for it in
`<deps>`; `-d n` changes the limit to `n`, and `-d 0` lifts it. The score of
a theorem splits into a part that depends only on the learned statistics,
computed once after learning, and one term per query feature that the theorem
has statistics for; these terms are kept in a list per feature, so a query
reads only the lists of its own features rather than every theorem. This
rounds differently from the scan over every theorem (kept as `nbayes-dense`),
so theorems whose scores come within a relative 1e-9 of each other are scored
again as the scan scores them: the two give the same predictions, ties
included.

`cosine` scores a theorem by the cosine between its TF-IDF vector and the
query's: the coordinate of a feature is its IDF in the theorem's vector, and
its IDF times its weight in the query's, which makes scaling every weight
alike leave the scores unchanged. The dot products are summed over the
theorems containing each query feature. Theorems sharing no feature with the
query score 0.

`mepo` is the relevance filter of Meng and Paulson ("Lightweight relevance
filtering for machine-generated resolution problems", Journal of Applied
Logic 7(1), 2009) as Isabelle's Sledgehammer runs it
(`src/HOL/Tools/Sledgehammer/sledgehammer_mepo.ML`; Blanchette et al.,
"Hammering towards QED", Journal of Formalized Reasoning 9(1), 2016), over
untyped symbols: each feature is a symbol, matched by its whole name, so a
constant written `Thy$Name` matches only `Thy$Name`, at every type. With
`freq` the number of theorems that have a symbol:

- A theorem scores `rel / (rel + irrel)`. `rel` sums the weights of its
  symbols in the relevant set, each `1 + 2 / ln(freq + 1)`; a theorem with
  none scores 0. `irrel` sums the weights of its other symbols, each
  `ln(freq + 1) / ln 100` below frequency 100 and `(1 + 2 / ln(freq + 1)) /
  (1 + 2 / ln 101)` from 100 on, so that symbols of frequency about 100 weigh
  most. A symbol that occurs only in the goal's assumptions weighs a quarter
  of that: the assumptions play the part of Sledgehammer's chained facts.
- A simp rule's `irrel` is lowered by 0.15, and a local theorem's by 0.55
  (by 0.15 if it is both).
- The relevant set starts as the symbols of the goal's conclusion, or of its
  assumptions if the conclusion has none. Symbols that occur only in the
  assumptions join it like any other once an accepted theorem has them.
- With `max_facts` the number of predictions (`-n`), each iteration accepts
  the theorems scoring at least the threshold: all those scoring above
  0.99999, and at most `ceil(11.5 ^ (remaining / max_facts))` others, up to
  the `remaining` free places. Their symbols join the relevant set, and the
  theorems with a symbol new to it are scored again. Every accepted theorem
  moves the threshold from its start of 0.45 toward 0.85: `threshold := 1 -
  (1 - threshold) * decay`, with `decay = ((1 - 0.85) / (1 - 0.45)) ^ (1 /
  (max_facts + 1))`.
- If the first iteration accepts nothing, it is run again with the threshold
  halved, as long as the threshold was at least 0.1. Theorems scoring below
  0.001 are dropped at the fifth iteration unless a new relevant symbol made
  them hopeful again. The filter stops when an iteration accepts nothing or
  `max_facts` theorems are accepted.
- The ranking is the order of acceptance, by score within an iteration;
  theorems not accepted follow, by their score against the final relevant
  set. `-S` prints the score a theorem was accepted with.

Sledgehammer's parts that need types or terms are left out: the
`1.05 ^ (order - 1)` factor on a higher-order symbol's `irrel` weight (every
symbol counts as order 1), the 1.5 multiplier on free variables, the lambda
and theory pseudo-constants, the treatment of set constants, and the facts
added after filtering (`ext`, set axioms).

`mepo-cosine` is upstream's `mepo`. It ranks the theorems by TF-IDF cosine
similarity to the query in batches of 100: after each batch, the symbols of
its theorems join the query, each weighted by the highest score of a theorem
in the batch that has it, and the remaining theorems are scored again.

`knn` lets each neighbour of the query credit its dependencies with part of
its similarity. `-c 0` turns that off, in `knn` and inside `mash` and `mesh`;
`-c 1`, the default, keeps it.

`mash` and `mesh` fuse rankings as Isabelle's Sledgehammer does
(`src/HOL/Tools/Sledgehammer/sledgehammer_mash.ML`, lines 164–200 at commit
`34ca456d44ca14641450b5d52f51b38688d4d1b5` of the GitHub mirror
`isabelle-prover/mirror-isabelle`: `scaled_avg`, `normalize_scores`,
`mesh_facts` and `steep_weight_of_fact`). For `M` predictions, each input
ranking is cut to some length, and in it the theorem at rank `r` (from 0)
weighs `0.62 ^ log2(r + 1)`, divided by the mean weight of the ranking's
first `M` theorems. The candidates are the first `M` theorems of each input.
A candidate scores the sum, over the inputs, of the input's weight times the
candidate's weight in it (0 if the input does not have it). Equal scores are
ranked in `<seq>` order. Isabelle also divides the sum by the number of
inputs and rounds it to a multiple of 1e-8, which can only make nearly equal
scores equal.

- `mash` fuses `nbayes` and `knn`, weighted 0.5 each, each cut to `M`
  theorems. Since both are cut alike, a theorem's score orders as the sum of
  its two rank weights.
- `mesh` fuses `mepo` and `mash`, weighted 0.5 each. `-T` sets how they are
  cut: `-T 2M+25`, the default, cuts `mepo` to `M` theorems and `mash` to
  `2M + 25`; `-T 51n/50` cuts both to `51n/50`, rounded down. Each input is
  computed for as many predictions as its cut, so `mash` inside `mesh` fuses
  `nbayes` and `knn` cut to that length.

`M` and `n` are both the number of predictions asked for with `-n`
(`no_adv` in `mesh.cpp`).

## Differences from upstream

- Option parsing works where `char` is unsigned (aarch64).
- Every method learns the last theorem of `<seq>`.
- Featureless theorems and empty queries are accepted, and give finite
  scores in `knn`, `nbayes`, `mepo`, `mepo-cosine`, `mash` and `mesh`;
  `rforest` accepts them too.
- Features may contain `"`, `\` and newlines, and ids may contain `:`, as
  escaped above; `rforest`'s exported trees (`-x`, read back with `-y`) escape
  feature and theorem names the same way inside their quoted labels.
- Dependencies later in `<seq>` than their dependent are kept.
- Repeated features count once in `<syms>` and in queries.
- Query features may carry weights.
- Equal scores are ranked in `<seq>` order.
- `nbayes` reads only the statistics of the query's features, ranking as the
  scan over every theorem does (kept as `nbayes-dense`), and learns from at
  most 20 dependencies per theorem by default (`-d`).
- `cosine`, `mash` and `mesh` are new, and `-c` turns off `knn`'s credit to
  the dependencies of neighbours.
- `mepo` is Meng and Paulson's relevance filter; upstream's `mepo` is
  `mepo-cosine`. `-F` gives `mepo` the theorems' statures.
- `knn` sorts only as many neighbours as it needs.
- `-S` prints scores; `--version` prints the version.

## Provenance and licence

The sources were taken from CoqHammer's `src/predict/` directory,
https://github.com/lukaszcz/coqhammer, at commit
`27b80e8f2aa1a53d8d2a0642302736bad14c20ed`. The first commit of this
repository contains those files unmodified, with CoqHammer's `LICENSE`.

`predict` is by Łukasz Czajka and Cezary Kaliszyk and is distributed under the
GNU Lesser General Public License, version 2.1 (see `LICENSE`). Every change
after the first commit is ours and is distributed under the same licence.
