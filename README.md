# predict

`predict` is a premise selector for hammers: given the features of a goal, it
ranks the theorems of a library by how likely they are to help prove it. It
offers k-nearest neighbours (`knn`), naive Bayes (`nbayes`), a TF-IDF cosine
filter (`mepo`) and random forests (`rforest`).

## Building and testing

```sh
make                 # builds ./predict with g++ -O3 -std=c++17
make test            # runs tests/run.sh against ./predict and against a
                     # build with address and bounds checking
./predict --version
```

`tests/golden/` holds the rankings of `knn`, `nbayes` and `mepo` on the test
corpus in `tests/data/`; `make test` compares against them, and `make golden`
rewrites them after an intended ranking change. The version string in
`main.cpp` is raised by every change to rankings, options or formats, so a
build from a source archive still says what it is.

## Usage

```
predict <syms> <deps> <seq> [-p knn|mepo|nbayes|rforest] [-n N] [-S] [-e <eval>]
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
contribution to a score by its weight; the other methods ignore weights.
Features no theorem has are ignored, a repeated feature counts once with its
largest weight, and an empty line is a query with no features.

## Differences from upstream

- Option parsing works where `char` is unsigned (aarch64).
- Every method learns the last theorem of `<seq>`.
- Featureless theorems and empty queries are accepted, and give finite
  scores in `knn`, `nbayes` and `mepo`; `rforest` accepts them too.
- Features may contain `"`, `\` and newlines, and ids may contain `:`, as
  escaped above; `rforest`'s exported trees (`-x`, read back with `-y`) escape
  feature and theorem names the same way inside their quoted labels.
- Dependencies later in `<seq>` than their dependent are kept.
- Repeated features count once in `<syms>` and in queries.
- Query features may carry weights.
- Equal scores are ranked in `<seq>` order.
- `-S` prints scores; `--version` prints the version.

## Provenance and licence

The sources were taken from CoqHammer's `src/predict/` directory,
https://github.com/lukaszcz/coqhammer, at commit
`27b80e8f2aa1a53d8d2a0642302736bad14c20ed`. The first commit of this
repository contains those files unmodified, with CoqHammer's `LICENSE`.

`predict` is by Łukasz Czajka and Cezary Kaliszyk and is distributed under the
GNU Lesser General Public License, version 2.1 (see `LICENSE`). Every change
after the first commit is ours and is distributed under the same licence.
