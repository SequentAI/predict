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
tests need `python3`).

`predict --version` prints the commit `predict` was built from. In a clone,
`make` asks git for it; a source archive made by `git archive`, as GitHub's
are, has it written into `main.cpp` (see `.gitattributes`). A build that has
neither says `unknown`.

## Usage

```
predict <syms> <deps> <seq> [-p knn|mepo|mepo-cosine|nbayes|cosine|mash|mesh|rforest] [-n N] [-S] [-e <eval>] [-F <statures>] [-P] [-i <snapshot>] [-o <snapshot>] [-L <theories>] [-b <boost>]
```

`predict` learns every theorem of `<seq>`, then reads one query per line on
standard input and prints, for each, one line with the `N` best theorem ids
(default 1000), each followed by a space. With `-S`, each id is followed by
its score in parentheses: `id (score) `. Run `predict` without arguments
for all options.

With `-e <eval>`, a file of theorem ids, one per line, `predict` instead
predicts for each of those theorems from its own features, having learned
only the theorems before it in `<seq>`, and prints `id:` before its line.

### Snapshots

Reading the text files takes most of `predict`'s start-up time. With
`-o <snapshot>`, `predict` reads the corpus, writes it to the file
`<snapshot>` in a binary form, and stops. With `-i <snapshot>`, it starts
from that corpus instead, and `<syms>`, `<deps>` and `<seq>` hold theorems to
add after it (empty files add none): every method then ranks as it would
from the text files the snapshot was written from, followed by the added
theorems' lines. The one exception is a dependency of a snapshot's theorem
on a theorem added after it, which the snapshot dropped when it was written,
as `<deps>` drops one missing from `<seq>`. `-i` and `-o` together write the
corpus with the added theorems to a new snapshot. Neither works with `-P`.

A snapshot holds the theorem names, the features' hashes (below) and the
numbers `predict` gives theorems and features, and nothing learned: learning
is fast next to reading. It can be read only by a `predict` built from the
commit that wrote it. If the snapshot was written by another build, is truncated, or is
not a snapshot, `predict` says why on standard error and exits with status 3
before reading anything else, so that a caller can fall back to the text
files. Since a snapshot holds no feature names, `rforest` cannot export its
trees (`-x`) from one.

`predict` knows a feature by the 64-bit FNV-1a hash of its text, so two
features whose texts hash alike count as one; for a million distinct
features, the chance that any two do is about 3 in 100 million.

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
- The list may be empty (`id:`). A theorem may have several lines; a feature
  repeated on them, or on one line, counts once.
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
symbol that occurs only in the goal's assumptions; `mash` passes the weights
to its `nbayes`, and `mesh` to its `mepo` and `mash`. The other methods ignore
weights.
Features no theorem has are ignored, a repeated feature counts once with its
largest weight, and an empty line is a query with no features.

### Typed symbols

For `mepo -P`, a feature in `<syms>` or a query may be followed by its type,
and in `<syms>` then by `*count`, how often the theorem has the symbol at
that type (1 if absent; at least 1 for kinds `c` and `v`):

```
id:"Thy$f"<c 1 _>*2, "x"<v 0>, "%thy%Thy"<t 0>, "Thy$g"<c 2 "Thy$list"(_)>
```

The type is `<kind order patterns>`: `kind` is `c` for a constant, `v` for a
free variable, `a` for the lambda pseudo-constant and `t` for a theory
pseudo-constant; `order` is the order of the symbol's type (0 for a
non-function, 1 for a first-order function); and each pattern, preceded by a
space, is a type argument of the symbol, `_` for a type variable or a quoted
type constructor, followed by its argument patterns in parentheses, separated
by a comma and one space. Orders and counts are at most 10^9. A feature
without a type is a constant of order 1
with no patterns. A feature repeated with the same type on a line adds up its
counts; repeated with other types, it is one feature with several types.
Without `-P`, types and counts are read and ignored, so a typed corpus ranks
as the corpus without them.

A feature in `<syms>` may instead be followed by `+count` (at least 1): a
frequency-only token, occurrences of the symbol at that type which count in
its frequencies (below) but don't make it a symbol of the theorem.
Sledgehammer's frequencies count every constant occurrence of a fact,
including the quantifiers it doesn't take as the fact's symbols, so
`"bool$!"<c 2 _>+3` gives three such occurrences of a quantifier. Without
`-P`, frequency-only tokens are ignored.

`<statures>`, given with `-F` and read only by `mepo` (also inside `mesh`),
flags theorems:

```
id:flag1 flag2 ...
```

A flag is `simp` (the theorem is a simplification rule) or `local` (it is
local to the goal's theory, for instance proved earlier in the same file).
Ids are written as in `<deps>`. Theorems without a line, and every theorem
when `-F` is not given, have no flag.

### Query scopes

A library's theories form a dependency DAG, and a goal in one theory may use
only part of the library: not the theories that build on its own (the
downstream cone, which would make a cycle), not its own theory's theorems
proved after the goal, and not the private theorems of other theories. When
one `predict` process serves goals from many theories, filtering its top `N`
afterwards can leave few or none of them usable, if the best-scoring
theorems are out of reach. A scope makes `predict` choose its predictions
among the usable theorems only, as Isabelle's MaSh does when it selects the
visible facts before ranking (`select_fact_idxs` in
`src/HOL/Tools/Sledgehammer/sledgehammer_mash.ML`).

Scopes of the default view need each theorem's theory, given with `-L
<theories>` (a view's versions bring their own, see "Versions and views"):

```
id:theory
id:theory private
```

Ids are written as in `<deps>`. A theory name is nonempty and has no space,
comma or `]`. A `private` theorem may be predicted only for a query of its
own theory. Theorems without a line have no theory, and every scope lets
them be predicted.

A query line may begin with a scope in square brackets, before its features:

```
[k=8 theory=T before=3 exclude=D1,D2 ancestors=A1,A2]"feature1", "feature2"@0.5
```

Each key may appear once, in any order, and each may be left out:

- `view=V`: the query ranks the theorems of view `V` (see below), and the
  other keys apply within it.
- `k=n`: the number of predictions for this query (at least 1), instead of
  `-n`'s.
- `theory=T`: the query's own theory. Its private theorems may be predicted.
- `before=n`: only the first `n` theorems of `T`, in corpus order, may be
  predicted (all if left out), so that a goal sees only the theorems proved
  before it. Needs `theory`.
- `exclude=D1,D2`: the theorems of these theories are never predicted;
  meant for the theories that depend on `T`.
- `ancestors=A1,A2`: the theorems of these theories are boosted. The first
  `max(k, 128)` theorems in scope are ranked again by the weight
  `0.62 ^ log2(r + 1)` of their rank `r` (from 0), as `mash` and `mesh` weigh
  ranks, multiplied by `-b`'s factor (default 4) for theorems of these
  theories; equal weights keep their order. The first `k` of them are
  printed, with `-S` each with the score it had before.

Theories no theorem has may be named. A query with a scope (even `[]`, but
not `[view=V]` alone, below) never predicts private theorems of theories
other than its `theory`. A query
without one ranks as it does without `-L`.

A theorem out of scope is treated as the `-e` mode treats the theorems after
the goal: it is never predicted, `knn` never takes it as a neighbour, and
nothing credits its dependencies. Unlike in the `-e` mode, the statistics
stay those learned from the whole corpus: the IDF of each feature, `nbayes`'s
dependency counts and `mepo`'s symbol frequencies count every theorem, in
scope or not. The rankings of a query without a scope, and of the `-e` mode,
are as without scopes.

### Changing the corpus

In interactive mode, a line starting with `!` is a command that changes the
corpus, so that a process can follow a library as it is edited:

```
!add <syms> <deps> <seq> <theories>
!replace <theory> <syms> <deps> <seq> <theories>
```

`!add` reads theorems from the four files, in the formats above, and adds
them after the others; their ids must be new. `!replace` first removes the
theorems of `<theory>` (none if it has none), then adds those of the files,
which are usually the theory's theorems as they now are. Paths are separated
by single spaces and cannot contain one. A file that cannot be read is an
error before the corpus changes. A dependency of an earlier theorem on
an added one is not read again, so it is missing, as with snapshots
(`-i`), and `-F`'s flags are only those of the theorems read at start-up.
`predict` then learns the whole corpus again, so that it ranks as a process
started on the changed corpus would, up to rounding (features may be
numbered otherwise, which changes the order of some sums), and prints the
number of theorems the process holds (see "Versions and views"). With `-P`, the files' types and
frequency-only tokens are read too. Commands work neither with `-x` nor with
`-y`.

### Versions and views

One process can also rank several variants of a library, such as the
library as each of several copies of a script edits it. It stores *versions*
of theories, and a *view* picks one version for each of some theories:

```
!version <theory> <id> <syms> <deps> <seq> <theories>
!view <view> <file>
!drop <view>
```

`!version` stores a version of `<theory>` under the caller's `<id>` (no
spaces), its theorems read from the four files, in the formats above. Every
theorem of `<seq>` must have a line in `<theories>`, of `<theory>`. Storing a
version of a theory under an id already stored is an error.

`!view` creates the view `<view>` (a name like a theory's), or changes it,
from the lines of `<file>`: `<theory> <id>` sets the theory's version, and
`<theory>` alone removes the theory from the view. A line naming a version
that is not stored is an error, and so is a view whose theories would share
a theorem id, as two theorems with one id are in `<seq>`. `!drop` removes a view. After each `!view`
and `!drop`, the versions no view uses are dropped, so a version is stored
right before the `!view` that uses it.

A query whose scope has `view=V` ranks exactly as a process started on the
theorems of `V` would rank it, with `-L` giving their theories: the theorems
of `V`'s versions, in the order the versions were stored, each version's in
`<seq>` order. So equal scores are ranked in that order, `before=n` counts
the theorems of `V`'s version of the theory, a dependency names a theorem
and means `V`'s version of it, a query feature that no theorem of `V` has is
dropped (with `-P`, one that `V`'s theorems have only in frequency-only
tokens is kept, as such a process keeps it), and the statistics (IDF, ...)
are learned from `V`'s theorems alone. `[view=V]` alone ranks as a query without a scope; other keys make a
scope within `V`. A view that does not exist is an error.

The theorems read at start-up and by `!add` and `!replace` form the default
view, which queries without `view=` rank, as before. Named views hold only
versions, never the default view's theorems, and storing versions changes
nothing for the default view.

Every command prints the number of theorems the process holds, in the
default view and in every stored version. A command that cannot be run (a
wrong number of arguments, a file that cannot be read) is an error before
anything changes. Versions have no `-F` flags. As with `!add` and
`!replace`, the commands work neither with `-x` nor with `-y`.

Every method has views (`mepo` also with `-P`), each ranking as a fresh
process on the view's theorems, as above. Each theorem's features and dependencies
(with `-P`, its typed symbols and frequency-only tokens) are stored once per
version, and each view has what its method learns from the view's theorems:
its own IDF table, over the features of its theorems, and for `cosine` the
norm of each of its theorems; for `nbayes`, the theorems that learn each
theorem's features as a dependency (`-d` applies per view), and each
theorem's baseline and the postings (`nbayes-dense`: each theorem's counts);
for `mepo`, each symbol's frequency (with `-P`, its occurrences at each type,
and the frequencies of each theorem's symbols at their types); `mash` has
those of `knn` and `nbayes`, and `mesh` those of `mepo` and `mash`;
`rforest` has its own forest, grown from the view's theorems in the view's
order as a process started on them grows it, and the tables it is grown from
(each theorem's features and dependencies, each feature's theorems).

`tests/views.py` measures the memory and the view changes of a process on a
synthetic corpus written by `tests/gen.py`.

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

With `-P`, `mepo` reads the symbols' types ("Typed symbols" above) and scores
as Sledgehammer does with them, which needs a query on standard input (not
`-e`). A symbol is a name with a type, and one type generalizes another if it
matches it with the type variables `_` instantiated; a type with more
patterns than another generalizes it if its first patterns do. A theorem's symbol counts as relevant if its type
generalizes the type of a symbol of that name in the relevant set, neither
relevant nor irrelevant if one of those generalizes its type or a relevant
symbol of the theorem with the same name generalizes it, and irrelevant
otherwise. A relevant symbol's `freq` sums the counts of its name at
the types its type generalizes, over all theorems (frequency-only tokens
included), and an irrelevant one's at the types that generalize its type. Free variables (`v`) weigh 1.5 times as
much; `irrel` weights are multiplied by `1.05 ^ (order - 1)`; `a` weighs 0.5
relevant and 2 irrelevant, and `t` 0.5 and 0.25, whatever their frequency. A
theorem whose relevant and irrelevant symbols are all `a` and `t` scores
0. If the query has no symbol, the relevant set starts as the symbols of
the local theorems (`-F`).

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
  `mepo-cosine`. `-F` gives `mepo` the theorems' statures, and `-P` makes it
  match symbols by their types.
- `knn` sorts only as many neighbours as it needs.
- `-S` prints scores; `--version` prints the commit `predict` was built from.
- `-o` and `-i` write and read snapshots of the corpus, and features are
  known by a hash of their text.
- Queries may carry scopes (`-L`, `-b`), and `!add` and `!replace` change the
  corpus of a running process.
- A process may hold versions of theories and views of them (`!version`,
  `!view`, `!drop`, `view=`).

## Provenance and licence

The sources were taken from CoqHammer's `src/predict/` directory,
https://github.com/lukaszcz/coqhammer, at commit
`27b80e8f2aa1a53d8d2a0642302736bad14c20ed`. The first commit of this
repository contains those files unmodified, with CoqHammer's `LICENSE`.

`predict` is by Łukasz Czajka and Cezary Kaliszyk and is distributed under the
GNU Lesser General Public License, version 2.1 (see `LICENSE`). Every change
after the first commit is ours and is distributed under the same licence.
