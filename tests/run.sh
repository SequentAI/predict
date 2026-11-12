#!/usr/bin/env bash
# Usage: tests/run.sh <predict binary> [--update-golden]
#
# Runs the regression tests against the given binary and exits non-zero if
# any fails. With --update-golden, rewrites tests/golden/ from the binary
# instead of comparing against it.

set -u

PREDICT=$(realpath "$1")
UPDATE=${2:-}
TESTS=$(cd "$(dirname "$0")" && pwd)
DATA=$TESTS/data
GOLDEN=$TESTS/golden
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

failures=0
number=0

ok() { echo "ok $number - $1"; }
fail() { echo "FAIL $number - $1: $2"; failures=$((failures + 1)); }

# run <name> <options...> < query: runs predict on a corpus written in
# $WORK/<name>.{syms,deps,seq}, with a limit of $LIMIT seconds (default 5)
run() {
  local name=$1
  shift
  timeout "${LIMIT:-5}" "$PREDICT" "$WORK/$name.syms" "$WORK/$name.deps" "$WORK/$name.seq" \
    "$@" 2>"$WORK/stderr"
}

# corpus <name> <seq> <syms> <deps>: writes a corpus; each argument is the
# file's contents
corpus() {
  printf '%s' "$2" > "$WORK/$1.seq"
  printf '%s' "$3" > "$WORK/$1.syms"
  printf '%s' "$4" > "$WORK/$1.deps"
}

# finite_scores: reads `-S` output and succeeds if every score is finite
finite_scores() {
  local out
  out=$(cat)
  [ -n "$out" ] && ! grep -qiE '\((-?nan|-?inf)' <<<"$out"
}

# scores_of: reads `-S` output and prints `id score` pairs, one per line
scores_of() {
  tr ' ' '\n' | grep -v '^$' | paste - - | sed 's/\t(\(.*\))$/ \1/'
}

# 1. option parsing ends where `char` is unsigned (aarch64)
number=1
corpus small $'a\nb\nc\n' $'a:"x"\nb:"x", "y"\nc:"z"\n' $'b:a\n'
out=$(run small -p knn -n 8 <<<'"x"')
status=$?
if [ $status -ne 0 ]; then
  fail "option parsing" "exit status $status (124 is the 5 s timeout)"
elif [ "$(wc -w <<<"$out")" -ne 3 ]; then
  fail "option parsing" "expected 3 predictions, got '$out'"
else
  ok "option parsing"
fi

# 2. the last theorem of <seq> is learned
number=2
corpus last $'a\nb\nc\n' $'a:"x"\nb:"y"\nc:"z"\n' $'b:a\n'
top=$(run last -p nbayes -n 1 <<<'"z"')
if [ "$top" != "c " ]; then
  fail "last theorem learned" "nbayes top for the last theorem's features is '$top', not 'c'"
elif ! printf '"z"\n"x"\n"x", "y", "z"\n' | run last -p nbayes -n 3 -S | finite_scores; then
  fail "last theorem learned" "a naive Bayes score is not finite"
else
  ok "last theorem learned"
fi

# 3. featureless theorems and empty queries
number=3
corpus empty $'c\nd\na\nb\n' $'c:"x"\nd:"y"\na:\nb:\n' $'d:c\n'
printf 'b\n' > "$WORK/empty.eval"
corpus lone $'a\n' $'a:\n' ''
msg=
out=$(run lone -p rforest -n 1 <<<'')
status=$?
if [ $status -ne 0 ] || [ "$out" != "a " ]; then
  msg="rforest on a featureless one-theorem corpus: exit status $status, '$out'"
fi
for m in knn nbayes mepo rforest; do
  [ -n "$msg" ] && break
  out=$(run empty -p $m -n 4 -S <<<'')
  status=$?
  if [ $status -ne 0 ]; then
    msg="$m on an empty query: exit status $status, $(head -1 "$WORK/stderr")"
    break
  elif [ "$(scores_of <<<"$out" | wc -l)" -ne 4 ] || ! finite_scores <<<"$out"; then
    msg="$m on an empty query: '$out'"
    break
  fi
done
if [ -z "$msg" ]; then
  # b's knn neighbours: if featureless theorems shared a feature, a would be
  # a neighbour with positive similarity and b's top score would exceed the
  # bare age of the first neighbour, 500000000
  out=$(run empty -p knn -n 1 -S -e "$WORK/empty.eval")
  if [[ $out != "b:"?" (500000000) " ]]; then
    msg="featureless theorems share a feature: '$out'"
  elif ! run empty -p knn -n 4 -S <<<'"x"' | finite_scores; then
    msg="knn score not finite with a neighbour without dependencies"
  fi
fi
if [ -n "$msg" ]; then fail "empty features" "$msg"; else ok "empty features"; fi

# 4. escaped features and theorem ids containing `:`
number=4
corpus escape $'p:q\nr\\s\nt\n' \
  $'p\\:q:"a\\"b", "c\\\\d"\nr\\\\s:"e\\nf"\nt:"g"\n' \
  $'t:p\\:q r\\\\s\n'
msg=
for q in '"a\"b"' '"c\\d"' '"e\nf"'; do
  case $q in '"e\nf"') want='r\s ' ;; *) want='p:q ' ;; esac
  out=$(run escape -p knn -n 1 <<<"$q")
  if [ "$out" != "$want" ]; then
    msg="query $q gave '$out', not '$want'"
    break
  fi
done
if [ -z "$msg" ]; then
  out=$(run escape -p knn -n 2 <<<'"g"')
  if [ "$out" != 'p:q r\s ' ] || grep -q missing "$WORK/stderr"; then
    msg="dependencies of t with escaped ids gave '$out', stderr '$(cat "$WORK/stderr")'"
  fi
fi
if [ -z "$msg" ]; then
  # rforest's exported trees keep features and ids with quotes and newlines
  corpus forest $'a\nb\nc\nd\ne"f\n' \
    $'a:"p\\"q", "z"\nb:"z"\nc:"p\\"q", "z"\nd:"z", "p"\ne"f:"n\\nl", "p"\n' ''
  mkdir "$WORK/trees"
  q='"p\"q", "z"'
  a=$(run forest -p rforest -t 1 -x "$WORK/trees" <<<"$q")
  b=$(run forest -p rforest -t 1 -y "$WORK/trees" <<<"$q")
  status=$?
  if [ -z "$a" ] || [ $status -ne 0 ] || [ "$a" != "$b" ]; then
    msg="rforest ranks '$a' before export and '$b' after import (exit status $status, $(head -1 "$WORK/stderr"))"
  fi
fi
if [ -n "$msg" ]; then fail "escaping" "$msg"; else ok "escaping"; fi

# 5. a dependency later in <seq> than its dependent is kept
number=5
corpus future $'a\nb\nc\nd\n' $'a:"x", "y"\nb:"z"\nc:"x"\nd:"w"\n' $'a:b\n'
out=$(run future -p knn -n 2 <<<'"x", "y"')
# with -e, predictions for c come only from a and b; a's dependency b is
# later than a but earlier than c, and must be predicted
printf 'c\n' > "$WORK/future.eval"
msg=
if [ "$out" != "b a " ]; then
  msg="knn gave '$out', not 'b a' (b is only reachable as a's dependency)"
else
  for m in knn nbayes mepo rforest; do
    out=$(run future -p $m -n 2 -e "$WORK/future.eval")
    status=$?
    if [ $status -ne 0 ] || [[ $out != "c:"[ab]" "[ab]" " ]]; then
      msg="$m with -e: exit status $status, '$out'"
      break
    fi
  done
fi
# a dependency at or after the evaluated theorem is not predicted for it
corpus ahead $'a\nb\nc\n' $'a:"x"\nb:"y"\nc:"z"\n' $'a:c\n'
printf 'b\n' > "$WORK/ahead.eval"
for m in knn nbayes mepo rforest; do
  [ -n "$msg" ] && break
  out=$(run ahead -p $m -n 2 -e "$WORK/ahead.eval")
  status=$?
  if [ $status -ne 0 ] || [ "$out" != "b:a " ]; then
    msg="$m with -e and a dependency past the evaluated theorem: exit status $status, '$out'"
  fi
done
if [ -n "$msg" ]; then fail "future dependency" "$msg"; else ok "future dependency"; fi

# 6. repeated features count once
number=6
cp "$DATA/lists.seq" "$WORK/dedup.seq"
cp "$DATA/lists.deps" "$WORK/dedup.deps"
cp "$DATA/lists.syms" "$WORK/dedup.syms"
cp "$DATA/lists.seq" "$WORK/dup.seq"
cp "$DATA/lists.deps" "$WORK/dup.deps"
sed -e 's/^\(rev_append:\)\(.*\)$/\1\2, "c:rev", \2/' \
    -e 's/^\(length_def:"c:length", \)/\1"c:length", /' \
    "$DATA/lists.syms" > "$WORK/dup.syms"
msg=
for m in knn nbayes mepo; do
  a=$(run dedup -p $m -n 16 -S < "$DATA/lists.queries")
  b=$(run dup -p $m -n 16 -S < "$DATA/lists.queries")
  c=$(sed 's/^\("[^"]*"\)\(.*\)$/\1\2, \1/' "$DATA/lists.queries" | run dedup -p $m -n 16 -S)
  if [ -z "$a" ] || [ "$a" != "$b" ]; then
    msg="$m: repeated features in <syms> change the ranking"
    break
  elif [ "$a" != "$c" ]; then
    msg="$m: repeated features in the query change the ranking"
    break
  fi
done
if [ -n "$msg" ]; then fail "repeated features" "$msg"; else ok "repeated features"; fi

# 7. query weights
number=7
cp "$DATA/lists.seq" "$WORK/lists.seq"
cp "$DATA/lists.deps" "$WORK/lists.deps"
cp "$DATA/lists.syms" "$WORK/lists.syms"
msg=
plain='"c:rev", "c:map", "t:list"'
for m in knn; do
  a=$(run lists -p $m -n 16 -S <<<"$plain")
  b=$(run lists -p $m -n 16 -S <<<'"c:rev"@0.01, "c:map"@3, "t:list"@0')
  if [ -z "$a" ] || [ "$a" != "$b" ]; then
    msg="$m does not treat weights as 1.0"
    break
  fi
done
if [ -z "$msg" ]; then
  a=$(run lists -p nbayes -n 16 -S <<<"$plain")
  b=$(run lists -p nbayes -n 16 -S <<<'"c:rev"@1, "c:map"@1.0, "t:list"@1e0')
  if [ -z "$a" ] || [ "$a" != "$b" ]; then
    msg="nbayes: weight 1 differs from no weight"
  fi
fi
if [ -z "$msg" ]; then
  # each query feature's term in a naive Bayes score is scaled by its weight,
  # so a score is linear in a weight: s(0.5) = (s(0) + s(1)) / 2
  for w in 0 0.5 1; do
    run lists -p nbayes -n 16 -S <<<"\"c:rev\"@$w, \"c:map\"" | scores_of | sort > "$WORK/w$w"
  done
  bad=$(join "$WORK/w0" "$WORK/w0.5" | join - "$WORK/w1" |
    awk '{ d = $3 - ($2 + $4) / 2; if (d < 0) d = -d; if (d > 1e-9 || NF != 4) print $1 }')
  if [ -n "$bad" ] || [ "$(wc -l < "$WORK/w0.5")" -ne 16 ]; then
    msg="nbayes scores are not linear in the weight for: $bad"
  fi
fi
if [ -z "$msg" ]; then
  a=$(run lists -p nbayes -n 1 <<<'"c:rev"@0.05, "c:map"')
  b=$(run lists -p nbayes -n 1 <<<'"c:rev", "c:map"@0.05')
  if [ "$a" != "map_rev " ] || [ "$b" != "rev_rev " ]; then
    msg="nbayes top predictions '$a' and '$b' do not follow the weights"
  fi
fi
if [ -z "$msg" ]; then
  # scaling every weight alike leaves cosine scores unchanged, also where the
  # squared weight overflows or underflows
  # or where the largest weight is that of a feature every theorem has
  corpus scale $'b\na\n' $'b:"y", "z"\na:"x", "z"\n' ''
  for m in cosine cosine-dense; do
    for q in '"x"@1e200' '"x"@1e-200' '"z"@1e200, "x"' '"z", "x"@1e-200'; do
      out=$(run scale -p $m -n 2 -S <<<"$q")
      if [ "$out" != "a (1) b (0) " ]; then
        msg="$m gave '$out' for $q"
        break 2
      fi
    done
  done
fi
if [ -n "$msg" ]; then fail "query weights" "$msg"; else ok "query weights"; fi

# 8. the number of dependencies naive Bayes learns
number=8
# t depends on d1 ... d30, which are in <seq> in reverse; naive Bayes ranks
# t first for t's feature, then the dependencies that learned it, then the
# others, equal scores in <seq> order
seq= syms= deps=t:
for i in $(seq 30 -1 1); do seq+="d$i"$'\n'; syms+="d$i:\"y$i\""$'\n'; done
for i in $(seq 1 30); do deps+=" d$i"; done
corpus many "${seq}t"$'\n' "${syms}t:\"x\""$'\n' "$deps"$'\n'
first20="t $(seq -f 'd%g' -s ' ' 20 -1 1) $(seq -f 'd%g' -s ' ' 30 -1 21) "
all30="t $(seq -f 'd%g' -s ' ' 30 -1 1) "
msg=
for m in nbayes nbayes-dense; do
  for opt in "" "-d 20" "-d 0" "-d 30" "-d 31"; do
    case $opt in "" | "-d 20") want=$first20 ;; *) want=$all30 ;; esac
    out=$(run many -p $m -n 31 $opt <<<'"x"')
    if [ "$out" != "$want" ]; then
      msg="$m ${opt:-without -d} gave '$out', not '$want'"
      break 2
    fi
  done
done
if [ -n "$msg" ]; then fail "naive Bayes dependency limit" "$msg"; else ok "naive Bayes dependency limit"; fi

# same <name> <method> <reference> <n> <query file> [options...]: succeeds if
# both methods give the same ranking of the n best theorems for every query
same() {
  local name=$1 m=$2 ref=$3 n=$4 queries=$5
  shift 5
  local a b
  a=$(run $name -p $m -n $n "$@" < "$queries") || return 1
  b=$(run $name -p $ref -n $n "$@" < "$queries") || return 1
  [ -n "$a" ] && [ "$a" = "$b" ]
}

# weight <query file>: the queries with some features weighted
weight() {
  sed -e 's/"\(c:[a-z]*\)"/"\1"@0.5/' -e 's/"\(t:[a-z]*\)"/"\1"@2.5/' \
      -e 's/"p:eq"/"p:eq"@0/' "$1"
}

# 9 and 10. sparse naive Bayes and cosine against the methods scanning every
# theorem, on the test corpus and on a synthetic corpus of 5000 theorems, on
# whole rankings
LIMIT=600
weight "$DATA/lists.queries" > "$WORK/lists.wqueries"
python3 "$TESTS/gen.py" "$WORK/synth" 5000 20 2000 1
for i in $(seq 0 50 4999); do echo "t$i"; done > "$WORK/synth.eval"
cat "$WORK/synth.queries" "$WORK/synth.wqueries" > "$WORK/synth.all"
corpus ties "$(printf 't%s\n' 0 1 2 3 4 5 6 7)" \
  't0:
t1:"f0"
t2:"f3", "f1"
t3:"f3"
t4:
t5:"f3"
t6:"f3"
t7:"f1", "f3"
' ''
corpus near $'a\nb\nc\n' $'a:"x", "y"\nb:\nc:"y"\n' ''
number=9
for m in nbayes cosine; do
  msg=
  for q in "$DATA/lists.queries" "$WORK/lists.wqueries"; do
    same lists $m $m-dense 16 "$q" || msg="$msg $(basename "$q")"
  done
  same synth $m $m-dense 5000 "$WORK/synth.all" || msg="$msg synth.queries+wqueries"
  same synth $m $m-dense 5000 /dev/null -e "$WORK/synth.eval" || msg="$msg synth.eval"
  # every theorem's features are in the query, with weight 0: naive Bayes
  # scores every theorem 30 ln 1000 and cosine 0, so both rank in <seq> order
  echo '"f0"@0, "f1"@0, "f3"@0' > "$WORK/ties.queries"
  same ties $m $m-dense 8 "$WORK/ties.queries" || msg="$msg ties"
  [ "$(run ties -p $m -n 8 < "$WORK/ties.queries")" = "t0 t1 t2 t3 t4 t5 t6 t7 " ] ||
    msg="$msg ties(order)"
  # the tie crosses the cutoff
  same ties $m $m-dense 3 "$WORK/ties.queries" || msg="$msg ties(-n 3)"
  # naive Bayes scores a above b by about 4e-11 for the first query, and
  # equal to it for the second; rounding apart from the scan over every
  # theorem would swap them
  printf '%s\n' '"x"@0.026002243966793848' '"x"@0.026002243965038505' > "$WORK/near.queries"
  same near $m $m-dense 3 "$WORK/near.queries" || msg="$msg near"
  [ $m = cosine ] || [ "$(run near -p $m -n 3 < "$WORK/near.queries")" = "a b c "$'\n'"a b c " ] ||
    msg="$msg near(order)"
  # the scores agree up to rounding
  for q in "$DATA/lists.queries" "$WORK/lists.wqueries"; do
    run lists -p $m -n 16 -S < "$q" | tr ' ' '\n' > "$WORK/sparse"
    run lists -p $m-dense -n 16 -S < "$q" | tr ' ' '\n' > "$WORK/dense"
    paste "$WORK/sparse" "$WORK/dense" | awk -F'\t' '
      $1 ~ /^\(/ { a = substr($1, 2) + 0; b = substr($2, 2) + 0; d = a - b
                   if (d < 0) d = -d; m = b < 0 ? -b : b
                   if (d > 1e-9 * (m + 1)) bad = 1; next }
      $1 != $2 { bad = 1 }
      END { exit bad || NR < 16 }' || msg="$msg scores($(basename "$q"))"
  done
  if [ -n "$msg" ]; then
    fail "$m equals $m-dense" "rankings differ on$msg"
  else
    ok "$m equals $m-dense"
  fi
  number=$((number + 1))
done
LIMIT=5

# golden rankings on the test corpus
number=11
msg=
for m in knn nbayes mepo cosine; do
  out=$(run lists -p $m -n 10 < "$DATA/lists.queries")
  if [ "$UPDATE" = --update-golden ]; then
    printf '%s\n' "$out" > "$GOLDEN/$m.out"
  elif [ "$out" != "$(cat "$GOLDEN/$m.out")" ]; then
    msg="$msg $m"
  fi
done
if [ -n "$msg" ]; then
  fail "golden rankings" "differ for$msg"
else
  ok "golden rankings"
fi

# MePo. Below, freq is the number of theorems containing a symbol, a relevant
# symbol weighs rel(f) = 1 + 2 / ln(f + 1), an irrelevant one
# irrel(f) = ln(f + 1) / ln 100 below frequency 100 and rel(f) / rel(100)
# from 100 on, and a theorem scores R / (R + I) with R and I the sums of
# those weights.

# score_of <id>: reads `-S` output and prints the score of theorem <id>
score_of() { scores_of | awk -v id="$1" '$1 == id { print $2 }'; }

# near <a> <b>: succeeds if the numbers a and b differ by at most 1e-6
near() { awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; exit !(a != "" && d <= 1e-6 && d >= -1e-6) }'; }

# ids_of: reads `-S` output and prints the ids, separated by spaces
ids_of() { scores_of | awk '{ printf "%s ", $1 }'; }

# repeat <n> <line>: prints <line> with `#` replaced by 1 ... n, one per line
repeat() { for i in $(seq 1 "$1"); do printf '%s\n' "${2//\#/$i}"; done; }

# 12. a theorem's score
# For the query k$q: a has k$q (freq 1, rel(1) = 1 + 2 / ln 2 = 3.885390)
# and k$u (freq 1, irrel(1) = ln 2 / ln 100 = 0.150515), and scores
# 3.885390 / (3.885390 + 0.150515) = 0.962706; d has no relevant symbol and
# scores 0. For the query k$r, k$s: b has k$r (freq 3, rel(3) = 1 + 2 / ln 4
# = 2.442695) and k$v (freq 2, irrel(2) = ln 3 / ln 100 = 0.238561), and
# scores 2.442695 / 2.681256 = 0.911027; c has only relevant symbols and
# scores 1.
number=12
corpus score $'a\nb\nc\nd\ne\n' \
  $'a:"k$q", "k$u"\nb:"k$r", "k$v"\nc:"k$r", "k$s"\nd:"k$v"\ne:"k$r", "k$s"\n' ''
msg=
out=$(run score -p mepo -n 5 -S <<<'"k$q"')
near "$(score_of a <<<"$out")" 0.962706 || msg="a: '$out'"
near "$(score_of d <<<"$out")" 0 || msg="d: '$out'"
out=$(run score -p mepo -n 4 -S <<<'"k$r", "k$s"')
if [ -z "$msg" ]; then
  near "$(score_of b <<<"$out")" 0.911027 || msg="b: '$out'"
  near "$(score_of c <<<"$out")" 1 || msg="c: '$out'"
fi
if [ -n "$msg" ]; then fail "mepo score" "$msg"; else ok "mepo score"; fi

# 13. the weight of an irrelevant symbol peaks at frequency 100
number=13
# a has the query's symbol k$q (freq 1, rel(1) = 3.885390) and k$u, which
# n - 1 other theorems also have. irrel(n) for n = 1, 10, 99, 100, 101 and
# 400 is ln 2 / ln 100 = 0.150515, ln 11 / ln 100 = 0.520696, ln 100 / ln 100
# = 1, rel(100) / rel(100) = 1, rel(101) / rel(100) = 1.433119 / 1.433370 =
# 0.999356 and rel(400) / rel(100) = 1.333735 / 1.433370 = 0.930451, so a
# scores 3.885390 / (3.885390 + irrel(n)): 0.962706, 0.881823, 0.795308,
# 0.795308, 0.795413 and 0.806794.
msg=
for nw in 1:0.962706 10:0.881823 99:0.795308 100:0.795308 101:0.795413 400:0.806794; do
  n=${nw%:*}
  corpus u "$(printf 'a\n'; repeat $((n - 1)) 'f#')"$'\n' \
    "$(printf 'a:"k$q", "k$u"\n'; repeat $((n - 1)) 'f#:"k$u"')"$'\n' ''
  out=$(run u -p mepo -n 1 -S <<<'"k$q"')
  near "$(score_of a <<<"$out")" "${nw#*:}" || { msg="freq $n: '$out'"; break; }
done
if [ -n "$msg" ]; then fail "mepo irrelevant weight" "$msg"; else ok "mepo irrelevant weight"; fi

# 14. symbols only in the goal's assumptions (weight below 1)
number=14
# a has k$c (freq 1, rel(1) = 3.885390) and k$h (freq 1). With k$h only in
# the assumptions, it is not relevant and weighs 0.25 irrel(1) = 0.25 *
# 0.150515 = 0.037629: a scores 3.885390 / 3.923019 = 0.990408. Outside the
# query k$h weighs irrel(1): 0.962706. In the conclusion too, it is relevant:
# 1. With only assumptions, their symbols are the relevant ones, and k$c
# weighs irrel(1): 0.962706.
corpus asm $'a\n' $'a:"k$c", "k$h"\n' ''
msg=
for qw in '"k$c", "k$h"@0.5:0.990408' '"k$c":0.962706' \
          '"k$c", "k$h"@0.5, "k$h":1' '"k$h"@0.5:0.962706'; do
  out=$(run asm -p mepo -n 1 -S <<<"${qw%:*}")
  near "$(score_of a <<<"$out")" "${qw##*:}" || { msg="query ${qw%:*}: '$out'"; break; }
done
if [ -n "$msg" ]; then fail "mepo assumption symbols" "$msg"; else ok "mepo assumption symbols"; fi

# 15. stature bonuses
number=15
# a, b, c and d have k$g and k$u (freq 4 each): R = rel(4) = 1 + 2 / ln 5 =
# 2.242670 and I = irrel(4) = ln 5 / ln 100 = 0.349485. a has no flag:
# 2.242670 / 2.592155 = 0.865176. b is a simp rule, I - 0.15 = 0.199485:
# 2.242670 / 2.442155 = 0.918316. c is local, I - 0.55 = -0.200515:
# 2.242670 / 2.042155 = 1.098188. d is both, and the simp bonus wins:
# 0.918316. All are accepted at once, by score.
corpus stat $'a\nb\nc\nd\n' "$(repeat 4 'x#:"k$g", "k$u"' | sed 's/^x1/a/; s/^x2/b/; s/^x3/c/; s/^x4/d/')"$'\n' ''
printf 'b:simp\nc:local\nd:local simp\n' > "$WORK/stat.statures"
msg=
out=$(run stat -p mepo -n 4 -S -F "$WORK/stat.statures" <<<'"k$g"')
for iw in a:0.865176 b:0.918316 c:1.098188 d:0.918316; do
  near "$(score_of ${iw%:*} <<<"$out")" ${iw#*:} || { msg="'$out'"; break; }
done
if [ -z "$msg" ] && [ "$(ids_of <<<"$out")" != "c b d a " ]; then
  msg="order '$out'"
fi
if [ -z "$msg" ]; then
  out=$(run stat -p mepo -n 4 -S <<<'"k$g"')
  for i in a b c d; do
    near "$(score_of $i <<<"$out")" 0.865176 || { msg="without -F: '$out'"; break; }
  done
fi
if [ -n "$msg" ]; then fail "mepo statures" "$msg"; else ok "mepo statures"; fi

# 16. at most ceil(11.5 ^ (remaining / max_facts)) imperfect facts per
# iteration
number=16
# p1 ... p20 have k$g and k$v (freq 20 each): R = rel(20) = 1 + 2 / ln 21 =
# 1.656917, I = irrel(20) = ln 21 / ln 100 = 0.661110, score 0.714796. With
# -n 20, the first iteration accepts ceil(11.5 ^ (20 / 20)) = 12 of them, in
# <seq> order; k$v becomes relevant, so p13 ... p20 score 1 (perfect) and
# are accepted next.
corpus cap "$(repeat 20 'p#')"$'\n' "$(repeat 20 'p#:"k$g", "k$v"')"$'\n' ''
want="$(repeat 20 'p#' | tr '\n' ' ')"
msg=
out=$(run cap -p mepo -n 20 -S <<<'"k$g"')
if [ "$(ids_of <<<"$out")" != "$want" ]; then
  msg="order '$out'"
else
  for i in 1 12; do near "$(score_of p$i <<<"$out")" 0.714796 || msg="p$i: '$out'"; done
  for i in 13 20; do near "$(score_of p$i <<<"$out")" 1 || msg="p$i: '$out'"; done
fi
if [ -n "$msg" ]; then fail "mepo imperfect cap" "$msg"; else ok "mepo imperfect cap"; fi

# 17. The ranking follows the order of acceptance, then the score; this
# also checks the cap with fewer facts remaining. Three iterations, with -n 20:
# the query is k$g. Iteration 0: p1 ... p8 have only k$g (score 1); b has
# k$g (freq 9, rel(9) = 1 + 2 / ln 10 = 1.868589) and k$h (freq 13,
# irrel(13) = ln 14 / ln 100 = 0.573064), scoring 1.868589 / 2.441653 =
# 0.765297. All nine pass the threshold 0.45 and are accepted, by score: p1 ...
# p8, b. The threshold decays by d = (0.15 / 0.55) ^ (1 / 21) = 0.939983 per
# accepted fact, to 1 - 0.55 d^9 = 0.684839. Iteration 1: k$h is relevant,
# so q1 ... q12, with k$h (rel(13) = 1 + 2 / ln 14 = 1.757846) and k$z (freq
# 12, irrel(12) = ln 13 / ln 100 = 0.556972), score 1.757846 / 2.314818 =
# 0.759389 >= 0.684839; with 11 of 20 remaining, ceil(11.5 ^ (11 / 20)) =
# ceil(3.832) = 4 of them are accepted, q1 ... q4. Iteration 2: k$z is
# relevant, q5 ... q12 score 1, and the 7 remaining slots take q5 ... q11.
number=17
corpus iters "$(printf 'b\n'; repeat 8 'p#'; repeat 12 'q#')"$'\n' \
  "$(printf 'b:"k$g", "k$h"\n'; repeat 8 'p#:"k$g"'; repeat 12 'q#:"k$h", "k$z"')"$'\n' ''
want="$(repeat 8 'p#' | tr '\n' ' ')b $(repeat 11 'q#' | tr '\n' ' ')"
msg=
out=$(run iters -p mepo -n 20 -S <<<'"k$g"')
if [ "$(ids_of <<<"$out")" != "$want" ]; then
  msg="order '$out'"
else
  for iw in p1:1 b:0.765297 q1:0.759389 q4:0.759389 q5:1 q11:1; do
    near "$(score_of ${iw%:*} <<<"$out")" ${iw#*:} || { msg="${iw%:*}: '$out'"; break; }
  done
fi
if [ -n "$msg" ]; then fail "mepo acceptance order" "$msg"; else ok "mepo acceptance order"; fi

# 18. the threshold decays with every accepted fact
number=18
# With -n 4, d = (0.15 / 0.55) ^ (1 / 5) = 0.771162. Iteration 0 accepts p
# (k$g only, score 1) and b (k$g, freq 2, rel(2) = 1 + 2 / ln 3 = 2.820478;
# k$h, freq 3, irrel(3) = ln 4 / ln 100 = 0.301030: 0.903563). The threshold
# becomes 1 - 0.55 d^2 = 0.672920. Iteration 1: k$h is relevant (rel(3) =
# 2.442695). x has k$h, k$c (freq 100, irrel 1) and k$x (freq 2, irrel(2) =
# 0.238561): 2.442695 / 3.681256 = 0.663549 < 0.672920, rejected though it
# passes 0.45. z has k$h, k$e (freq 100, irrel 1) and k$z (freq 1, irrel(1) =
# 0.150515): 2.442695 / 3.593210 = 0.679809, accepted. Iteration 2 makes k$e
# relevant, and e1 (k$e only) takes the last slot.
corpus decay "$(printf 'p\nb\nx\nz\n'; repeat 99 'c#'; repeat 99 'e#')"$'\n' \
  "$(printf 'p:"k$g"\nb:"k$g", "k$h"\nx:"k$h", "k$c", "k$x"\nz:"k$h", "k$e", "k$z"\n'
     repeat 98 'c#:"k$c"'; printf 'c99:"k$c", "k$x"\n'; repeat 99 'e#:"k$e"')"$'\n' ''
msg=
out4=$(run decay -p mepo -n 4 -S <<<'"k$g"')
if [ "$(ids_of <<<"$out4")" != "p b z e1 " ]; then
  msg="order '$out4'"
else
  for iw in p:1 b:0.903563 z:0.679809; do
    near "$(score_of ${iw%:*} <<<"$out4")" ${iw#*:} || { msg="${iw%:*}: '$out4'"; break; }
  done
fi
if [ -n "$msg" ]; then fail "mepo threshold decay" "$msg"; else ok "mepo threshold decay"; fi

# 19. symbols are matched by their whole name
number=19
corpus names $'a\nb\nc\n' $'a:"list$LENGTH"\nb:"rich_list$LENGTH"\nc:"LENGTH"\n' ''
out=$(run names -p mepo -n 3 -S <<<'"list$LENGTH"')
if [ "$out" != "a (1) b (0) c (0) " ]; then
  fail "mepo symbol names" "'$out'"
else
  ok "mepo symbol names"
fi

[ $failures -eq 0 ]
