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
for m in knn mepo; do
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
number=9
for m in nbayes cosine; do
  msg=
  for q in "$DATA/lists.queries" "$WORK/lists.wqueries"; do
    same lists $m $m-dense 16 "$q" || msg="$msg $(basename "$q")"
  done
  same synth $m $m-dense 5000 "$WORK/synth.all" || msg="$msg synth.queries+wqueries"
  same synth $m $m-dense 5000 /dev/null -e "$WORK/synth.eval" || msg="$msg synth.eval"
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

[ $failures -eq 0 ]
