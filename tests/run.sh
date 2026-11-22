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

# a test passes only if none of its checked runs exited non-zero; a failure
# names the first such run
ok() { if [ -s "$WORK/failed" ]; then fail "$1"; else echo "ok $number - $1"; fi; }
fail() {
  local why=${2:-}
  if [ -s "$WORK/failed" ]; then
    why="${why:+$why; }$(head -n 1 "$WORK/failed")"
    rm "$WORK/failed"
  fi
  echo "FAIL $number - $1: $why"
  failures=$((failures + 1))
}

# run <name> <options...> < query: runs predict on a corpus written in
# $WORK/<name>.{syms,deps,seq}, with a limit of $LIMIT seconds (default 5)
run() {
  local name=$1
  shift
  timeout "${LIMIT:-5}" "$PREDICT" "$WORK/$name.syms" "$WORK/$name.deps" "$WORK/$name.seq" \
    "$@" 2>"$WORK/stderr"
}

# checked <name> <options...> < query: run, for a run that must succeed; a
# non-zero exit status, which output piped on or captured in $(...) hides,
# is written to $WORK/failed and fails the test. $note, if set, tells the
# run from others with the same options.
checked() {
  run "$@"
  local status=$?
  [ $status -eq 0 ] || echo "'$*'${note:+ ($note)} exited with status $status" >> "$WORK/failed"
  return $status
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
for m in knn nbayes mepo mepo-cosine mash mesh rforest; do
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
  for m in knn nbayes mepo mepo-cosine mash mesh rforest; do
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
for m in knn nbayes mepo mepo-cosine mash mesh rforest; do
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
for m in knn nbayes mepo mepo-cosine mash mesh; do
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
for m in knn mepo-cosine; do
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
for m in knn nbayes mepo mepo-cosine cosine mash mesh; do
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

# 20. mepo-cosine adds the symbols of each batch of 100 accepted theorems to
# the query, weighted by their scores
number=20
# The query is k$q. p1 ... p99 have only k$q and score 1; m has k$q and k$w
# and scores below 1 but above the rest, which score 0, so the first batch is
# p1 ... p99, m. With 102 theorems, tfidf(k$q) = ln 102 - ln 100 = 0.0198026
# and tfidf(k$w) = ln 102 - ln 2 = 3.931826; m scores
# q^2 / (q^2 + w^2) = 2.536566e-05 with q, w those tfidfs, and k$w joins the
# query with that weight c. Against the new query, z (k$w only) scores
# w^2 c^2 / (q^2 + w^2 c^2) = 2.536437e-05 and takes the 101st place ahead of
# n (k$n only), which still scores 0 and comes before z in <seq>. A single
# pass ranks n first.
corpus batch "$(repeat 99 'p#'; printf 'm\nn\nz\n')" \
  "$(repeat 99 'p#:"k$q"'; printf '%s\n' 'm:"k$q", "k$w"' 'n:"k$n"' 'z:"k$w"')" ''
out=$(run batch -p mepo-cosine -n 102 -S <<<'"k$q"')
ids=$(ids_of <<<"$out")
msg=
[ "$ids" = "$(repeat 99 'p# ' | tr -d '\n')m z n " ] || msg="order: '$ids'"
# the scores, times 1e5
e5() { awk -v x="$1" 'BEGIN { if (x != "") printf "%.9f\n", x * 1e5 }'; }
near "$(e5 "$(score_of m <<<"$out")")" 2.536566 || msg="m: '$out'"
near "$(e5 "$(score_of z <<<"$out")")" 2.536437 || msg="z: '$out'"
# a single pass, as cosine makes, ranks n before z
cos=$(run batch -p cosine -n 102 <<<'"k$q"')
[ "${cos% }" != "${cos%n z }" ] || msg="cosine: '$cos'"
if [ -n "$msg" ]; then
  fail "mepo-cosine batches" "$msg"
else
  ok "mepo-cosine batches"
fi

# MaSh and MeSh. A theorem at rank r (from 0) of an input ranking weighs
# s(r) = 0.62 ^ log2(r + 1): s(0) = 1, s(1) = 0.62, s(2) = 0.468906,
# s(3) = 0.3844, s(4) = 0.329720, s(5) = 0.290721. The fused ranking orders
# theorems by the sum of their weights, each leg's weights scaled alike.

# features <from> <to>: prints `, "e<from>", ..., "e<to>"`
features() { for i in $(seq "$1" "$2"); do printf ', "e%s"' "$i"; done; }

# rank_of <id>: reads predictions and prints the rank (from 0) of <id>
rank_of() { tr ' ' '\n' | grep -v '^$' | grep -n -x -F "$1" | cut -d: -f1 | awk '{ print $1 - 1 }'; }

# a and b share both query features but have many others, which naive Bayes
# penalises and knn ignores
corpus fuse $'a\nb\nc\nd\ne\nf\n' \
  "a:\"x\", \"y\"$(features 1 10)
b:\"x\", \"y\"$(features 20 30)
c:\"x\", \"e40\"
d:\"x\"
e:\"z\"
f:\"w\"
" ''

# 21. mash sums the steep weights of knn's and naive Bayes's rankings
number=21
# For the query x, y with -n 4, knn ranks a b c d and naive Bayes d c a b, so
# a scores s(0) + s(2) = 1.468906, b s(1) + s(3) = 1.0044, c s(2) + s(1) =
# 1.088906 and d s(3) + s(0) = 1.3844: a d c b.
msg=
knn=$(run fuse -p knn -n 4 <<<'"x", "y"')
nb=$(run fuse -p nbayes -n 4 <<<'"x", "y"')
out=$(run fuse -p mash -n 4 <<<'"x", "y"')
if [ "$knn" != "a b c d " ] || [ "$nb" != "d c a b " ]; then
  msg="the inputs are knn '$knn' and nbayes '$nb', not 'a b c d' and 'd c a b'"
elif [ "$out" != "a d c b " ]; then
  msg="'$out', not 'a d c b'"
fi
if [ -n "$msg" ]; then fail "mash fusion" "$msg"; else ok "mash fusion"; fi

# 22. -c turns off the credit knn gives the dependencies of neighbours
number=22
# a depends on f. With credit (-c 1, the default), knn ranks f a b c and
# naive Bayes f d c a: f scores s(0) + s(0) = 2, a s(1) + s(3) = 1.0044,
# b s(2) = 0.468906, c s(3) + s(2) = 0.853306 and d s(1) = 0.62: f a c d.
# Without it (-c 0), knn ranks a b c d: a scores s(0) + s(3) = 1.3844, b
# s(1) = 0.62, c s(2) + s(2) = 0.937812, d s(3) + s(1) = 1.0044 and f s(0)
# = 1: a d f c.
cp "$WORK/fuse.seq" "$WORK/credit.seq"
cp "$WORK/fuse.syms" "$WORK/credit.syms"
printf 'a:f\n' > "$WORK/credit.deps"
msg=
for cw in "-c 1:f a b c :f d c a :f a c d " ":f a b c :f d c a :f a c d " \
          "-c 0:a b c d :f d c a :a d f c "; do
  IFS=: read -r opt wknn wnb want <<<"$cw"
  knn=$(run credit -p knn -n 4 $opt <<<'"x", "y"')
  nb=$(run credit -p nbayes -n 4 $opt <<<'"x", "y"')
  out=$(run credit -p mash -n 4 $opt <<<'"x", "y"')
  if [ "$knn" != "$wknn" ] || [ "$nb" != "$wnb" ] || [ "$out" != "$want" ]; then
    msg="${opt:-without -c}: knn '$knn', nbayes '$nb', mash '$out'"
    break
  fi
done
if [ -z "$msg" ] && run credit -p mash -c 2 <<<'"x"' >/dev/null; then
  msg="-c 2 is accepted"
fi
if [ -n "$msg" ]; then fail "mash dependency credit" "$msg"; else ok "mash dependency credit"; fi

# MeSh fixtures: g and h join the corpus above
cp "$WORK/fuse.seq" "$WORK/mesh.seq"
printf 'g\nh\n' >> "$WORK/mesh.seq"
cp "$WORK/fuse.syms" "$WORK/mesh.syms"
printf '%s\n' 'g:"y", "z"' 'h:"y"' >> "$WORK/mesh.syms"
: > "$WORK/mesh.deps"
# p1 ... p48 have only x; w has x and u, z x and v, q1 ... q12 x and v, and
# r1 ... r20 only y
corpus long "$(repeat 48 'p#'; printf 'w\nz\n'; repeat 12 'q#'; repeat 20 'r#')"$'\n' \
  "$(repeat 48 'p#:"x"'; printf '%s\n' 'w:"x", "u"' 'z:"x", "v"'
     repeat 12 'q#:"x", "v"'; repeat 20 'r#:"y"')"$'\n' ''

# 23. mesh with MePo cut to M and MaSh to 2M + 25 (the default)
number=23
# For the query x, y with -n 4 (M = 4), mepo -n 4 ranks d h c g and mash,
# cut to 2M + 25 = 33 or the 8 theorems there are, d a h b c g e f. The
# candidates are the first 4 of each; d scores s(0) + s(0) = 2, h s(1) +
# s(2) = 1.088906, c s(2) + s(4) = 0.798626, g s(3) + s(5) = 0.675121, a
# s(1) = 0.62 and b s(3) = 0.3844: d h c g.
# On the long corpus with -n 50, mepo -n 50 ranks w 48th and mash -n 80
# (2M + 25 = 125 cut to 80 theorems) ranks w 54th, so w scores
# 0.5 (s(48) + s(54)) / m = 0.399700, where m = 0.164305 is the mean of
# s(0) ... s(49).
msg=
mepo=$(run mesh -p mepo -n 4 <<<'"x", "y"')
mash=$(run mesh -p mash -n 8 <<<'"x", "y"')
if [ "$mepo" != "d h c g " ] || [ "$mash" != "d a h b c g e f " ]; then
  msg="the inputs are mepo '$mepo' and mash '$mash', not 'd h c g' and 'd a h b c g e f'"
fi
for opt in "" "-T 2M+25"; do
  [ -n "$msg" ] && break
  out=$(run mesh -p mesh -n 4 $opt <<<'"x", "y"')
  [ "$out" = "d h c g " ] || msg="${opt:-without -T}: '$out', not 'd h c g'"
done
if [ -z "$msg" ]; then
  mepo=$(run long -p mepo -n 50 <<<'"x"' | rank_of w)
  mash=$(run long -p mash -n 80 <<<'"x"' | rank_of w)
  out=$(run long -p mesh -n 50 -S <<<'"x"')
  if [ "$mepo" != 48 ] || [ "$mash" != 54 ]; then
    msg="w ranks $mepo in mepo and $mash in mash, not 48 and 54"
  else
    near "$(score_of w <<<"$out")" 0.399700 || msg="w: '$out'"
  fi
fi
if [ -n "$msg" ]; then fail "mesh cut 2M+25" "$msg"; else ok "mesh cut 2M+25"; fi

# 24. mesh with both inputs cut to 51n/50
number=24
# With -n 4, 51n/50 = 4: mepo -n 4 ranks d h c g and mash -n 4 d a c b. d
# scores s(0) + s(0) = 2, c s(2) + s(2) = 0.937812, a s(1) = 0.62, h s(1) =
# 0.62, g s(3) = 0.3844 and b s(3) = 0.3844: d c a h, a before h in <seq>.
# On the long corpus with -n 50, 51n/50 = 51: mepo -n 51 ranks w 48th and
# mash -n 51 ranks it 50th, the 51st theorem, which counts: w scores
# 0.5 (s(48) + s(50)) / m = 0.409957 (0.207812 without it).
msg=
mash=$(run mesh -p mash -n 4 <<<'"x", "y"')
out=$(run mesh -p mesh -n 4 -T 51n/50 <<<'"x", "y"')
if [ "$mash" != "d a c b " ]; then
  msg="the input is mash '$mash', not 'd a c b'"
elif [ "$out" != "d c a h " ]; then
  msg="'$out', not 'd c a h'"
else
  mepo=$(run long -p mepo -n 51 <<<'"x"' | rank_of w)
  mash=$(run long -p mash -n 51 <<<'"x"' | rank_of w)
  out=$(run long -p mesh -n 50 -S -T 51n/50 <<<'"x"')
  if [ "$mepo" != 48 ] || [ "$mash" != 50 ]; then
    msg="w ranks $mepo in mepo and $mash in mash, not 48 and 50"
  else
    near "$(score_of w <<<"$out")" 0.409957 || msg="w: '$out'"
  fi
fi
if [ -z "$msg" ] && run mesh -p mesh -T 51n <<<'"x"' >/dev/null; then
  msg="-T 51n is accepted"
fi
if [ -n "$msg" ]; then fail "mesh cut 51n/50" "$msg"; else ok "mesh cut 51n/50"; fi

# 25. typed symbols: a fact's symbol whose type patterns generalize the
# goal's is relevant
number=25
# Each case has its own symbols, each in one fact (freq 1): rel(1) = 3.885390,
# irrel(1) = 0.150515. The goal has k$f at type k$num; a has k$f at any type
# (`_`), which generalizes it, and k$u. a's k$f also generalizes b's, so its
# freq is 2, rel(2) = 2.820478: 2.820478 / (2.820478 + 0.150515) = 0.949338.
# Untyped, k$f matches by name alike, but a fact with k$f at k$bool would be
# relevant too: typed, b is not, and scores 0.
corpus types $'a\nb\nc\nd\ne\n' \
  $'a:"k$f"<c 1 _>, "k$u"\nb:"k$f"<c 1 "k$bool">, "k$u2"\nc:"k$g", "k$f2"<c 1 "k$num">, "k$v"\nd:"k$h", "k$e"<c 1 "k$bool">, "k$w"\ne:"k$z"\n' ''
msg=
out=$(run types -p mepo -P -n 5 -S <<<'"k$f"<c 1 "k$num">')
near "$(score_of a <<<"$out")" 0.949338 || msg="a: '$out'"
near "$(score_of b <<<"$out")" 0 || msg="b: '$out'"
if [ -n "$msg" ]; then fail "typed mepo: generalizing fact relevant" "$msg"; else ok "typed mepo: generalizing fact relevant"; fi

# 26. a fact's symbol whose type patterns the goal's generalize counts on
# neither side
number=26
# The goal has k$g and k$f2 at any type; c has k$g (relevant), k$f2 at k$num
# (neither) and k$v (irrelevant): 3.885390 / (3.885390 + 0.150515) =
# 0.962706. Were k$f2 relevant, c would score 2 * 3.885390 / (2 * 3.885390
# + 0.150515) = 0.980999; irrelevant, 3.885390 / (3.885390 + 2 * 0.150515)
# = 0.928094.
msg=
out=$(run types -p mepo -P -n 5 -S <<<'"k$g", "k$f2"<c 1 _>')
near "$(score_of c <<<"$out")" 0.962706 || msg="c: '$out'"
if [ -n "$msg" ]; then fail "typed mepo: generalized fact neutral" "$msg"; else ok "typed mepo: generalized fact neutral"; fi

# 27. a fact's symbol at a type incompatible with the goal's is irrelevant
number=27
# The goal has k$h and k$e at k$num; d has k$h (relevant), k$e at k$bool
# (irrelevant) and k$w (irrelevant): 3.885390 / (3.885390 + 2 * 0.150515) =
# 0.928094. Untyped, k$e is relevant: 0.980999.
msg=
out=$(run types -p mepo -P -n 5 -S <<<'"k$h", "k$e"<c 1 "k$num">')
near "$(score_of d <<<"$out")" 0.928094 || msg="typed d: '$out'"
out=$(run types -p mepo -n 5 -S <<<'"k$h", "k$e"<c 1 "k$num">')
[ -z "$msg" ] && { near "$(score_of d <<<"$out")" 0.980999 || msg="untyped d: '$out'"; }
if [ -n "$msg" ]; then fail "typed mepo: incompatible fact irrelevant" "$msg"; else ok "typed mepo: incompatible fact irrelevant"; fi

# 28. typed weights: variables, theories, abstractions, orders and counts
number=28
# a has k$g (freq 1, rel 3.885390), the variable x (1.5 * 3.885390 =
# 5.828085), the theory t (0.5), all relevant to the goal: R = 10.213475.
# The abstraction l is irrelevant (2.0) and so is k$h, of order 3 and
# occurring twice: irrel(2) * 1.05 ^ 2 = ln 3 / ln 100 * 1.1025 = 0.263013.
# a scores 10.213475 / (10.213475 + 2.263013) = 0.818618. With k$h in the
# goal's assumptions only, it weighs a quarter: 10.213475 / (10.213475 + 2 +
# 0.065753) = 0.831768.
corpus tw $'a\nb\n' $'a:"k$g", "x"<v 0>, "t"<t 0>, "l"<a 2>, "k$h"<c 3>*2\nb:"k$y"\n' ''
msg=
out=$(run tw -p mepo -P -n 2 -S <<<'"k$g", "x"<v 0>, "t"<t 0>')
near "$(score_of a <<<"$out")" 0.818618 || msg="a: '$out'"
out=$(run tw -p mepo -P -n 2 -S <<<'"k$g", "x"<v 0>, "t"<t 0>, "k$h"<c 3>@0.5')
[ -z "$msg" ] && { near "$(score_of a <<<"$out")" 0.831768 || msg="a, k\$h assumed: '$out'"; }
# an irrelevant symbol of the largest order weighs 1.05 ^ (10^9 - 1) = inf:
# c scores 0
corpus hi $'c\n' $'c:"k$g", "k$u"<c 1000000000>\n' ''
[ -z "$msg" ] && { out=$(run hi -p mepo -P -n 1 -S <<<'"k$g"') &&
  near "$(score_of c <<<"$out")" 0 || msg="c, the largest order: '$out'"; }
if [ -n "$msg" ]; then fail "typed mepo weights" "$msg"; else ok "typed mepo weights"; fi

# 29. a goal with no symbol starts from the local facts' symbols; one with
# only a theory symbol doesn't
number=29
# With no symbol, l is local and has k$p, so k$p is relevant: l scores 1,
# and m, with k$p (freq 2, rel(2) = 1 + 2 / ln 3 = 2.820478) and k$q
# (irrel(1)), 2.820478 / (2.820478 + 0.150515) = 0.949338 (it is not local).
# With "t" alone, only t has a relevant symbol, and it is a theory, so every
# theorem scores 0.
corpus fb $'l\nm\nt\n' $'l:"k$p"\nm:"k$p", "k$q"\nt:"t"<t 0>\n' ''
printf 'l:local\n' > "$WORK/fb.statures"
msg=
out=$(run fb -p mepo -P -n 3 -S -F "$WORK/fb.statures" <<<'')
near "$(score_of m <<<"$out")" 0.949338 || msg="m, empty query: '$out'"
out=$(run fb -p mepo -P -n 3 -S -F "$WORK/fb.statures" <<<'"t"<t 0>')
[ -z "$msg" ] && { near "$(score_of m <<<"$out")" 0 || msg="m, query t: '$out'"; }
if [ -n "$msg" ]; then fail "typed mepo local fallback" "$msg"; else ok "typed mepo local fallback"; fi

# 30. untyped corpora and queries load with -P, and typed ones without it
number=30
# Without types, every symbol is a constant of order 1 with no patterns, so
# -P ranks the test corpus as mepo does. Without -P, types and counts are
# read and ignored: the typed corpus ranks as its untyped copy. (mepo on the
# synthetic corpus is too slow for the checked build.)
msg=
if ! run lists -p mepo -P -n 1000 -S < "$DATA/lists.queries" > "$WORK/typed.out"; then
  msg="-P on the untyped lists corpus fails: $(head -1 "$WORK/stderr")"
elif ! run lists -p mepo -n 1000 -S < "$DATA/lists.queries" > "$WORK/untyped.out"; then
  msg="mepo on the lists corpus fails: $(head -1 "$WORK/stderr")"
elif [ ! -s "$WORK/typed.out" ] || ! cmp -s "$WORK/typed.out" "$WORK/untyped.out"; then
  msg="-P on the untyped lists corpus ranks differently"
fi
if [ -z "$msg" ]; then
  sed 's/<[^>]*>//g; s/\*[0-9]*//g' "$WORK/types.syms" > "$WORK/stripped.syms"
  cp "$WORK/types.seq" "$WORK/stripped.seq"; cp "$WORK/types.deps" "$WORK/stripped.deps"
  for m in knn nbayes mepo; do
    a=$(run types -p $m -n 5 -S <<<'"k$h", "k$e"')
    b=$(run stripped -p $m -n 5 -S <<<'"k$h", "k$e"')
    [ -n "$a" ] && [ "$a" = "$b" ] || { msg="$m: typed '$a', stripped '$b'"; break; }
  done
fi
if [ -n "$msg" ]; then fail "typed and untyped files" "$msg"; else ok "typed and untyped files"; fi

# 31. malformed types are errors, -P needs -p mepo and queries
number=31
msg=
for bad in '"k$f"<x 1>' '"k$f"<c>' '"k$f"<c 1 k>' '"k$f"<c 1 "k"(_>' '"k$f"*2' \
    '"k$f"<c 1000000001>'; do
  if run types -p mepo -P -n 1 <<<"$bad" >/dev/null 2>&1; then
    msg="query $bad accepted"; break
  fi
done
for bad in '"k$f"<c 1>*1000000001' '"k$f"<c 1>*0, "k$g"' '"x"<v 0>*0'; do
  corpus big $'a\n' "a:$bad"$'\n' ''
  if [ -z "$msg" ] && run big -p mepo -P -n 1 <<<'"k$f"' >/dev/null 2>&1; then
    msg="corpus $bad accepted"
  fi
done
# abstractions and theories may have count 0: their weights ignore counts
corpus abs0 $'a\n' $'a:"k$f", "l"<a 1>*0\n' ''
if [ -z "$msg" ] && ! run abs0 -p mepo -P -n 1 <<<'"k$f"' >/dev/null 2>&1; then
  msg="corpus abstraction count 0 rejected"
fi
printf 'a\n' > "$WORK/t.eval"
if [ -z "$msg" ] && run types -p knn -P -n 1 <<<'"k$f"' >/dev/null 2>&1; then
  msg="-P accepted with knn"
elif [ -z "$msg" ] && run types -p mepo -P -n 1 -e "$WORK/t.eval" >/dev/null 2>&1; then
  msg="-P accepted with -e"
fi
if [ -n "$msg" ]; then fail "typed input errors" "$msg"; else ok "typed input errors"; fi

# 32. frequency-only tokens count in typed frequencies and nowhere else
number=32
# a has k$g (freq 1, rel 3.885390) relevant and k$h irrelevant. b's 98
# frequency-only occurrences of k$h at a's type make k$h's freq 99, so
# irrel(99) = ln 100 / ln 100 = 1 and a scores 3.885390 / 4.885390 =
# 0.795308. At an incompatible type they don't count: k$h's freq stays 1,
# irrel(1) = ln 2 / ln 100 = 0.150515 and a scores 0.962706. They are no
# symbols of b: with the query k$z, which only a frequency-only token has,
# every theorem scores 0. Without -P they are ignored.
corpus fo $'a\nb\n' $'a:"k$g", "k$h"<c 1 "k$num">\nb:"k$h"<c 1 "k$num">+98, "k$z"+5\n' ''
corpus fox $'a\nb\n' $'a:"k$g", "k$h"<c 1 "k$num">\nb:"k$h"<c 1 "k$bool">+98\n' ''
corpus fos $'a\nb\n' $'a:"k$g", "k$h"<c 1 "k$num">\nb:\n' ''
msg=
out=$(run fo -p mepo -P -n 2 -S <<<'"k$g"')
near "$(score_of a <<<"$out")" 0.795308 || msg="a: '$out'"
[ -z "$msg" ] && { out=$(run fox -p mepo -P -n 2 -S <<<'"k$g"') &&
  near "$(score_of a <<<"$out")" 0.962706 || msg="a, other type: '$out'"; }
[ -z "$msg" ] && { out=$(run fo -p mepo -P -n 2 -S <<<'"k$z"') &&
  near "$(score_of a <<<"$out")" 0 && near "$(score_of b <<<"$out")" 0 ||
  msg="query k\$z: '$out'"; }
if [ -z "$msg" ]; then
  for m in knn nbayes mepo; do
    a=$(run fo -p $m -n 2 -S <<<'"k$g", "k$z"')
    b=$(run fos -p $m -n 2 -S <<<'"k$g", "k$z"')
    [ -n "$a" ] && [ "$a" = "$b" ] || { msg="$m: with '$a', without '$b'"; break; }
  done
fi
for bad in '"k$f"+0' '"k$f"<c 1>+1000000001'; do
  corpus big $'a\n' "a:\"k\$g\", $bad"$'\n' ''
  if [ -z "$msg" ] && run big -p mepo -P -n 1 <<<'"k$g"' >/dev/null 2>&1; then
    msg="corpus $bad accepted"
  fi
done
if [ -z "$msg" ] && run fo -p mepo -P -n 1 <<<'"k$g"+2' >/dev/null 2>&1; then
  msg="query with +2 accepted"
fi
if [ -n "$msg" ]; then fail "frequency-only tokens" "$msg"; else ok "frequency-only tokens"; fi

# 33. a snapshot ranks as the text corpus it was written from, and so does a
# snapshot with theorems added after it
number=33
corpus none '' '' ''
# from_snapshot <name> <options...> < query: runs predict on the snapshot
# $WORK/<name>.snap, with no theorems added
from_snapshot() {
  local name=$1
  shift
  run none -i "$WORK/$name.snap" "$@"
}
LIMIT=600
msg=
for name in lists synth ties; do
  run $name -o "$WORK/$name.snap" || { msg="writing $name.snap: $(head -1 "$WORK/stderr")"; break; }
done
for m in knn nbayes mepo mepo-cosine cosine mash mesh; do
  [ -n "$msg" ] && break
  for q in "$DATA/lists.queries" "$WORK/lists.wqueries"; do
    a=$(run lists -p $m -n 16 -S < "$q")
    b=$(from_snapshot lists -p $m -n 16 -S < "$q")
    [ -n "$a" ] && [ "$a" = "$b" ] || { msg="$m on $(basename "$q")"; break 2; }
  done
done
if [ -z "$msg" ]; then
  # ties: equal scores keep <seq> order, also across the cutoff
  for n in 8 3; do
    a=$(run ties -p cosine -n $n -S < "$WORK/ties.queries")
    b=$(from_snapshot ties -p cosine -n $n -S < "$WORK/ties.queries")
    [ -n "$a" ] && [ "$a" = "$b" ] || { msg="cosine on the ties, -n $n"; break; }
  done
fi
if [ -z "$msg" ]; then
  a=$(run synth -p cosine -n 5000 -S < "$WORK/synth.all")
  b=$(from_snapshot synth -p cosine -n 5000 -S < "$WORK/synth.all")
  [ -n "$a" ] && [ "$a" = "$b" ] || msg="cosine on synth"
fi
if [ -z "$msg" ]; then
  a=$(run synth -p cosine -n 5000 -S -e "$WORK/synth.eval")
  b=$(from_snapshot synth -p cosine -n 5000 -S -e "$WORK/synth.eval")
  [ -n "$a" ] && [ "$a" = "$b" ] || msg="cosine on synth with -e"
fi
if [ -z "$msg" ]; then
  # the first 4000 theorems as a snapshot, the other 1000 as text after it;
  # each theorem has one line in each file, in <seq> order
  for f in seq syms deps; do
    head -n 4000 "$WORK/synth.$f" > "$WORK/head.$f"
    tail -n +4001 "$WORK/synth.$f" > "$WORK/rest.$f"
  done
  run head -o "$WORK/head.snap" 2>/dev/null
  a=$(run synth -p cosine -n 5000 -S < "$WORK/synth.all")
  b=$(run rest -i "$WORK/head.snap" -p cosine -n 5000 -S < "$WORK/synth.all")
  [ -n "$a" ] && [ "$a" = "$b" ] || msg="cosine on a snapshot of synth's first 4000 theorems and the rest as text"
fi
LIMIT=5
if [ -n "$msg" ]; then fail "snapshot rankings" "rankings differ: $msg"; else ok "snapshot rankings"; fi

# 34. a snapshot from another version, truncated, or not a snapshot at all is
# refused with exit status 3, and a caller falling back to the text corpus
# ranks as the text
number=34
version=$("$PREDICT" --version | cut -d' ' -f2)
python3 - "$WORK/lists.snap" "$WORK/old.snap" "$version" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
version = sys.argv[3].encode()
assert data.count(version) == 1
open(sys.argv[2], "wb").write(data.replace(version, b"0" * len(version)))
PY
size=$(wc -c < "$WORK/lists.snap")
head -c $((size - 1)) "$WORK/lists.snap" > "$WORK/short.snap"
head -c $((size / 2)) "$WORK/lists.snap" > "$WORK/half.snap"
head -c 30 "$WORK/lists.snap" > "$WORK/head30.snap"
# caller <snapshot> < query: ranks from the snapshot, or from the text corpus
# if predict refuses it
caller() {
  local q out status
  q=$(cat)
  out=$(from_snapshot "$1" -p cosine -n 16 -S <<<"$q")
  status=$?
  if [ $status -eq 3 ]; then run lists -p cosine -n 16 -S <<<"$q"; else printf '%s\n' "$out"; return $status; fi
}
want=$(run lists -p cosine -n 16 -S < "$DATA/lists.queries")
msg=
for snap in old short half head30 lists.syms; do
  case $snap in lists.syms) f=$DATA/lists.syms ;; *) f=$WORK/$snap.snap ;; esac
  out=$(run none -i "$f" -p cosine -n 16 < "$DATA/lists.queries")
  status=$?
  if [ $status -ne 3 ] || [ -n "$out" ] || ! grep -q "^Error: cannot use snapshot" "$WORK/stderr"; then
    msg="$snap: exit status $status, output '$out', stderr '$(head -1 "$WORK/stderr")'"
    break
  fi
  case $snap in
    old) grep -q "written by predict 0" "$WORK/stderr" || msg="old: stderr '$(cat "$WORK/stderr")'" ;;
    short | half | head30) grep -q "truncated" "$WORK/stderr" || msg="$snap: stderr '$(cat "$WORK/stderr")'" ;;
  esac
  [ -n "$msg" ] && break
  cp "$f" "$WORK/fallback.snap"
  out=$(caller fallback < "$DATA/lists.queries")
  [ "$out" = "$want" ] || { msg="$snap: the caller's fallback ranks '$out'"; break; }
done
if [ -z "$msg" ]; then
  run none -i "$WORK/lists.snap" -p mepo -P -n 1 <<<'' >/dev/null && msg="-P accepted with -i"
fi
if [ -z "$msg" ]; then
  mkdir -p "$WORK/snaptrees"
  run none -i "$WORK/lists.snap" -p rforest -t 1 -x "$WORK/snaptrees" -n 1 <<<'' >/dev/null &&
    msg="-x accepted with -i"
fi
if [ -n "$msg" ]; then fail "snapshot refused" "$msg"; else ok "snapshot refused"; fi


# Query scopes. The fixture's theories form a DAG: Mid builds on Base,
# Script (the querying script) on Mid, and Down on Script; Other is apart,
# and its theorem Other.o1 is private.
dag_seq=$'Base.b1\nBase.b2\nBase.t\nMid.m1\nMid.t\nOther.o1\nOther.o2\nScript.s1\nScript.s2\nScript.s3\nDown.d1\nDown.d2\n'
dag_syms=$'Base.b1:"p", "q"\nBase.b2:"q"\nBase.t:"t"\nMid.m1:"p", "r"\nMid.t:"t"\nOther.o1:"p", "q"\nOther.o2:"r"\nScript.s1:"p"\nScript.s2:"p", "q"\nScript.s3:"p", "q", "r", "u"\nDown.d1:"p", "q"\nDown.d2:"p", "q", "r"\n'
dag_deps=$'Mid.m1:Base.b1\nScript.s2:Mid.m1 Base.b2\nScript.s3:Script.s1\nDown.d1:Script.s2\nDown.d2:Script.s3 Mid.m1\n'
corpus dag "$dag_seq" "$dag_syms" "$dag_deps"
sed 's/^\([A-Za-z]*\)\..*/&:\1/; s/^Other.o1:Other$/& private/' "$WORK/dag.seq" > "$WORK/dag.thy"
methods="knn nbayes mepo mepo-cosine cosine mash mesh"

# words: reads predict's output and prints its ids sorted, on one line
words() { tr ' ' '\n' | grep -v '^$' | sort | tr '\n' ' '; }

# scoped <query> <expected ids>: for every method, succeeds if the query's
# predictions are exactly the expected ids, in any order; else sets msg
scoped() {
  local m out
  for m in $methods; do
    out=$(run dag -p $m -L "$WORK/dag.thy" -n 20 <<<"$1" | words)
    if [ "$out" != "$(words <<<"$2")" ]; then
      msg="$m on '$1': '$out'"
      return 1
    fi
  done
}

# 35. a scope's excluded theories (the cone downstream of the script) are
# never predicted
number=35
msg=
scoped '[exclude=Down]"p", "q", "r"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o2 Script.s1 Script.s2 Script.s3' &&
  scoped '[exclude=Script,Down,Nothing]"p"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o2'
if [ -n "$msg" ]; then fail "scope excludes theories" "$msg"; else ok "scope excludes theories"; fi

# 36. a private theorem is predicted only for a query of its own theory
number=36
msg=
scoped '[theory=Script exclude=Down]"p"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o2 Script.s1 Script.s2 Script.s3' &&
  scoped '[theory=Other exclude=Down]"p"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o1 Other.o2 Script.s1 Script.s2 Script.s3'
if [ -n "$msg" ]; then fail "scope hides private theorems" "$msg"; else ok "scope hides private theorems"; fi

# 37. before=n keeps only the first n theorems of the query's theory
number=37
msg=
scoped '[theory=Script before=2 exclude=Down]"p"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o2 Script.s1 Script.s2' &&
  scoped '[theory=Script before=0 exclude=Down]"p"' 'Base.b1 Base.b2 Base.t Mid.m1 Mid.t Other.o2'
if [ -n "$msg" ]; then fail "scope cuts the script at before" "$msg"; else ok "scope cuts the script at before"; fi

# 38. the ancestors' boost decides the order of two theorems with equal
# scores, Base.t and Mid.t, which otherwise come in <seq> order
number=38
# first_of <a> <b>: reads predict's output and prints whichever of the ids a
# and b comes first
first_of() { tr ' ' '\n' | grep -m 1 -x -e "$1" -e "$2"; }
msg=
for m in $methods; do
  plain=$(run dag -p $m -L "$WORK/dag.thy" <<<'"t"' | first_of Base.t Mid.t)
  boosted=$(run dag -p $m -L "$WORK/dag.thy" <<<'[ancestors=Mid]"t"' | first_of Base.t Mid.t)
  unboosted=$(run dag -p $m -L "$WORK/dag.thy" -b 1 <<<'[ancestors=Mid]"t"' | first_of Base.t Mid.t)
  if [ "$plain $boosted $unboosted" != "Base.t Mid.t Base.t" ]; then
    msg="$m: first $plain, boosted $boosted, with -b 1 $unboosted"
    break
  fi
done
if [ -n "$msg" ]; then fail "ancestors boost" "$msg"; else ok "ancestors boost"; fi

# 39. k sets the number of predictions of one query
number=39
out=$(checked dag -p cosine -L "$WORK/dag.thy" -n 5 <<<$'[k=3]"p"\n"p"\n[k=50]"p"')
if [ "$(sed -n 1p <<<"$out" | wc -w)" -ne 3 ] || [ "$(sed -n 2p <<<"$out" | wc -w)" -ne 5 ] ||
   [ "$(sed -n 3p <<<"$out" | wc -w)" -ne 11 ]; then
  fail "scope k" "got '$out'"
else
  ok "scope k"
fi

# 40. the scope filters before the best k are chosen: with the 100 best
# theorems out of scope, a query still gets k = 8 in scope
number=40
{ for i in $(seq 100); do echo "Hot.h$i"; done; for i in $(seq 20); do echo "Cold.c$i"; done; } > "$WORK/hot.seq"
{ for i in $(seq 100); do echo "Hot.h$i:\"x\", \"y\""; done; for i in $(seq 20); do echo "Cold.c$i:\"x\", \"z$i\""; done; } > "$WORK/hot.syms"
: > "$WORK/hot.deps"
sed 's/^\([A-Za-z]*\)\..*/&:\1/' "$WORK/hot.seq" > "$WORK/hot.thy"
msg=
top=$(checked hot -p cosine -n 100 <<<'"x", "y"' | tr ' ' '\n' | grep -c '^Hot\.')
[ "$top" -eq 100 ] || msg="the 100 best of cosine hold $top Hot theorems"
for m in $methods; do
  [ -n "$msg" ] && break
  out=$(checked hot -p $m -L "$WORK/hot.thy" <<<'[k=8 exclude=Hot]"x", "y"')
  [ "$(wc -w <<<"$out")" -eq 8 ] && ! grep -q Hot <<<"$out" || msg="$m: '$out'"
done
if [ -n "$msg" ]; then fail "scope filters before selection" "$msg"; else ok "scope filters before selection"; fi

# 41. without a scope, -L changes no ranking
number=41
sed 's/.*/&:lists/' "$DATA/lists.seq" > "$WORK/lists.thy"
msg=
for m in $methods; do
  a=$(checked lists -p $m -n 16 -S < "$DATA/lists.queries")
  b=$(checked lists -p $m -n 16 -S -L "$WORK/lists.thy" < "$DATA/lists.queries")
  [ -n "$a" ] && [ "$a" = "$b" ] || { msg="$m"; break; }
done
if [ -n "$msg" ]; then fail "-L alone ranks as before" "differ for $msg"; else ok "-L alone ranks as before"; fi

# 42. !add and !replace change the corpus of a running predict: it then ranks
# as a predict started on the changed corpus. Script.s3 alone has "u", so
# after it is replaced, "u" is a feature no theorem has. With -P, on a typed
# copy of the fixture: only the added Down theorems have the type "k$set",
# Script.s3 alone has "w", as a frequency-only token, and Down.d2 alone has
# "x", as one, which stays after the replacement. Base.b1 is local, so a
# query left with no known symbol starts from its symbols.
number=42
# part <from> <to> <pattern>: the theorems of corpus <from> matching the grep
# pattern, as files $WORK/<to>.{seq,syms,deps,thy}
part() {
  local f
  for f in seq syms deps thy; do grep -E "$3" "$WORK/$1.$f" > "$WORK/$2.$f"; done
}
for f in seq deps thy; do cp "$WORK/dag.$f" "$WORK/tdag.$f"; done
printf '%s\n' \
  'Base.b1:"p"<c 1 "k$num">, "q"<c 1 _>*2' \
  'Base.b2:"q"<c 1 "k$num">, "p"<c 1 "k$num">+3' \
  'Base.t:"t"<c 1>' \
  'Mid.m1:"p"<c 1 _>, "r"<c 2 "k$num" "k$bool">' \
  'Mid.t:"t"<c 1>*3' \
  'Other.o1:"p"<c 1 "k$num">, "q"<c 1 "k$num">' \
  'Other.o2:"r"<c 2 "k$num" "k$bool">, "q"<c 1 _>+2' \
  'Script.s1:"p"<c 1 "k$num">' \
  'Script.s2:"p"<c 1 _>, "q"<c 1 "k$num">*2' \
  'Script.s3:"p"<c 1 "k$num">, "q"<c 1 _>, "r"<c 2 _ "k$bool">, "u"<v 0>, "w"+4' \
  'Down.d1:"p"<c 1 "k$set">, "q"<c 1 "k$num">' \
  'Down.d2:"p"<c 1 "k$set">*2, "q"<c 1 _>, "r"<c 2 "k$set" "k$bool">, "x"+2' \
  > "$WORK/tdag.syms"
printf 'Script.n1\nScript.n2\n' > "$WORK/new.seq"
printf 'Script.n1:"p", "v"\nScript.n2:"v"\n' > "$WORK/new.syms"
printf 'Script.n2:Script.n1 Base.b1\n' > "$WORK/new.deps"
printf 'Script.n1:Script private\nScript.n2:Script\n' > "$WORK/new.thy"
for f in seq deps thy; do cp "$WORK/new.$f" "$WORK/tnew.$f"; done
printf '%s\n' 'Script.n1:"p"<c 1 "k$set">, "v"<c 1 "k$num">*2, "q"<c 1 "k$set">+2' \
  'Script.n2:"v"<c 1 _>' > "$WORK/tnew.syms"
for c in "" t; do
  part ${c}dag ${c}up '^(Base|Mid|Other|Script)\.'
  part ${c}dag ${c}down '^Down\.'
  part ${c}dag ${c}kept '^(Base|Mid|Other|Down)\.'
  for f in seq syms deps thy; do cat "$WORK/${c}new.$f" >> "$WORK/${c}kept.$f"; done
done
queries=$'"p", "q"\n[theory=Script exclude=Down]"p", "r"@0.5\n"u", "v"\n"t"'
typed_queries=$'"p"<c 1 "k$set">, "q"\n"u"<v 0>, "p"<c 1 "k$num">@0.5\n"w", "q"@0.5\n"x", "p"@0.5\n"u"<v 0>\n"w"'
printf 'Base.b1:local\n' > "$WORK/tdag.statures"
# same_as_fresh <c> <queries> <options...>: sets msg unless predict with the
# options, started on corpus <c>up, answers the queries after `!add <c>down`
# and after `!replace Script <c>new` as when started on <c>dag and <c>kept
same_as_fresh() {
  local c=$1 q=$2 out want
  shift 2
  out=$(note='with !add and !replace' checked ${c}up -L "$WORK/${c}up.thy" -n 20 -S "$@" <<<"$q
!add $WORK/${c}down.syms $WORK/${c}down.deps $WORK/${c}down.seq $WORK/${c}down.thy
$q
!replace Script $WORK/${c}new.syms $WORK/${c}new.deps $WORK/${c}new.seq $WORK/${c}new.thy
$q")
  want="$(checked ${c}up -L "$WORK/${c}up.thy" -n 20 -S "$@" <<<"$q")
12
$(checked ${c}dag -L "$WORK/${c}dag.thy" -n 20 -S "$@" <<<"$q")
11
$(checked ${c}kept -L "$WORK/${c}kept.thy" -n 20 -S "$@" <<<"$q")"
  [ "$out" = "$want" ] || msg="$*: '$out', not '$want'"
}
msg=
for m in $methods rforest; do
  [ -n "$msg" ] && break
  same_as_fresh "" "$queries" -p $m
done
[ -n "$msg" ] || same_as_fresh t "$queries"$'\n'"$typed_queries" -p mepo -P -F "$WORK/tdag.statures"
if [ -n "$msg" ]; then fail "!add and !replace" "$msg"; else ok "!add and !replace"; fi

# 43. malformed scopes and commands are errors, as are command files giving
# an earlier theorem a feature, commands need a corpus that can change, so
# not -x or -y, and -P still does not work with snapshots
number=43
msg=
for q in '[before=2]"p"' '[k=0]"p"' '[k=x]"p"' '[size=3]"p"' '[k=2 k=3]"p"' '[exclude=]"p"' '[k=2"p"' '!remove Script' '!add a b c'; do
  run dag -p cosine -L "$WORK/dag.thy" <<<"$q" >/dev/null && { msg="'$q' accepted"; break; }
done
for c in "!add $WORK/down.syms $WORK/down.deps $WORK/missing.seq $WORK/down.thy" \
         "!replace Script $WORK/missing.syms $WORK/new.deps $WORK/new.seq $WORK/new.thy" \
         "!replace Script $WORK/new.syms $WORK/missing.deps $WORK/new.seq $WORK/new.thy"; do
  [ -n "$msg" ] && break
  out=$(run dag -p cosine -L "$WORK/dag.thy" <<<"$c")
  status=$?
  if [ $status -eq 0 ] || [ -n "$out" ] || ! grep -q "Cannot read" "$WORK/stderr"; then
    msg="'$c' with a missing file: exit status $status, output '$out'"
  fi
done
if [ -z "$msg" ]; then
  { cat "$WORK/down.syms"; printf '%s:"zz"\n' "$(head -1 "$WORK/up.seq")"; } > "$WORK/old.syms"
  out=$(run up -p cosine -L "$WORK/up.thy" <<<"!add $WORK/old.syms $WORK/down.deps $WORK/down.seq $WORK/down.thy")
  status=$?
  if [ $status -eq 0 ] || [ -n "$out" ] || ! grep -q "which the command does not add" "$WORK/stderr"; then
    msg="!add giving a feature to an earlier theorem: exit status $status, output '$out'"
  fi
fi
for o in -x -y; do
  [ -n "$msg" ] && break
  out=$(run dag -p cosine $o "$WORK/none" <<<"!add $WORK/down.syms $WORK/down.deps $WORK/down.seq $WORK/down.thy")
  status=$?
  if [ $status -eq 0 ] || [ -n "$out" ] || ! grep -q "does not work with -y or -x" "$WORK/stderr"; then
    msg="!add with $o: exit status $status, output '$out', stderr '$(head -1 "$WORK/stderr")'"
  fi
done
for o in -i -o; do
  [ -n "$msg" ] && break
  out=$(run dag -p mepo -P $o "$WORK/typed.snap" <<<'"p"')
  status=$?
  if [ $status -eq 0 ] || [ -n "$out" ] || ! grep -q -- "-P does not work with snapshots" "$WORK/stderr"; then
    msg="-P with $o: exit status $status, output '$out', stderr '$(head -1 "$WORK/stderr")'"
  fi
done
if [ -z "$msg" ]; then
  printf 'Base.b1:Base secret\n' > "$WORK/bad.thy"
  run dag -p cosine -L "$WORK/bad.thy" <<<'"p"' >/dev/null && msg="flag secret accepted"
fi
if [ -n "$msg" ]; then fail "bad scopes and commands" "$msg"; else ok "bad scopes and commands"; fi

# 44. --version prints the commit predict was built from
number=44
out=$("$PREDICT" --version)
if [[ $out =~ ^predict\ [0-9a-f]{40}$ ]]; then ok "version is a commit"; else fail "version is a commit" "--version printed '$out'"; fi

# Versions and views. Each version below is a theory's files, written to
# $WORK/<theory><id>.{seq,syms,deps,thy}. A's version 2 changes A.a2 and adds
# A.a4, whose "z" no other theorem has; C's version 1 alone has "w"; B.b2
# depends on A.a2 and A.a4; A.a1 and B.b1 have equal features, and A.a3 is
# private. Versions come in a view in the order they were stored: in v2, B's
# version 1 before A's version 2. For mepo -P, typed copies of the versions,
# written to $WORK/t<theory><id>.*, have types, counts and frequency-only
# tokens: the type "int" is only in A's version 2, which only v2 has, and
# "y" only in a frequency-only token of C's version 1.
# version <theory> <id> <syms lines>...: writes a version whose theorems are
# those of the syms lines, with B.b2's dependencies, to files prefixed $VP
VP=
version() {
  local f=$WORK/$VP$1$2 l
  shift 2
  : > "$f.seq"; : > "$f.thy"
  for l in "$@"; do
    echo "${l%%:*}" >> "$f.seq"
    echo "${l%%:*}:${l%%.*}" >> "$f.thy"
  done
  printf '%s\n' "$@" > "$f.syms"
  grep -q '^B\.b2:' "$f.syms" && echo 'B.b2:A.a2 A.a4' > "$f.deps" || : > "$f.deps"
  sed -i 's/^A\.a3:A$/& private/' "$f.thy"
}
version A 1 'A.a1:"p", "q"' 'A.a2:"q", "r"' 'A.a3:"p", "s"'
version A 2 'A.a1:"p", "q"' 'A.a2:"r", "s", "t"' 'A.a3:"p", "s"' 'A.a4:"z", "p"'
version B 1 'B.b1:"p", "q"' 'B.b2:"q", "s"' 'B.b3:"t", "u"'
version C 1 'C.c1:"u", "p"' 'C.c2:"w"'
version C 2 'C.c1:"u", "q", "r"' 'C.c3:"p"'
VP=t
version A 1 'A.a1:"p"<c 1 _>*2, "q"' 'A.a2:"q"<c 0 "nat">, "r"' \
  'A.a3:"p"<c 1 "nat">, "s", "q"+3'
version A 2 'A.a1:"p"<c 1 _>*2, "q"' 'A.a2:"r", "s"<c 0 "int">, "t"' \
  'A.a3:"p"<c 1 "nat">, "s", "q"+3' 'A.a4:"z"<v 0>, "p"<c 1 "int">*2'
version B 1 'B.b1:"p"<c 1 _>*2, "q"' 'B.b2:"q"<c 0 "nat">*3, "s"' \
  'B.b3:"t", "u"<c 1 "list"(_)>, "p"+2'
version C 1 'C.c1:"u", "p", "y"+1' 'C.c2:"w"<v 0>'
version C 2 'C.c1:"u"<c 1 "list"("nat")>, "q", "r"' 'C.c3:"p"<c 1 "nat">*2'
VP=
# store <theory> <id>: the command storing that version
store() { local f=$WORK/$VP$1$2; echo "!version $1 $2 $f.syms $f.deps $f.seq $f.thy"; }
# fresh <name> <theory><id>...: the corpus of those versions, in that order,
# as files $WORK/$VP<name>.{seq,syms,deps,thy}
fresh() {
  local name=$1 f v
  shift
  for f in seq syms deps thy; do
    for v in "$@"; do cat "$WORK/$VP$v.$f"; done > "$WORK/$VP$name.$f"
  done
}
for VP in "" t; do
  fresh v1 A1 B1 C1
  fresh v2 B1 A2
  fresh v1b A1 C2
done
VP=
printf 'A 1\nB 1\nC 1\n' > "$WORK/v1.view"
printf 'A 2\nB 1\n' > "$WORK/v2.view"
printf 'C 2\nB\n' > "$WORK/v1b.view"
# On v2, the second query's scores round otherwise when its features are
# added up in the order the process numbers them rather than the view's.
view_queries=$'"p", "q"\n"r"@1.628, "t"@1.653, "q"@1.197, "u"@2.584, "z"@0.703, "s"@0.463\n"s", "t"@0.5\n"z", "w", "u"\n"q"\n[k=2]"p", "q"\n[theory=A]"p", "s"\n[theory=B]"p", "s"\n[theory=A before=1]"p", "q", "r"\n[exclude=B]"p", "q"\n[ancestors=B]"p", "q"\n[theory=C before=1 exclude=A ancestors=B k=3]"u", "p"'
# with mepo -P, these too
typed_view_queries=$'"p"<c 1 "nat">, "q"\n"s"<c 0 "int">, "p"<c 1 _>@0.5\n"u"<c 1 "list"(_)>, "y"\n"y"\n"y", "p"@0.5\n"z"<v 0>, "q"<c 0 "nat">@0.5\n[theory=A]"p"<c 1 "int">, "s"'
# in_view <view>: the queries $q, each with view=<view> in its scope
in_view() { sed "s/^\[/[view=$1 /; t; s/^/[view=$1]/" <<<"$q"; }
# every method has views
view_methods=(cosine cosine-dense knn "knn -c 0" mepo-cosine mepo "mepo -P"
  nbayes nbayes-dense "nbayes -d 1" mash "mash -c 0" mesh rforest)

# 45. a query on a view ranks as a process started on the view's theorems,
# after views are created, changed and dropped, and the default view is as
# before. Dropping v2 drops A's version 2 and B's version 1, which only it
# uses: the count falls by their 7 theorems; stored again, they give v2
# back. mepo -P has the typed versions and the typed queries too.
number=45
msg=
for m in "${view_methods[@]}"; do
  q=$view_queries
  VP=
  if [ "$m" = "mepo -P" ]; then
    q+=$'\n'$typed_view_queries
    VP=t
  fi
  out=$(note='with views' checked dag -p $m -L "$WORK/dag.thy" -n 20 -S <<<"$(store A 1)
$(store B 1)
$(store C 1)
!view v1 $WORK/v1.view
$(store A 2)
!view v2 $WORK/v2.view
$(in_view v1)
$(in_view v2)
$q
$(store C 2)
!view v1 $WORK/v1b.view
$(in_view v1)
!drop v2
$(in_view v1)
$(store B 1)
$(store A 2)
!view v2 $WORK/v2.view
$(in_view v2)")
  v1b=$(checked ${VP}v1b -p $m -L "$WORK/${VP}v1b.thy" -n 20 -S <<<"$q")
  want="15
18
20
20
24
24
$(checked ${VP}v1 -p $m -L "$WORK/${VP}v1.thy" -n 20 -S <<<"$q")
$(checked ${VP}v2 -p $m -L "$WORK/${VP}v2.thy" -n 20 -S <<<"$q")
$(checked dag -p $m -L "$WORK/dag.thy" -n 20 -S <<<"$q")
26
24
$v1b
17
$v1b
20
24
24
$(checked ${VP}v2 -p $m -L "$WORK/${VP}v2.thy" -n 20 -S <<<"$q")"
  if [ "$out" != "$want" ]; then
    msg="$m: '$out', not '$want'"
    break
  fi
done
VP=
# Storing a version leaves the default view's numbering of features alone:
# were "b" and "c" numbered before "a", the query's norm would round
# otherwise. In the version D, D.x has two <syms> lines, so it has "g"
# twice, as in a process started on D's files.
corpus one $'s\n' $'s:"z"\n' ''
corpus more $'p\nz1\nz2\n' $'p:"a", "b", "c"\nz1:"z"\nz2:"z"\n' ''
sed 's/.*/&:X/' "$WORK/more.seq" > "$WORK/more.thy"
corpus both $'s\np\nz1\nz2\n' $'s:"z"\np:"a", "b", "c"\nz1:"z"\nz2:"z"\n' ''
version M 1 'M.m:"b", "c"'
printf 'D.x\nD.y\nD.k1\nD.k2\nD.k3\nD.k4\n' > "$WORK/D1.seq"
printf '%s\n' 'D.x:"f", "g"' 'D.y:"g", "h"' 'D.x:"g"' 'D.k1:"k"' 'D.k2:"k"' 'D.k3:"k"' 'D.k4:"k"' > "$WORK/D1.syms"
: > "$WORK/D1.deps"
sed 's/.*/&:D/' "$WORK/D1.seq" > "$WORK/D1.thy"
printf 'D 1\n' > "$WORK/d.view"
fresh d D1
q='"a"@1, "b"@8e-17, "c"@8e-17'
for m in "${view_methods[@]}"; do
  [ -n "$msg" ] && break
  out=$(note='with a version and !add' checked one -p $m -n 20 -S <<<"$(store M 1)
!add $WORK/more.syms $WORK/more.deps $WORK/more.seq $WORK/more.thy
$q
$(store D 1)
!view d $WORK/d.view
[view=d]\"g\", \"f\"")
  want="2
5
$(checked both -p $m -n 20 -S <<<"$q")
11
10
$(checked d -p $m -L "$WORK/d.thy" -n 20 -S <<<'"g", "f"')"
  [ "$out" = "$want" ] || msg="$m: '$out', not '$want'"
done
if [ -n "$msg" ]; then fail "views rank as fresh processes" "$msg"; else ok "views rank as fresh processes"; fi

# 46. malformed version and view commands, and unknown views, are errors
# before any change
number=46
msg=
cp "$WORK/A1.thy" "$WORK/bad.thy"
sed -i 's/^A\.a2:A$/A.a2:B/' "$WORK/bad.thy"
# lines of a theorem not in the version: of another theory, and its deps
{ cat "$WORK/A1.thy"; echo 'B.b1:B private'; } > "$WORK/foreign.thy"
echo 'B.b1:A.a1' > "$WORK/foreign.deps"
# A.a1 given to B, then to A
{ echo 'A.a1:B'; cat "$WORK/A1.thy"; } > "$WORK/moved.thy"
printf 'A 9\n' > "$WORK/unstored.view"
printf 'A 1 x\n' > "$WORK/long.view"
# version 1 of B, with A's theorem A.a1
version X 1 'A.a1:"p"'
sed -i 's/:A$/:B/' "$WORK/X1.thy"
printf 'A 1\nB 1\n' > "$WORK/twice.view"
# rejected <cmds> <ok lines> <error>: sets msg unless predict, given the
# commands, prints <ok lines> lines and fails with the error on stderr
rejected() {
  local out status
  out=$(run dag -p cosine -L "$WORK/dag.thy" <<<"$1")
  status=$?
  if [ $status -eq 0 ] || [ "$(grep -c . <<<"$out")" -ne "$2" ] ||
     ! grep -q -- "$3" "$WORK/stderr"; then
    msg="'$1': exit status $status, output '$out', stderr '$(head -1 "$WORK/stderr")'"
    return 1
  fi
}
A1="$WORK/A1.syms $WORK/A1.deps $WORK/A1.seq"
rejected "!version A 1 $A1" 0 'Expected four files' &&
  rejected '!version A' 0 'Expected `!version' &&
  rejected "!view v1" 0 'Expected `!view' &&
  rejected "!view v1 $WORK/v1.view x" 0 'Expected `!view' &&
  rejected '!drop' 0 'Expected `!drop' &&
  rejected "!drop v1 v2" 0 'Expected `!drop' &&
  rejected "!version A 1 $WORK/missing.syms $WORK/A1.deps $WORK/A1.seq $WORK/A1.thy" 0 'Cannot read' &&
  rejected "$(store A 1)
!view v1 $WORK/missing.view" 1 'Cannot read' &&
  rejected "$(store A 1)
$(store A 1)" 1 'already stored' &&
  rejected "!version A 1 $A1 $WORK/bad.thy" 0 'not of theory `A' &&
  rejected "!version A 1 $A1 $WORK/moved.thy" 0 'Theorem `A.a1. is not of theory `A' &&
  rejected "!version A 1 $A1 $WORK/foreign.thy" 0 'Theorem `B.b1. of .*foreign.thy. is not in' &&
  rejected "!version A 1 $WORK/A1.syms $WORK/foreign.deps $WORK/A1.seq $WORK/A1.thy" 0 'Theorem `B.b1. of .*foreign.deps. is not in' &&
  rejected "$(store A 1)
!view v1 $WORK/unstored.view" 1 'No version `9' &&
  rejected "$(store A 1)
!view v1 $WORK/long.view" 1 'Expected `<theory> <id>' &&
  rejected "$(store A 1)
!view v1 $WORK/v1.view" 1 'No version `1. of `B' &&
  rejected "$(store A 1)
!view v1 $WORK/v2.view" 1 'No version' &&
  rejected "!version A 1 $WORK $WORK/A1.deps $WORK/A1.seq $WORK/A1.thy" 0 'Cannot read' &&
  rejected "$(store A 1)
!view v1 $WORK" 1 'Cannot read' &&
  rejected "$(store A 1)
!version B 1 $WORK/X1.syms $WORK/X1.deps $WORK/X1.seq $WORK/X1.thy
!view v1 $WORK/twice.view" 2 'Theorem `A.a1. is in two theories' &&
  rejected '!drop v1' 0 'Unknown view' &&
  rejected '[view=v1]"p"' 0 'Unknown view' &&
  rejected '[view=]"p"' 0 'Bad view'
if [ -n "$msg" ]; then fail "bad versions and views" "$msg"; else ok "bad versions and views"; fi

# 47. on the first 3000 theorems of the synthetic corpus, cut into theories
# S0, ..., S5 of 500, a view ranks as a process started on its theorems,
# scores included: the view leaves S4 out, has S0 at a version whose
# theorems have the features of theorems 1500 places later, stored last.
# Its theorems include ones with more than 20 dependencies, dependencies on
# later theorems and on theorems of S4, and equal features; naive Bayes adds
# a theorem's counts up in the order the process learned them. MePo's
# filter, which scores the theorems again after each batch it accepts, takes
# minutes for 300 predictions on the checked build, so mepo and mesh predict
# 50, and mepo -P answers 3 plain and 2 scoped queries.
number=47
LIMIT=600
msg=
for k in $(seq 0 5); do
  sed -n "$((500 * k + 1)),$((500 * k + 500))p" "$WORK/synth.seq" > "$WORK/S${k}a.seq"
  sed -n "$((500 * k + 1)),$((500 * k + 500))p" "$WORK/synth.syms" > "$WORK/S${k}a.syms"
  sed -n "$((500 * k + 1)),$((500 * k + 500))p" "$WORK/synth.deps" > "$WORK/S${k}a.deps"
  sed "s/\$/:S$k/" "$WORK/S${k}a.seq" > "$WORK/S${k}a.thy"
done
cp "$WORK/S0a.seq" "$WORK/S0b.seq"
cp "$WORK/S0a.deps" "$WORK/S0b.deps"
cp "$WORK/S0a.thy" "$WORK/S0b.thy"
paste -d: "$WORK/S0a.seq" <(sed -n 1501,2000p "$WORK/synth.syms" | cut -d: -f2-) > "$WORK/S0b.syms"
stored=(S1a S2a S3a S4a S5a S0b)
printf 'S%s a\n' 1 2 3 5 > "$WORK/s.view"
echo 'S0 b' >> "$WORK/s.view"
fresh s S1a S2a S3a S5a S0b
scoped=$(head -5 "$WORK/synth.queries" | sed 's/^/[theory=S3 before=200 exclude=S5]/')
{ cat "$WORK/synth.wqueries"; echo "$scoped"; } > "$WORK/s.queries"
{ head -3 "$WORK/synth.wqueries"; head -2 <<<"$scoped"; } > "$WORK/s.few"
for m in "${view_methods[@]}"; do
  n=300
  qs=$WORK/s.queries
  case $m in
    mepo|mesh) n=50 ;;
    "mepo -P") n=50; qs=$WORK/s.few ;;
  esac
  out=$(note='with a view of the synthetic corpus' checked one -p $m -n $n -S <<<"$(for v in "${stored[@]}"; do echo "!version ${v%?} ${v: -1} $WORK/$v.syms $WORK/$v.deps $WORK/$v.seq $WORK/$v.thy"; done)
!view s $WORK/s.view
$(sed 's/^\[/[view=s /; t; s/^/[view=s]/' "$qs")" | tail -n +8)
  want=$(checked s -p $m -L "$WORK/s.thy" -n $n -S < "$qs")
  if [ -z "$want" ] || [ "$out" != "$want" ]; then
    msg="$m: the view's rankings differ from a fresh process's"
    break
  fi
done
if [ -n "$msg" ]; then fail "views of a synthetic corpus" "$msg"; else ok "views of a synthetic corpus"; fi
LIMIT=5

# 48. a feature on two <syms> lines of a theorem counts once, as on one line:
# naive Bayes scores stay finite, the same as the one-line corpus's, and
# nbayes and nbayes-dense agree on them, with and without the feature in the
# query
number=48
corpus twolines $'x\ny\n' $'x:"f", "g"\nx:"g"\ny:"g"\n' ''
corpus oneline $'x\ny\n' $'x:"f", "g"\ny:"g"\n' ''
msg=
declare -A nb
for q in '"g"' '"f"' '"f", "g"'; do
  for m in nbayes nbayes-dense; do
    a=$(checked twolines -p $m -S <<<"$q")
    b=$(checked oneline -p $m -S <<<"$q")
    if ! finite_scores <<<"$a" || [ "$a" != "$b" ]; then
      msg="$m, '$q': two lines give '$a', one line '$b'"
      break 2
    fi
    nb[$m]=$a
  done
  sparse=${nb[nbayes]} dense=${nb[nbayes-dense]}
  if [ "$(ids_of <<<"$sparse")" != "$(ids_of <<<"$dense")" ] ||
     ! near "$(score_of x <<<"$sparse")" "$(score_of x <<<"$dense")" ||
     ! near "$(score_of y <<<"$sparse")" "$(score_of y <<<"$dense")"; then
    msg="'$q': nbayes gives '$sparse', nbayes-dense '$dense'"
    break
  fi
done
if [ -n "$msg" ]; then fail "feature repeated across lines" "$msg"; else ok "feature repeated across lines"; fi

# 49. a process started on a snapshot of the first 1000 theorems of the
# synthetic corpus and a theory T, or on those theorems alone and `!add`ing
# T, ranks after each `!replace` of its last theory (to fewer theorems, to
# theorems with other features, to none) and `!add` as a
# process started on the snapshot and the theorems it then holds after it,
# scores included, for every method that unlearns (mash is mesh's part);
# so does one whose <syms> gives T's features before those of a theory
# before it. T depends on the theorems before and
# on later ones; MePo predicts 50 on the checked build, as in 47.
number=49
LIMIT=600
msg=
for f in seq syms deps; do cat "$WORK"/S[01]a.$f > "$WORK/b.$f"; done
checked b -o "$WORK/b.snap" </dev/null
# tail <name> <theory> <first line> <last line> [<syms from line>]: theorems
# of S5a, or of S4a for <first line> 0, as <theory>; their features those of
# the synthetic corpus from the given line on, if given
tail_theory() {
  local name=$1 thy=$2 first=$3 last=$4 from=${5:-} src=S5a
  [ "$first" = 0 ] && { src=S4a; first=1; }
  sed -n "${first},${last}p" "$WORK/$src.seq" > "$WORK/$name.seq"
  sed -n "${first},${last}p" "$WORK/$src.deps" > "$WORK/$name.deps"
  if [ -n "$from" ]; then
    paste -d: "$WORK/$name.seq" <(sed -n "${from},\$p" "$WORK/synth.syms" |
      head -n "$(wc -l < "$WORK/$name.seq")" | cut -d: -f2-) > "$WORK/$name.syms"
  else
    sed -n "${first},${last}p" "$WORK/$src.syms" > "$WORK/$name.syms"
  fi
  sed "s/\$/:$thy/" "$WORK/$name.seq" > "$WORK/$name.thy"
}
tail_theory T1 T 0 250
tail_theory T2 T 1 50
tail_theory T3 T 51 150 501
tail_theory U U 151 200
# U, added before T, depends on none of it, as in a process started on both
sed -i 's/:.*/:/' "$WORK/U.deps"
# features no theorem before has, numbered otherwise by T's two versions
sed -i '1,5s/$/, "n0", "n1", "n2", "n3", "n4", "n5"/' "$WORK/T1.syms"
sed -i '1,5s/$/, "n5", "n4", "n3", "n2", "n1", "n0"/' "$WORK/T2.syms"
for f in seq syms deps thy; do : > "$WORK/E.$f"; done
fresh UT1 U T1
fresh UT2 U T2
fresh UT3 U T3
{ cat "$WORK/T3.syms"; cat "$WORK/U.syms"; } > "$WORK/UT3.syms"
files() { echo "$WORK/$1.syms $WORK/$1.deps $WORK/$1.seq $WORK/$1.thy"; }
qs=$(head -3 "$WORK/synth.wqueries"; echo '"n0", "n2", "n4", "f1"')
# and with the features of a theorem of U and of one of T2
uqs=$(echo "$qs"; head -1 "$WORK/U.syms" | cut -d: -f2-; head -1 "$WORK/T2.syms" | cut -d: -f2-)
commands="!replace T $(files T2)
$qs
!replace T $(files T3)
$qs
!replace T $(files E)
$qs
!add $(files U)
$qs
!replace T $(files T1)
$qs
!replace T $(files T2)
$qs"
for m in knn nbayes nbayes-dense cosine cosine-dense mepo mesh; do
  n=300
  case $m in mepo|mesh) n=50 ;; esac
  # from_snap <tail>: the answers of a process started on the snapshot and
  # <tail>
  from_snap() { checked $1 -i "$WORK/b.snap" -L "$WORK/$1.thy" -p $m -n $n -S <<<"$qs"; }
  want="$(from_snap T1)
1050
$(from_snap T2)
1100
$(from_snap T3)
1000
$(from_snap E)
1050
$(from_snap U)
1300
$(from_snap UT1)
1100
$(from_snap UT2)"
  out=$(note='with !replace of the last theory' checked T1 -i "$WORK/b.snap" -L "$WORK/T1.thy" -p $m -n $n -S <<<"$qs
$commands")
  if [ "$out" != "$want" ]; then
    msg="$m, started on the snapshot and T: '$out', not '$want'"
    break
  fi
  want="$(from_snap E)
1250
$(from_snap T1)
1050
$(from_snap T2)"
  out=$(note='with !add and !replace of the last theory' checked b -p $m -n $n -S <<<"$qs
!add $(files T1)
$qs
!replace T $(files T2)
$qs")
  if [ "$out" != "$want" ]; then
    msg="$m, started on the first theorems: '$out', not '$want'"
    break
  fi
  want="1100
$(checked UT2 -i "$WORK/b.snap" -L "$WORK/UT2.thy" -p $m -n $n -S <<<"$uqs")"
  out=$(note='with !replace of a theory given features first' checked UT3 -i "$WORK/b.snap" -L "$WORK/UT3.thy" -p $m -n $n -S <<<"!replace T $(files T2)
$uqs")
  if [ "$out" != "$want" ]; then
    msg="$m, its <syms> giving T's features first: '$out', not '$want'"
    break
  fi
done
if [ -n "$msg" ]; then fail "!replace of the last theory" "$msg"; else ok "!replace of the last theory"; fi
LIMIT=5

[ $failures -eq 0 ]
