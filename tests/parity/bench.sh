#!/bin/sh
# Decode time of level 0 of the benchmark fixtures, Go and Rust on the same files, as markdown
# tables: tests/parity/bench.sh [N [ROUNDS]]. Needs `oracle` (Go) and `bench` (Rust) on PATH (or
# $ORACLE and $BENCH: other commands for the first two tables); run from the repo root.
#   default parallelism: Go at its default GOMAXPROCS, Rust at its default max_threads;
#   single thread:       Go with GOMAXPROCS=1, Rust with max_threads = 1.
# Both carry the pixel-hash check and the 1.10 gate: the script exits non-zero, after printing every
# table, if a pixel hash differs or a Rust / Go ratio is above 1.10 (a FAIL line says which).
# When `oracle-v4` (Go built with GOAMD64=v4) and `bench-v4` (Rust built with -C target-cpu=x86-64-v4
# and the false-deps features, docs/performance.md) are on PATH and $ORACLE and $BENCH are not set,
# two more tables follow, "v4 build, default threads" and "v4 build, single thread": the same
# comparison of the v4 builds. They are not gated: a pixel hash that differs there is a warning.
# Each oracle and bench line is `name, bytes, ns, pixel hash, minor page faults per decode`. The
# tables show the fastest of ROUNDS rounds, the faults of that round, and `spread`, the slowest over
# the fastest Rust time of the last batch of rounds. After ROUNDS rounds, a fixture whose Go or Rust
# spread is above SPREAD (default 1.25) is run alone for RETRY_ROUNDS (default ROUNDS) more rounds,
# up to twice; the table takes the fastest over all the rounds. The retries are listed on stderr.
set -eu
export LC_ALL=C
n=${1:-10}
rounds=${2:-7}
spread=${SPREAD:-1.25}
retry_rounds=${RETRY_ROUNDS:-$rounds}
oracle=${ORACLE:-oracle} bench=${BENCH:-bench}
bad=0
tab=$(printf '\t')
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
grep -m1 'model name' /proc/cpuinfo >&2 || true
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|bmi2|avx512(f|bw|vl))$' | tr '\n' ' ' >&2 || true
echo >&2
d=$t/fixtures
"$oracle" -genbench "$d"
ls -l "$d" >&2
# The single-threaded runs are pinned to one CPU where taskset exists: no migrations, no noise.
cpu=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //; s/[-,].*//') || cpu=
pin1=
if [ -n "$cpu" ] && taskset -c "$cpu" true 2>/dev/null; then pin1="taskset -c $cpu"; fi
echo "pin: ${pin1:-none}" >&2
# Fastest of the rounds per file and side: the machine is shared, and one run can be slow.
best() {
  awk -F'\t' '!($1 in m) || $3 < m[$1] { m[$1] = $3; b[$1] = $2; h[$1] = $4; f[$1] = $5 } END { for (k in m) printf "%s\t%s\t%s\t%s\t%s\n", k, b[k], m[k], h[k], f[k] }' "$@" | sort
}
# spreads SIDE BATCH: per file, the slowest over the fastest time of that side in the rounds of BATCH.
spreads() {
  awk -F'\t' '!($1 in mn) || $3 < mn[$1] { mn[$1] = $3 } $3 > mx[$1] { mx[$1] = $3 }
    END { for (k in mn) printf "%s\t%.3f\n", k, (mn[k] > 0 ? mx[k] / mn[k] : 1) }' "$w/$1.$2".*
}
# run_rounds DIR BATCH COUNT: COUNT rounds on the files of DIR, Go and Rust alternating, into
# $w/go.BATCH.I and $w/rs.BATCH.I. $genv, $rthreads and $pin are those of `table`.
run_rounds() {
  i=0
  while [ "$i" -lt "$3" ]; do
    i=$((i + 1))
    $pin env $genv "$oracle" -bench "$n" "$1" >"$w/go.$2.$i" </dev/null
    $pin "$bench" "$1" "$n" $rthreads >"$w/rs.$2.$i" </dev/null
  done
}
# table TITLE LABEL GO_ENV RUST_THREADS PIN GATE: Go with GO_ENV in its environment (var=value, or
# empty) against Rust with `max_threads` RUST_THREADS (empty: the default), both under PIN (a
# command prefix, or empty). Go and Rust alternate round by round. The pixels must be the same.
table() {
  title=$1 label=$2 genv=$3 rthreads=$4 pin=$5 gate=$6
  w=$t/$label
  mkdir -p "$w"
  run_rounds "$d" 0 "$rounds"
  b=0
  while [ "$b" -lt 2 ]; do
    spreads go "$b" | sort >"$w/sg"
    spreads rs "$b" | sort >"$w/sr"
    join -t "$tab" "$w/sg" "$w/sr" | awk -F'\t' -v s="$spread" '$2 > s || $3 > s' >"$w/retry"
    [ -s "$w/retry" ] || break
    b=$((b + 1))
    k=0
    while IFS="$tab" read -r c sg sr; do
      k=$((k + 1))
      echo "$label: retry $b of $c: spread Go $sg, Rust $sr (limit $spread)" >&2
      rm -rf "$w/one"
      mkdir "$w/one"
      ln "$d/$c" "$w/one/$c" 2>/dev/null || cp "$d/$c" "$w/one/$c"
      run_rounds "$w/one" "$b.$k" "$retry_rounds"
    done <"$w/retry"
  done
  best "$w"/go.* >"$w/bgo"
  best "$w"/rs.* >"$w/brs"
  # The spread of a file is that of the last batch it ran in.
  k=0
  while [ "$k" -le "$b" ]; do
    spreads rs "$k"
    k=$((k + 1))
  done | awk -F'\t' '{ v[$1] = $2 } END { for (c in v) printf "%s\t%s\n", c, v[c] }' | sort >"$w/spread"
  # name, Go bytes, ns, hash, faults, Rust bytes, ns, hash, faults, Rust spread.
  join -t "$tab" "$w/bgo" "$w/brs" | join -t "$tab" - "$w/spread" >"$w/j"
  awk -F'\t' -v gate="$gate" '$4 != $8 { bad = 1; print "pixel hash differs for " $1 ": Go " $4 ", Rust " $8 > "/dev/stderr" }
    END { exit (bad && gate) ? 1 : 0 }' "$w/j" || bad=1
  # The fastest rounds are not the only ones that count: every round of both sides, retries
  # included, must give one hash per file.
  cat "$w"/go.* "$w"/rs.* | awk -F'\t' -v gate="$gate" '!($1 in h) { h[$1] = $4 } $4 != h[$1] { bad = 1; print "pixel hash differs between rounds for " $1 > "/dev/stderr" }
    END { exit (bad && gate) ? 1 : 0 }' || bad=1
  echo
  echo "$title"
  echo
  echo "| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |"
  echo "|---|---|---|---|---|---|---|---|"
  awk -F'\t' '{ printf "| %s | %.1f | %.2f | %.2f | %.2f | %s | %s | %.2f |\n", $1, $2 / 1e6, $3 / 1e6, $7 / 1e6, $7 / $3, $5, $9, $10 }' "$w/j"
  if [ "$gate" = 1 ]; then
    awk -F'\t' '$7 / $3 > 1.10 { printf "\nFAIL: %s is %.2f times Go (limit 1.10)\n", $1, $7 / $3; bad = 1 } END { exit bad }' "$w/j" || bad=1
  fi
}
table "Default parallelism: Go (GOMAXPROCS = all CPUs) against Rust (max_threads = available_parallelism)" default "" "" "" 1
table "Single-threaded: Go with GOMAXPROCS=1 against Rust with max_threads = 1" 1t "GOMAXPROCS=1" "1" "$pin1" 1
if [ -z "${ORACLE:-}${BENCH:-}" ] && command -v oracle-v4 >/dev/null 2>&1 && command -v bench-v4 >/dev/null 2>&1 && grep -q avx512bw /proc/cpuinfo; then
  oracle=oracle-v4 bench=bench-v4
  table "v4 build, default parallelism, not gated: Go GOAMD64=v4 against Rust x86-64-v4" v4 "" "" "" 0
  table "v4 build, single-threaded, not gated: Go GOAMD64=v4 with GOMAXPROCS=1 against Rust x86-64-v4 with max_threads = 1" v4-1t "GOMAXPROCS=1" "1" "$pin1" 0
fi
exit "$bad"
