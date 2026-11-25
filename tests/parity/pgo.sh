#!/bin/sh
# Times the bench example built three ways against Go: tests/parity/pgo.sh [N [ROUNDS]]. Needs
# `oracle` (Go) and `bench` (the default build), `bench-pgo` (cargo-pgo, trained on the fixtures of
# `oracle -genbench`) and, when BOLT was available, `bench-bolt` (PGO then BOLT) on PATH; run from
# the repo root (stage `pgo` of .github/ci/Dockerfile, workflow pgo.yml). Each build runs
# tests/parity/bench.sh (its Rust binary is $BENCH); the table gives the Rust ms of each fixture,
# and the sum of them, in the two tables of bench.sh, and the change against the default build.
set -eu
export LC_ALL=C
n=${1:-10}
rounds=${2:-7}
t=$(mktemp -d)
have=
labels=
for v in bench:default bench-pgo:PGO bench-bolt:"PGO + BOLT"; do
  b=${v%%:*}
  if ! command -v "$b" >/dev/null 2>&1; then
    echo >&2 "no $b: skipped"
    continue
  fi
  # bench.sh exits 1 when a fixture is above its 1.10 gate; the tables are still complete.
  BENCH=$b sh tests/parity/bench.sh "$n" "$rounds" >"$t/$b.md" || echo >&2 "$b: bench.sh exited $? (a fixture above the gate, or a failure above)"
  # An empty file (bench.sh died before printing) would shift the labels of the later columns.
  [ -s "$t/$b.md" ] || { echo >&2 "$b: no output: skipped"; continue; }
  have="$have $t/$b.md"
  labels="$labels,${v#*:}"
done
[ -n "$have" ] || { echo >&2 "no bench binary on PATH"; exit 1; }
# A table starts at its `| input |` header line; a fixture row is `| name | MB out | Go ms | Rust
# ms | ...`, Rust ms in column 5.
# shellcheck disable=SC2086
awk -F'|' -v labels="${labels#,}" '
FNR == 1 { nf++; p = 0 }
/^[|] input [|]/ { p++; if (p > np) np = p; next }
/^[|] [^ |]+ [|] [0-9.]+ [|]/ {
  name = $2; gsub(/^ +| +$/, "", name)
  v[nf, p, name] = $5 + 0
  v[nf, p, "all"] += $5
  if (!((p, name) in seen)) { seen[p, name] = 1; names[p, ++cnt[p]] = name }
}
END {
  split(labels, lab, ",")
  split("default threads,single thread", pass, ",")
  for (p = 1; p <= np; p++) {
    print "#### Rust ms, " pass[p] " (lower is better; the change is against the default build)"
    print ""
    s = "| fixture |"; r = "|---|"
    for (f = 1; f <= nf; f++) { s = s " " lab[f] " |"; r = r "---|"; if (f > 1) { s = s " change |"; r = r "---|" } }
    print s; print r
    names[p, ++cnt[p]] = "all"
    for (i = 1; i <= cnt[p]; i++) {
      k = names[p, i]
      s = "| " k " |"
      for (f = 1; f <= nf; f++) {
        s = s sprintf(" %.2f |", v[f, p, k])
        if (f > 1 && v[1, p, k] > 0) s = s sprintf(" %+.1f%% |", (v[f, p, k] / v[1, p, k] - 1) * 100)
      }
      print s
    }
    print ""
  }
}' $have
for f in $have; do
  echo "<details><summary>$(basename "$f" .md): the full bench.sh tables</summary>"
  echo
  cat "$f"
  echo
  echo "</details>"
done
rm -rf "$t"
