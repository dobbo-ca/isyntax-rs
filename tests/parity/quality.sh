#!/bin/sh
# The quality gate over the parity corpus (byte-exact):
#   tests/parity/quality.sh <inputs> <go-out> <report>
# <inputs> holds the files (the bench fixtures and the largest non-mutant inputs of `oracle -gen`),
# <go-out> the P6 files that `oracle -ppm <inputs> <go-out>` wrote (stage `go-quality` of
# .github/ci/Dockerfile). This script decodes <inputs> with examples/ppm.rs, runs `diff -r` on
# the two trees into <report>/diff.txt and sets <report>/status: 0 only if every tile is byte-identical,
# Rust wrote as many tiles as Go, and at least FLOOR tiles were compared.
# Needs $PPM (default target/release/examples/ppm).
# FLOOR (default 500) keeps the gate from passing on an empty tree. The six bench fixtures are
# 3000 x 2000 with three levels; their grid (fixture.Grid) is 12 x 8 tiles at level 0, 6 x 4 at
# level 1, 3 x 2 at level 2, plus the derived level of 3 x 2 128 x 128 tiles: 96 + 24 + 6 + 6 = 132
# tiles each, 792 for the six. The floor counts less than that, and nothing from the other files.
set -u
in=$1 go=$2 report=$3
floor=${FLOOR:-500}
ppm=${PPM:-target/release/examples/ppm}
rs=$(mktemp -d)
trap 'rm -rf "$rs"' EXIT
mkdir -p "$report"
st=0
"$ppm" "$in" "$rs" >"$report/rs.log" 2>"$report/rs.err" || { st=1; echo "ppm failed" >&2; }
diff -r "$go" "$rs" >"$report/diff.txt" || st=1
compared=$(ls "$go" | wc -l)
[ "$compared" -ge "$floor" ] || st=1
# Rust must have written as many tiles as Go: a backstop, as `diff -r` should already fail on a tile in one set only.
[ "$(ls "$go" | wc -l)" -eq "$(ls "$rs" | wc -l)" ] || st=1
{
  echo "- tiles compared: $compared"
  echo "- inputs: $(ls "$in" | wc -l)"
  echo "- tiles written by Go: $(ls "$go" | wc -l), by Rust: $(ls "$rs" | wc -l)"
  echo "- tiles that failed to decode (Rust stderr lines): $(grep -c . "$report/rs.err")"
  echo "- floor: $floor tiles compared, got $compared"
  if [ "$st" -eq 0 ]; then echo "- result: PASS"; else echo "- result: FAIL"; fi
} >>"$report/summary.md"
echo "$st" >"$report/status"
cat "$report/summary.md"
