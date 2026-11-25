# Parity with Go

Status: parity with Go (`tests/parity/`, CI jobs `parity`, `parity-arm64`) is ported and running; the part table is in [status.md](status.md).

Everything lives in `tests/parity/`. `tests/parity/go/upstream/` is the non-test source of dobbo-ca/isyntax-go (with `go.mod` and
`LICENSE`; the repo has no NOTICE). `tests/parity/go/oracle/` is a Go program that writes the inputs and decodes
them; `examples/oracle.rs` decodes the same inputs with this crate; the CI job `parity` (and
`parity-arm64`, the Rust side built and run natively on aarch64) runs both and does `diff -r` on
the two output trees. A floor on the number of outputs and of `ok` lines keeps the diff from passing
on an empty tree.

The oracle's module path is `github.com/dobbo-ca/isyntax-go/parity-oracle`, under the
upstream module's path, so that it may import upstream's `internal/fixture` (the synthetic file
builder and Hulsken encoder of the Go tests) and `internal/container`, which Go allows for any path
below the parent of `internal`. `upstream/` is not edited (beyond comments), except for one added file,
`internal/container/oracle_export.go`, which exports `decodeBase64` and the cluster-table parser for
the `.b64` and `.ct` inputs.

## Operations and output

`oracle <inputs> <outputs>` writes one file per input, `<name>.out`, with a line per operation:
`<op> ok <digest>` (FNV-1a 64 of the output bytes, and their length) or `<op> err <class>`.
`open` comes first; if it fails the file has that line only. Otherwise:

- `info`: levels, tile size, bits stored, MPP (as float bits), label, macro and ICC (digest and
  whether nil), barcode, model, serial, acquisition time, calibration fields, rack and slot,
  interface version, and per level its size, tile grid, downsample, `SourceBounds` and `Envelopes`.
- tiles: per level, five tiles (the corner, the middle, the last, one past the grid and one at
  x = -1) as RGBA, and the middle one as RGBA64, YCbCr 4:2:0 and 4:4:4; then again after the
  regions on the same `Image` (`warm.`, a warm cache).
- `all.<level>.<x>.<y>`: for an image of at most 64 level-0 tiles, every tile of every level as
  RGBA, a digest line each, so that no codeblock of a small image goes undecoded.
- regions: per level, two rectangles (the top left, up to 400 x 300, and one that crosses the
  bottom right edge) as RGBA, and for the first also RGBA64, samples, signed samples and YCbCr 4:2:0;
  samples and signed samples of `TestSignedSamples`' two more rectangles (`-20,-10,10,10` and
  `250,250,263,520`, region index 2 and 3);
  bad arguments (a level out of range, a destination of the wrong size or stride, and
  `TestRGBA64BadBounds`' RGBA64 region and 128 x 128 tile).
- the same regions on the `SDKTruncated` view (`sdk.`, or its error class) and on `WideGrid`
  (`wide.`, with its level geometry).
- `cancel`: a tile decode with a cancelled context (a cancel flag set) on a fresh image.
- `size-1`, `size+1`: the open with a reader size one short and one over the real length, `ok` or
  the class (Go's `readXML` tolerance for a short header read, and the index checks against the
  size). The open reads the header only, so the codeblock-read deviation below does not arise.

`.b64` and `.ct` inputs have one line: `base64 ok <digest>` of the decoded bytes (or `err <class>`)
from `decodeBase64`, and `ct ok <codeblocks>` (or `err <class>`) from the cluster-table parser with
`FuzzParseClusterTable`'s two fixed templates.

Partial output: a tile decode writes into the caller's destination, which is prefilled with 0xAA, and
after an `err` line of a tile operation (RGBA, RGBA64, YCbCr 4:2:0 and 4:4:4) both oracles add `<op> partial <digest>` of the destination
as the decoder left it, so what was written before the error is compared. Go decodes full tiles
straight into the destination, as does the Rust tile decode. A region decode on a level of
one tile runs one worker, so it is compared partially too, and both sides paint the tile that did
not finish white. With two or more tiles and several workers which tiles finished depends on
scheduling, so the default pass compares only the `err` class there. If any region of the default
pass failed, a second pass (`w1.`) runs all the regions again on a fresh `Image` with one worker
(`runtime.GOMAXPROCS(1)` in Go, restored after, as `internal/parallel.For` runs `GOMAXPROCS`
workers; `Limits { max_threads: 1, .. }` in Rust) and every region line has a partial digest, so
the tiles finished before the error and the white ones are compared on multi-tile levels too. The
same pass also runs the regions of the `SDKTruncated` and `WideGrid` views of that one-worker image
(`w1.sdk.`, `w1.wide.`, in the same order and format in both oracles).

The `parity` job also runs the Rust oracle a second time on the whole input set with one thread
(`ISYNTAX_ORACLE_THREADS=1`, which sets `Limits::max_threads`) and diffs it against the same Go
tree: the lines of the default pass do not depend on scheduling, so the serial path is compared with
Go on every successful multi-tile decode, as the output is the same for every thread count.

Go runs with `SetMemoryLimit(0)` and a soft 1 GiB heap limit, Rust with `max_alloc = u64::MAX`;
inputs are processed one at a time in both. The inputs are bounded by `-gen` itself: a mutant that
parses into more than 2^28 pixels or 2^16 level-0 tiles is dropped, so neither side allocates a
huge grid (Rust would stop at `Limits::max_pixels` where Go has no cap). The memory limit
(`ErrMemoryLimit`, `errTooBig`) is process-wide in Go and per image here, so it is covered by the
unit tests, not by parity.

## Inputs (`oracle -gen`)

- The synthetic files of the Go tests: `randSpec` (compressors 16 and 19, and the v1 variant),
  `constSpec` at several grey levels and 8 and 9 bits, `missingAt` (tiles missing at scale 0 and at
  the top scale), the container tests' `spec` and v1 spec, the three seeds of `FuzzOpenDecode`, the
  calibration-field cases of `TestCalibrationFields` and some metadata, the edits of
  `TestUnsupported` (compressor 17, 16 bits in iSyntax2, v1 with compressor 19, unknown plane names),
  and odd geometries (1 x 1, 5 x 3, 255 x 257, a wide image with four levels, a tall one).
- `wide-*`: `wideSpec` of the decoder tests (LL samples outside the clamp range) as compressor 16 at
  8 bits, 19 at 9 and 8 bits, v1 at 16 bits, and with colour planes (`wide-rgb-*`, the tests'
  `planes(...)` variants).
- `seed-*`: the seeds of the other Go fuzz tests: `FuzzParse`'s `hello\x04`, the empty file and the
  halves of the `FuzzOpenDecode` files (`seed-open-*`; the whole files are `fuzz-seed*`),
  `FuzzDecode`'s PCG blocks with `CounterBits` 4 (`seed-hulsken-*.blk`), `FuzzDecodeBase64`'s
  strings (`seed-b64-*.b64`) and `FuzzParseClusterTable`'s tables (`seed-ct-*.ct`).
- Colour planes R, G, B in iSyntax2 8 and 9 bits and in v1 (the `planes` functions of the decode),
  and `Quality=0..3` in the header, so the SDK truncation presets (Q2 at 8 and 9 bits,
  Q1 RGB) and their `Unsupported` cases (v1 16 bits among them) are decoded. These are the fixtures with a header edit that
  keeps the file length (`patched` in `gen.go`).
- `TestRoundTrip`'s random codeblocks (`randomBlock`: bits used 1 to 15, zero share 0 to 1) as the
  coefficients of whole files, for both Hulsken versions. Whole files always use the fixture's own
  encoder parameters (zero symbol 0, 7 or 14, counter bits 3 to 5, 128 x 128 blocks), so the
  blocks themselves are also inputs: `hulsken-*.blk`, `[version-1, planes/3, size index][block]`
  as in the hulsken fuzz target, decoded alone with `hulsken.Decode` and `fuzz_api::Hulsken`. They
  are `TestRoundTrip`'s 480 trials (the test's PRNG and draws: zero symbols 0, 0x7f, 0xff and 1,
  counter bits 1 to 8, block sides 8, 16 and 128, 1 and 3 planes) and the two of `TestAllPlanes`;
  the line is `hulsken ok <digest>` of the coefficients, or `hulsken err codeblock`. Each block also
  has three mutants (`-m0..2`: a truncation, a bit flip anywhere, a bit flip in the first 12 bytes).
  Both oracles decode all `.blk` inputs, in sorted order, through one shared scratch (`blkScratch`
  in Go, one `Hulsken` in Rust), so blocks of different parameters reuse it.
- `xml-*`: corners of `encoding/xml` that a bit flip cannot reach, as same-length edits: a name
  with a non-letter rune (U+00D7), the directives `<!-x>`, `<!'>`, `<!<>` and one with a nested
  comment, `<?xml?>` parameters that need a second look (`version=1 version="2.0"`,
  `encoding=x encoding="UTF-16"`), and a 5000-deep subtree no struct field takes.
- Mutants of ten small files (every format, bit depth, the missing-tile rule, labels, calibration
  tags, one to three levels): truncations (around the end of the XML header and at random), single
  bit flips (half in the header), digit edits in the XML header (dimensions, counts, offsets),
  edits of the base64 of the cluster and block header tables, and flips and length-field extremes
  in the first bytes of codeblocks (the Hulsken header). A fixed PRNG seeded by the file name makes
  them the same on every run. The counts are printed in the CI log (`go-run`: files, mutants and
  the mutants dropped for size).

`parity` runs every input on the x86_64 SIMD tier and the subset (every file that is not a mutant,
and every 16th mutant) on the scalar tier; `parity-arm64` runs the subset natively on aarch64 on
NEON and on scalar (both jobs floor the number of `ok` lines of the Go output).

## Go tests and what happens to them

Go has 75 `Test` functions, 5 `Fuzz` and 9 `Benchmark` in `internal/`. The ported ones are unit
tests in `src/` (96 `#[test]` in all, with the Rust-only tests); what [go-differences.md](go-differences.md) lists as not
ported needs slide data, the pre-refactor reference implementations, `legacyMode` or Go's allocation
counters. The tests that build random inputs (`randSpec`, `constSpec`, `missingAt`, `randomBlock`,
the container specs, `wideSpec`, the fuzz seeds) are generated here by `-gen` so that Go provides the expected
output. The five fuzz tests are the fuzz targets of `tests/fuzz/` (`open`, `tables`, `hulsken`, `decode`),
seeded by `oracle -genseeds` with their `f.Add` values (the Go repo has no `testdata/fuzz`).

## Deliberate deviations

- A codeblock of size 0 whose offset is the end of the file: Go's `Index` accepts it, and Go then
  calls `ReadAt` with an empty buffer, which fails with `io.EOF` for a `bytes.Reader` (class `io`)
  but returns nil for an `*os.File`, so Go itself depends on the reader. Rust never reads an empty
  buffer and decodes the block as zeros (`ReadAt` has no such split), so no input of this kind is
  generated. `src/types.rs` tests it.
- The XML header limit is tested as in Go, after each 1 MiB chunk without the terminator, so a
  header shorter than `max_header` plus one chunk opens (`src/container.rs` tests the boundary at
  one chunk; Go's 256 MiB + 1 is the same rule).
- `YCbCr::new` takes `&Limits`, and sizes 4:2:0 chroma as the decoder indexes it, which is one
  sample more than `image.NewYCbCr` when `max_x` or `max_y` is 0 (Go's own decode panics there).

## Error mapping

| Go | class | Rust |
|---|---|---|
| `context.Canceled` | `cancelled` | `Error::Cancelled` |
| `ErrUnsupported` (wrapped) | `unsupported` | `Error::Unsupported` |
| `ErrMemoryLimit` | `memory-limit` | `Error::MemoryLimit` |
| `errTooBig` (needs more than the whole limit) | `alloc-limit` | `Error::AllocLimit` |
| `isyntax: XML header: ...` | `xml` | `Error::Xml` |
| `isyntax: XML header over 256 MiB` | `header-limit` | `Error::HeaderLimit` |
| every other `fmt.Errorf` of `internal/container` | `header` | `Error::Header` |
| `isyntax: codeblock at N: hulsken: ...` | `codeblock` | `Error::Codeblock` |
| read errors of the `io.ReaderAt` (`isyntax: reading ...`) | `io` | `Error::Io` |
| bad level, destination or stride | `invalid-argument` | `Error::InvalidArgument` |
| worker panic (`isyntax: internal error: panic`) | `internal` | `Error::Internal` |
| (no Go counterpart) `Limits::max_pixels` | `pixel-limit` | `Error::PixelLimit` |
| (no Go counterpart) allocator refusal | `out-of-memory` | `Error::OutOfMemory` |

## Real-file quality gate

A gate on real files would use the OpenSlide test data, which has no iSyntax file, so
this crate uses the parity corpus instead (`quality.yml`, workflow `quality`). The files are the six bench
fixtures (`oracle -genbench`, 3000 x 2000) and the 24 largest inputs of `oracle -gen` that are not
mutants (no `~` in the name) and not `.blk`, `.b64` or `.ct`. `oracle -ppm <in> <out>` and
`examples/ppm.rs` open each file, in sorted order, and for every level and every tile of that
level's grid decode the tile as RGBA and write a binary P6 (alpha dropped) named
`<file name with . as _>.l<level>.<x>.<y>.ppm`. Go opens without memory limit; Rust uses
`max_alloc` and `max_pixels` of `u64::MAX`, as `examples/oracle.rs` does, and `examples/ppm.rs` opens
every file with one shared `Decoder` (`.with_decoder(&decoder)`), so the gate also checks that the
buffers of one file serve the next, byte for byte against Go. A file that does not open
and a tile that does not decode write nothing and are named on stderr; the two sides must agree on
which tiles exist (`diff -r` fails on a tile in one set only), and the error classes are what
the parity job above compares.

`diff -r` compares the two trees byte for byte. `tests/parity/quality.sh` also floors the
tiles compared at 500: the six fixtures alone have 132 tiles each (a 12 x 8 grid at level 0, 6 x 4,
3 x 2, and the derived level's 3 x 2), 792 in all, so the gate cannot pass on an empty tree.
