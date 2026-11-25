# Status and tests

## Status

| Part | State |
|---|---|
| Header: XML, base64, cluster table, v1 block header table and seektable, tile index | ported, tested |
| Hulsken v1 and v2 codeblock decoder | ported, tested |
| Inverse LeGall 5/3, YCoCg, RGBA to YCbCr (scalar) | ported, tested |
| Tiles and regions (RGBA, RGBA64, YCbCr, samples), caches, memory budget, SDK truncation, wide grid | ported, tested |
| SIMD kernels (AVX2, NEON; AVX-512 on x86_64, not in Go), `ISYNTAX_SIMD`, cross-tier tests, arm64 CI | ported, tested, measured (kernels bench, see [safety.md](safety.md)) |
| Parity with Go (`tests/parity/`, CI jobs `parity`, `parity-arm64`), fuzz (`tests/fuzz/`, `fuzz.yml`), bench (`bench.yml`) | ported, running |

## Tests

`cargo test` runs on synthetic inputs only, as in Go: a test-only Hulsken encoder and synthetic
iSyntax2 and v1 files (`src/fixture.rs`) drive the unit tests of every module, the decoder
(`src/decoder_tests.rs`: tiles, regions, YCbCr, samples, RGBA64, truncation, the missing-tile
rules, the wide grid, the cache, the memory limit, cancellation, concurrency, files with random
edits). No slide data is in git. CI runs everything in `.github/ci/Dockerfile`.
