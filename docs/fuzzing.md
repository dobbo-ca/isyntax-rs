# Fuzzing

`tests/fuzz/` has a target for the header and tile decode (`open`,
`decode`, the latter reusing images across inputs and comparing warm with cold decodes; the warm images share one `Decoder` across inputs, and about a quarter of seeds use `max_threads` 1), for the
tables under the header (`tables`: base64, cluster table, XML), for the Hulsken decoder with a reused
scratch (`hulsken`) and for every SIMD kernel on every tier of the CPU against its scalar
reference (`kernels`); they reach internal code through `isyntax::fuzz_api`, which is hidden and not
API. Every target also runs natively on aarch64 (ubuntu-24.04-arm), with ASan.

CI: `.github/workflows/fuzz.yml` (nightly, and on dispatch with `max_total_time`); the crate is built with `cargo fuzz ... --fuzz-dir tests/fuzz`.
