# Performance

How to build and call this crate for speed. The measured numbers are in [benchmarks.md](benchmarks.md).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. For a worker that decodes tiles:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the kernels and the loops that call them inline across
  modules and across the crate boundary.
- `panic = "abort"` removes the unwinding tables. It also changes one thing here: the crate catches a
  panic on a worker thread and returns `Error::Internal` ([parity.md](parity.md)), and with
  `abort` the process stops at the panic instead. The crate is written not to panic on any input,
  so this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 (F, BW, CD, DQ, VL) everywhere, not only in
  the kernels. **Use it only when every machine that runs the binary has AVX-512.** On a CPU without
  it the program dies with `SIGILL` (illegal instruction) at the first such instruction, with no
  message. GitHub-hosted x86_64 runners do not all have AVX-512; the
  default build needs no flag and runs everywhere. Do not set it for an aarch64 build.
- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` is part of the v4 recommendation. Zen 4 and
  Zen 5 make `tzcnt`, `blsr`, `blsi` and `blsmsk` wait for the last write to their output register,
  and LLVM breaks that wait only under `target-cpu=znver4`/`znver5` tuning, not under `x86-64-v4`.
  jpeg-rs measured its Huffman loop 30 to 50% slower without the two features; the v4 column here
  is built with them, and whether the Hulsken decoder needs them is not measured. rustc prints a
  warning that it does not know the two features and passes them to LLVM, which does.
  `-C target-cpu=znver5` gives the same speed but runs only on Zen 5.
- The library keeps its runtime SIMD dispatch, and the default build already uses AVX-512: the
  process picks one tier on first use: AVX-512 (AVX2, F and BW; the Hulsken unpack also needs GFNI)
  when the CPU has it, else AVX2, NEON or scalar. On the AVX-512 tier `lift_odd` and `trunc_row`
  run their AVX2 kernels ([safety.md](safety.md) lists the kernels of each tier). So the default
  build uses the AVX-512 tier on a machine that has it, with no flag. The v4 build
  only widens the code around the kernels (the Hulsken decoder, the wavelet loops, the copies),
  where the compiler may use the wider baseline. The output is the same: `ci` runs the `dsp::` unit
  tests, which hold every kernel against scalar, under the v4 build, released.

## What the v4 tables measure

`bench.sh` prints two more tables, "v4 build, default parallelism" and "v4 build, single-threaded":
the same comparison as the default-build tables, with Go built with `GOAMD64=v4` (`oracle-v4`) and
Rust built with the flags above (`bench-v4`, its own cargo cache mount). It is the number for a
consumer that follows the profile on this page. The two tables are not gated: the 1.10 gate is for
the default build, which every consumer has. The numbers are in [benchmarks.md](benchmarks.md).

## Reusing buffers

Keep one `Decoder` per worker and give it to every image the worker opens
(`Image::open(..)?.with_decoder(&decoder)`). The images then take their coefficient cache entries,
tile scratch and staging buffers from it instead of allocating them, so a worker that goes from
image to image does not page-fault about 56 MB of fresh memory per 3000x2000 level 0 again. With
`Limits::max_threads` 1 the second image's decodes make no allocator call (`tests/reuse_alloc.rs`).
The limits, the output and the errors do not change. Details: [api.md](api.md#reusing-buffers-decoder).
The bench (`examples/bench.rs`) uses one `Decoder` per fixture; its fault column shows the effect.

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
worker that links this crate); the library does not change. `pgo.yml` (dispatch only) measures how
much they gain on the bench example: it builds the example with `cargo-pgo`, trains it on the
fixtures of `oracle -genbench` (`bench <fixtures> 5 1`, then `bench <fixtures> 5`), then times the
default, the PGO and the PGO plus BOLT binaries with `tests/parity/bench.sh` and prints the table in
the job summary. BOLT is optional there: if the package is not available the job reports PGO only.

`pgo.yml` on an AMD Ryzen 7 9800X3D (BOLT was available), Rust ms per decode
of level 0, the sum over the six fixtures and the fixtures that moved most, change against the
default build:

| table | fixture | default ms | PGO ms | PGO + BOLT ms |
|---|---|---|---|---|
| default parallelism | all | 53.07 | 52.62 (-0.8%) | 52.64 (-0.8%) |
| default parallelism | flat-v2-8 | 1.65 | 1.53 (-7.3%) | 1.62 (-1.8%) |
| default parallelism | dense-v1-16 | 15.39 | 15.15 (-1.6%) | 14.98 (-2.7%) |
| single-threaded | all | 226.96 | 227.70 (+0.3%) | 225.70 (-0.6%) |
| single-threaded | flat-v2-8 | 6.38 | 5.99 (-6.1%) | 6.25 (-2.0%) |
| single-threaded | dense-v1-16 | 65.46 | 65.99 (+0.8%) | 64.43 (-1.6%) |

PGO and BOLT gain under 1% on the whole set. The time is in the SIMD kernels and the Hulsken
decode loop, which the default build already compiles well; only the small flat and sparse
fixtures, where the per-tile overhead weighs more, gain 2 to 7% from PGO.

The training input is the benchmark's own fixtures, so a gain here is an upper bound for another
workload, and a gain of a few percent is not worth a PGO build step in a consumer. The multi-threaded
table moves with thread scheduling on a shared runner as well as with the build.
