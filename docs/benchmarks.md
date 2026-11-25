# Benchmarks

`bench.yml` runs `tests/parity/bench.sh` in one container: `oracle -genbench` writes six 3000 x 2000
files with three levels (dense and sparse random detail in iSyntax2 8 and 9 bits and v1 16 bits,
and a flat one), and `oracle -bench` and `examples/bench.rs` time the decode of the whole of level
0 into RGBA from a fresh `Image` (a cold coefficient cache; `examples/bench.rs` gives every fresh
`Image` one `Decoder` shared per fixture, so the working buffers are warm, see
[performance.md](performance.md#reusing-buffers); Go has no equivalent), best of ROUNDS (7) runs of N (10) decodes each. The job
summary has two gated tables with the ratio per file: Go at its default parallelism against Rust at the
default `max_threads`, and both single-threaded (`GOMAXPROCS=1` for Go, `bench <dir> <n> 1`, that is
`max_threads` 1, for Rust). Each checks the pixel hash of both sides in every round, and the ratio is gated at 1.10. The bench image also prints the two v4 tables (see
[v4 tables](#v4-tables)).

`bench.yml` times Go and Rust on the same 3000 x 2000 files and prints the ratio. A second step runs
`examples/kernels.rs`, which times each SIMD kernel on each tier (scalar, AVX2, AVX-512) and prints a
table of ns per call and the AVX-512 / AVX2 ratio.

## Columns

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|

- `Go faults`, `Rust faults`: minor page faults per decode, from `getrusage` (Go) and field 10 of
  `/proc/self/stat` (Rust, 0 if unreadable), read before and after the timed loop and divided by N,
  for the fastest round. A fault is a first touch of a page, so this is a count of fresh memory a
  decode asks for ([performance.md](performance.md), "Reusing buffers").
- `spread`: the slowest over the fastest Rust time of the last batch of rounds, for that file.

## Retry rule

After ROUNDS rounds, `bench.sh` takes the slowest over the fastest time of each file, for Go and for
Rust. If either is above SPREAD (default 1.25), the machine was noisy for that file: it is run alone
(a copy of just that file in its own directory) for RETRY_ROUNDS (default ROUNDS) more rounds, Go
and Rust alternating, and this is done at most twice. The table shows the fastest time over all the
rounds, and the spread of the last batch. The files that were retried, and their spreads, are on
stderr (`retry <n> of <file>: spread Go .., Rust ..`).

## v4 tables

When `oracle-v4` (`GOAMD64=v4`) and `bench-v4` (`-C target-cpu=x86-64-v4` and the two `false-deps`
features) are installed, as in the `bench` image, `bench.sh` prints two more tables: v4 build at
default parallelism and v4 build single-threaded. They are not gated. See
[performance.md](performance.md).

## Hosted baseline

`bench.yml` run 45706608176 on a GitHub-hosted `ubuntu-24.04` runner (2 vCPU, Intel Xeon 6975P-C,
AVX-512 F/BW/VL), the baseline for later hosted runs. Shared VMs are noisy, and these numbers are
not comparable with the dedicated-machine numbers below. Same inputs and columns.

#### Default parallelism: Go (GOMAXPROCS = all CPUs) against Rust (max_threads = available_parallelism)

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 180.30 | 99.49 | 0.55 | 9452 | 26 | 1.07 |
| dense-v2-8 | 24.0 | 153.66 | 86.63 | 0.56 | 4184 | 25 | 1.06 |
| dense-v2-9 | 24.0 | 167.99 | 97.29 | 0.58 | 2084 | 25 | 1.05 |
| flat-v2-8 | 24.0 | 23.17 | 14.80 | 0.64 | 16 | 0 | 1.16 |
| sparse-v1-16 | 24.0 | 58.69 | 28.55 | 0.49 | 12 | 0 | 1.15 |
| sparse-v2-8 | 24.0 | 59.15 | 28.81 | 0.49 | 1370 | 0 | 1.12 |

#### Single-threaded: Go with GOMAXPROCS=1 against Rust with max_threads = 1

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 192.04 | 108.61 | 0.57 | 10403 | 26 | 1.03 |
| dense-v2-8 | 24.0 | 162.05 | 94.57 | 0.58 | 2433 | 26 | 1.04 |
| dense-v2-9 | 24.0 | 180.35 | 106.30 | 0.59 | 3645 | 25 | 1.03 |
| flat-v2-8 | 24.0 | 27.16 | 15.82 | 0.58 | 1022 | 26 | 1.11 |
| sparse-v1-16 | 24.0 | 62.62 | 32.33 | 0.52 | 57 | 0 | 1.06 |
| sparse-v2-8 | 24.0 | 61.38 | 32.83 | 0.53 | 68 | 26 | 1.07 |

#### v4 build, default parallelism, not gated: Go GOAMD64=v4 against Rust x86-64-v4

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 177.71 | 98.34 | 0.55 | 9295 | 26 | 1.03 |
| dense-v2-8 | 24.0 | 150.19 | 85.47 | 0.57 | 4160 | 26 | 1.03 |
| dense-v2-9 | 24.0 | 164.63 | 96.66 | 0.59 | 2074 | 26 | 1.02 |
| flat-v2-8 | 24.0 | 23.21 | 14.87 | 0.64 | 16 | 0 | 1.07 |
| sparse-v1-16 | 24.0 | 57.73 | 28.64 | 0.50 | 9 | 0 | 1.05 |
| sparse-v2-8 | 24.0 | 57.08 | 28.67 | 0.50 | 21 | 0 | 1.14 |

#### v4 build, single-threaded, not gated: Go GOAMD64=v4 with GOMAXPROCS=1 against Rust x86-64-v4 with max_threads = 1

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 190.52 | 108.36 | 0.57 | 9730 | 26 | 1.03 |
| dense-v2-8 | 24.0 | 160.08 | 94.74 | 0.59 | 1918 | 26 | 1.04 |
| dense-v2-9 | 24.0 | 178.02 | 107.33 | 0.60 | 3135 | 26 | 1.01 |
| flat-v2-8 | 24.0 | 26.36 | 15.95 | 0.60 | 99 | 25 | 1.05 |
| sparse-v1-16 | 24.0 | 61.85 | 32.83 | 0.53 | 46 | 26 | 1.05 |
| sparse-v2-8 | 24.0 | 60.84 | 33.24 | 0.55 | 115 | 26 | 1.04 |

#### Kernels (`examples/kernels.rs`, ns per call)

| kernel | size | scalar ns | avx2 ns | avx512 ns | avx512/avx2 |
|---|---|---|---|---|---|
| lift_odd | 272 | 84 | 18 | 18 | 1.00 |
| lift_odd | 256 | 79 | 17 | 17 | 0.97 |
| lift_odd | 65536 | 20687 | 5411 | 5410 | 1.00 |
| lift_even | 272 | 29 | 16 | 15 | 0.93 |
| lift_even | 256 | 28 | 16 | 14 | 0.88 |
| lift_even | 65536 | 5249 | 4713 | 4516 | 0.96 |
| idwt_row | 136 | 85 | 30 | 23 | 0.78 |
| idwt_row | 256 | 162 | 47 | 46 | 0.98 |
| idwt_row | 65536 | 41398 | 9170 | 8030 | 0.88 |
| ycocg_to_rgba | 272 | 237 | 36 | 53 | 1.46 |
| ycocg_to_rgba | 256 | 223 | 36 | 29 | 0.81 |
| ycocg_to_rgba | 65536 | 55974 | 8415 | 7043 | 0.84 |
| ycocg_to_rgb16 | 272 | 98 | 40 | 45 | 1.11 |
| ycocg_to_rgb16 | 256 | 94 | 39 | 35 | 0.89 |
| ycocg_to_rgb16 | 65536 | 22591 | 9614 | 9216 | 0.96 |
| ycocg_to_samples | 272 | 140 | 40 | 46 | 1.15 |
| ycocg_to_samples | 256 | 130 | 39 | 35 | 0.89 |
| ycocg_to_samples | 65536 | 33322 | 9373 | 9223 | 0.98 |
| planes_to_samples | 272 | 114 | 33 | 42 | 1.30 |
| planes_to_samples | 256 | 106 | 30 | 30 | 1.02 |
| planes_to_samples | 65536 | 26774 | 8023 | 8602 | 1.07 |
| trunc_row | 136 | 15 | 8 | 8 | 1.00 |
| trunc_row | 256 | 25 | 11 | 11 | 1.00 |
| trunc_row | 65536 | 6399 | 2642 | 2640 | 1.00 |
| pack64 | 272 | 99 | 98 | 56 | 0.57 |
| pack64 | 256 | 93 | 93 | 50 | 0.54 |
| pack64 | 65536 | 23745 | 23811 | 4352 | 0.18 |
| rgba_to_ycbcr | 272 | 495 | 495 | 92 | 0.19 |
| rgba_to_ycbcr | 256 | 357 | 357 | 67 | 0.19 |
| rgba_to_ycbcr | 65536 | 91274 | 91223 | 17026 | 0.19 |
| hulsken_unpack | 4096 | 6264 | 8966 | 829 | 0.09 |
| hulsken_unpack | 16384 | 24480 | 24456 | 3204 | 0.13 |
| hulsken_unpack | 65536 | 99271 | 99184 | 12869 | 0.13 |

## Earlier result (dedicated machine)

`bench.yml` on an AMD Ryzen 7 9800X3D (8 cores), before the move to hosted runners: 3000 x 2000
level 0 RGBA (24.0 MB out), best of 7 rounds x n=10, ms per decode (lower is better), faults =
minor page faults per decode, spread = slowest over fastest Rust round of the final batch. The
default and single-threaded tables are gated at 1.10; the v4 tables are not.

#### Default parallelism: Go (GOMAXPROCS = all CPUs) against Rust (max_threads = available_parallelism)

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 28.28 | 15.60 | 0.55 | 8160 | 78 | 1.02 |
| dense-v2-8 | 24.0 | 24.47 | 13.22 | 0.54 | 2171 | 62 | 1.02 |
| dense-v2-9 | 24.0 | 27.47 | 15.20 | 0.55 | 9313 | 62 | 1.03 |
| flat-v2-8 | 24.0 | 4.48 | 1.79 | 0.40 | 4729 | 0 | 1.14 |
| sparse-v1-16 | 24.0 | 8.87 | 4.34 | 0.49 | 2632 | 0 | 1.09 |
| sparse-v2-8 | 24.0 | 9.32 | 4.35 | 0.47 | 961 | 46 | 1.15 |

#### Single-threaded: Go with GOMAXPROCS=1 against Rust with max_threads = 1

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 120.73 | 65.82 | 0.55 | 9732 | 26 | 1.08 |
| dense-v2-8 | 24.0 | 102.43 | 56.88 | 0.56 | 5167 | 26 | 1.07 |
| dense-v2-9 | 24.0 | 113.12 | 64.77 | 0.57 | 1561 | 26 | 1.04 |
| flat-v2-8 | 24.0 | 12.17 | 6.51 | 0.53 | 87 | 25 | 1.16 |
| sparse-v1-16 | 24.0 | 34.60 | 17.22 | 0.50 | 340 | 25 | 1.11 |
| sparse-v2-8 | 24.0 | 34.51 | 17.44 | 0.51 | 5 | 26 | 1.08 |

#### v4 build, default parallelism, not gated: Go GOAMD64=v4 against Rust x86-64-v4

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 28.78 | 15.26 | 0.53 | 7208 | 77 | 2.76 |
| dense-v2-8 | 24.0 | 25.26 | 13.01 | 0.51 | 5280 | 61 | 1.22 |
| dense-v2-9 | 24.0 | 27.47 | 14.73 | 0.54 | 9099 | 28 | 2.85 |
| flat-v2-8 | 24.0 | 4.34 | 1.73 | 0.40 | 2118 | 0 | 1.25 |
| sparse-v1-16 | 24.0 | 8.39 | 4.03 | 0.48 | 343 | 75 | 1.08 |
| sparse-v2-8 | 24.0 | 8.66 | 4.63 | 0.53 | 1886 | 46 | 1.13 |

#### v4 build, single-threaded, not gated: Go GOAMD64=v4 with GOMAXPROCS=1 against Rust x86-64-v4 with max_threads = 1

| input | MB out | Go ms | Rust ms | Rust / Go | Go faults | Rust faults | spread |
|---|---|---|---|---|---|---|---|
| dense-v1-16 | 24.0 | 119.11 | 64.79 | 0.54 | 10386 | 26 | 1.12 |
| dense-v2-8 | 24.0 | 103.27 | 55.77 | 0.54 | 2270 | 26 | 1.03 |
| dense-v2-9 | 24.0 | 115.04 | 63.41 | 0.55 | 204 | 26 | 1.01 |
| flat-v2-8 | 24.0 | 12.97 | 6.62 | 0.51 | 2306 | 26 | 1.16 |
| sparse-v1-16 | 24.0 | 34.44 | 17.10 | 0.50 | 30 | 25 | 1.01 |
| sparse-v2-8 | 24.0 | 34.33 | 17.39 | 0.51 | 120 | 26 | 1.02 |

#### Kernels (`examples/kernels.rs`, ns per call)

On the AVX-512 tier `lift_odd` and `trunc_row` run their AVX2 kernels, and `pack64`,
`rgba_to_ycbcr` and `hulsken_unpack` have no AVX2 kernel (their avx2 column is scalar); see
[safety.md](safety.md).

| kernel | size | scalar ns | avx2 ns | avx512 ns | avx512/avx2 |
|---|---|---|---|---|---|
| lift_odd | 272 | 93 | 9 | 9 | 1.00 |
| lift_odd | 256 | 87 | 8 | 8 | 1.00 |
| lift_odd | 65536 | 22356 | 2149 | 2134 | 0.99 |
| lift_even | 272 | 13 | 9 | 7 | 0.81 |
| lift_even | 256 | 12 | 8 | 7 | 0.85 |
| lift_even | 65536 | 2634 | 1993 | 1839 | 0.92 |
| idwt_row | 136 | 82 | 16 | 12 | 0.75 |
| idwt_row | 256 | 155 | 28 | 31 | 1.12 |
| idwt_row | 65536 | 40934 | 5253 | 3669 | 0.70 |
| ycocg_to_rgba | 272 | 177 | 17 | 20 | 1.15 |
| ycocg_to_rgba | 256 | 167 | 18 | 11 | 0.65 |
| ycocg_to_rgba | 65536 | 42461 | 4177 | 2517 | 0.60 |
| ycocg_to_rgb16 | 272 | 59 | 24 | 13 | 0.53 |
| ycocg_to_rgb16 | 256 | 56 | 24 | 11 | 0.46 |
| ycocg_to_rgb16 | 65536 | 14286 | 5880 | 3644 | 0.62 |
| ycocg_to_samples | 272 | 90 | 24 | 16 | 0.68 |
| ycocg_to_samples | 256 | 85 | 23 | 11 | 0.47 |
| ycocg_to_samples | 65536 | 21781 | 5862 | 3590 | 0.61 |
| planes_to_samples | 272 | 68 | 18 | 16 | 0.86 |
| planes_to_samples | 256 | 64 | 16 | 11 | 0.68 |
| planes_to_samples | 65536 | 16370 | 4099 | 3127 | 0.76 |
| trunc_row | 136 | 9 | 5 | 5 | 1.00 |
| trunc_row | 256 | 15 | 6 | 6 | 1.00 |
| trunc_row | 65536 | 3908 | 1297 | 1284 | 0.99 |
| pack64 | 272 | 62 | 62 | 21 | 0.34 |
| pack64 | 256 | 59 | 58 | 17 | 0.29 |
| pack64 | 65536 | 15165 | 15174 | 1294 | 0.09 |
| rgba_to_ycbcr | 272 | 194 | 194 | 25 | 0.13 |
| rgba_to_ycbcr | 256 | 184 | 184 | 24 | 0.13 |
| rgba_to_ycbcr | 65536 | 44920 | 44965 | 5918 | 0.13 |
| hulsken_unpack | 4096 | 4235 | 4241 | 452 | 0.11 |
| hulsken_unpack | 16384 | 16877 | 16964 | 1679 | 0.10 |
| hulsken_unpack | 65536 | 67782 | 67798 | 6641 | 0.10 |
