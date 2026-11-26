# Safety: `unsafe` and SIMD

`unsafe` is confined to the SIMD kernel modules (`src/simd/`), each block with a `// SAFETY:` comment.

## SIMD

The sample kernels of the decode (inverse 5/3 lifting and row interleave, YCoCg to RGBA, RGB16 and
clamped samples, truncation) have a scalar reference, an AVX2 kernel on x86_64 and a NEON kernel on
aarch64, ported from Go's `dsp_amd64.s` and `dsp_arm64.s`. On x86_64 there is also an AVX-512 tier
(F and BW, 32 int16 lanes, `src/simd/avx512.rs`), which Go does not have. The tier is detected
once at first use. It is the best the CPU has: AVX-512 if `avx2`, `avx512f` and `avx512bw` are all
present, else AVX2, else scalar. `ISYNTAX_SIMD=scalar` forces the scalar tier and
`ISYNTAX_SIMD=avx2` the AVX2 one (on a CPU without it, scalar); `ISYNTAX_SIMD=avx512` panics on a
CPU without AVX-512 F, BW and GFNI (for CI under an emulator); any other value, or none, picks
the best. The tier is chosen per kernel: on the AVX-512 tier a kernel without an AVX-512 version
runs its AVX2 version (one word in the dispatch table of `src/simd.rs`). Every tier is bit-exact with
scalar; the unit tests compare each tier the CPU has with scalar on random and edge data, assert
that AVX2 was detected, and expect the AVX-512 tier when the CPU has AVX-512 F and BW. CI runs the
suite on x86_64 under the default tier, `ISYNTAX_SIMD=scalar` and `ISYNTAX_SIMD=avx2`, and the
`dsp::` tests natively on aarch64. The AVX-512 kernels (F, BW, GFNI) run under Intel SDE in the
`avx512-sde` job (the unit tests on `-spr` with `ISYNTAX_SIMD=avx512`, which fails on a CPU
without AVX-512), and natively when the hosted runner has AVX-512.
Unlike Go, the kernels take any shift (not only 0 and 1).

The AVX-512 kernels are `lift_even`, `idwt_row`, `ycocg_to_rgba` and `samples` (the RGB16 and
clamped-sample variants), each the AVX2 kernel at 32 lanes; `lift_odd` and `trunc_row` run their
AVX2 kernels on the AVX-512 tier (measured, see below). Three more have no AVX2
or NEON version and run only on the AVX-512 tier, scalar elsewhere (`dispatch512!` in `simd.rs`):

- `pack64` (`region.rs`, the RGBA64 packing, 32 pixels at a time): clamp to `[0, 2^bits - 1]` with
  `max`/`min`, replicate as `v << s1 | v >> s2` (8 bits: 8 and 0; 9 bits: 7 and 2, as the scalar
  `v << 7 | v >> 2`), gather the R, G, B lanes of 32 pixels (96 int16) into four vectors with
  `permutex2var_epi16` and set the alpha lanes with `mask_mov_epi16`. The kernel does `bits` 8 and 9
  only (the only values the decoder passes: `8 + shift`, `shift` 0 or 1) and nothing otherwise, so
  the scalar code runs for any other value.
- `rgba_to_ycbcr` (`dsp.rs`, 16 pixels at a time): the scalar formula is Go's `color.RGBToYCbCr`,
  integer arithmetic on int32 with the chroma clamp spelled as a branch that equals
  `clamp(v >> 16, 0, 255)` (the unit test checks that form). The kernel evaluates the same sums in
  int32 lanes (`mullo_epi32`, so no int16 overflow: the Y coefficient 38470 and the chroma
  coefficient 32768 do not fit a signed int16 for `madd_epi16`), `srli`/`srai` by 16, `max`/`min`
  and `cvtepi32_epi8`. It is bit-exact, with no rounding or overflow difference.
- the Hulsken `unpack` (`hulsken.rs`, 64 byte positions of 16 bit planes at a time, the hottest
  loop of the codeblock decode). Per plane half (planes 0 to 7 and 8 to 15): byte interleaves
  (`unpack{lo,hi}_epi8`, `_epi16`, `_epi32`) bring the eight plane bytes of each position into one
  qword, then `gf2p8affine_epi64_epi8` with the selector `0x8040201008040201` and the qword as the
  matrix is the 8x8 bit transpose of `transpose8`; the planes are fed in reverse order because the
  affine instruction reads matrix row `7 - r` for result bit `r`. `unpack{lo,hi}_epi8` of the low
  and high halves is `spread` (coefficient `i` in 16-bit lane `i`), the bit 15 sign rule is
  `mask_sub_epi16(v, movepi16_mask(v), 0, v & 0x7fff)`, and each position's two 4-coefficient rows
  are written at `blk[j >> 1] + (j & 1) * 2 * bw` and `bw` after it. Positions whose plane bytes
  are all zero are skipped (`test_epi64_mask`), as in scalar, in groups of 8 positions (no stores
  for a group whose 8 are zero); a block of 64 is transposed whatever its zeros, except an absent
  half (no plane 0 to 7, or 8 to 15), which is not.
  Only blocks of 64 positions run in the kernel; the scalar loop does the rest. The kernel needs
  GFNI as well, detected where it is called (`simd::unpack`: tier Avx512 and
  `is_x86_feature_detected!("gfni")`), so on an AVX-512 CPU without GFNI the unpack runs scalar.
  It cannot panic on a hostile `blk` or a short `d`: each row is stored through a checked
  `get_mut`, a negative or too large entry stops the kernel, and it reports the positions of the
  blocks it finished, so the scalar loop redoes the rest (and panics there exactly as before on an
  index that does not fit). For a valid `blk` (distinct blocks, as `Scratch::blocks` builds it) the
  result is identical to scalar; for one with overlapping blocks the write order differs.

Features detected: `avx2`, `avx512f`, `avx512bw` (the tier) and `gfni` (the unpack only).

`unsafe` inventory of the AVX-512 tier, as for AVX2:

- the call of a kernel in the dispatch (`simd.rs`): the tier is Avx512 only if `avx2`, `avx512f` and
  `avx512bw` were detected, the features the AVX2 and AVX-512 kernels enable;
- the call of `simd::unpack`: as above, plus `gfni`, checked in the same `if`;
- the load and store helpers of `avx512.rs` (`ld`, `st`, `st2`, `st3`, `st_bytes`, `ld_bytes`,
  `ld3`, `st4_u16`, `st16`), which take `&[i16; 32]`, `&mut [i16; 32]` and the larger arrays the
  loads and stores need, so the array type proves the bounds. No other `unsafe`: the kernels are
  safe `target_feature` functions and cut their slices into arrays in safe code.

For `pack64`, `rgba_to_ycbcr` and `hulsken_unpack` (no AVX2 version) the `avx2 ns` column of the
bench is the scalar code, so `avx512/avx2` there is the ratio to scalar. The `hulsken_unpack`
avx512 column shows `-` on a CPU without GFNI.

`examples/kernels.rs` times every kernel on every tier the CPU has (`bench.yml` prints its table).

## AVX-512 kernels measured and dropped

`bench.yml` (`kernels`, best of 7 rounds, ns per call, AMD Ryzen 7 9800X3D) measured every AVX-512
kernel against the AVX2 one at the row lengths the decoder uses: 272 for the column lifts of
`inverse53` (n = 136), 136 for `idwt_row`, 256 for the output rows. Two lost and were removed; the
AVX-512 tier runs their AVX2 kernels (the `avx2` entry in the `dispatch!` table of `src/simd.rs`):

| kernel | size | avx2 ns | avx512 ns | avx512 / avx2 |
|---|---|---|---|---|
| lift_odd | 272 | 8 | 11 | 1.32 |
| lift_odd | 256 | 8 | 8 | 1.00 |
| trunc_row | 136 | 5 | 5 | 1.00 |
| trunc_row | 256 | 7 | 7 | 0.99 |

They only won on a 65536-element row (0.87 and 0.71), a length the decoder never passes.

The kernels that stayed, at the decoder's sizes: lift_even 0.85 (272), idwt_row 0.75 (136),
ycocg_to_rgba 0.69, ycocg_to_rgb16 0.51, ycocg_to_samples 0.53, planes_to_samples 0.75 (256);
and against scalar, as they have no AVX2 kernel: pack64 0.29, rgba_to_ycbcr 0.13 (256), the
Hulsken unpack 0.11 (size 4096 coefficients: a 64 x 64 block, 512 bytes per plane).

