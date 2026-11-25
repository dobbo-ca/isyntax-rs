//! NEON kernels, 8 int16 lanes; the same arithmetic as `avx2.rs` and the scalar code. These are
//! safe `target_feature` functions (the dispatch in `simd.rs` proves NEON is present); the only
//! `unsafe` is the load and store helpers, whose array arguments prove the bounds; the kernels cut
//! their slices into such arrays in safe code. The structured stores
//! (`vst2q`, `vst3q`, `vst4`) do the interleaving the AVX2 kernels do with shuffles.

use std::arch::aarch64::*;

const LANES: usize = 8;

/// The `N` elements of `s` from `i`, if there are that many.
fn at<T, const N: usize>(s: &[T], i: usize) -> Option<&[T; N]> {
    s.get(i..)?.first_chunk()
}

fn at_mut<T, const N: usize>(s: &mut [T], i: usize) -> Option<&mut [T; N]> {
    s.get_mut(i..)?.first_chunk_mut()
}

/// The only `unsafe` of this tier: array arguments prove the bounds.
#[allow(unsafe_code)]
mod mem {
    use super::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn ld(s: &[i16; LANES]) -> int16x8_t {
        // SAFETY: `s` is 8 int16 (16 bytes), which the array type proves; vld1q has no alignment
        // requirement.
        unsafe { vld1q_s16(s.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn st(s: &mut [i16; LANES], v: int16x8_t) {
        // SAFETY: `s` is 8 int16 (16 bytes), which the array type proves; vst1q has no alignment
        // requirement.
        unsafe { vst1q_s16(s.as_mut_ptr(), v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn st2(s: &mut [i16; 2 * LANES], v: int16x8x2_t) {
        // SAFETY: `s` is 16 int16, which the array type proves and vst2q writes; no alignment
        // requirement.
        unsafe { vst2q_s16(s.as_mut_ptr(), v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn st3(s: &mut [i16; 3 * LANES], v: int16x8x3_t) {
        // SAFETY: `s` is 24 int16, which the array type proves and vst3q writes; no alignment
        // requirement.
        unsafe { vst3q_s16(s.as_mut_ptr(), v) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn st4(s: &mut [u8; 4 * LANES], v: uint8x8x4_t) {
        // SAFETY: `s` is 32 bytes, which the array type proves and vst4 writes; no alignment
        // requirement.
        unsafe { vst4_u8(s.as_mut_ptr(), v) }
    }
}

use mem::{ld, st, st2, st3, st4};

/// `(l - ((h0 + h1 + 2) >> 2))` per lane, from the halving add `t = (h0 + h1) >> 1`.
#[inline]
#[target_feature(enable = "neon")]
fn lift_odd_lane(l: int16x8_t, h0: int16x8_t, h1: int16x8_t) -> int16x8_t {
    let t = vhaddq_s16(h0, h1);
    vsubq_s16(l, vsubq_s16(t, vshrq_n_s16::<1>(t)))
}

#[target_feature(enable = "neon")]
pub(super) fn lift_odd(out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) -> usize {
    let mut n = 0;
    let (out, l) = (out.as_chunks_mut::<LANES>().0, l.as_chunks::<LANES>().0);
    let (h0, h1) = (h0.as_chunks::<LANES>().0, h1.as_chunks::<LANES>().0);
    for (((o, l), h0), h1) in out.iter_mut().zip(l).zip(h0).zip(h1) {
        st(o, lift_odd_lane(ld(l), ld(h0), ld(h1)));
        n += 1;
    }
    n * LANES
}

#[target_feature(enable = "neon")]
pub(super) fn lift_even(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) -> usize {
    let mut n = 0;
    let (out, h) = (out.as_chunks_mut::<LANES>().0, h.as_chunks::<LANES>().0);
    let (o0, o1) = (o0.as_chunks::<LANES>().0, o1.as_chunks::<LANES>().0);
    for (((out, h), o0), o1) in out.iter_mut().zip(h).zip(o0).zip(o1) {
        st(out, vaddq_s16(ld(h), vhaddq_s16(ld(o0), ld(o1))));
        n += 1;
    }
    n * LANES
}

/// See `avx2::idwt_row`: whole blocks of 8 positions while `k + 9 <= n`; returns the first
/// position not done.
#[target_feature(enable = "neon")]
pub(super) fn idwt_row(out: &mut [i16], l: &[i16], h: &[i16]) -> usize {
    let n = l.len().min(h.len()).min(out.len() / 2);
    if n < LANES + 1 {
        return 0;
    }
    // d[-1] = d[0].
    let d0 = (i32::from(l[0]) - ((i32::from(h[0]) + i32::from(h[1]) + 2) >> 2)) as i16;
    let mut prev = vdupq_n_s16(d0);
    let mut k = 0;
    while k + LANES < n {
        // `k + LANES < n` puts all four in bounds.
        let (Some(lk), Some(hk), Some(hk1), Some(o)) =
            (at(l, k), at(h, k), at(h, k + 1), at_mut(out, 2 * k))
        else {
            break;
        };
        let h0 = ld(hk);
        let d = lift_odd_lane(ld(lk), h0, ld(hk1));
        // [prev[7], d[0..7]]
        let dp = vextq_s16::<7>(prev, d);
        let e = vaddq_s16(h0, vhaddq_s16(dp, d));
        st2(o, int16x8x2_t(e, d));
        prev = d;
        k += LANES;
    }
    k
}

/// `(r, g, b)` of YCoCg, wrapping int16.
#[inline]
#[target_feature(enable = "neon")]
fn ycocg(y: int16x8_t, co: int16x8_t, cg: int16x8_t) -> (int16x8_t, int16x8_t, int16x8_t) {
    let t = vsubq_s16(y, vshrq_n_s16::<1>(cg));
    let g = vaddq_s16(t, cg);
    let b = vsubq_s16(t, vshrq_n_s16::<1>(co));
    let r = vaddq_s16(b, co);
    (r, g, b)
}

#[target_feature(enable = "neon")]
pub(super) fn ycocg_to_rgba(
    dst: &mut [u8],
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    shift: u32,
) -> usize {
    let mut n = 0;
    let dst = dst.as_chunks_mut::<{ 4 * LANES }>().0;
    let (y, co, cg) = (
        y.as_chunks::<LANES>().0,
        co.as_chunks::<LANES>().0,
        cg.as_chunks::<LANES>().0,
    );
    // A negative count shifts right, arithmetically for signed lanes.
    let sh = vdupq_n_s16(-((shift & 15) as i16));
    let alpha = vdup_n_u8(255);
    for (((d, y), co), cg) in dst.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(ld(y), ld(co), ld(cg));
        // vqmovun saturates to [0, 255].
        let px = uint8x8x4_t(
            vqmovun_s16(vshlq_s16(r, sh)),
            vqmovun_s16(vshlq_s16(g, sh)),
            vqmovun_s16(vshlq_s16(b, sh)),
            alpha,
        );
        st4(d, px);
        n += 1;
    }
    n * LANES
}

/// See `avx2::samples`.
#[target_feature(enable = "neon")]
pub(super) fn samples(
    dst: &mut [i16],
    a: &[i16],
    b: &[i16],
    c: &[i16],
    lo: i16,
    hi: i16,
    planes: bool,
) -> usize {
    let mut n = 0;
    let dst = dst.as_chunks_mut::<{ 3 * LANES }>().0;
    let (a, b, c) = (
        a.as_chunks::<LANES>().0,
        b.as_chunks::<LANES>().0,
        c.as_chunks::<LANES>().0,
    );
    let (lo, hi) = (vdupq_n_s16(lo), vdupq_n_s16(hi));
    let cl = |v| vminq_s16(vmaxq_s16(v, lo), hi);
    for (((d, a), b), c) in dst.iter_mut().zip(a).zip(b).zip(c) {
        let (r, g, b) = if planes {
            (ld(a), ld(b), ld(c))
        } else {
            ycocg(ld(a), ld(b), ld(c))
        };
        st3(d, int16x8x3_t(cl(r), cl(g), cl(b)));
        n += 1;
    }
    n * LANES
}

#[target_feature(enable = "neon")]
pub(super) fn trunc_row(row: &mut [i16], mask: i16, o: i16) -> usize {
    let row = row.as_chunks_mut::<LANES>().0;
    let n = row.len();
    let (mask, o) = (vdupq_n_s16(mask), vdupq_n_s16(o));
    for r in row {
        let v = ld(r);
        // abs wraps at -32768 like the scalar code; add `o` to the nonzero magnitudes.
        let a = vbicq_s16(vabsq_s16(v), mask);
        let zero = vreinterpretq_s16_u16(vceqzq_s16(a));
        let a = vaddq_s16(a, vbicq_s16(o, zero));
        st(r, vbslq_s16(vcltzq_s16(v), vnegq_s16(a), a));
    }
    n * LANES
}
