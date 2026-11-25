//! AVX2 kernels, 16 int16 lanes. Each is a safe `target_feature` function: the caller (the
//! dispatch in `simd.rs`) proves AVX2 is present. The only `unsafe` here is the unaligned
//! load and store helpers, whose array arguments prove the bounds; the kernels cut their
//! slices into such arrays in safe code. Values are int16 with
//! wraparound, as the scalar code; averages use the halving add (`(a & b) + ((a ^ b) >> 1)`),
//! which cannot overflow where `a + b` would.

use std::arch::x86_64::*;

const LANES: usize = 16;

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
    #[target_feature(enable = "avx2")]
    pub(super) fn ld(s: &[i16; LANES]) -> __m256i {
        // SAFETY: `s` is 16 int16 (32 bytes), which the array type proves; `loadu` needs no alignment.
        unsafe { _mm256_loadu_si256(s.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn st(s: &mut [i16; LANES], v: __m256i) {
        // SAFETY: `s` is 16 int16 (32 bytes), which the array type proves; `storeu` needs no alignment.
        unsafe { _mm256_storeu_si256(s.as_mut_ptr().cast(), v) }
    }

    /// Stores `a` then `b`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn st2(s: &mut [i16; 2 * LANES], a: __m256i, b: __m256i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 32 int16 (64 bytes), which the array type proves; both 32-byte stores are
        // inside it; `storeu` needs no alignment.
        unsafe {
            _mm256_storeu_si256(p.cast(), a);
            _mm256_storeu_si256(p.add(LANES).cast(), b);
        }
    }

    /// Stores `a`, `b`, then `c`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn st3(s: &mut [i16; 3 * LANES], a: __m256i, b: __m256i, c: __m256i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 48 int16 (96 bytes), which the array type proves; the three 32-byte stores
        // are inside it; `storeu` needs no alignment.
        unsafe {
            _mm256_storeu_si256(p.cast(), a);
            _mm256_storeu_si256(p.add(LANES).cast(), b);
            _mm256_storeu_si256(p.add(2 * LANES).cast(), c);
        }
    }

    /// Stores `a` then `b`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn st_bytes(s: &mut [u8; 64], a: __m256i, b: __m256i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 64 bytes, which the array type proves; both 32-byte stores are inside it;
        // `storeu` needs no alignment.
        unsafe {
            _mm256_storeu_si256(p.cast(), a);
            _mm256_storeu_si256(p.add(32).cast(), b);
        }
    }
}

use mem::{ld, st, st_bytes, st2, st3};

/// `floor((a + b) / 2)` per lane.
#[inline]
#[target_feature(enable = "avx2")]
fn avg(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi16(
        _mm256_and_si256(a, b),
        _mm256_srai_epi16::<1>(_mm256_xor_si256(a, b)),
    )
}

#[target_feature(enable = "avx2")]
pub(super) fn lift_odd(out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) -> usize {
    let mut n = 0;
    let (out, l) = (out.as_chunks_mut::<LANES>().0, l.as_chunks::<LANES>().0);
    let (h0, h1) = (h0.as_chunks::<LANES>().0, h1.as_chunks::<LANES>().0);
    for (((o, l), h0), h1) in out.iter_mut().zip(l).zip(h0).zip(h1) {
        let t = avg(ld(h0), ld(h1));
        // (h0 + h1 + 2) >> 2 == t - (t >> 1)
        let r = _mm256_sub_epi16(t, _mm256_srai_epi16::<1>(t));
        st(o, _mm256_sub_epi16(ld(l), r));
        n += 1;
    }
    n * LANES
}

#[target_feature(enable = "avx2")]
pub(super) fn lift_even(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) -> usize {
    let mut n = 0;
    let (out, h) = (out.as_chunks_mut::<LANES>().0, h.as_chunks::<LANES>().0);
    let (o0, o1) = (o0.as_chunks::<LANES>().0, o1.as_chunks::<LANES>().0);
    for (((out, h), o0), o1) in out.iter_mut().zip(h).zip(o0).zip(o1) {
        st(out, _mm256_add_epi16(ld(h), avg(ld(o0), ld(o1))));
        n += 1;
    }
    n * LANES
}

/// The 1D inverse lift of `idwt_row` for whole blocks of 16 positions, in registers: the detail
/// `d[k]` of one block, shifted by one lane against the last block's, gives `d[k-1]`. A block
/// needs `h[k + 16]`, so it runs while `k + 17 <= n`. Returns the first position not done.
#[target_feature(enable = "avx2")]
pub(super) fn idwt_row(out: &mut [i16], l: &[i16], h: &[i16]) -> usize {
    let n = l.len().min(h.len()).min(out.len() / 2);
    if n < LANES + 1 {
        return 0;
    }
    // d[-1] = d[0].
    let d0 = (i32::from(l[0]) - ((i32::from(h[0]) + i32::from(h[1]) + 2) >> 2)) as i16;
    let mut prev = _mm256_set1_epi16(d0);
    let mut k = 0;
    while k + LANES < n {
        // `k + LANES < n` puts all four in bounds.
        let (Some(lk), Some(hk), Some(hk1), Some(o)) =
            (at(l, k), at(h, k), at(h, k + 1), at_mut(out, 2 * k))
        else {
            break;
        };
        let h0 = ld(hk);
        let t = avg(h0, ld(hk1));
        let d = _mm256_sub_epi16(ld(lk), _mm256_sub_epi16(t, _mm256_srai_epi16::<1>(t)));
        // [prev[15], d[0..15]]
        let dp = _mm256_alignr_epi8::<14>(d, _mm256_permute2x128_si256::<0x21>(prev, d));
        let e = _mm256_add_epi16(h0, avg(dp, d));
        let lo = _mm256_unpacklo_epi16(e, d);
        let hi = _mm256_unpackhi_epi16(e, d);
        st2(
            o,
            _mm256_permute2x128_si256::<0x20>(lo, hi),
            _mm256_permute2x128_si256::<0x31>(lo, hi),
        );
        prev = d;
        k += LANES;
    }
    k
}

/// `(r, g, b)` of YCoCg, wrapping int16.
#[inline]
#[target_feature(enable = "avx2")]
fn ycocg(y: __m256i, co: __m256i, cg: __m256i) -> (__m256i, __m256i, __m256i) {
    let t = _mm256_sub_epi16(y, _mm256_srai_epi16::<1>(cg));
    let g = _mm256_add_epi16(t, cg);
    let b = _mm256_sub_epi16(t, _mm256_srai_epi16::<1>(co));
    let r = _mm256_add_epi16(b, co);
    (r, g, b)
}

#[target_feature(enable = "avx2")]
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
    let sh = _mm_cvtsi32_si128((shift & 15) as i32);
    let alpha = _mm256_set1_epi16(255);
    for (((d, y), co), cg) in dst.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(ld(y), ld(co), ld(cg));
        let (r, g, b) = (
            _mm256_sra_epi16(r, sh),
            _mm256_sra_epi16(g, sh),
            _mm256_sra_epi16(b, sh),
        );
        // packus saturates to [0, 255]; then interleave bytes r g b a per pixel.
        let rb = _mm256_packus_epi16(r, b);
        let ga = _mm256_packus_epi16(g, alpha);
        let rg = _mm256_unpacklo_epi8(rb, ga);
        let ba = _mm256_unpackhi_epi8(rb, ga);
        let p0 = _mm256_unpacklo_epi16(rg, ba);
        let p1 = _mm256_unpackhi_epi16(rg, ba);
        st_bytes(
            d,
            _mm256_permute2x128_si256::<0x20>(p0, p1),
            _mm256_permute2x128_si256::<0x31>(p0, p1),
        );
        n += 1;
    }
    n * LANES
}

/// `a, b, c` are Y, Co, Cg (or R, G, B if `planes`), clamped to `[lo, hi]` and written as
/// R, G, B triples.
#[target_feature(enable = "avx2")]
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
    let (lo, hi) = (_mm256_set1_epi16(lo), _mm256_set1_epi16(hi));
    // Byte shuffles that put each channel's words at their RGB slots (per 128-bit lane), so
    // each output vector is two blends of the three.
    let shuf = |hi: i64, lo: i64| _mm256_broadcastsi128_si256(_mm_set_epi64x(hi, lo));
    let sr = shuf(0x0B0A_0504_0F0E_0908, 0x0302_0D0C_0706_0100);
    let sg = shuf(0x0504_0F0E_0908_0302, 0x0D0C_0706_0100_0B0A);
    let sb = shuf(0x0F0E_0908_0302_0D0C, 0x0706_0100_0B0A_0504);
    for (((d, a), b), c) in dst.iter_mut().zip(a).zip(b).zip(c) {
        let (r, g, b) = if planes {
            (ld(a), ld(b), ld(c))
        } else {
            ycocg(ld(a), ld(b), ld(c))
        };
        let cl = |v| _mm256_min_epi16(_mm256_max_epi16(v, lo), hi);
        let r = _mm256_shuffle_epi8(cl(r), sr);
        let g = _mm256_shuffle_epi8(cl(g), sg);
        let b = _mm256_shuffle_epi8(cl(b), sb);
        let v0 = _mm256_blend_epi16::<0x24>(_mm256_blend_epi16::<0x92>(r, g), b);
        let v1 = _mm256_blend_epi16::<0x24>(_mm256_blend_epi16::<0x92>(b, r), g);
        let v2 = _mm256_blend_epi16::<0x24>(_mm256_blend_epi16::<0x92>(g, b), r);
        st3(
            d,
            _mm256_permute2x128_si256::<0x20>(v0, v1),
            _mm256_permute2x128_si256::<0x30>(v2, v0),
            _mm256_permute2x128_si256::<0x31>(v1, v2),
        );
        n += 1;
    }
    n * LANES
}

#[target_feature(enable = "avx2")]
pub(super) fn trunc_row(row: &mut [i16], mask: i16, o: i16) -> usize {
    let row = row.as_chunks_mut::<LANES>().0;
    let n = row.len();
    let (mask, o) = (_mm256_set1_epi16(mask), _mm256_set1_epi16(o));
    let zero = _mm256_setzero_si256();
    for r in row {
        let v = ld(r);
        // abs wraps at -32768 like the scalar code; add `o` to the nonzero magnitudes.
        let a = _mm256_andnot_si256(mask, _mm256_abs_epi16(v));
        let add = _mm256_andnot_si256(_mm256_cmpeq_epi16(a, zero), o);
        st(r, _mm256_sign_epi16(_mm256_add_epi16(a, add), v));
    }
    n * LANES
}
