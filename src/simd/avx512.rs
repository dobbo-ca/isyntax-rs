//! AVX-512 kernels (F and BW), 32 int16 lanes. They mirror `avx2.rs`: each is a safe
//! `target_feature` function whose caller (the dispatch in `simd.rs`) proves AVX-512F and BW are
//! present, and the only `unsafe` is the unaligned load and store helpers, whose array arguments
//! prove the bounds. Values are int16 with wraparound, as the scalar code. Each kernel handles
//! whole blocks of 32 and returns how many elements it did (`pack64`: pixels; `rgba_to_ycbcr`:
//! blocks of 16 pixels; `unpack`, which also needs GFNI: blocks of 64 byte positions).

use std::arch::x86_64::*;

const LANES: usize = 32;

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
    #[target_feature(enable = "avx512f")]
    pub(super) fn ld(s: &[i16; LANES]) -> __m512i {
        // SAFETY: `s` is 32 int16 (64 bytes), which the array type proves; `loadu` needs no alignment.
        unsafe { _mm512_loadu_si512(s.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st(s: &mut [i16; LANES], v: __m512i) {
        // SAFETY: `s` is 32 int16 (64 bytes), which the array type proves; `storeu` needs no alignment.
        unsafe { _mm512_storeu_si512(s.as_mut_ptr().cast(), v) }
    }

    /// Stores `a` then `b`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st2(s: &mut [i16; 2 * LANES], a: __m512i, b: __m512i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 64 int16 (128 bytes), which the array type proves; both 64-byte stores are
        // inside it; `storeu` needs no alignment.
        unsafe {
            _mm512_storeu_si512(p.cast(), a);
            _mm512_storeu_si512(p.add(LANES).cast(), b);
        }
    }

    /// Stores `a`, `b`, then `c`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st3(s: &mut [i16; 3 * LANES], a: __m512i, b: __m512i, c: __m512i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 96 int16 (192 bytes), which the array type proves; the three 64-byte
        // stores are inside it; `storeu` needs no alignment.
        unsafe {
            _mm512_storeu_si512(p.cast(), a);
            _mm512_storeu_si512(p.add(LANES).cast(), b);
            _mm512_storeu_si512(p.add(2 * LANES).cast(), c);
        }
    }

    /// Stores `a` then `b`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st_bytes(s: &mut [u8; 128], a: __m512i, b: __m512i) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 128 bytes, which the array type proves; both 64-byte stores are inside it;
        // `storeu` needs no alignment.
        unsafe {
            _mm512_storeu_si512(p.cast(), a);
            _mm512_storeu_si512(p.add(64).cast(), b);
        }
    }

    /// Loads 64 bytes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn ld_bytes(s: &[u8; 64]) -> __m512i {
        // SAFETY: `s` is 64 bytes, which the array type proves; `loadu` needs no alignment.
        unsafe { _mm512_loadu_si512(s.as_ptr().cast()) }
    }

    /// Loads three vectors of 32 int16.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn ld3(s: &[i16; 3 * LANES]) -> (__m512i, __m512i, __m512i) {
        let p = s.as_ptr();
        // SAFETY: `s` is 96 int16 (192 bytes), which the array type proves; the three 64-byte
        // loads are inside it; `loadu` needs no alignment.
        unsafe {
            (
                _mm512_loadu_si512(p.cast()),
                _mm512_loadu_si512(p.add(LANES).cast()),
                _mm512_loadu_si512(p.add(2 * LANES).cast()),
            )
        }
    }

    /// Stores `a`, `b`, `c`, then `d`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st4_u16(
        s: &mut [u16; 4 * LANES],
        a: __m512i,
        b: __m512i,
        c: __m512i,
        d: __m512i,
    ) {
        let p = s.as_mut_ptr();
        // SAFETY: `s` is 128 uint16 (256 bytes), which the array type proves; the four 64-byte
        // stores are inside it; `storeu` needs no alignment.
        unsafe {
            _mm512_storeu_si512(p.cast(), a);
            _mm512_storeu_si512(p.add(LANES).cast(), b);
            _mm512_storeu_si512(p.add(2 * LANES).cast(), c);
            _mm512_storeu_si512(p.add(3 * LANES).cast(), d);
        }
    }

    /// Stores 16 bytes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) fn st16(s: &mut [u8; 16], v: __m128i) {
        // SAFETY: `s` is 16 bytes, which the array type proves; `storeu` needs no alignment.
        unsafe { _mm_storeu_si128(s.as_mut_ptr().cast(), v) }
    }
}

use mem::{ld, ld_bytes, ld3, st, st_bytes, st2, st3, st4_u16, st16};

/// `floor((a + b) / 2)` per lane.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn avg(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(
        _mm512_and_si512(a, b),
        _mm512_srai_epi16::<1>(_mm512_xor_si512(a, b)),
    )
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn lift_even(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) -> usize {
    let mut n = 0;
    let (out, h) = (out.as_chunks_mut::<LANES>().0, h.as_chunks::<LANES>().0);
    let (o0, o1) = (o0.as_chunks::<LANES>().0, o1.as_chunks::<LANES>().0);
    for (((out, h), o0), o1) in out.iter_mut().zip(h).zip(o0).zip(o1) {
        st(out, _mm512_add_epi16(ld(h), avg(ld(o0), ld(o1))));
        n += 1;
    }
    n * LANES
}

/// `[31 + i]`: with `permutex2var(prev, SHIFT, d)`, lane 0 is `prev[31]` and lane `i` is `d[i - 1]`.
const SHIFT: [i16; LANES] = {
    let mut t = [0; LANES];
    let mut i = 0;
    while i < LANES {
        t[i] = 31 + i as i16;
        i += 1;
    }
    t
};

/// Indices into `(e, d)` that interleave them: output half `h` (0 or 1) holds `e[16h + i]` at
/// lane `2i` and `d[16h + i]` (index 32 and up) at lane `2i + 1`.
const WEAVE: [[i16; LANES]; 2] = {
    let mut t = [[0; LANES]; 2];
    let mut h = 0;
    while h < 2 {
        let mut i = 0;
        while i < LANES / 2 {
            t[h][2 * i] = (16 * h + i) as i16;
            t[h][2 * i + 1] = (32 + 16 * h + i) as i16;
            i += 1;
        }
        h += 1;
    }
    t
};

/// The 1D inverse lift of `idwt_row` for whole blocks of 32 positions, in registers: the detail
/// `d[k]` of one block, shifted by one lane against the last block's, gives `d[k-1]`. A block
/// needs `h[k + 32]`, so it runs while `k + 33 <= n`. Returns the first position not done.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn idwt_row(out: &mut [i16], l: &[i16], h: &[i16]) -> usize {
    let n = l.len().min(h.len()).min(out.len() / 2);
    if n < LANES + 1 {
        return 0;
    }
    // d[-1] = d[0].
    let d0 = (i32::from(l[0]) - ((i32::from(h[0]) + i32::from(h[1]) + 2) >> 2)) as i16;
    let mut prev = _mm512_set1_epi16(d0);
    let shift = ld(&SHIFT);
    let (w0, w1) = (ld(&WEAVE[0]), ld(&WEAVE[1]));
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
        let d = _mm512_sub_epi16(ld(lk), _mm512_sub_epi16(t, _mm512_srai_epi16::<1>(t)));
        // [prev[31], d[0..31]]
        let dp = _mm512_permutex2var_epi16(prev, shift, d);
        let e = _mm512_add_epi16(h0, avg(dp, d));
        st2(
            o,
            _mm512_permutex2var_epi16(e, w0, d),
            _mm512_permutex2var_epi16(e, w1, d),
        );
        prev = d;
        k += LANES;
    }
    k
}

/// `(r, g, b)` of YCoCg, wrapping int16.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn ycocg(y: __m512i, co: __m512i, cg: __m512i) -> (__m512i, __m512i, __m512i) {
    let t = _mm512_sub_epi16(y, _mm512_srai_epi16::<1>(cg));
    let g = _mm512_add_epi16(t, cg);
    let b = _mm512_sub_epi16(t, _mm512_srai_epi16::<1>(co));
    let r = _mm512_add_epi16(b, co);
    (r, g, b)
}

#[target_feature(enable = "avx512f,avx512bw")]
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
    let alpha = _mm512_set1_epi16(255);
    // After the per-128-bit-lane interleave, `p0` holds pixels 8j..8j+4 and `p1` pixels
    // 8j+4..8j+8 in lane j; these pick the qwords (index 8 and up: `p1`) of pixels 0..16 and
    // 16..32.
    let first = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    let second = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    for (((d, y), co), cg) in dst.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(ld(y), ld(co), ld(cg));
        let (r, g, b) = (
            _mm512_sra_epi16(r, sh),
            _mm512_sra_epi16(g, sh),
            _mm512_sra_epi16(b, sh),
        );
        // packus saturates to [0, 255]; then interleave bytes r g b a per pixel.
        let rb = _mm512_packus_epi16(r, b);
        let ga = _mm512_packus_epi16(g, alpha);
        let rg = _mm512_unpacklo_epi8(rb, ga);
        let ba = _mm512_unpackhi_epi8(rb, ga);
        let p0 = _mm512_unpacklo_epi16(rg, ba);
        let p1 = _mm512_unpackhi_epi16(rg, ba);
        st_bytes(
            d,
            _mm512_permutex2var_epi64(p0, first, p1),
            _mm512_permutex2var_epi64(p0, second, p1),
        );
        n += 1;
    }
    n * LANES
}

/// For output vector `v` (of 3, 32 int16 each), lane `l`: position `p = 32v + l` of the R, G, B
/// triples is channel `p % 3` of pixel `p / 3`. `RG[v]` indexes `permutex2var(r, _, g)` for the R
/// and G lanes, `B[v]` indexes `b` for the B lanes, and `KB[v]` is the mask of the B lanes.
const fn triple(v: usize, l: usize) -> (usize, usize) {
    let p = LANES * v + l;
    (p / 3, p % 3)
}

const RG: [[i16; LANES]; 3] = {
    let mut t = [[0; LANES]; 3];
    let mut v = 0;
    while v < 3 {
        let mut l = 0;
        while l < LANES {
            let (px, c) = triple(v, l);
            t[v][l] = match c {
                0 => px as i16,
                1 => (LANES + px) as i16,
                _ => 0,
            };
            l += 1;
        }
        v += 1;
    }
    t
};

const B: [[i16; LANES]; 3] = {
    let mut t = [[0; LANES]; 3];
    let mut v = 0;
    while v < 3 {
        let mut l = 0;
        while l < LANES {
            t[v][l] = triple(v, l).0 as i16;
            l += 1;
        }
        v += 1;
    }
    t
};

const KB: [u32; 3] = {
    let mut t = [0; 3];
    let mut v = 0;
    while v < 3 {
        let mut l = 0;
        while l < LANES {
            if triple(v, l).1 == 2 {
                t[v] |= 1u32 << l;
            }
            l += 1;
        }
        v += 1;
    }
    t
};

/// `a, b, c` are Y, Co, Cg (or R, G, B if `planes`), clamped to `[lo, hi]` and written as
/// R, G, B triples.
#[target_feature(enable = "avx512f,avx512bw")]
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
    let (lo, hi) = (_mm512_set1_epi16(lo), _mm512_set1_epi16(hi));
    let (rg0, rg1, rg2) = (ld(&RG[0]), ld(&RG[1]), ld(&RG[2]));
    let (bi0, bi1, bi2) = (ld(&B[0]), ld(&B[1]), ld(&B[2]));
    for (((d, a), b), c) in dst.iter_mut().zip(a).zip(b).zip(c) {
        let (r, g, b) = if planes {
            (ld(a), ld(b), ld(c))
        } else {
            ycocg(ld(a), ld(b), ld(c))
        };
        let (r, g, b) = (clamp(r, lo, hi), clamp(g, lo, hi), clamp(b, lo, hi));
        let v0 = _mm512_permutex2var_epi16(r, rg0, g);
        let v1 = _mm512_permutex2var_epi16(r, rg1, g);
        let v2 = _mm512_permutex2var_epi16(r, rg2, g);
        st3(
            d,
            _mm512_mask_permutexvar_epi16(v0, KB[0], bi0, b),
            _mm512_mask_permutexvar_epi16(v1, KB[1], bi1, b),
            _mm512_mask_permutexvar_epi16(v2, KB[2], bi2, b),
        );
        n += 1;
    }
    n * LANES
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn clamp(v: __m512i, lo: __m512i, hi: __m512i) -> __m512i {
    _mm512_min_epi16(_mm512_max_epi16(v, lo), hi)
}

/// Indices for `permutex2var` that gather output vector `v` of `pack64` (pixels `8v..8v + 8`, lane
/// `l` = channel `l % 4` of pixel `8v + l / 4`) from the three input vectors, `a`, `b`, `c` for
/// `v` = 0 (from `a, b`), 1 (`a, b`), 2 (`b, c`) and 3 (`c, c`). The source index of a colour lane
/// is `24v + 3(l / 4) + l % 4`, less the first input's start (0, 0, 32, 64). The alpha lanes take
/// index 0 and are overwritten.
const PACK: [[i16; LANES]; 4] = {
    let mut t = [[0; LANES]; 4];
    let first = [0, 0, 32, 64];
    let mut v = 0;
    while v < 4 {
        let mut l = 0;
        while l < LANES {
            if l % 4 < 3 {
                t[v][l] = (24 * v + 3 * (l / 4) + l % 4 - first[v]) as i16;
            }
            l += 1;
        }
        v += 1;
    }
    t
};

/// Clamps to `[0, hi]` and replicates 8 bits to 16 as `v << s1 | v >> s2`, which is `v << 8 | v`
/// (`s1` 8, `s2` 0) or, for 9 bits, `v << 7 | v >> 2` (7, 2).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn replicate(v: __m512i, hi: __m512i, s1: __m128i, s2: __m128i) -> __m512i {
    let v = clamp(v, _mm512_setzero_si512(), hi);
    _mm512_or_si512(_mm512_sll_epi16(v, s1), _mm512_srl_epi16(v, s2))
}

/// `pack64` for whole blocks of 32 pixels, `bits` 8 or 9 (any other value: nothing done).
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn pack64(dst: &mut [u16], src: &[i16], bits: u32) -> usize {
    let (s1, s2) = match bits {
        8 => (8, 0),
        9 => (7, 2),
        _ => return 0,
    };
    let mut n = 0;
    let dst = dst.as_chunks_mut::<{ 4 * LANES }>().0;
    let src = src.as_chunks::<{ 3 * LANES }>().0;
    let hi = _mm512_set1_epi16((1i16 << bits) - 1);
    let (s1, s2) = (_mm_cvtsi32_si128(s1), _mm_cvtsi32_si128(s2));
    let (i0, i1, i2, i3) = (ld(&PACK[0]), ld(&PACK[1]), ld(&PACK[2]), ld(&PACK[3]));
    let alpha = _mm512_set1_epi16(-1);
    for (d, s) in dst.iter_mut().zip(src) {
        let (a, b, c) = ld3(s);
        let (a, b, c) = (
            replicate(a, hi, s1, s2),
            replicate(b, hi, s1, s2),
            replicate(c, hi, s1, s2),
        );
        // Lanes 3, 7, ...: the alpha of each pixel.
        const A: __mmask32 = 0x8888_8888;
        st4_u16(
            d,
            _mm512_mask_mov_epi16(_mm512_permutex2var_epi16(a, i0, b), A, alpha),
            _mm512_mask_mov_epi16(_mm512_permutex2var_epi16(a, i1, b), A, alpha),
            _mm512_mask_mov_epi16(_mm512_permutex2var_epi16(b, i2, c), A, alpha),
            _mm512_mask_mov_epi16(_mm512_permutex2var_epi16(c, i3, c), A, alpha),
        );
        n += 1;
    }
    n * LANES
}

/// `r * c[0] + g * c[1] + b * c[2] + bias` per int32 lane, wrapping.
#[inline]
#[target_feature(enable = "avx512f")]
fn dot(r: __m512i, g: __m512i, b: __m512i, c: [i32; 3], bias: i32) -> __m512i {
    let t = _mm512_add_epi32(
        _mm512_mullo_epi32(r, _mm512_set1_epi32(c[0])),
        _mm512_mullo_epi32(g, _mm512_set1_epi32(c[1])),
    );
    let t = _mm512_add_epi32(t, _mm512_mullo_epi32(b, _mm512_set1_epi32(c[2])));
    _mm512_add_epi32(t, _mm512_set1_epi32(bias))
}

/// `clamp(v >> 16, 0, 255)` as 16 bytes.
#[inline]
#[target_feature(enable = "avx512f")]
fn chroma(v: __m512i) -> __m128i {
    let v = _mm512_srai_epi32::<16>(v);
    let v = _mm512_min_epi32(
        _mm512_max_epi32(v, _mm512_setzero_si512()),
        _mm512_set1_epi32(255),
    );
    _mm512_cvtepi32_epi8(v)
}

/// `rgb_to_ycbcr` for whole blocks of 16 pixels.
#[target_feature(enable = "avx512f,avx512bw")]
pub(super) fn rgba_to_ycbcr(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], rgba: &[u8]) -> usize {
    let mut n = 0;
    let (y, cb, cr) = (
        y.as_chunks_mut::<16>().0,
        cb.as_chunks_mut::<16>().0,
        cr.as_chunks_mut::<16>().0,
    );
    let rgba = rgba.as_chunks::<64>().0;
    let ff = _mm512_set1_epi32(0xff);
    for (((y, cb), cr), px) in y.iter_mut().zip(cb).zip(cr).zip(rgba) {
        let p = ld_bytes(px);
        // Pixel lanes are R, G, B, A bytes from the lowest.
        let r = _mm512_and_si512(p, ff);
        let g = _mm512_and_si512(_mm512_srli_epi32::<8>(p), ff);
        let b = _mm512_and_si512(_mm512_srli_epi32::<16>(p), ff);
        // Y is below 2^24 and the sums of the chroma fit int32, so no wraparound happens.
        let yy = dot(r, g, b, [19595, 38470, 7471], 1 << 15);
        st16(y, _mm512_cvtepi32_epi8(_mm512_srli_epi32::<16>(yy)));
        st16(cb, chroma(dot(r, g, b, [-11056, -21712, 32768], 257 << 15)));
        st16(cr, chroma(dot(r, g, b, [32768, -27440, -5328], 257 << 15)));
        n += 1;
    }
    n * 16
}

static ZERO: [u8; 64] = [0; 64];

/// Eight planes' bytes at 64 positions to bit-transposed qwords: qword `2l + q` of result `m` is
/// `transpose8` of the planes' bytes at position `16l + 2m + q`, which has coefficient `i`'s eight
/// bits for these planes in byte `i`. `r[k]` is plane `k`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn half(r: [&[u8; 64]; 8]) -> [__m512i; 8] {
    // Planes 7 down to 0, so that byte p of a position's qword is plane 7 - p.
    let [r0, r1, r2, r3, r4, r5, r6, r7] = r;
    let (x0, x1, x2, x3) = (ld_bytes(r7), ld_bytes(r6), ld_bytes(r5), ld_bytes(r4));
    let (x4, x5, x6, x7) = (ld_bytes(r3), ld_bytes(r2), ld_bytes(r1), ld_bytes(r0));
    // Per 128-bit lane (16 positions): interleave bytes, then words, then dwords, which leaves
    // each position's eight plane bytes in one qword.
    let (a0, a1) = (_mm512_unpacklo_epi8(x0, x1), _mm512_unpackhi_epi8(x0, x1));
    let (b0, b1) = (_mm512_unpacklo_epi8(x2, x3), _mm512_unpackhi_epi8(x2, x3));
    let (c0, c1) = (_mm512_unpacklo_epi8(x4, x5), _mm512_unpackhi_epi8(x4, x5));
    let (d0, d1) = (_mm512_unpacklo_epi8(x6, x7), _mm512_unpackhi_epi8(x6, x7));
    let (f0, f1) = (_mm512_unpacklo_epi16(a0, b0), _mm512_unpackhi_epi16(a0, b0));
    let (f2, f3) = (_mm512_unpacklo_epi16(a1, b1), _mm512_unpackhi_epi16(a1, b1));
    let (e0, e1) = (_mm512_unpacklo_epi16(c0, d0), _mm512_unpackhi_epi16(c0, d0));
    let (e2, e3) = (_mm512_unpacklo_epi16(c1, d1), _mm512_unpackhi_epi16(c1, d1));
    // Byte b of the affine result, bit r, is bit b of byte 7 - r of the qword, which is plane r.
    let sel = _mm512_set1_epi64(0x8040_2010_0804_0201u64 as i64);
    [
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpacklo_epi32(f0, e0)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpackhi_epi32(f0, e0)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpacklo_epi32(f1, e1)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpackhi_epi32(f1, e1)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpacklo_epi32(f2, e2)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpackhi_epi32(f2, e2)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpacklo_epi32(f3, e3)),
        _mm512_gf2p8affine_epi64_epi8::<0>(sel, _mm512_unpackhi_epi32(f3, e3)),
    ]
}

/// Writes the two rows of coefficients `c` of position `j`, as `unpack` does; `None`, with
/// possibly the first row written, if either does not fit in `d`.
fn put(d: &mut [i16], bw: usize, blk: &[i32], j: usize, c: &[i16; 8]) -> Option<()> {
    let base = usize::try_from(*blk.get(j >> 1)?).ok()?;
    let base = if j & 1 == 1 {
        base.checked_add(bw.checked_mul(2)?)?
    } else {
        base
    };
    let [c0, c1, c2, c3, c4, c5, c6, c7] = *c;
    *at_mut::<i16, 4>(d, base)? = [c0, c1, c2, c3];
    *at_mut::<i16, 4>(d, base.checked_add(bw)?)? = [c4, c5, c6, c7];
    Some(())
}

/// The Hulsken `unpack` for whole blocks of 64 positions `j`: transposes the 16 bit planes `p`
/// (`lo`: any of planes 0 to 7 is not all zero, `hi`: of 8 to 15) to the coefficients of each
/// position, applies the bit 15 sign and writes rows `blk[j >> 1] + (j & 1) * 2 * bw` and `bw`
/// after it, skipping the positions whose planes' bytes are all zero. Stops at the first block
/// with a row that does not fit in `d` or a `blk` entry that is missing or negative, and
/// returns the positions of the blocks done (its own block may be partly written; scalar redoes it).
#[target_feature(enable = "avx512f,avx512bw,gfni")]
pub(super) fn unpack(
    d: &mut [i16],
    bw: usize,
    blk: &[i32],
    p: &[&[u8]; 16],
    lo: bool,
    hi: bool,
) -> usize {
    let c: [&[[u8; 64]]; 16] = (*p).map(|s| s.as_chunks::<64>().0);
    let blocks = c.iter().map(|s| s.len()).min().unwrap_or(0);
    let zero = _mm512_setzero_si512();
    let mag = _mm512_set1_epi16(0x7fff);
    for b in 0..blocks {
        let mut rows = [&ZERO; 16];
        for (r, s) in rows.iter_mut().zip(&c) {
            let Some(x) = s.get(b) else {
                return 64 * b;
            };
            *r = x;
        }
        let [
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
        ] = rows;
        let tl = if lo {
            half([l0, l1, l2, l3, l4, l5, l6, l7])
        } else {
            [zero; 8]
        };
        let th = if hi {
            half([h0, h1, h2, h3, h4, h5, h6, h7])
        } else {
            [zero; 8]
        };
        for (m, (&l, &h)) in tl.iter().zip(&th).enumerate() {
            // Bit 2l + q: position 16l + 2m + q has a nonzero plane byte.
            let any = _mm512_or_si512(l, h);
            let nz = _mm512_test_epi64_mask(any, any);
            if nz == 0 {
                continue;
            }
            for q in 0..2 {
                // Coefficient i of a position is lane i of its 128 bits: byte i of `l` below byte i of `h`.
                let v = if q == 0 {
                    _mm512_unpacklo_epi8(l, h)
                } else {
                    _mm512_unpackhi_epi8(l, h)
                };
                // Lanes with bit 15 set become -(v & 0x7fff).
                let v = _mm512_mask_sub_epi16(
                    v,
                    _mm512_movepi16_mask(v),
                    zero,
                    _mm512_and_si512(v, mag),
                );
                let mut buf = [0i16; LANES];
                st(&mut buf, v);
                for (lane, row) in buf.as_chunks::<8>().0.iter().enumerate() {
                    if (nz >> (2 * lane + q)) & 1 == 0 {
                        continue;
                    }
                    let j = 64 * b + 16 * lane + 2 * m + q;
                    if put(d, bw, blk, j, row).is_none() {
                        return 64 * b;
                    }
                }
            }
        }
    }
    64 * blocks
}
