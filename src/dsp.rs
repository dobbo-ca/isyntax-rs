//! The sample-level stages of a decode: the inverse LeGall 5/3 wavelet transform, YCoCg to RGB and
//! RGB to YCbCr. Port of `internal/dsp`: the `*_scalar` functions are its `*Go` functions, the
//! oracle and the tail of every SIMD kernel (`simd.rs`, AVX-512, AVX2 and NEON, whole lanes
//! only; the functions without a suffix dispatch, and those with `_on` take the tier). Arithmetic
//! is int32 with every output stored as int16 before later steps read it, so int16 wraps as in Go.

use crate::simd::{self, Tier};

/// Inverts one level of the 2D LeGall 5/3 transform in place. `buf` is 2n x 2n with the quadrants
/// LL | HL over LH | HH (`n >= 2`); `tmp` is scratch of the same size. Rows go first, then
/// columns, each with the lifting in isyntax-go's docs/format.md.
pub(crate) fn inverse53(buf: &mut [i16], tmp: &mut [i16], n: usize) {
    let w = 2 * n;
    let buf = &mut buf[..w * w];
    let tmp = &mut tmp[..w * w];
    for y in 0..w {
        let row = &buf[y * w..(y + 1) * w];
        idwt_row(&mut tmp[y * w..(y + 1) * w], &row[..n], &row[n..]);
    }
    // Columns: the same lift, transposed.
    let l = |k: usize| k * w..(k + 1) * w;
    let h = |k: usize| (n + k) * w..(n + k + 1) * w;
    for k in 0..n - 1 {
        lift_odd(
            &mut buf[l(2 * k + 1)],
            &tmp[l(k)],
            &tmp[h(k)],
            &tmp[h(k + 1)],
        );
    }
    lift_odd(
        &mut buf[l(2 * n - 1)],
        &tmp[l(n - 1)],
        &tmp[h(n - 1)],
        &tmp[h(n - 1)],
    );
    for k in 0..n {
        // Row 2k reads the odd rows around it, which are final.
        let (head, tail) = buf.split_at_mut(2 * k * w);
        let (out, rest) = tail.split_at_mut(w);
        let o1 = &rest[..w];
        let o0 = if k == 0 { o1 } else { &head[l(2 * k - 1)] };
        lift_even(out, &tmp[h(k)], o0, o1);
    }
}

/// The 1D inverse lift of one row (`l.len() >= 2`).
pub(crate) fn idwt_row(out: &mut [i16], l: &[i16], h: &[i16]) {
    idwt_row_on(simd::tier(), out, l, h);
}

pub(crate) fn idwt_row_on(t: Tier, out: &mut [i16], l: &[i16], h: &[i16]) {
    let k0 = simd::idwt_row(t, out, l, h);
    idwt_row_from(out, l, h, k0);
}

/// `idwt_row` from position `k0` on, given that the positions before it are done.
pub(crate) fn idwt_row_from(out: &mut [i16], l: &[i16], h: &[i16], k0: usize) {
    let n = l.len();
    let h = &h[..n];
    let out = &mut out[..2 * n];
    let hv = |k: usize| i32::from(h[k]);
    let d = |k: usize, hn: i32| (i32::from(l[k]) - ((hv(k) + hn + 2) >> 2)) as i16 as i32;
    let mut k = k0;
    // d[-1] = d[0].
    let mut dc = if k0 == 0 {
        let dc = d(0, hv(1));
        out[0] = (hv(0) + dc) as i16;
        k = 1;
        dc
    } else {
        d(k0 - 1, hv(k0))
    };
    while k < n - 1 {
        let dn = d(k, hv(k + 1));
        out[2 * k - 1] = dc as i16;
        out[2 * k] = (hv(k) + ((dc + dn) >> 1)) as i16;
        dc = dn;
        k += 1;
    }
    let dn = (i32::from(l[n - 1]) - ((hv(n - 1) + 1) >> 1)) as i16 as i32;
    out[2 * n - 3] = dc as i16;
    out[2 * n - 2] = (hv(n - 1) + ((dc + dn) >> 1)) as i16;
    out[2 * n - 1] = dn as i16;
}

/// `out = l - ((h0 + h1 + 2) >> 2)`.
pub(crate) fn lift_odd(out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) {
    lift_odd_on(simd::tier(), out, l, h0, h1);
}

pub(crate) fn lift_odd_on(t: Tier, out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) {
    let m = simd::lift_odd(t, out, l, h0, h1);
    lift_odd_scalar(&mut out[m..], &l[m..], &h0[m..], &h1[m..]);
}

pub(crate) fn lift_odd_scalar(out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) {
    for (((o, &l), &a), &b) in out.iter_mut().zip(l).zip(h0).zip(h1) {
        *o = (i32::from(l) - ((i32::from(a) + i32::from(b) + 2) >> 2)) as i16;
    }
}

/// `out = h + ((o0 + o1) >> 1)`.
pub(crate) fn lift_even(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) {
    lift_even_on(simd::tier(), out, h, o0, o1);
}

pub(crate) fn lift_even_on(t: Tier, out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) {
    let m = simd::lift_even(t, out, h, o0, o1);
    lift_even_scalar(&mut out[m..], &h[m..], &o0[m..], &o1[m..]);
}

pub(crate) fn lift_even_scalar(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) {
    for (((o, &h), &a), &b) in out.iter_mut().zip(h).zip(o0).zip(o1) {
        *o = (i32::from(h) + ((i32::from(a) + i32::from(b)) >> 1)) as i16;
    }
}

/// YCoCg to R, G, B in int16 with wrapping, unshifted and unclamped (docs/format.md, "Colour
/// transform").
fn ycocg(y: i16, co: i16, cg: i16) -> (i16, i16, i16) {
    let t = y.wrapping_sub(cg >> 1);
    let g = t.wrapping_add(cg);
    let b = t.wrapping_sub(co >> 1);
    let r = b.wrapping_add(co);
    (r, g, b)
}

fn clamp(v: i16) -> u8 {
    if (v as u16) > 255 {
        (!(v >> 15)) as u8 // 0 below, 255 above
    } else {
        v as u8
    }
}

/// `dst` is 4 bytes per pixel: R, G, B shifted right by `shift` and clamped, then 255.
pub(crate) fn ycocg_to_rgba(dst: &mut [u8], y: &[i16], co: &[i16], cg: &[i16], shift: u32) {
    ycocg_to_rgba_on(simd::tier(), dst, y, co, cg, shift);
}

pub(crate) fn ycocg_to_rgba_on(
    t: Tier,
    dst: &mut [u8],
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    shift: u32,
) {
    let m = simd::ycocg_to_rgba(t, dst, y, co, cg, shift);
    ycocg_to_rgba_scalar(&mut dst[4 * m..], &y[m..], &co[m..], &cg[m..], shift);
}

pub(crate) fn ycocg_to_rgba_scalar(dst: &mut [u8], y: &[i16], co: &[i16], cg: &[i16], shift: u32) {
    let shift = shift & 15;
    for (((d, &yv), &c), &g) in dst.as_chunks_mut::<4>().0.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(yv, c, g);
        d[0] = clamp(r >> shift);
        d[1] = clamp(g >> shift);
        d[2] = clamp(b >> shift);
        d[3] = 255;
    }
}

/// `ycocg_to_rgba` for files whose colour planes are already R, G, B: shift and clamp only.
pub(crate) fn planes_to_rgba(dst: &mut [u8], r: &[i16], g: &[i16], b: &[i16], shift: u32) {
    let shift = shift & 15;
    for (((d, &r), &g), &b) in dst.as_chunks_mut::<4>().0.iter_mut().zip(r).zip(g).zip(b) {
        d[0] = clamp(r >> shift);
        d[1] = clamp(g >> shift);
        d[2] = clamp(b >> shift);
        d[3] = 255;
    }
}

/// `dst` is 3 int16 per pixel: the signed, unshifted, unclamped R, G, B.
pub(crate) fn ycocg_to_rgb16(dst: &mut [i16], y: &[i16], co: &[i16], cg: &[i16]) {
    ycocg_to_rgb16_on(simd::tier(), dst, y, co, cg);
}

pub(crate) fn ycocg_to_rgb16_on(t: Tier, dst: &mut [i16], y: &[i16], co: &[i16], cg: &[i16]) {
    let m = simd::samples(t, dst, y, co, cg, i16::MIN, i16::MAX, false);
    ycocg_to_rgb16_scalar(&mut dst[3 * m..], &y[m..], &co[m..], &cg[m..]);
}

pub(crate) fn ycocg_to_rgb16_scalar(dst: &mut [i16], y: &[i16], co: &[i16], cg: &[i16]) {
    for (((d, &yv), &c), &g) in dst.as_chunks_mut::<3>().0.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(yv, c, g);
        *d = [r, g, b];
    }
}

/// `ycocg_to_rgb16` for R, G, B planes: interleaves them unchanged.
pub(crate) fn planes_to_rgb16(dst: &mut [i16], r: &[i16], g: &[i16], b: &[i16]) {
    planes_to_rgb16_on(simd::tier(), dst, r, g, b);
}

pub(crate) fn planes_to_rgb16_on(t: Tier, dst: &mut [i16], r: &[i16], g: &[i16], b: &[i16]) {
    let m = simd::samples(t, dst, r, g, b, i16::MIN, i16::MAX, true);
    planes_to_rgb16_scalar(&mut dst[3 * m..], &r[m..], &g[m..], &b[m..]);
}

pub(crate) fn planes_to_rgb16_scalar(dst: &mut [i16], r: &[i16], g: &[i16], b: &[i16]) {
    for (((d, &r), &g), &b) in dst.as_chunks_mut::<3>().0.iter_mut().zip(r).zip(g).zip(b) {
        *d = [r, g, b];
    }
}

/// `ycocg_to_rgb16`, each value clamped to `[0, hi]`.
pub(crate) fn ycocg_to_samples(dst: &mut [i16], y: &[i16], co: &[i16], cg: &[i16], hi: i16) {
    ycocg_to_samples_on(simd::tier(), dst, y, co, cg, hi);
}

pub(crate) fn ycocg_to_samples_on(
    t: Tier,
    dst: &mut [i16],
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    hi: i16,
) {
    let m = simd::samples(t, dst, y, co, cg, 0, hi, false);
    ycocg_to_samples_scalar(&mut dst[3 * m..], &y[m..], &co[m..], &cg[m..], hi);
}

pub(crate) fn ycocg_to_samples_scalar(dst: &mut [i16], y: &[i16], co: &[i16], cg: &[i16], hi: i16) {
    for (((d, &yv), &c), &g) in dst.as_chunks_mut::<3>().0.iter_mut().zip(y).zip(co).zip(cg) {
        let (r, g, b) = ycocg(yv, c, g);
        *d = [r.clamp(0, hi), g.clamp(0, hi), b.clamp(0, hi)];
    }
}

/// `planes_to_rgb16`, each value clamped to `[0, hi]`.
pub(crate) fn planes_to_samples(dst: &mut [i16], r: &[i16], g: &[i16], b: &[i16], hi: i16) {
    planes_to_samples_on(simd::tier(), dst, r, g, b, hi);
}

pub(crate) fn planes_to_samples_on(
    t: Tier,
    dst: &mut [i16],
    r: &[i16],
    g: &[i16],
    b: &[i16],
    hi: i16,
) {
    let m = simd::samples(t, dst, r, g, b, 0, hi, true);
    planes_to_samples_scalar(&mut dst[3 * m..], &r[m..], &g[m..], &b[m..], hi);
}

pub(crate) fn planes_to_samples_scalar(dst: &mut [i16], r: &[i16], g: &[i16], b: &[i16], hi: i16) {
    for (((d, &r), &g), &b) in dst.as_chunks_mut::<3>().0.iter_mut().zip(r).zip(g).zip(b) {
        *d = [r.clamp(0, hi), g.clamp(0, hi), b.clamp(0, hi)];
    }
}

/// Go's `color.RGBToYCbCr` (full-range BT.601).
pub(crate) fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (r1, g1, b1) = (i32::from(r), i32::from(g), i32::from(b));
    let yy = (19595 * r1 + 38470 * g1 + 7471 * b1 + (1 << 15)) >> 16;
    let mut cb = -11056 * r1 - 21712 * g1 + 32768 * b1 + (257 << 15);
    if (cb as u32) & 0xff00_0000 == 0 {
        cb >>= 16;
    } else {
        cb = !(cb >> 31);
    }
    let mut cr = 32768 * r1 - 27440 * g1 - 5328 * b1 + (257 << 15);
    if (cr as u32) & 0xff00_0000 == 0 {
        cr >>= 16;
    } else {
        cr = !(cr >> 31);
    }
    (yy as u8, cb as u8, cr as u8)
}

/// Converts `y.len()` RGBA pixels with `color.RGBToYCbCr`.
pub(crate) fn rgba_to_ycbcr(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], rgba: &[u8]) {
    rgba_to_ycbcr_on(simd::tier(), y, cb, cr, rgba);
}

pub(crate) fn rgba_to_ycbcr_on(t: Tier, y: &mut [u8], cb: &mut [u8], cr: &mut [u8], rgba: &[u8]) {
    let m = simd::rgba_to_ycbcr(t, y, cb, cr, rgba);
    rgba_to_ycbcr_scalar(&mut y[m..], &mut cb[m..], &mut cr[m..], &rgba[4 * m..]);
}

pub(crate) fn rgba_to_ycbcr_scalar(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], rgba: &[u8]) {
    for (((y, cb), cr), px) in y
        .iter_mut()
        .zip(cb.iter_mut())
        .zip(cr.iter_mut())
        .zip(rgba.as_chunks::<4>().0)
    {
        (*y, *cb, *cr) = rgb_to_ycbcr(px[0], px[1], px[2]);
    }
}

/// `|v| &^ mask`, plus `o` if nonzero, sign restored; int16 wraparound.
pub(crate) fn trunc_row(row: &mut [i16], mask: i16, o: i16) {
    trunc_row_on(simd::tier(), row, mask, o);
}

pub(crate) fn trunc_row_on(t: Tier, row: &mut [i16], mask: i16, o: i16) {
    let m = simd::trunc_row(t, row, mask, o);
    trunc_row_scalar(&mut row[m..], mask, o);
}

pub(crate) fn trunc_row_scalar(row: &mut [i16], mask: i16, o: i16) {
    for v in row {
        let mut a = (*v).max(v.wrapping_neg()) & !mask;
        if a != 0 {
            a = a.wrapping_add(o);
        }
        if *v < 0 {
            a = a.wrapping_neg();
        }
        *v = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    /// docs/format.md's lifting, written literally.
    fn ref53(input: &[i16]) -> Vec<i16> {
        let n = input.len() / 2;
        let l = |k: usize| i32::from(input[k]);
        let h = |k: usize| i32::from(input[n + k]);
        let mut out = vec![0i16; 2 * n];
        for k in 0..=n - 2 {
            out[2 * k + 1] = (l(k) - ((h(k) + h(k + 1) + 2) >> 2)) as i16;
        }
        out[2 * n - 1] = (l(n - 1) - ((h(n - 1) + 1) >> 1)) as i16;
        out[0] = (h(0) + i32::from(out[1])) as i16;
        for k in 1..n {
            out[2 * k] =
                (h(k) + ((i32::from(out[2 * k - 1]) + i32::from(out[2 * k + 1])) >> 1)) as i16;
        }
        out
    }

    fn ref2d(buf: &[i16], n: usize) -> Vec<i16> {
        let w = 2 * n;
        let mut a = buf.to_vec();
        for y in 0..w {
            let r = ref53(&a[y * w..(y + 1) * w]);
            a[y * w..(y + 1) * w].copy_from_slice(&r);
        }
        for x in 0..w {
            let col: Vec<i16> = (0..w).map(|y| a[y * w + x]).collect();
            for (y, v) in ref53(&col).into_iter().enumerate() {
                a[y * w + x] = v;
            }
        }
        a
    }

    #[test]
    fn inverse53_matches_the_literal_lifting() {
        let mut r = Rng::new(1);
        for n in [2usize, 3, 8, 136] {
            for span in [16u64, 600, 1 << 16] {
                // 1 << 16 wraps int16.
                let w = 2 * n;
                let buf: Vec<i16> = (0..w * w)
                    .map(|_| (r.below(span) as i64 - (span / 2) as i64) as i16)
                    .collect();
                let want = ref2d(&buf, n);
                let mut got = buf.clone();
                let mut tmp = vec![0i16; w * w];
                inverse53(&mut got, &mut tmp, n);
                assert_eq!(got, want, "n={n} span={span}");
            }
        }
    }

    /// docs/format.md's transform, 16-bit wrapping math.
    fn ref_rgb(y: i16, co: i16, cg: i16, shift: u32) -> [u8; 3] {
        let t = y.wrapping_sub(cg >> 1);
        let g = t.wrapping_add(cg);
        let b = t.wrapping_sub(co >> 1);
        let r = b.wrapping_add(co);
        let c = |v: i16| (v >> shift).clamp(0, 255) as u8;
        [c(r), c(g), c(b)]
    }

    #[test]
    fn ycocg_cases() {
        // (y, co, cg, shift, want): signed Y, arithmetic shifts floor.
        for (y, co, cg, shift, want) in [
            (200i16, 0i16, 0i16, 0u32, [200u8, 200, 200]),
            (100, 50, -20, 0, [135, 90, 85]),
            (100, -51, 21, 0, [65, 111, 116]),
            (300, 0, 0, 0, [255, 255, 255]),
            (10, -40, -40, 0, [10, 0, 50]),
            (510, 0, 0, 1, [255, 255, 255]),
            (255, 0, 0, 1, [127, 127, 127]),
            (32767, 32767, 0, 0, [0, 255, 255]), // R wraps negative
            (-200, 0, 0, 0, [0, 0, 0]),          // signed Y (rule 4)
            (-100, 50, -20, 0, [0, 0, 0]),
        ] {
            let mut dst = [0u8; 4];
            ycocg_to_rgba(&mut dst, &[y], &[co], &[cg], shift);
            assert_eq!(dst[..3], want, "{y} {co} {cg}");
            assert_eq!(dst[3], 255);
            assert_eq!(ref_rgb(y, co, cg, shift), want);
        }
        let mut rnd = Rng::new(2);
        let n = 1 << 16;
        let mut y = vec![0i16; n];
        let mut co = vec![0i16; n];
        let mut cg = vec![0i16; n];
        for i in 0..n {
            y[i] = rnd.next_u64() as i16;
            co[i] = rnd.next_u64() as i16;
            cg[i] = rnd.next_u64() as i16;
        }
        for shift in [0u32, 1] {
            let mut dst = vec![0u8; 4 * n];
            ycocg_to_rgba(&mut dst, &y, &co, &cg, shift);
            for i in 0..n {
                let want = ref_rgb(y[i], co[i], cg[i], shift);
                assert_eq!(dst[4 * i..4 * i + 3], want, "shift {shift} i {i}");
                assert_eq!(dst[4 * i + 3], 255);
            }
        }
    }

    #[test]
    fn rgba_to_ycbcr_matches_the_clamped_formula() {
        let n = 1 << 12;
        let mut rgba = vec![255u8; 4 * n];
        let (mut yy, mut cb, mut cr) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
        for base in (0..1 << 24).step_by(n * 11) {
            for i in 0..n {
                let v = base + i;
                rgba[4 * i] = (v >> 16) as u8;
                rgba[4 * i + 1] = (v >> 8) as u8;
                rgba[4 * i + 2] = v as u8;
            }
            rgba_to_ycbcr(&mut yy, &mut cb, &mut cr, &rgba);
            for i in 0..n {
                let (r, g, b) = (
                    i32::from(rgba[4 * i]),
                    i32::from(rgba[4 * i + 1]),
                    i32::from(rgba[4 * i + 2]),
                );
                // The integer form of the JFIF formulas, with the clamp spelled out.
                let y = (19595 * r + 38470 * g + 7471 * b + (1 << 15)) >> 16;
                let u = ((-11056 * r - 21712 * g + 32768 * b + (257 << 15)) >> 16).clamp(0, 255);
                let v = ((32768 * r - 27440 * g - 5328 * b + (257 << 15)) >> 16).clamp(0, 255);
                assert_eq!(
                    (i32::from(yy[i]), i32::from(cb[i]), i32::from(cr[i])),
                    (y, u, v),
                    "rgb {r} {g} {b}"
                );
            }
        }
    }

    #[test]
    fn planes() {
        let (r, g, b) = (
            [-1i16, 0, 200, 300, 511],
            [5i16, 255, 256, -300, 20],
            [0i16, 1, 2, 3, 4],
        );
        for shift in [0u32, 1] {
            let mut dst = vec![0u8; 4 * r.len()];
            planes_to_rgba(&mut dst, &r, &g, &b, shift);
            let c = |v: i16| (v >> shift).clamp(0, 255) as u8;
            for i in 0..r.len() {
                assert_eq!(dst[4 * i..4 * i + 4], [c(r[i]), c(g[i]), c(b[i]), 255]);
            }
        }
        let mut dst = vec![0i16; 3 * r.len()];
        planes_to_rgb16(&mut dst, &r, &g, &b);
        for i in 0..r.len() {
            assert_eq!(dst[3 * i..3 * i + 3], [r[i], g[i], b[i]]);
        }
    }

    #[test]
    fn samples_are_rgb16_clamped() {
        let mut rnd = Rng::new(9);
        let n = 1 << 12;
        let mut a: Vec<i16> = (0..n).map(|_| rnd.next_u64() as i16).collect();
        let mut b: Vec<i16> = (0..n).map(|_| rnd.next_u64() as i16).collect();
        let mut c: Vec<i16> = (0..n).map(|_| rnd.next_u64() as i16).collect();
        (a[0], b[0], c[0]) = (-32768, 32767, -1);
        let (mut raw, mut got) = (vec![0i16; 3 * n], vec![0i16; 3 * n]);
        for hi in [255i16, 511] {
            ycocg_to_rgb16(&mut raw, &a, &b, &c);
            ycocg_to_samples(&mut got, &a, &b, &c, hi);
            for (i, v) in raw.iter().enumerate() {
                assert_eq!(got[i], (*v).clamp(0, hi), "ycocg hi {hi} at {i}");
            }
            planes_to_rgb16(&mut raw, &a, &b, &c);
            planes_to_samples(&mut got, &a, &b, &c, hi);
            for (i, v) in raw.iter().enumerate() {
                assert_eq!(got[i], (*v).clamp(0, hi), "planes hi {hi} at {i}");
            }
        }
    }

    #[test]
    fn trunc_wraps() {
        let mut row = [-32768i16, 32767, -1, 0, 1];
        trunc_row(&mut row, 0, 0);
        assert_eq!(row, [-32768, 32767, -1, 0, 1]);
        let mut row = [0i16, 15, 16, -16, 31, 32, -47];
        trunc_row(&mut row, 15, 7);
        assert_eq!(row, [0, 0, 23, -23, 23, 39, -39]);
    }

    /// The tier a CPU with SIMD must pick (x86_64 CI runners have AVX2, some AVX-512; aarch64
    /// runners have NEON), or the one `ISYNTAX_SIMD` names.
    fn expected_tier() -> crate::simd::Tier {
        use crate::simd::Tier;
        #[cfg(target_arch = "x86_64")]
        let simd = if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            Tier::Avx512
        } else {
            Tier::Avx2
        };
        #[cfg(target_arch = "aarch64")]
        let simd = Tier::Neon;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let simd = Tier::Scalar;
        match std::env::var_os("ISYNTAX_SIMD") {
            Some(v) if v == "scalar" => Tier::Scalar,
            #[cfg(target_arch = "x86_64")]
            Some(v) if v == "avx2" => Tier::Avx2,
            _ => simd,
        }
    }

    /// Every tier of the CPU is in `available()`: x86_64 CI runners have AVX2 (AVX-512 F and BW on
    /// some), and aarch64 runners have NEON.
    #[test]
    fn every_tier_of_the_cpu_is_available() {
        let all = crate::simd::available();
        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::Tier;
            assert!(is_x86_feature_detected!("avx2"));
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                assert_eq!(all, [Tier::Scalar, Tier::Avx2, Tier::Avx512]);
            } else {
                assert_eq!(all, [Tier::Scalar, Tier::Avx2]);
            }
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(all, [crate::simd::Tier::Scalar, crate::simd::Tier::Neon]);
        assert!(all.contains(&crate::simd::tier()));
    }

    /// Also logs the tier in use (`--nocapture`), so a CI log shows what the run tested.
    #[test]
    fn the_simd_tier_is_detected() {
        eprintln!(
            "SIMD tier: {:?}, AVX-512 kernels: {}",
            crate::simd::tier(),
            crate::simd::avx512_kernels()
        );
        if std::env::var_os("ISYNTAX_SIMD").is_some_and(|v| v == "avx512") {
            assert!(
                crate::simd::avx512_kernels(),
                "ISYNTAX_SIMD=avx512, but no AVX-512 kernels"
            );
            #[cfg(target_arch = "x86_64")]
            assert_eq!(crate::simd::tier(), crate::simd::Tier::Avx512);
        }
        assert_eq!(crate::simd::tier(), expected_tier());
    }

    /// int16 values: full range, small, or the wrap and clamp edges.
    fn plane(r: &mut Rng, n: usize, mode: u64) -> Vec<i16> {
        const EDGE: [i16; 9] = [i16::MIN, i16::MIN + 1, -256, -1, 0, 1, 255, 511, i16::MAX];
        (0..n)
            .map(|_| match mode {
                0 => r.next_u64() as i16,
                1 => r.below(1200) as i16 - 400,
                _ => EDGE[r.below(9) as usize],
            })
            .collect()
    }

    /// Every kernel, dispatched against its scalar function: bit-exact on random and edge data,
    /// at lengths around the lane counts (0, 1, lane - 1, lane, lane + 1, ...) and unaligned starts.
    #[test]
    fn simd_kernels_match_scalar() {
        // Not scalar against scalar by accident.
        assert_eq!(crate::simd::tier(), expected_tier());
        for t in crate::simd::available() {
            let mut r = Rng::new(42);
            for n in [
                0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 18, 24, 31, 32, 33, 40, 63, 64, 65, 95, 96,
                97, 100, 129, 257,
            ] {
                for off in 0..3 {
                    for mode in 0..3 {
                        let k = n + off;
                        let (a, b, c, d) = (
                            plane(&mut r, k + 1, mode),
                            plane(&mut r, k + 1, mode),
                            plane(&mut r, k + 1, mode),
                            plane(&mut r, k + 1, mode),
                        );
                        let (a, b, c, d) = (&a[off..k], &b[off..k], &c[off..k], &d[off..k]);
                        let ctx = format!("tier={t:?} n={n} off={off} mode={mode}");
                        let (mut got, mut want) = (vec![0x5aa5i16; n], vec![0x5aa5i16; n]);
                        lift_odd_on(t, &mut got, a, b, c);
                        lift_odd_scalar(&mut want, a, b, c);
                        assert_eq!(got, want, "lift_odd {ctx}");
                        lift_even_on(t, &mut got, a, b, c);
                        lift_even_scalar(&mut want, a, b, c);
                        assert_eq!(got, want, "lift_even {ctx}");
                        if n >= 2 {
                            let (mut got, mut want) =
                                (vec![0x5aa5i16; 2 * n], vec![0x5aa5i16; 2 * n]);
                            idwt_row_on(t, &mut got, a, b);
                            idwt_row_from(&mut want, a, b, 0);
                            assert_eq!(got, want, "idwt_row {ctx}");
                        }
                        for shift in [0u32, 1, 5, 15] {
                            let (mut got, mut want) = (vec![0xa5u8; 4 * n], vec![0xa5u8; 4 * n]);
                            ycocg_to_rgba_on(t, &mut got, a, b, c, shift);
                            ycocg_to_rgba_scalar(&mut want, a, b, c, shift);
                            assert_eq!(got, want, "ycocg_to_rgba {ctx} shift={shift}");
                        }
                        let (mut got, mut want) = (vec![0x5aa5i16; 3 * n], vec![0x5aa5i16; 3 * n]);
                        ycocg_to_rgb16_on(t, &mut got, a, b, c);
                        ycocg_to_rgb16_scalar(&mut want, a, b, c);
                        assert_eq!(got, want, "ycocg_to_rgb16 {ctx}");
                        planes_to_rgb16_on(t, &mut got, a, b, c);
                        planes_to_rgb16_scalar(&mut want, a, b, c);
                        assert_eq!(got, want, "planes_to_rgb16 {ctx}");
                        for hi in [255i16, 511] {
                            ycocg_to_samples_on(t, &mut got, a, b, c, hi);
                            ycocg_to_samples_scalar(&mut want, a, b, c, hi);
                            assert_eq!(got, want, "ycocg_to_samples {ctx} hi={hi}");
                            planes_to_samples_on(t, &mut got, a, b, c, hi);
                            planes_to_samples_scalar(&mut want, a, b, c, hi);
                            assert_eq!(got, want, "planes_to_samples {ctx} hi={hi}");
                        }
                        for (mask, o) in [(0i16, 0i16), (15, 7), (255, -3), (-1, 5), (7, i16::MAX)]
                        {
                            let (mut got, mut want) = (d.to_vec(), d.to_vec());
                            trunc_row_on(t, &mut got, mask, o);
                            trunc_row_scalar(&mut want, mask, o);
                            assert_eq!(got, want, "trunc_row {ctx} mask={mask} o={o}");
                        }
                    }
                }
            }
        }
    }

    /// The same, with an output longer than the inputs and inputs of different lengths: the SIMD
    /// kernel takes the shortest, the tail runs in scalar, and the rest of the output is not touched.
    #[test]
    fn simd_kernels_match_scalar_on_unequal_lengths() {
        assert_eq!(crate::simd::tier(), expected_tier());
        for t in crate::simd::available() {
            let mut r = Rng::new(7);
            for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 100] {
                for (ea, eb, ec) in [(0, 1, 2), (3, 0, 17), (16, 5, 0), (1, 33, 8)] {
                    for mode in 0..3 {
                        let (a, b, c) = (
                            plane(&mut r, n + ea, mode),
                            plane(&mut r, n + eb, mode),
                            plane(&mut r, n + ec, mode),
                        );
                        let ctx = format!("tier={t:?} n={n} extra=({ea}, {eb}, {ec}) mode={mode}");
                        let (mut got, mut want) =
                            (vec![0x5aa5i16; n + 40], vec![0x5aa5i16; n + 40]);
                        lift_odd_on(t, &mut got, &a, &b, &c);
                        lift_odd_scalar(&mut want, &a, &b, &c);
                        assert_eq!(got, want, "lift_odd {ctx}");
                        lift_even_on(t, &mut got, &a, &b, &c);
                        lift_even_scalar(&mut want, &a, &b, &c);
                        assert_eq!(got, want, "lift_even {ctx}");
                        let (mut got, mut want) =
                            (vec![0xa5u8; 4 * n + 40], vec![0xa5u8; 4 * n + 40]);
                        ycocg_to_rgba_on(t, &mut got, &a, &b, &c, 3);
                        ycocg_to_rgba_scalar(&mut want, &a, &b, &c, 3);
                        assert_eq!(got, want, "ycocg_to_rgba {ctx}");
                        let (mut got, mut want) =
                            (vec![0x5aa5i16; 3 * n + 40], vec![0x5aa5i16; 3 * n + 40]);
                        ycocg_to_rgb16_on(t, &mut got, &a, &b, &c);
                        ycocg_to_rgb16_scalar(&mut want, &a, &b, &c);
                        assert_eq!(got, want, "ycocg_to_rgb16 {ctx}");
                        planes_to_samples_on(t, &mut got, &a, &b, &c, 255);
                        planes_to_samples_scalar(&mut want, &a, &b, &c, 255);
                        assert_eq!(got, want, "planes_to_samples {ctx}");
                        // idwt_row takes equal-length inputs; its output may be longer.
                        if n >= 2 {
                            let (mut got, mut want) =
                                (vec![0x5aa5i16; 2 * n + 40], vec![0x5aa5i16; 2 * n + 40]);
                            idwt_row_on(t, &mut got, &a[..n], &b[..n]);
                            idwt_row_from(&mut want, &a[..n], &b[..n], 0);
                            assert_eq!(got, want, "idwt_row {ctx}");
                        }
                    }
                }
            }
        }
    }

    /// `pack64` in every tier, with 8 and 9 bits, full-range, clamp-edge and small samples, whole
    /// pixels and a partial one, and an output longer than the pixels.
    #[test]
    fn pack64_matches_scalar() {
        use crate::region::{pack64_on, pack64_scalar};
        for t in crate::simd::available() {
            let mut r = Rng::new(21);
            for n in [0usize, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257] {
                for mode in 0..3 {
                    for extra in 0..3 {
                        let src = plane(&mut r, 3 * n + extra, mode);
                        for bits in [8u32, 9] {
                            let (mut got, mut want) =
                                (vec![0x5555u16; 4 * n + 3], vec![0x5555u16; 4 * n + 3]);
                            pack64_on(t, &mut got, &src, bits);
                            pack64_scalar(&mut want, &src, bits);
                            assert_eq!(
                                got, want,
                                "pack64 tier={t:?} n={n} mode={mode} extra={extra} bits={bits}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `rgba_to_ycbcr` in every tier, on random pixels and on the corners of the colour cube.
    #[test]
    fn rgba_to_ycbcr_matches_scalar() {
        const EDGE: [u8; 6] = [0, 1, 127, 128, 254, 255];
        for t in crate::simd::available() {
            let mut r = Rng::new(22);
            for n in [0usize, 1, 2, 15, 16, 17, 31, 32, 33, 100, 257] {
                for mode in 0..2 {
                    let rgba: Vec<u8> = (0..4 * n)
                        .map(|_| {
                            if mode == 0 {
                                r.next_u64() as u8
                            } else {
                                EDGE[r.below(6) as usize]
                            }
                        })
                        .collect();
                    let mut got = (
                        vec![0x55u8; n + 3],
                        vec![0x55u8; n + 3],
                        vec![0x55u8; n + 3],
                    );
                    let mut want = got.clone();
                    rgba_to_ycbcr_on(t, &mut got.0, &mut got.1, &mut got.2, &rgba);
                    rgba_to_ycbcr_scalar(&mut want.0, &mut want.1, &mut want.2, &rgba);
                    assert_eq!(got, want, "rgba_to_ycbcr tier={t:?} n={n} mode={mode}");
                }
            }
        }
    }

    /// The first index of each 4x4 block of a `bw` x `bw` plane, as the decoder lays them out.
    fn block_starts(bw: usize) -> Vec<i32> {
        (0..bw * bw / 16)
            .map(|a| (a % (bw / 4) * 4 + a / (bw / 4) * 4 * bw) as i32)
            .collect()
    }

    /// The Hulsken unpack in every tier, on blocks with whole blocks of 64 positions and a tail, with
    /// absent planes, all-zero halves and zero groups of positions, over a poisoned output (so a
    /// skipped position must stay as it was).
    #[test]
    fn hulsken_unpack_matches_scalar() {
        use crate::hulsken::{unpack_on, unpack_scalar};
        for t in crate::simd::available() {
            let mut r = Rng::new(23);
            for bw in [8usize, 16, 24, 28, 32, 64, 68, 128] {
                let (area, n) = (bw * bw, bw * bw / 8);
                let blk = block_starts(bw);
                for present in [0xffffu16, 0x00ff, 0xff00, 0x8001, 0, 0x7ff0] {
                    for live in [0u64, 1, 4, 8] {
                        // Groups of 8 positions are all zero unless live; a live group's bytes are
                        // zero a quarter of the time.
                        let on: Vec<bool> = (0..n.div_ceil(8)).map(|_| r.below(8) < live).collect();
                        let store: Vec<Vec<u8>> = (0..16)
                            .map(|k| {
                                if (present >> k) & 1 == 0 {
                                    return Vec::new();
                                }
                                (0..n)
                                    .map(|j| {
                                        if on[j / 8] && r.below(4) != 0 {
                                            r.next_u64() as u8
                                        } else {
                                            0
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        let zero = vec![0u8; n];
                        let pl: [Option<&[u8]>; 16] =
                            std::array::from_fn(|k| (!store[k].is_empty()).then(|| &store[k][..]));
                        let (mut got, mut want) = (vec![0x5aa5i16; area], vec![0x5aa5i16; area]);
                        unpack_on(t, &mut got, bw, &blk, &pl, &zero);
                        unpack_scalar(&mut want, bw, &blk, &pl, &zero);
                        assert_eq!(
                            got, want,
                            "unpack tier={t:?} bw={bw} present={present:#06x} live={live}"
                        );
                    }
                }
            }
        }
    }

    /// The unpack kernel on a `blk` that is missing, negative or beyond `d` does nothing and does
    /// not panic: it reports no position done.
    #[test]
    fn hulsken_unpack_kernel_survives_a_hostile_blk() {
        let mut r = Rng::new(24);
        let store: Vec<Vec<u8>> = (0..16)
            .map(|_| (0..128).map(|_| (r.next_u64() as u8) | 1).collect())
            .collect();
        let planes: [&[u8]; 16] = std::array::from_fn(|k| &store[k][..]);
        for t in crate::simd::available() {
            for blk in [
                vec![-1i32; 64],
                vec![i32::MAX; 64],
                vec![0; 3],
                vec![60; 64],
            ] {
                let mut d = vec![0i16; 64];
                let done = crate::simd::unpack(t, &mut d, 8, &blk, &planes, true, true);
                assert_eq!(done, 0, "tier={t:?} blk[0]={:?}", blk.first());
            }
        }
    }
}
