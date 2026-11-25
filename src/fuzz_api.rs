//! Entry points of the fuzz targets (`tests/fuzz/fuzz_targets`) to code that is not public API: the
//! parsers under `Image::open`, the Hulsken block decoder and each SIMD kernel against its scalar
//! reference. Not part of the API. What a function allocates from a caller's lengths is checked
//! against its `max_alloc` first.

use crate::budget::Acct;
use crate::hulsken::{self, Scratch};
use crate::{Error, container, decoder, dsp, region, simd, xml};

/// The width of the inverse-transform buffer, and the side of a tile.
pub const W: usize = decoder::W;
pub const TS: usize = decoder::TS;

fn name(t: simd::Tier) -> &'static str {
    match t {
        simd::Tier::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        simd::Tier::Avx2 => "avx2",
        #[cfg(target_arch = "x86_64")]
        simd::Tier::Avx512 => "avx512",
        #[cfg(target_arch = "aarch64")]
        simd::Tier::Neon => "neon",
    }
}

/// This CPU's SIMD tier as the kernels use it (`ISYNTAX_SIMD` applies): `"avx512"`, `"avx2"`,
/// `"neon"` or `"scalar"`.
pub fn tier() -> &'static str {
    name(simd::tier())
}

/// A SIMD tier to run a kernel on.
#[derive(Clone, Copy)]
pub struct Tier(simd::Tier);

impl Tier {
    /// `"scalar"`, `"avx2"`, `"avx512"` or `"neon"`.
    pub fn name(self) -> &'static str {
        name(self.0)
    }
}

/// Every tier this CPU has, scalar first, whatever `ISYNTAX_SIMD` says.
pub fn tiers() -> Vec<Tier> {
    simd::available().into_iter().map(Tier).collect()
}

/// The tolerant base64 decoder of the header.
pub fn decode_base64(s: &str, max_alloc: u64) -> Result<Vec<u8>, Error> {
    container::decode_base64(s, &mut Acct::new(max_alloc))
}

/// The cluster table against two fixed templates: the codeblocks it holds.
pub fn cluster_table(b: &[u8], max_alloc: u64) -> Result<usize, Error> {
    container::fuzz_cluster_table(b, max_alloc)
}

/// The XML reader of the header.
pub fn xml(b: &[u8], max_alloc: u64) -> Result<(), Error> {
    xml::parse(b, &mut Acct::new(max_alloc)).map(|_| ())
}

/// A Hulsken decoder with its scratch, to be reused across inputs.
#[derive(Default)]
pub struct Hulsken(Scratch);

impl Hulsken {
    /// Decodes `src` as `n` (1 or 3) planes of `bw` x `bw` coefficients, `version` 1 or 2; the
    /// output is charged to `max_alloc`.
    pub fn decode(
        &mut self,
        src: &[u8],
        n: usize,
        version: u8,
        bw: usize,
        max_alloc: u64,
    ) -> Result<Vec<i16>, Error> {
        if !matches!(n, 1 | 3)
            || !matches!(version, 1 | 2)
            || bw == 0
            || !bw.is_multiple_of(4)
            || bw > 256
        {
            return Err(Error::InvalidArgument("hulsken shape".into()));
        }
        let len = n * bw * bw;
        if len as u64 * 2 > max_alloc {
            return Err(Error::AllocLimit);
        }
        let mut dst = Vec::new();
        dst.try_reserve_exact(len).map_err(|_| Error::OutOfMemory)?;
        dst.resize(len, 0);
        hulsken::decode(&mut dst, src, n, bw, bw, version, &mut self.0)?;
        Ok(dst)
    }
}

fn same<T: PartialEq>(what: &str, t: Tier, got: &T, want: &T) -> Result<(), Error> {
    if got == want {
        Ok(())
    } else {
        Err(Error::Internal(format!(
            "{what}: the {} tier differs from scalar",
            t.name()
        )))
    }
}

const POISON: i16 = 0x5555;
/// How many elements the output of a kernel has beyond what the shortest input fills: the kernel
/// must leave them alone.
const SPARE: usize = 3;

/// Two buffers of `n` copies of `fill`, charged to `max_alloc` before they are allocated.
fn pair<T: Clone>(n: usize, fill: T, max_alloc: u64) -> Result<(Vec<T>, Vec<T>), Error> {
    let bytes = n
        .checked_mul(2 * std::mem::size_of::<T>())
        .ok_or(Error::AllocLimit)?;
    if bytes as u64 > max_alloc {
        return Err(Error::AllocLimit);
    }
    let one = || -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        v.resize(n, fill.clone());
        Ok(v)
    };
    Ok((one()?, one()?))
}

/// `skip + n * k + SPARE` for the output of a kernel that writes `k` elements per input element
/// from element `skip` on.
fn out_len(n: usize, k: usize, skip: usize) -> Result<usize, Error> {
    n.checked_mul(k)
        .and_then(|v| v.checked_add(SPARE))
        .and_then(|v| v.checked_add(skip))
        .ok_or(Error::AllocLimit)
}

/// `lift_odd` in tier `t` against scalar. The inputs may differ in length and the output is
/// longer than all of them.
pub fn lift_odd(
    t: Tier,
    l: &[i16],
    h0: &[i16],
    h1: &[i16],
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    let n = l.len().max(h0.len()).max(h1.len());
    let (mut got, mut want) = pair(out_len(n, 1, skip)?, POISON, max_alloc)?;
    dsp::lift_odd_on(t.0, &mut got[skip..], l, h0, h1);
    dsp::lift_odd_scalar(&mut want[skip..], l, h0, h1);
    same("lift_odd", t, &got, &want)
}

/// `lift_even` against scalar.
pub fn lift_even(
    t: Tier,
    h: &[i16],
    o0: &[i16],
    o1: &[i16],
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    let n = h.len().max(o0.len()).max(o1.len());
    let (mut got, mut want) = pair(out_len(n, 1, skip)?, POISON, max_alloc)?;
    dsp::lift_even_on(t.0, &mut got[skip..], h, o0, o1);
    dsp::lift_even_scalar(&mut want[skip..], h, o0, o1);
    same("lift_even", t, &got, &want)
}

/// `idwt_row` (the 1D inverse lift of a row) against scalar, on the common length of the inputs
/// (at least 2 positions) and an output longer than the row.
pub fn idwt_row(t: Tier, l: &[i16], h: &[i16], skip: usize, max_alloc: u64) -> Result<(), Error> {
    let n = l.len().min(h.len());
    if n < 2 {
        return Ok(());
    }
    let (l, h) = (&l[..n], &h[..n]);
    let (mut got, mut want) = pair(out_len(n, 2, skip)?, POISON, max_alloc)?;
    dsp::idwt_row_on(t.0, &mut got[skip..], l, h);
    dsp::idwt_row_from(&mut want[skip..], l, h, 0);
    same("idwt_row", t, &got, &want)
}

/// `ycocg_to_rgba` against scalar, with any `shift`.
pub fn ycocg_to_rgba(
    t: Tier,
    y: &[i16],
    co: &[i16],
    cg: &[i16],
    shift: u32,
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    let n = y.len().max(co.len()).max(cg.len());
    let (mut got, mut want) = pair(out_len(n, 4, skip)?, 0x55u8, max_alloc)?;
    dsp::ycocg_to_rgba_on(t.0, &mut got[skip..], y, co, cg, shift);
    dsp::ycocg_to_rgba_scalar(&mut want[skip..], y, co, cg, shift);
    same("ycocg_to_rgba", t, &got, &want)
}

/// The four sample kernels against scalar: `kind` 0 `ycocg_to_rgb16`, 1 `planes_to_rgb16`, 2
/// `ycocg_to_samples`, 3 `planes_to_samples` (the last two with `hi`, at least 0).
#[allow(clippy::too_many_arguments)]
pub fn samples(
    t: Tier,
    a: &[i16],
    b: &[i16],
    c: &[i16],
    hi: i16,
    kind: u8,
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    let n = a.len().max(b.len()).max(c.len());
    let hi = hi.max(0);
    let (mut got, mut want) = pair(out_len(n, 3, skip)?, POISON, max_alloc)?;
    match kind & 3 {
        0 => {
            dsp::ycocg_to_rgb16_on(t.0, &mut got[skip..], a, b, c);
            dsp::ycocg_to_rgb16_scalar(&mut want[skip..], a, b, c);
        }
        1 => {
            dsp::planes_to_rgb16_on(t.0, &mut got[skip..], a, b, c);
            dsp::planes_to_rgb16_scalar(&mut want[skip..], a, b, c);
        }
        2 => {
            dsp::ycocg_to_samples_on(t.0, &mut got[skip..], a, b, c, hi);
            dsp::ycocg_to_samples_scalar(&mut want[skip..], a, b, c, hi);
        }
        _ => {
            dsp::planes_to_samples_on(t.0, &mut got[skip..], a, b, c, hi);
            dsp::planes_to_samples_scalar(&mut want[skip..], a, b, c, hi);
        }
    }
    same("samples", t, &got, &want)
}

/// `trunc_row` against scalar, with any `mask` and `o`.
pub fn trunc_row(
    t: Tier,
    row: &[i16],
    mask: i16,
    o: i16,
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    let n = row.len().checked_add(skip).ok_or(Error::AllocLimit)?;
    let (mut got, mut want) = pair(n, 0, max_alloc)?;
    got[skip..].copy_from_slice(row);
    want[skip..].copy_from_slice(row);
    dsp::trunc_row_on(t.0, &mut got[skip..], mask, o);
    dsp::trunc_row_scalar(&mut want[skip..], mask, o);
    same("trunc_row", t, &got, &want)
}

/// `pack64` (a row of RGB samples to RGBA64) against scalar, with 8 bits or, if `bits` is odd, 9.
pub fn pack64(t: Tier, src: &[i16], bits: u8, skip: usize, max_alloc: u64) -> Result<(), Error> {
    let bits = 8 + u32::from(bits & 1);
    let (mut got, mut want) = pair(out_len(src.len() / 3, 4, skip)?, 0x5555u16, max_alloc)?;
    region::pack64_on(t.0, &mut got[skip..], src, bits);
    region::pack64_scalar(&mut want[skip..], src, bits);
    same("pack64", t, &got, &want)
}

/// `rgba_to_ycbcr` against scalar, on `rgba.len() / 4` pixels and outputs longer than that.
pub fn rgba_to_ycbcr(t: Tier, rgba: &[u8], skip: usize, max_alloc: u64) -> Result<(), Error> {
    let len = out_len(rgba.len() / 4, 1, skip)?;
    let (mut gy, mut wy) = pair(len, 0x55u8, max_alloc)?;
    let (mut gb, mut wb) = pair(len, 0x55u8, max_alloc)?;
    let (mut gr, mut wr) = pair(len, 0x55u8, max_alloc)?;
    dsp::rgba_to_ycbcr_on(t.0, &mut gy[skip..], &mut gb[skip..], &mut gr[skip..], rgba);
    dsp::rgba_to_ycbcr_scalar(&mut wy[skip..], &mut wb[skip..], &mut wr[skip..], rgba);
    same("rgba_to_ycbcr", t, &(gy, gb, gr), &(wy, wb, wr))
}

/// The block layout of the decoder: the first index of each 4x4 block, in raster order.
fn block_starts(bw: usize) -> Vec<i32> {
    (0..bw * bw / 16)
        .map(|a| (a % (bw / 4) * 4 + a / (bw / 4) * 4 * bw) as i32)
        .collect()
}

/// The Hulsken `unpack` of one colour of a `bw` x `bw` block (`bw` a multiple of 4 up to 256, else
/// nothing is checked) against scalar. Plane `k` is present if bit `k` of `present` is set, and
/// holds `bytes` cycled, with the 16-byte groups whose index has a bit in common with `sparse`
/// zeroed, so that the kernel meets zero groups, absent planes and all-zero halves.
pub fn unpack(
    t: Tier,
    bw: usize,
    bytes: &[u8],
    present: u16,
    sparse: u8,
    skip: usize,
    max_alloc: u64,
) -> Result<(), Error> {
    if bw == 0 || !bw.is_multiple_of(4) || bw > 256 || bytes.is_empty() {
        return Ok(());
    }
    let (area, n) = (bw * bw, bw * bw / 8);
    if (16 * n) as u64 > max_alloc {
        return Err(Error::AllocLimit);
    }
    let store: Vec<Vec<u8>> = (0..16)
        .map(|k| {
            if (present >> k) & 1 == 0 {
                return Vec::new();
            }
            (0..n)
                .map(|j| {
                    if (j / 16) as u8 & sparse != 0 {
                        0
                    } else {
                        bytes[(k * n + j) % bytes.len()]
                    }
                })
                .collect()
        })
        .collect();
    let zero = vec![0u8; n];
    let pl: [Option<&[u8]>; 16] =
        std::array::from_fn(|k| (!store[k].is_empty()).then(|| &store[k][..]));
    let blk = block_starts(bw);
    let (mut got, mut want) = pair(out_len(area, 1, skip)?, POISON, max_alloc)?;
    hulsken::unpack_on(t.0, &mut got[skip..skip + area], bw, &blk, &pl, &zero);
    hulsken::unpack_scalar(&mut want[skip..skip + area], bw, &blk, &pl, &zero);
    same("unpack", t, &got, &want)
}

/// Inputs and outputs for timing one kernel on a tier (`examples/kernels.rs`): `n` elements of
/// deterministic xorshift data, small like decoded coefficients. Each method runs one kernel
/// over all of them, the SIMD prefix and the scalar tail, as the decoder does.
pub struct Bench {
    a: Vec<i16>,
    b: Vec<i16>,
    c: Vec<i16>,
    out: Vec<i16>,
    /// RGBA pixels, for `rgba_to_ycbcr`.
    bytes: Vec<u8>,
    wide: Vec<u16>,
    ycc: Vec<u8>,
    unpack: UnpackBench,
}

/// One colour of a `bw` x `bw` Hulsken block for `Bench::unpack`: planes 0 to 10 and 15 present,
/// the rest absent.
struct UnpackBench {
    bw: usize,
    planes: Vec<Vec<u8>>,
    zero: Vec<u8>,
    blk: Vec<i32>,
    coefs: Vec<i16>,
}

impl Bench {
    /// For kernels over `n` elements (`n >= 2`).
    pub fn new(n: usize) -> Bench {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut plane = || -> Vec<i16> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % 2048) as i16 - 1024
                })
                .collect()
        };
        let (a, b, c) = (plane(), plane(), plane());
        let bytes = (0..4 * n)
            .map(|i| (a[i / 4] >> (i % 2 * 4)) as u8)
            .collect();
        // The `unpack` block: the largest `bw` (a multiple of 4) with `bw * bw <= n`; a byte of
        // each plane holds 8 coefficients.
        let bw = (n.isqrt() / 4 * 4).max(4);
        let bytes_per_plane = bw * bw / 8;
        let planes = (0..16)
            .map(|k| {
                if k > 10 && k != 15 {
                    return Vec::new();
                }
                (0..bytes_per_plane)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        x as u8
                    })
                    .collect()
            })
            .collect();
        let unpack = UnpackBench {
            bw,
            planes,
            zero: vec![0; bytes_per_plane],
            blk: block_starts(bw),
            coefs: vec![0; bw * bw],
        };
        Bench {
            a,
            b,
            c,
            out: vec![0; 3 * n],
            bytes,
            wide: vec![0; 4 * n],
            ycc: vec![0; 3 * n],
            unpack,
        }
    }

    pub fn lift_odd(&mut self, t: Tier) {
        dsp::lift_odd_on(t.0, &mut self.out, &self.a, &self.b, &self.c);
    }

    pub fn lift_even(&mut self, t: Tier) {
        dsp::lift_even_on(t.0, &mut self.out, &self.a, &self.b, &self.c);
    }

    pub fn idwt_row(&mut self, t: Tier) {
        dsp::idwt_row_on(t.0, &mut self.out, &self.a, &self.b);
    }

    pub fn ycocg_to_rgba(&mut self, t: Tier) {
        dsp::ycocg_to_rgba_on(t.0, &mut self.bytes, &self.a, &self.b, &self.c, 2);
    }

    pub fn ycocg_to_rgb16(&mut self, t: Tier) {
        dsp::ycocg_to_rgb16_on(t.0, &mut self.out, &self.a, &self.b, &self.c);
    }

    pub fn ycocg_to_samples(&mut self, t: Tier) {
        dsp::ycocg_to_samples_on(t.0, &mut self.out, &self.a, &self.b, &self.c, 511);
    }

    pub fn planes_to_samples(&mut self, t: Tier) {
        dsp::planes_to_samples_on(t.0, &mut self.out, &self.a, &self.b, &self.c, 255);
    }

    /// `pack64` over the `a` plane read as RGB samples, 8 bits.
    pub fn pack64(&mut self, t: Tier) {
        region::pack64_on(t.0, &mut self.wide, &self.a, 8);
    }

    /// `rgba_to_ycbcr` over `n` pixels.
    pub fn rgba_to_ycbcr(&mut self, t: Tier) {
        let n = self.a.len();
        let (y, rest) = self.ycc.split_at_mut(n);
        let (cb, cr) = rest.split_at_mut(n);
        dsp::rgba_to_ycbcr_on(t.0, y, cb, cr, &self.bytes);
    }

    /// `unpack` of one colour of a `bw` x `bw` block, `bw` the largest multiple of 4 with
    /// `bw * bw <= n`, planes 0 to 10 and 15 present, no sparsity.
    pub fn unpack(&mut self, t: Tier) {
        let u = &mut self.unpack;
        let pl: [Option<&[u8]>; 16] =
            std::array::from_fn(|k| (!u.planes[k].is_empty()).then(|| &u.planes[k][..]));
        hulsken::unpack_on(t.0, &mut u.coefs, u.bw, &u.blk, &pl, &u.zero);
    }

    /// Truncates the `a` plane in place.
    pub fn trunc_row(&mut self, t: Tier) {
        dsp::trunc_row_on(t.0, &mut self.a, 15, 7);
    }
}
