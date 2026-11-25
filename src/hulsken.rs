//! iSyntax codeblocks: the Huffman and zero-run entropy coder of compressors 16 (version 1) and
//! 19 (version 2), the bit-plane unpack, the 4x4 reorder and the sign-magnitude conversion.
//! Port of `internal/hulsken`; see isyntax-go's docs/format.md, "Entropy coding (Hulsken)".

use crate::Error;
use crate::error::codeblock;
use crate::simd::{self, Tier};

const FAST_BITS: usize = 11;
const FAST_MASK: u64 = (1 << FAST_BITS) - 1;

#[derive(Clone, Copy)]
struct LongCode {
    code: u32,
    len: u8,
    sym: u8,
}

/// A decode's working memory; reuse it to avoid allocation.
pub(crate) struct Scratch {
    msg: Vec<u8>,
    /// `symbol | len << 8`; 0 if the code is longer than [`FAST_BITS`].
    fast: [u16; 1 << FAST_BITS],
    long: Vec<LongCode>,
    code: [u32; 256],
    size: [u8; 256],
    in_use: [bool; 256],
    /// Each 4x4 block's first index, for `blk_bw`.
    blk: Vec<i32>,
    blk_bw: usize,
    /// Stands in for absent planes.
    zero: Vec<u8>,
}

impl Scratch {
    pub(crate) fn new() -> Scratch {
        Scratch {
            msg: Vec::new(),
            fast: [0; 1 << FAST_BITS],
            long: Vec::new(),
            code: [0; 256],
            size: [0; 256],
            in_use: [false; 256],
            blk: Vec::new(),
            blk_bw: 0,
            zero: Vec::new(),
        }
    }

    fn blocks(&mut self, bw: usize, area: usize) -> Result<(), Error> {
        if self.blk_bw != bw || self.blk.len() != area / 16 {
            self.blk.clear();
            self.blk_bw = bw;
            self.blk
                .try_reserve_exact(area / 16)
                .map_err(|_| Error::OutOfMemory)?;
            for a in 0..area / 16 {
                self.blk
                    .push((a % (bw / 4) * 4 + a / (bw / 4) * 4 * bw) as i32);
            }
        }
        Ok(())
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

fn le16(b: &[u8], i: usize) -> u16 {
    match b.get(i..i + 2) {
        Some(s) => u16::from_le_bytes([s[0], s[1]]),
        None => 0,
    }
}

fn le32(b: &[u8], i: usize) -> u32 {
    match b.get(i..i + 4) {
        Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0,
    }
}

/// Decodes the codeblock `src` into `dst`: `n` planes (1 for LL, 3 for HL, LH, HH) of `bw * bh`
/// coefficients in raster order. `version` is 1 for compressor 16 and 2 for compressor 19. `bw`
/// must be a multiple of 4 and `bw * bh` a multiple of 16, and `dst` at least `n * bw * bh` long.
pub(crate) fn decode(
    dst: &mut [i16],
    src: &[u8],
    n: usize,
    bw: usize,
    bh: usize,
    version: u8,
    s: &mut Scratch,
) -> Result<(), Error> {
    let area = bw * bh;
    let dst = &mut dst[..n * area];
    dst.fill(0);
    if src.len() <= 8 {
        return Ok(());
    }
    let plane_bytes = area / 8;
    let mut masks = [0u16; 3];
    let ser_len: usize;
    let mut r = BitReader { b: src, pos: 0 };
    if version == 1 {
        ser_len = le32(src, 0) as usize;
        masks = [0xFFFF; 3];
        r.pos = 32;
    } else {
        let mut total = 0usize;
        for (c, m) in masks.iter_mut().enumerate().take(n) {
            *m = le16(src, 2 * c);
            total += m.count_ones() as usize;
        }
        ser_len = total * plane_bytes;
        r.pos = 16 * n;
    }
    if ser_len > 4 * n * area {
        return Err(codeblock(format!(
            "hulsken: serialized length {ser_len} over {}",
            4 * n * area
        )));
    }
    let zero_sym = src[r.pos / 8];
    let counter_bits = u32::from(src[r.pos / 8 + 1]);
    r.pos += 16;
    if version == 2 {
        if ser_len == 0 {
            return Ok(());
        }
        let agg = masks[..n].iter().fold(0u16, |a, m| a | m);
        let ptr_bits = (usize::BITS - ser_len.leading_zeros()) as usize - 1 + 5;
        // The bit-plane seek table, unused.
        r.pos += (agg.count_ones() as usize).saturating_sub(1) * ptr_bits;
    }
    if counter_bits > 32 {
        return Err(codeblock(format!(
            "hulsken: zero counter of {counter_bits} bits"
        )));
    }
    s.read_tree(&mut r)?;
    let got = s.message(&mut r, ser_len, zero_sym, counter_bits, version)?;

    // A short message: the rest reads as zero.
    if version == 1 && got < 16 * n * plane_bytes {
        if got < 2 * n {
            return Err(codeblock("hulsken: message shorter than its bitmasks"));
        }
        let t = &s.msg[got - 2 * n..];
        for (c, m) in masks.iter_mut().enumerate().take(n) {
            *m = le16(t, 2 * c);
        }
    }
    let planes: usize = masks[..n].iter().map(|m| m.count_ones() as usize).sum();
    if planes * plane_bytes > s.msg.len() {
        return Err(codeblock(format!(
            "hulsken: {planes} bit planes in a {}-byte message",
            s.msg.len()
        )));
    }

    s.blocks(bw, area)?;
    if s.zero.len() < plane_bytes {
        s.zero.clear();
        s.zero
            .try_reserve_exact(plane_bytes)
            .map_err(|_| Error::OutOfMemory)?;
        s.zero.resize(plane_bytes, 0);
    }
    let Scratch { msg, blk, zero, .. } = &*s;
    let zero = &zero[..plane_bytes];
    // Each colour's planes by value bit.
    let mut pl: [[Option<&[u8]>; 16]; 3] = [[None; 16]; 3];
    let mut p = 0usize;
    if version == 1 {
        for (c, &mask) in masks.iter().enumerate().take(n) {
            let mut m = mask;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                let t = if b > 0 { b - 1 } else { 15 };
                pl[c][t] = Some(&msg[p * plane_bytes..(p + 1) * plane_bytes]);
                p += 1;
                m &= m - 1;
            }
        }
    } else {
        for b in 0..16 {
            for (c, &mask) in masks.iter().enumerate().take(n) {
                if (mask >> b) & 1 != 0 {
                    pl[c][15 - b] = Some(&msg[p * plane_bytes..(p + 1) * plane_bytes]);
                    p += 1;
                }
            }
        }
    }
    for (c, plc) in pl.iter().enumerate().take(n) {
        unpack(&mut dst[c * area..(c + 1) * area], bw, blk, plc, zero);
    }
    Ok(())
}

/// Transposes a colour's bit planes into coefficients, one 4x4-pair byte at a time, and applies
/// the bit 15 sign.
fn unpack<'a>(d: &mut [i16], bw: usize, blk: &[i32], pl: &[Option<&'a [u8]>; 16], zero: &'a [u8]) {
    unpack_on(simd::tier(), d, bw, blk, pl, zero);
}

/// Planes 0 to 7 and 8 to 15: whether each half has any data.
fn halves(pl: &[Option<&[u8]>; 16]) -> (bool, bool) {
    let any = |h: &[Option<&[u8]>]| h.iter().any(Option::is_some);
    (any(&pl[..8]), any(&pl[8..]))
}

pub(crate) fn unpack_on<'a>(
    t: Tier,
    d: &mut [i16],
    bw: usize,
    blk: &[i32],
    pl: &[Option<&'a [u8]>; 16],
    zero: &'a [u8],
) {
    let (lo, hi) = halves(pl);
    // Cut to the scalar length, so the kernel never writes a position scalar would not.
    let planes: [&[u8]; 16] = std::array::from_fn(|k| {
        let s = pl[k].unwrap_or(zero);
        &s[..s.len().min(zero.len())]
    });
    let done = simd::unpack(t, d, bw, blk, &planes, lo, hi);
    unpack_from(d, bw, blk, pl, zero, done);
}

pub(crate) fn unpack_scalar<'a>(
    d: &mut [i16],
    bw: usize,
    blk: &[i32],
    pl: &[Option<&'a [u8]>; 16],
    zero: &'a [u8],
) {
    unpack_from(d, bw, blk, pl, zero, 0);
}

/// The scalar loop from position `from` on.
fn unpack_from<'a>(
    d: &mut [i16],
    bw: usize,
    blk: &[i32],
    pl: &[Option<&'a [u8]>; 16],
    zero: &'a [u8],
    from: usize,
) {
    let (lo, hi) = halves(pl);
    let n = zero.len();
    let g = |k: usize| pl[k].unwrap_or(zero);
    let (p0, p1, p2, p3, p4, p5, p6, p7) = (g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7));
    let (q0, q1, q2, q3, q4, q5, q6, q7) = (g(8), g(9), g(10), g(11), g(12), g(13), g(14), g(15));
    for j in from..n {
        // Byte k: plane k's (or k+8's) byte j.
        let mut l = 0u64;
        let mut h = 0u64;
        if lo {
            l = u64::from(p0[j])
                | u64::from(p1[j]) << 8
                | u64::from(p2[j]) << 16
                | u64::from(p3[j]) << 24
                | u64::from(p4[j]) << 32
                | u64::from(p5[j]) << 40
                | u64::from(p6[j]) << 48
                | u64::from(p7[j]) << 56;
        }
        if hi {
            h = u64::from(q0[j])
                | u64::from(q1[j]) << 8
                | u64::from(q2[j]) << 16
                | u64::from(q3[j]) << 24
                | u64::from(q4[j]) << 32
                | u64::from(q5[j]) << 40
                | u64::from(q6[j]) << 48
                | u64::from(q7[j]) << 56;
        }
        if l | h == 0 {
            continue;
        }
        let (l, h) = (transpose8(l), transpose8(h));
        let base = blk[j >> 1] as usize + (j & 1) * 2 * bw;
        put4(
            &mut d[base..base + 4],
            sign(spread(l as u32) | (spread(h as u32) << 8)),
        );
        put4(
            &mut d[base + bw..base + bw + 4],
            sign(spread((l >> 32) as u32) | (spread((h >> 32) as u32) << 8)),
        );
    }
}

/// Stores `v`'s 16-bit lanes, lane 0 lowest, as `r[0..4]`.
fn put4(r: &mut [i16], v: u64) {
    r[0] = v as i16;
    r[1] = (v >> 16) as i16;
    r[2] = (v >> 32) as i16;
    r[3] = (v >> 48) as i16;
}

/// Swaps bit i of byte k with bit k of byte i.
fn transpose8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// Puts byte k of `x` in 16-bit lane k.
fn spread(x: u32) -> u64 {
    let mut v = u64::from(x);
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    (v | (v << 8)) & 0x00FF_00FF_00FF_00FF
}

/// Lanes with bit 15 set become `-(v & 0x7FFF)`, per lane.
fn sign(r: u64) -> u64 {
    const H: u64 = 0x8000_8000_8000_8000;
    const L: u64 = 0x0001_0001_0001_0001;
    let s = ((r & H) >> 15) * 0xFFFF;
    let t = (r & !H) ^ s;
    // Carry-free lane add.
    ((t & !H) + (s & L)) ^ (t & H)
}

impl Scratch {
    /// Reads the Huffman tree: the pre-order bits `writeTree` writes.
    fn read_tree(&mut self, r: &mut BitReader) -> Result<(), Error> {
        self.fast.fill(0);
        self.long.clear();
        self.in_use = [false; 256];
        let mut code: u32 = 0;
        let mut depth: usize = 0;
        loop {
            if r.pos >= 8 * r.b.len() {
                return Err(codeblock("hulsken: Huffman tree runs past the codeblock"));
            }
            let mut blob = r.peek();
            while blob & 1 == 0 {
                depth += 1;
                if depth > 16 {
                    return Err(codeblock("hulsken: Huffman code over 16 bits"));
                }
                blob >>= 1;
                r.pos += 1;
            }
            let sym = (blob >> 1) as u8;
            r.pos += 9;
            let si = usize::from(sym);
            self.code[si] = code;
            self.size[si] = depth as u8;
            self.in_use[si] = true;
            if depth <= FAST_BITS {
                let e = u16::from(sym) | ((depth.max(1) as u16) << 8);
                for i in 0..(1u32 << (FAST_BITS - depth)) {
                    self.fast[((i << depth) | code) as usize] = e;
                }
            } else {
                if self.long.len() == 256 {
                    return Err(codeblock("hulsken: too many Huffman codes"));
                }
                self.long.push(LongCode {
                    code,
                    len: depth as u8,
                    sym,
                });
            }
            // The next code: drop trailing ones, take one.
            while depth > 0 && (code >> (depth - 1)) & 1 == 1 {
                depth -= 1;
                code &= !(1u32 << depth);
            }
            if depth == 0 {
                return Ok(());
            }
            code |= 1 << (depth - 1);
        }
    }

    fn symbol(&self, r: &mut BitReader) -> Result<u8, Error> {
        let blob = r.peek();
        let e = self.fast[(blob & FAST_MASK) as usize];
        if e != 0 {
            r.pos += usize::from(e >> 8);
            return Ok(e as u8);
        }
        for l in &self.long {
            if (blob as u32) & ((1u32 << l.len) - 1) == l.code {
                r.pos += usize::from(l.len);
                return Ok(l.sym);
            }
        }
        Err(codeblock("hulsken: unknown Huffman code"))
    }

    /// The zero-run decode into `self.msg` (`ser_len` bytes, the rest zero); returns how many
    /// bytes the stream filled.
    fn message(
        &mut self,
        r: &mut BitReader,
        ser_len: usize,
        zero_sym: u8,
        counter_bits: u32,
        version: u8,
    ) -> Result<usize, Error> {
        self.msg.clear();
        self.msg.resize(ser_len, 0);
        let end = 8 * r.b.len();
        let zi = usize::from(zero_sym);
        let (mut z_code, mut z_len) = (self.code[zi], u32::from(self.size[zi].max(1)));
        if !self.in_use[zi] {
            (z_code, z_len) = (0, 1);
        }
        let counter_mask = (1u64 << counter_bits) - 1;
        let run = |cnt: u64| -> usize {
            if version == 2 {
                (cnt as usize).saturating_add(1)
            } else {
                cnt as usize
            }
        };
        let mut n = 0usize;
        while n < ser_len && r.pos < end {
            // Plain symbols: 5 x 11 bits fit a peek.
            let mut blob = r.peek();
            let (mut got, mut pos) = (0, r.pos);
            while got < 5 && n < ser_len && pos < end {
                let e = self.fast[(blob & FAST_MASK) as usize];
                if e == 0 || e as u8 == zero_sym {
                    break;
                }
                self.msg[n] = e as u8;
                n += 1;
                pos += usize::from(e >> 8);
                blob >>= u32::from(e >> 8);
                got += 1;
            }
            r.pos = pos;
            if got > 0 {
                continue;
            }
            let sym = self.symbol(r)?;
            if sym != zero_sym {
                self.msg[n] = sym;
                n += 1;
                continue;
            }
            let mut cnt = r.peek() & counter_mask;
            r.pos += counter_bits as usize;
            if cnt == 0 {
                // An escaped zero-run symbol.
                self.msg[n] = sym;
                n += 1;
                continue;
            }
            if n.saturating_add(run(cnt)) < ser_len && r.pos < end {
                // Following zero-run codes extend the counter.
                loop {
                    let blob = r.peek();
                    if (blob as u32) & ((1u32 << z_len) - 1) != z_code {
                        break;
                    }
                    cnt = (cnt << counter_bits) | ((blob >> z_len) & counter_mask);
                    r.pos += (z_len + counter_bits) as usize;
                    if cnt > (1 << 31) || n.saturating_add(run(cnt)) >= ser_len || r.pos >= end {
                        break;
                    }
                }
            }
            n += run(cnt).min(ser_len - n);
        }
        Ok(n)
    }
}

/// LSB first; reads past the end are 0.
struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    /// At least 57 bits from `pos`.
    fn peek(&self) -> u64 {
        let i = self.pos >> 3;
        if let Some(s) = self.b.get(i..i + 8) {
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            return u64::from_le_bytes(a) >> (self.pos & 7);
        }
        self.peek_tail()
    }

    /// `peek` near the end.
    fn peek_tail(&self) -> u64 {
        let i = self.pos >> 3;
        let mut v = 0u64;
        for k in 0..8 {
            if let Some(&b) = self.b.get(i + k) {
                v |= u64::from(b) << (8 * k);
            }
        }
        v >> (self.pos & 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{HulskenParams, encode_hulsken};
    use crate::testutil::Rng;

    /// Random data with seeded zero runs.
    fn random_block(
        r: &mut Rng,
        n: usize,
        bw: usize,
        bh: usize,
        bits_used: u32,
        zero_share: f64,
    ) -> Vec<i16> {
        (0..n * bw * bh)
            .map(|_| {
                if (r.below(1 << 20) as f64) / f64::from(1 << 20) < zero_share {
                    return 0;
                }
                let m = r.below(1 << bits_used) as i16;
                if r.below(2) == 0 { -m } else { m }
            })
            .collect()
    }

    #[test]
    fn round_trip() {
        let mut r = Rng::new(1);
        let mut s = Scratch::new();
        for version in [1u8, 2] {
            for n in [1usize, 3] {
                for bw in [8usize, 16, 128] {
                    let trials = if bw == 128 { 8 } else { 40 };
                    for trial in 0..trials {
                        let bits_used = 1 + r.below(15) as u32;
                        let zero = [0.0, 0.5, 0.9, 0.999, 1.0][trial % 5];
                        let want = random_block(&mut r, n, bw, bw, bits_used, zero);
                        let p = HulskenParams {
                            version,
                            zero_symbol: [0u8, 0x7f, 0xff, 1][trial % 4],
                            counter_bits: 1 + r.below(8) as usize,
                        };
                        let src = encode_hulsken(&want, n, bw, bw, p);
                        if src.len() <= 8 {
                            continue; // decode treats a block this short as empty, as Go does
                        }
                        let mut got = vec![12345i16; want.len()]; // must be overwritten
                        decode(&mut got, &src, n, bw, bw, version, &mut s).unwrap_or_else(|e| {
                            panic!("v{version} n={n} bw={bw} trial {trial} {p:?}: {e}")
                        });
                        assert_eq!(got, want, "v{version} n={n} bw={bw} trial {trial} {p:?}");
                    }
                }
            }
        }
    }

    /// Full-range values hit all 16 planes.
    #[test]
    fn all_planes() {
        let mut r = Rng::new(3);
        let mut s = Scratch::new();
        for version in [1u8, 2] {
            let mut want: Vec<i16> = (0..3 * 16 * 16)
                .map(|_| (r.below(1 << 16) as i64 - (1 << 15)).max(-32767) as i16)
                .collect();
            for i in 0..3 {
                want[i * 256] = 0x7fff;
                want[i * 256 + 1] = -0x7fff;
            }
            let p = HulskenParams {
                version,
                zero_symbol: 0,
                counter_bits: 3,
            };
            let src = encode_hulsken(&want, 3, 16, 16, p);
            let mut got = vec![0i16; want.len()];
            decode(&mut got, &src, 3, 16, 16, version, &mut s).unwrap();
            assert_eq!(got, want, "v{version}");
        }
    }

    #[test]
    fn empty_block() {
        let mut s = Scratch::new();
        let mut got: Vec<i16> = (1..=16).collect();
        decode(&mut got, &[0u8; 8], 1, 4, 4, 2, &mut s).unwrap();
        assert!(got.iter().all(|&v| v == 0), "got {got:?}, want zeros");
    }

    #[test]
    fn corrupt() {
        let mut s = Scratch::new();
        let mut dst = vec![0i16; 3 * 128 * 128];
        let cases: [(&str, &[u8]); 2] = [
            (
                "v1 length too large",
                &[0xff, 0xff, 0xff, 0x7f, 0, 4, 0xff, 0xff, 0xff],
            ),
            ("tree runs off end", &[0, 0x10, 0, 0, 0, 4, 0, 0, 0]), // only internal nodes
        ];
        for (name, src) in cases {
            let r = decode(&mut dst, src, 3, 128, 128, 1, &mut s);
            assert!(matches!(r, Err(Error::Codeblock(_))), "{name}: {r:?}");
        }
    }

    /// Hostile codeblocks error, not panic: random bytes and bit flips of valid blocks.
    #[test]
    fn garbage_does_not_panic() {
        let mut r = Rng::new(5);
        let mut s = Scratch::new();
        let mut dst = vec![0i16; 3 * 16 * 16];
        let mut seeds = Vec::new();
        for v in [1u8, 2] {
            for n in [1usize, 3] {
                let blk = random_block(&mut r, n, 16, 16, 6, 0.5);
                seeds.push((
                    encode_hulsken(
                        &blk,
                        n,
                        16,
                        16,
                        HulskenParams {
                            version: v,
                            zero_symbol: 0,
                            counter_bits: 4,
                        },
                    ),
                    v,
                    n,
                ));
            }
        }
        for i in 0..20000 {
            let (src, v, n) = &seeds[i % seeds.len()];
            let mut src = src.clone();
            match i % 3 {
                0 => {
                    for _ in 0..1 + r.below(4) {
                        let at = r.below(src.len() as u64) as usize;
                        src[at] ^= 1 << r.below(8);
                    }
                }
                1 => src.truncate(r.below(src.len() as u64 + 1) as usize),
                _ => src = (0..r.below(200)).map(|_| r.next_u64() as u8).collect(),
            }
            let _ = decode(&mut dst, &src, *n, 16, 16, *v, &mut s);
        }
    }
}
