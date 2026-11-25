//! Output formats, tiles and regions: RGBA, RGBA64, YCbCr and the unclamped or clamped samples,
//! the white of what lies outside the level, the fan-out of a region's tiles. Port of
//! `internal/decoder/rgb16.go` and the output half of `image.go`.

use crate::budget::{Charge, lock};
use crate::decoder::{BS, Image, RGBA_BYTES, TS};
use crate::error::check_cancel;
use crate::reuse::{Lent, Stage};
use crate::simd::{self, Tier};
use crate::types::{Rect, Subsample, YCbCr, chroma_len};
use crate::{Error, dsp, parallel};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

/// Coordinates beyond this are refused: they cannot be a window onto an image the limits allow.
const MAX_COORD: i64 = 1 << 40;

/// Where the rows of a tile go.
enum Rows<'a, 'b, T> {
    /// Row `y` starts at `buf[y * stride]`.
    Strided { buf: &'a mut [T], stride: usize },
    /// Row `y` is `rows[y]`: the disjoint pieces of a larger buffer that a tile owns.
    Split(&'b mut [&'a mut [T]]),
}

/// Where decoded samples go, and `white`, the samples of one white pixel (its length is the
/// channels per pixel).
pub(crate) struct Out<'a, 'b, T> {
    rows: Rows<'a, 'b, T>,
    pub white: &'a [T],
}

impl<'a, 'b, T> Out<'a, 'b, T> {
    pub fn strided(buf: &'a mut [T], stride: usize, white: &'a [T]) -> Self {
        Out {
            rows: Rows::Strided { buf, stride },
            white,
        }
    }

    fn split(rows: &'b mut [&'a mut [T]], white: &'a [T]) -> Self {
        Out {
            rows: Rows::Split(rows),
            white,
        }
    }

    /// The first `len` samples of row `r`.
    fn row(&mut self, r: usize, len: usize) -> &mut [T] {
        match &mut self.rows {
            Rows::Strided { buf, stride } => &mut buf[r * *stride..][..len],
            Rows::Split(rows) => &mut rows[r][..len],
        }
    }
}

/// The part of a region's output that one tile owns, and whether the tile finished.
struct Piece<'a, T> {
    rows: Vec<&'a mut [T]>,
    done: bool,
}

fn fill_white<T: Copy>(dst: &mut [T], white: &[T]) {
    for px in dst.chunks_exact_mut(white.len()) {
        px.copy_from_slice(white);
    }
}

fn invalid(s: String) -> Error {
    Error::InvalidArgument(s)
}

/// `rows` rows of `row_len` elements at `stride` fit in `len`.
fn check_dst(len: usize, stride: usize, rows: usize, row_len: usize) -> Result<(), Error> {
    if rows == 0 {
        return Ok(());
    }
    let need = (rows - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(row_len));
    if stride < row_len || need.is_none_or(|n| n > len) {
        return Err(invalid(format!(
            "{len} samples at stride {stride}, too small for {rows} rows of {row_len}"
        )));
    }
    Ok(())
}

/// The size of a non-empty `r`, after refusing coordinates no image has.
fn check_rect(r: Rect) -> Result<(usize, usize), Error> {
    for c in [r.min_x, r.min_y, r.max_x, r.max_y] {
        if c.unsigned_abs() > MAX_COORD as u64 {
            return Err(invalid(format!("region {r:?} has coordinates over 2^40")));
        }
    }
    let size = |n: i64| usize::try_from(n).map_err(|_| invalid(format!("region {r:?} too large")));
    Ok((size(r.dx())?, size(r.dy())?))
}

/// `ch` samples for each of `dx` pixels.
fn row_len(dx: usize, ch: usize) -> Result<usize, Error> {
    dx.checked_mul(ch)
        .ok_or_else(|| invalid("region size overflows".into()))
}

fn check_ycbcr(dst: &YCbCr, w: i64, h: i64) -> Result<(), Error> {
    if dst.rect.dx() != w || dst.rect.dy() != h {
        return Err(invalid(format!(
            "destination is {}x{}, want {w}x{h}",
            dst.rect.dx(),
            dst.rect.dy()
        )));
    }
    if !dst.fits() {
        return Err(invalid("YCbCr planes too small for their rectangle".into()));
    }
    Ok(())
}

/// Calls `f` on `r` minus `inner`, as up to four rectangles.
fn around(r: Rect, inner: Rect, f: &mut dyn FnMut(Rect) -> Result<(), Error>) -> Result<(), Error> {
    if inner.is_empty() {
        return f(r);
    }
    for a in [
        Rect::new(r.min_x, r.min_y, r.max_x, inner.min_y),
        Rect::new(r.min_x, inner.max_y, r.max_x, r.max_y),
        Rect::new(r.min_x, inner.min_y, inner.min_x, inner.max_y),
        Rect::new(inner.max_x, inner.min_y, r.max_x, inner.max_y),
    ] {
        if !a.is_empty() {
            f(a)?;
        }
    }
    Ok(())
}

/// Clamps to `[0, 2^bits - 1]` and replicates to 16 bits (`v<<8|v`; 9 bits: `v<<7|v>>2`), alpha
/// 0xFFFF.
fn pack64(dst: &mut [u16], src: &[i16], bits: u32) {
    pack64_on(simd::tier(), dst, src, bits);
}

pub(crate) fn pack64_on(t: Tier, dst: &mut [u16], src: &[i16], bits: u32) {
    let m = simd::pack64(t, dst, src, bits);
    pack64_scalar(&mut dst[4 * m..], &src[3 * m..], bits);
}

pub(crate) fn pack64_scalar(dst: &mut [u16], src: &[i16], bits: u32) {
    let hi = (1i16 << bits) - 1;
    for (d, s) in dst
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        for c in 0..3 {
            let v = s[c].clamp(0, hi) as u16;
            d[c] = if bits == 9 {
                (v << 7) | (v >> 2)
            } else {
                (v << 8) | v
            };
        }
        d[3] = 0xFFFF;
    }
}

/// White over `a`, in pixels of the buffer's origin; with `cancel`, stops when it is set (checked
/// every 64 rows: `a` is as large as the caller's buffer).
fn paint<T: Copy>(
    buf: &mut [T],
    stride: usize,
    white: &[T],
    a: Rect,
    cancel: Option<&AtomicBool>,
) -> Result<(), Error> {
    let w = white.len() * a.dx() as usize;
    for y in a.min_y..a.max_y {
        if let Some(c) = cancel
            && (y - a.min_y) % 64 == 0
        {
            check_cancel(c)?;
        }
        let o = y as usize * stride + white.len() * a.min_x as usize;
        fill_white(&mut buf[o..o + w], white);
    }
    Ok(())
}

impl Image {
    /// A buffer of `n` copies of `v`, from the decoder's free ones if the image has one. Charge
    /// it first, as before.
    fn lend<T: Stage>(&self, n: usize, v: T) -> Result<Lent<T>, Error> {
        Lent::new(self.dec.as_ref(), n, v)
    }

    fn check_level(&self, level: usize) -> Result<(), Error> {
        if level >= self.levels() {
            return Err(invalid(format!("level {level} of {}", self.levels())));
        }
        Ok(())
    }

    fn tile_side(&self, level: usize) -> Result<usize, Error> {
        self.check_level(level)?;
        Ok(if level == self.h.levels { BS } else { TS })
    }

    pub(crate) fn to_rgba(&self, d: &mut [u8], a: &[i16], b: &[i16], c: &[i16]) {
        if self.planes {
            dsp::planes_to_rgba(d, a, b, c, self.shift);
        } else {
            dsp::ycocg_to_rgba(d, a, b, c, self.shift);
        }
    }

    pub(crate) fn to_rgb16(&self, d: &mut [i16], a: &[i16], b: &[i16], c: &[i16]) {
        if self.planes {
            dsp::planes_to_rgb16(d, a, b, c);
        } else {
            dsp::ycocg_to_rgb16(d, a, b, c);
        }
    }

    pub(crate) fn to_samples(&self, d: &mut [i16], a: &[i16], b: &[i16], c: &[i16], hi: i16) {
        if self.planes {
            dsp::planes_to_samples(d, a, b, c, hi);
        } else {
            dsp::ycocg_to_samples(d, a, b, c, hi);
        }
    }

    /// Decodes tile `(x, y)` of `level` into `out`, converting each row of three planes with
    /// `conv`; a tile outside the grid is white.
    #[allow(clippy::type_complexity)]
    pub(crate) fn tile_into<T: Copy>(
        &self,
        level: usize,
        x: i64,
        y: i64,
        out: &mut Out<T>,
        cancel: &AtomicBool,
        conv: &dyn Fn(&mut [T], &[i16], &[i16], &[i16]),
    ) -> Result<(), Error> {
        check_cancel(cancel)?;
        let side = if level == self.h.levels { BS } else { TS };
        let white = out.white;
        let ch = white.len();
        if !self.in_grid(level.min(self.h.levels - 1), x, y) {
            for r in 0..side {
                fill_white(out.row(r, ch * side), white);
            }
            return Ok(());
        }
        self.tile_rows(level, x, y, cancel, &mut |r, a, b, c| {
            conv(out.row(r, ch * side), a, b, c);
        })
    }

    /// Decodes tile `(tx, ty)` of `level`'s native grid into `dst`, RGBA bytes, row `y` at
    /// `dst[y * stride]`: 256x256 pixels, or 128x128 at the derived level `levels() - 1` (the top
    /// scale's LL). A tile outside the grid is white; one missing from the file decodes as zero
    /// detail over its parent's LL instead of a blank fill (isyntax-go docs/format.md). 9-bit
    /// samples are halved. Cancelling stops the decode between codeblocks.
    ///
    /// Limits: the scratch and the coefficients the call caches are charged to the image's
    /// `max_alloc`, which it shares with the image's views and other calls; it runs on the
    /// calling thread.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst` or a stride under a row;
    /// [`Error::Cancelled`]; [`Error::MemoryLimit`] when other decodes hold the memory it needs
    /// (transient) and [`Error::AllocLimit`] when it needs more than the whole `max_alloc`;
    /// [`Error::Io`], [`Error::Codeblock`], [`Error::Header`] or [`Error::Unsupported`] from the
    /// file; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_tile_rgba(
        &self,
        level: usize,
        tx: i64,
        ty: i64,
        dst: &mut [u8],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let side = self.tile_side(level)?;
        check_dst(dst.len(), stride, side, 4 * side)?;
        let mut out = Out::strided(dst, stride, &[255; 4]);
        self.tile_into(level, tx, ty, &mut out, cancel, &|d, a, b, c| {
            self.to_rgba(d, a, b, c);
        })
    }

    /// [`decode_tile_rgba`](Image::decode_tile_rgba) into 16-bit RGBA (4 `u16` per pixel): the
    /// decoder's unshifted sample clamped to `[0, 2^b-1]` (`b` = 9 for 9-bit files, else 8), then
    /// bit-replicated to 16 bits (`b=8: v<<8|v`; `b=9: v<<7|v>>2`). Alpha and white are 0xFFFF.
    ///
    /// Limits: the scratch and the coefficients the call caches are charged to the image's
    /// `max_alloc`, which it shares with the image's views and other calls; it runs on the
    /// calling thread.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst` or a stride under a row;
    /// [`Error::Cancelled`]; [`Error::MemoryLimit`] when other decodes hold the memory it needs
    /// (transient) and [`Error::AllocLimit`] when it needs more than the whole `max_alloc`;
    /// [`Error::Io`], [`Error::Codeblock`], [`Error::Header`] or [`Error::Unsupported`] from the
    /// file; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_tile_rgba64(
        &self,
        level: usize,
        tx: i64,
        ty: i64,
        dst: &mut [u16],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let side = self.tile_side(level)?;
        check_dst(dst.len(), stride, side, 4 * side)?;
        let mut out = Out::strided(dst, stride, &[0xFFFF; 4]);
        self.tile_into(level, tx, ty, &mut out, cancel, &|d, a, b, c| {
            self.pack_row(d, a, b, c);
        })
    }

    fn pack_row(&self, d: &mut [u16], a: &[i16], b: &[i16], c: &[i16]) {
        let n = 3 * a.len();
        let mut tmp = [0i16; 3 * TS];
        self.to_rgb16(&mut tmp[..n], a, b, c);
        pack64(d, &tmp[..n], 8 + self.shift);
    }

    /// [`decode_tile_rgba`](Image::decode_tile_rgba) into a planar YCbCr of the tile's size
    /// (`YCbCr::new(Rect::new(0, 0, 256, 256), .., &limits)`), full-range BT.601, 4:2:0 chroma the rounded
    /// mean of each 2x2 block.
    ///
    /// Limits: the scratch and the coefficients the call caches are charged to the image's
    /// `max_alloc`, which it shares with the image's views and other calls; it runs on the
    /// calling thread.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level or a `dst` whose rect is not the tile's size or
    /// whose planes or strides do not cover it;
    /// [`Error::Cancelled`]; [`Error::MemoryLimit`] when other decodes hold the memory it needs
    /// (transient) and [`Error::AllocLimit`] when it needs more than the whole `max_alloc`;
    /// [`Error::Io`], [`Error::Codeblock`], [`Error::Header`] or [`Error::Unsupported`] from the
    /// file; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_tile_ycbcr(
        &self,
        level: usize,
        tx: i64,
        ty: i64,
        dst: &mut YCbCr,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let side = self.tile_side(level)?;
        check_ycbcr(dst, side as i64, side as i64)?;
        let _charge = self.reserve(RGBA_BYTES)?;
        let mut px = self.lend(4 * side * side, 0u8)?;
        let mut out = Out::strided(&mut px, 4 * side, &[255; 4]);
        self.tile_into(level, tx, ty, &mut out, cancel, &|d, a, b, c| {
            self.to_rgba(d, a, b, c);
        })?;
        self.to_ycbcr(dst, &px, 4 * side, cancel)
    }

    /// Decodes the rectangle `r` of `level` into `dst`, RGBA bytes, row `y` of `r` at
    /// `dst[y * stride]`. `level` may be `levels() - 1`, the derived top-scale LL.
    ///
    /// `r` is in pixels of the level, with (0, 0) at the image's top-left: level pixel `(x, y)`
    /// is pixel `(x+o, y+o)` of the level's native tile grid, where `n` is the native level count
    /// (`levels() - 1`) and `o = ((((3 << n) - 2) >> level) - 1)`. This matches Philips SDK
    /// SourceView (isyntax-go docs/format.md); libisyntax's `o = ((3 << n) - 3) >> level` differs
    /// by one from level 2. Pixels outside the level's width and height are white; a tile missing
    /// from the file decodes instead of a blank fill.
    ///
    /// Tiles are decoded in parallel on up to `Limits::max_threads` threads (the caller's included; 1 spawns none and never touches a pool), from the pool of [`Image::with_pool`] or the shared one. The output is the same for every thread count. Cancelling stops
    /// the decode between codeblocks and tiles; the pixels of tiles not decoded are white.
    ///
    /// Limits: the staging buffer of each edge tile in flight (a tile cut by the edge of `r` or
    /// of the level; the others decode straight into `dst`), the list of row pieces the tiles own,
    /// the scratch and the cached coefficients are charged to the image's `max_alloc`, shared with
    /// its views and other calls, all threads together (`dst` is the caller's and is not
    /// charged); at most `Limits::max_threads` threads work on it.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst`, a stride under a row or a
    /// coordinate beyond +-2^40; [`Error::Cancelled`]; [`Error::MemoryLimit`] when
    /// other decodes hold the memory it needs (transient) and [`Error::AllocLimit`] when one
    /// tile, or the list of row pieces (16 bytes per row of `r` per tile column), needs more than
    /// the whole `max_alloc`; [`Error::Io`], [`Error::Codeblock`],
    /// [`Error::Header`] or [`Error::Unsupported`] from the file; [`Error::Internal`] if a worker
    /// panicked; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_region_rgba(
        &self,
        level: usize,
        r: Rect,
        dst: &mut [u8],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.check_level(level)?;
        if r.is_empty() {
            return Ok(());
        }
        let (dx, dy) = check_rect(r)?;
        check_dst(dst.len(), stride, dy, row_len(dx, 4)?)?;
        let out = Out::strided(dst, stride, &[255; 4]);
        self.region(level, r, out, cancel, &|tx, ty, o| {
            self.tile_into(level, tx, ty, o, cancel, &|d, a, b, c| {
                self.to_rgba(d, a, b, c);
            })
        })
    }

    /// [`decode_region_rgba`](Image::decode_region_rgba) into 16-bit RGBA, as
    /// [`decode_tile_rgba64`](Image::decode_tile_rgba64) (its scratch is charged to the memory
    /// limit).
    ///
    /// Limits: the staging buffer of each edge tile in flight (a tile cut by the edge of `r` or
    /// of the level; the others decode straight into `dst`), the list of row pieces the tiles own,
    /// the scratch and the cached coefficients are charged to the image's `max_alloc`, shared with
    /// its views and other calls, all threads together (`dst` is the caller's and is not
    /// charged); at most `Limits::max_threads` threads work on it.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst`, a stride under a row or a
    /// coordinate beyond +-2^40; [`Error::Cancelled`]; [`Error::MemoryLimit`] when
    /// other decodes hold the memory it needs (transient) and [`Error::AllocLimit`] when one
    /// tile, or the list of row pieces (16 bytes per row of `r` per tile column), needs more than
    /// the whole `max_alloc`; [`Error::Io`], [`Error::Codeblock`],
    /// [`Error::Header`] or [`Error::Unsupported`] from the file; [`Error::Internal`] if a worker
    /// panicked; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_region_rgba64(
        &self,
        level: usize,
        r: Rect,
        dst: &mut [u16],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.check_level(level)?;
        if r.is_empty() {
            return Ok(());
        }
        let (dx, dy) = check_rect(r)?;
        check_dst(dst.len(), stride, dy, row_len(dx, 4)?)?;
        let out = Out::strided(dst, stride, &[0xFFFF; 4]);
        self.region(level, r, out, cancel, &|tx, ty, o| {
            self.tile_into(level, tx, ty, o, cancel, &|d, a, b, c| {
                self.pack_row(d, a, b, c);
            })
        })
    }

    /// [`decode_region_rgba`](Image::decode_region_rgba) into a planar YCbCr whose rectangle is
    /// `r`'s size: pixel `r.min` lands on `dst.rect.min`. Full-range BT.601; 4:2:0 chroma is the
    /// rounded mean of each 2x2 block's pixels inside `dst.rect`.
    ///
    /// Limits: an RGBA buffer of `4 * r.dx() * r.dy()` bytes for the whole region, the chroma
    /// rows of the conversion, the staging buffer of each edge tile in flight, the list of row
    /// pieces the tiles own, the scratch and the cached coefficients are charged to the image's
    /// `max_alloc`, shared with its views and other calls, all threads together (`dst` is the
    /// caller's and is not charged); at most `Limits::max_threads` threads work on it.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a `dst` whose rect is not the size of `r` or
    /// whose planes or strides do not cover it, or a coordinate beyond +-2^40;
    /// [`Error::Cancelled`]; [`Error::MemoryLimit`] when other decodes hold the memory it needs
    /// (transient), or when the call's own buffers together (the RGBA buffer of the region plus a
    /// tile's scratch, staging and row pieces) are over `max_alloc` (not transient: size the limit
    /// at least `4 * dx * dy` plus a few MiB); [`Error::AllocLimit`] when the RGBA buffer of the
    /// region alone is more than the whole `max_alloc`;
    /// [`Error::Io`], [`Error::Codeblock`],
    /// [`Error::Header`] or [`Error::Unsupported`] from the file; [`Error::Internal`] if a worker
    /// panicked; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_region_ycbcr(
        &self,
        level: usize,
        r: Rect,
        dst: &mut YCbCr,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.check_level(level)?;
        check_ycbcr(dst, r.dx(), r.dy())?;
        if r.is_empty() {
            return Ok(());
        }
        let (dx, dy) = check_rect(r)?;
        let row = row_len(dx, 4)?;
        let n = row
            .checked_mul(dy)
            .ok_or_else(|| invalid("region size overflows".into()))?;
        let _charge = self.reserve(n as u64)?;
        let mut px = self.lend(n, 0u8)?;
        self.decode_region_rgba(level, r, &mut px, row, cancel)?;
        self.to_ycbcr(dst, &px, row, cancel)
    }

    /// Decodes `r` as [`decode_region_rgba`](Image::decode_region_rgba) does, but writes 3 `u16`
    /// per pixel (R, G, B), each the sample clamped to `[0, 2^b-1]` (`b` = 9 for 9-bit files, else
    /// 8) and not bit-replicated; pixels painted white get `2^b-1`. Row `y` of `r` starts at
    /// `dst[y * stride]`; a stride under `3 * r.dx()` or a short `dst` is an error. For a sample
    /// `s`, the RGBA64 channel is `s<<7|s>>2` for 9-bit files and `s<<8|s` otherwise.
    ///
    /// Limits: the staging buffer of each edge tile in flight (a tile cut by the edge of `r` or
    /// of the level; the others decode straight into `dst`), the list of row pieces the tiles own,
    /// the scratch and the cached coefficients are charged to the image's `max_alloc`, shared with
    /// its views and other calls, all threads together (`dst` is the caller's and is not
    /// charged); at most `Limits::max_threads` threads work on it.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst`, a stride under a row or a
    /// coordinate beyond +-2^40; [`Error::Cancelled`]; [`Error::MemoryLimit`] when
    /// other decodes hold the memory it needs (transient) and [`Error::AllocLimit`] when one
    /// tile, or the list of row pieces (16 bytes per row of `r` per tile column), needs more than
    /// the whole `max_alloc`; [`Error::Io`], [`Error::Codeblock`],
    /// [`Error::Header`] or [`Error::Unsupported`] from the file; [`Error::Internal`] if a worker
    /// panicked; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_region_samples(
        &self,
        level: usize,
        r: Rect,
        dst: &mut [u16],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.check_level(level)?;
        if r.is_empty() {
            return Ok(());
        }
        let (dx, dy) = check_rect(r)?;
        check_dst(dst.len(), stride, dy, row_len(dx, 3)?)?;
        let hi = (1i16 << (8 + self.shift)) - 1;
        let white = [hi as u16; 3];
        let out = Out::strided(dst, stride, &white);
        self.region(level, r, out, cancel, &|tx, ty, o| {
            self.tile_into(level, tx, ty, o, cancel, &|d: &mut [u16], a, b, c| {
                let mut tmp = [0i16; 3 * TS];
                let tmp = &mut tmp[..d.len()];
                self.to_samples(tmp, a, b, c, hi);
                for (dv, &v) in d.iter_mut().zip(tmp.iter()) {
                    *dv = v as u16;
                }
            })
        })
    }

    /// [`decode_region_samples`](Image::decode_region_samples) without the clamp: each sample is
    /// the colour transform's signed int16 output, which can be negative or above `2^b-1`.
    /// Pixels painted white still get `2^b-1`.
    ///
    /// Limits: the staging buffer of each edge tile in flight (a tile cut by the edge of `r` or
    /// of the level; the others decode straight into `dst`), the list of row pieces the tiles own,
    /// the scratch and the cached coefficients are charged to the image's `max_alloc`, shared with
    /// its views and other calls, all threads together (`dst` is the caller's and is not
    /// charged); at most `Limits::max_threads` threads work on it.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad level, a short `dst`, a stride under a row or a
    /// coordinate beyond +-2^40; [`Error::Cancelled`]; [`Error::MemoryLimit`] when
    /// other decodes hold the memory it needs (transient) and [`Error::AllocLimit`] when one
    /// tile, or the list of row pieces (16 bytes per row of `r` per tile column), needs more than
    /// the whole `max_alloc`; [`Error::Io`], [`Error::Codeblock`],
    /// [`Error::Header`] or [`Error::Unsupported`] from the file; [`Error::Internal`] if a worker
    /// panicked; [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn decode_region_signed_samples(
        &self,
        level: usize,
        r: Rect,
        dst: &mut [i16],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.check_level(level)?;
        if r.is_empty() {
            return Ok(());
        }
        let (dx, dy) = check_rect(r)?;
        check_dst(dst.len(), stride, dy, row_len(dx, 3)?)?;
        let hi = (1i16 << (8 + self.shift)) - 1;
        let white = [hi; 3];
        let out = Out::strided(dst, stride, &white);
        self.region(level, r, out, cancel, &|tx, ty, o| {
            self.tile_into(level, tx, ty, o, cancel, &|d, a, b, c| {
                self.to_rgb16(d, a, b, c);
            })
        })
    }

    /// Paints `r` of `level` into `out`: the white around the level, and every tile that
    /// touches the level decoded by `tile` on parallel threads. A tile inside `r` is decoded
    /// straight into `out` (it owns disjoint rows of it, split up front); one cut by the edge of
    /// `r` or of the level goes through a staging buffer, charged to the budget and reused by the
    /// call's later edge tiles, and only its visible part is copied. After an error the tiles not
    /// done are white.
    #[allow(clippy::type_complexity)]
    fn region<T: Copy + Send + Sync + Stage>(
        &self,
        level: usize,
        r: Rect,
        out: Out<T>,
        cancel: &AtomicBool,
        tile: &(dyn Fn(i64, i64, &mut Out<T>) -> Result<(), Error> + Sync),
    ) -> Result<(), Error> {
        check_cancel(cancel)?;
        let Some(lv) = self.level(level) else {
            return Err(invalid(format!("level {level} of {}", self.levels())));
        };
        let white = out.white;
        let Rows::Strided { buf, stride } = out.rows else {
            return Err(Error::Internal("region output is not strided".into()));
        };
        let ch = white.len();
        let vis = r.intersect(Rect::new(0, 0, lv.width as i64, lv.height as i64));
        around(r, vis, &mut |a| {
            paint(buf, stride, white, a.sub(r.min_x, r.min_y), Some(cancel))
        })?;
        if vis.is_empty() {
            return Ok(());
        }
        let side = (if level == self.h.levels { BS } else { TS }) as i64;
        let grid_level = level.min(self.h.levels - 1);
        // The SDK SourceView origin, which differs from libisyntax's from level 2.
        let off = (((3i64 << self.h.levels) - 2) >> level) - 1;
        let g = vis.add(off, off);
        let (tx0, ty0) = (g.min_x / side, g.min_y / side);
        let nx = (g.max_x - 1) / side - tx0 + 1;
        let ny = (g.max_y - 1) / side - ty0 + 1;
        let (nxu, nyu) = (nx as usize, ny as usize);
        let n = nxu * nyu;
        let geom = |i: usize| -> (i64, i64, Rect, Rect) {
            let (tx, ty) = (tx0 + i as i64 % nx, ty0 + i as i64 / nx);
            let tr = Rect::new(tx * side, ty * side, (tx + 1) * side, (ty + 1) * side);
            (tx, ty, tr, tr.intersect(g))
        };
        // Charged before it is allocated: the tiles' list and the row pieces they own.
        let a = vis.sub(r.min_x, r.min_y);
        let (ax, aw) = (ch * a.min_x as usize, ch * a.dx() as usize);
        let overflow = || invalid("region size overflows".into());
        let piece_bytes = (a.dy() as usize)
            .checked_mul(nxu)
            .and_then(|rows| rows.checked_mul(size_of::<&mut [T]>()))
            .and_then(|b| b.checked_add(n.checked_mul(size_of::<Mutex<Piece<T>>>())?))
            .ok_or_else(overflow)?;
        // Refused by the memory limit before any tile ran: the visible area is white, as after a
        // decode error. (An allocator refusal below returns OutOfMemory with dst as it is.)
        let _pieces_charge = match self.reserve(piece_bytes as u64) {
            Ok(c) => c,
            Err(e) => {
                paint(buf, stride, white, a, None)?;
                return Err(e);
            }
        };
        let side_u = side as usize;
        let staging = (side_u * side_u * ch * size_of::<T>()) as u64;
        if self.threads.min(n) < 2 {
            // One worker: there are no row pieces to split off, each tile writes straight into
            // `buf`. (The pieces are charged above all the same, so the limits do not change.)
            let one = |buf: &mut [T],
                       stage: &mut Option<(Lent<T>, Charge)>,
                       i: usize|
             -> Result<(), Error> {
                let (tx, ty, tr, part) = geom(i);
                check_cancel(cancel)?;
                let at = part.sub(off, off).sub(r.min_x, r.min_y);
                let (ox, oy) = (ch * at.min_x as usize, at.min_y as usize);
                if !self.in_grid(grid_level, tx, ty) {
                    return paint(buf, stride, white, at, None);
                }
                if part == tr {
                    let dst = &mut buf[oy * stride + ox..];
                    return tile(tx, ty, &mut Out::strided(dst, stride, white));
                }
                if stage.is_none() {
                    let c = self.reserve(staging)?;
                    *stage = Some((self.lend(side_u * side_u * ch, white[0])?, c));
                }
                let (st, _) = stage
                    .as_mut()
                    .ok_or_else(|| Error::Internal("staging buffer".into()))?;
                tile(
                    tx,
                    ty,
                    &mut Out::strided(st.as_mut_slice(), ch * side_u, white),
                )?;
                let (sx, sy) = (
                    (part.min_x - tr.min_x) as usize,
                    (part.min_y - tr.min_y) as usize,
                );
                let w = ch * part.dx() as usize;
                for k in 0..part.dy() as usize {
                    let s = (sy + k) * ch * side_u + ch * sx;
                    let d = (oy + k) * stride + ox;
                    buf[d..d + w].copy_from_slice(&st[s..s + w]);
                }
                Ok(())
            };
            let mut stage = None;
            // The panic net of `parallel::for_each`, which this branch does not go through.
            let mut next = 0;
            let run =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), Error> {
                    while next < n {
                        one(&mut *buf, &mut stage, next)?;
                        next += 1;
                    }
                    Ok(())
                }));
            let err = match run {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => e,
                Err(p) => Error::Internal(format!("panic: {}", parallel::panic_message(p))),
            };
            // The tiles not done are white, the failed one too.
            for j in next..n {
                let at = geom(j).3.sub(off, off).sub(r.min_x, r.min_y);
                // Without a cancel flag, `paint` cannot fail.
                let _ = paint(&mut *buf, stride, white, at, None);
            }
            return Err(err);
        }
        let mut tiles: Vec<Mutex<Piece<T>>> = Vec::new();
        tiles.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        let mut rows = buf[a.min_y as usize * stride..].chunks_mut(stride);
        for cy in 0..nyu {
            let first = tiles.len();
            let h = geom(cy * nxu).3.dy() as usize;
            for _ in 0..nxu {
                let mut v = Vec::new();
                v.try_reserve_exact(h).map_err(|_| Error::OutOfMemory)?;
                tiles.push(Mutex::new(Piece {
                    rows: v,
                    done: false,
                }));
            }
            for _ in 0..h {
                let Some(row) = rows.next() else {
                    return Err(Error::Internal("region output too short".into()));
                };
                let mut rest = row.split_at_mut(ax).1.split_at_mut(aw).0;
                for cx in 0..nxu {
                    let (p, tail) = rest.split_at_mut(ch * geom(first + cx).3.dx() as usize);
                    let t = &mut tiles[first + cx];
                    t.get_mut().unwrap_or_else(|e| e.into_inner()).rows.push(p);
                    rest = tail;
                }
            }
        }
        // Buffers for the edge tiles, each with its charge (dropped after the buffer), kept for the
        // rest of the call.
        let spare: Mutex<Vec<(Lent<T>, Charge)>> = Mutex::new(Vec::new());
        let res = parallel::for_each(self.threads, self.pool.as_ref(), &self.gauge, n, |i| {
            let (tx, ty, tr, part) = geom(i);
            check_cancel(cancel)?;
            let mut guard = lock(&tiles[i]);
            let p = &mut *guard;
            if !self.in_grid(grid_level, tx, ty) {
                for row in &mut p.rows {
                    fill_white(row, white);
                }
            } else if part == tr {
                tile(tx, ty, &mut Out::split(&mut p.rows, white))?;
            } else {
                let pooled = lock(&spare).pop();
                let (mut stage, charge) = match pooled {
                    Some(s) => s,
                    None => {
                        let c = self.reserve(staging)?;
                        (self.lend(side_u * side_u * ch, white[0])?, c)
                    }
                };
                // Every row of the tile is overwritten, so a reused buffer needs no fill.
                let res = tile(tx, ty, &mut Out::strided(&mut stage, ch * side_u, white));
                if res.is_ok() {
                    let (sx, sy) = (
                        (part.min_x - tr.min_x) as usize,
                        (part.min_y - tr.min_y) as usize,
                    );
                    let w = ch * part.dx() as usize;
                    for (k, row) in p.rows.iter_mut().enumerate() {
                        let s = (sy + k) * ch * side_u + ch * sx;
                        row.copy_from_slice(&stage[s..s + w]);
                    }
                }
                lock(&spare).push((stage, charge));
                res?;
            }
            p.done = true;
            Ok(())
        });
        if res.is_err() {
            for t in &mut tiles {
                let p = t.get_mut().unwrap_or_else(|e| e.into_inner());
                if !p.done {
                    // Not cancelled: the tiles not done are white even after a cancel.
                    for row in &mut p.rows {
                        fill_white(row, white);
                    }
                }
            }
        }
        res
    }

    /// RGBA to YCbCr: luma per pixel, 4:2:0 chroma the rounded mean of the pixels of each block
    /// that lie inside `dst.rect`.
    fn to_ycbcr(
        &self,
        dst: &mut YCbCr,
        src: &[u8],
        stride: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let b = dst.rect;
        let wd = b.dx() as usize;
        // The chroma index follows c_offset's truncating x/2.
        let cw = if dst.subsample == Subsample::S420 {
            chroma_len(b.min_x, b.max_x) as usize
        } else {
            0
        };
        let _charge = self.reserve((2 * wd + 12 * cw) as u64)?;
        let mut cb = self.lend(wd, 0u8)?;
        let mut cr = self.lend(wd, 0u8)?;
        if dst.subsample == Subsample::S444 {
            for y in b.min_y..b.max_y {
                if (y - b.min_y) % 64 == 0 {
                    check_cancel(cancel)?;
                }
                let o = dst.y_offset(b.min_x, y);
                let c = dst.c_offset(b.min_x, y);
                let row = (y - b.min_y) as usize * stride;
                dsp::rgba_to_ycbcr(
                    &mut dst.y[o..o + wd],
                    &mut dst.cb[c..c + wd],
                    &mut dst.cr[c..c + wd],
                    &src[row..],
                );
            }
            return Ok(());
        }
        let mut sum_b = self.lend(cw, 0i32)?;
        let mut sum_r = self.lend(cw, 0i32)?;
        let mut cnt = self.lend(cw, 0i32)?;
        for y in b.min_y..b.max_y {
            if (y - b.min_y) % 64 == 0 {
                check_cancel(cancel)?;
            }
            let o = dst.y_offset(b.min_x, y);
            let row = (y - b.min_y) as usize * stride;
            dsp::rgba_to_ycbcr(&mut dst.y[o..o + wd], &mut cb, &mut cr, &src[row..]);
            for i in 0..wd {
                let k = ((b.min_x + i as i64) / 2 - b.min_x / 2) as usize;
                sum_b[k] += i32::from(cb[i]);
                sum_r[k] += i32::from(cr[i]);
                cnt[k] += 1;
            }
            if y == b.max_y - 1 || (y + 1) / 2 != y / 2 {
                let c = dst.c_offset(b.min_x, y);
                for k in 0..cw {
                    let n = cnt[k].max(1);
                    dst.cb[c + k] = ((sum_b[k] + n / 2) / n) as u8;
                    dst.cr[c + k] = ((sum_r[k] + n / 2) / n) as u8;
                    (sum_b[k], sum_r[k], cnt[k]) = (0, 0, 0);
                }
            }
        }
        Ok(())
    }
}
