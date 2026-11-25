use crate::{Error, Limits};
use std::io;
use std::sync::Arc;

/// A pixel position. Go: `image.Point`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    /// Column.
    pub x: i64,
    /// Row.
    pub y: i64,
}

/// A rectangle of pixels, `min` inclusive, `max` exclusive. Go: `image.Rectangle`. One with
/// `max <= min` in either axis is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// First column, inclusive.
    pub min_x: i64,
    /// First row, inclusive.
    pub min_y: i64,
    /// End column, exclusive.
    pub max_x: i64,
    /// End row, exclusive.
    pub max_y: i64,
}

impl Rect {
    /// `image.Rect`: swaps the corners where they are the wrong way round. No limits apply: it
    /// allocates nothing and cannot fail.
    pub fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
        Rect {
            min_x: x0.min(x1),
            min_y: y0.min(y1),
            max_x: x0.max(x1),
            max_y: y0.max(y1),
        }
    }

    /// The width, 0 if empty. No limits apply; it cannot fail.
    pub fn dx(&self) -> i64 {
        self.max_x.saturating_sub(self.min_x).max(0)
    }

    /// The height, 0 if empty. No limits apply; it cannot fail.
    pub fn dy(&self) -> i64 {
        self.max_y.saturating_sub(self.min_y).max(0)
    }

    /// Whether the rectangle has no pixels. No limits apply; it cannot fail.
    pub fn is_empty(&self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }

    /// The common part, or the zero rectangle if there is none. No limits apply; it cannot fail.
    pub fn intersect(&self, s: Rect) -> Rect {
        let r = Rect {
            min_x: self.min_x.max(s.min_x),
            min_y: self.min_y.max(s.min_y),
            max_x: self.max_x.min(s.max_x),
            max_y: self.max_y.min(s.max_y),
        };
        if r.is_empty() { Rect::default() } else { r }
    }

    pub(crate) fn add(&self, dx: i64, dy: i64) -> Rect {
        Rect {
            min_x: self.min_x + dx,
            min_y: self.min_y + dy,
            max_x: self.max_x + dx,
            max_y: self.max_y + dy,
        }
    }

    pub(crate) fn sub(&self, dx: i64, dy: i64) -> Rect {
        self.add(-dx, -dy)
    }
}

/// One resolution level. Go: `Level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Native tiles across.
    pub tiles_x: usize,
    /// Native tiles down.
    pub tiles_y: usize,
    /// Level 0 pixels per pixel of this level along an axis.
    pub downsample: usize,
}

/// Chroma subsampling of a [`YCbCr`] destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsample {
    /// 4:4:4.
    S444,
    /// 4:2:0.
    S420,
}

/// A planar YCbCr image, the layout of Go's `image.YCbCr` (full-range BT.601): the pixel `(x, y)`
/// of `rect` has luma `y[y_offset(x, y)]` and chroma at `c_offset(x, y)`.
#[derive(Debug, Clone)]
pub struct YCbCr {
    /// Luma plane.
    pub y: Vec<u8>,
    /// Blue-difference chroma plane.
    pub cb: Vec<u8>,
    /// Red-difference chroma plane.
    pub cr: Vec<u8>,
    /// Bytes from one luma row to the next.
    pub y_stride: usize,
    /// Bytes from one chroma row to the next.
    pub c_stride: usize,
    /// How the chroma planes are subsampled.
    pub subsample: Subsample,
    /// The pixels the planes cover.
    pub rect: Rect,
}

/// The 4:2:0 chroma samples along an axis spanning `[min, max)`, as `c_offset` indexes them: by the
/// truncating `v/2`, so one more than `(max+1)/2 - min/2` when `max` is 0 and `min` negative.
pub(crate) fn chroma_len(min: i64, max: i64) -> i64 {
    if max <= min {
        return 0;
    }
    ((max - 1) / 2 - min / 2).saturating_add(1)
}

impl YCbCr {
    /// Allocates the planes for `rect` (`image.NewYCbCr`), after checking the pixels against
    /// `limits.max_pixels` and the plane bytes against `limits.max_alloc`. Fails if the size
    /// overflows or the allocator refuses it. Not `image.NewYCbCr` exactly: 4:2:0 chroma is sized
    /// as the decoder indexes it (by the truncating `v/2`), one sample more than Go when `rect.max_x`
    /// or `rect.max_y` is 0 (where Go's own decode would index past its planes and panic).
    ///
    /// Limits: `max_pixels` and `max_alloc` (see above); `max_threads` does not apply, it runs
    /// on the calling thread.
    ///
    /// # Errors
    /// [`Error::PixelLimit`] over `max_pixels`; [`Error::AllocLimit`] when the planes are over
    /// `max_alloc`; [`Error::InvalidArgument`] when the size overflows; [`Error::OutOfMemory`]
    /// when the allocator refuses it.
    pub fn new(rect: Rect, subsample: Subsample, limits: &Limits) -> Result<YCbCr, Error> {
        let (w, h) = (rect.dx(), rect.dy());
        let (cw, ch) = match subsample {
            Subsample::S444 => (w, h),
            Subsample::S420 => (
                chroma_len(rect.min_x, rect.max_x),
                chroma_len(rect.min_y, rect.max_y),
            ),
        };
        let size = |a: i64, b: i64| -> Result<usize, Error> {
            a.checked_mul(b)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::InvalidArgument("YCbCr size overflows".into()))
        };
        let (yn, cn) = (size(w, h)?, size(cw.max(0), ch.max(0))?);
        if yn as u64 > limits.max_pixels {
            return Err(Error::PixelLimit);
        }
        let bytes = (yn as u64).saturating_add((cn as u64).saturating_mul(2));
        if bytes > limits.max_alloc {
            return Err(Error::AllocLimit);
        }
        let plane = |n: usize| -> Result<Vec<u8>, Error> {
            let mut v = Vec::new();
            v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
            v.resize(n, 0);
            Ok(v)
        };
        Ok(YCbCr {
            y: plane(yn)?,
            cb: plane(cn)?,
            cr: plane(cn)?,
            y_stride: w as usize,
            c_stride: cw.max(0) as usize,
            subsample,
            rect,
        })
    }

    /// Whether the planes and strides cover `rect`.
    pub(crate) fn fits(&self) -> bool {
        let (w, h) = (self.rect.dx() as usize, self.rect.dy() as usize);
        if w == 0 || h == 0 {
            return true;
        }
        let (cw, ch) = match self.subsample {
            Subsample::S444 => (w, h),
            Subsample::S420 => (
                chroma_len(self.rect.min_x, self.rect.max_x) as usize,
                chroma_len(self.rect.min_y, self.rect.max_y) as usize,
            ),
        };
        let covers = |len: usize, stride: usize, w: usize, h: usize| {
            stride >= w && (h - 1).saturating_mul(stride).saturating_add(w) <= len
        };
        covers(self.y.len(), self.y_stride, w, h)
            && covers(self.cb.len(), self.c_stride, cw, ch)
            && covers(self.cr.len(), self.c_stride, cw, ch)
    }

    /// `YCbCr.YOffset`. No limits apply; it cannot fail.
    pub fn y_offset(&self, x: i64, y: i64) -> usize {
        y.wrapping_sub(self.rect.min_y)
            .wrapping_mul(self.y_stride as i64)
            .wrapping_add(x.wrapping_sub(self.rect.min_x)) as usize
    }

    /// `YCbCr.COffset`: for 4:2:0, the chroma sample of the 2x2 block, grouped by the truncating
    /// `x/2` and `y/2` of absolute coordinates. No limits apply; it cannot fail.
    pub fn c_offset(&self, x: i64, y: i64) -> usize {
        let cs = self.c_stride as i64;
        let o = match self.subsample {
            Subsample::S444 => y
                .wrapping_sub(self.rect.min_y)
                .wrapping_mul(cs)
                .wrapping_add(x.wrapping_sub(self.rect.min_x)),
            Subsample::S420 => (y / 2 - self.rect.min_y / 2)
                .wrapping_mul(cs)
                .wrapping_add(x / 2 - self.rect.min_x / 2),
        };
        o as usize
    }
}

/// A source of bytes at an offset, as Go's `io.ReaderAt`. An `Image` reads codeblocks from it
/// concurrently.
pub trait ReadAt: Send + Sync {
    /// Reads up to `buf.len()` bytes at `offset` and returns how many (0 at the end).
    ///
    /// # Errors
    /// Any `io::Error` of the source; it is returned as [`Error::Io`] by the call that read. An
    /// `Interrupted` error is retried, and a short read is continued until the buffer is full or
    /// the source returns 0 (then [`Error::Io`] with `UnexpectedEof`).
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
}

impl ReadAt for [u8] {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let Ok(start) = usize::try_from(offset) else {
            return Ok(0);
        };
        let Some(src) = self.get(start..) else {
            return Ok(0);
        };
        let n = src.len().min(buf.len());
        buf[..n].copy_from_slice(&src[..n]);
        Ok(n)
    }
}

impl ReadAt for Vec<u8> {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.as_slice().read_at(buf, offset)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Arc<T> {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        (**self).read_at(buf, offset)
    }
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buf, offset)
    }
}

/// Fills `buf` from `r` at `off`, or fails (`io.ReaderAt`'s short read is an error here).
pub(crate) fn read_full(r: &dyn ReadAt, buf: &mut [u8], off: u64) -> Result<(), Error> {
    let mut done = 0usize;
    while done < buf.len() {
        match r.read_at(&mut buf[done..], off.saturating_add(done as u64)) {
            Ok(0) => return Err(Error::Io(io::ErrorKind::UnexpectedEof)),
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::Io(e.kind())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_new_takes_limits() {
        let r = Rect::new(0, 0, 100, 100);
        let px = Limits {
            max_pixels: 9999,
            ..Limits::default()
        };
        assert_eq!(
            YCbCr::new(r, Subsample::S444, &px).unwrap_err(),
            Error::PixelLimit
        );
        let mem = Limits {
            max_alloc: 29999,
            ..Limits::default()
        };
        assert_eq!(
            YCbCr::new(r, Subsample::S444, &mem).unwrap_err(),
            Error::AllocLimit
        );
        assert!(YCbCr::new(r, Subsample::S444, &Limits::default()).is_ok());
    }

    /// A codeblock of size 0 at the end of the file reads as nothing, whatever the reader does at
    /// the end (Go's `bytes.Reader` fails, its `os.File` does not): a deliberate deviation, see
    /// docs/parity.md.
    #[test]
    fn empty_read_at_the_end_is_ok() {
        let v = vec![1u8, 2, 3];
        assert!(read_full(&v, &mut [], 3).is_ok());
        assert!(read_full(&v, &mut [0u8; 1], 3).is_err());
    }
}
