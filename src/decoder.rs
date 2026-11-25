//! Tiles from codeblocks: tile dependencies (LL from the parent's inverse transform, recursively to
//! the top scale, and 4-coefficient margins from the 8 neighbours), the coefficient caches, the
//! memory budget. Port of `internal/decoder/image.go`; see isyntax-go's docs/format.md. Output
//! formats and regions are in `region.rs`.

use crate::budget::{Acct, Budget, Charge, lock};
use crate::cache::{Buf, Cache, Key};
use crate::container::{self, BLOCK_SIZE, Block, Header, Tile, TileKey};
use crate::error::check_cancel;
use crate::parallel::{Gauge, ThreadPool};
use crate::reuse::Decoder;
use crate::trunc::{TruncTable, parent_variant, sdk_preset};
use crate::types::{Level, Point, ReadAt, Rect, read_full};
use crate::{Error, Limits, dsp, hulsken};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// The codeblock side.
pub(crate) const BS: usize = BLOCK_SIZE as usize;
/// The tile side.
pub(crate) const TS: usize = 2 * BS;
pub(crate) const AREA: usize = BS * BS;
/// A quadrant of the inverse-transform buffer: a block plus two margins.
pub(crate) const Q: usize = BS + 8;
/// The inverse-transform buffer side.
pub(crate) const W: usize = 2 * Q;
/// The first output pixel in the buffer.
pub(crate) const FIRST: usize = 7;
pub(crate) const PLANE: usize = TS * TS;

/// 3 colours x HL, LH, HH.
const COEF_LEN: usize = 9 * AREA;
const RECON_LEN: usize = 3 * PLANE;
/// A block's four 4-wide strips.
const EDGE_LEN: usize = 16 * BS;

/// The background luma of a tile with nothing under it (the 9-bit background differs, see
/// isyntax-go's docs/format.md).
const FILL_Y: i16 = 255;

/// The codeblock read buffer a scratch keeps.
pub(crate) const READ_BYTES: usize = 1 << 18;
/// Bytes charged for a scratch, from its creation to its drop.
pub(crate) const SCRATCH_BYTES: u64 = (4 * W * W * 2 + 4 * AREA * 3 + READ_BYTES) as u64;
/// Bytes charged for a tile of RGBA staging.
pub(crate) const RGBA_BYTES: u64 = (4 * PLANE) as u64;

/// 64ths of the cache total, by kind.
const POOL_SHARE: [u64; 4] = [23, 26, 8, 7];

/// The floor for `default_cache_total`.
const MIN_CACHE_TOTAL: u64 = 256 << 20;

/// What a cache holds.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Coef = 0,
    Recon = 1,
    CoefEdge = 2,
    ReconEdge = 3,
}

/// A missing top-scale tile's LL.
static ZERO_LL: [i16; 3 * AREA] = [0; 3 * AREA];

/// The cache total: 1.125 MiB per level-0 tile column, at least 256 MiB, at most half the limit.
fn default_cache_total(grid_w: i64, limit: u64) -> u64 {
    let n = (grid_w.max(0) as u64)
        .saturating_mul(9 << 17)
        .max(MIN_CACHE_TOTAL);
    n.min(limit / 2)
}

fn new_pools(total: u64, budget: &Arc<Budget>) -> [Cache; 4] {
    std::array::from_fn(|k| {
        let mut c = Cache::new(total * POOL_SHARE[k] / 64, budget.clone());
        c.kind = k;
        c
    })
}

fn key(v: u8, s: usize, x: i64, y: i64) -> Key {
    Key {
        v,
        s: s as i32,
        x: x as i32,
        y: y as i32,
    }
}

/// An opened iSyntax file. It is safe for concurrent use: decodes share caches of wavelet
/// coefficients, charged to the memory limit (see [`Image::set_memory_limit`]). The limit
/// (`Limits::max_alloc`) is the enforced peak of every call on the image and its views together,
/// all threads included; `Limits::max_threads` caps the threads of one call.
pub struct Image {
    pub(crate) r: Arc<dyn ReadAt>,
    pub(crate) h: Arc<Header>,
    pub(crate) tiles: Arc<HashMap<TileKey, Tile>>,
    /// Hulsken 1 or 2.
    pub(crate) version: u8,
    /// 1 for 9-bit files.
    pub(crate) shift: u32,
    /// The colour planes are R, G, B: no YCoCg.
    pub(crate) planes: bool,
    /// One cache per [`Kind`].
    pub(crate) pools: [Cache; 4],
    pub(crate) scratch: Mutex<Vec<Box<Scratch>>>,
    /// The SDK default truncation; `None` is none.
    pub(crate) tt: Option<&'static TruncTable>,
    /// The recon variant of output tiles.
    pub(crate) out: u8,
    pub(crate) budget: Arc<Budget>,
    /// `Limits::max_threads`, at least 1.
    pub(crate) threads: usize,
    /// The caller's pool; `None` is the shared pool.
    pub(crate) pool: Option<ThreadPool>,
    /// The threads of this image's calls (views share it).
    pub(crate) gauge: Arc<Gauge>,
    /// Where freed buffers go and new ones come from; `None` allocates.
    pub(crate) dec: Option<Decoder>,
}

/// Pooled tile-decode working memory.
pub(crate) struct Scratch {
    /// Y, Co, Cg.
    idwt: [Vec<i16>; 3],
    tmp: Vec<i16>,
    hul: hulsken::Scratch,
    /// Up to `READ_BYTES` stays allocated; a larger codeblock is charged while it is read.
    pub(crate) read: Vec<u8>,
    /// `SCRATCH_BYTES`, held while this scratch exists, in use or pooled; `None` while it waits
    /// in a [`Decoder`].
    _charge: Option<Charge>,
}

/// A scratch in use: returned to its image's pool when dropped (it stays charged).
pub(crate) struct ScratchGuard<'a> {
    img: &'a Image,
    sc: Option<Box<Scratch>>,
}

impl Deref for ScratchGuard<'_> {
    type Target = Scratch;
    fn deref(&self) -> &Scratch {
        match &self.sc {
            Some(sc) => sc,
            None => unreachable_scratch(),
        }
    }
}

impl DerefMut for ScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut Scratch {
        match &mut self.sc {
            Some(sc) => sc,
            None => unreachable_scratch(),
        }
    }
}

/// The scratch is only taken out in `drop`; a guard used after that cannot exist.
#[cold]
fn unreachable_scratch() -> ! {
    std::process::abort()
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        if let Some(sc) = self.sc.take() {
            lock(&self.img.scratch).push(sc);
        }
    }
}

impl Drop for Image {
    /// The idle scratch goes to the decoder; the caches give back their entries as they drop.
    fn drop(&mut self) {
        let idle = std::mem::take(&mut *lock(&self.scratch));
        for sc in idle {
            self.recycle(sc);
        }
    }
}

/// What the header (the label, macro and ICC blobs, its strings, the envelopes) and the tile index hold.
fn kept_bytes(h: &Header, tiles: &HashMap<TileKey, Tile>) -> u64 {
    let blob = |b: &Option<Arc<Vec<u8>>>| b.as_ref().map_or(0, |v| v.len());
    let envelopes: usize = (h.envelopes.iter())
        .map(|p| size_of::<Vec<Point>>() + p.len() * size_of::<Point>())
        .sum();
    let index = tiles.capacity() * 2 * (size_of::<TileKey>() + size_of::<Tile>());
    let tokens: usize = (h.calibration_date.iter().chain(h.calibration_time.iter()))
        .map(|t| size_of::<String>() + t.len())
        .sum();
    let strings = [
        &h.interface_version,
        &h.model_name,
        &h.device_serial,
        &h.acquisition_date_time,
        &h.calibration_status,
    ]
    .iter()
    .map(|s| s.len())
    .sum::<usize>();
    (blob(&h.icc)
        + strings
        + blob(&h.label)
        + blob(&h.macro_image)
        + h.barcode.len()
        + envelopes
        + index
        + tokens) as u64
}

/// Tile (s, x, y) if it has codeblocks.
fn tile_in<'a>(
    h: &Header,
    tiles: &'a HashMap<TileKey, Tile>,
    s: usize,
    x: i64,
    y: i64,
) -> Option<&'a Tile> {
    if x < 0 || y < 0 || x >= h.grid_w >> s || y >= h.grid_h >> s {
        return None;
    }
    let t = tiles.get(&TileKey {
        scale: s as i32,
        x: x as i32,
        y: y as i32,
    })?;
    t.exists.then_some(t)
}

impl Image {
    /// Reads the header of the iSyntax file `reader[0..size]`. Opening reads only the header; the
    /// pixels are read by the decode calls.
    ///
    /// Limits: `max_header` bounds the XML header, `max_pixels` the image, and `max_alloc` the
    /// header parse and, from then on, the image's coefficient caches and the scratch of every
    /// call on it and its views; `max_threads` is kept for the calls ([`Image::with_pool`] picks
    /// the pool). The cancel flag is polled while the tile index is built.
    ///
    /// # Errors
    /// [`Error::HeaderLimit`], [`Error::PixelLimit`] and [`Error::AllocLimit`] for the limits
    /// above; [`Error::Cancelled`]; [`Error::Io`] from the reader; [`Error::Xml`],
    /// [`Error::Header`] and [`Error::Unsupported`] for a header that does not parse or is not
    /// decoded; [`Error::InvalidArgument`] if `size` is over 2^63 bytes;
    /// [`Error::OutOfMemory`] if the allocator refuses a size within the limits.
    pub fn open<R: ReadAt + 'static>(
        reader: R,
        size: u64,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Image, Error> {
        let size = i64::try_from(size)
            .map_err(|_| Error::InvalidArgument("file size over 2^63 bytes".into()))?;
        let r: Arc<dyn ReadAt> = Arc::new(reader);
        // The header parse and the tile index share one total, and `h` is still alive while the
        // index is built.
        let mut acct = Acct::with_cancel(limits.max_alloc, cancel);
        let mut h = container::parse(&*r, size, limits, cancel, &mut acct)?;
        let tiles = h.index(size, &mut acct)?;
        h.codeblocks = Vec::new(); // the tile index has them now
        occupancy_envelopes(&mut h, &tiles, &mut acct)?;
        let budget = Arc::new(Budget::new(limits.max_alloc));
        // What the image keeps of its header counts against the limit for its whole life; a
        // header that alone is over the limit does not open.
        budget.reserve(kept_bytes(&h, &tiles))?;
        let total = default_cache_total(h.grid_w, limits.max_alloc);
        Ok(Image {
            r,
            version: if h.compressor == 19 { 2 } else { 1 },
            shift: u32::from(h.bits_stored == 9),
            planes: h.rgb,
            pools: new_pools(total, &budget),
            scratch: Mutex::new(Vec::new()),
            tt: None,
            out: 0,
            tiles: Arc::new(tiles),
            h: Arc::new(h),
            budget,
            threads: limits.max_threads.max(1),
            pool: None,
            gauge: Arc::new(Gauge::new()),
            dec: None,
        })
    }

    /// Sets how many bytes this image and its views ([`Image::sdk_truncated`],
    /// [`Image::wide_grid`]) may hold together in coefficient caches and scratch, and returns the
    /// previous limit (`Limits::max_alloc` at first); `u64::MAX` is none. A decode that would pass
    /// the limit evicts its image's cached coefficients first, then fails with
    /// [`Error::MemoryLimit`] rather than allocate; one that needs more than the whole limit fails
    /// with [`Error::AllocLimit`]. Go's `SetMemoryLimit` is process-wide; this one is per image.
    /// It replaces `Limits::max_alloc` for the image and its views; it cannot fail.
    pub fn set_memory_limit(&self, n: u64) -> u64 {
        self.budget.set_limit(n)
    }

    /// This image with `pool` as the threads of its calls (and its views'), in place of the
    /// crate's shared pool: a caller's pool always wins. `Limits::max_threads` still caps the
    /// workers of a call, the calling thread included, and a call uses at most one more than the
    /// pool has threads. The pool is not charged to `max_alloc`. It cannot fail.
    #[must_use]
    pub fn with_pool(mut self, pool: ThreadPool) -> Image {
        self.pool = Some(pool);
        self
    }

    /// This image with `d` as the store of its buffers (and its views'): it takes the coefficient
    /// cache entries, the scratch, the staging buffers and the cache tables it needs from `d`
    /// where it would allocate them, and gives them back when it drops or evicts them. The
    /// decodes of an image whose `d` is warm allocate nothing (with `max_threads` 1; see
    /// [`Decoder`]). Every buffer is still charged to this image's `max_alloc` where a new one
    /// would be, so the limits, the errors, the output and the budget's peak do not change.
    /// What `d` keeps between images is not charged to any of them. It cannot fail.
    #[must_use]
    pub fn with_decoder(mut self, d: &Decoder) -> Image {
        for c in &mut self.pools {
            c.set_decoder(d.clone());
        }
        // Room for the scratch of a call per thread, so the pool of an image does not allocate.
        let _ = lock(&self.scratch).try_reserve(self.threads.min(64));
        self.dec = Some(d.clone());
        self
    }

    /// A view of this image with its own caches, `total` bytes in all.
    fn view(&self, h: Arc<Header>, total: u64) -> Image {
        let mut pools = new_pools(total, &self.budget);
        let mut scratch = Vec::new();
        if let Some(d) = &self.dec {
            for c in &mut pools {
                c.set_decoder(d.clone());
            }
            // As in `with_decoder`: the pool of a view does not allocate either.
            let _ = scratch.try_reserve(self.threads.min(64));
        }
        Image {
            r: self.r.clone(),
            h,
            tiles: self.tiles.clone(),
            version: self.version,
            shift: self.shift,
            planes: self.planes,
            pools,
            scratch: Mutex::new(scratch),
            tt: self.tt,
            out: self.out,
            budget: self.budget.clone(),
            threads: self.threads,
            pool: self.pool.clone(),
            gauge: self.gauge.clone(),
            dec: self.dec.clone(),
        }
    }

    /// A view that decodes like the Philips SDK SourceView with default parameters, which
    /// truncates detail coefficients of Q1 and Q2 files (isyntax-go docs/format.md, "SDK default
    /// truncation"). `self` is unchanged. The view shares the file, so the file must stay open
    /// while either is used. It has its own caches, sized as [`Image::open`] sizes them now, and
    /// both count against the same memory limit. [`Error::Unsupported`] when the file's quality
    /// preset and bit depth have no measured SDK table. The view inherits `max_threads` and the
    /// pool; its caches count against `max_alloc` with the image's.
    ///
    /// # Errors
    /// [`Error::Unsupported`] as above.
    pub fn sdk_truncated(&self) -> Result<Image, Error> {
        let tt = sdk_preset(&self.h)?;
        let mut v = self.view(
            self.h.clone(),
            default_cache_total(self.h.grid_w, self.budget.limit()),
        );
        v.tt = tt;
        v.out = u8::from(tt.is_some());
        Ok(v)
    }

    /// A view whose native tile grid is one top-scale tile wider and taller, on the right and
    /// bottom. The added tiles decode as tiles missing from the file (isyntax-go docs/format.md):
    /// zero detail, the parent's LL, zero LL at the top scale, where `self`'s tile decodes are
    /// white. The Philips SDK display filters read these pixels past the grid. Every other pixel
    /// decodes as in `self`, and the view's [`Image::level`] reports the wider grid. The view has
    /// its own caches, an eighth of `self`'s size, charged to the same `max_alloc`; it inherits
    /// `max_threads` and the pool. It cannot fail.
    pub fn wide_grid(&self) -> Image {
        let mut h = (*self.h).clone();
        let extra = 1i64 << (h.levels - 1);
        h.grid_w += extra;
        h.grid_h += extra;
        self.view(
            Arc::new(h),
            default_cache_total(self.h.grid_w, self.budget.limit()) / 8,
        )
    }

    /// The number of resolution levels: the native levels plus one, the SDK's derived level, the
    /// top scale's own LL, at half its size.
    ///
    /// No limits apply.
    pub fn levels(&self) -> usize {
        self.h.levels + 1
    }

    /// Level `i`, in `0..levels()`; `None` outside. Level `levels() - 1` is the derived level:
    /// the top scale's LL codeblocks, one 128x128 tile per top-scale tile, at the same grid.
    ///
    /// No limits apply.
    pub fn level(&self, i: usize) -> Option<Level> {
        if i >= self.levels() {
            return None;
        }
        if i == self.h.levels {
            let top = self.level(i - 1)?;
            return Some(Level {
                width: top.width / 2,
                height: top.height / 2,
                tiles_x: top.tiles_x,
                tiles_y: top.tiles_y,
                downsample: 2 * top.downsample,
            });
        }
        Some(Level {
            width: (self.h.width >> i) as usize,
            height: (self.h.height >> i) as usize,
            tiles_x: (self.h.grid_w >> i) as usize,
            tiles_y: (self.h.grid_h >> i) as usize,
            downsample: 1 << i,
        })
    }

    /// The side of a native tile: 256 at every level below `levels() - 1`, whose tiles are 128.
    ///
    /// No limits apply.
    pub fn tile_size(&self) -> usize {
        TS
    }

    /// 8, 9 or 16: iSyntax2 files are 8 or 9, v1 files may be 8, 9 or 16. 9-bit samples are
    /// halved to 8 bits in RGBA and YCbCr output.
    ///
    /// No limits apply.
    pub fn bits_stored(&self) -> u32 {
        self.h.bits_stored as u32
    }

    /// The label image's JPEG bytes.
    ///
    /// No limits apply.
    pub fn label(&self) -> Option<&[u8]> {
        self.h.label.as_deref().map(Vec::as_slice)
    }

    /// The macro (overview) image's JPEG bytes.
    ///
    /// No limits apply.
    pub fn macro_image(&self) -> Option<&[u8]> {
        self.h.macro_image.as_deref().map(Vec::as_slice)
    }

    /// The WSI image's ICC profile. Decoded pixels are not colour-managed.
    ///
    /// No limits apply.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.h.icc.as_deref().map(Vec::as_slice)
    }

    /// The size of a level-0 pixel in micrometres, 0 when the file does not say.
    ///
    /// No limits apply.
    pub fn mpp(&self) -> (f64, f64) {
        (self.h.mpp_x, self.h.mpp_y)
    }

    /// The scanned barcode (bytes, not necessarily UTF-8), empty if none. Sensitive: callers must
    /// not log or persist it outside access-controlled storage.
    ///
    /// No limits apply.
    pub fn barcode(&self) -> &[u8] {
        &self.h.barcode
    }

    /// The scanner model, empty if none.
    ///
    /// No limits apply.
    pub fn model_name(&self) -> &str {
        &self.h.model_name
    }

    /// The scanner's serial number, empty if none.
    ///
    /// No limits apply.
    pub fn device_serial(&self) -> &str {
        &self.h.device_serial
    }

    /// The DICOM acquisition timestamp, empty if none.
    ///
    /// No limits apply.
    pub fn acquisition_date_time(&self) -> &str {
        &self.h.acquisition_date_time
    }

    /// The scanner calibration status, empty if none.
    ///
    /// No limits apply.
    pub fn calibration_status(&self) -> &str {
        &self.h.calibration_status
    }

    /// The IStringArray tokens, empty if absent.
    ///
    /// No limits apply.
    pub fn date_of_last_calibration(&self) -> &[String] {
        &self.h.calibration_date
    }

    /// The IStringArray tokens, empty if absent.
    ///
    /// No limits apply.
    pub fn time_of_last_calibration(&self) -> &[String] {
        &self.h.calibration_time
    }

    /// The scanner rack and slot numbers, `None` when absent.
    ///
    /// No limits apply.
    pub fn rack_slot(&self) -> (Option<u16>, Option<u16>) {
        (
            u16::try_from(self.h.rack).ok(),
            u16::try_from(self.h.slot).ok(),
        )
    }

    /// The raw `PIM_DP_UFS_INTERFACE_VERSION`, e.g. "5.0" or "100.5"; empty if none.
    ///
    /// No limits apply.
    pub fn interface_version(&self) -> &str {
        &self.h.interface_version
    }

    /// `level`'s rectangle in level pixels: the SDK SourceView's dimension ranges, background
    /// border excluded, with `min` the level's first valid column and row.
    ///
    /// No limits apply.
    pub fn source_bounds(&self, level: usize) -> Rect {
        pyramid_shrink(Rect::new(0, 0, self.h.width, self.h.height), level).unwrap_or_default()
    }

    /// `UFS_IMAGE_VALID_DATA_ENVELOPES`, the file's polygons in the file's vertex order, moved to
    /// image coordinates (the level-0 padding subtracted) and scaled to level pixels (each vertex
    /// divided by the level's downsample). A v1 file without the tag gets the whole image, or
    /// rectangles over its present scale-0 tiles when some are missing. Pixels of
    /// [`source_bounds`](Image::source_bounds) outside every envelope are background. Computed
    /// as it is read: polygons of vertices, nothing is copied.
    ///
    /// No limits apply.
    pub fn envelopes(
        &self,
        level: usize,
    ) -> impl Iterator<Item = impl Iterator<Item = Point> + '_> + '_ {
        let step = u32::try_from(level)
            .ok()
            .and_then(|l| 1i64.checked_shl(l))
            .filter(|&s| s > 0)
            .unwrap_or(i64::MAX);
        self.h.envelopes.iter().map(move |poly| {
            poly.iter().map(move |v| Point {
                x: v.x / step,
                y: v.y / step,
            })
        })
    }

    /// Calls `emit(row, plane0, plane1, plane2)` for each row of tile `(level, x, y)`'s three
    /// colour planes: `TS` rows of `TS` samples, or `BS` of `BS` at the derived level. The tile
    /// must be in the grid; a tile missing from the file decodes as zero detail over the parent's LL.
    #[allow(clippy::type_complexity)]
    pub(crate) fn tile_rows(
        &self,
        level: usize,
        x: i64,
        y: i64,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(usize, &[i16], &[i16], &[i16]),
    ) -> Result<(), Error> {
        if level == self.h.levels {
            // The derived level: the top scale's LL.
            let top = self.h.levels - 1;
            let cf;
            let ll: &[i16] = if self.tile(top, x, y).is_some() {
                cf = self.coef(top, x, y, cancel)?;
                &cf.data[COEF_LEN..]
            } else {
                &ZERO_LL
            };
            for r in 0..BS {
                let o = r * BS;
                emit(
                    r,
                    &ll[o..o + BS],
                    &ll[AREA + o..AREA + o + BS],
                    &ll[2 * AREA + o..2 * AREA + o + BS],
                );
            }
            return Ok(());
        }
        if level > 0 {
            let rec = self.recon(level, x, y, self.out, cancel)?;
            // An output, not a centre.
            self.pools[Kind::Recon as usize].demote(&key(self.out, level, x, y));
            for r in 0..TS {
                emit(
                    r,
                    &rec.data[r * TS..][..TS],
                    &rec.data[PLANE + r * TS..][..TS],
                    &rec.data[2 * PLANE + r * TS..][..TS],
                );
            }
            return Ok(());
        }
        let d = self.gather(0, x, y, self.out, cancel)?;
        let mut sc = self.get_scratch()?;
        self.transform(&d, &mut sc, self.out);
        for r in 0..TS {
            let o = (FIRST + r) * W + FIRST;
            emit(
                r,
                &sc.idwt[0][o..][..TS],
                &sc.idwt[1][o..][..TS],
                &sc.idwt[2][o..][..TS],
            );
        }
        Ok(())
    }

    /// Charges `n` bytes, evicting this image's cached coefficients first (the SDK-truncated and
    /// wide views have their own).
    pub(crate) fn reserve(&self, n: u64) -> Result<Charge, Error> {
        loop {
            match Charge::new(&self.budget, n) {
                Err(Error::MemoryLimit) => {
                    let mut evicted = false;
                    for c in &self.pools {
                        evicted = evicted || c.evict_oldest(None);
                    }
                    // Then a pooled scratch nobody is using.
                    let idle = if evicted {
                        None
                    } else {
                        lock(&self.scratch).pop()
                    };
                    if let Some(sc) = idle {
                        self.recycle(sc);
                        evicted = true;
                    }
                    if !evicted {
                        return Err(Error::MemoryLimit);
                    }
                }
                r => return r,
            }
        }
    }

    /// Releases `sc`'s charge and gives it to the decoder, if there is one.
    fn recycle(&self, mut sc: Box<Scratch>) {
        sc._charge = None;
        if let Some(d) = &self.dec {
            d.put_scratch(sc);
        }
    }

    /// Tile `(s, x, y)`'s codeblocks, if it has any.
    pub(crate) fn tile(&self, s: usize, x: i64, y: i64) -> Option<&Tile> {
        tile_in(&self.h, &self.tiles, s, x, y)
    }

    /// Whether `(x, y)` is in scale `s`'s grid.
    pub(crate) fn in_grid(&self, s: usize, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && x < self.h.grid_w >> s && y < self.h.grid_h >> s
    }

    /// Pooled working memory for one tile decode.
    pub(crate) fn get_scratch(&self) -> Result<ScratchGuard<'_>, Error> {
        let pooled = lock(&self.scratch).pop();
        let sc = match pooled {
            Some(sc) => sc,
            None => {
                let charge = self.reserve(SCRATCH_BYTES)?;
                match self.dec.as_ref().and_then(|d| d.take_scratch()) {
                    Some(mut sc) => {
                        sc._charge = Some(charge);
                        sc
                    }
                    None => Box::new(Scratch {
                        idwt: std::array::from_fn(|_| vec![0; W * W]),
                        tmp: vec![0; W * W],
                        hul: hulsken::Scratch::new(),
                        read: Vec::new(),
                        _charge: Some(charge),
                    }),
                }
            }
        };
        Ok(ScratchGuard {
            img: self,
            sc: Some(sc),
        })
    }

    /// A tile's coefficients, per colour HL, LH, HH (and LL at the top scale).
    pub(crate) fn coef(
        &self,
        s: usize,
        x: i64,
        y: i64,
        cancel: &AtomicBool,
    ) -> Result<Arc<Buf>, Error> {
        let mut n = COEF_LEN;
        if s == self.h.levels - 1 {
            n += 3 * AREA;
        }
        self.pools[Kind::Coef as usize].get(key(0, s, x, y), n, cancel, |data| {
            let Some(t) = self.tile(s, x, y) else {
                return Err(Error::Internal(format!("tile ({s}, {x}, {y}) is absent")));
            };
            let mut sc = self.get_scratch()?;
            for (c, (h, ll)) in t.h.iter().zip(&t.ll).enumerate() {
                let dst = &mut data[c * 3 * AREA..(c + 1) * 3 * AREA];
                self.codeblock(cancel, &mut sc, *h, dst, 3)?;
                if n > COEF_LEN {
                    let dst = &mut data[COEF_LEN + c * AREA..COEF_LEN + (c + 1) * AREA];
                    self.codeblock(cancel, &mut sc, *ll, dst, 1)?;
                }
            }
            Ok(())
        })
    }

    pub(crate) fn codeblock(
        &self,
        cancel: &AtomicBool,
        sc: &mut Scratch,
        b: Block,
        dst: &mut [i16],
        n: usize,
    ) -> Result<(), Error> {
        check_cancel(cancel)?;
        let size = usize::try_from(b.size).unwrap_or(0);
        // The scratch is charged for `READ_BYTES` of it; the rest of a large codeblock is charged
        // for as long as it is held.
        let extra = size.saturating_sub(READ_BYTES);
        let _extra = if extra > 0 {
            Some(self.reserve(extra as u64)?)
        } else {
            None
        };
        let r = self.read_decode(sc, b, size, dst, n);
        // On every exit, a failed read included: the oversize buffer is charged only until here.
        if sc.read.capacity() > READ_BYTES {
            sc.read = Vec::new();
        }
        r.map_err(|e| match e {
            Error::Codeblock(m) => Error::Codeblock(format!("at {}: {m}", b.offset)),
            other => other,
        })
    }

    fn read_decode(
        &self,
        sc: &mut Scratch,
        b: Block,
        size: usize,
        dst: &mut [i16],
        n: usize,
    ) -> Result<(), Error> {
        sc.read.clear();
        sc.read
            .try_reserve_exact(size)
            .map_err(|_| Error::OutOfMemory)?;
        sc.read.resize(size, 0);
        read_full(&*self.r, &mut sc.read, b.offset as u64)?;
        hulsken::decode(dst, &sc.read, n, BS, BS, self.version, &mut sc.hul)
    }

    /// Inverse transform of tile `(s, x, y)`, variant `v`'s truncation: 3 planes of `TS x TS`.
    pub(crate) fn recon(
        &self,
        s: usize,
        x: i64,
        y: i64,
        v: u8,
        cancel: &AtomicBool,
    ) -> Result<Arc<Buf>, Error> {
        self.pools[Kind::Recon as usize].get(key(v, s, x, y), RECON_LEN, cancel, |data| {
            let d = self.gather(s, x, y, v, cancel)?;
            let mut sc = self.get_scratch()?;
            self.transform(&d, &mut sc, v);
            for (c, idwt) in sc.idwt.iter().enumerate() {
                for r in 0..TS {
                    data[c * PLANE + r * TS..][..TS]
                        .copy_from_slice(&idwt[(FIRST + r) * W + FIRST..][..TS]);
                }
            }
            Ok(())
        })
    }

    /// The margins of `coef` that the neighbours read.
    pub(crate) fn coef_edge(
        &self,
        s: usize,
        x: i64,
        y: i64,
        cancel: &AtomicBool,
    ) -> Result<Arc<Buf>, Error> {
        let nb = if s == self.h.levels - 1 { 12 } else { 9 };
        self.pools[Kind::CoefEdge as usize].get(key(0, s, x, y), nb * EDGE_LEN, cancel, |e| {
            let cf = self.coef(s, x, y, cancel)?;
            for b in 0..nb {
                put_edge(&mut e[b * EDGE_LEN..], &cf.data[b * AREA..], BS);
            }
            Ok(())
        })
    }

    /// The margins of `recon`'s quadrants, colour-major.
    pub(crate) fn recon_edge(
        &self,
        s: usize,
        x: i64,
        y: i64,
        v: u8,
        cancel: &AtomicBool,
    ) -> Result<Arc<Buf>, Error> {
        self.pools[Kind::ReconEdge as usize].get(key(v, s, x, y), 12 * EDGE_LEN, cancel, |e| {
            let rec = self.recon(s, x, y, v, cancel)?;
            for b in 0..12 {
                let q = b % 4;
                put_edge(
                    &mut e[b * EDGE_LEN..],
                    &rec.data[b / 4 * PLANE + q / 2 * BS * TS + q % 2 * BS..],
                    TS,
                );
            }
            Ok(())
        })
    }

    /// What tile `(s, x, y)` reads: the full centre and the neighbours' edges.
    fn gather(&self, s: usize, x: i64, y: i64, v: u8, cancel: &AtomicBool) -> Result<Deps, Error> {
        let mut d = Deps::default();
        let top = self.h.levels - 1;
        let pv = parent_variant(v);
        for i in 0..9usize {
            let (nx, ny) = (x + (i % 3) as i64 - 1, y + (i / 3) as i64 - 1);
            if !self.in_grid(s, nx, ny) {
                continue; // the edge: zero margin, or Y fill
            }
            let real = self.tile(s, nx, ny).is_some();
            if real {
                let cf = if i == 4 {
                    let cf = self.coef(s, nx, ny, cancel)?;
                    self.retire(Kind::Coef, key(0, s, nx, ny), 1, cancel)?;
                    cf
                } else {
                    self.coef_edge(s, nx, ny, cancel)?
                };
                d.h[i] = Some(cf);
            }
            // A missing tile: zero detail, the parent's LL.
            if s == top {
                d.ll[i] = match (&d.h[i], i == 4) {
                    (Some(cf), true) => LlSrc::of(cf, COEF_LEN, AREA, 0, BS),
                    (Some(cf), false) => LlSrc::of(cf, 0, EDGE_LEN, 9 * EDGE_LEN, 0),
                    (None, _) => LlSrc {
                        zero: true,
                        stride: BS,
                        ..LlSrc::default()
                    }, // rule 3
                };
                continue;
            }
            let (px, py) = (nx / 2, ny / 2);
            if !self.in_grid(s + 1, px, py) {
                continue;
            }
            if i == 4 {
                let rec = self.recon(s + 1, px, py, pv, cancel)?;
                let n = (2i64.min((self.h.grid_w >> s) - px * 2)
                    * 2i64.min((self.h.grid_h >> s) - py * 2)) as i32;
                self.retire(Kind::Recon, key(pv, s + 1, px, py), n, cancel)?;
                let off = (ny % 2) as usize * BS * TS + (nx % 2) as usize * BS;
                d.ll[i] = LlSrc::of(&rec, 0, PLANE, off, TS);
                continue;
            }
            let e = self.recon_edge(s + 1, px, py, pv, cancel)?;
            let off = ((ny % 2) * 2 + nx % 2) as usize * EDGE_LEN;
            d.ll[i] = LlSrc::of(&e, 0, 4 * EDGE_LEN, off, 0);
        }
        Ok(d)
    }

    /// After the `n`th centre use of an entry, keeps its edge and evicts the full entry first.
    fn retire(&self, kind: Kind, k: Key, n: i32, cancel: &AtomicBool) -> Result<(), Error> {
        let pool = &self.pools[kind as usize];
        if !pool.used(&k, n) {
            return Ok(());
        }
        let (s, x, y) = (k.s as usize, i64::from(k.x), i64::from(k.y));
        match kind {
            Kind::Coef => drop(self.coef_edge(s, x, y, cancel)?),
            Kind::Recon => drop(self.recon_edge(s, x, y, k.v, cancel)?),
            _ => {}
        }
        pool.demote(&k);
        Ok(())
    }

    /// Stitches the margins, truncates per `v`, inverts.
    fn transform(&self, d: &Deps, sc: &mut Scratch, v: u8) {
        let Scratch { idwt, tmp, .. } = sc;
        for (c, buf) in idwt.iter_mut().enumerate() {
            for i in 0..9 {
                let ll = &d.ll[i];
                if let Some(src) = ll.data() {
                    blit(buf, 0, 0, &src[c * ll.plane_len + ll.off..], ll.stride, i);
                } else {
                    fill(buf, 0, 0, i, if c == 0 { FILL_Y } else { 0 });
                }
                let Some(h) = &d.h[i] else {
                    // A virtual tile: zero detail.
                    fill(buf, 0, Q, i, 0);
                    fill(buf, Q, 0, i, 0);
                    fill(buf, Q, Q, i, 0);
                    continue;
                };
                let n = if i == 4 { AREA } else { EDGE_LEN };
                let hd = &h.data[c * 3 * n..];
                blit(buf, 0, Q, &hd[..n], BS, i); // HL
                blit(buf, Q, 0, &hd[n..2 * n], BS, i); // LH
                blit(buf, Q, Q, &hd[2 * n..3 * n], BS, i); // HH
            }
            if v != 0
                && let Some(tt) = self.tt
            {
                tt.apply(buf, usize::from(v) - 1, c);
            }
            dsp::inverse53(buf, tmp, Q);
        }
    }
}

/// An unfinished tile's inputs: the full centre, the neighbours' edges.
#[derive(Default)]
struct Deps {
    h: [Option<Arc<Buf>>; 9],
    ll: [LlSrc; 9],
}

/// A tile's LL source: `buf[base..]`, or zeros, or (with neither) the fill.
#[derive(Default)]
struct LlSrc {
    buf: Option<Arc<Buf>>,
    zero: bool,
    base: usize,
    plane_len: usize,
    off: usize,
    stride: usize,
}

impl LlSrc {
    fn of(buf: &Arc<Buf>, base: usize, plane_len: usize, off: usize, stride: usize) -> LlSrc {
        LlSrc {
            buf: Some(buf.clone()),
            zero: false,
            base,
            plane_len,
            off,
            stride,
        }
    }

    fn data(&self) -> Option<&[i16]> {
        if self.zero {
            return Some(&ZERO_LL[..]);
        }
        self.buf.as_ref().map(|b| &b.data[self.base..])
    }
}

/// The margins neighbours read: 4 rows top and bottom, 4 columns left and right (by row).
fn put_edge(e: &mut [i16], src: &[i16], stride: usize) {
    for k in 0..4 {
        e[k * BS..][..BS].copy_from_slice(&src[k * stride..][..BS]);
        e[(4 + k) * BS..][..BS].copy_from_slice(&src[(BS - 4 + k) * stride..][..BS]);
    }
    for k in 0..BS {
        let row = &src[k * stride..][..BS];
        e[8 * BS + 4 * k..][..4].copy_from_slice(&row[..4]);
        e[12 * BS + 4 * k..][..4].copy_from_slice(&row[BS - 4..]);
    }
}

/// Neighbour span `[lo, hi)` lands at `at`.
struct Span {
    lo: usize,
    hi: usize,
    at: usize,
}

const SPAN: [Span; 3] = [
    Span {
        lo: BS - 4,
        hi: BS,
        at: 0,
    },
    Span {
        lo: 0,
        hi: BS,
        at: 4,
    },
    Span {
        lo: 0,
        hi: 4,
        at: BS + 4,
    },
];

/// Neighbour `i`'s block (the centre, or the margin strips the neighbour keeps): start and stride.
fn view(blk: &[i16], stride: usize, i: usize) -> (&[i16], usize) {
    match i {
        4 => (blk, stride),
        // Above: its bottom rows.
        0..=2 => (&blk[4 * BS + SPAN[i % 3].lo..], BS),
        // Below: its top rows.
        6..=8 => (&blk[SPAN[i % 3].lo..], BS),
        // Left: its right columns.
        3 => (&blk[12 * BS..], 4),
        // Right: its left columns.
        _ => (&blk[8 * BS..], 4),
    }
}

/// Copies neighbour `i`'s span of `blk` into quadrant `(qr, qc)` of the buffer.
fn blit(dst: &mut [i16], qr: usize, qc: usize, blk: &[i16], stride: usize, i: usize) {
    let (src, stride) = view(blk, stride, i);
    let (r, c) = (&SPAN[i / 3], &SPAN[i % 3]);
    let n = c.hi - c.lo;
    for k in 0..r.hi - r.lo {
        let o = (qr + r.at + k) * W + qc + c.at;
        dst[o..o + n].copy_from_slice(&src[k * stride..][..n]);
    }
}

/// Fills neighbour `i`'s span of quadrant `(qr, qc)`. The spans tile the quadrant, so only gaps
/// need it.
fn fill(dst: &mut [i16], qr: usize, qc: usize, i: usize, v: i16) {
    let (r, c) = (&SPAN[i / 3], &SPAN[i % 3]);
    for k in 0..r.hi - r.lo {
        let o = (qr + r.at + k) * W + qc + c.at;
        dst[o..o + (c.hi - c.lo)].fill(v);
    }
}

/// `r` shrunk per level by the wavelet borders; `None` once it is empty.
fn pyramid_shrink(r: Rect, level: usize) -> Option<Rect> {
    let (mut min_x, mut max_x) = (r.min_x, r.max_x - 1);
    let (mut min_y, mut max_y) = (r.min_y, r.max_y - 1);
    for _ in 0..level {
        (min_x, max_x) = ((min_x + 1) / 2 + 1, max_x / 2 - 1);
        (min_y, max_y) = ((min_y + 1) / 2 + 1, max_y / 2 - 1);
        if min_x > max_x || min_y > max_y {
            return None;
        }
    }
    Some(Rect::new(min_x, min_y, max_x + 1, max_y + 1))
}

/// Replaces a v1 file's default envelope with rectangles over the present scale-0 tiles, a row's
/// adjacent tiles merged, when any scale-0 tile over the image is missing or sits under a missing
/// top-scale tile: the SDK leaves such tiles out of the data envelope (isyntax-go docs/format.md,
/// "SourceBounds and Envelopes").
pub(crate) fn occupancy_envelopes(
    h: &mut Header,
    tiles: &HashMap<TileKey, Tile>,
    acct: &mut Acct,
) -> Result<(), Error> {
    if h.major >= 100 || !h.envelope_default {
        return Ok(());
    }
    let top = h.levels - 1;
    let off = (3i64 << h.levels) - 2 - 1; // the region rule 1 at level 0
    let ts = TS as i64;
    let bounds = Rect::new(0, 0, h.width, h.height);
    let mut env: Vec<Vec<Point>> = Vec::new();
    let mut missing = false;
    let flush = |cur: &mut Rect, env: &mut Vec<Vec<Point>>, acct: &mut Acct| -> Result<(), Error> {
        if !cur.is_empty() {
            acct.add((size_of::<Vec<Point>>() + 4 * size_of::<Point>()) as u64)?;
            env.push(vec![
                Point {
                    x: cur.min_x,
                    y: cur.min_y,
                },
                Point {
                    x: cur.max_x,
                    y: cur.min_y,
                },
                Point {
                    x: cur.max_x,
                    y: cur.max_y,
                },
                Point {
                    x: cur.min_x,
                    y: cur.max_y,
                },
            ]);
        }
        *cur = Rect::default();
        Ok(())
    };
    // Tiles past the image are empty after the intersect; skipping them keeps the work to the
    // image, whatever a hostile header says of the grid.
    let x_end = ((h.width + off + ts - 1) / ts).min(h.grid_w);
    let y_end = ((h.height + off + ts - 1) / ts).min(h.grid_h);
    for y in 0..y_end {
        let mut cur = Rect::default();
        for x in 0..x_end {
            acct.tick()?;
            let r = Rect::new(
                x * ts - off,
                y * ts - off,
                (x + 1) * ts - off,
                (y + 1) * ts - off,
            )
            .intersect(bounds);
            if r.is_empty() {
                continue;
            }
            // Absent subtrees may list blocks.
            if tile_in(h, tiles, 0, x, y).is_none()
                || tile_in(h, tiles, top, x >> top, y >> top).is_none()
            {
                missing = true;
                flush(&mut cur, &mut env, acct)?;
                continue;
            }
            if cur.is_empty() {
                cur = r;
            } else {
                cur.max_x = r.max_x;
            }
        }
        flush(&mut cur, &mut env, acct)?;
    }
    if missing {
        h.envelopes = Arc::new(env);
    }
    Ok(())
}
