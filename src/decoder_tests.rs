//! Tests of the tile and region decoder on synthetic files: port of `internal/decoder`'s and
//! `internal/isyntax`'s tests that do not need slide data (`ISYNTAX_SLIDES`, `ISYNTAX_ORACLE`).

use crate::budget::{Acct, lock};
use crate::cache::{Cache, Key};
use crate::container::{Block, Header, Tile, TileKey};
use crate::decoder::{AREA, BS, Image, Kind, READ_BYTES, SCRATCH_BYTES, TS, occupancy_envelopes};
use crate::dsp::rgb_to_ycbcr;
use crate::fixture::{self, FileSpec};
use crate::testutil::Rng;
use crate::trunc::{TRUNC_Q1_RGB, TRUNC_Q2B8, TRUNC_Q2B9, TruncTable, sdk_preset};
use crate::types::{Level, Point, ReadAt, Rect, Subsample, YCbCr};
use crate::{Decoder, Error, Limits};
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

fn go() -> AtomicBool {
    AtomicBool::new(false)
}

/// Flat grey, LL only.
fn const_spec(y: i16, bits: u32) -> FileSpec {
    let mut s = FileSpec::new(move |_, _, _, c, ll| {
        if !ll {
            return vec![0; 3 * AREA];
        }
        let mut v = vec![0; AREA];
        if c == 0 {
            v.fill(y);
        }
        v
    });
    s.compressor = 19;
    s.bits_stored = bits;
    s
}

/// Random detail, LL, missing tiles.
fn rand_spec(compressor: u32) -> FileSpec {
    let mut s = FileSpec::new(|sc, x, y, c, ll| {
        let mut r = Rng::new((sc * 1000 + x * 31 + y) as u64 * 8 + c as u64 + 1);
        let n = if ll { 1 } else { 3 };
        let mut v = vec![0i16; n * AREA];
        for e in &mut v {
            *e = if ll && c == 0 {
                100 + r.below(120) as i16
            } else if ll {
                r.below(60) as i16 - 30
            } else if r.below(4) == 0 {
                r.below(40) as i16 - 20
            } else {
                0
            };
        }
        v
    });
    s.compressor = compressor;
    s.offset_x = 3;
    s.offset_y = 5;
    s.set_exists(|s, x, y| s > 0 || !(x * 7 + y * 3).is_multiple_of(5));
    s
}

/// A v1 header, `rand_spec`'s layout.
fn v1_spec() -> FileSpec {
    let mut s = rand_spec(16);
    s.version = "5.0".into();
    s.bits_stored = 16;
    s
}

/// Samples exceed the clamp range.
fn wide_spec(compressor: u32, bits: u32) -> FileSpec {
    let mut s = rand_spec(compressor);
    s.bits_stored = bits;
    let base = s.coef.clone();
    s.set_coef(move |sc, x, y, c, ll| {
        let mut v = base(sc, x, y, c, ll);
        if ll {
            let mut r = Rng::new((sc * 1000 + x * 31 + y) as u64 * 8 + c as u64 + 100);
            for e in &mut v {
                *e = if c == 0 {
                    r.below(900) as i16 - 200
                } else {
                    r.below(500) as i16 - 250
                };
            }
        }
        v
    });
    s
}

fn open(s: &FileSpec) -> Image {
    let (file, _) = fixture::file(s);
    let n = file.len() as u64;
    Image::open(file, n, &Limits::default(), &go()).unwrap()
}

fn side(m: &Image, level: usize) -> usize {
    if level == m.h.levels { BS } else { TS }
}

fn tile_rgba(m: &Image, level: usize, x: i64, y: i64) -> Vec<u8> {
    let s = side(m, level);
    let mut dst = vec![0u8; 4 * s * s];
    m.decode_tile_rgba(level, x, y, &mut dst, 4 * s, &go())
        .unwrap();
    dst
}

fn lv(m: &Image, level: usize) -> Level {
    m.level(level).unwrap()
}

fn is_white(px: &[u8]) -> bool {
    px.iter().all(|&v| v == 255)
}

#[test]
fn geometry() {
    let m = open(&rand_spec(19));
    assert_eq!((m.levels(), m.tile_size(), m.bits_stored()), (4, 256, 8));
    // Level 3: derived, the top scale's LL.
    let want = [
        (1500, 900, 8, 4, 1),
        (750, 450, 4, 2, 2),
        (375, 225, 2, 1, 4),
        (187, 112, 2, 1, 8),
    ];
    for (i, (width, height, tiles_x, tiles_y, downsample)) in want.into_iter().enumerate() {
        let w = Level {
            width,
            height,
            tiles_x,
            tiles_y,
            downsample,
        };
        assert_eq!(m.level(i), Some(w), "level {i}");
    }
    assert_eq!(m.level(4), None);
    assert_eq!(m.mpp(), (0.25, 0.25));
}

#[test]
fn constant_image() {
    for bits in [8, 9] {
        let y = if bits == 9 { 200 } else { 100 };
        let m = open(&const_spec(y, bits));
        // Level-0 tiles away from the fill.
        for (x, y) in [(3, 1), (4, 2)] {
            let img = tile_rgba(&m, 0, x, y);
            for (i, px) in img.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(
                    *px,
                    [100, 100, 100, 255],
                    "bits {bits} tile ({x},{y}) pixel {i}"
                );
            }
        }
    }
}

/// Out-of-grid tiles are white; in-grid missing tiles decode virtually instead
/// (`missing_tile_virtual`).
#[test]
fn out_of_grid_tiles_white() {
    let m = open(&rand_spec(19));
    for (x, y) in [(-1, 0), (8, 0), (0, 4)] {
        assert!(is_white(&tile_rgba(&m, 0, x, y)), "tile ({x}, {y})");
    }
}

/// `decode_region_rgba`'s target: `decode_tile_rgba`'s output stitched.
fn region_via_tiles(m: &Image, level: usize, r: Rect) -> Vec<u8> {
    let (w, h) = (r.dx() as usize, r.dy() as usize);
    let mut out = vec![255u8; 4 * w * h];
    let l = lv(m, level);
    let off = (((3i64 << m.h.levels) - 2) >> level) - 1;
    let s = side(m, level) as i64;
    let vis = r.intersect(Rect::new(0, 0, l.width as i64, l.height as i64));
    let mut tiles: HashMap<(i64, i64), Vec<u8>> = HashMap::new();
    for y in vis.min_y..vis.max_y {
        for x in vis.min_x..vis.max_x {
            let (gx, gy) = (x + off, y + off);
            let t = tiles
                .entry((gx / s, gy / s))
                .or_insert_with(|| tile_rgba(m, level, gx / s, gy / s));
            let src = 4 * ((gy % s) * s + gx % s) as usize;
            let o = 4 * ((y - r.min_y) as usize * w + (x - r.min_x) as usize);
            out[o..o + 4].copy_from_slice(&t[src..src + 4]);
        }
    }
    out
}

fn region_rgba(m: &Image, level: usize, r: Rect) -> Vec<u8> {
    let mut got = vec![0u8; 4 * r.dx() as usize * r.dy() as usize];
    m.decode_region_rgba(level, r, &mut got, 4 * r.dx() as usize, &go())
        .unwrap();
    got
}

const RECTS: [(i64, i64, i64, i64); 4] = [
    (0, 0, 40, 30),
    (230, 250, 300, 280), // across tile edges
    (-20, -10, 10, 10),   // outside the image
    (1480, 880, 1530, 920),
];

#[test]
fn region_matches_tiles() {
    let m = open(&rand_spec(16));
    for level in 0..m.levels() {
        for (x0, y0, x1, y1) in RECTS {
            let d = 1i64 << level;
            let r = Rect::new(x0 / d, y0 / d, x1 / d, y1 / d);
            if r.is_empty() {
                continue;
            }
            assert!(
                region_rgba(&m, level, r) == region_via_tiles(&m, level, r),
                "level {level} rect {r:?}"
            );
        }
    }
}

#[test]
fn ycbcr() {
    let m = open(&rand_spec(19));
    let r = Rect::new(201, 197, 330, 300); // odd origin and size
    let w = r.dx() as usize;
    let rgba = region_rgba(&m, 0, r);
    let shifted = Rect::new(r.min_x - 250, r.min_y - 250, r.max_x - 250, r.max_y - 250);
    for (sub, bounds) in [
        (Subsample::S444, r),
        (Subsample::S420, r),
        (Subsample::S420, shifted),
        // Chroma rows and columns at negative and zero maxima (c_offset truncates toward zero).
        (Subsample::S420, Rect::new(-r.dx(), -r.dy(), 0, 0)),
        (Subsample::S420, Rect::new(1 - r.dx(), 1 - r.dy(), 1, 1)),
        (Subsample::S420, Rect::new(-2, -2, r.dx() - 2, r.dy() - 2)),
    ] {
        let (ox, oy) = (bounds.min_x - r.min_x, bounds.min_y - r.min_y);
        let mut got = YCbCr::new(bounds, sub, &Limits::default()).unwrap();
        m.decode_region_ycbcr(0, r, &mut got, &go()).unwrap();
        let mut sums: HashMap<usize, (i32, i32, i32)> = HashMap::new();
        for y in r.min_y..r.max_y {
            for x in r.min_x..r.max_x {
                let o = 4 * ((y - r.min_y) as usize * w + (x - r.min_x) as usize);
                let (yy, cb, cr) = rgb_to_ycbcr(rgba[o], rgba[o + 1], rgba[o + 2]);
                assert_eq!(
                    got.y[got.y_offset(x + ox, y + oy)],
                    yy,
                    "{sub:?} ({x},{y}): Y"
                );
                let e = sums.entry(got.c_offset(x + ox, y + oy)).or_default();
                *e = (e.0 + i32::from(cb), e.1 + i32::from(cr), e.2 + 1);
            }
        }
        for (o, (cb, cr, n)) in sums {
            let (wcb, wcr) = ((cb + n / 2) / n, (cr + n / 2) / n);
            assert_eq!(
                (i32::from(got.cb[o]), i32::from(got.cr[o])),
                (wcb, wcr),
                "{sub:?} chroma {o}"
            );
        }
    }
    // A tile too.
    let want = tile_rgba(&m, 1, 1, 1);
    let mut got = YCbCr::new(
        Rect::new(0, 0, 256, 256),
        Subsample::S444,
        &Limits::default(),
    )
    .unwrap();
    m.decode_tile_ycbcr(1, 1, 1, &mut got, &go()).unwrap();
    let n = 256 * 256;
    let (mut y, mut cb, mut cr) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    crate::dsp::rgba_to_ycbcr(&mut y, &mut cb, &mut cr, &want);
    assert!(
        got.y == y && got.cb == cb && got.cr == cr,
        "tile YCbCr mismatch"
    );
    // A white tile outside the grid, into 4:2:0 planes at a negative origin that ends at 0.
    let mut white = YCbCr::new(
        Rect::new(-256, -256, 0, 0),
        Subsample::S420,
        &Limits::default(),
    )
    .unwrap();
    m.decode_tile_ycbcr(0, -1, -1, &mut white, &go()).unwrap();
    assert!(white.y.iter().all(|&v| v == 255), "white luma");
    for (x, y) in [(-256, -256), (-1, -1), (-1, -256)] {
        assert_eq!(
            (
                white.cb[white.c_offset(x, y)],
                white.cr[white.c_offset(x, y)]
            ),
            (128, 128)
        );
    }
}

fn all_tiles(m: &Image) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for level in (0..m.levels()).rev() {
        let l = lv(m, level);
        for y in 0..l.tiles_y {
            for x in 0..l.tiles_x {
                out.push(tile_rgba(m, level, x as i64, y as i64));
            }
        }
    }
    out
}

#[test]
fn cache_size_invariant() {
    let want = all_tiles(&open(&rand_spec(19)));
    let mut m = open(&rand_spec(19));
    for c in &mut m.pools {
        c.max = 1; // every entry evicted on insert
    }
    let got = all_tiles(&m);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g == w, "tile {i} differs with no cache");
    }
    // Level 0 first, backwards; parents compute lazily.
    let m = open(&rand_spec(19));
    let l = lv(&m, 0);
    for y in (0..l.tiles_y).rev() {
        for x in (0..l.tiles_x).rev() {
            let got = tile_rgba(&m, 0, x as i64, y as i64);
            assert!(
                got == want[want.len() - l.tiles_x * l.tiles_y + y * l.tiles_x + x],
                "level 0 tile ({x},{y}) differs in reverse order"
            );
        }
    }
}

#[test]
fn concurrent() {
    let mut m = open(&rand_spec(16));
    let mut rnd = Rng::new(9);
    let mut jobs = Vec::new();
    for _ in 0..48 {
        let l = rnd.below(3) as usize;
        let (x, y) = (rnd.below(1500 >> l) as i64, rnd.below(900 >> l) as i64);
        let r = Rect::new(
            x,
            y,
            x + 1 + rnd.below(400) as i64,
            y + 1 + rnd.below(300) as i64,
        );
        jobs.push((l, r));
    }
    let serial = open(&rand_spec(16));
    let want: Vec<Vec<u8>> = jobs
        .iter()
        .map(|&(l, r)| region_rgba(&serial, l, r))
        .collect();
    for c in &mut m.pools {
        c.max = 4 << 20; // force eviction under contention
    }
    std::thread::scope(|s| {
        let hs: Vec<_> = jobs
            .iter()
            .zip(&want)
            .map(|(&(l, r), w)| {
                let m = &m;
                s.spawn(move || region_rgba(m, l, r) == *w)
            })
            .collect();
        for (i, h) in hs.into_iter().enumerate() {
            assert!(h.join().unwrap(), "concurrent region {i} differs");
        }
    });
}

#[test]
fn bad_arguments() {
    let m = open(&rand_spec(19));
    let c = go();
    let mut tile = vec![0u8; 4 * 256 * 256];
    assert!(
        matches!(
            m.decode_tile_rgba(4, 0, 0, &mut tile, 1024, &c),
            Err(Error::InvalidArgument(_))
        ),
        "level"
    );
    assert!(
        matches!(
            m.decode_tile_rgba(0, 1, 1, &mut tile[..1000], 1024, &c),
            Err(Error::InvalidArgument(_))
        ),
        "short dst"
    );
    assert!(
        matches!(
            m.decode_tile_rgba(0, 1, 1, &mut tile, 1000, &c),
            Err(Error::InvalidArgument(_))
        ),
        "short stride"
    );
    let mut small = YCbCr::new(
        Rect::new(0, 0, 255, 256),
        Subsample::S444,
        &Limits::default(),
    )
    .unwrap();
    assert!(
        m.decode_tile_ycbcr(0, 1, 1, &mut small, &c).is_err(),
        "tile size"
    );
    let mut bad = YCbCr::new(
        Rect::new(0, 0, 256, 256),
        Subsample::S444,
        &Limits::default(),
    )
    .unwrap();
    bad.cb.truncate(10);
    assert!(
        m.decode_tile_ycbcr(0, 1, 1, &mut bad, &c).is_err(),
        "short planes"
    );
    let mut px = vec![0u8; 400];
    assert!(
        m.decode_region_rgba(0, Rect::new(0, 0, 10, 11), &mut px, 40, &c)
            .is_err(),
        "region dst"
    );
    assert!(
        m.decode_region_rgba(0, Rect::new(0, 0, 1 << 41, 1), &mut px, 40, &c)
            .is_err(),
        "huge region"
    );
    assert!(
        matches!(
            m.decode_region_rgba(0, Rect::new(1 << 41, 0, (1 << 41) + 8, 8), &mut px, 40, &c),
            Err(Error::InvalidArgument(_))
        ),
        "far region (Go returns white)"
    );
    // TestRGBA64BadBounds: a region one column short, a 128 x 128 tile at level 0.
    let mut d64 = vec![0u16; 4 * 9 * 10];
    assert!(
        matches!(
            m.decode_region_rgba64(0, Rect::new(0, 0, 10, 10), &mut d64, 4 * 9, &c),
            Err(Error::InvalidArgument(_))
        ),
        "rgba64 region dst"
    );
    let mut t64 = vec![0u16; 4 * 128 * 128];
    assert!(
        matches!(
            m.decode_tile_rgba64(0, 0, 0, &mut t64, 4 * 128, &c),
            Err(Error::InvalidArgument(_))
        ),
        "rgba64 tile dst"
    );
    let mut yc = YCbCr::new(Rect::new(0, 0, 10, 11), Subsample::S420, &Limits::default()).unwrap();
    assert!(
        m.decode_region_ycbcr(0, Rect::new(0, 0, 10, 10), &mut yc, &c)
            .is_err(),
        "ycbcr size"
    );
}

#[test]
fn cancel() {
    let m = open(&rand_spec(19));
    let c = AtomicBool::new(true);
    let mut tile = vec![0u8; 4 * 256 * 256];
    assert_eq!(
        m.decode_tile_rgba(0, 2, 1, &mut tile, 1024, &c),
        Err(Error::Cancelled)
    );
    let mut px = vec![0u8; 4 * 900 * 800];
    assert_eq!(
        m.decode_region_rgba(0, Rect::new(0, 0, 900, 800), &mut px, 4 * 900, &c),
        Err(Error::Cancelled)
    );
    // Even where nothing is decoded.
    let far = Rect::new(1 << 20, 1 << 20, (1 << 20) + 8, (1 << 20) + 8);
    let mut s = vec![0u16; 3 * 64];
    assert_eq!(
        m.decode_region_samples(0, far, &mut s, 24, &c),
        Err(Error::Cancelled)
    );
    let mut s16 = vec![0i16; 3 * 64];
    assert_eq!(
        m.decode_region_signed_samples(0, far, &mut s16, 24, &c),
        Err(Error::Cancelled)
    );
    let mut p64 = vec![0u16; 4 * 64];
    assert_eq!(
        m.decode_region_rgba64(0, far, &mut p64, 32, &c),
        Err(Error::Cancelled)
    );
    assert_eq!(
        m.decode_tile_rgba64(
            0,
            1 << 10,
            1 << 10,
            &mut vec![0u16; 4 * 256 * 256],
            1024,
            &c
        ),
        Err(Error::Cancelled)
    );
}

/// Delays the first read.
struct Blocking {
    r: Arc<dyn ReadAt>,
    /// (started, released)
    gate: Mutex<(bool, bool)>,
    cv: Condvar,
}

impl ReadAt for Blocking {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut g = self.gate.lock().unwrap();
        if !g.0 {
            g.0 = true;
            self.cv.notify_all();
            while !g.1 {
                g = self.cv.wait(g).unwrap();
            }
        }
        drop(g);
        self.r.read_at(buf, offset)
    }
}

/// Cancels mid-decode, not before the start: the call returns promptly.
#[test]
fn cancel_during_decode() {
    for name in ["tile", "region"] {
        let mut m = open(&rand_spec(19));
        let b = Arc::new(Blocking {
            r: m.r.clone(),
            gate: Mutex::new((false, false)),
            cv: Condvar::new(),
        });
        m.r = b.clone(); // blocks codeblock reads, not header reads
        let c = go();
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                if name == "tile" {
                    m.decode_tile_rgba(0, 2, 1, &mut vec![0u8; 4 * 256 * 256], 1024, &c)
                } else {
                    m.decode_region_rgba(
                        0,
                        Rect::new(0, 0, 900, 800),
                        &mut vec![0u8; 4 * 900 * 800],
                        4 * 900,
                        &c,
                    )
                }
            });
            let mut g = b.gate.lock().unwrap();
            while !g.0 {
                g = b.cv.wait(g).unwrap(); // a codeblock read is blocked mid-decode
            }
            c.store(true, Ordering::Relaxed);
            g.1 = true;
            b.cv.notify_all();
            drop(g);
            assert_eq!(h.join().unwrap(), Err(Error::Cancelled), "{name}");
        });
    }
}

#[test]
fn memory_limit() {
    let m = open(&rand_spec(19));
    let mut tile = vec![0u8; 4 * 256 * 256];
    let prev = m.set_memory_limit(1 << 20); // under one tile's scratch
    let err = m
        .decode_tile_rgba(0, 2, 1, &mut tile, 1024, &go())
        .unwrap_err();
    assert_eq!(err, Error::AllocLimit, "never fits");
    // Held by another decode: transient.
    m.set_memory_limit(64 << 20);
    lock(&m.scratch).clear(); // idle scratch holds its charge
    let held = m.budget.free() - 1000;
    m.budget.reserve(held).unwrap();
    let full = Rect::new(0, 0, 1500, 900);
    let mut px = vec![0u8; 4 * 1500 * 900];
    let err = m.decode_region_rgba(0, full, &mut px, 4 * 1500, &go());
    m.budget.release(held);
    assert_eq!(err, Err(Error::MemoryLimit), "held");
    assert_eq!(m.set_memory_limit(prev), 64 << 20);
    m.decode_region_rgba(0, full, &mut px, 4 * 1500, &go())
        .unwrap();
}

/// A scratch must evict this image's cache, not blame others.
#[test]
fn scratch_evicts_own_cache() {
    let m = open(&rand_spec(19));
    let held = 8 << 20; // charged by the fake entry below
    let k = Key {
        v: 0,
        s: 99,
        x: 0,
        y: 0,
    };
    drop(
        m.pools[Kind::Coef as usize]
            .get(k, held / 2, &go(), |_| Ok(()))
            .unwrap(),
    );
    // One byte short without eviction.
    m.set_memory_limit(held as u64 + SCRATCH_BYTES - 1);
    drop(
        m.get_scratch()
            .expect("get_scratch did not evict this image's own cache"),
    );
    assert!(!m.pools[Kind::Coef as usize].contains(&k));
}

/// An idle pooled scratch stays charged, and a reserve that needs the room frees it.
#[test]
fn pooled_scratch_is_charged() {
    let m = open(&rand_spec(19));
    let before = m.budget.free();
    drop(m.get_scratch().unwrap());
    assert_eq!(m.budget.free(), before - SCRATCH_BYTES, "pooled, charged");
    let _all = m.reserve(before).unwrap();
    assert!(lock(&m.scratch).is_empty(), "evicted for the room");
}

/// The part of a codeblock past what the scratch is charged for is charged while it is read.
#[test]
fn big_codeblock_is_charged() {
    let m = open(&rand_spec(19));
    let mut sc = m.get_scratch().unwrap();
    m.set_memory_limit(m.budget.limit() - m.budget.free() + 1000);
    let mut dst = vec![0i16; 3 * AREA];
    let b = Block {
        offset: 0,
        size: 1 << 20,
    };
    assert_eq!(
        m.codeblock(&go(), &mut sc, b, &mut dst, 3),
        Err(Error::MemoryLimit)
    );
}

/// A codeblock whose read fails (the reader is shorter than the declared size) does not leave its
/// oversize buffer in the pooled scratch, and its charge is released.
#[test]
fn failed_big_read_releases_the_buffer() {
    let m = open(&rand_spec(19));
    let free = m.budget.free();
    let mut dst = vec![0i16; 3 * AREA];
    let b = Block {
        offset: 1 << 40,
        size: 1 << 24,
    };
    {
        let mut sc = m.get_scratch().unwrap();
        let free = m.budget.free();
        let r = m.codeblock(&go(), &mut sc, b, &mut dst, 3);
        assert!(matches!(r, Err(Error::Io(_))), "{r:?}");
        assert!(sc.read.capacity() <= READ_BYTES, "{}", sc.read.capacity());
        assert_eq!(m.budget.free(), free, "the extra charge is released");
    }
    assert!(m.budget.free() < free, "the pooled scratch stays charged");
}

/// The open charges the tile index and the header parse against one `max_alloc`: the smallest
/// limit that opens is an `AllocLimit` away from the one below it.
#[test]
fn open_total_is_charged_once() {
    let spec = rand_spec(19);
    let opens = |max_alloc: u64| {
        let l = Limits {
            max_alloc,
            ..Limits::default()
        };
        open_with(&spec, &l).map(|_| ())
    };
    let (mut lo, mut hi) = (0u64, 64 << 20);
    assert!(opens(hi).is_ok());
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        match opens(mid) {
            Ok(()) => hi = mid,
            Err(e) => {
                assert_eq!(e, Error::AllocLimit, "max_alloc {mid}");
                lo = mid;
            }
        }
    }
    assert_eq!(opens(hi - 1), Err(Error::AllocLimit));
    assert_eq!(opens(hi), Ok(()));
}

/// The header's label, macro, ICC and tile index count against the limit, and the wide view
/// shares the blobs.
#[test]
fn header_is_charged_and_shared() {
    let m = open(&api_spec(false));
    assert!(m.budget.free() < m.budget.limit(), "the header is charged");
    let w = m.wide_grid();
    let (a, b) = (m.h.label.as_ref().unwrap(), w.h.label.as_ref().unwrap());
    assert!(Arc::ptr_eq(a, b), "label shared");
    assert!(
        Arc::ptr_eq(&m.h.envelopes, &w.h.envelopes),
        "envelopes shared"
    );
}

/// A retired entry goes before older ones.
#[test]
fn demote_evicts_first() {
    let c = Cache::new(2 * 2 * 8, Arc::new(crate::budget::Budget::new(u64::MAX))); // room for two 8-sample entries
    let key = |x: i32| Key {
        v: 0,
        s: 0,
        x,
        y: 0,
    };
    for x in 0..3 {
        let _buf = c.get(key(x), 8, &go(), |_| Ok(())).unwrap();
        if x == 1 {
            c.demote(&key(x));
        }
    }
    for (x, want) in [true, false, true].into_iter().enumerate() {
        assert_eq!(c.contains(&key(x as i32)), want, "entry {x}");
    }
}

/// A fill that panics leaves no pending slot: the next get computes the entry.
#[test]
fn cache_fill_panic_frees_the_slot() {
    let c = Cache::new(1 << 20, Arc::new(crate::budget::Budget::new(u64::MAX)));
    let k = Key {
        v: 0,
        s: 0,
        x: 0,
        y: 0,
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = c.get(k, 8, &go(), |_| panic!("fill"));
    }));
    assert!(r.is_err());
    let b = c.get(k, 8, &go(), |d| {
        d.fill(3);
        Ok(())
    });
    assert!(b.unwrap().data.iter().all(|&v| v == 3));
}

#[test]
fn cache_computes_once_and_reports_failures() {
    let c = Cache::new(1 << 20, Arc::new(crate::budget::Budget::new(u64::MAX)));
    let k = Key {
        v: 0,
        s: 1,
        x: 2,
        y: 3,
    };
    let calls = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                let b = c
                    .get(k, 16, &go(), |d| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_millis(30));
                        d.fill(7);
                        Ok(())
                    })
                    .unwrap();
                assert!(b.data.iter().all(|&v| v == 7));
            });
        }
    });
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    // A failed fill leaves nothing behind, and the next caller computes it again.
    let k2 = Key {
        v: 0,
        s: 1,
        x: 2,
        y: 4,
    };
    assert_eq!(
        c.get(k2, 16, &go(), |_| Err(Error::Cancelled)).err(),
        Some(Error::Cancelled)
    );
    assert!(!c.contains(&k2));
    c.get(k2, 16, &go(), |_| Ok(())).unwrap();
    assert!(c.contains(&k2));
}

// ----- samples and RGBA64 -----

fn planes(mut m: Image) -> Image {
    m.planes = true;
    m
}

fn sdk(m: &Image) -> Image {
    m.sdk_truncated().unwrap()
}

fn rects_for(l: Level) -> Vec<Rect> {
    let (w, h) = (l.width as i64, l.height as i64);
    vec![
        Rect::new(0, 0, 40, 30),
        Rect::new(3, 1, 4, 2),
        Rect::new(w / 2 - 100, h / 2 - 90, w / 2 + 120, h / 2 + 70),
        Rect::new(250, 250, 263, 520), // crosses tile edges
        Rect::new(-20, -10, 10, 10),
        Rect::new(w - 30, h - 25, w + 40, h + 30),
        Rect::new(0, 0, w, h),
        Rect::new(w + 5, 0, w + 20, 10), // fully outside
    ]
}

/// The RGBA64 channel is the sample replicated.
#[test]
fn samples_match_rgba64() {
    let cases: Vec<(&str, Image)> = vec![
        ("8bit", open(&wide_spec(16, 8))),
        ("9bit", open(&wide_spec(19, 9))),
        ("9bit-rgb", planes(open(&wide_spec(19, 9)))),
        ("8bit-rgb", planes(open(&wide_spec(16, 8)))),
        ("v1-16bit", open(&v1_spec())),
        ("8bit-sdk", sdk(&open(&rand_spec(19)))),
    ];
    for (name, m) in &cases {
        for level in 0..m.levels() {
            for r in rects_for(lv(m, level)) {
                let (w, h) = (r.dx() as usize, r.dy() as usize);
                let mut b = vec![0u16; 4 * w * h];
                m.decode_region_rgba64(level, r, &mut b, 4 * w, &go())
                    .unwrap();
                let stride = 3 * w + 5; // padding left untouched
                let mut dst = vec![0xBEEFu16; (h - 1) * stride + 3 * w];
                m.decode_region_samples(level, r, &mut dst, stride, &go())
                    .unwrap();
                let hi = (1u16 << (8 + m.shift)) - 1;
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..3 {
                            let s = dst[y * stride + 3 * x + c];
                            let want = if m.shift == 1 {
                                (s << 7) | (s >> 2)
                            } else {
                                (s << 8) | s
                            };
                            assert!(
                                s <= hi && b[4 * (y * w + x) + c] == want,
                                "{name} level {level} {r:?} ({x},{y}) ch {c}: sample {s}, rgba64 {:04x}",
                                b[4 * (y * w + x) + c]
                            );
                        }
                    }
                    if y < h - 1 {
                        assert!(
                            dst[y * stride + 3 * w..(y + 1) * stride]
                                .iter()
                                .all(|&v| v == 0xBEEF),
                            "{name} level {level} {r:?} row {y}: padding written"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn samples_errors() {
    let m = open(&rand_spec(19));
    let r = Rect::new(0, 0, 10, 10);
    let c = go();
    let mut s = vec![0u16; 300];
    let bad = |e: Result<(), Error>| matches!(e, Err(Error::InvalidArgument(_)));
    assert!(
        bad(m.decode_region_samples(m.levels(), r, &mut s, 30, &c)),
        "level Levels() accepted"
    );
    assert!(
        bad(m.decode_region_samples(0, r, &mut s, 29, &c)),
        "short stride accepted"
    );
    assert!(
        bad(m.decode_region_samples(0, r, &mut s[..299], 30, &c)),
        "short dst accepted"
    );
    assert_eq!(
        m.decode_region_samples(0, Rect::default(), &mut [], 0, &c),
        Ok(())
    );
    let mut sg = vec![0i16; 300];
    assert!(
        bad(m.decode_region_signed_samples(m.levels(), r, &mut sg, 30, &c)),
        "level Levels() accepted"
    );
    assert!(
        bad(m.decode_region_signed_samples(0, r, &mut sg, 29, &c)),
        "short stride accepted"
    );
    assert!(
        bad(m.decode_region_signed_samples(0, r, &mut sg[..299], 30, &c)),
        "short dst accepted"
    );
    assert_eq!(
        m.decode_region_signed_samples(0, Rect::default(), &mut [], 0, &c),
        Ok(())
    );
    // Memory held by others.
    m.set_memory_limit(64 << 20);
    lock(&m.scratch).clear(); // idle scratch holds its charge
    let held = m.budget.free() - 1000;
    m.budget.reserve(held).unwrap();
    let big = Rect::new(0, 0, 300, 300);
    let mut s = vec![0u16; 3 * 300 * 300];
    let e = m.decode_region_samples(0, big, &mut s, 3 * 300, &c);
    m.budget.release(held);
    assert_eq!(e, Err(Error::MemoryLimit), "held");
}

/// Signed vs clamped on every transform and depth: the clamped is the signed one clamped, white
/// is `2^b-1` in both, and the padding stays untouched.
#[test]
fn signed_samples_layouts() {
    let mut v1 = wide_spec(16, 16);
    v1.version = "5.0".into();
    let cases: Vec<(&str, Image)> = vec![
        ("9bit-rgb", planes(open(&wide_spec(19, 9)))),
        ("9bit-ycocg", open(&wide_spec(19, 9))),
        ("8bit-ycocg", open(&wide_spec(19, 8))),
        ("8bit-rgb", planes(open(&wide_spec(19, 8)))),
        ("v1-16bit", open(&v1)),
    ];
    const PAD: usize = 5;
    const SENTINEL: i16 = 0x7EEF;
    for (name, m) in &cases {
        let hi = (1i16 << (8 + m.shift)) - 1;
        let mut out = 0;
        for level in 0..m.levels() {
            let l = lv(m, level);
            let mut rects = rects_for(l);
            rects.push(Rect::new(
                -20,
                -10,
                l.width as i64 + 30,
                l.height as i64 + 25,
            ));
            for r in rects {
                let (w, h) = (r.dx() as usize, r.dy() as usize);
                let stride = 3 * w + PAD;
                let n = (h - 1) * stride + 3 * w;
                let mut got = vec![SENTINEL; n];
                let mut clamped = vec![SENTINEL as u16; n];
                m.decode_region_signed_samples(level, r, &mut got, stride, &go())
                    .unwrap();
                m.decode_region_samples(level, r, &mut clamped, stride, &go())
                    .unwrap();
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..3 {
                            let i = y * stride + 3 * x + c;
                            let v = got[i];
                            assert_eq!(
                                clamped[i],
                                v.clamp(0, hi) as u16,
                                "{name} level {level} {r:?} ({x},{y}) ch {c}"
                            );
                            if v < 0 || v > hi {
                                out += 1;
                            }
                            let p = (r.min_x + x as i64, r.min_y + y as i64);
                            let inside = p.0 >= 0
                                && p.1 >= 0
                                && p.0 < l.width as i64
                                && p.1 < l.height as i64;
                            assert!(
                                inside || (v == hi && clamped[i] == hi as u16),
                                "{name} level {level} {r:?} ({x},{y}): outside the level, want {hi}"
                            );
                        }
                    }
                    if y < h - 1 {
                        assert!(
                            got[y * stride + 3 * w..(y + 1) * stride]
                                .iter()
                                .all(|&v| v == SENTINEL),
                            "{name}: padding written"
                        );
                    }
                }
            }
        }
        assert!(
            out > 0,
            "{name}: no sample outside [0, {hi}]: the test does not bite"
        );
    }
}

fn rgba64_region(m: &Image, level: usize, r: Rect) -> Vec<u16> {
    let mut b = vec![0u16; 4 * r.dx() as usize * r.dy() as usize];
    m.decode_region_rgba64(level, r, &mut b, 4 * r.dx() as usize, &go())
        .unwrap();
    b
}

/// RGBA64 >> 8 equals RGBA, the colour channels are replicated, alpha is 0xFFFF.
fn check_shift(m: &Image, what: &str, a: &[u8], b: &[u16]) {
    assert_eq!(a.len(), b.len(), "{what}: sizes");
    for (i, (p, q)) in a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .enumerate()
    {
        for c in 0..4 {
            assert_eq!(
                u16::from(p[c]),
                q[c] >> 8,
                "{what} pixel {i} ch {c}: rgba {}, rgba64 {:04x}",
                p[c],
                q[c]
            );
        }
        for &u in &q[..3] {
            let want = if m.shift == 1 {
                ((u >> 7) << 7) | (u >> 7 >> 2)
            } else {
                ((u >> 8) << 8) | (u >> 8)
            };
            assert_eq!(u, want, "{what} pixel {i}: {u:04x} not replicated");
        }
        assert_eq!(q[3], 0xFFFF, "{what} pixel {i}: alpha");
    }
}

#[test]
fn rgba64_matches_rgba() {
    let cases: Vec<(&str, Image)> = vec![
        ("8bit", open(&wide_spec(16, 8))),
        ("9bit", open(&wide_spec(19, 9))),
        ("9bit-rgb", planes(open(&wide_spec(19, 9)))),
        ("8bit-rgb", planes(open(&wide_spec(16, 8)))),
        ("v1-16bit", open(&v1_spec())),
    ];
    for (name, m) in &cases {
        for level in 0..m.levels() {
            let l = lv(m, level);
            for r in rects_for(l) {
                if r.is_empty() {
                    continue;
                }
                check_shift(
                    m,
                    &format!("{name} level {level} region {r:?}"),
                    &region_rgba(m, level, r),
                    &rgba64_region(m, level, r),
                );
            }
            let s = side(m, level);
            for y in -1..=l.tiles_y as i64 {
                for x in -1..=l.tiles_x as i64 {
                    let mut b = vec![0u16; 4 * s * s];
                    m.decode_tile_rgba64(level, x, y, &mut b, 4 * s, &go())
                        .unwrap();
                    check_shift(
                        m,
                        &format!("{name} level {level} tile ({x},{y})"),
                        &tile_rgba(m, level, x, y),
                        &b,
                    );
                }
            }
        }
    }
}

/// Exact bits from the signed sample.
#[test]
fn rgba64_replication() {
    for bits in [8u32, 9] {
        let m = open(&wide_spec(19, bits));
        let r = Rect::new(100, 100, 400, 300);
        let n = 3 * r.dx() as usize * r.dy() as usize;
        let mut s = vec![0i16; n];
        m.decode_region_signed_samples(0, r, &mut s, 3 * r.dx() as usize, &go())
            .unwrap();
        let b = rgba64_region(&m, 0, r);
        let hi = (1i32 << (8 + m.shift)) - 1;
        let mut clamped = 0;
        for i in 0..n / 3 {
            for c in 0..3 {
                let raw = i32::from(s[3 * i + c]);
                let v = raw.clamp(0, hi);
                clamped += usize::from(v != raw);
                let want = if bits == 9 {
                    (v << 7) | (v >> 2)
                } else {
                    (v << 8) | v
                };
                assert_eq!(i32::from(b[4 * i + c]), want, "bits {bits} sample {raw}");
            }
        }
        assert!(clamped > 0, "bits {bits}: the fixture never clamps");
    }
}

#[test]
fn rgba64_budget_and_cancel() {
    let m = open(&rand_spec(19));
    m.set_memory_limit(64 << 20);
    lock(&m.scratch).clear(); // idle scratch holds its charge
    let held = m.budget.free() - 1000;
    m.budget.reserve(held).unwrap();
    let r = Rect::new(0, 0, 300, 300);
    let mut reg = vec![0u16; 4 * 300 * 300];
    let e1 = m.decode_region_rgba64(0, r, &mut reg, 4 * 300, &go());
    let mut tile = vec![0u16; 4 * 256 * 256];
    let e2 = m.decode_tile_rgba64(0, 0, 0, &mut tile, 4 * 256, &go());
    m.budget.release(held);
    assert_eq!(
        (e1, e2),
        (Err(Error::MemoryLimit), Err(Error::MemoryLimit)),
        "held"
    );
}

// ----- truncation -----

/// One coefficient (isyntax-go docs/format.md).
fn trunc_ref(k: u8, round: bool, v: i16) -> i16 {
    let mut a = i32::from(v).abs();
    a &= !((1 << k) - 1);
    if a != 0 && round && k > 0 {
        a += (1 << (k - 1)) - 1;
    }
    (if v < 0 { -a } else { a }) as i16
}

#[test]
fn trunc_apply() {
    use crate::decoder::{Q, W};
    for (k, round, input, want) in [
        (
            4u8,
            true,
            vec![0i16, 15, 16, -16, 31, 32, -47],
            vec![0i16, 0, 23, -23, 23, 39, -39],
        ),
        (1, true, vec![1, -1, 2, 3, -3], vec![0, 0, 2, 2, -2]),
        (3, true, vec![7, 8, -8, 15, 16], vec![0, 11, -11, 11, 19]),
        (4, false, vec![15, 16, -31], vec![0, 16, -16]),
    ] {
        for (v, w) in input.iter().zip(&want) {
            assert_eq!(trunc_ref(k, round, *v), *w, "ref k={k} round={round} {v}");
        }
    }
    let orig: Vec<i16> = (0..W * W).map(|i| (i % 97) as i16 - 48).collect();
    for round in [true, false] {
        let tt = TruncTable {
            k: [[1, 4, 0], [0, 2, 2]],
            round,
        };
        for (j, c) in [(0, 0), (0, 1), (0, 2), (1, 1)] {
            let mut buf = orig.clone();
            tt.apply(&mut buf, j, c);
            let k = tt.k[j][c];
            for (i, &v) in orig.iter().enumerate() {
                let want = if i / W < Q && i % W < Q {
                    v
                } else {
                    trunc_ref(k, round, v)
                }; // LL stays
                assert_eq!(
                    buf[i],
                    want,
                    "round={round} j={j} c={c} at ({},{})",
                    i % W,
                    i / W
                );
            }
        }
    }
}

#[test]
fn sdk_presets() {
    let h = |major, bits_stored, quality, rgb, levels| Header {
        major,
        bits_stored,
        quality,
        rgb,
        levels,
        ..Header::default()
    };
    type Want = Option<&'static TruncTable>;
    let cases: Vec<(&str, Header, Want, bool)> = vec![
        ("Q2 8-bit", h(100, 8, 2, false, 3), Some(&TRUNC_Q2B8), false),
        (
            "no token 8-bit",
            h(100, 8, -1, false, 3),
            Some(&TRUNC_Q2B8),
            false,
        ),
        ("Q2 9-bit", h(100, 9, 2, false, 3), Some(&TRUNC_Q2B9), false),
        ("no token 9-bit", h(100, 9, -1, false, 3), None, true),
        ("Q1 RGB v1", h(5, 8, 1, true, 1), Some(&TRUNC_Q1_RGB), false),
        ("Q1 RGB v1, 2 scales", h(5, 8, 1, true, 2), None, true),
        ("Q1 YCoCg", h(100, 8, 1, false, 3), None, true),
        ("Q2 v1 16-bit", h(5, 16, 2, false, 3), None, true),
        ("no token v1 16-bit", h(5, 16, -1, false, 3), None, true),
        ("Q1 v1 16-bit", h(5, 16, 1, false, 3), None, true),
        ("Q2 v1 RGB 16-bit", h(5, 16, 2, true, 3), None, true),
        ("Q2 v1 8-bit", h(5, 8, 2, false, 3), None, true),
        ("Q2 v1 9-bit", h(5, 9, 2, false, 3), None, true),
        ("Q0", h(100, 9, 0, false, 3), None, false),
    ];
    for (name, hdr, want, is_err) in cases {
        match sdk_preset(&hdr) {
            Ok(got) => {
                assert!(!is_err, "{name}: no error");
                assert_eq!(
                    got.map(std::ptr::from_ref),
                    want.map(std::ptr::from_ref),
                    "{name}"
                );
            }
            Err(e) => assert!(
                is_err && matches!(e, Error::Unsupported(_)),
                "{name}: {e:?}"
            ),
        }
    }
}

/// The view at level `l` equals a plain decode of the file with its coefficients truncated.
#[test]
fn sdk_truncated_matches_spec() {
    let spec = rand_spec(19);
    let m = open(&spec);
    let v = m.sdk_truncated().unwrap();
    let tt = v.tt.unwrap();
    for level in 0..m.levels() {
        let mut re = spec.clone();
        let base = spec.coef.clone();
        re.set_coef(move |s, x, y, c, ll| {
            let mut cf = base(s, x, y, c, ll);
            if !ll && s >= level && s - level < 2 {
                for e in &mut cf {
                    *e = trunc_ref(tt.k[s - level][c], tt.round, *e);
                }
            }
            cf
        });
        let want = open(&re);
        let l = lv(&m, level);
        let r = Rect::new(0, 0, l.width as i64, l.height as i64);
        assert!(
            region_rgba(&v, level, r) == region_rgba(&want, level, r),
            "level {level}: RGBA view differs from the truncated file"
        );
        assert!(
            rgba64_region(&v, level, r) == rgba64_region(&want, level, r),
            "level {level}: RGBA64 view differs from the truncated file"
        );
    }
    assert!(
        open(&v1_spec()).sdk_truncated().is_err(),
        "v1 16-bit has no SDK table"
    );
    assert!(
        open(&const_spec(100, 9)).sdk_truncated().is_err(),
        "9-bit without a Quality token"
    );
}

// ----- rules -----

#[test]
fn origin_formula() {
    // The SDK offset (rule 1) differs from libisyntax's above level 2.
    let levels = 3;
    let sdk = |l: usize| (((3i64 << levels) - 2) >> l) - 1;
    let lib = |l: usize| ((3i64 << levels) - 3) >> l;
    for (l, w) in [21, 10, 4].into_iter().enumerate() {
        assert_eq!(sdk(l), w, "sdk offset level {l}");
    }
    assert!(
        sdk(0) == lib(0) && sdk(1) == lib(1),
        "the formulas agree below level 2"
    );
    assert_ne!(
        sdk(2),
        lib(2),
        "rule 1 regressed: sdk and libisyntax offsets must differ at level 2"
    );
}

/// `const_spec` with one tile absent.
fn missing_at(y: i16, bits: u32, s: usize, tx: usize, ty: usize) -> FileSpec {
    let mut spec = const_spec(y, bits);
    spec.set_exists(move |sc, x, yy| sc != s || x != tx || yy != ty);
    spec
}

/// A missing tile is zero detail plus the parent's LL (rule 2), not white: here it equals its
/// existing neighbours.
#[test]
fn missing_tile_virtual() {
    for bits in [8, 9] {
        let y = if bits == 9 { 200 } else { 100 };
        let m = open(&missing_at(y, bits, 0, 3, 1)); // far from the grid edge
        for (i, px) in tile_rgba(&m, 0, 3, 1).as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                *px,
                [100, 100, 100, 255],
                "bits {bits} pixel {i}: want grey 100 like a real neighbour"
            );
        }
    }
}

/// A missing top-scale LL is zero, not white (rule 3).
#[test]
fn top_scale_missing_is_zero() {
    let top = 2; // const_spec has 3 levels
    let m = open(&missing_at(100, 8, top, 1, 0));
    let l = lv(&m, top);
    assert!(
        l.tiles_x >= 2 && l.tiles_y >= 1,
        "grid too small, got {l:?}"
    );
    let img = tile_rgba(&m, top, 1, 0);
    // Core pixels come from its own LL.
    for y in 8..BS - 8 {
        for x in 8..BS - 8 {
            let i = 4 * (y * BS + x);
            assert_eq!(img[i..i + 3], [0, 0, 0], "pixel ({x},{y}): want black");
        }
    }
}

/// Y is signed (rule 4): a negative Y clamps to black.
#[test]
fn signed_y() {
    let m = open(&const_spec(-50, 8));
    assert_eq!(tile_rgba(&m, 0, 3, 1)[..3], [0, 0, 0]);
}

/// The view decodes as a file one top-scale tile wider and taller whose added tiles are missing;
/// in-grid tiles as `m`'s.
#[test]
fn wide_grid() {
    let a = rand_spec(19);
    let m = open(&a);
    let top = m.h.levels - 1;
    let (gw, gh) = (m.h.grid_w, m.h.grid_h);
    let mut b = a.clone();
    b.width = 2500;
    b.height = 1500; // one top-scale tile more
    let a_exists = a.exists.clone().unwrap();
    b.set_exists(move |s, x, y| (x as i64) < gw >> s && (y as i64) < gh >> s && a_exists(s, x, y));
    let reference = open(&b);
    assert_eq!(
        (reference.h.grid_w, reference.h.grid_h),
        (gw + (1 << top), gh + (1 << top)),
        "reference grid"
    );
    let (ms, refs) = (sdk(&m), sdk(&reference));
    for (m, reference) in [(&m, &reference), (&ms, &refs)] {
        let v = m.wide_grid();
        for k in 0..4 {
            assert!(v.pools[k].max <= m.pools[k].max / 8, "pool {k}");
        }
        for level in 0..=v.h.levels {
            let (vl, inl, want) = (lv(&v, level), lv(m, level), lv(reference, level));
            assert_eq!(
                (vl.tiles_x, vl.tiles_y),
                (want.tiles_x, want.tiles_y),
                "level {level}: grid"
            );
            assert_eq!(
                (vl.width, vl.height),
                (inl.width, inl.height),
                "level {level}: size"
            );
            for ty in 0..vl.tiles_y {
                for tx in 0..vl.tiles_x {
                    let got = tile_rgba(&v, level, tx as i64, ty as i64);
                    assert!(
                        got == tile_rgba(reference, level, tx as i64, ty as i64),
                        "level {level} tile ({tx},{ty}): differs from the wider file's"
                    );
                    if tx < inl.tiles_x && ty < inl.tiles_y {
                        assert!(
                            got == tile_rgba(m, level, tx as i64, ty as i64),
                            "level {level} tile ({tx},{ty}): in-grid tile differs from m's"
                        );
                    }
                }
            }
            // Past the grid: m white, the view not.
            assert!(
                tile_rgba(&v, level, inl.tiles_x as i64, inl.tiles_y as i64)
                    != tile_rgba(m, level, inl.tiles_x as i64, inl.tiles_y as i64),
                "level {level}: corner tile past the grid decodes as m's"
            );
        }
    }
}

// ----- occupancy -----

/// A v1 header with 4x4 scale-0 tiles under 2x2 top-scale tiles.
fn occ_image(major: i64, missing: &[TileKey]) -> (Header, HashMap<TileKey, Tile>) {
    let mut h = Header {
        major,
        levels: 2,
        grid_w: 4,
        grid_h: 4,
        width: 1000,
        height: 1000,
        envelope_default: true,
        ..Header::default()
    };
    let p = |x, y| Point { x, y };
    h.envelopes = Arc::new(vec![vec![p(0, 0), p(1000, 0), p(1000, 1000), p(0, 1000)]]);
    let mut tiles = HashMap::new();
    for s in 0..2 {
        for y in 0..4 >> s {
            for x in 0..4 >> s {
                tiles.insert(
                    TileKey { scale: s, x, y },
                    Tile {
                        exists: true,
                        ..Tile::default()
                    },
                );
            }
        }
    }
    for k in missing {
        tiles.remove(k);
    }
    (h, tiles)
}

fn env_rects(env: &[Vec<Point>]) -> Vec<Rect> {
    env.iter()
        .map(|p| {
            let r = Rect {
                min_x: p[0].x,
                min_y: p[0].y,
                max_x: p[2].x,
                max_y: p[2].y,
            };
            assert!(
                p.len() == 4
                    && p[1]
                        == Point {
                            x: r.max_x,
                            y: r.min_y
                        }
                    && p[3]
                        == Point {
                            x: r.min_x,
                            y: r.max_y
                        },
                "polygon {p:?} is not a rectangle"
            );
            r
        })
        .collect()
}

#[test]
fn occupancy_envelopes_rules() {
    let key = |scale, x, y| TileKey { scale, x, y };
    let (mut full, tiles) = occ_image(5, &[]);
    let want = full.envelopes.clone();
    occupancy_envelopes(&mut full, &tiles, &mut Acct::new(u64::MAX)).unwrap();
    assert_eq!(full.envelopes, want, "all tiles present: the default");

    let (mut v2, tiles) = occ_image(100, &[key(0, 1, 1)]);
    occupancy_envelopes(&mut v2, &tiles, &mut Acct::new(u64::MAX)).unwrap();
    assert_eq!(v2.envelopes, want, "v2: the default");

    // Tile edges at 256x-9 for offset (3<<2)-2-1 = 9: 247, 503, 759.
    let (mut m, tiles) = occ_image(5, &[key(0, 1, 1)]);
    occupancy_envelopes(&mut m, &tiles, &mut Acct::new(u64::MAX)).unwrap();
    let want_r = vec![
        Rect::new(0, 0, 1000, 247),
        Rect::new(0, 247, 247, 503),
        Rect::new(503, 247, 1000, 503),
        Rect::new(0, 503, 1000, 759),
        Rect::new(0, 759, 1000, 1000),
    ];
    assert_eq!(env_rects(&m.envelopes), want_r, "scale-0 tile missing");

    // Scale-0 tiles under the missing top tile may still be listed.
    let want_top = vec![
        Rect::new(0, 0, 1000, 247),
        Rect::new(0, 247, 1000, 503),
        Rect::new(0, 503, 503, 759),
        Rect::new(0, 759, 503, 1000),
    ];
    for miss in [
        vec![key(1, 1, 1)],
        vec![
            key(1, 1, 1),
            key(0, 2, 2),
            key(0, 3, 2),
            key(0, 2, 3),
            key(0, 3, 3),
        ],
    ] {
        let (mut top, tiles) = occ_image(5, &miss);
        occupancy_envelopes(&mut top, &tiles, &mut Acct::new(u64::MAX)).unwrap();
        assert_eq!(
            env_rects(&top.envelopes),
            want_top,
            "top tile missing ({} keys removed)",
            miss.len()
        );
    }
}

// ----- the public API -----

fn api_spec(v1: bool) -> FileSpec {
    let mut s = FileSpec::new(|_, _, _, c, ll| {
        if ll {
            let mut v = vec![0i16; AREA];
            if c == 0 {
                v.fill(300);
            }
            v
        } else {
            vec![0; 3 * AREA]
        }
    });
    s.levels = 2;
    s.width = 600;
    s.height = 500;
    s.mpp = 0.5;
    s.label = vec![0xff, 0xd8, 1];
    s.macro_image = vec![0xff, 0xd8, 2];
    s.icc = b"icc".to_vec();
    if v1 {
        s.version = "5.0".into();
        s.compressor = 16;
        s.bits_stored = 16;
    } else {
        s.compressor = 19;
        s.bits_stored = 9;
    }
    s
}

#[test]
fn api() {
    let spec = api_spec(false);
    let (file, _) = fixture::file(&spec);
    let n = file.len() as u64;
    let im = Image::open(file.clone(), n, &Limits::default(), &go()).unwrap();
    assert_eq!(
        (
            im.levels(),
            im.tile_size(),
            im.bits_stored(),
            im.level(1).unwrap().width
        ),
        (3, 256, 9, 300)
    );
    assert_eq!(im.mpp(), (0.5, 0.5));
    assert_eq!(
        (im.label(), im.macro_image(), im.icc_profile()),
        (
            Some(&spec.label[..]),
            Some(&spec.macro_image[..]),
            Some(&b"icc"[..])
        )
    );
    let mut dst = vec![0u8; 4 * 20 * 20];
    im.decode_region_rgba(0, Rect::new(290, 240, 310, 260), &mut dst, 80, &go())
        .unwrap();
    let c = &dst[4 * (10 * 20 + 10)..][..3];
    assert_eq!(c, [150, 150, 150], "centre pixel: grey 150 (300 >> 1)");
    im.decode_tile_rgba(0, 1, 1, &mut vec![0u8; 4 * 256 * 256], 1024, &go())
        .unwrap();
    let (g, l) = (im.wide_grid().level(0).unwrap(), im.level(0).unwrap());
    assert!(
        g.tiles_x == l.tiles_x + 2 && g.tiles_y == l.tiles_y + 2 && g.width == l.width,
        "wide grid level 0 {g:?}, want {l:?} with 2 more tiles each way"
    );
    assert_eq!(im.rack_slot(), (None, None));

    let mut unsupported = fixture::file(&spec).0;
    let at = unsupported
        .windows(19)
        .position(|w| w == b"PMSVR=\"IString\">19<")
        .unwrap();
    unsupported[at + 16..at + 18].copy_from_slice(b"17");
    let n = unsupported.len() as u64;
    let e = Image::open(unsupported, n, &Limits::default(), &go()).err();
    assert!(
        matches!(e, Some(Error::Unsupported(_))),
        "bad compressor: {e:?}"
    );
}

#[test]
fn api_v1() {
    let (file, _) = fixture::file(&api_spec(true));
    let n = file.len() as u64;
    let im = Image::open(file, n, &Limits::default(), &go()).unwrap();
    assert_eq!((im.levels(), im.bits_stored()), (3, 16));
    let mut dst = vec![0u8; 4 * 20 * 20];
    im.decode_region_rgba(0, Rect::new(290, 240, 310, 260), &mut dst, 80, &go())
        .unwrap();
    let c = &dst[4 * (10 * 20 + 10)..][..3];
    assert_eq!(
        c,
        [255, 255, 255],
        "centre pixel: white (Y 300, no shift, clamped)"
    );
}

#[test]
fn open_limits() {
    let (file, _) = fixture::file(&rand_spec(19));
    let n = file.len() as u64;
    let few = Limits {
        max_pixels: 1000,
        ..Limits::default()
    };
    assert_eq!(
        Image::open(file.clone(), n, &few, &go()).err(),
        Some(Error::PixelLimit)
    );
    assert_eq!(
        Image::open(file.clone(), n, &Limits::default(), &AtomicBool::new(true)).err(),
        Some(Error::Cancelled)
    );
    // A reader that fails.
    struct Fail;
    impl ReadAt for Fail {
        fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<usize> {
            Err(io::Error::other("read failed"))
        }
    }
    assert_eq!(
        Image::open(Fail, 100, &Limits::default(), &go()).err(),
        Some(Error::Io(io::ErrorKind::Other))
    );
    // A file read from a path.
    let dir = std::env::temp_dir().join(format!("isyntax-rs-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("slide.isyntax");
    std::fs::write(&path, &file).unwrap();
    let f = std::fs::File::open(&path).unwrap();
    let im = Image::open(f, n, &Limits::default(), &go()).unwrap();
    assert_eq!(im.levels(), 4);
    assert_eq!(std::fs::read(&path).unwrap(), file);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Hostile files error, not panic: byte edits of valid files, opened and decoded.
#[test]
fn mutated_files_do_not_panic() {
    let mut small = rand_spec(19);
    small.width = 300;
    small.height = 200;
    small.levels = 2;
    let mut small_v1 = small.clone();
    small_v1.version = "5.0".into();
    small_v1.compressor = 16;
    small_v1.bits_stored = 16;
    for spec in [small, small_v1] {
        let (file, _) = fixture::file(&spec);
        let mut rnd = Rng::new(11);
        for i in 0..120 {
            let mut f = file.clone();
            for _ in 0..1 + rnd.below(4) {
                let at = rnd.below(f.len() as u64) as usize;
                f[at] = if i % 2 == 0 {
                    rnd.next_u64() as u8
                } else {
                    f[at] ^ (1 << rnd.below(8))
                };
            }
            let n = f.len() as u64;
            let limits = Limits {
                max_alloc: 1 << 30,
                ..Limits::default()
            };
            let Ok(im) = Image::open(f, n, &limits, &go()) else {
                continue;
            };
            let mut px = vec![0u8; 4 * 256 * 256];
            for level in 0..im.levels() {
                let side = if level == im.levels() - 1 { 128 } else { 256 };
                let _ = im.decode_tile_rgba(level, 0, 0, &mut px, 4 * side, &go());
                let _ =
                    im.decode_region_rgba(level, Rect::new(0, 0, 64, 64), &mut px, 4 * 64, &go());
            }
        }
    }
}

/// Rectangles with corners far outside any image are refused, not a panic, by every region call.
#[test]
fn extreme_rects() {
    let m = open(&rand_spec(19));
    let big = 1i64 << 40;
    let mut yc = YCbCr::new(Rect::new(0, 0, 4, 4), Subsample::S420, &Limits::default()).unwrap();
    for (lo, hi) in [
        (i64::MIN, i64::MAX),
        (i64::MIN, 0),
        (0, i64::MAX),
        (-big, big),
        (-big - 1, big + 1),
        (0, big),
        (0, big + 1),
        (-big, 0),
        (-big - 1, 0),
    ] {
        let r = Rect {
            min_x: lo,
            min_y: lo,
            max_x: hi,
            max_y: hi,
        };
        let (mut a, mut b, mut c) = ([0u8; 16], [0u16; 16], [0i16; 16]);
        let got = [
            m.decode_region_rgba(0, r, &mut a, 4, &go()).err(),
            m.decode_region_rgba64(0, r, &mut b, 4, &go()).err(),
            m.decode_region_samples(0, r, &mut b, 4, &go()).err(),
            m.decode_region_signed_samples(0, r, &mut c, 4, &go()).err(),
            m.decode_region_ycbcr(0, r, &mut yc, &go()).err(),
        ];
        for e in got {
            assert!(matches!(e, Some(Error::InvalidArgument(_))), "{r:?}: {e:?}");
        }
    }
}

/// The header strings an image keeps are charged to its budget, and `wide_grid` shares them.
#[test]
fn header_strings_are_charged_and_shared() {
    let free = |extra: String| {
        let mut s = rand_spec(19);
        s.extra = extra;
        let m = open(&s);
        (m.budget.free(), m)
    };
    let (base, _) = free(String::new());
    let model = "m".repeat(300_000);
    let (with, m) = free(format!(
        "<Attribute Group=\"0x0008\" Element=\"0x1090\">{model}</Attribute>"
    ));
    assert_eq!(m.model_name(), model);
    assert_eq!(base - with, model.len() as u64);
    let v = m.wide_grid();
    assert!(Arc::ptr_eq(&m.h.model_name, &v.h.model_name));
}

#[test]
fn envelopes_scale_by_level() {
    let m = open(&rand_spec(19));
    for level in 0..m.levels() {
        let got: Vec<Vec<Point>> = m.envelopes(level).map(Iterator::collect).collect();
        let want: Vec<Vec<Point>> = (m.h.envelopes.iter())
            .map(|p| {
                (p.iter())
                    .map(|v| Point {
                        x: v.x / (1 << level),
                        y: v.y / (1 << level),
                    })
                    .collect()
            })
            .collect();
        assert_eq!(got, want, "level {level}");
    }
}

fn open_with(s: &FileSpec, limits: &Limits) -> Result<Image, Error> {
    let (file, _) = fixture::file(s);
    let n = file.len() as u64;
    Image::open(file, n, limits, &go())
}

fn with_threads(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

/// Several tiles on every level: 8 x 4 at level 0.
const MULTI: Rect = Rect {
    min_x: 0,
    min_y: 0,
    max_x: 1500,
    max_y: 900,
};

/// The threads of a call never pass `max_threads`, whatever the pool, for the image and its
/// views.
#[test]
fn thread_peak_within_max_threads() {
    let pool = crate::ThreadPool::new(8).unwrap();
    for mt in [1, 2, 3, Limits::default().max_threads] {
        let m = open_with(&rand_spec(19), &with_threads(mt))
            .unwrap()
            .with_pool(pool.clone());
        let v = m.wide_grid();
        assert_eq!((v.threads, v.pool.is_some()), (mt, true), "a view inherits");
        for im in [&m, &v] {
            im.decode_region_rgba(0, MULTI, &mut vec![0; 4 * 1500 * 900], 4 * 1500, &go())
                .unwrap();
            assert!(im.gauge.peak() <= mt.max(1), "max_threads {mt}");
        }
        if mt == 1 {
            assert_eq!(m.gauge.peak(), 1);
        }
    }
    // 0 counts as 1.
    let m = open_with(&rand_spec(19), &with_threads(0)).unwrap();
    assert_eq!(m.threads, 1);
}

/// `max_threads` 1 and a caller's pool never start the shared pool. The test thread is the
/// caller of its own decodes, so its own count of asks for the pool tells.
#[test]
fn shared_pool_only_when_needed() {
    use crate::parallel::SHARED_ASKS;
    let decode = |m: &Image| {
        m.decode_region_rgba(0, MULTI, &mut vec![0; 4 * 1500 * 900], 4 * 1500, &go())
            .unwrap();
    };
    let asks = || SHARED_ASKS.with(|c| c.get());
    decode(&open_with(&rand_spec(19), &with_threads(1)).unwrap());
    assert_eq!(asks(), 0, "max_threads 1");
    let pool = crate::ThreadPool::new(2).unwrap();
    decode(
        &open_with(&rand_spec(19), &with_threads(4))
            .unwrap()
            .with_pool(pool),
    );
    assert_eq!(asks(), 0, "a caller's pool wins");
    decode(&open_with(&rand_spec(19), &with_threads(4)).unwrap());
    assert!(asks() > 0, "no pool: the shared pool");
}

/// The pixels do not depend on the thread count or the pool.
#[test]
fn output_is_the_same_for_every_thread_count() {
    let pool = crate::ThreadPool::new(3).unwrap();
    for spec in [rand_spec(16), rand_spec(19), v1_spec()] {
        let one = {
            let m = open_with(&spec, &with_threads(1)).unwrap();
            [
                region_rgba(&m, 0, MULTI),
                region_rgba(&m, 1, Rect::new(5, 7, 700, 430)),
            ]
        };
        for mt in [2, 3, Limits::default().max_threads] {
            for p in [None, Some(pool.clone())] {
                let mut m = open_with(&spec, &with_threads(mt)).unwrap();
                if let Some(p) = p {
                    m = m.with_pool(p);
                }
                assert!(region_rgba(&m, 0, MULTI) == one[0], "max_threads {mt}");
                assert!(
                    region_rgba(&m, 1, Rect::new(5, 7, 700, 430)) == one[1],
                    "max_threads {mt}, level 1"
                );
            }
        }
    }
}

/// Tiles inside the region decode straight into `dst` and the edge tiles through staging: with a
/// padded stride the pixels are those of a tight decode and the padding is not touched, for
/// every thread count.
#[test]
fn region_with_padded_stride_matches_and_spares_the_padding() {
    let spec = rand_spec(19);
    let want = region_rgba(&open_with(&spec, &with_threads(1)).unwrap(), 0, MULTI);
    let (row, pad) = (4 * MULTI.dx() as usize, 12);
    for mt in [1, 3] {
        let m = open_with(&spec, &with_threads(mt)).unwrap();
        let mut got = vec![0xABu8; (row + pad) * MULTI.dy() as usize];
        m.decode_region_rgba(0, MULTI, &mut got, row + pad, &go())
            .unwrap();
        for (y, (g, w)) in got.chunks(row + pad).zip(want.chunks(row)).enumerate() {
            assert!(g[..row] == *w, "max_threads {mt}, row {y}");
            assert!(g[row..].iter().all(|&b| b == 0xAB), "padding of row {y}");
        }
    }
}

/// The measured peak of the budget stays under `max_alloc`, tight values included, and a decode
/// that fits gives the pixels of the unlimited one; one that does not is a limit error.
#[test]
fn budget_peak_within_max_alloc() {
    let spec = rand_spec(19);
    let base = open_with(&spec, &with_threads(3)).unwrap();
    let want_region = region_rgba(&base, 0, MULTI);
    let want_tile = tile_rgba(&base, 0, 2, 1);
    let pool = crate::ThreadPool::new(3).unwrap();
    let mut decoded = 0;
    let mut refused = 0;
    let limits = [
        u64::MAX,
        64 << 20,
        16 << 20,
        4 << 20,
        2 << 20,
        1 << 20,
        512 << 10,
        SCRATCH_BYTES,
        2 * SCRATCH_BYTES,
        3 * SCRATCH_BYTES,
        4 * SCRATCH_BYTES,
        6 * SCRATCH_BYTES,
        100_000,
    ];
    for (max_threads, max_alloc) in [1, 2, Limits::default().max_threads]
        .into_iter()
        .flat_map(|t| limits.map(|a| (t, a)))
    {
        let l = Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        };
        let Ok(m) = open_with(&spec, &l) else {
            continue; // the header alone is over it
        };
        let m = m.with_pool(pool.clone());
        let mut px = vec![0u8; 4 * 1500 * 900];
        let r = m.decode_region_rgba(0, MULTI, &mut px, 4 * 1500, &go());
        match r {
            Ok(()) => {
                assert!(
                    px == want_region,
                    "max_alloc {max_alloc}, max_threads {max_threads}"
                );
                decoded += 1;
            }
            Err(e) => {
                assert!(matches!(e, Error::MemoryLimit | Error::AllocLimit), "{e:?}");
                refused += 1;
            }
        }
        let mut tile = vec![0u8; 4 * 256 * 256];
        match m.decode_tile_rgba(0, 2, 1, &mut tile, 4 * 256, &go()) {
            Ok(()) => assert!(tile == want_tile, "max_alloc {max_alloc}, tile"),
            Err(e) => assert!(matches!(e, Error::MemoryLimit | Error::AllocLimit), "{e:?}"),
        }
        let mut yc = YCbCr::new(
            Rect::new(0, 0, 300, 200),
            Subsample::S420,
            &Limits::default(),
        )
        .unwrap();
        if let Err(e) = m.decode_region_ycbcr(0, Rect::new(0, 0, 300, 200), &mut yc, &go()) {
            assert!(matches!(e, Error::MemoryLimit | Error::AllocLimit), "{e:?}");
        }
        assert!(
            m.budget.peak() <= max_alloc,
            "max_alloc {max_alloc}, max_threads {max_threads}: peak {}",
            m.budget.peak()
        );
    }
    assert!(
        decoded > 0 && refused > 0,
        "{decoded} decoded, {refused} refused"
    );
}

// ----- Decoder: buffer reuse -----

fn hash<T: Copy + Into<i64>>(v: &[T]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        let x: i64 = x.into();
        (h ^ x as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The results (errors included) and output hashes of a fixed run of calls on `m`, and the peak
/// of its budget.
fn observe(m: &Image) -> (Vec<String>, u64) {
    let c = go();
    let mut log = Vec::new();
    let full = Rect::new(5, 7, 605, 457);
    let mut rgba = vec![0u8; 4 * 600 * 450];
    let r = m.decode_region_rgba(0, full, &mut rgba, 4 * 600, &c);
    log.push(format!("rgba {r:?} {:x}", hash(&rgba)));
    let mut rgba64 = vec![0u16; 4 * 300 * 200];
    let r = m.decode_region_rgba64(1, Rect::new(0, 0, 300, 200), &mut rgba64, 4 * 300, &c);
    log.push(format!("rgba64 {r:?} {:x}", hash(&rgba64)));
    let part = Rect::new(100, 50, 400, 300);
    let mut samples = vec![0u16; 3 * 300 * 250];
    let r = m.decode_region_samples(0, part, &mut samples, 3 * 300, &c);
    log.push(format!("samples {r:?} {:x}", hash(&samples)));
    let mut signed = vec![0i16; 3 * 300 * 250];
    let r = m.decode_region_signed_samples(0, part, &mut signed, 3 * 300, &c);
    log.push(format!("signed {r:?} {:x}", hash(&signed)));
    let mut tile = vec![0u8; 4 * 256 * 256];
    for (tx, ty) in [(2, 1), (99, 99)] {
        let r = m.decode_tile_rgba(0, tx, ty, &mut tile, 4 * 256, &c);
        log.push(format!("tile {tx} {ty} {r:?} {:x}", hash(&tile)));
    }
    let area = Rect::new(0, 0, 300, 200);
    let mut yc = YCbCr::new(area, Subsample::S420, &Limits::default()).unwrap();
    let r = m.decode_region_ycbcr(0, area, &mut yc, &c);
    log.push(format!(
        "ycbcr {r:?} {:x} {:x} {:x}",
        hash(&yc.y),
        hash(&yc.cb),
        hash(&yc.cr)
    ));
    let r = m.decode_region_rgba(99, full, &mut rgba, 4 * 600, &c);
    log.push(format!("bad level {r:?}"));
    let r = m.decode_region_rgba(0, full, &mut rgba, 4 * 600, &AtomicBool::new(true));
    log.push(format!("cancelled {r:?}"));
    (log, m.budget.peak())
}

/// With a `Decoder` the outputs, the errors and the budget's peak are those of an image without
/// one, on the first image and on the next one that takes the first one's buffers (of the same
/// file or of another, one `Decoder` serves every spec), for generous and tight limits (the tight ones evict entries and scratch).
#[test]
fn decoder_changes_nothing_observable() {
    let specs = [
        rand_spec(19),
        rand_spec(16),
        v1_spec(),
        wide_spec(19, 9),
        wide_spec(16, 8),
        const_spec(300, 8),
        const_spec(100, 9),
        api_spec(false),
        api_spec(true),
    ];
    for max_alloc in [u64::MAX, 4 << 20, 2 << 20] {
        let limits = Limits {
            max_threads: 1,
            max_alloc,
            ..Limits::default()
        };
        // One Decoder for every spec: buffers move from one file to a different one.
        let d = Decoder::new(&limits);
        for (i, spec) in specs.iter().enumerate() {
            let (file, _) = fixture::file(spec);
            let file = Arc::new(file);
            let n = file.len() as u64;
            let Ok(plain) = Image::open(file.clone(), n, &limits, &go()) else {
                continue;
            };
            let want = observe(&plain);
            drop(plain);
            for round in 0..2 {
                let im = Image::open(file.clone(), n, &limits, &go())
                    .unwrap()
                    .with_decoder(&d);
                let got = observe(&im);
                let tag = format!("fixture {i}, max_alloc {max_alloc}, round {round}");
                if max_alloc == u64::MAX {
                    assert_eq!(got, want, "{tag}");
                    continue;
                }
                // Which of several equally old entries a tight limit evicts is up to the hash
                // map, so a call may be refused with one image and not with the other: what
                // succeeds is the same, and the peak is within the limit.
                assert!(got.1 <= max_alloc, "{tag}");
                for (g, w) in got.0.iter().zip(&want.0) {
                    if g.contains("Ok(())") && w.contains("Ok(())") {
                        assert_eq!(g, w, "{tag}");
                    }
                }
            }
        }
    }
}

/// A decoder keeps at most `Limits::max_alloc` bytes, and the views of an image use its decoder.
#[test]
fn decoder_keeps_at_most_max_alloc() {
    let mut rgba = vec![0u8; 4 * 300 * 200];
    let r = Rect::new(0, 0, 300, 200);
    for max_alloc in [0, 1 << 20, 64 << 20, u64::MAX] {
        let d = Decoder::new(&Limits {
            max_alloc,
            ..Limits::default()
        });
        {
            let im = open(&rand_spec(19)).with_decoder(&d);
            im.decode_region_rgba(0, r, &mut rgba, 4 * 300, &go())
                .unwrap();
            let wide = im.wide_grid();
            assert!(wide.dec.is_some(), "max_alloc {max_alloc}");
            wide.decode_region_rgba(0, r, &mut rgba, 4 * 300, &go())
                .unwrap();
        }
        assert!(d.held() <= max_alloc, "max_alloc {max_alloc}: {}", d.held());
        assert!(
            max_alloc < 64 << 20 || d.held() > 0,
            "max_alloc {max_alloc}"
        );
    }
}
