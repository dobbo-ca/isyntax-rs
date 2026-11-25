//! With a warm `Decoder` and `max_threads` 1, a decode makes no allocator call: image A decodes
//! with the decoder and is dropped, image B opens on the same bytes with the same decoder, and the
//! same decodes on B (a cold coefficient cache) call the allocator zero times, and again on a second
//! run. A counting global allocator counts the calls (`alloc`, `alloc_zeroed` and `realloc`) of
//! the test's own thread, so other tests do not count.
//!
//! What still allocates, and is not covered here: `Image::open`, an error (its message), a
//! codeblock over 256 KiB (read into a buffer of its own), a call with `max_threads` above 1 (the
//! row pieces and the staging list of a region, and the rayon jobs), and the first calls on a
//! cold `Decoder`.

#![allow(dead_code)]

#[path = "../src/fixture.rs"]
mod fixture;
#[path = "../src/testutil.rs"]
mod testutil;

use isyntax::{Decoder, Image, Limits, Rect, Subsample, YCbCr};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

thread_local! {
    /// The allocator calls of this thread. A `Cell` of a plain integer needs no lazy set-up, so
    /// the allocator can touch it.
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

fn bump() {
    CALLS.with(|c| c.set(c.get() + 1));
}

fn calls() -> u64 {
    CALLS.with(Cell::get)
}

struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments and only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        // SAFETY: the caller's contract for `alloc` is `System`'s.
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        // SAFETY: the caller's contract for `alloc_zeroed` is `System`'s.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` and `l` are the caller's, from this allocator, so from `System`.
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        bump();
        // SAFETY: `p` and `l` are the caller's, from this allocator, so from `System`.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Random detail with missing tiles, as the unit tests' `rand_spec`; small, with several tiles.
fn spec(compressor: u32, bits: u32, v1: bool) -> fixture::FileSpec {
    let mut s = fixture::FileSpec::new(|sc, x, y, c, ll| {
        let mut r = testutil::Rng::new((sc * 1000 + x * 31 + y) as u64 * 8 + c as u64 + 1);
        let n = if ll { 1 } else { 3 };
        let mut v = vec![0i16; n * 128 * 128];
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
    s.bits_stored = bits;
    if v1 {
        s.version = "5.0".into();
    }
    s.levels = 2;
    s.width = 700;
    s.height = 500;
    s.offset_x = 3;
    s.offset_y = 5;
    s.set_exists(|s, x, y| s > 0 || !(x * 7 + y * 3).is_multiple_of(5));
    s
}

/// The caller's buffers, made before anything is counted.
struct Bufs {
    rgba: Vec<u8>,
    tile: Vec<u8>,
    tile64: Vec<u16>,
    samples: Vec<u16>,
    signed: Vec<i16>,
    ycc: YCbCr,
}

fn fnv(h: u64, it: impl Iterator<Item = u64>) -> u64 {
    it.fold(h, |h, v| (h ^ v).wrapping_mul(0x0100_0000_01b3))
}

/// The decodes of the test; returns a hash of every output.
fn run(im: &Image, b: &mut Bufs) -> u64 {
    let no = AtomicBool::new(false);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    // A region of several tiles, cut by the edge of the region and of the level.
    let r = Rect::new(5, 7, 605, 457);
    im.decode_region_rgba(0, r, &mut b.rgba, 4 * 600, &no)
        .unwrap();
    h = fnv(h, b.rgba.iter().map(|&v| u64::from(v)));
    let lv = im.level(0).unwrap();
    for ty in 0..lv.tiles_y as i64 {
        for tx in 0..lv.tiles_x as i64 {
            im.decode_tile_rgba(0, tx, ty, &mut b.tile, 4 * 256, &no)
                .unwrap();
            h = fnv(h, b.tile.iter().map(|&v| u64::from(v)));
            im.decode_tile_rgba64(0, tx, ty, &mut b.tile64, 4 * 256, &no)
                .unwrap();
            h = fnv(h, b.tile64.iter().map(|&v| u64::from(v)));
            im.decode_tile_ycbcr(0, tx, ty, &mut b.ycc, &no).unwrap();
            h = fnv(h, b.ycc.y.iter().map(|&v| u64::from(v)));
            h = fnv(h, b.ycc.cb.iter().map(|&v| u64::from(v)));
            h = fnv(h, b.ycc.cr.iter().map(|&v| u64::from(v)));
        }
    }
    let s = Rect::new(100, 50, 400, 300);
    im.decode_region_samples(0, s, &mut b.samples, 3 * 300, &no)
        .unwrap();
    h = fnv(h, b.samples.iter().map(|&v| u64::from(v)));
    im.decode_region_signed_samples(0, s, &mut b.signed, 3 * 300, &no)
        .unwrap();
    fnv(h, b.signed.iter().map(|&v| v as u64))
}

#[test]
fn second_decode_allocates_nothing() {
    let limits = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let no = AtomicBool::new(false);
    // iSyntax2 8 bits (compressors 19 and 16), 9 bits, and v1 16 bits.
    for (compressor, bits, v1) in [
        (19, 8, false),
        (16, 8, false),
        (19, 9, false),
        (16, 16, true),
    ] {
        let tag = format!("compressor {compressor}, {bits} bits, v1 {v1}");
        let (file, _) = fixture::file(&spec(compressor, bits, v1));
        let file = Arc::new(file);
        let size = file.len() as u64;
        let mut b = Bufs {
            rgba: vec![0; 4 * 600 * 450],
            tile: vec![0; 4 * 256 * 256],
            tile64: vec![0; 4 * 256 * 256],
            samples: vec![0; 3 * 300 * 250],
            signed: vec![0; 3 * 300 * 250],
            ycc: YCbCr::new(Rect::new(0, 0, 256, 256), Subsample::S420, &limits).unwrap(),
        };
        // The output without a decoder, for the comparison.
        let plain = Image::open(file.clone(), size, &limits, &no).unwrap();
        let want = run(&plain, &mut b);
        drop(plain);

        let d = Decoder::new(&limits);
        let a = Image::open(file.clone(), size, &limits, &no)
            .unwrap()
            .with_decoder(&d);
        assert_eq!(run(&a, &mut b), want, "A, {tag}");
        drop(a);

        let im = Image::open(file.clone(), size, &limits, &no)
            .unwrap()
            .with_decoder(&d);
        for round in 1..=2 {
            let before = calls();
            let got = run(&im, &mut b);
            let n = calls() - before;
            assert_eq!(n, 0, "{n} allocator calls in round {round} of B, {tag}");
            assert_eq!(got, want, "B round {round}, {tag}");
        }
    }
}
