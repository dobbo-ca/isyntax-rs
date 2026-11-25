//! The real heap peak of an open and of decodes stays within `Limits::max_alloc`, for several
//! `max_threads` and `max_alloc` including tight ones. A counting global allocator (this binary
//! only) measures the live heap, so what the budget does not charge shows up here. It is one
//! `#[test]`, so nothing else allocates while a case runs.
//!
//! Not heap, so not counted: the stacks of the shared pool's threads. The pool is started before
//! the first measurement, so its thread bookkeeping is not in a peak either.

#![allow(dead_code)]

#[path = "../src/fixture.rs"]
mod fixture;
#[path = "../src/testutil.rs"]
mod testutil;

use isyntax::{Error, Image, Limits, Rect};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

fn add(n: usize) {
    let now = LIVE.fetch_add(n, SeqCst) + n;
    PEAK.fetch_max(now, SeqCst);
}

// SAFETY: every method forwards to `System` with the caller's arguments and only counts bytes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc` is `System`'s.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), SeqCst);
        // SAFETY: `p` and `l` are the caller's, from this allocator, so from `System`.
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // The old and the new block can be live together while the data moves: count both.
        add(new);
        // SAFETY: `p` and `l` are the caller's, from this allocator, so from `System`.
        let q = unsafe { System.realloc(p, l, new) };
        if q.is_null() {
            LIVE.fetch_sub(new, SeqCst);
        } else {
            LIVE.fetch_sub(l.size(), SeqCst);
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Starts a measurement and returns the live heap it is relative to.
fn reset() -> usize {
    let now = LIVE.load(SeqCst);
    PEAK.store(now, SeqCst);
    now
}

/// Bytes the heap held above `base` at its highest since [`reset`].
fn peak_over(base: usize) -> usize {
    PEAK.load(SeqCst).saturating_sub(base)
}

/// What the budget does not charge and a call still allocates: the job and error boxes of the
/// workers, the cache's bookkeeping, a thread's lazily made thread-locals.
const SLACK: usize = 512 << 10;

fn spec() -> fixture::FileSpec {
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
    s.compressor = 19;
    s.offset_x = 3;
    s.offset_y = 5;
    s
}

#[test]
fn heap_peak_within_max_alloc() {
    let no = AtomicBool::new(false);
    let (file, _) = fixture::file(&spec());
    let size = file.len() as u64;
    let full = Rect::new(0, 0, 1500, 900);
    let part = Rect::new(100, 50, 700, 400);
    let default_threads = Limits::default().max_threads;

    // The caller's buffers are not the call's.
    let mut px = vec![0u8; 4 * 1500 * 900];
    let mut px2 = vec![0u8; 4 * 600 * 350];

    // Starts the shared pool and warms the allocator before any measurement.
    let im = Image::open(file.clone(), size, &Limits::default(), &no).unwrap();
    im.decode_region_rgba(0, full, &mut px, 4 * 1500, &no)
        .unwrap();
    let want = px.clone();
    drop(im);

    for max_threads in [1, 2, default_threads] {
        let mut decoded = 0;
        for max_alloc in [
            64 << 20,
            16 << 20,
            4 << 20,
            3 << 20,
            2 << 20,
            1 << 20,
            100_000,
        ] {
            let limits = Limits {
                max_alloc,
                max_threads,
                ..Limits::default()
            };
            let reader = file.clone();
            let base = reset();
            let opened = Image::open(reader, size, &limits, &no);
            // `base` already holds the reader that the open takes.
            let worst = peak_over(base);
            let tag = format!("max_alloc {max_alloc}, max_threads {max_threads}");
            let im = match opened {
                Ok(im) => im,
                Err(e) => {
                    assert_eq!(e, Error::AllocLimit, "{tag}");
                    assert!(worst <= max_alloc as usize + SLACK, "open, {tag}: {worst}");
                    continue;
                }
            };
            assert!(worst <= max_alloc as usize + SLACK, "open, {tag}: {worst}");
            let base = reset();
            let r1 = im.decode_region_rgba(0, full, &mut px, 4 * 1500, &no);
            let r2 = im.decode_region_rgba(1, part, &mut px2, 4 * 600, &no);
            let worst = peak_over(base);
            assert!(
                worst <= max_alloc as usize + SLACK,
                "decode, {tag}: {worst}"
            );
            for r in [&r1, &r2] {
                if let Err(e) = r {
                    assert!(
                        matches!(e, Error::MemoryLimit | Error::AllocLimit),
                        "{tag}: {e:?}"
                    );
                }
            }
            if r1.is_ok() {
                assert!(px == want, "{tag}");
                decoded += 1;
            }
        }
        assert!(decoded > 0, "max_threads {max_threads}: nothing decoded");
    }
}
