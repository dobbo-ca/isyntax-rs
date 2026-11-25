//! Shared by the `open` and `decode` targets (`#[path = "../common.rs"] mod common;`).

use isyntax::{Limits, ReadAt};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

/// A reader that is cut at `at`: from there on it returns an error, or end of file (a short read).
pub struct Flaky {
    pub data: Vec<u8>,
    pub at: usize,
    pub short: bool,
}

impl ReadAt for Flaky {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let end = self.at.min(self.data.len());
        let off = usize::try_from(offset).unwrap_or(usize::MAX);
        if off >= end {
            return if self.short {
                Ok(0)
            } else {
                Err(io::Error::other("fault"))
            };
        }
        let n = buf.len().min(end - off);
        buf[..n].copy_from_slice(&self.data[off..off + n]);
        Ok(n)
    }
}

/// Limits from `h`, so that a small limit is the one that fires: half the inputs get the largest
/// (`max_alloc` 128 MiB; with the doubling of a `Vec`, no single allocation is over 256 MiB, under
/// `-malloc_limit_mb=300`), the others `max_alloc` 1 to 128 MiB, `max_pixels` 2^10 to 2^32 and
/// `max_header` 1 KiB to 8 MiB. A quarter of the seeds (bits 40 and 41 of `h` both set) get
/// `max_threads` 1, the serial region path.
pub fn limits(h: u64) -> Limits {
    let max_threads = if h >> 40 & 3 == 3 {
        1
    } else {
        Limits::default().max_threads
    };
    if h & 1 == 0 {
        return Limits {
            max_alloc: 128 << 20,
            max_pixels: 1 << 32,
            max_header: 8 << 20,
            max_threads,
            ..Limits::default()
        };
    }
    Limits {
        max_alloc: 1 << (20 + (h >> 1) % 8),
        max_pixels: 1 << (10 + (h >> 4) % 23),
        max_header: 1 << (10 + (h >> 9) % 14),
        max_threads,
        ..Limits::default()
    }
}

/// Runs `f` with a cancel flag that another thread sets after `delay` yields (never if 0).
pub fn with_cancel<T>(delay: u64, f: impl FnOnce(&AtomicBool) -> T) -> T {
    let cancel = AtomicBool::new(false);
    std::thread::scope(|sc| {
        if delay > 0 {
            sc.spawn(|| {
                for _ in 0..delay {
                    std::thread::yield_now();
                }
                cancel.store(true, Ordering::Relaxed);
            });
        }
        f(&cancel)
    })
}
