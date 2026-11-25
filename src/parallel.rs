//! Fan-out of a region's tiles over threads, with a worker's panic returned as an error. Port of
//! `internal/parallel`, on a rayon pool: the caller's ([`ThreadPool`], see
//! [`Image::with_pool`](crate::Image::with_pool)) or the crate's lazy shared one. A call's workers
//! are at most `Limits::max_threads`, the calling thread included. Rayon is here only because
//! `std::thread::scope` cannot reuse a thread: a pool's threads outlive the call, so a small
//! region pays a wake-up, not a thread start. Each tile's data is a staging buffer of its own and
//! rayon's scope lets a pool thread borrow the caller's data, so nothing here is `unsafe`.
//!
//! Memory: a pool's thread stacks belong to the pool, not to a call, so the budget books a
//! call's own scratch only (staging buffers, coefficient caches, scratch).

use crate::Error;
use crate::budget::lock;
use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Worker threads that decodes share. Create one, keep it, and pass it to
/// [`Image::with_pool`](crate::Image::with_pool): each call on that image then runs its tiles on
/// these threads. An image without one uses the crate's lazy shared pool, which starts on the
/// first call with `max_threads` above 1 that has more than one tile, and never has more than
/// [`std::thread::available_parallelism`] threads; a caller's pool always wins.
/// `Limits::max_threads` still caps the workers of one call, the calling thread included, and a
/// call uses at most one more than the pool has threads. Clones share the threads, which stop
/// when the last clone drops. A pool is not charged to `Limits::max_alloc`; its threads' stacks
/// and bookkeeping are its owner's.
#[derive(Clone, Debug)]
pub struct ThreadPool(Arc<rayon::ThreadPool>);

impl ThreadPool {
    /// A pool of `threads` workers (at least 1), started now. No `Limits` apply: the pool is not
    /// charged to `max_alloc`.
    ///
    /// # Errors
    /// [`Error::OutOfMemory`] if the operating system will not start the threads.
    pub fn new(threads: usize) -> Result<ThreadPool, Error> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("isyntax-worker-{i}"))
            .build()
            .map(|p| ThreadPool(Arc::new(p)))
            .map_err(|_| Error::OutOfMemory)
    }

    /// How many worker threads the pool has. No limits apply.
    pub fn threads(&self) -> usize {
        self.0.current_num_threads()
    }
}

/// The crate's shared pool: built on first use, `available_parallelism` threads, no other
/// state. `None` if the threads could not be started (calls then run serially).
static SHARED: OnceLock<Option<ThreadPool>> = OnceLock::new();

#[cfg(test)]
thread_local! {
    /// How often this thread asked for the shared pool (a test thread is the caller of its own
    /// decodes, so this tells a test that `max_threads` 1 or a caller's pool never reached it).
    pub(crate) static SHARED_ASKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn shared_pool() -> Option<&'static ThreadPool> {
    #[cfg(test)]
    SHARED_ASKS.with(|c| c.set(c.get() + 1));
    SHARED
        .get_or_init(|| {
            ThreadPool::new(std::thread::available_parallelism().map_or(1, |n| n.get())).ok()
        })
        .as_ref()
}

/// The threads of an image's calls that are running now and the most at once, the calling thread
/// included. Per image (its views share it), so a test can read it without global state.
pub(crate) struct Gauge {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Gauge {
    pub(crate) fn new() -> Self {
        Self {
            live: AtomicUsize::new(1),
            peak: AtomicUsize::new(1),
        }
    }

    #[cfg(test)]
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

/// A pool worker counted in a [`Gauge`] while it runs.
struct Live<'a>(&'a Gauge);

impl<'a> Live<'a> {
    fn enter(g: &'a Gauge) -> Self {
        let n = g.live.fetch_add(1, Ordering::Relaxed) + 1;
        g.peak.fetch_max(n, Ordering::Relaxed);
        Live(g)
    }
}

impl Drop for Live<'_> {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(crate) fn panic_message(p: Box<dyn Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    match p.downcast::<String>() {
        Ok(s) => *s,
        Err(_) => "panic".to_string(),
    }
}

/// Calls `f(i)` for every `i` in `0..n` on at most `threads` threads, the caller's included, from
/// `pool` or else the shared pool: `min(threads, n, pool threads + 1)` workers. With one worker
/// (`threads` 1, or one item) it runs on the caller and never starts the shared pool. After the
/// first error no new items start; `for_each` returns it once every worker has stopped. A panic
/// in `f` is [`Error::Internal`].
pub(crate) fn for_each<F>(
    threads: usize,
    pool: Option<&ThreadPool>,
    gauge: &Gauge,
    n: usize,
    f: F,
) -> Result<(), Error>
where
    F: Fn(usize) -> Result<(), Error> + Sync,
{
    let want = threads.min(n).max(1);
    let pool = match pool {
        _ if want < 2 => None,
        Some(p) => Some(p),
        None => shared_pool(),
    };
    let nw = pool.map_or(1, |p| want.min(p.threads() + 1));
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let first: Mutex<Option<Error>> = Mutex::new(None);
    let fail = |e: Error| {
        let mut g = lock(&first);
        if g.is_none() {
            *g = Some(e);
        }
        stop.store(true, Ordering::Relaxed);
    };
    let run = || {
        // Else a panic would unwind through the scope.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !stop.load(Ordering::Relaxed) {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return;
                }
                if let Err(e) = f(i) {
                    fail(e);
                    return;
                }
            }
        }));
        if let Err(p) = r {
            fail(Error::Internal(format!("panic: {}", panic_message(p))));
        }
    };
    match pool {
        Some(p) if nw > 1 => p.0.in_place_scope(|s| {
            let run = &run;
            for _ in 1..nw {
                s.spawn(move |_| {
                    let _live = Live::enter(gauge);
                    run();
                });
            }
            run();
        }),
        _ => run(),
    }
    match lock(&first).take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn for_each_sums_and_stops() {
        let g = Gauge::new();
        let sum = AtomicU64::new(0);
        for_each(4, None, &g, 1000, |i| {
            sum.fetch_add(i as u64, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();
        assert_eq!(sum.load(Ordering::Relaxed), 999 * 1000 / 2);

        let r = for_each(4, None, &g, 100, |i| {
            if i == 50 {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert_eq!(r, Err(Error::Cancelled));

        for_each(4, None, &g, 0, |_| Err(Error::Cancelled)).unwrap();
    }

    #[test]
    fn for_each_returns_a_panic_as_an_error() {
        let g = Gauge::new();
        let r = for_each(4, None, &g, 100, |i| {
            assert!(i != 7, "worker");
            Ok(())
        });
        match r {
            Err(Error::Internal(m)) => assert!(m.contains("worker"), "{m}"),
            other => panic!("got {other:?}"),
        }
    }
}
