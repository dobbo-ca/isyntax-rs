//! The coefficient cache of one kind of entry: budget-charged, each entry computed once even when
//! several threads ask for it, evicted least recently used first. Port of `internal/decoder`'s
//! `cache` with `Arc` for Go's pins: an evicted entry that a decode still holds stays alive, and
//! stays charged, until the decode drops it.

use crate::Error;
use crate::budget::{Budget, Charge, lock};
use crate::reuse::Decoder;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// How often a thread waiting for another's entry looks at its cancel flag.
const WAIT: Duration = Duration::from_millis(20);

/// A cached run of int16, charged to the budget while it lives.
pub(crate) struct Buf {
    pub data: Vec<i16>,
    /// `None` only while the entry waits in a [`Decoder`] to be reused.
    _charge: Option<Charge>,
}

impl Buf {
    /// `a` without its charge, to be kept for reuse; `None` if another holder has it.
    pub(crate) fn free(mut a: Arc<Buf>) -> Option<Arc<Buf>> {
        Arc::get_mut(&mut a)?._charge = None;
        Some(a)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Key {
    /// The recon variant (truncation), 0 for the rest.
    pub v: u8,
    pub s: i32,
    pub x: i32,
    pub y: i32,
}

pub(crate) enum Slot {
    /// Another thread is computing it.
    Pending,
    Ready {
        buf: Arc<Buf>,
        /// The clock at the last use; 0 evicts first.
        used: u64,
        /// Centre uses, see `used`.
        uses: i32,
    },
}

struct Inner {
    map: HashMap<Key, Slot>,
    /// Bytes of the ready entries.
    bytes: u64,
    clock: u64,
}

pub(crate) struct Cache {
    inner: Mutex<Inner>,
    cv: Condvar,
    /// Bytes the ready entries may hold.
    pub max: u64,
    budget: Arc<Budget>,
    /// Which of an image's four caches this is, to find its map again in a [`Decoder`].
    pub(crate) kind: usize,
    dec: Option<Decoder>,
}

enum Look {
    Hit(Arc<Buf>),
    Wait,
    Create,
}

impl Cache {
    pub(crate) fn new(max: u64, budget: Arc<Budget>) -> Cache {
        Cache {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                bytes: 0,
                clock: 0,
            }),
            cv: Condvar::new(),
            max,
            budget,
            kind: 0,
            dec: None,
        }
    }

    /// Takes entries, maps and their buffers from `d` from now on, and gives them back to it.
    pub(crate) fn set_decoder(&mut self, d: Decoder) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        if inner.map.is_empty()
            && let Some(map) = d.take_map(self.kind)
        {
            inner.map = map;
        }
        self.dec = Some(d);
    }

    /// Returns `key`'s entry, computing it with `fill` (into `n` zeroed int16) if no thread has.
    pub(crate) fn get(
        &self,
        key: Key,
        n: usize,
        cancel: &AtomicBool,
        fill: impl FnOnce(&mut [i16]) -> Result<(), Error>,
    ) -> Result<Arc<Buf>, Error> {
        loop {
            let mut g = lock(&self.inner);
            let look = {
                let inner = &mut *g;
                inner.clock += 1;
                let now = inner.clock;
                match inner.map.get_mut(&key) {
                    Some(Slot::Ready { buf, used, .. }) => {
                        *used = now;
                        Look::Hit(buf.clone())
                    }
                    Some(Slot::Pending) => Look::Wait,
                    None => {
                        inner.map.insert(key, Slot::Pending);
                        Look::Create
                    }
                }
            };
            match look {
                Look::Hit(b) => return Ok(b),
                Look::Wait => {
                    let (g, _) = self
                        .cv
                        .wait_timeout(g, WAIT)
                        .unwrap_or_else(|e| e.into_inner());
                    drop(g);
                    if cancel.load(Ordering::Relaxed) {
                        return Err(Error::Cancelled);
                    }
                }
                Look::Create => {
                    drop(g);
                    return self.fill(key, n, fill);
                }
            }
        }
    }

    fn fill(
        &self,
        key: Key,
        n: usize,
        fill: impl FnOnce(&mut [i16]) -> Result<(), Error>,
    ) -> Result<Arc<Buf>, Error> {
        let size = 2 * n as u64;
        let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<Arc<Buf>, Error> {
                let charge = self.reserve(size)?;
                let (mut buf, reused) = match self.dec.as_ref().and_then(|d| d.take_buf(n)) {
                    Some(b) => (b, true),
                    None => {
                        let mut data: Vec<i16> = Vec::new();
                        data.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
                        data.resize(n, 0);
                        (
                            Arc::new(Buf {
                                data,
                                _charge: None,
                            }),
                            false,
                        )
                    }
                };
                let b = Arc::get_mut(&mut buf)
                    .ok_or_else(|| Error::Internal("cache entry is shared".into()))?;
                if reused {
                    b.data.fill(0); // as a new entry's
                }
                b._charge = Some(charge);
                if let Err(e) = fill(&mut b.data) {
                    if let (Some(d), Some(b)) = (&self.dec, Buf::free(buf)) {
                        d.put_buf(b);
                    }
                    return Err(e);
                }
                Ok(buf)
            },
        ));
        // A panic in `fill` must not leave the slot pending for good: waiters would never wake.
        let made = match made {
            Ok(m) => m,
            Err(p) => {
                lock(&self.inner).map.remove(&key);
                self.cv.notify_all();
                std::panic::resume_unwind(p)
            }
        };
        let mut g = lock(&self.inner);
        let out = match made {
            Ok(buf) => {
                g.clock += 1;
                let used = g.clock;
                g.map.insert(
                    key,
                    Slot::Ready {
                        buf: buf.clone(),
                        used,
                        uses: 0,
                    },
                );
                g.bytes += size;
                Ok(buf)
            }
            Err(e) => {
                g.map.remove(&key);
                Err(e)
            }
        };
        drop(g);
        self.cv.notify_all();
        if out.is_ok() {
            loop {
                // The guard must be gone before evict_oldest takes it again.
                let over = lock(&self.inner).bytes > self.max;
                if !over || !self.evict_oldest(Some(key)) {
                    break;
                }
            }
        }
        out
    }

    /// Charges `n` bytes, evicting this cache's entries while others hold the memory.
    fn reserve(&self, n: u64) -> Result<Charge, Error> {
        loop {
            match Charge::new(&self.budget, n) {
                Err(Error::MemoryLimit) if self.evict_oldest(None) => {}
                r => return r,
            }
        }
    }

    /// Drops the least recently used ready entry other than `keep`; false if there is none.
    /// ponytail: a scan of the entries, a few thousand at most; an ordered index if it shows up.
    pub(crate) fn evict_oldest(&self, keep: Option<Key>) -> bool {
        let mut g = lock(&self.inner);
        let oldest = g
            .map
            .iter()
            .filter_map(|(k, s)| match s {
                Slot::Ready { used, .. } if Some(*k) != keep => Some((*used, *k)),
                _ => None,
            })
            .min_by_key(|&(used, _)| used)
            .map(|(_, k)| k);
        let Some(k) = oldest else {
            return false;
        };
        if let Some(Slot::Ready { buf, .. }) = g.map.remove(&k) {
            g.bytes -= 2 * buf.data.len() as u64;
            if let (Some(d), Some(b)) = (&self.dec, Buf::free(buf)) {
                d.put_buf(b);
            }
        }
        true
    }

    /// Counts a centre use of `key`; true at the `n`th (see `Image::retire`).
    pub(crate) fn used(&self, key: &Key, n: i32) -> bool {
        match lock(&self.inner).map.get_mut(key) {
            Some(Slot::Ready { uses, .. }) => {
                *uses += 1;
                *uses == n
            }
            _ => false,
        }
    }

    /// Makes `key` the next entry to evict.
    pub(crate) fn demote(&self, key: &Key) {
        if let Some(Slot::Ready { used, .. }) = lock(&self.inner).map.get_mut(key) {
            *used = 0;
        }
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, key: &Key) -> bool {
        matches!(lock(&self.inner).map.get(key), Some(Slot::Ready { .. }))
    }
}

impl Drop for Cache {
    /// An image's entries and map go back to its [`Decoder`].
    fn drop(&mut self) {
        let Some(d) = self.dec.take() else {
            return;
        };
        let mut map =
            std::mem::take(&mut self.inner.get_mut().unwrap_or_else(|e| e.into_inner()).map);
        for (_, slot) in map.drain() {
            if let Slot::Ready { buf, .. } = slot
                && let Some(b) = Buf::free(buf)
            {
                d.put_buf(b);
            }
        }
        d.put_map(self.kind, map);
    }
}
