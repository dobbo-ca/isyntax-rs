//! [`Decoder`]: the working buffers of a decode that are kept between images, so a worker that
//! decodes tile after tile does not allocate after the first ones.

use crate::Limits;
use crate::budget::lock;
use crate::cache::{Buf, Key, Slot};
use crate::decoder::{SCRATCH_BYTES, Scratch};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

/// The free buffers. `bytes` is what they hold, at most `max`.
pub(crate) struct Store {
    max: u64,
    bytes: u64,
    /// Coefficient cache entries nobody else holds, without their charge.
    bufs: Vec<Arc<Buf>>,
    /// Tile scratch without its charge.
    scratch: Vec<Box<Scratch>>,
    /// The emptied maps of the caches, by cache kind.
    maps: Vec<(usize, HashMap<Key, Slot>)>,
    u8s: Vec<Vec<u8>>,
    u16s: Vec<Vec<u16>>,
    i16s: Vec<Vec<i16>>,
    i32s: Vec<Vec<i32>>,
}

impl Store {
    /// Whether `n` more bytes fit; if so they are counted.
    fn keep(&mut self, n: u64) -> bool {
        if self.bytes.saturating_add(n) > self.max {
            return false;
        }
        self.bytes += n;
        true
    }

    fn release(&mut self, n: u64) {
        self.bytes = self.bytes.saturating_sub(n);
    }
}

/// The element types of the staging and conversion buffers.
pub(crate) trait Stage: Copy {
    fn list(s: &mut Store) -> &mut Vec<Vec<Self>>;
}

macro_rules! stage {
    ($t:ty, $f:ident) => {
        impl Stage for $t {
            fn list(s: &mut Store) -> &mut Vec<Vec<$t>> {
                &mut s.$f
            }
        }
    };
}

stage!(u8, u8s);
stage!(u16, u16s);
stage!(i16, i16s);
stage!(i32, i32s);

/// Working buffers kept for the next image: a worker that decodes tile after tile from one
/// [`Image`](crate::Image) after another allocates nothing once the buffers are warm. Make one
/// with [`Decoder::new`], give it to each image with [`Image::with_decoder`](crate::Image::with_decoder),
/// and keep it as long as the worker lives. A `Decoder` is a handle: clones share the same buffers,
/// and it is `Send` and `Sync`, so every worker thread can hold one or share one.
///
/// What it keeps, when an image that uses it drops or evicts them: the coefficient cache entries
/// (found by length), the tile scratch, the staging and conversion buffers of the region and
/// YCbCr calls, and the emptied tables of the caches. A decode asks it for a buffer where it
/// would allocate one, after charging the buffer to the image's `max_alloc` as it always does, so
/// limits, errors, outputs and the peak of the budget are the same with and without a `Decoder`.
///
/// The free buffers are not charged to any image. A `Decoder` holds at most `Limits::max_alloc`
/// bytes of them (the `limits` of [`Decoder::new`]); a buffer that would pass that is freed.
/// It never evicts: cache entries and staging buffers are found by exact length, so buffers of a length that never comes
/// back stay held (up to that `max_alloc`) until the `Decoder` is dropped. When an image evicts a
/// cache entry under memory pressure, the entry moves to the `Decoder` instead of being freed: the
/// image's budget drops, the process's memory does not. Build it with a `max_alloc` the worker can
/// afford to keep resident, and drop it and every image that uses it to free what it holds.
///
/// ```no_run
/// use isyntax::{Decoder, Image, Limits, Rect};
/// use std::sync::atomic::AtomicBool;
///
/// # fn main() -> Result<(), isyntax::Error> {
/// let limits = Limits { max_threads: 1, ..Limits::default() };
/// let decoder = Decoder::new(&limits);
/// let cancel = AtomicBool::new(false);
/// let mut rgba = vec![0u8; 4 * 256 * 256];
/// for path in ["a.isyntax", "b.isyntax"] {
///     let f = std::fs::File::open(path)?;
///     let size = f.metadata()?.len();
///     let im = Image::open(f, size, &limits, &cancel)?.with_decoder(&decoder);
///     im.decode_region_rgba(0, Rect::new(0, 0, 256, 256), &mut rgba, 4 * 256, &cancel)?;
/// } // the second image reuses what the first one used
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Decoder(Arc<Mutex<Store>>);

impl fmt::Debug for Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decoder").finish_non_exhaustive()
    }
}

impl Decoder {
    /// An empty `Decoder` that will keep at most `limits.max_alloc` bytes of free buffers. No
    /// other limit applies.
    pub fn new(limits: &Limits) -> Decoder {
        Decoder(Arc::new(Mutex::new(Store {
            max: limits.max_alloc,
            bytes: 0,
            bufs: Vec::new(),
            scratch: Vec::new(),
            maps: Vec::new(),
            u8s: Vec::new(),
            u16s: Vec::new(),
            i16s: Vec::new(),
            i32s: Vec::new(),
        })))
    }

    /// The bytes of free buffers it holds now.
    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        lock(&self.0).bytes
    }

    /// A cache entry of `n` int16, not zeroed, with no charge.
    pub(crate) fn take_buf(&self, n: usize) -> Option<Arc<Buf>> {
        let mut s = lock(&self.0);
        let i = s.bufs.iter().position(|b| b.data.len() == n)?;
        let b = s.bufs.swap_remove(i);
        s.release(2 * n as u64);
        Some(b)
    }

    /// Keeps `b`, a unique entry without its charge ([`Buf::free`]).
    pub(crate) fn put_buf(&self, b: Arc<Buf>) {
        let mut s = lock(&self.0);
        if s.keep(2 * b.data.len() as u64) {
            s.bufs.push(b);
        }
    }

    pub(crate) fn take_scratch(&self) -> Option<Box<Scratch>> {
        let mut s = lock(&self.0);
        let sc = s.scratch.pop()?;
        s.release(SCRATCH_BYTES);
        Some(sc)
    }

    /// Keeps `sc`, which no longer holds its charge.
    pub(crate) fn put_scratch(&self, sc: Box<Scratch>) {
        let mut s = lock(&self.0);
        if s.keep(SCRATCH_BYTES) {
            s.scratch.push(sc);
        }
    }

    /// The emptied map of cache `kind` with the most room.
    pub(crate) fn take_map(&self, kind: usize) -> Option<HashMap<Key, Slot>> {
        let mut s = lock(&self.0);
        let i = (s.maps.iter().enumerate())
            .filter(|(_, (k, _))| *k == kind)
            .max_by_key(|(_, (_, m))| m.capacity())
            .map(|(i, _)| i)?;
        let (_, m) = s.maps.swap_remove(i);
        s.release(map_bytes(&m));
        Some(m)
    }

    /// Keeps the emptied map `m` of cache `kind`.
    pub(crate) fn put_map(&self, kind: usize, m: HashMap<Key, Slot>) {
        let mut s = lock(&self.0);
        if m.capacity() > 0 && s.keep(map_bytes(&m)) {
            s.maps.push((kind, m));
        }
    }

    /// A free buffer of exactly `n` elements, every one set to `v`.
    fn take<T: Stage>(&self, n: usize, v: T) -> Option<Vec<T>> {
        let mut s = lock(&self.0);
        let list = T::list(&mut s);
        let i = list.iter().position(|b| b.len() == n)?;
        let mut b = list.swap_remove(i);
        s.release((b.capacity() * size_of::<T>()) as u64);
        drop(s);
        b.fill(v);
        Some(b)
    }

    fn give<T: Stage>(&self, v: Vec<T>) {
        let mut s = lock(&self.0);
        if s.keep((v.capacity() * size_of::<T>()) as u64) {
            T::list(&mut s).push(v);
        }
    }
}

fn map_bytes(m: &HashMap<Key, Slot>) -> u64 {
    (m.capacity() * size_of::<(Key, Slot)>()) as u64
}

/// A buffer on loan from a [`Decoder`] (or a plain one without it), given back when dropped.
pub(crate) struct Lent<T: Stage> {
    v: Vec<T>,
    home: Option<Decoder>,
}

impl<T: Stage> Lent<T> {
    /// A buffer of `n` elements set to `v`: a free one of `home` if it has one, else a new one.
    pub(crate) fn new(home: Option<&Decoder>, n: usize, v: T) -> Result<Lent<T>, crate::Error> {
        let v = match home.and_then(|d| d.take(n, v)) {
            Some(b) => b,
            None => {
                let mut b = Vec::new();
                b.try_reserve_exact(n)
                    .map_err(|_| crate::Error::OutOfMemory)?;
                b.resize(n, v);
                b
            }
        };
        Ok(Lent {
            v,
            home: home.cloned(),
        })
    }
}

impl<T: Stage> Deref for Lent<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.v
    }
}

impl<T: Stage> DerefMut for Lent<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.v
    }
}

impl<T: Stage> Drop for Lent<T> {
    fn drop(&mut self) {
        if let Some(d) = &self.home {
            d.give(std::mem::take(&mut self.v));
        }
    }
}
