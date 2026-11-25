use std::sync::OnceLock;

/// Caps on opening and decoding. Go had one process-wide memory limit and the constants below
/// (`maxHeader`); the defaults are finite.
///
/// | Field | Unit | Default | Over the limit |
/// |---|---|---|---|
/// | `max_alloc` | bytes, the peak of the image, its views and all their calls | 4 GiB | [`Error::AllocLimit`](crate::Error::AllocLimit), or [`Error::MemoryLimit`](crate::Error::MemoryLimit) when other decodes hold the room |
/// | `max_pixels` | pixels (width times height) of level 0 | 2^36 | [`Error::PixelLimit`](crate::Error::PixelLimit) |
/// | `max_header` | bytes of XML header | 256 MiB | [`Error::HeaderLimit`](crate::Error::HeaderLimit) |
/// | `max_threads` | threads, the caller's included | [`std::thread::available_parallelism`] | none: the call runs on fewer threads, same output |
///
/// ```
/// use isyntax::Limits;
/// // One thread (no pool is touched) and 64 MiB for the image and every call on it.
/// let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
/// # assert_eq!(limits.max_pixels, 1 << 36);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Bytes of working memory of an [`Image`](crate::Image) and its views together: the header
    /// parse while opening, then the coefficient caches and the scratch of every decode, which are
    /// charged before they are allocated. It is the enforced peak of every call on the image and
    /// its views together, all threads included (a pool's thread stacks are the pool's, not
    /// charged). Go: `SetMemoryLimit`, default half the cgroup limit, else none; `u64::MAX` is
    /// none here (see [`Image::set_memory_limit`](crate::Image::set_memory_limit)).
    pub max_alloc: u64,
    /// Most pixels of the full-size level (width times height, without the wavelet padding).
    /// Go had none; it bounds the tile grid the open loops over.
    pub max_pixels: u64,
    /// Most bytes of XML header before the 0x04 terminator. Go: 256 MiB. As in Go, it is tested after
    /// each 1 MiB chunk without the terminator, so a header under `max_header` plus 1 MiB opens.
    pub max_header: u64,
    /// Most threads one call uses, the calling thread included. 1 runs on the calling thread and
    /// never touches a pool; 0 counts as 1. The default is [`std::thread::available_parallelism`].
    /// [`Image::open`](crate::Image::open) stores it and the image's views inherit it. A region
    /// uses as many threads as it has tiles, and no more than the pool has plus the caller
    /// ([`ThreadPool`](crate::ThreadPool)), so the output is the same for every value.
    pub max_threads: usize,
}

/// `available_parallelism`, read once: it reads cgroup files on every call.
fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_alloc: 4 << 30,
            max_pixels: 1 << 36,
            max_header: 256 << 20,
            max_threads: default_threads(),
        }
    }
}
