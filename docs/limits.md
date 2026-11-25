# Resource limits

`Image::open` takes `&Limits` and a cancel flag; the image and its views keep the limits, and every
decode call takes a cancel flag. A constrained call, on the calling thread only (no pool touched) and
at most 64 MiB for the image and every call on it:

```rust,no_run
# use isyntax::{Image, Limits, Rect};
# use std::sync::atomic::AtomicBool;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let cancel = AtomicBool::new(false);
# let f = std::fs::File::open("slide.isyntax")?;
# let size = f.metadata()?.len();
# let r = Rect::new(0, 0, 1024, 1024);
# let mut rgba = vec![0u8; 4 * 1024 * 1024];
let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
let im = Image::open(f, size, &limits, &cancel)?;
im.decode_region_rgba(0, r, &mut rgba, 4 * 1024, &cancel)?;
# Ok(())
# }
```

| Field | Type, unit | Default | Over the limit |
|---|---|---|---|
| `max_alloc` | `u64`, bytes | 4 GiB | `Error::AllocLimit` when a call needs more than the whole limit; `Error::MemoryLimit` (transient: retry, or answer HTTP 503) when other decodes hold the room |
| `max_pixels` | `u64`, pixels of level 0 (width times height) | 2^36 | `Error::PixelLimit`, from the header before the tile index is built |
| `max_header` | `u64`, bytes of XML header | 256 MiB | `Error::HeaderLimit` |
| `max_threads` | `usize`, threads, the calling thread included | `std::thread::available_parallelism()` | none: the call uses fewer threads and gives the same output |

- `max_alloc` is the enforced peak of the image, its views (`sdk_truncated`, `wide_grid`) and all
  calls on them together, every thread included. Memory is charged before it is allocated (the header
  parse, then the coefficient caches, the scratch of every decode, the staging buffer of every edge
  tile in flight and the row-piece list of a region) and released when dropped, so the peak never passes it. The destination buffer is the
  caller's and is not charged; a pool's thread stacks are the pool's. There are no nested crates here.
  `Image::set_memory_limit` changes the limit afterwards, as Go's `SetMemoryLimit`.
- A YCbCr region decode (`decode_region_ycbcr`) also charges a `4 * dx * dy` byte RGBA buffer for
  the whole region plus its chroma rows; it is `Error::AllocLimit` when that alone is over
  `max_alloc`, and `Error::MemoryLimit` (not transient) when it and a tile's scratch together are,
  so a large YCbCr region needs a limit of at least `4 * dx * dy` plus a few MiB.
- `max_alloc` is a `u64`, not a `usize`: `set_memory_limit` takes a `u64` (as Go's `SetMemoryLimit`
  takes an `int64`), `u64::MAX` means no limit, and a 32-bit target can then still express a limit
  over 4 GiB.
- A `Decoder` (`Image::with_decoder`, see [api.md](api.md)) keeps freed buffers between images. What it
  keeps is outside every image's `max_alloc` ledger: an image still charges each buffer it uses, where a
  new allocation would be charged and for as much, before it takes the buffer from the `Decoder`, so
  the limits, the errors, the output and the peak of the ledger are the same with and without one.
  The `Decoder` holds at most the `Limits::max_alloc` it was made with in free buffers and frees the
  rest. A worker's memory is its images' peaks plus what its `Decoder` keeps.
- A `Decoder` never evicts. Cache entries and staging buffers are found by exact length, so a
  buffer of a length that never comes back (for example the staging buffer of a region size the worker no longer asks for) stays
  held, up to `max_alloc`, and a new buffer that would pass that is freed instead. When an image
  evicts a cache entry under memory pressure (its own `max_alloc` is full), the entry moves to the
  `Decoder` instead of being freed, so the image's ledger drops but the process's memory does not.
  A worker should build the `Decoder` with a `max_alloc` it can afford to keep resident, and drop
  it and every image that uses it to free everything it holds.
- Cancellation: every call that reads takes an `&AtomicBool`, checked between codeblocks and tiles;
  set it from any thread and the call returns `Error::Cancelled`. A region's pixels of tiles not
  decoded are white.
- One `Error` enum, one variant per Go error ([parity.md](parity.md)). No decode panics on hostile input:
  sizes taken from the file are checked before they are allocated, and a worker thread's panic is
  returned as `Error::Internal`. Every public function's rustdoc lists the limits it enforces and its
  errors.

Threads and the worker pool: see [threading.md](threading.md).
