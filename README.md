# isyntax-rs

Philips iSyntax whole-slide image decoder in Rust: iSyntax2 (UFS interface 100 and later, compressors
16 and 19, 8 or 9 bits stored) and v1 (UFS interface under 100, compressor 16, 8, 9 or 16 bits stored).
A port of [dobbo-ca/isyntax-go](https://github.com/dobbo-ca/isyntax-go).
The public tool that reads iSyntax is [libisyntax](https://github.com/amspath/libisyntax) (OpenSlide does not
support iSyntax); this crate follows its decoding algorithm.
Pure Rust, one dependency (`rayon`), SIMD kernels (AVX-512, AVX2, NEON); `unsafe` only in `src/simd/`.

```toml
isyntax = { git = "https://github.com/dobbo-ca/isyntax-rs", rev = "<sha>" }
```

## Usage

```rust,no_run
use isyntax::{Image, Limits, Rect, Subsample, YCbCr};
use std::sync::atomic::AtomicBool;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let limits = Limits { max_threads: 4, max_alloc: 1 << 30, ..Limits::default() };
let cancel = AtomicBool::new(false);

let f = std::fs::File::open("slide.isyntax")?;
let size = f.metadata()?.len();
let im = Image::open(f, size, &limits, &cancel)?; // reads the header only

// Level 2 is a quarter of full size; r is in its pixels.
let r = Rect::new(1024, 1024, 2048, 2048);
let mut rgba = vec![0u8; 4 * 1024 * 1024];
im.decode_region_rgba(2, r, &mut rgba, 4 * 1024, &cancel)?;

// Or one native 256x256 tile, into YCbCr 4:2:0 for a WebP encoder.
let mut tile = YCbCr::new(Rect::new(0, 0, 256, 256), Subsample::S420, &limits)?;
im.decode_tile_ycbcr(0, 40, 30, &mut tile, &cancel)?;
# Ok(())
# }
```

## Resource limits

`Image::open` takes `&Limits` and a cancel flag, as the quick start shows; `Limits { max_threads: 1,
max_alloc: 64 << 20, ..Limits::default() }` is a constrained call on the calling thread (example in
[limits.md](docs/limits.md)).

- `max_alloc` (default 4 GiB): enforced peak of the image and all calls on it; `Error::AllocLimit` or
  `Error::MemoryLimit` over it.
- `max_pixels` (2^36), `max_header` (256 MiB): checked from the header, before any large allocation.
- `max_threads` (default `available_parallelism`): `max_threads = 1` spawns no thread and touches no
  pool; a pool passed with `Image::with_pool` always wins over the shared one.

## Docs

[limits](docs/limits.md), [threading](docs/threading.md), [API](docs/api.md) (and reusing buffers with `Decoder`), [safety (unsafe, SIMD)](docs/safety.md),
[parity with Go](docs/parity.md), [fuzzing](docs/fuzzing.md), [benchmarks](docs/benchmarks.md), [performance](docs/performance.md),
[differences from Go](docs/go-differences.md), [status and tests](docs/status.md).
