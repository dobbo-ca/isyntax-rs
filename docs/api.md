# API

`level(i)` gives each level's size and tile grid, `mpp`, `label`, `macro_image` and `icc_profile`
the pixel size and the associated JPEGs and profile. `decode_region_rgba64` writes 16-bit RGBA;
`decode_region_samples` writes a region as clamped R, G, B samples, 3 `u16` per pixel, unhalved for
9-bit files, and `decode_region_signed_samples` the same as `i16` without the clamp. An `Image` is
`Sync`: decodes share a cache of wavelet coefficients charged to the memory limit.
`sdk_truncated` gives a view that decodes like the Philips SDK SourceView's defaults, which
truncate detail coefficients of Q1 and Q2 files (`Error::Unsupported` for presets not measured);
`wide_grid` a view one top-scale tile wider and taller, for the SDK display filters.

## Reusing buffers: Decoder

A worker that decodes tile after tile from one image after another can keep its working buffers.
Make one `Decoder::new(&limits)`, give it to each image with `Image::with_decoder(&decoder)` (views
inherit it) and keep it as long as the worker lives. `Decoder` is a cheap `Clone` handle that is `Send`
and `Sync`.

```rust,no_run
# use isyntax::{Decoder, Image, Limits, Rect};
# use std::sync::atomic::AtomicBool;
# fn main() -> Result<(), isyntax::Error> {
# let cancel = AtomicBool::new(false);
# let mut rgba = vec![0u8; 4 * 256 * 256];
let limits = Limits { max_threads: 1, ..Limits::default() };
let decoder = Decoder::new(&limits);
for path in ["a.isyntax", "b.isyntax"] {
    let f = std::fs::File::open(path)?;
    let size = f.metadata()?.len();
    let im = Image::open(f, size, &limits, &cancel)?.with_decoder(&decoder);
    im.decode_region_rgba(0, Rect::new(0, 0, 256, 256), &mut rgba, 4 * 256, &cancel)?;
}
# Ok(())
# }
```

- It keeps what an image frees: the coefficient cache entries (about 56 MB for a 3000x2000 level 0),
  the tile scratch, the staging and conversion buffers of the region and YCbCr calls, and the cleared
  tables of the caches. An image that has one takes these from it instead of allocating, and gives
  them back when it drops or evicts them. The coefficient cache is still cold for each image; only the
  memory is warm.
- The limits do not change. Every buffer is charged to the image's `max_alloc` where a new one would
  be, then taken from the `Decoder`. What the `Decoder` keeps between images is not charged to any
  image ([limits.md](limits.md)); it keeps at most its own `max_alloc` and frees the rest.
- With `max_threads` 1, once a `Decoder` has seen an image of the same size, the decode calls on the
  next image allocate nothing: `tests/reuse_alloc.rs` counts the allocator calls of 8-bit (compressors
  19 and 16), 9-bit and v1 16-bit files, for a multi-tile RGBA region, all the tiles as RGBA, RGBA64 and
  YCbCr, and the sample regions.
- What still allocates: `Image::open`; an error (its message); a codeblock over 256 KiB (read into a
  buffer of its own); a cache entry that the cache evicts while a decode still holds it (freed, not
  kept); buffers of a size the `Decoder` has not seen (found by exact length); and calls with
  `max_threads` above 1 (the list of row pieces and the staging list of a region, the jobs of the
  pool).

