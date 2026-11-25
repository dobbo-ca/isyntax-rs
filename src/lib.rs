//! Philips iSyntax whole-slide image decoder: iSyntax2 (UFS interface 100 and later, compressors
//! 16 and 19, 8 or 9 bits stored) and v1 (UFS interface under 100, compressor 16, 8, 9 or 16 bits
//! stored). Rust port of dobbo-ca/isyntax-go; see the README.
//!
//! Every entry point takes [`Limits`] (through the image it opens) and a cancel flag.
//!
//! # Resource limits
//!
//! [`Image::open`] takes `&Limits`; the image and its views keep them. A constrained call, on the
//! calling thread only (no pool is touched) and at most 64 MiB for the image and every call on it:
//!
//! ```no_run
//! use isyntax::{Image, Limits, Rect};
//! use std::sync::atomic::AtomicBool;
//!
//! # fn main() -> Result<(), isyntax::Error> {
//! let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
//! let cancel = AtomicBool::new(false);
//! let f = std::fs::File::open("slide.isyntax")?;
//! let size = f.metadata()?.len();
//! let im = Image::open(f, size, &limits, &cancel)?;
//! let mut rgba = vec![0u8; 4 * 256 * 256];
//! im.decode_region_rgba(0, Rect::new(0, 0, 256, 256), &mut rgba, 4 * 256, &cancel)?;
//! # Ok(())
//! # }
//! ```
//!
//! - `max_alloc` (bytes, default 4 GiB) is the enforced peak of the image, its views and all
//!   their calls together, every thread included: the header parse, the coefficient caches, the
//!   scratch and the staging buffers are charged before they are allocated and released when
//!   dropped. A decode refused because other decodes hold the memory is [`Error::MemoryLimit`]
//!   (transient); one that needs more than the whole limit is [`Error::AllocLimit`].
//! - `max_pixels` (pixels of level 0, default 2^36) is [`Error::PixelLimit`]; `max_header`
//!   (bytes of XML header, default 256 MiB) is [`Error::HeaderLimit`].
//! - `max_threads` (threads, the caller's included, default
//!   [`std::thread::available_parallelism`]) caps the threads of one call. 1 spawns no thread
//!   and never touches a pool; 0 counts as 1. The output is the same for every value.
//! - A region's tiles run on a [`ThreadPool`]: the one given to [`Image::with_pool`] always wins;
//!   without one, the crate's lazy shared pool (built on the first call that has more than one
//!   tile and `max_threads` above 1, at most `available_parallelism` threads).
//! - Cancellation: every call that reads takes an `&AtomicBool`, polled between codeblocks and
//!   tiles; setting it from any thread makes the call return [`Error::Cancelled`].
#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

/// The README's examples, compiled as doc-tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// The example of docs/limits.md, compiled as a doc-test.
#[cfg(doctest)]
#[doc = include_str!("../docs/limits.md")]
struct LimitsDoctests;

mod budget;
mod cache;
mod container;
mod decoder;
#[cfg(test)]
mod decoder_tests;
mod dsp;
mod error;
#[cfg(test)]
mod fixture;
#[doc(hidden)]
pub mod fuzz_api;
mod hulsken;
mod limits;
mod parallel;
mod region;
mod reuse;
mod simd;
#[cfg(test)]
mod testutil;
mod trunc;
mod types;
mod xml;
mod xml_names;

pub use decoder::Image;
pub use error::Error;
pub use limits::Limits;
pub use parallel::ThreadPool;
pub use reuse::Decoder;
pub use types::{Level, Point, ReadAt, Rect, Subsample, YCbCr};
