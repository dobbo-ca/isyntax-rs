# Threading and dependencies

`Limits::max_threads` and the worker pool (the other limits are in [limits.md](limits.md)):

- `max_threads = 1` runs on the calling thread: no thread is spawned and no pool is touched. 0 counts
  as 1. Views inherit it from the image. A region decodes its tiles on up to `max_threads` threads
  and gives the same bytes for every value.
- The workers are an `isyntax::ThreadPool`. Pass your own (`Image::with_pool`, views inherit it): it
  always wins. Without one, a call with `max_threads` above 1 and more than one tile uses the crate's
  shared pool, a `OnceLock` built on the first such call, with `available_parallelism` threads at most
  and no other state. Each call takes at most `max_threads` workers from either pool, the caller
  included, and never more than the pool's threads plus the caller.
- A `Decoder` is a cheap handle (clones share one store behind a mutex, locked for a push or a pop per
  buffer), so workers can share one or hold one each. With `max_threads = 1` and a warm `Decoder` a
  call allocates nothing. With more threads a region still allocates its list of row pieces and the
  rayon jobs, and its staging list (see [api.md](api.md)).

## Dependencies

One third-party dependency, `rayon` (pinned `=1.12.0`, default features), with what it pulls in
(`rayon-core`, `crossbeam-deque`, `crossbeam-epoch`, `crossbeam-utils`, `either`; exact versions in
`Cargo.lock`). All are pure Rust: no C code, no `-sys` crate, no build script that compiles anything.
It is here only for `ThreadPool`: safe Rust cannot reuse a worker thread for jobs that borrow
short-lived buffers (`std::thread::scope` starts and joins its threads on every call, which made
small regions slower than Go), and rayon's scope can. It backs the caller's pool and the lazy shared
pool (see above); `src/parallel.rs` is the only module that uses it, and the CI image
fetches it in a layer before the source is copied. `libfuzzer-sys` lives only in the separate `tests/fuzz/`
crate. `src/xml.rs`, the base64 decoder in `src/container.rs` (libisyntax's tolerant alphabet, as Go's
`decodeBase64`) and `rgb_to_ycbcr` in `src/dsp.rs` are written here.
