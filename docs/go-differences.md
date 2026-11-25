# Differences from Go

- The memory limit is per image (and its views), not process-wide, and its default is
  `Limits::max_alloc` (4 GiB), not half the cgroup limit.
- A region with a coordinate beyond +-2^40 is refused (`InvalidArgument`), where Go returns the
  white a region off the image gets; no image the limits allow comes near it, and the bound keeps
  the size arithmetic from overflowing.
- As in Go, tiles inside the region decode straight into the destination (through disjoint row
  slices split up front, so without `unsafe`); only tiles cut by the region or level edge go
  through a staging buffer. The workers are a rayon pool (see [threading.md](threading.md)), not goroutines.
- The reader is a `ReadAt` (`io.ReaderAt`) that must return the bytes the header says; Go
  tolerated an `io.EOF` with a short read of a codeblock and decoded the stale scratch bytes.
- `encoding/xml` and `encoding/base64` are replaced by the readers in `src/xml.rs` (the subset of
  strict `encoding/xml` the header needs: no DTD, no namespaces, UTF-8) and `decode_base64`.
- Not ported: Go's libisyntax-compatibility mode (`legacyMode`, used by `TestConformance` only),
  `cmd/commentlint`, `cmd/tojson` and `bench/`; the tests that need slides
  (`ISYNTAX_SLIDES`, `ISYNTAX_ORACLE`: conformance, parity with the SDK, memory sweep, `TestMetadata`,
  `TestSourceViewInfo`); the pre-refactor reference implementations kept as test oracles in Go
  (`ref_test.go`, `regionRGBAOld`, `applyOld`); the Go allocation-count tests.
