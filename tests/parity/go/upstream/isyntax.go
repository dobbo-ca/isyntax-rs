// Package isyntax decodes Philips iSyntax whole-slide images in pure
// Go: iSyntax2 (UFS interface version 100 or later, compressors 16 and
// 19, 8 or 9 bits stored) and v1 (UFS interface version under 100,
// compressor 16, 8, 9, or 16 bits stored). Other compressors and bit
// depths fail with an error wrapping ErrUnsupported.
//
// Open an image, then decode tiles of its native 256x256 grid with
// DecodeTile or any rectangle with DecodeRegion, into an *image.RGBA or
// an *image.YCbCr. DecodeRegionSamples writes a rectangle's unhalved
// samples, 3 uint16 per pixel. An *Image is safe for concurrent use; its
// decodes share a cache of wavelet coefficients charged to the memory
// limit (see SetMemoryLimit).
//
// The decoding algorithm follows libisyntax
// (https://github.com/amspath/libisyntax), whose licence is:
//
//	BSD 2-Clause License
//
//	Copyright (c) 2019-2026, Pieter Valkema
//	Copyright (c) 2019-2025, Pieter Valkema, Alexandr Virodov (isyntax_reader.c)
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this
//	   list of conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	   this list of conditions and the following disclaimer in the documentation
//	   and/or other materials provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
package isyntax

import (
	"context"
	"io"

	isynt "github.com/dobbo-ca/isyntax-go/internal/isyntax"
)

// ErrUnsupported is wrapped by the error Open returns for files this
// package does not decode: unknown compressors, v1 compressor 19, bit
// depths other than 8, 9 (iSyntax2) and 16 (v1), unknown colour plane
// names, or codeblocks other than 128x128.
var ErrUnsupported = isynt.ErrUnsupported

// ErrMemoryLimit is wrapped by the error of a decode refused because
// other decodes hold the memory it needs. It is transient: retry, or
// answer HTTP 503.
var ErrMemoryLimit = isynt.ErrMemoryLimit

// SetMemoryLimit sets how many bytes all decodes in the process may hold
// together in coefficient caches and scratch, and returns the previous
// limit; n <= 0 means none. A decode that would pass the limit evicts
// its Image's cached coefficients first, then fails with an error
// wrapping ErrMemoryLimit rather than allocate; one that needs more than
// the whole limit fails with an error that does not wrap it.
//
// Until it is called, the limit is half of the Go runtime's memory limit
// (GOMEMLIMIT or runtime/debug.SetMemoryLimit) when one is set, else, on
// Linux, half of the cgroup memory limit, else none.
func SetMemoryLimit(n int64) int64 {
	return isynt.SetMemoryLimit(n)
}

// Level is one resolution level.
type Level = isynt.Level

// Image is an opened iSyntax file. It is safe for concurrent use. Its
// decode cache holds up to 1.125 MiB per level-0 tile column, at
// least 256 MiB and at most half the memory limit, and is released
// when the Image is garbage-collected.
type Image = isynt.Image

// Polygon is one UFS_IMAGE_VALID_DATA_ENVELOPES vertex list, in the
// order the file stores it.
type Polygon = isynt.Polygon

// Open reads the header of the iSyntax file in r[0:size].
func Open(r io.ReaderAt, size int64) (*Image, error) {
	return isynt.Open(r, size)
}

// OpenContext is Open with a context, checked between header reads.
func OpenContext(ctx context.Context, r io.ReaderAt, size int64) (*Image, error) {
	return isynt.OpenContext(ctx, r, size)
}
