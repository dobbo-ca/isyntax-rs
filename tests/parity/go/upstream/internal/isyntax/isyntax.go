package isyntax

import (
	"context"
	"image"
	"io"

	"github.com/dobbo-ca/isyntax-go/internal/container"
	"github.com/dobbo-ca/isyntax-go/internal/decoder"
)

// ErrUnsupported is wrapped by the error Open returns for files this
// package does not decode: unknown compressors, v1 compressor 19, bit
// depths other than 8, 9 (iSyntax2) and 16 (v1), unknown colour plane
// names, or codeblocks other than 128x128.
var ErrUnsupported error = container.ErrUnsupported

// ErrMemoryLimit is wrapped by the error of a decode refused because
// other decodes hold the memory it needs. It is transient: retry, or
// answer HTTP 503.
var ErrMemoryLimit error = decoder.ErrMemoryLimit

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
	return decoder.SetMemoryLimit(n)
}

// Level is one resolution level.
type Level struct {
	Width, Height  int // pixels
	TilesX, TilesY int // native 256x256 tile grid
	Downsample     int // 1 << level
}

// Image is an opened iSyntax file. It is safe for concurrent use. Its
// decode cache holds up to 1.125 MiB per level-0 tile column, at
// least 256 MiB and at most half the memory limit, and is released
// when the Image is garbage-collected.
type Image struct {
	m *decoder.Image
}

// Open reads the header of the iSyntax file in r[0:size].
func Open(r io.ReaderAt, size int64) (*Image, error) {
	return OpenContext(context.Background(), r, size)
}

// OpenContext is Open with a context, checked between header reads.
func OpenContext(ctx context.Context, r io.ReaderAt, size int64) (*Image, error) {
	m, err := decoder.Open(ctx, r, size)
	if err != nil {
		return nil, err
	}
	return &Image{m}, nil
}

// SDKTruncated returns a view of im whose DecodeTile and DecodeRegion
// reproduce the Philips SDK SourceView with default parameters, which
// truncate detail coefficients of Q1 and Q2 files (docs/format.md,
// "SDK default truncation"). im itself is unchanged. The view shares
// im's file, so the file must stay open while either is used. It has
// its own cache, sized as Open sizes one at the time of the call, and
// both caches count against the same memory limit. The error wraps
// ErrUnsupported when the file's quality preset and bit depth have no
// measured SDK table.
func (im *Image) SDKTruncated() (*Image, error) {
	m, err := im.m.SDKTruncated()
	if err != nil {
		return nil, err
	}
	return &Image{m}, nil
}

// WideGrid returns a view of im whose native tile grid is one top-scale
// tile wider and taller, on the right and bottom. The added tiles decode
// as tiles missing from the file (docs/format.md): zero detail, the
// parent's LL, zero LL at the top scale, where im's DecodeTile returns
// white. The Philips SDK display filters read these pixels past the
// grid. Every other pixel decodes as in im, the view's Level reports the
// wider TilesX and TilesY, and im is unchanged. The view shares im's
// file and truncation and has its own cache, an eighth of im's size.
func (im *Image) WideGrid() *Image { return &Image{im.m.WideGrid()} }

// Levels is the number of resolution levels, native levels plus one:
// the SDK's derived level, the top scale's own LL, at half its size.
func (im *Image) Levels() int { return im.m.Levels() }

// Level describes level i, in [0, Levels()). Level Levels()-1 is the
// derived level: 128x128 tiles at the top native scale's grid.
func (im *Image) Level(i int) Level { return Level(im.m.Level(i)) }

// TileSize is the side of a native tile, 256 at every level below
// Levels()-1, the derived level, whose tiles are 128.
func (im *Image) TileSize() int { return im.m.TileSize() }

// MPP is the size of a level-0 pixel in micrometres, 0 when the file
// does not say.
func (im *Image) MPP() (x, y float64) { return im.m.MPP() }

// Label is the label image's JPEG bytes, nil if absent.
func (im *Image) Label() []byte { return im.m.Label() }

// Macro is the macro (overview) image's JPEG bytes, nil if absent.
func (im *Image) Macro() []byte { return im.m.Macro() }

// ICCProfile is the WSI image's ICC profile, nil if absent. Decoded
// pixels are not colour-managed.
func (im *Image) ICCProfile() []byte { return im.m.ICC() }

// Barcode is the scanned barcode, empty if none. Sensitive: callers
// must not log or persist it outside access-controlled storage.
func (im *Image) Barcode() string { return im.m.Barcode() }

// ModelName is the scanner model, empty if none.
func (im *Image) ModelName() string { return im.m.ModelName() }

// DeviceSerial is the scanner's serial number, empty if none.
func (im *Image) DeviceSerial() string { return im.m.DeviceSerial() }

// AcquisitionDateTime is the DICOM acquisition timestamp, empty if none.
func (im *Image) AcquisitionDateTime() string { return im.m.AcquisitionDateTime() }

// CalibrationStatus is the scanner calibration status, empty if none.
func (im *Image) CalibrationStatus() string { return im.m.CalibrationStatus() }

// DateOfLastCalibration is the IStringArray tokens, nil if absent.
func (im *Image) DateOfLastCalibration() []string { return im.m.DateOfLastCalibration() }

// TimeOfLastCalibration is the IStringArray tokens, nil if absent.
func (im *Image) TimeOfLastCalibration() []string { return im.m.TimeOfLastCalibration() }

// RackSlot is the scanner rack and slot numbers, -1 when absent.
func (im *Image) RackSlot() (rack, slot int) { return im.m.RackSlot() }

// InterfaceVersion is the raw header value, e.g. "5.0" or "100.5".
func (im *Image) InterfaceVersion() string { return im.m.InterfaceVersion() }

// SourceBounds is level's rectangle in level pixels: the SDK
// SourceView's dimensionRanges, background border excluded, start
// forced so Min is the level's first valid column and row (see
// docs/format.md for the end convention).
func (im *Image) SourceBounds(level int) image.Rectangle { return im.m.SourceBounds(level) }

// Polygon is one UFS_IMAGE_VALID_DATA_ENVELOPES vertex list, in the
// order the file stores it.
type Polygon []image.Point

// Envelopes is level's UFS_IMAGE_VALID_DATA_ENVELOPES polygons in
// the file's vertex order, moved to image coordinates (the level-0
// padding subtracted) and scaled to level pixels (each vertex divided
// by the level's downsample; docs/format.md). A v1 file without the
// tag gets the whole image, or rectangles over its present scale-0
// tiles when some are missing. Pixels of
// SourceBounds outside every envelope are background.
func (im *Image) Envelopes(level int) []Polygon {
	raw := im.m.Envelopes(level)
	out := make([]Polygon, len(raw))
	for i, p := range raw {
		out[i] = Polygon(p)
	}
	return out
}

// BitsStored is 8, 9 or 16: iSyntax2 files are 8 or 9, v1 files may
// be 8, 9 or 16. 9-bit samples are halved to 8 bits in RGBA and YCbCr
// output; an *image.RGBA64 keeps them unhalved.
func (im *Image) BitsStored() int { return im.m.BitsStored() }

// DecodeRegion decodes the rectangle r of level into dst, an *image.RGBA,
// an *image.RGBA64 or an *image.YCbCr with 4:4:4 or 4:2:0 subsampling,
// whose bounds must be r's size: pixel r.Min lands on dst.Bounds().Min.
// level may be Levels()-1, the derived top-scale LL.
//
// r is in pixels of the level, with (0, 0) at the image's top-left:
// level pixel (x, y) is pixel (x+o, y+o) of the level's native tile
// grid, where o = (((3 << n) - 2) >> level) - 1, n the native level
// count (Levels()-1), matches Philips SDK SourceView (docs/format.md;
// libisyntax_read_region's o = ((3 << n) - 3) >> level differs by one
// from level 2). Pixels outside the level's Width and Height are white;
// a tile missing from the file decodes instead of a blank fill
// (docs/format.md).
//
// RGBA64 channels are the decoder's unshifted sample, clamped to
// [0, 2^b-1] with b = 9 for 9-bit files, else 8, then bit-replicated to
// 16 bits (b=8: v<<8|v; b=9: v<<7|v>>2). Alpha is 0xFFFF; white is
// 0xFFFF. Every channel >> 8 equals the RGBA byte; for 9-bit files the
// 9-bit value is channel>>7. Its per-tile scratch is pooled and
// charged to the memory limit.
//
// YCbCr is full-range BT.601, as color.RGBToYCbCr; 4:2:0 chroma is the
// rounded mean of each 2x2 block's pixels inside dst's bounds.
//
// Tiles are decoded in parallel on up to GOMAXPROCS goroutines.
// Cancelling ctx stops the decode between codeblocks and tiles.
func (im *Image) DecodeRegion(ctx context.Context, level int, r image.Rectangle, dst image.Image) error {
	return im.m.DecodeRegion(ctx, level, r, dst)
}

// DecodeRegionSamples decodes level rect r as DecodeRegion does into an
// *image.RGBA64, but writes 3 uint16 per pixel (R, G, B), each the
// sample clamped to [0, 2^b-1] (b = 9 for 9-bit files, else 8) and not
// bit-replicated; pixels DecodeRegion paints white get 2^b-1. Row y of
// r starts at dst[y*stride:]; a stride under 3*r.Dx() or a short dst is
// an error. For a sample s, DecodeRegion's RGBA64 channel is s<<7|s>>2 for
// 9-bit files and s<<8|s otherwise. Level range, memory charging and
// parallelism are DecodeRegion's.
func (im *Image) DecodeRegionSamples(ctx context.Context, level int, r image.Rectangle, dst []uint16, stride int) error {
	return im.m.DecodeRegionSamples(ctx, level, r, dst, stride)
}

// DecodeRegionSignedSamples is DecodeRegionSamples without the clamp:
// each sample is the colour transform's signed int16 output, which can
// be negative or above 2^b-1. Pixels DecodeRegion paints white still get
// 2^b-1.
func (im *Image) DecodeRegionSignedSamples(ctx context.Context, level int, r image.Rectangle, dst []int16, stride int) error {
	return im.m.DecodeRegionSignedSamples(ctx, level, r, dst, stride)
}

// DecodeTile decodes tile (tx, ty) of level's native grid into dst, an
// *image.RGBA, *image.RGBA64 (see DecodeRegion) or *image.YCbCr (4:4:4
// or 4:2:0), 256x256 pixels, or 128x128 at the derived level Levels()-1.
// Tiles outside the grid are white; one missing from the file decodes
// instead of a blank fill (docs/format.md).
func (im *Image) DecodeTile(ctx context.Context, level, tx, ty int, dst image.Image) error {
	return im.m.DecodeTile(ctx, level, tx, ty, dst)
}
