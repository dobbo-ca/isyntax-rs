package decoder

import (
	"fmt"

	"github.com/dobbo-ca/isyntax-go/internal/container"
	"github.com/dobbo-ca/isyntax-go/internal/dsp"
)

// truncTable: Philips SDK default truncation, docs/format.md.
type truncTable struct {
	k     [2][3]uint8 // [scale-level][colour] bits dropped
	round bool        // adds 2^(k-1)-1 when nonzero
}

// Measured presets only: the Philips SDK SourceView default truncation.
var (
	truncQ2b8  = truncTable{[2][3]uint8{{1, 4, 4}, {0, 2, 2}}, true}
	truncQ2b9  = truncTable{[2][3]uint8{{3, 3, 3}, {2, 3, 3}}, true}
	truncQ1RGB = truncTable{[2][3]uint8{{0, 4, 4}}, false}
)

// parentVariant: recon 1 (output) reads 2.
func parentVariant(v uint8) uint8 {
	if v == 1 {
		return 2
	}
	return 0
}

// sdkPreset: nil means no truncation.
func sdkPreset(h *container.Header) (*truncTable, error) {
	q := h.Quality
	if q < 0 && h.Major >= 100 && h.BitsStored == 8 {
		q = 2 // the SDK reports Q2
	}
	switch {
	case q == 0:
		return nil, nil
	case q == 2 && h.Major >= 100 && !h.RGB && h.BitsStored == 8:
		return &truncQ2b8, nil
	case q == 2 && h.Major >= 100 && !h.RGB && h.BitsStored == 9:
		return &truncQ2b9, nil
	case q == 1 && h.Major < 100 && h.RGB && h.BitsStored == 8 && h.Levels == 1:
		return &truncQ1RGB, nil
	}
	return nil, fmt.Errorf("%w: no measured SDK truncation for Quality=%d, %d bits stored, interface %s", container.ErrUnsupported, h.Quality, h.BitsStored, h.InterfaceVersion)
}

// SDKTruncated is a view of m that decodes like the SDK SourceView
// with default parameters. It shares m's file and header and has its
// own caches, sized as Open sizes them now.
func (m *Image) SDKTruncated() (*Image, error) {
	tt, err := sdkPreset(m.h)
	if err != nil {
		return nil, err
	}
	v := &Image{r: m.r, h: m.h, tiles: m.tiles, version: m.version, shift: m.shift, toRGBA: m.toRGBA, toRGB16: m.toRGB16, toSamples: m.toSamples, tt: tt}
	if tt != nil {
		v.out = 1
	}
	v.newPools(defaultCacheTotal(m.h.GridW))
	return v, nil
}

// WideGrid is a view of m whose grid is one top-scale tile wider and
// taller, the added tiles missing. It shares m's file, header fields
// and truncation. Its caches total an eighth of m's: it decodes only
// the tiles past m's grid and their neighbours.
func (m *Image) WideGrid() *Image {
	h := *m.h
	h.GridW += 1 << (h.Levels - 1)
	h.GridH += 1 << (h.Levels - 1)
	v := &Image{r: m.r, h: &h, tiles: m.tiles, version: m.version, shift: m.shift, toRGBA: m.toRGBA, toRGB16: m.toRGB16, toSamples: m.toSamples, tt: m.tt, out: m.out}
	v.newPools(defaultCacheTotal(m.h.GridW) / 8)
	return v
}

// apply: colour c's detail quadrants only.
func (t *truncTable) apply(buf []int16, j, c int) {
	k := t.k[j][c]
	if k == 0 {
		return
	}
	mask := int16(1)<<k - 1
	var o int16
	if t.round {
		o = int16(1)<<(k-1) - 1
	}
	for r := 0; r < w; r++ {
		row := buf[r*w : (r+1)*w]
		if r < q {
			row = row[q:] // LL stays
		}
		dsp.TruncRow(row, mask, o)
	}
}
