package decoder

import (
	"context"
	"fmt"
	"image"
	"sync"
	"unsafe"
)

// Parity path: signed, unshifted samples.

// rgb16Func: toRGB16, or clamped for DecodeRegionSamples.
type rgb16Func = func(dst, a, b, c []int16, yAbs bool)

// tileRGB16: tileRGBA before the shift.
func (m *Image) tileRGB16(ctx context.Context, s, x, y int, dst []int16, stride int, conv rgb16Func) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.tile(s, x, y) == nil && (legacyMode || !m.inGrid(s, x, y)) {
		clear(dst)
		return nil
	}
	if s > 0 {
		rec, rel, err := m.recon(ctx, s, x, y, m.out)
		if err != nil {
			return err
		}
		defer rel.release()
		rel.demote() // output, not a centre
		for r := 0; r < ts; r++ {
			conv(dst[r*stride:][:3*ts], rec[r*ts:][:ts], rec[plane+r*ts:][:ts], rec[2*plane+r*ts:][:ts], legacyMode)
		}
		return nil
	}
	d, err := m.gather(ctx, s, x, y, m.out)
	if err != nil {
		return err
	}
	defer d.release()
	sc, err := getScratch(m)
	if err != nil {
		return err
	}
	defer putScratch(sc)
	m.transform(&d, sc, m.out)
	for r := 0; r < ts; r++ {
		o := (first+r)*w + first
		conv(dst[r*stride:][:3*ts], sc.idwt[0][o:][:ts], sc.idwt[1][o:][:ts], sc.idwt[2][o:][:ts], legacyMode)
	}
	return nil
}

// topLLTileRGB16: topLLTileRGBA before the shift.
func (m *Image) topLLTileRGB16(ctx context.Context, x, y int, dst []int16, stride int, conv rgb16Func) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	top := m.h.Levels - 1
	real := m.tile(top, x, y) != nil
	if !real && (legacyMode || !m.inGrid(top, x, y)) {
		clear(dst)
		return nil
	}
	var ll [3 * area]int16
	if real {
		cf, rel, err := m.coef(ctx, top, x, y)
		if err != nil {
			return err
		}
		copy(ll[:], cf[coefLen:])
		rel.release()
	}
	for r := 0; r < bs; r++ {
		o := r * bs
		conv(dst[r*stride:][:3*bs], ll[o:o+bs], ll[area+o:area+o+bs], ll[2*area+o:2*area+o+bs], legacyMode)
	}
	return nil
}

// rgb16Pool: int16 tile staging buffers.
var rgb16Pool = sync.Pool{New: func() any { return new([3 * plane]int16) }}

// regionTiles: per-tile puts; absent is white.
func (m *Image) regionTiles(ctx context.Context, level int, r image.Rectangle, conv rgb16Func, put func(buf []int16, side int, tr, part image.Rectangle, off int, absent bool)) error {
	side, lvl := ts, level
	tile := func(x, y int, dst []int16, stride int) error { return m.tileRGB16(ctx, level, x, y, dst, stride, conv) }
	if level == m.h.Levels {
		side, lvl = bs, m.h.Levels-1
		tile = func(x, y int, dst []int16, stride int) error { return m.topLLTileRGB16(ctx, x, y, dst, stride, conv) }
	}
	lv := m.Level(level)
	vis := r.Intersect(image.Rect(0, 0, lv.Width, lv.Height))
	off := (3<<m.h.Levels-2)>>level - 1 // rule 1
	o := image.Pt(off, off)
	around(r, vis, func(a image.Rectangle) { put(nil, side, image.Rectangle{}, a.Add(o), off, true) })
	if vis.Empty() {
		return nil
	}
	g := vis.Add(o)
	tx0, ty0 := g.Min.X/side, g.Min.Y/side
	nx, ny := (g.Max.X-1)/side-tx0+1, (g.Max.Y-1)/side-ty0+1
	return tilesFor(nx*ny, func(i int) error {
		tx, ty := tx0+i%nx, ty0+i/nx
		tr := image.Rect(tx*side, ty*side, (tx+1)*side, (ty+1)*side)
		if err := ctx.Err(); err != nil {
			return err
		}
		if m.tile(lvl, tx, ty) == nil && (legacyMode || !m.inGrid(lvl, tx, ty)) {
			put(nil, side, tr, tr.Intersect(g), off, true)
			return nil
		}
		n := int64(6 * plane) // pooled buffer size
		if err := m.reserve(n); err != nil {
			return err
		}
		defer release(n)
		p := rgb16Pool.Get().(*[3 * plane]int16)
		defer rgb16Pool.Put(p)
		buf := p[:3*side*side]
		if err := tile(tx, ty, buf, 3*side); err != nil {
			return err
		}
		put(buf, side, tr, tr.Intersect(g), off, false)
		return nil
	}, func(i int) {
		tx, ty := tx0+i%nx, ty0+i/nx
		tr := image.Rect(tx*side, ty*side, (tx+1)*side, (ty+1)*side)
		put(nil, side, tr, tr.Intersect(g), off, true)
	})
}

// regionRGB16: unshifted, any level.
func (m *Image) regionRGB16(ctx context.Context, level int, r image.Rectangle) ([]int16, error) {
	dst := make([]int16, 3*r.Dx()*r.Dy())
	stride := 3 * r.Dx()
	err := m.regionTiles(ctx, level, r, m.toRGB16, func(buf []int16, side int, tr, part image.Rectangle, off int, absent bool) {
		if absent {
			return
		}
		sx, sy := part.Min.X-tr.Min.X, part.Min.Y-tr.Min.Y
		at := dst[(part.Min.Y-off-r.Min.Y)*stride+(part.Min.X-off-r.Min.X)*3:]
		for k := 0; k < part.Dy(); k++ {
			copy(at[k*stride:][:3*part.Dx()], buf[(sy+k)*3*side+3*sx:])
		}
	})
	return dst, err
}

// pack64: clamp, replicate to 16 bits.
func pack64(dst []byte, src []int16, bits uint) {
	hi := int16(1)<<bits - 1
	for i := 0; i < len(src)/3; i++ {
		d := dst[8*i : 8*i+8 : 8*i+8]
		for c := 0; c < 3; c++ {
			v := uint16(min(max(src[3*i+c], 0), hi))
			if bits == 9 {
				v = v<<7 | v>>2
			} else {
				v = v<<8 | v
			}
			d[2*c], d[2*c+1] = byte(v>>8), byte(v)
		}
		d[6], d[7] = 0xff, 0xff
	}
}

// fill255: doubling copy, memmove speed.
func fill255(b []byte) {
	b[0] = 0xff
	for n := 1; n < len(b); n *= 2 {
		copy(b[n:], b[:n])
	}
}

// tileRGBA64: white if absent.
func (m *Image) tileRGBA64(ctx context.Context, level, x, y int, px []byte, stride int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	side, lvl := ts, level
	if level == m.h.Levels {
		side, lvl = bs, m.h.Levels-1
	}
	if m.tile(lvl, x, y) == nil && (legacyMode || !m.inGrid(lvl, x, y)) {
		for r := 0; r < side; r++ {
			fill255(px[r*stride:][:8*side])
		}
		return nil
	}
	n := int64(6 * plane) // pooled buffer size
	if err := m.reserve(n); err != nil {
		return err
	}
	defer release(n)
	p := rgb16Pool.Get().(*[3 * plane]int16)
	defer rgb16Pool.Put(p)
	buf := p[:3*side*side]
	var err error
	if level == m.h.Levels {
		err = m.topLLTileRGB16(ctx, x, y, buf, 3*side, m.toRGB16)
	} else {
		err = m.tileRGB16(ctx, level, x, y, buf, 3*side, m.toRGB16)
	}
	if err != nil {
		return err
	}
	for r := 0; r < side; r++ {
		pack64(px[r*stride:][:8*side], buf[3*side*r:][:3*side], 8+m.shift)
	}
	return nil
}

// regionRGBA64: allocates, budget-charged.
func (m *Image) regionRGBA64(ctx context.Context, level int, r image.Rectangle, px []byte, stride int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.regionTiles(ctx, level, r, m.toRGB16, func(buf []int16, side int, tr, part image.Rectangle, off int, absent bool) {
		at := px[(part.Min.Y-off-r.Min.Y)*stride+(part.Min.X-off-r.Min.X)*8:]
		if absent {
			for k := 0; k < part.Dy(); k++ {
				fill255(at[k*stride:][:8*part.Dx()])
			}
			return
		}
		sx, sy := part.Min.X-tr.Min.X, part.Min.Y-tr.Min.Y
		for k := 0; k < part.Dy(); k++ {
			pack64(at[k*stride:][:8*part.Dx()], buf[(sy+k)*3*side+3*sx:][:3*part.Dx()], 8+m.shift)
		}
	})
}

// DecodeRegionSamples decodes r as DecodeRegion does into an
// *image.RGBA64, but writes 3 uint16 per pixel (R, G, B): the sample
// clamped to [0, 2^b-1], b = 9 for 9-bit files else 8, not
// bit-replicated; white is 2^b-1. Row y of r starts at dst[y*stride:].
func (m *Image) DecodeRegionSamples(ctx context.Context, level int, r image.Rectangle, dst []uint16, stride int) error {
	var px []int16
	if len(dst) > 0 {
		// Same bits: every value is in [0, hi].
		px = unsafe.Slice((*int16)(unsafe.Pointer(&dst[0])), len(dst))
	}
	return m.regionSamples(ctx, level, r, px, stride, true)
}

// DecodeRegionSignedSamples is DecodeRegionSamples without the clamp:
// the colour transform's int16 output as is; white is still 2^b-1.
func (m *Image) DecodeRegionSignedSamples(ctx context.Context, level int, r image.Rectangle, dst []int16, stride int) error {
	return m.regionSamples(ctx, level, r, dst, stride, false)
}

func (m *Image) regionSamples(ctx context.Context, level int, r image.Rectangle, px []int16, stride int, clamped bool) error {
	if level < 0 || level >= m.Levels() {
		return fmt.Errorf("isyntax: level %d of %d", level, m.Levels())
	}
	if r.Empty() {
		return nil
	}
	if stride < 3*r.Dx() || len(px) < (r.Dy()-1)*stride+3*r.Dx() {
		return fmt.Errorf("isyntax: %d samples at stride %d, too small for %v", len(px), stride, r.Size())
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	hi := int16(1)<<(8+m.shift) - 1
	conv := m.toRGB16
	if clamped {
		conv = func(d, a, b, c []int16, yAbs bool) { m.toSamples(d, a, b, c, hi, yAbs) }
	}
	return m.regionTiles(ctx, level, r, conv, func(buf []int16, side int, tr, part image.Rectangle, off int, absent bool) {
		at := px[(part.Min.Y-off-r.Min.Y)*stride+(part.Min.X-off-r.Min.X)*3:]
		if absent {
			for k := 0; k < part.Dy(); k++ {
				row := at[k*stride:][:3*part.Dx()]
				row[0] = hi
				for n := 1; n < len(row); n *= 2 {
					copy(row[n:], row[:n])
				}
			}
			return
		}
		sx, sy := part.Min.X-tr.Min.X, part.Min.Y-tr.Min.Y
		for k := 0; k < part.Dy(); k++ {
			copy(at[k*stride:][:3*part.Dx()], buf[(sy+k)*3*side+3*sx:])
		}
	})
}
