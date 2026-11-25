// Package decoder turns codeblocks into pixels: tile dependencies (LL
// from the parent's inverse transform, recursively to the top scale, and
// 4-coefficient margins from the 8 neighbours), the coefficient cache,
// the memory budget and region assembly. See docs/format.md.
package decoder

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"runtime"
	"sync"

	"github.com/dobbo-ca/isyntax-go/internal/container"
	"github.com/dobbo-ca/isyntax-go/internal/dsp"
	"github.com/dobbo-ca/isyntax-go/internal/hulsken"
	"github.com/dobbo-ca/isyntax-go/internal/parallel"
)

const (
	bs    = container.BlockSize // codeblock side
	ts    = 2 * bs              // tile side
	area  = bs * bs
	q     = bs + 8 // quadrant: block plus two margins
	w     = 2 * q  // inverse-transform buffer side
	first = 7      // first output pixel in the buffer
	plane = ts * ts

	coefLen  = 9 * area // 3 colours x HL, LH, HH
	reconLen = 3 * plane
	edgeLen  = 16 * bs // a block's four 4-wide strips

	// fillY: 9-bit background differs, see format.md.
	fillY = 255
)

// poolShare: 64ths of total, by kind.
var poolShare = [4]int64{kindCoef: 23, kindRecon: 26, kindCoefEdge: 8, kindReconEdge: 7}

// minCacheTotal: floor for defaultCacheTotal.
const minCacheTotal = 256 << 20

// defaultCacheTotal: gridW-sized.
func defaultCacheTotal(gridW int) int64 {
	n := int64(gridW) * (9 << 17) // 1.125 MiB per column
	if n < minCacheTotal {
		n = minCacheTotal
	}
	if half := currentLimit() / 2; n > half {
		n = half
	}
	return n
}

// Level is one resolution level.
type Level struct {
	Width, Height  int // pixels
	TilesX, TilesY int // native tile grid
	Downsample     int
}

// legacyMode: libisyntax compatibility mode (libisyntax's rules), for TestConformance.
var legacyMode bool

// zeroLL: missing top-scale tile's LL.
var zeroLL [3 * area]int16

// Image is an opened iSyntax file.
type Image struct {
	r       io.ReaderAt
	h       *container.Header
	tiles   map[container.TileKey]*container.Tile
	version int       // Hulsken 1 or 2
	shift   uint      // 1 for 9-bit files
	pools   [4]*cache // one per entry kind

	// RGB files skip YCoCg.
	toRGBA    func(dst []byte, a, b, c []int16, shift uint, yAbs bool)
	toRGB16   func(dst, a, b, c []int16, yAbs bool)
	toSamples func(dst, a, b, c []int16, hi int16, yAbs bool)

	tt  *truncTable // SDK default truncation; nil is none
	out uint8       // recon variant of output tiles
}

// Open reads the header of the iSyntax file r[0:size].
func Open(ctx context.Context, r io.ReaderAt, size int64) (*Image, error) {
	h, err := container.Parse(ctx, r, size)
	if err != nil {
		return nil, err
	}
	tiles, err := h.Index(size)
	if err != nil {
		return nil, err
	}
	m := &Image{r: r, h: h, tiles: tiles, version: 1}
	m.occupancyEnvelopes()
	m.newPools(defaultCacheTotal(h.GridW))
	if h.Compressor == 19 {
		m.version = 2
	}
	if h.BitsStored == 9 {
		m.shift = 1
	}
	m.toRGBA, m.toRGB16, m.toSamples = dsp.YCoCgToRGBA, dsp.YCoCgToRGB16, dsp.YCoCgToSamples
	if h.RGB {
		m.toRGBA, m.toRGB16, m.toSamples = dsp.PlanesToRGBA, dsp.PlanesToRGB16, dsp.PlanesToSamples
	}
	return m, nil
}

func (m *Image) newPools(total int64) {
	for k := range m.pools {
		m.pools[k] = newCache(total * poolShare[k] / 64)
		runtime.AddCleanup(m, (*cache).drop, m.pools[k])
	}
}

// reserve: evicts own cache.
func (m *Image) reserve(n int64) error {
	for {
		err := reserve(n)
		if !errors.Is(err, ErrMemoryLimit) {
			return err
		}
		evicted := false
		for _, c := range m.pools {
			evicted = evicted || c.evictOldest(nil)
		}
		if !evicted {
			return err
		}
	}
}

// Levels is the number of native levels plus one: SDK numDerivedLevels
// exposes the top-scale LL as a further, half-size derived level.
func (m *Image) Levels() int     { return m.h.Levels + 1 }
func (m *Image) TileSize() int   { return ts }
func (m *Image) BitsStored() int { return m.h.BitsStored }
func (m *Image) Label() []byte   { return m.h.Label }
func (m *Image) Macro() []byte   { return m.h.Macro }
func (m *Image) ICC() []byte     { return m.h.ICC }

// MPP is the level-0 pixel size in micrometres, 0 if unknown.
func (m *Image) MPP() (x, y float64) { return m.h.MPPX, m.h.MPPY }

// Barcode is the scanned barcode, empty if none. Sensitive: callers
// must not log or persist it outside access-controlled storage.
func (m *Image) Barcode() string { return m.h.Barcode }

// ModelName is the scanner model, empty if none.
func (m *Image) ModelName() string { return m.h.ModelName }

// DeviceSerial is the scanner's serial number, empty if none.
func (m *Image) DeviceSerial() string { return m.h.DeviceSerial }

// AcquisitionDateTime is the DICOM acquisition timestamp, empty if none.
func (m *Image) AcquisitionDateTime() string { return m.h.AcquisitionDateTime }

// CalibrationStatus is the scanner calibration status, empty if none.
func (m *Image) CalibrationStatus() string { return m.h.CalibrationStatus }

// DateOfLastCalibration is the IStringArray tokens, nil if absent.
func (m *Image) DateOfLastCalibration() []string { return m.h.CalibrationDate }

// TimeOfLastCalibration is the IStringArray tokens, nil if absent.
func (m *Image) TimeOfLastCalibration() []string { return m.h.CalibrationTime }

// RackSlot is the scanner rack and slot numbers, -1 when absent.
func (m *Image) RackSlot() (rack, slot int) { return m.h.Rack, m.h.Slot }

// InterfaceVersion is the raw PIM_DP_UFS_INTERFACE_VERSION, empty if none.
func (m *Image) InterfaceVersion() string { return m.h.InterfaceVersion }

// SourceBounds is level's rectangle in level pixels: the SDK SourceView
// dimensionRanges, background border excluded (docs/format.md).
func (m *Image) SourceBounds(level int) image.Rectangle {
	r, _ := pyramidShrink(image.Rect(0, 0, m.h.Width, m.h.Height), level)
	return r
}

// Envelopes is UFS_IMAGE_VALID_DATA_ENVELOPES, the file's polygons
// exactly as stored, scaled to level pixels: each level-0 vertex
// divided by the level's downsample (docs/format.md).
func (m *Image) Envelopes(level int) [][]image.Point {
	step := 1 << level
	out := make([][]image.Point, len(m.h.Envelopes))
	for i, poly := range m.h.Envelopes {
		p := make([]image.Point, len(poly))
		for j, v := range poly {
			p[j] = image.Pt(v.X/step, v.Y/step)
		}
		out[i] = p
	}
	return out
}

// occupancyEnvelopes replaces a v1 file's default envelope with
// rectangles over the present scale-0 tiles, a row's adjacent tiles
// merged, when any scale-0 tile over the image is missing or sits
// under a missing top-scale tile: the SDK leaves such tiles out of
// the data envelope (docs/format.md "SourceBounds and Envelopes").
func (m *Image) occupancyEnvelopes() {
	h := m.h
	if h.Major >= 100 || !h.EnvelopeDefault {
		return
	}
	top := h.Levels - 1
	off := (3<<h.Levels - 2) - 1 // regionTiles' rule 1 at level 0
	bounds := image.Rect(0, 0, h.Width, h.Height)
	var env [][]image.Point
	missing := false
	for y := 0; y < h.GridH; y++ {
		var cur image.Rectangle
		flush := func() {
			if !cur.Empty() {
				env = append(env, []image.Point{cur.Min, {X: cur.Max.X, Y: cur.Min.Y}, cur.Max, {X: cur.Min.X, Y: cur.Max.Y}})
			}
			cur = image.Rectangle{}
		}
		for x := 0; x < h.GridW; x++ {
			r := image.Rect(x*ts-off, y*ts-off, (x+1)*ts-off, (y+1)*ts-off).Intersect(bounds)
			if r.Empty() {
				continue
			}
			// Absent subtrees may list blocks.
			if m.tile(0, x, y) == nil || m.tile(top, x>>top, y>>top) == nil {
				missing = true
				flush()
				continue
			}
			if cur.Empty() {
				cur = r
			} else {
				cur.Max.X = r.Max.X
			}
		}
		flush()
	}
	if missing {
		h.Envelopes = env
	}
}

// pyramidShrink: r shrunk per-level, wavelet borders.
func pyramidShrink(r image.Rectangle, level int) (image.Rectangle, bool) {
	minX, maxX := r.Min.X, r.Max.X-1
	minY, maxY := r.Min.Y, r.Max.Y-1
	for range level {
		minX, maxX = (minX+1)/2+1, maxX/2-1
		minY, maxY = (minY+1)/2+1, maxY/2-1
		if minX > maxX || minY > maxY {
			return image.Rectangle{}, false
		}
	}
	return image.Rect(minX, minY, maxX+1, maxY+1), true
}

// Level describes level i, which must be in [0, Levels()). Level
// Levels()-1 is the derived level: the top scale's LL codeblocks, one
// 128x128 tile per top-scale tile, at the same grid.
func (m *Image) Level(i int) Level {
	if i == m.h.Levels {
		top := m.Level(i - 1)
		return Level{Width: top.Width / 2, Height: top.Height / 2, TilesX: top.TilesX, TilesY: top.TilesY, Downsample: 2 * top.Downsample}
	}
	return Level{
		Width: m.h.Width >> i, Height: m.h.Height >> i,
		TilesX: m.h.GridW >> i, TilesY: m.h.GridH >> i,
		Downsample: 1 << i,
	}
}

// tile returns (s, x, y)'s tile.
func (m *Image) tile(s, x, y int) *container.Tile {
	if x < 0 || y < 0 || x >= m.h.GridW>>s || y >= m.h.GridH>>s {
		return nil
	}
	t := m.tiles[container.TileKey{Scale: int32(s), X: int32(x), Y: int32(y)}]
	if t == nil || !t.Exists {
		return nil
	}
	return t
}

// inGrid: (x, y) in scale s?
func (m *Image) inGrid(s, x, y int) bool {
	return x >= 0 && y >= 0 && x < m.h.GridW>>s && y < m.h.GridH>>s
}

// scratch: pooled tile-decode working memory.
type scratch struct {
	idwt [3][w * w]int16 // Y, Co, Cg
	tmp  [w * w]int16
	row  dsp.RowScratch
	hul  hulsken.Scratch
	read []byte
}

// scratchBytes: charged while a scratch used.
const scratchBytes = 4*w*w*2 + 4*area*3 + 1<<18

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

// getScratch evicts m's caches, avoids self-block.
func getScratch(m *Image) (*scratch, error) {
	if err := m.reserve(scratchBytes); err != nil {
		return nil, err
	}
	return scratchPool.Get().(*scratch), nil
}

func putScratch(s *scratch) {
	scratchPool.Put(s)
	release(scratchBytes)
}

// coef: per-colour HL, LH, HH, LL.
func (m *Image) coef(ctx context.Context, s, x, y int) ([]int16, pin, error) {
	n := coefLen
	if s == m.h.Levels-1 {
		n += 3 * area
	}
	return m.pools[kindCoef].get(ctx, ckey{kindCoef, 0, int32(s), int32(x), int32(y)}, n, func(data []int16) error {
		t := m.tile(s, x, y)
		sc, err := getScratch(m)
		if err != nil {
			return err
		}
		defer putScratch(sc)
		for c := 0; c < 3; c++ {
			if err := m.codeblock(ctx, sc, t.H[c], data[c*3*area:(c+1)*3*area], 3); err != nil {
				return err
			}
			if len(data) > coefLen {
				if err := m.codeblock(ctx, sc, t.LL[c], data[coefLen+c*area:coefLen+(c+1)*area], 1); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

func (m *Image) codeblock(ctx context.Context, sc *scratch, b container.Block, dst []int16, n int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if cap(sc.read) < int(b.Size) {
		sc.read = make([]byte, b.Size)
	}
	buf := sc.read[:b.Size]
	if _, err := m.r.ReadAt(buf, b.Offset); err != nil && (!errors.Is(err, io.EOF) || b.Size <= 0) {
		return fmt.Errorf("isyntax: reading a codeblock: %w", err)
	}
	if err := hulsken.Decode(dst, buf, n, bs, bs, m.version, &sc.hul); err != nil {
		return fmt.Errorf("isyntax: codeblock at %d: %w", b.Offset, err)
	}
	return nil
}

// recon: inverse transform, variant v's truncation.
func (m *Image) recon(ctx context.Context, s, x, y int, v uint8) ([]int16, pin, error) {
	return m.pools[kindRecon].get(ctx, ckey{kindRecon, v, int32(s), int32(x), int32(y)}, reconLen, func(data []int16) error {
		d, err := m.gather(ctx, s, x, y, v)
		if err != nil {
			return err
		}
		defer d.release()
		sc, err := getScratch(m)
		if err != nil {
			return err
		}
		defer putScratch(sc)
		m.transform(&d, sc, v)
		for c := 0; c < 3; c++ {
			for r := 0; r < ts; r++ {
				copy(data[c*plane+r*ts:][:ts], sc.idwt[c][(first+r)*w+first:][:ts])
			}
		}
		return nil
	})
}

// deps: full centre, neighbours' edges.
type deps struct {
	h   [9][]int16
	ll  [9]llSrc
	rel [18]pin // pins to release after use
}

// release unpins every dependency gather fetched.
func (d *deps) release() {
	for _, r := range d.rel {
		r.release()
	}
}

// llSrc: tile LL; nil means fill.
type llSrc struct {
	data                  []int16
	planeLen, off, stride int
}

// putEdge: the margins neighbours read.
func putEdge(e, src []int16, stride int) {
	for k := 0; k < 4; k++ {
		copy(e[k*bs:][:bs], src[k*stride:])
		copy(e[(4+k)*bs:][:bs], src[(bs-4+k)*stride:])
	}
	l, r := e[8*bs:12*bs], e[12*bs:16*bs]
	for k := 0; k < bs; k++ {
		row := src[k*stride:][:bs]
		copy(l[4*k:4*k+4], row[:4])
		copy(r[4*k:4*k+4], row[bs-4:])
	}
}

// coefEdge: coef's margins; v unused.
func (m *Image) coefEdge(ctx context.Context, s, x, y int, _ uint8) ([]int16, pin, error) {
	nb := 9
	if s == m.h.Levels-1 {
		nb = 12
	}
	return m.pools[kindCoefEdge].get(ctx, ckey{kindCoefEdge, 0, int32(s), int32(x), int32(y)}, nb*edgeLen, func(e []int16) error {
		cf, rel, err := m.coef(ctx, s, x, y)
		if err != nil {
			return err
		}
		defer rel.release()
		for b := 0; b < nb; b++ {
			putEdge(e[b*edgeLen:], cf[b*area:], bs)
		}
		return nil
	})
}

// reconEdge: recon's quadrant margins, colour-major.
func (m *Image) reconEdge(ctx context.Context, s, x, y int, v uint8) ([]int16, pin, error) {
	return m.pools[kindReconEdge].get(ctx, ckey{kindReconEdge, v, int32(s), int32(x), int32(y)}, 12*edgeLen, func(e []int16) error {
		rec, rel, err := m.recon(ctx, s, x, y, v)
		if err != nil {
			return err
		}
		defer rel.release()
		for b := 0; b < 12; b++ {
			q := b % 4
			putEdge(e[b*edgeLen:], rec[b/4*plane+q/2*bs*ts+q%2*bs:], ts)
		}
		return nil
	})
}

func (m *Image) gather(ctx context.Context, s, x, y int, v uint8) (deps, error) {
	var d deps
	top := m.h.Levels - 1
	pv := parentVariant(v)
	for i := 0; i < 9; i++ {
		nx, ny := x+i%3-1, y+i/3-1
		if !m.inGrid(s, nx, ny) {
			continue // edge: zero margin, or Y fill
		}
		real := m.tile(s, nx, ny) != nil
		if legacyMode && !real {
			continue
		}
		if real {
			var cf []int16
			var rel pin
			var err error
			if i == 4 {
				cf, rel, err = m.coef(ctx, s, nx, ny)
			} else {
				cf, rel, err = m.coefEdge(ctx, s, nx, ny, 0)
			}
			if err == nil && i == 4 {
				d.rel[i] = rel
				err = m.retire(ctx, rel, 1, m.coefEdge, s, nx, ny, 0)
			}
			if err != nil {
				d.release()
				return d, err
			}
			d.h[i], d.rel[i] = cf, rel
		}
		// Missing tile: zero detail, parent's LL.
		if s == top {
			if real && i == 4 {
				d.ll[i] = llSrc{d.h[i][coefLen:], area, 0, bs}
			} else if real {
				d.ll[i] = llSrc{d.h[i], edgeLen, 9 * edgeLen, 0}
			} else if !legacyMode {
				d.ll[i] = llSrc{zeroLL[:], 0, 0, bs} // rule 3
			}
			continue
		}
		if legacyMode {
			if m.tile(s+1, nx/2, ny/2) == nil {
				continue
			}
		} else if !m.inGrid(s+1, nx/2, ny/2) {
			continue
		}
		if i == 4 {
			rec, rel, err := m.recon(ctx, s+1, nx/2, ny/2, pv)
			if err == nil {
				d.rel[9+i] = rel
				px, py := nx/2*2, ny/2*2
				n := int32(min(2, m.h.GridW>>s-px) * min(2, m.h.GridH>>s-py))
				err = m.retire(ctx, rel, n, m.reconEdge, s+1, nx/2, ny/2, pv)
			}
			if err != nil {
				d.release()
				return d, err
			}
			d.ll[i], d.rel[9+i] = llSrc{rec, plane, ny%2*bs*ts + nx%2*bs, ts}, rel
			continue
		}
		e, rel, err := m.reconEdge(ctx, s+1, nx/2, ny/2, pv)
		if err != nil {
			d.release()
			return d, err
		}
		d.ll[i], d.rel[9+i] = llSrc{e, 4 * edgeLen, (ny%2*2 + nx%2) * edgeLen, 0}, rel
	}
	return d, nil
}

// retire: edge kept, full evicted first.
func (m *Image) retire(ctx context.Context, p pin, n int32, edge func(context.Context, int, int, int, uint8) ([]int16, pin, error), s, x, y int, v uint8) error {
	if !p.used(n) {
		return nil
	}
	_, e, err := edge(ctx, s, x, y, v)
	if err != nil {
		return err
	}
	e.release()
	p.demote()
	return nil
}

// Neighbour span [lo,hi) lands at [at,...).
var span = [3]struct{ lo, hi, at int }{{bs - 4, bs, 0}, {0, bs, 4}, {0, 4, bs + 4}}

// view: span start and stride.
func view(blk []int16, stride, i int) ([]int16, int) {
	switch {
	case i == 4:
		return blk, stride
	case i < 3: // above: its bottom rows
		return blk[4*bs+span[i%3].lo:], bs
	case i > 5: // below: its top rows
		return blk[span[i%3].lo:], bs
	case i == 3: // left: its right columns
		return blk[12*bs:], 4
	}
	return blk[8*bs:], 4
}

// transform: stitch, truncate per v, invert.
func (m *Image) transform(d *deps, sc *scratch, v uint8) {
	blit := func(dst []int16, qr, qc int, blk []int16, stride, i int) {
		src, stride := view(blk, stride, i)
		r, c := span[i/3], span[i%3]
		for k := 0; k < r.hi-r.lo; k++ {
			copy(dst[(qr+r.at+k)*w+qc+c.at:][:c.hi-c.lo], src[k*stride:])
		}
	}
	// Spans tile quadrants; fill only gaps.
	fill := func(dst []int16, qr, qc, i int, v int16) {
		r, c := span[i/3], span[i%3]
		for k := 0; k < r.hi-r.lo; k++ {
			row := dst[(qr+r.at+k)*w+qc+c.at:][:c.hi-c.lo]
			if v == 0 {
				clear(row) // memclr; the loop is scalar
				continue
			}
			for j := range row {
				row[j] = v
			}
		}
	}
	for c := 0; c < 3; c++ {
		buf := sc.idwt[c][:]
		for i := 0; i < 9; i++ {
			if ll := d.ll[i]; ll.data != nil {
				blit(buf, 0, 0, ll.data[c*ll.planeLen+ll.off:], ll.stride, i)
			} else if c == 0 {
				fill(buf, 0, 0, i, fillY)
			} else {
				fill(buf, 0, 0, i, 0)
			}
			if d.h[i] == nil {
				fill(buf, 0, q, i, 0) // virtual tile: zero detail
				fill(buf, q, 0, i, 0)
				fill(buf, q, q, i, 0)
				continue
			}
			n := edgeLen
			if i == 4 {
				n = area
			}
			h := d.h[i][c*3*n:]
			blit(buf, 0, q, h[:n], bs, i)      // HL
			blit(buf, q, 0, h[n:2*n], bs, i)   // LH
			blit(buf, q, q, h[2*n:3*n], bs, i) // HH
		}
		if v != 0 {
			m.tt.apply(buf, int(v)-1, c)
		}
		dsp.Inverse53(buf, sc.tmp[:], q, &sc.row)
	}
}

// tileRGBA: white if missing, else decoded.
func (m *Image) tileRGBA(ctx context.Context, s, x, y int, px []byte, stride int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.tile(s, x, y) == nil && (legacyMode || !m.inGrid(s, x, y)) {
		for r := 0; r < ts; r++ {
			fill255(px[r*stride:][:4*ts])
		}
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
			m.toRGBA(px[r*stride:][:4*ts], rec[r*ts:][:ts], rec[plane+r*ts:][:ts], rec[2*plane+r*ts:][:ts], m.shift, legacyMode)
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
		m.toRGBA(px[r*stride:][:4*ts], sc.idwt[0][o:][:ts], sc.idwt[1][o:][:ts], sc.idwt[2][o:][:ts], m.shift, legacyMode)
	}
	return nil
}

// topLLTileRGBA: derived tile from top LL.
func (m *Image) topLLTileRGBA(ctx context.Context, x, y int, px []byte, stride int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	top := m.h.Levels - 1
	real := m.tile(top, x, y) != nil
	if !real && (legacyMode || !m.inGrid(top, x, y)) {
		for r := 0; r < bs; r++ {
			fill255(px[r*stride:][:4*bs])
		}
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
		m.toRGBA(px[r*stride:][:4*bs], ll[o:o+bs], ll[area+o:area+o+bs], ll[2*area+o:2*area+o+bs], m.shift, legacyMode)
	}
	return nil
}

// rgbaPool: tile-sized buffers for cut tiles.
var rgbaPool = sync.Pool{New: func() any { return new([4 * plane]byte) }}

const rgbaBytes = 4 * plane

// DecodeTile decodes tile (x, y) of level's native grid into dst, an
// *image.RGBA, *image.RGBA64 or *image.YCbCr (4:4:4 or 4:2:0) of 256x256
// pixels, or 128x128 at the derived level Levels()-1 (the top scale's LL).
func (m *Image) DecodeTile(ctx context.Context, level, x, y int, dst image.Image) error {
	side := ts
	if level == m.h.Levels {
		side = bs
	}
	if err := m.check(level, dst, image.Pt(side, side)); err != nil {
		return err
	}
	if d, ok := dst.(*image.RGBA64); ok {
		b := d.Bounds()
		return m.tileRGBA64(ctx, level, x, y, d.Pix[d.PixOffset(b.Min.X, b.Min.Y):], d.Stride)
	}
	decode := func(px []byte, stride int) error { return m.tileRGBA(ctx, level, x, y, px, stride) }
	if level == m.h.Levels {
		decode = func(px []byte, stride int) error { return m.topLLTileRGBA(ctx, x, y, px, stride) }
	}
	switch d := dst.(type) {
	case *image.RGBA:
		b := d.Bounds()
		return decode(d.Pix[d.PixOffset(b.Min.X, b.Min.Y):], d.Stride)
	default:
		if err := m.reserve(rgbaBytes); err != nil {
			return err
		}
		defer release(rgbaBytes)
		px := rgbaPool.Get().(*[4 * plane]byte)
		defer rgbaPool.Put(px)
		if err := decode(px[:], 4*side); err != nil {
			return err
		}
		return m.toYCbCr(dst.(*image.YCbCr), px[:4*side*side], 4*side)
	}
}

func (m *Image) check(level int, dst image.Image, size image.Point) error {
	if level < 0 || level >= m.Levels() {
		return fmt.Errorf("isyntax: level %d of %d", level, m.Levels())
	}
	switch d := dst.(type) {
	case *image.RGBA, *image.RGBA64:
	case *image.YCbCr:
		if d.SubsampleRatio != image.YCbCrSubsampleRatio444 && d.SubsampleRatio != image.YCbCrSubsampleRatio420 {
			return fmt.Errorf("isyntax: YCbCr subsampling %v, want 4:4:4 or 4:2:0", d.SubsampleRatio)
		}
	default:
		return fmt.Errorf("isyntax: destination %T, want *image.RGBA, *image.RGBA64 or *image.YCbCr", dst)
	}
	if dst.Bounds().Size() != size {
		return fmt.Errorf("isyntax: destination is %v, want %v", dst.Bounds().Size(), size)
	}
	return nil
}

// DecodeRegion decodes r, in pixels of level with the origin at the
// image's top-left, into dst, whose bounds must be r's size. level may
// be Levels()-1, the derived top-scale LL.
func (m *Image) DecodeRegion(ctx context.Context, level int, r image.Rectangle, dst image.Image) error {
	if err := m.check(level, dst, r.Size()); err != nil {
		return err
	}
	if r.Empty() {
		return nil
	}
	if d, ok := dst.(*image.RGBA); ok {
		b := d.Bounds()
		return m.regionRGBA(ctx, level, r, d.Pix[d.PixOffset(b.Min.X, b.Min.Y):], d.Stride)
	}
	if d, ok := dst.(*image.RGBA64); ok {
		b := d.Bounds()
		return m.regionRGBA64(ctx, level, r, d.Pix[d.PixOffset(b.Min.X, b.Min.Y):], d.Stride)
	}
	n := 4 * int64(r.Dx()) * int64(r.Dy())
	buf, err := getRegionBuf(m, n)
	if err != nil {
		return err
	}
	defer putRegionBuf(buf, n)
	px := (*buf)[:n]
	if err := m.regionRGBA(ctx, level, r, px, 4*r.Dx()); err != nil {
		return err
	}
	return m.toYCbCr(dst.(*image.YCbCr), px, 4*r.Dx())
}

// regionPool: RGBA buffers for non-RGBA output.
var regionPool = sync.Pool{New: func() any { return new([]byte) }}

func getRegionBuf(m *Image, n int64) (*[]byte, error) {
	if err := m.reserve(n); err != nil {
		return nil, err
	}
	p := regionPool.Get().(*[]byte)
	if int64(cap(*p)) < n {
		*p = make([]byte, n)
	}
	return p, nil
}

func putRegionBuf(p *[]byte, n int64) {
	regionPool.Put(p)
	release(n)
}

func (m *Image) regionRGBA(ctx context.Context, level int, r image.Rectangle, px []byte, stride int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// white: a in level pixels.
	white := func(a image.Rectangle) {
		a = a.Sub(r.Min)
		for y := a.Min.Y; y < a.Max.Y; y++ {
			fill255(px[y*stride+4*a.Min.X:][:4*a.Dx()])
		}
	}
	lv := m.Level(level)
	vis := r.Intersect(image.Rect(0, 0, lv.Width, lv.Height))
	around(r, vis, white)
	if vis.Empty() {
		return nil
	}
	side := ts
	tile := func(x, y int, dst []byte, stride int) error { return m.tileRGBA(ctx, level, x, y, dst, stride) }
	if level == m.h.Levels {
		side = bs
		tile = func(x, y int, dst []byte, stride int) error { return m.topLLTileRGBA(ctx, x, y, dst, stride) }
	}
	// SDK SourceView origin, differs elsewhere.
	off := (3<<m.h.Levels-2)>>level - 1
	g := vis.Add(image.Pt(off, off))
	tx0, ty0 := g.Min.X/side, g.Min.Y/side
	nx, ny := (g.Max.X-1)/side-tx0+1, (g.Max.Y-1)/side-ty0+1
	return tilesFor(nx*ny, func(i int) error {
		tx, ty := tx0+i%nx, ty0+i/nx
		tr := image.Rect(tx*side, ty*side, (tx+1)*side, (ty+1)*side)
		part := tr.Intersect(g)
		// Destination of part's top-left pixel.
		at := px[(part.Min.Y-off-r.Min.Y)*stride+(part.Min.X-off-r.Min.X)*4:]
		if part == tr {
			return tile(tx, ty, at, stride)
		}
		if err := m.reserve(rgbaBytes); err != nil {
			return err
		}
		defer release(rgbaBytes)
		buf := rgbaPool.Get().(*[4 * plane]byte)
		defer rgbaPool.Put(buf)
		if err := tile(tx, ty, buf[:], 4*side); err != nil {
			return err
		}
		sx, sy := part.Min.X-tr.Min.X, part.Min.Y-tr.Min.Y
		for k := 0; k < part.Dy(); k++ {
			copy(at[k*stride:][:4*part.Dx()], buf[(sy+k)*4*side+4*sx:])
		}
		return nil
	}, func(i int) {
		tx, ty := tx0+i%nx, ty0+i/nx
		white(image.Rect(tx*side, ty*side, (tx+1)*side, (ty+1)*side).Intersect(g).Sub(image.Pt(off, off)))
	})
}

// around: f on r minus in.
func around(r, in image.Rectangle, f func(image.Rectangle)) {
	if in.Empty() {
		f(r)
		return
	}
	for _, a := range [4]image.Rectangle{
		image.Rect(r.Min.X, r.Min.Y, r.Max.X, in.Min.Y),
		image.Rect(r.Min.X, in.Max.Y, r.Max.X, r.Max.Y),
		image.Rect(r.Min.X, in.Min.Y, in.Min.X, in.Max.Y),
		image.Rect(in.Max.X, in.Min.Y, r.Max.X, in.Max.Y),
	} {
		if !a.Empty() {
			f(a)
		}
	}
}

// tilesFor: on error, miss unfinished tiles.
func tilesFor(n int, fn func(i int) error, miss func(i int)) error {
	done := make([]bool, n)
	err := parallel.For(n, func(i int) error {
		err := fn(i)
		done[i] = err == nil
		return err
	})
	if err != nil {
		for i, ok := range done {
			if !ok {
				miss(i)
			}
		}
	}
	return err
}

// ySc: toYCbCr's pooled chroma scratch.
type ySc struct {
	cb, cr          []byte
	sumB, sumR, cnt []int32
}

var yPool = sync.Pool{New: func() any { return new(ySc) }}

// toYCbCr: RGBA to 4:2:0, chroma averaged.
func (m *Image) toYCbCr(dst *image.YCbCr, src []byte, stride int) error {
	b := dst.Bounds()
	wd := b.Dx()
	// Chroma index follows COffset's truncating x/2.
	cw := 0
	if dst.SubsampleRatio == image.YCbCrSubsampleRatio420 {
		cw = (b.Max.X-1)/2 - b.Min.X/2 + 1
	}
	n := int64(2*wd + 12*cw)
	if err := m.reserve(n); err != nil {
		return err
	}
	defer release(n)
	s := yPool.Get().(*ySc)
	defer yPool.Put(s)
	if cap(s.cb) < wd {
		s.cb, s.cr = make([]byte, wd), make([]byte, wd)
	}
	cb, cr := s.cb[:wd], s.cr[:wd]
	if dst.SubsampleRatio == image.YCbCrSubsampleRatio444 {
		for y := b.Min.Y; y < b.Max.Y; y++ {
			o := dst.YOffset(b.Min.X, y)
			c := dst.COffset(b.Min.X, y)
			dsp.RGBAToYCbCr(dst.Y[o:o+wd], dst.Cb[c:c+wd], dst.Cr[c:c+wd], src[(y-b.Min.Y)*stride:])
		}
		return nil
	}
	if cap(s.sumB) < cw {
		s.sumB, s.sumR, s.cnt = make([]int32, cw), make([]int32, cw), make([]int32, cw)
	}
	sumB, sumR, cnt := s.sumB[:cw], s.sumR[:cw], s.cnt[:cw]
	for y := b.Min.Y; y < b.Max.Y; y++ {
		o := dst.YOffset(b.Min.X, y)
		dsp.RGBAToYCbCr(dst.Y[o:o+wd], cb, cr, src[(y-b.Min.Y)*stride:])
		for i := 0; i < wd; i++ {
			k := (b.Min.X+i)/2 - b.Min.X/2
			sumB[k] += int32(cb[i])
			sumR[k] += int32(cr[i])
			cnt[k]++
		}
		if y == b.Max.Y-1 || (y+1)/2 != y/2 {
			c := dst.COffset(b.Min.X, y)
			for k := range cnt {
				dst.Cb[c+k] = byte((sumB[k] + cnt[k]/2) / cnt[k])
				dst.Cr[c+k] = byte((sumR[k] + cnt[k]/2) / cnt[k])
				sumB[k], sumR[k], cnt[k] = 0, 0, 0
			}
		}
	}
	return nil
}
