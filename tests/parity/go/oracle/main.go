// Command oracle is the Go side of the parity check (see docs/parity.md).
//
//	oracle -gen <inputs>          write the test inputs
//	oracle -genbench <fixtures>   write the benchmark fixtures
//	oracle -genseeds <seeds>      write the fuzz seeds, a directory per target of tests/fuzz/
//	oracle <inputs> <outputs>     open and decode every input
//	oracle -bench N <fixtures>    time the decode of every fixture
//	oracle -ppm <in> <out>        write every tile of every level of every file as a P6 (quality gate)
//
// Decode mode writes one file per input, <name>.out: a line per operation, "<op> ok <digest>" or
// "<op> err <class>", and "<op> partial <digest>" after the error line of a tile decode (the
// destination as the decoder left it). examples/oracle.rs does the same with the Rust crate.
package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"syscall"
	"time"

	isyntax "github.com/dobbo-ca/isyntax-go"
	"github.com/dobbo-ca/isyntax-go/internal/container"
	"github.com/dobbo-ca/isyntax-go/internal/hulsken"
)

func main() {
	gen := flag.Bool("gen", false, "write the inputs: -gen <inputs>")
	genBench := flag.Bool("genbench", false, "write the benchmark fixtures: -genbench <fixtures>")
	genSeeds := flag.Bool("genseeds", false, "write the fuzz seeds: -genseeds <seeds>")
	bench := flag.Int("bench", 0, "decode every fixture N times: -bench N <fixtures>")
	ppm := flag.Bool("ppm", false, "write every tile as a P6: -ppm <in> <out>")
	flag.Parse()
	a := flag.Args()
	var err error
	switch {
	case *gen && len(a) == 1:
		err = generate(a[0])
	case *genBench && len(a) == 1:
		err = generateBench(a[0])
	case *genSeeds && len(a) == 1:
		err = generateSeeds(a[0])
	case *bench > 0 && len(a) == 1:
		err = runBench(*bench, a[0])
	case *ppm && len(a) == 2:
		err = ppmDir(a[0], a[1])
	case !*gen && !*genBench && !*genSeeds && *bench == 0 && !*ppm && len(a) == 2:
		err = decodeAll(a[0], a[1])
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "oracle:", err)
		os.Exit(1)
	}
}

// class is the short stable token of an error; the table is in docs/parity.md.
func class(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, isyntax.ErrUnsupported):
		return "unsupported"
	case errors.Is(err, isyntax.ErrMemoryLimit):
		return "memory-limit"
	}
	s := err.Error()
	// "isyntax: codeblock at 1234: hulsken: ..." is a codeblock that does not decode; the container
	// errors "codeblock at scale", "codeblock at (x, y)" and "codeblock of N bytes" are header errors.
	if rest, ok := strings.CutPrefix(s, "isyntax: codeblock at "); ok && rest != "" && rest[0] >= '0' && rest[0] <= '9' {
		return "codeblock"
	}
	for _, c := range []struct{ has, class string }{
		{"XML header over", "header-limit"},
		{"isyntax: XML header:", "xml"},
		{"decode needs more than the memory limit", "alloc-limit"},
		{"isyntax: reading", "io"},
		{"isyntax: internal error", "internal"},
		{"isyntax: level ", "invalid-argument"},
		{"isyntax: destination", "invalid-argument"},
		{"isyntax: YCbCr", "invalid-argument"},
		{"samples at stride", "invalid-argument"},
	} {
		if strings.Contains(s, c.has) {
			return c.class
		}
	}
	return "header"
}

// digest is FNV-1a 64 and the length of the parts together.
func digest(parts ...[]byte) string {
	h, n := fnv.New64a(), 0
	for _, p := range parts {
		h.Write(p)
		n += len(p)
	}
	return fmt.Sprintf("%016x %d", h.Sum64(), n)
}

func u16bytes[T ~uint16 | ~int16](v []T) []byte {
	b := make([]byte, 0, 2*len(v))
	for _, x := range v {
		b = append(b, byte(x), byte(uint16(x)>>8))
	}
	return b
}

// report collects the lines of one input.
type report struct {
	b strings.Builder
	// oneWorker: the regions run with one worker, so every region has a partial line.
	oneWorker bool
	// regionErr: a region decode failed.
	regionErr bool
}

func (r *report) line(f string, a ...any) { fmt.Fprintf(&r.b, f+"\n", a...) }

func (r *report) res(op string, err error, data ...[]byte) {
	if err != nil {
		r.line("%s err %s", op, class(err))
		return
	}
	r.line("%s ok %s", op, digest(data...))
}

// tileRes is res for a tile decode into a destination prefilled with 0xAA: the partial line shows
// what the decoder wrote before it failed.
func (r *report) tileRes(op string, err error, data ...[]byte) {
	r.res(op, err, data...)
	if err != nil {
		r.line("%s partial %s", op, digest(data...))
	}
}

// regionRes is res for a region decode. With one tile in the level, or one worker, which tiles
// were written before an error does not depend on scheduling and the destination is compared as a
// partial line too.
func (r *report) regionRes(single bool, op string, err error, data ...[]byte) {
	r.regionErr = r.regionErr || err != nil
	if single || r.oneWorker {
		r.tileRes(op, err, data...)
		return
	}
	r.res(op, err, data...)
}

func hx(s string) string { return hex.EncodeToString([]byte(s)) }

func hxs(v []string) string {
	p := make([]string, len(v))
	for i, s := range v {
		p[i] = hx(s)
	}
	return strings.Join(p, ",")
}

func rect(r image.Rectangle) string {
	return fmt.Sprintf("%d,%d,%d,%d", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
}

func info(r *report, im *isyntax.Image) {
	r.line("info levels=%d tile=%d bits=%d", im.Levels(), im.TileSize(), im.BitsStored())
	mx, my := im.MPP()
	r.line("info mpp=%016x,%016x", math.Float64bits(mx), math.Float64bits(my))
	r.line("info label=%s macro=%s icc=%s", digest(im.Label()), digest(im.Macro()), digest(im.ICCProfile()))
	r.line("info label-nil=%v macro-nil=%v icc-nil=%v", im.Label() == nil, im.Macro() == nil, im.ICCProfile() == nil)
	r.line("info barcode=%s model=%s serial=%s acq=%s cal=%s iface=%s", hx(im.Barcode()), hx(im.ModelName()),
		hx(im.DeviceSerial()), hx(im.AcquisitionDateTime()), hx(im.CalibrationStatus()), hx(im.InterfaceVersion()))
	rack, slot := im.RackSlot()
	r.line("info dates=%s times=%s rack=%d slot=%d", hxs(im.DateOfLastCalibration()), hxs(im.TimeOfLastCalibration()), rack, slot)
	for l := 0; l < im.Levels(); l++ {
		lv := im.Level(l)
		var env strings.Builder
		for _, poly := range im.Envelopes(l) {
			for _, p := range poly {
				fmt.Fprintf(&env, "%d,%d ", p.X, p.Y)
			}
			env.WriteString("| ")
		}
		r.line("info level %d: %dx%d tiles %dx%d ds %d bounds %s env %s", l, lv.Width, lv.Height,
			lv.TilesX, lv.TilesY, lv.Downsample, rect(im.SourceBounds(l)), env.String())
	}
}

const fill = 0xAA

func filled(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = fill
	}
	return b
}

func tiles(r *report, ctx context.Context, im *isyntax.Image, tag string) {
	for l := 0; l < im.Levels(); l++ {
		lv := im.Level(l)
		side := im.TileSize()
		if l == im.Levels()-1 {
			side = 128
		}
		for _, t := range [][2]int{{0, 0}, {lv.TilesX / 2, lv.TilesY / 2}, {lv.TilesX - 1, lv.TilesY - 1}, {lv.TilesX, lv.TilesY}, {-1, 0}} {
			op := fmt.Sprintf("%stile.%d.%d.%d", tag, l, t[0], t[1])
			d := image.NewRGBA(image.Rect(0, 0, side, side))
			copy(d.Pix, filled(len(d.Pix)))
			err := im.DecodeTile(ctx, l, t[0], t[1], d)
			r.tileRes(op, err, d.Pix)
		}
		t := [2]int{lv.TilesX / 2, lv.TilesY / 2}
		d := image.NewRGBA64(image.Rect(0, 0, side, side))
		copy(d.Pix, filled(len(d.Pix)))
		err := im.DecodeTile(ctx, l, t[0], t[1], d)
		r.tileRes(fmt.Sprintf("%stile64.%d", tag, l), err, d.Pix)
		for _, sub := range []struct {
			name string
			s    image.YCbCrSubsampleRatio
		}{{"420", image.YCbCrSubsampleRatio420}, {"444", image.YCbCrSubsampleRatio444}} {
			y := image.NewYCbCr(image.Rect(0, 0, side, side), sub.s)
			err := im.DecodeTile(ctx, l, t[0], t[1], y)
			r.tileRes(fmt.Sprintf("%stileycc%s.%d", tag, sub.name, l), err, y.Y, y.Cb, y.Cr)
		}
	}
}

// allTiles decodes every tile of every level as RGBA, a digest line each, so that no codeblock of a
// small image goes undecoded.
func allTiles(r *report, ctx context.Context, im *isyntax.Image) {
	if lv := im.Level(0); lv.TilesX*lv.TilesY > 64 {
		return
	}
	for l := 0; l < im.Levels(); l++ {
		lv := im.Level(l)
		side := im.TileSize()
		if l == im.Levels()-1 {
			side = 128
		}
		for y := 0; y < lv.TilesY; y++ {
			for x := 0; x < lv.TilesX; x++ {
				d := image.NewRGBA(image.Rect(0, 0, side, side))
				r.res(fmt.Sprintf("all.%d.%d.%d", l, x, y), im.DecodeTile(ctx, l, x, y, d), d.Pix)
			}
		}
	}
}

func regions(r *report, ctx context.Context, im *isyntax.Image, tag string, all bool) {
	for l := 0; l < im.Levels(); l++ {
		lv := im.Level(l)
		single := lv.TilesX == 1 && lv.TilesY == 1
		a := image.Rect(0, 0, min(lv.Width, 400), min(lv.Height, 300))
		b := image.Rect(lv.Width-100, lv.Height-90, lv.Width+60, lv.Height+50)
		for i, rc := range []image.Rectangle{a, b} {
			op := func(k string) string { return fmt.Sprintf("%sregion%s.%d.%d", tag, k, l, i) }
			d := image.NewRGBA(image.Rect(0, 0, rc.Dx(), rc.Dy()))
			r.regionRes(single, op("rgba"), im.DecodeRegion(ctx, l, rc, d), d.Pix)
			if i == 1 && !all {
				continue
			}
			if i == 0 || all {
				d64 := image.NewRGBA64(image.Rect(0, 0, rc.Dx(), rc.Dy()))
				r.regionRes(single, op("rgba64"), im.DecodeRegion(ctx, l, rc, d64), d64.Pix)
				u := make([]uint16, 3*rc.Dx()*rc.Dy())
				r.regionRes(single, op("samples"), im.DecodeRegionSamples(ctx, l, rc, u, 3*rc.Dx()), u16bytes(u))
				s := make([]int16, 3*rc.Dx()*rc.Dy())
				r.regionRes(single, op("signed"), im.DecodeRegionSignedSamples(ctx, l, rc, s, 3*rc.Dx()), u16bytes(s))
				y := image.NewYCbCr(image.Rect(0, 0, rc.Dx(), rc.Dy()), image.YCbCrSubsampleRatio420)
				r.regionRes(single, op("ycc420"), im.DecodeRegion(ctx, l, rc, y), y.Y, y.Cb, y.Cr)
			}
		}
		// TestSignedSamples' rectangles: one partly above and left of the level, one narrow and tall.
		if all {
			for i, rc := range []image.Rectangle{image.Rect(-20, -10, 10, 10), image.Rect(250, 250, 263, 520)} {
				op := func(k string) string { return fmt.Sprintf("%sregion%s.%d.%d", tag, k, l, i+2) }
				u := make([]uint16, 3*rc.Dx()*rc.Dy())
				r.regionRes(single, op("samples"), im.DecodeRegionSamples(ctx, l, rc, u, 3*rc.Dx()), u16bytes(u))
				s := make([]int16, 3*rc.Dx()*rc.Dy())
				r.regionRes(single, op("signed"), im.DecodeRegionSignedSamples(ctx, l, rc, s, 3*rc.Dx()), u16bytes(s))
			}
		}
	}
	// Bad arguments: a level out of range, a destination of the wrong size.
	d := image.NewRGBA(image.Rect(0, 0, 8, 8))
	r.res(tag+"bad.level", im.DecodeRegion(ctx, im.Levels(), image.Rect(0, 0, 8, 8), d))
	r.res(tag+"bad.size", im.DecodeRegion(ctx, 0, image.Rect(0, 0, 9, 8), d))
	r.res(tag+"bad.stride", im.DecodeRegionSamples(ctx, 0, image.Rect(0, 0, 8, 8), make([]uint16, 24), 24))
	// TestRGBA64BadBounds: a region into a destination one column short, a 128 x 128 tile.
	r.res(tag+"bad.size64", im.DecodeRegion(ctx, 0, image.Rect(0, 0, 10, 10), image.NewRGBA64(image.Rect(0, 0, 9, 10))))
	r.res(tag+"bad.tile64", im.DecodeTile(ctx, 0, 0, 0, image.NewRGBA64(image.Rect(0, 0, 128, 128))))
}

// blkScratch is shared by all `.blk` inputs, which are decoded in sorted order: blocks with
// different parameters reuse one scratch, as the fuzz target does.
var blkScratch hulsken.Scratch

// hulskenOps decodes a `.blk` input, [version-1, planes/3, size index][codeblock] (see gen.go), with
// hulsken.Decode: one line, the digest of the coefficients or an error (always class codeblock).
func hulskenOps(data []byte) string {
	var r report
	if len(data) < 3 {
		r.line("hulsken err header")
		return r.b.String()
	}
	version, n, bw := 1+int(data[0]&1), 1, [4]int{4, 8, 16, 128}[data[2]&3]
	if data[1]&1 != 0 {
		n = 3
	}
	dst := make([]int16, n*bw*bw)
	if err := hulsken.Decode(dst, data[3:], n, bw, bw, version, &blkScratch); err != nil {
		r.line("hulsken err codeblock")
	} else {
		r.line("hulsken ok %s", digest(u16bytes(dst)))
	}
	return r.b.String()
}

// ops runs every operation on one input.
func ops(name string, data []byte) string {
	switch {
	case strings.HasSuffix(name, ".blk"):
		return hulskenOps(data)
	case strings.HasSuffix(name, ".b64"):
		var r report
		b, err := container.OracleDecodeBase64(string(data))
		r.res("base64", err, b)
		return r.b.String()
	case strings.HasSuffix(name, ".ct"):
		var r report
		n, err := container.OracleClusterTable(data)
		if err != nil {
			r.res("ct", err)
		} else {
			r.line("ct ok %d", n)
		}
		return r.b.String()
	}
	var r report
	ctx := context.Background()
	im, err := isyntax.OpenContext(ctx, bytes.NewReader(data), int64(len(data)))
	if err != nil {
		r.res("open", err)
		return r.b.String()
	}
	r.line("open ok")
	info(&r, im)
	// The full set of tiles and regions on a first view, the tiles again after the regions on the
	// same Image (its cache is warm), then the SDK-truncated and the wide-grid views.
	tiles(&r, ctx, im, "")
	regions(&r, ctx, im, "", true)
	tiles(&r, ctx, im, "warm.")
	allTiles(&r, ctx, im)
	sdk, err := im.SDKTruncated()
	r.res("sdk", err)
	if err == nil {
		regions(&r, ctx, sdk, "sdk.", false)
	}
	wide := im.WideGrid()
	for l := 0; l < wide.Levels(); l++ {
		lv := wide.Level(l)
		r.line("wide level %d: %dx%d tiles %dx%d", l, lv.Width, lv.Height, lv.TilesX, lv.TilesY)
	}
	regions(&r, ctx, wide, "wide.", false)
	// Where a region failed: the regions again with one worker (parallel.For runs GOMAXPROCS) on a
	// fresh Image, so the tiles written before the error are the same on every run.
	if r.regionErr {
		if im1, err := isyntax.OpenContext(ctx, bytes.NewReader(data), int64(len(data))); err == nil {
			prev := runtime.GOMAXPROCS(1)
			r.oneWorker = true
			regions(&r, ctx, im1, "w1.", true)
			sdk1, err := im1.SDKTruncated()
			if err == nil {
				regions(&r, ctx, sdk1, "w1.sdk.", false)
			}
			regions(&r, ctx, im1.WideGrid(), "w1.wide.", false)
			r.oneWorker = false
			runtime.GOMAXPROCS(prev)
		}
	}
	// A cancelled context stops the decode (on a fresh Image: a cached tile needs no reads).
	if im2, err := isyntax.OpenContext(ctx, bytes.NewReader(data), int64(len(data))); err == nil {
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		d := image.NewRGBA(image.Rect(0, 0, 256, 256))
		r.res("cancel", im2.DecodeTile(cctx, 0, 0, 0, d))
	}
	// A reader size one short and one over the real length (the open reads the header only).
	for _, d := range []int64{-1, 1} {
		_, err := isyntax.OpenContext(ctx, bytes.NewReader(data), int64(len(data))+d)
		r.res(fmt.Sprintf("size%+d", d), err)
	}
	return r.b.String()
}

func decodeAll(in, out string) error {
	// Bound the heap: a soft limit, as the inputs are small and gen filters the ones that are not.
	debug.SetMemoryLimit(1 << 30)
	isyntax.SetMemoryLimit(0) // no limit: the Rust oracle runs with max_alloc = u64::MAX
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].Name() < ents[j].Name() })
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(in, e.Name()))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(out, e.Name()+".out"), []byte(ops(e.Name(), data)), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// minflt is the minor page faults of this process so far.
func minflt() int64 {
	var ru syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		return 0
	}
	return int64(ru.Minflt)
}

// runBench decodes the whole of level 0 of every fixture n times, each time from a fresh Image
// (a cold cache), and prints name, output bytes, ns per decode, the FNV-1a 64 of the pixels and the
// minor page faults per decode.
func runBench(n int, dir string) error {
	isyntax.SetMemoryLimit(0)
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range ents {
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return err
		}
		ctx := context.Background()
		var dst *image.RGBA
		one := func() error {
			im, err := isyntax.Open(bytes.NewReader(data), int64(len(data)))
			if err != nil {
				return err
			}
			lv := im.Level(0)
			if dst == nil {
				dst = image.NewRGBA(image.Rect(0, 0, lv.Width, lv.Height))
			}
			return im.DecodeRegion(ctx, 0, dst.Rect, dst)
		}
		if err := one(); err != nil {
			return fmt.Errorf("%s: %w", e.Name(), err)
		}
		f0 := minflt()
		t := time.Now()
		for i := 0; i < n; i++ {
			if err := one(); err != nil {
				return err
			}
		}
		ns := time.Since(t).Nanoseconds() / int64(n)
		faults := (minflt() - f0) / int64(n)
		h := fnv.New64a()
		h.Write(dst.Pix)
		fmt.Printf("%s\t%d\t%d\t%016x\t%d\n", e.Name(), len(dst.Pix), ns, h.Sum64(), faults)
	}
	return nil
}
