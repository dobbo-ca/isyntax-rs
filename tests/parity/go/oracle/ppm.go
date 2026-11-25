package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"

	isyntax "github.com/dobbo-ca/isyntax-go"
)

// ppmDir is the Go side of the quality gate: for every file of in, in sorted order, it
// opens the file as the parity decode does, with no memory limit, and writes every tile of every
// level, decoded as RGBA, as a binary P6 (alpha dropped) named
// <file name with . as _>.l<level>.<x>.<y>.ppm to out. A file that does not open, and a tile that
// does not decode, write nothing and are named on stderr. examples/ppm.rs is the Rust side.
func ppmDir(in, out string) error {
	isyntax.SetMemoryLimit(0)
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].Name() < ents[j].Name() })
	ctx := context.Background()
	count := 0
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(in, e.Name()))
		if err != nil {
			return err
		}
		im, err := isyntax.OpenContext(ctx, bytes.NewReader(data), int64(len(data)))
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: open: %v\n", e.Name(), err)
			continue
		}
		stem := strings.ReplaceAll(e.Name(), ".", "_")
		for l := 0; l < im.Levels(); l++ {
			lv := im.Level(l)
			side := im.TileSize()
			if l == im.Levels()-1 {
				side = 128
			}
			for y := 0; y < lv.TilesY; y++ {
				for x := 0; x < lv.TilesX; x++ {
					d := image.NewRGBA(image.Rect(0, 0, side, side))
					if err := im.DecodeTile(ctx, l, x, y, d); err != nil {
						fmt.Fprintf(os.Stderr, "%s: tile %d.%d.%d: %v\n", e.Name(), l, x, y, err)
						continue
					}
					buf := fmt.Appendf(make([]byte, 0, 32+3*side*side), "P6\n%d %d\n255\n", side, side)
					for i := 0; i < len(d.Pix); i += 4 {
						buf = append(buf, d.Pix[i:i+3]...)
					}
					name := fmt.Sprintf("%s.l%d.%d.%d.ppm", stem, l, x, y)
					if err := os.WriteFile(filepath.Join(out, name), buf, 0o644); err != nil {
						return err
					}
					count++
				}
			}
		}
	}
	fmt.Printf("%d tiles decoded\n", count)
	return nil
}
