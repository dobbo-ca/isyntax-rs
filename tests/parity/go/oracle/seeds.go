package main

// The seeds of the fuzz targets (tests/fuzz/fuzz_targets), from the f.Add calls of the Go fuzz tests:
// FuzzOpenDecode and FuzzParse (internal/isyntax, internal/container: the synthetic files and their
// halves), FuzzDecodeBase64, FuzzParseClusterTable and hulsken's FuzzDecode, in the input formats of
// the Rust targets. The Go repo has no testdata/fuzz directory.

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/dobbo-ca/isyntax-go/internal/fixture"
)

func generateSeeds(dir string) error {
	var err error
	put := func(target string, n int, b []byte) {
		if err != nil {
			return
		}
		d := filepath.Join(dir, target)
		if err = os.MkdirAll(d, 0o755); err == nil {
			err = os.WriteFile(filepath.Join(d, fmt.Sprintf("seed%02d", n)), b, 0o644)
		}
	}

	// open and decode: the files of FuzzOpenDecode, FuzzParse's halves, a small file with real
	// coefficients and one with colour planes.
	files := [][]byte{[]byte("hello\x04"), {}}
	for _, s := range fuzzSpecs() {
		f, _ := fixture.File(s)
		files = append(files, f, f[:len(f)/2])
	}
	for _, in := range []input{build("l1", sparse(300, 300, 1)), buildPatched("rgb", sparse(300, 300, 1), planes(rgbNames))} {
		files = append(files, in.file, in.file[:len(in.file)/2])
	}
	for i, f := range files {
		put("open", i, f)
	}
	n := 0
	for i, f := range files[2:] {
		for seed := byte(1); seed <= 3; seed++ {
			put("decode", n, append([]byte{seed * byte(i+1), 0, 0, 0, 0, 0, 0, 0}, f...))
			n++
		}
	}

	// hulsken: FuzzDecode's seeds, [version-1, planes, size][block]; 128x128 blocks too.
	r := rand.New(rand.NewPCG(5, 6))
	for i, b := range hulskenSeeds(r) {
		put("hulsken", i, b)
	}

	// tables: [0] base64 (FuzzDecodeBase64), [1] the cluster table (FuzzParseClusterTable), [2] the
	// XML header of a file.
	n = 0
	for _, s := range base64Seeds {
		put("tables", n, append([]byte{0}, s...))
		n++
	}
	for _, t := range clusterTableSeeds() {
		put("tables", n, append([]byte{1}, t...))
		n++
	}
	for _, f := range files[2:] {
		if eot := bytes.IndexByte(f, 4); eot > 0 {
			put("tables", n, append([]byte{2}, f[:eot]...))
			n++
		}
	}

	// kernels: random bytes, one seed per kernel and a few shapes.
	n = 0
	for k := 0; k < 8; k++ {
		for _, size := range []int{20, 40, 300} {
			s := make([]byte, 8+size)
			for i := range s {
				s[i] = byte(r.IntN(256))
			}
			s[0], s[1] = byte(k), byte(size/2)
			put("kernels", n, s)
			n++
		}
	}
	return err
}

// hulskenSeeds are FuzzDecode's seeds (internal/hulsken): PCG blocks of 16 x 16 and 128 x 128,
// CounterBits 4, as `[version-1, planes/3, size index][block]`. The draws come from r.
func hulskenSeeds(r *rand.Rand) [][]byte {
	randomBlock := func(n, bw int, bitsUsed int, zeroShare float64) []int16 {
		c := make([]int16, n*bw*bw)
		for i := range c {
			if r.Float64() < zeroShare {
				continue
			}
			m := int16(r.IntN(1 << bitsUsed))
			if r.IntN(2) == 0 {
				m = -m
			}
			c[i] = m
		}
		return c
	}
	var out [][]byte
	for v := 1; v <= 2; v++ {
		for _, planes := range []int{1, 3} {
			p := fixture.HulskenParams{Version: v, CounterBits: 4}
			b := fixture.EncodeHulsken(randomBlock(planes, 16, 6, 0.5), planes, 16, 16, p)
			out = append(out, append([]byte{byte(v - 1), byte(planes / 3), 2}, b...))
			b = fixture.EncodeHulsken(randomBlock(planes, 128, 9, 0.9), planes, 128, 128, p)
			out = append(out, append([]byte{byte(v - 1), byte(planes / 3), 3}, b...))
		}
	}
	return out
}

// base64Seeds are FuzzDecodeBase64's.
var base64Seeds = []string{"aGVsbG8=", "aGVs\r\nbG8=", "aGVsbG8=/", "", "aGVsbA===", "===="}

// clusterTableSeeds are FuzzParseClusterTable's.
func clusterTableSeeds() [][]byte {
	var b []byte
	le := func(v uint64, n int) {
		for i := 0; i < n; i++ {
			b = append(b, byte(v>>(8*i)))
		}
	}
	tag := func(g, e uint16, size int) { le(uint64(g), 2); le(uint64(e), 2); le(uint64(size), 4) }
	tag(0xFFFE, 0xE000, 8+8+8+4+8+8+8+8+8+8+8+8)
	tag(0x301D, 0x2020, 8)
	le(100, 4)
	le(200, 4)
	tag(0x301D, 0x2019, 4)
	le(0, 4)
	tag(0x301D, 0x2010, 8)
	le(5000, 8)
	tag(0x301D, 0x2011, 8)
	le(30, 8)
	tag(0x301D, 0x201B, 8)
	le(0, 4)
	le(10, 4)
	tag(0x301D, 0x201C, 8)
	le(10, 4)
	le(20, 4)
	table := append([]byte{byte(len(b)), byte(len(b) >> 8), 0, 0}, b...)
	return [][]byte{table, table[:len(table)-5], {}, {0, 0, 0, 0}}
}
