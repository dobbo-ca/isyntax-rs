package main

// The inputs of the parity job: the synthetic files of the Go tests (the specs of
// internal/decoder/decode_test.go, rules_test.go, internal/container/container_test.go and the
// seeds of the two fuzz_test.go), the random codeblocks of internal/hulsken/hulsken_test.go
// (TestRoundTrip's randomBlock) as the coefficients of whole files, and deterministic mutations of
// ten small files: truncations, bit flips, digit edits in the XML header, edits of the cluster and
// block header tables, and edits of the first bytes of codeblocks (the Hulsken length fields).

import (
	"bytes"
	"context"
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"github.com/dobbo-ca/isyntax-go/internal/container"
	"github.com/dobbo-ca/isyntax-go/internal/fixture"
)

const area = 128 * 128

type coefFn = func(s, x, y, c int, ll bool) []int16

// randCoef is randSpec's generator of internal/decoder/decode_test.go: 1 in density detail
// coefficients is nonzero, in [-amp, amp).
func randCoef(density, amp int) coefFn {
	return func(s, x, y, c int, ll bool) []int16 {
		r := rand.New(rand.NewPCG(uint64(s*1000+x*31+y), uint64(c)))
		n := 3
		if ll {
			n = 1
		}
		v := make([]int16, n*area)
		for i := range v {
			switch {
			case ll && c == 0:
				v[i] = int16(100 + r.IntN(120))
			case ll:
				v[i] = int16(r.IntN(60) - 30)
			case r.IntN(density) == 0:
				v[i] = int16(r.IntN(2*amp) - amp)
			}
		}
		return v
	}
}

// blockCoef is randomBlock of internal/hulsken/hulsken_test.go: values of up to bitsUsed bits, a
// share zero of them zero.
func blockCoef(seed uint64, bitsUsed int, zero float64) coefFn {
	return func(s, x, y, c int, ll bool) []int16 {
		r := rand.New(rand.NewPCG(seed+uint64(s*1000+x*31+y), uint64(c)))
		n := 3
		if ll {
			n = 1
		}
		v := make([]int16, n*area)
		for i := range v {
			if r.Float64() < zero {
				continue
			}
			m := int16(r.IntN(1 << bitsUsed))
			if r.IntN(2) == 0 {
				m = -m
			}
			v[i] = m
		}
		return v
	}
}

// constCoef is constSpec's: flat grey y, LL only.
func constCoef(y int16) coefFn {
	return func(s, x, yy, c int, ll bool) []int16 {
		if !ll {
			return make([]int16, 3*area)
		}
		v := make([]int16, area)
		if c == 0 {
			for i := range v {
				v[i] = y
			}
		}
		return v
	}
}

func randSpec(compressor int) fixture.FileSpec {
	return fixture.FileSpec{
		Compressor: compressor, BitsStored: 8, Levels: 3, Width: 1500, Height: 900, MPP: 0.25,
		OffsetX: 3, OffsetY: 5,
		Exists: func(s, x, y int) bool { return s > 0 || (x*7+y*3)%5 != 0 },
		Coef:   randCoef(4, 20),
	}
}

// wideSpec is wideSpec of internal/decoder/rgba64_test.go: LL samples outside the clamp range.
func wideSpec(compressor, bits int) fixture.FileSpec {
	s := randSpec(compressor)
	s.BitsStored = bits
	coef := s.Coef
	s.Coef = func(sc, x, y, c int, ll bool) []int16 {
		v := coef(sc, x, y, c, ll)
		if ll {
			r := rand.New(rand.NewPCG(uint64(sc*1000+x*31+y), uint64(c)+9))
			for i := range v {
				if c == 0 {
					v[i] = int16(r.IntN(900) - 200)
				} else {
					v[i] = int16(r.IntN(500) - 250)
				}
			}
		}
		return v
	}
	return s
}

func constSpec(y int16, bits int) fixture.FileSpec {
	return fixture.FileSpec{Compressor: 19, BitsStored: bits, Levels: 3, Width: 1500, Height: 900, MPP: 0.25, Coef: constCoef(y)}
}

func missingAt(y int16, bits, s, tx, ty int) fixture.FileSpec {
	spec := constSpec(y, bits)
	spec.Exists = func(sc, x, yy int) bool { return sc != s || x != tx || yy != ty }
	return spec
}

// containerSpec and containerV1 are spec and v1Spec of internal/container/container_test.go.
func containerSpec() fixture.FileSpec {
	return fixture.FileSpec{
		Compressor: 19, BitsStored: 9, Levels: 3,
		Width: 1500, Height: 900, OffsetX: 7, OffsetY: 11, MPP: 0.25,
		Exists: func(s, x, y int) bool { return (x+y+s)%3 != 0 },
		Coef: func(s, x, y, c int, ll bool) []int16 {
			n := 3
			if ll {
				n = 1
			}
			v := make([]int16, n*area)
			v[x+y+c] = int16(s + 1)
			return v
		},
		Label: []byte("label jpeg"), Macro: []byte("macro jpeg bytes"), ICC: []byte("icc!"),
	}
}

func containerV1() fixture.FileSpec {
	s := containerSpec()
	s.Version, s.Compressor, s.BitsStored = "5.0", 16, 16
	return s
}

// fuzzSpecs are the seeds of FuzzOpenDecode (internal/isyntax/fuzz_test.go).
func fuzzSpecs() []fixture.FileSpec {
	zero := func(s, x, y, c int, ll bool) []int16 {
		n := 3
		if ll {
			n = 1
		}
		return make([]int16, n*area)
	}
	return []fixture.FileSpec{
		{Compressor: 19, BitsStored: 9, Levels: 2, Width: 600, Height: 500, MPP: 0.5, Coef: zero,
			Label: []byte{0xff, 0xd8, 1}, Macro: []byte{0xff, 0xd8, 2}, ICC: []byte("icc")},
		{Compressor: 16, BitsStored: 8, Levels: 3, Width: 900, Height: 700,
			Exists: func(s, x, y int) bool { return (x+y+s)%2 == 0 }, Coef: zero},
		{Version: "5.0", Compressor: 16, BitsStored: 16, Levels: 2, Width: 600, Height: 500, MPP: 0.5, Coef: zero,
			Label: []byte{0xff, 0xd8, 1}, Macro: []byte{0xff, 0xd8, 2}, ICC: []byte("icc")},
	}
}

// attr is the XML of one attribute, as the fixture writes them.
func attr(g, e uint16, vr, v string) string {
	return fmt.Sprintf("<Attribute Name=\"X\" Group=\"0x%04x\" Element=\"0x%04x\" PMSVR=%q>%s</Attribute>\n", g, e, vr, v)
}

// patched builds s, then makes the replacements, once each, without changing the file's length (the
// cluster table holds absolute offsets): s.Extra carries an XML comment as long as the growth, and
// it is removed after.
func patched(s fixture.FileSpec, pairs ...[2]string) []byte {
	grow := 0
	for _, p := range pairs {
		grow += len(p[1]) - len(p[0])
	}
	if grow < 7 {
		panic("patched: the edits must grow the file by at least 7 bytes")
	}
	s.Extra = "<!--" + strings.Repeat("x", grow-7) + "-->"
	file, _ := fixture.File(s)
	for _, p := range pairs {
		if !bytes.Contains(file, []byte(p[0])) {
			panic("patched: no " + p[0])
		}
		file = bytes.Replace(file, []byte(p[0]), []byte(p[1]), 1)
	}
	return bytes.Replace(file, []byte(s.Extra), nil, 1)
}

// planes is the edit that gives the file colour plane names (internal/container TestUnsupported).
func planes(vals string) [2]string {
	return [2]string{
		"<DataObject ObjectType=\"UFSImageDimension\">\n</DataObject>\n",
		"<DataObject ObjectType=\"UFSImageDimension\">\n" +
			"<Attribute Name=\"UFS_IMAGE_DIMENSION_TYPE\" Group=\"0x301D\" Element=\"0x2005\" PMSVR=\"IString\">colour component</Attribute>\n" +
			"<Attribute Name=\"UFS_IMAGE_DIMENSION_DISCRETE_VALUES_STRING\" Group=\"0x301D\" Element=\"0x2008\" PMSVR=\"IString\">" + vals + "</Attribute>\n" +
			"</DataObject>\n",
	}
}

// quality is the edit that gives the WSI image a DICOM_DERIVATION_DESCRIPTION with Quality=q.
func quality(q int) [2]string {
	wsi := "PMSVR=\"IString\">WSI</Attribute>\n"
	return [2]string{wsi, wsi + attr(0x0008, 0x2111, "IString", fmt.Sprintf("PHILIPS UFS B300 V1.1.1.1 | Quality=%d | DWT=LeGall53 | Compressor=Hulsken2", q))}
}

const rgbNames = "&quot;R&quot; &quot;G&quot; &quot;B&quot;"

type input struct {
	name string
	file []byte
	blk  []fixture.Block
}

func build(name string, s fixture.FileSpec) input {
	f, b := fixture.File(s)
	return input{name, f, b}
}

func buildPatched(name string, s fixture.FileSpec, pairs ...[2]string) input {
	// The blocks do not move: the length is the same.
	_, b := fixture.File(s)
	return input{name, patched(s, pairs...), b}
}

// sparse is a small file of the given shape: sparse random detail, 8 bits, iSyntax2 compressor 19.
func sparse(w, h, levels int) fixture.FileSpec {
	return fixture.FileSpec{Compressor: 19, BitsStored: 8, Levels: levels, Width: w, Height: h, MPP: 0.5, Coef: randCoef(64, 6)}
}

// bases are the files written as they are.
func bases() []input {
	var in []input
	add := func(name string, s fixture.FileSpec) { in = append(in, build(name, s)) }
	add("rand19", randSpec(19))
	add("rand16", randSpec(16))
	v1 := randSpec(16)
	v1.Version, v1.BitsStored = "5.0", 16
	add("rand-v1", v1)
	// The wideSpec files of the decoder tests, in YCoCg and with colour planes, and the v1 one.
	for _, w := range [][2]int{{16, 8}, {19, 9}, {19, 8}} {
		s := wideSpec(w[0], w[1])
		add(fmt.Sprintf("wide-c%d-b%d", w[0], w[1]), s)
		in = append(in, buildPatched(fmt.Sprintf("wide-rgb-c%d-b%d", w[0], w[1]), s, planes(rgbNames)))
	}
	wv1 := wideSpec(16, 16)
	wv1.Version = "5.0"
	add("wide-v1-16", wv1)
	for _, c := range []struct {
		y    int16
		bits int
	}{{0, 8}, {100, 8}, {200, 9}, {-50, 8}, {300, 8}, {-300, 9}} {
		add(fmt.Sprintf("const-y%d-b%d", c.y, c.bits), constSpec(c.y, c.bits))
	}
	add("missing-0-3-1", missingAt(100, 8, 0, 3, 1))
	add("missing-0-3-1-b9", missingAt(200, 9, 0, 3, 1))
	add("missing-top", missingAt(100, 8, 2, 1, 0))
	add("cont-v2", containerSpec())
	add("cont-v1", containerV1())
	for i, s := range fuzzSpecs() {
		add(fmt.Sprintf("fuzz-seed%d", i), s)
	}
	for _, g := range [][3]int{{1, 1, 1}, {5, 3, 2}, {255, 257, 1}, {2000, 100, 4}, {130, 3000, 3}} {
		add(fmt.Sprintf("geo-%dx%d-l%d", g[0], g[1], g[2]), sparse(g[0], g[1], g[2]))
	}
	// TestCalibrationFields.
	for name, extra := range map[string]string{
		"cal-present": attr(0x101D, 0x100A, "IString", " OK ") +
			attr(0x0018, 0x1200, "IStringArray", `"20190612"`) + attr(0x0018, 0x1201, "IStringArray", `"101500.000000"`) +
			attr(0x101D, 0x1007, "IUInt16", "3") + attr(0x101D, 0x1008, "IUInt16", " 12 "),
		"cal-multi": attr(0x0018, 0x1200, "IStringArray", `"20190612" "20190701"`) +
			attr(0x0018, 0x1201, "IStringArray", `"1" "2" "3"`),
		"cal-malformed": attr(0x0018, 0x1200, "IStringArray", `20190612`) + attr(0x0018, 0x1201, "IStringArray", `"open`) +
			attr(0x101D, 0x1007, "IUInt16", "70000") + attr(0x101D, 0x1008, "IUInt16", "-1"),
		"cal-malformed-keeps": attr(0x0018, 0x1200, "IStringArray", `"a" "b`) + attr(0x101D, 0x1007, "IUInt16", "x") +
			attr(0x101D, 0x1008, "IUInt16", "0"),
		"meta": attr(0x0008, 0x002A, "IString", " 20190612101500 ") + attr(0x0008, 0x1090, "IString", "Model X") +
			attr(0x0018, 0x1000, "IString", "SN123") + attr(0x301D, 0x1002, "IString", "YmFyY29kZQ=="),
	} {
		s := containerSpec()
		s.Extra = extra
		add(name, s)
	}
	// TestUnsupported: the edits that make a supported file unsupported.
	c16 := containerSpec()
	c16.Compressor = 16
	f16, _ := fixture.File(c16)
	in = append(in,
		input{name: "unsup-comp17", file: bytes.ReplaceAll(f16, []byte("PMSVR=\"IString\">16<"), []byte("PMSVR=\"IString\">17<"))},
		input{name: "unsup-bits16", file: bytes.ReplaceAll(f16, []byte("PMSVR=\"IUInt16\">9<"), []byte("PMSVR=\"IUInt16\">16<"))})
	v1c19 := containerV1()
	v1c19.Compressor = 19
	add("unsup-v1c19", v1c19)
	in = append(in, buildPatched("unsup-planes", containerSpec(), planes("&quot;X&quot; &quot;Y&quot; &quot;Z&quot;")))
	// Colour planes R, G, B (the planes functions of the decode), with the SDK truncation presets.
	rgb2 := sparse(600, 500, 2)
	in = append(in, buildPatched("rgb-v2-8", rgb2, planes(rgbNames)))
	rgb9 := rgb2
	rgb9.BitsStored = 9
	in = append(in, buildPatched("rgb-v2-9", rgb9, planes(`"R" "G" "B"`)))
	rgbV1 := rgb2
	rgbV1.Version, rgbV1.Compressor, rgbV1.BitsStored = "5.0", 16, 8
	in = append(in, buildPatched("rgb-v1-8", rgbV1, planes(rgbNames)))
	rgbQ1 := rgbV1
	rgbQ1.Levels = 1
	in = append(in, buildPatched("rgb-v1-8-q1", rgbQ1, planes(rgbNames), quality(1)))
	for _, q := range []int{0, 1, 2, 3} {
		in = append(in, buildPatched(fmt.Sprintf("q%d-v2-8", q), sparse(600, 500, 2), quality(q)))
	}
	q9 := sparse(600, 500, 2)
	q9.BitsStored = 9
	in = append(in, buildPatched("q2-v2-9", q9, quality(2)))
	q1 := sparse(600, 500, 2)
	q1.Compressor = 16
	in = append(in, buildPatched("q1-v2c16-8", q1, quality(1)))
	v1q := sparse(600, 500, 2)
	v1q.Version, v1q.Compressor, v1q.BitsStored = "5.0", 16, 16
	in = append(in, buildPatched("q2-v1-16", v1q, quality(2)))
	// TestRoundTrip's random blocks, as whole files: every Hulsken version, bit count and zero share.
	n := uint64(0)
	for _, comp := range []int{16, 19} {
		for _, bitsUsed := range []int{1, 3, 6, 9, 12, 15} {
			for _, zero := range []float64{0, 0.5, 0.9, 0.999, 1} {
				n++
				if (zero == 0 && bitsUsed > 6) || (zero == 0.5 && bitsUsed > 9) {
					continue // the dense ones are megabytes each
				}
				s := fixture.FileSpec{Compressor: comp, BitsStored: 8, Levels: 1, Width: 300, Height: 300, MPP: 0.25, Coef: blockCoef(n, bitsUsed, zero)}
				add(fmt.Sprintf("hk-c%d-b%d-z%v", comp, bitsUsed, zero), s)
			}
		}
	}
	return append(in, xmlEdgeInputs()...)
}

// xmlEdgeInputs are the corners of encoding/xml that a single bit flip cannot reach, as edits that
// keep the file's length: a name with a non-letter rune (U+00D7), the directives `<!-x>`, `<!'>` and
// `<!<>` in text, `<?xml?>` parameters that need a second look (`version=1 version="2.0"`), and a
// 5000-deep subtree that no struct field takes.
func xmlEdgeInputs() []input {
	sameLen := func(name string, s fixture.FileSpec, from, to string) input {
		if len(to) > len(from) {
			panic("xmlEdgeInputs: " + name + " grows")
		}
		f, b := fixture.File(s)
		if !bytes.Contains(f, []byte(from)) {
			panic("xmlEdgeInputs: no " + from)
		}
		to += strings.Repeat(" ", len(from)-len(to))
		return input{name, bytes.Replace(f, []byte(from), []byte(to), 1), b}
	}
	var in []input
	in = append(in, sameLen("xml-name-times", containerSpec(), "PMSVR=", "PMS\xC3\x97="))
	const head = `version="1.0" encoding="UTF-8" `
	in = append(in,
		sameLen("xml-pi-version2", containerSpec(), head, `version=1 version="2.0"`),
		sameLen("xml-pi-version1", containerSpec(), head, `version=1 version="1.0"`),
		sameLen("xml-pi-encoding", containerSpec(), head, `encoding=x encoding="UTF-16"`),
		sameLen("xml-pi-encoding-utf8", containerSpec(), head, `encoding=x encoding="utf-8"`))
	for name, d := range map[string]string{"bad-comment": "<!-x>", "quote": "<!'>x", "lt": "<!<>x", "comment-in": "<!x<!-- > -->>"} {
		s := containerSpec()
		s.Extra = attr(0x0001, 0x0001, "IString", strings.Repeat("a", len(d)))
		in = append(in, sameLen("xml-dir-"+name, s, ">"+strings.Repeat("a", len(d))+"<", ">"+d+"<"))
	}
	deep := containerSpec()
	deep.Extra = strings.Repeat("<x>", 5000) + strings.Repeat("</x>", 5000)
	return append(in, build("xml-deep-skip", deep))
}

// mutationBases are the ten small files that get mutated.
func mutationBases() []input {
	label := func(s fixture.FileSpec) fixture.FileSpec {
		s.Label, s.Macro, s.ICC = []byte{0xff, 0xd8, 1}, []byte{0xff, 0xd8, 2}, []byte("icc")
		return s
	}
	v19 := label(sparse(600, 500, 2))
	v16 := sparse(600, 500, 2)
	v16.Compressor = 16
	v16.Exists = func(s, x, y int) bool { return (x+y+s)%2 == 0 }
	v19b9 := sparse(600, 500, 2)
	v19b9.BitsStored = 9
	v1 := label(sparse(600, 500, 2))
	v1.Version, v1.Compressor, v1.BitsStored = "5.0", 16, 16
	v1b8 := sparse(600, 500, 2)
	v1b8.Version, v1b8.Compressor, v1b8.BitsStored = "5.0", 16, 8
	v1b8.Exists = func(s, x, y int) bool { return s > 0 || (x*7+y*3)%5 != 0 }
	cal := sparse(600, 500, 2)
	cal.Extra = attr(0x101D, 0x100A, "IString", " OK ") + attr(0x0018, 0x1200, "IStringArray", `"20190612"`) +
		attr(0x101D, 0x1007, "IUInt16", "3")
	return []input{
		build("m-v19-8", v19),
		build("m-v16-8", v16),
		build("m-v19-9", v19b9),
		build("m-v1-16", v1),
		build("m-v1-8", v1b8),
		buildPatched("m-rgb", sparse(600, 500, 2), planes(rgbNames)),
		buildPatched("m-q2", sparse(600, 500, 2), quality(2)),
		build("m-cal", cal),
		build("m-lev1", sparse(300, 300, 1)),
		build("m-lev3", sparse(1000, 700, 3)),
	}
}

type mutant struct {
	name string
	file []byte
}

// mutations are the deterministic mutants of in.
func mutations(in input) []mutant {
	h := fnv.New64a()
	h.Write([]byte(in.name))
	r := rand.New(rand.NewPCG(h.Sum64(), 1))
	file := in.file
	head := bytes.IndexByte(file, 4) // the end of the XML header
	var out []mutant
	add := func(kind string, i int, f []byte) {
		out = append(out, mutant{fmt.Sprintf("%s~%s%02d", in.name, kind, i), f})
	}
	edit := func(f func(b []byte)) []byte {
		b := bytes.Clone(file)
		f(b)
		return b
	}
	// Truncations: around the end of the header, and at random.
	cuts := []int{0, 1, head - 1, head, head + 1, head + 9, len(file) - 1, len(file) / 2}
	for i := 0; i < 6; i++ {
		cuts = append(cuts, r.IntN(len(file)))
	}
	for i, c := range cuts {
		add("t", i, bytes.Clone(file[:max(0, c)]))
	}
	// Bit flips, half in the header.
	for i := 0; i < 24; i++ {
		p := r.IntN(head)
		if i%2 == 1 {
			p = head + r.IntN(len(file)-head)
		}
		bit := byte(1) << r.IntN(8)
		add("f", i, edit(func(b []byte) { b[p] ^= bit }))
	}
	// Digit edits in the XML header: dimensions, counts, offsets.
	for i := 0; i < 10; i++ {
		p := -1
		for try := 0; try < 2000 && p < 0; try++ {
			if q := r.IntN(head); file[q] >= '0' && file[q] <= '9' {
				p = q
			}
		}
		if p < 0 {
			continue
		}
		d := byte('0' + r.IntN(10))
		add("d", i, edit(func(b []byte) { b[p] = d }))
	}
	// Edits of the base64 of the cluster header table or the block header table.
	for _, tag := range []string{"UFS_IMAGE_CLUSTER_HEADER_TABLE", "UFS_IMAGE_BLOCK_HEADER_TABLE"} {
		at := bytes.Index(file, []byte(tag))
		if at < 0 {
			continue
		}
		from := at + bytes.IndexByte(file[at:], '>') + 1
		to := from + bytes.Index(file[from:], []byte("</Attribute>"))
		if to <= from {
			continue
		}
		const b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
		for i := 0; i < 8; i++ {
			p := from + r.IntN(to-from)
			c := b64[r.IntN(len(b64))]
			add("x", len(out), edit(func(b []byte) { b[p] = c }))
		}
	}
	// Codeblocks: a bit flip in the first bytes (the Hulsken header), and a 4-byte word set to a
	// length-field extreme.
	for i := 0; i < 16 && len(in.blk) > 0; i++ {
		b := in.blk[r.IntN(len(in.blk))]
		if b.Size < 1 {
			continue
		}
		p := int(b.Offset) + r.IntN(int(min(b.Size, 32)))
		bit := byte(1) << r.IntN(8)
		add("b", i, edit(func(f []byte) { f[p] ^= bit }))
	}
	for i := 0; i < 10 && len(in.blk) > 0; i++ {
		b := in.blk[r.IntN(len(in.blk))]
		if b.Size < 12 {
			continue
		}
		w := []uint32{0, 0xFFFFFFFF, 0x7FFFFFFF, uint32(b.Size) + 1, 1 << 16, uint32(b.Size) - 1}[r.IntN(6)]
		p := int(b.Offset) + 4*r.IntN(3)
		add("z", i, edit(func(f []byte) {
			f[p], f[p+1], f[p+2], f[p+3] = byte(w), byte(w>>8), byte(w>>16), byte(w>>24)
		}))
	}
	return out
}

// tame reports whether the decoders can take f on a shared builder: whatever fails to parse is
// fine, but a file that parses into an image with a huge tile grid would make Go allocate it.
func tame(f []byte) bool {
	h, err := container.Parse(context.Background(), bytes.NewReader(f), int64(len(f)))
	if err != nil {
		return true
	}
	return h.Width*h.Height <= 1<<28 && h.GridW*h.GridH <= 1<<16
}

// hulskenBlock is a codeblock for the `.blk` inputs: [version-1, planes/3, size index][codeblock],
// the format of the hulsken fuzz target.
type hulskenBlock struct {
	name string
	data []byte
}

// hulskenBlocks encodes the codeblocks of hulsken's TestRoundTrip (versions 1 and 2, 1 and 3
// planes, block sides 8, 16 and 128, 40 trials each, with the test's zero symbols, counter bits 1
// to 8, bits used 1 to 15 and zero shares) and of TestAllPlanes (the full int16 range), so that the
// Hulsken decoder is compared on the parameters the whole files never use. The PRNGs and the order
// of draws are the tests'.
func hulskenBlocks() []hulskenBlock {
	randomBlock := func(r *rand.Rand, n, bw, bitsUsed int, zeroShare float64) []int16 {
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
	sizeIdx := map[int]byte{4: 0, 8: 1, 16: 2, 128: 3}
	var out []hulskenBlock
	put := func(name string, version, n, bw int, src []byte) {
		out = append(out, hulskenBlock{name, append([]byte{byte(version - 1), byte(n / 3), sizeIdx[bw]}, src...)})
	}
	r := rand.New(rand.NewPCG(1, 2))
	for _, version := range []int{1, 2} {
		for _, n := range []int{1, 3} {
			for _, bw := range []int{8, 16, 128} {
				for trial := 0; trial < 40; trial++ {
					bitsUsed := 1 + r.IntN(15)
					zero := []float64{0, 0.5, 0.9, 0.999, 1}[trial%5]
					want := randomBlock(r, n, bw, bitsUsed, zero)
					p := fixture.HulskenParams{
						Version:     version,
						ZeroSymbol:  []byte{0, 0x7f, 0xff, 1}[trial%4],
						CounterBits: 1 + r.IntN(8),
					}
					put(fmt.Sprintf("hulsken-v%d-n%d-b%d-%02d.blk", version, n, bw, trial), version, n, bw,
						fixture.EncodeHulsken(want, n, bw, bw, p))
				}
			}
		}
	}
	r = rand.New(rand.NewPCG(3, 4))
	for _, version := range []int{1, 2} {
		want := make([]int16, 3*16*16)
		for i := range want {
			want[i] = int16(r.IntN(1<<16) - 1<<15)
			if want[i] == -32768 {
				want[i] = -32767
			}
		}
		for i := 0; i < 3; i++ {
			want[i*256] = 0x7fff
			want[i*256+1] = -0x7fff
		}
		put(fmt.Sprintf("hulsken-v%d-allplanes.blk", version), version, 3, 16,
			fixture.EncodeHulsken(want, 3, 16, 16, fixture.HulskenParams{Version: version, ZeroSymbol: 0, CounterBits: 3}))
	}
	return out
}

// blkMutants are a truncation, a bit flip anywhere in the codeblock and one in its first bytes (the
// Hulsken header) of a `.blk` input: the error paths with that block's own parameters.
func blkMutants(r *rand.Rand, data []byte) [][]byte {
	const hdr = 3
	body := len(data) - hdr
	if body <= 0 {
		return nil
	}
	cut := append([]byte(nil), data[:hdr+r.IntN(body)]...)
	flip := func(pos int) []byte {
		m := append([]byte(nil), data...)
		m[pos] ^= 1 << r.IntN(8)
		return m
	}
	anywhere := flip(hdr + r.IntN(body))
	first := flip(hdr + r.IntN(min(body, 12)))
	return [][]byte{cut, anywhere, first}
}

// fuzzSeedInputs are the seeds of the Go fuzz tests that are not synthetic files of bases(): the
// small files and the halves of FuzzParse, FuzzDecode's blocks (`.blk`), FuzzDecodeBase64's strings
// (`.b64`) and FuzzParseClusterTable's tables (`.ct`), which the oracles feed to the parsers directly.
func fuzzSeedInputs() []input {
	in := []input{{name: "seed-open-hello", file: []byte("hello\x04")}, {name: "seed-open-empty"}}
	for i, s := range fuzzSpecs() {
		f, _ := fixture.File(s)
		in = append(in, input{name: fmt.Sprintf("seed-open-half%d", i), file: f[:len(f)/2]})
	}
	for i, b := range hulskenSeeds(rand.New(rand.NewPCG(5, 6))) {
		in = append(in, input{name: fmt.Sprintf("seed-hulsken-%02d.blk", i), file: b})
	}
	for i, s := range base64Seeds {
		in = append(in, input{name: fmt.Sprintf("seed-b64-%d.b64", i), file: []byte(s)})
	}
	for i, t := range clusterTableSeeds() {
		in = append(in, input{name: fmt.Sprintf("seed-ct-%d.ct", i), file: t})
	}
	return in
}

func generate(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	write := func(name string, f []byte) error { return os.WriteFile(filepath.Join(dir, name), f, 0o644) }
	nb, nm, dropped := 0, 0, 0
	mr := rand.New(rand.NewPCG(5, 6))
	for _, b := range hulskenBlocks() {
		if err := write(b.name, b.data); err != nil {
			return err
		}
		nb++
		for k, m := range blkMutants(mr, b.data) {
			if err := write(fmt.Sprintf("%s-m%d.blk", b.name[:len(b.name)-len(".blk")], k), m); err != nil {
				return err
			}
			nm++
		}
	}
	for _, in := range append(bases(), fuzzSeedInputs()...) {
		if err := write(in.name, in.file); err != nil {
			return err
		}
		nb++
	}
	for _, in := range mutationBases() {
		if err := write(in.name, in.file); err != nil {
			return err
		}
		nb++
		for _, m := range mutations(in) {
			if !tame(m.file) {
				dropped++
				continue
			}
			if err := write(m.name, m.file); err != nil {
				return err
			}
			nm++
		}
	}
	fmt.Printf("%d files, %d mutants (%d dropped as too large)\n", nb, nm, dropped)
	return nil
}

// generateBench writes the benchmark fixtures: 3000x2000 files, four levels of Go-style random
// detail of different densities, in the three formats.
func generateBench(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	mk := func(comp, bits int, version string, c coefFn) fixture.FileSpec {
		return fixture.FileSpec{Version: version, Compressor: comp, BitsStored: bits, Levels: 3, Width: 3000, Height: 2000, MPP: 0.25, Coef: c}
	}
	for name, s := range map[string]fixture.FileSpec{
		"dense-v2-8":   mk(19, 8, "", randCoef(4, 20)),
		"dense-v2-9":   mk(19, 9, "", randCoef(4, 40)),
		"sparse-v2-8":  mk(19, 8, "", randCoef(40, 3)),
		"dense-v1-16":  mk(16, 16, "5.0", randCoef(4, 20)),
		"sparse-v1-16": mk(16, 16, "5.0", randCoef(40, 3)),
		"flat-v2-8":    mk(19, 8, "", constCoef(100)),
	} {
		f, _ := fixture.File(s)
		if err := os.WriteFile(filepath.Join(dir, name), f, 0o644); err != nil {
			return err
		}
	}
	return nil
}
