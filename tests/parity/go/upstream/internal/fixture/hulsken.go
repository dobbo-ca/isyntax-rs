// Package fixture builds synthetic iSyntax inputs for tests: Hulsken
// codeblocks and whole iSyntax2 files. It is test-only; nothing here
// comes from slide data.
package fixture

import (
	"container/heap"
	"encoding/binary"
	"math/bits"
)

// HulskenParams chooses the encoder's free parameters.
type HulskenParams struct {
	Version     int  // 1 or 2, per compressor
	ZeroSymbol  byte // symbol that starts a zero run
	CounterBits int  // bits per zero-run counter chunk, 1..8
}

// EncodeHulsken encodes n coefficient planes of bw*bh values in raster
// order as one codeblock, the inverse of docs/format.md "Entropy coding".
func EncodeHulsken(coef []int16, n, bw, bh int, p HulskenParams) []byte {
	area := bw * bh
	planeBytes := area / 8
	// Sign-magnitude, in 4x4 element order.
	sm := make([]uint16, n*area)
	for c := 0; c < n; c++ {
		for e := 0; e < area; e++ {
			a, w := e/16, e%16
			x := (a%(bw/4))*4 + w%4
			y := (a/(bw/4))*4 + w/4
			v := int32(coef[c*area+y*bw+x])
			u := uint16(v)
			if v < 0 {
				u = uint16(-v) | 0x8000
			}
			sm[c*area+e] = u
		}
	}
	target := func(b int) int { // value bit for mask bit b
		if p.Version == 1 {
			if b == 0 {
				return 15
			}
			return b - 1
		}
		return 15 - b
	}
	masks := make([]uint16, n)
	for c := range masks {
		for b := 0; b < 16; b++ {
			for e := 0; e < area; e++ {
				if sm[c*area+e]>>target(b)&1 != 0 {
					masks[c] |= 1 << b
					break
				}
			}
		}
	}
	plane := func(c, bit int) []byte {
		out := make([]byte, planeBytes)
		for e := 0; e < area; e++ {
			out[e/8] |= byte(sm[c*area+e]>>bit&1) << (e % 8)
		}
		return out
	}
	var data []byte
	if p.Version == 1 {
		for c := 0; c < n; c++ {
			for b := 0; b < 16; b++ {
				if masks[c]>>b&1 != 0 {
					data = append(data, plane(c, target(b))...)
				}
			}
		}
		all := true
		for _, m := range masks {
			all = all && m == 0xFFFF
		}
		if !all {
			for _, m := range masks {
				data = binary.LittleEndian.AppendUint16(data, m)
			}
		}
	} else {
		for b := 0; b < 16; b++ {
			for c := 0; c < n; c++ {
				if masks[c]>>b&1 != 0 {
					data = append(data, plane(c, target(b))...)
				}
			}
		}
	}

	var out []byte
	if p.Version == 1 {
		out = binary.LittleEndian.AppendUint32(out, uint32(len(data)))
	} else {
		for _, m := range masks {
			out = binary.LittleEndian.AppendUint16(out, m)
		}
	}
	out = append(out, p.ZeroSymbol, byte(p.CounterBits))
	w := &bitWriter{buf: out, nbit: len(out) * 8}
	if p.Version == 2 && len(data) > 0 {
		var agg uint16
		for _, m := range masks {
			agg |= m
		}
		ptrBits := bits.Len(uint(len(data))) - 1 + 5
		for i := 0; i < bits.OnesCount16(agg)-1; i++ {
			w.write(0, ptrBits) // seek table, skipped by decoders
		}
	}

	toks := tokenize(data, p)
	var freq [256]int
	for _, t := range toks {
		freq[t.sym]++
	}
	codes, lens, tree := huffman(freq)
	writeTree(w, tree)
	for _, t := range toks {
		w.write(uint64(codes[t.sym]), max(1, lens[t.sym]))
		if t.chunk >= 0 {
			w.write(uint64(t.chunk), p.CounterBits)
		}
	}
	return w.bytes()
}

// token: symbol, plus optional counter chunk.
type token struct {
	sym   byte
	chunk int
}

func tokenize(data []byte, p HulskenParams) []token {
	minRun := 1
	if p.Version == 2 {
		minRun = 2
	}
	k := p.CounterBits
	var toks []token
	emitRun := func(r int) {
		s := r
		if p.Version == 2 {
			s = r - 1
		}
		var chunks []int
		for s > 0 {
			chunks = append([]int{s & (1<<k - 1)}, chunks...)
			s >>= k
		}
		for _, c := range chunks {
			toks = append(toks, token{p.ZeroSymbol, c})
		}
	}
	literal := func(b byte) {
		if b == p.ZeroSymbol {
			toks = append(toks, token{b, 0}) // escape
		} else {
			toks = append(toks, token{b, -1})
		}
	}
	for i := 0; i < len(data); {
		if data[i] != 0 {
			literal(data[i])
			i++
			continue
		}
		j := i
		for j < len(data) && data[j] == 0 {
			j++
		}
		r := j - i
		// Runs before an escape absorb it.
		nextEscape := j < len(data) && data[j] == p.ZeroSymbol && p.ZeroSymbol != 0
		switch {
		case p.ZeroSymbol == 0 && r < minRun:
			literal(0) // escape of the zero symbol itself
		case p.ZeroSymbol != 0 && (r < minRun || nextEscape && r-1 < minRun):
			for ; r > 0; r-- {
				literal(0)
			}
		case nextEscape:
			emitRun(r - 1)
			literal(0)
		default:
			emitRun(r)
		}
		i = j
	}
	return toks
}

type hnode struct {
	freq        int
	sym         int // -1 for internal
	left, right *hnode
	order       int
}

type hheap []*hnode

func (h hheap) Len() int { return len(h) }
func (h hheap) Less(i, j int) bool {
	if h[i].freq != h[j].freq {
		return h[i].freq < h[j].freq
	}
	return h[i].order < h[j].order
}
func (h hheap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *hheap) Push(x any)   { *h = append(*h, x.(*hnode)) }
func (h *hheap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// huffman: codes to 16 bits deep.
func huffman(freq [256]int) (codes [256]uint32, lens [256]int, root *hnode) {
	f := freq
	for {
		var h hheap
		for s, n := range f {
			if n > 0 {
				h = append(h, &hnode{freq: n, sym: s, order: s})
			}
		}
		if len(h) == 0 {
			h = append(h, &hnode{freq: 1, sym: 0})
		}
		heap.Init(&h)
		order := 256
		for h.Len() > 1 {
			a := heap.Pop(&h).(*hnode)
			b := heap.Pop(&h).(*hnode)
			order++
			heap.Push(&h, &hnode{freq: a.freq + b.freq, sym: -1, left: a, right: b, order: order})
		}
		root = h[0]
		ok := true
		var walk func(n *hnode, code uint32, depth int)
		walk = func(n *hnode, code uint32, depth int) {
			if n.sym >= 0 {
				codes[n.sym], lens[n.sym] = code, depth
				ok = ok && depth <= 16
				return
			}
			walk(n.left, code, depth+1)
			walk(n.right, code|1<<depth, depth+1)
		}
		walk(root, 0, 0)
		if ok {
			return codes, lens, root
		}
		for s := range f {
			if f[s] > 0 {
				f[s] = f[s]/2 + 1
			}
		}
	}
}

// writeTree: pre-order, internal then leaf bits.
func writeTree(w *bitWriter, n *hnode) {
	if n.sym >= 0 {
		w.write(1, 1)
		w.write(uint64(n.sym), 8)
		return
	}
	w.write(0, 1)
	writeTree(w, n.left)
	writeTree(w, n.right)
}

type bitWriter struct {
	buf  []byte
	nbit int
}

func (w *bitWriter) write(v uint64, n int) {
	for i := 0; i < n; i++ {
		if w.nbit/8 >= len(w.buf) {
			w.buf = append(w.buf, 0)
		}
		w.buf[w.nbit/8] |= byte(v>>i&1) << (w.nbit % 8)
		w.nbit++
	}
}

func (w *bitWriter) bytes() []byte { return w.buf }
