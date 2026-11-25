// Package hulsken decodes iSyntax codeblocks: the Huffman and zero-run
// entropy coder of compressors 16 (version 1) and 19 (version 2), the
// bit-plane unpack, the 4x4 reorder and the sign-magnitude conversion.
// See docs/format.md, "Entropy coding (Hulsken)".
package hulsken

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"
)

const fastBits = 11

// Scratch holds a decode's working memory; reuse it to avoid allocation.
type Scratch struct {
	msg   []byte
	fast  [1 << fastBits]uint16 // symbol | len<<8; 0 if longer
	long  []longCode
	code  [256]uint32
	size  [256]uint8
	inUse [256]bool
	blk   []int32 // 4x4 block starts, for blkBW
	blkBW int
	zero  []byte // stands in for absent planes
}

// blocks: each 4x4 block's first index.
func (s *Scratch) blocks(bw, area int) []int32 {
	if s.blkBW != bw || len(s.blk) != area/16 {
		s.blk, s.blkBW = s.blk[:0], bw
		for a := 0; a < area/16; a++ {
			s.blk = append(s.blk, int32(a%(bw/4)*4+a/(bw/4)*4*bw))
		}
	}
	return s.blk
}

type longCode struct {
	code uint32
	len  uint8
	sym  byte
}

// Decode decodes the codeblock src into dst: n planes (1 for LL, 3 for
// HL, LH, HH) of bw*bh coefficients in raster order. version is 1 for
// compressor 16 and 2 for compressor 19. bw must be a multiple of 4 and
// bw*bh a multiple of 16.
func Decode(dst []int16, src []byte, n, bw, bh, version int, s *Scratch) error {
	area := bw * bh
	dst = dst[:n*area]
	clear(dst)
	if len(src) <= 8 {
		return nil
	}
	planeBytes := area / 8
	var masks [3]uint16
	var serLen int
	r := bitReader{b: src}
	if version == 1 {
		serLen = int(binary.LittleEndian.Uint32(src))
		masks = [3]uint16{0xFFFF, 0xFFFF, 0xFFFF}
		r.pos = 32
	} else {
		total := 0
		for c := 0; c < n; c++ {
			masks[c] = binary.LittleEndian.Uint16(src[2*c:])
			total += bits.OnesCount16(masks[c])
		}
		serLen = total * planeBytes
		r.pos = 16 * n
	}
	if serLen > 4*n*area {
		return fmt.Errorf("hulsken: serialized length %d over %d", serLen, 4*n*area)
	}
	zeroSym := src[r.pos/8]
	counterBits := uint(src[r.pos/8+1])
	r.pos += 16
	if version == 2 {
		if serLen == 0 {
			return nil
		}
		var agg uint16
		for _, m := range masks[:n] {
			agg |= m
		}
		ptrBits := bits.Len(uint(serLen)) - 1 + 5
		r.pos += max(0, bits.OnesCount16(agg)-1) * ptrBits // bit-plane seek table, unused
	}
	if counterBits > 32 {
		return fmt.Errorf("hulsken: zero counter of %d bits", counterBits)
	}
	if err := s.readTree(&r); err != nil {
		return err
	}
	msg, got, err := s.message(&r, serLen, zeroSym, counterBits, version)
	if err != nil {
		return err
	}

	// Short message: rest reads as zero.
	if version == 1 && got < 16*n*planeBytes {
		if got < 2*n {
			return errors.New("hulsken: message shorter than its bitmasks")
		}
		t := msg[got-2*n:]
		for c := 0; c < n; c++ {
			masks[c] = binary.LittleEndian.Uint16(t[2*c:])
		}
	}
	planes := 0
	for _, m := range masks[:n] {
		planes += bits.OnesCount16(m)
	}
	if planes*planeBytes > len(msg) {
		return fmt.Errorf("hulsken: %d bit planes in a %d-byte message", planes, len(msg))
	}

	blk := s.blocks(bw, area)
	if len(s.zero) < planeBytes {
		s.zero = make([]byte, planeBytes)
	}
	// Each colour's planes by value bit.
	var pl [3][16][]byte
	p := 0
	if version == 1 {
		for c := 0; c < n; c++ {
			for m := masks[c]; m != 0; m &= m - 1 {
				b := bits.TrailingZeros16(m)
				t := 15
				if b > 0 {
					t = b - 1
				}
				pl[c][t] = msg[p*planeBytes : (p+1)*planeBytes]
				p++
			}
		}
	} else {
		for b := 0; b < 16; b++ {
			for c := 0; c < n; c++ {
				if masks[c]>>b&1 != 0 {
					pl[c][15-b] = msg[p*planeBytes : (p+1)*planeBytes]
					p++
				}
			}
		}
	}
	for c := 0; c < n; c++ {
		unpack(dst[c*area:(c+1)*area], bw, blk, &pl[c], s.zero[:planeBytes])
	}
	return nil
}

// unpack transposes a colour's bit planes into coefficients, one
// 4x4-pair byte at a time, and applies the bit 15 sign.
func unpack(d []int16, bw int, blk []int32, pl *[16][]byte, zero []byte) {
	var lo, hi bool
	for k, p := range pl {
		switch {
		case p == nil:
			pl[k] = zero
		case k < 8:
			lo = true
		default:
			hi = true
		}
	}
	n := len(zero)
	p0, p1, p2, p3, p4, p5, p6, p7 := pl[0][:n], pl[1][:n], pl[2][:n], pl[3][:n], pl[4][:n], pl[5][:n], pl[6][:n], pl[7][:n]
	q0, q1, q2, q3, q4, q5, q6, q7 := pl[8][:n], pl[9][:n], pl[10][:n], pl[11][:n], pl[12][:n], pl[13][:n], pl[14][:n], pl[15][:n]
	for j := 0; j < n; j++ {
		// Byte k: plane k's (or k+8's) byte j.
		var l, h uint64
		if lo {
			l = uint64(p0[j]) | uint64(p1[j])<<8 | uint64(p2[j])<<16 | uint64(p3[j])<<24 |
				uint64(p4[j])<<32 | uint64(p5[j])<<40 | uint64(p6[j])<<48 | uint64(p7[j])<<56
		}
		if hi {
			h = uint64(q0[j]) | uint64(q1[j])<<8 | uint64(q2[j])<<16 | uint64(q3[j])<<24 |
				uint64(q4[j])<<32 | uint64(q5[j])<<40 | uint64(q6[j])<<48 | uint64(q7[j])<<56
		}
		if l|h == 0 {
			continue
		}
		l, h = transpose8(l), transpose8(h)
		base := int(blk[j>>1]) + j&1*2*bw
		put4(d[base:base+4:base+4], sign(spread(uint32(l))|spread(uint32(h))<<8))
		put4(d[base+bw:base+bw+4:base+bw+4], sign(spread(uint32(l>>32))|spread(uint32(h>>32))<<8))
	}
}

// transpose8 swaps bit i of byte k with bit k of byte i.
func transpose8(x uint64) uint64 {
	t := (x ^ x>>7) & 0x00AA00AA00AA00AA
	x ^= t ^ t<<7
	t = (x ^ x>>14) & 0x0000CCCC0000CCCC
	x ^= t ^ t<<14
	t = (x ^ x>>28) & 0x00000000F0F0F0F0
	x ^= t ^ t<<28
	return x
}

// spread puts byte k of x in 16-bit lane k.
func spread(x uint32) uint64 {
	v := uint64(x)
	v = (v | v<<16) & 0x0000FFFF0000FFFF
	return (v | v<<8) & 0x00FF00FF00FF00FF
}

// sign: lanes with bit 15 set become -(v & 0x7FFF), per lane.
func sign(r uint64) uint64 {
	const h, l = 0x8000800080008000, 0x0001000100010001
	s := (r & h) >> 15 * 0xFFFF
	t := r&^h ^ s
	return (t&^h + s&l) ^ t&h // carry-free lane add
}

// readTree: mirrors writeTree's pre-order bits.
func (s *Scratch) readTree(r *bitReader) error {
	s.fast = [1 << fastBits]uint16{}
	s.long = s.long[:0]
	s.inUse = [256]bool{}
	var code uint32
	depth := 0
	for {
		if r.pos >= 8*len(r.b) {
			return errors.New("hulsken: Huffman tree runs past the codeblock")
		}
		blob := r.peek()
		for blob&1 == 0 {
			depth++
			if depth > 16 {
				return errors.New("hulsken: Huffman code over 16 bits")
			}
			blob >>= 1
			r.pos++
		}
		sym := byte(blob >> 1)
		r.pos += 9
		s.code[sym], s.size[sym], s.inUse[sym] = code, uint8(depth), true
		if depth <= fastBits {
			e := uint16(sym) | uint16(max(depth, 1))<<8
			for i := uint32(0); i < 1<<(fastBits-depth); i++ {
				s.fast[i<<depth|code] = e
			}
		} else {
			if len(s.long) == 256 {
				return errors.New("hulsken: too many Huffman codes")
			}
			s.long = append(s.long, longCode{code, uint8(depth), sym})
		}
		// Next: drop trailing ones, take one.
		for depth > 0 && code>>(depth-1)&1 == 1 {
			depth--
			code &^= 1 << depth
		}
		if depth == 0 {
			return nil
		}
		code |= 1 << (depth - 1)
	}
}

func (s *Scratch) symbol(r *bitReader) (byte, error) {
	blob := r.peek()
	if e := s.fast[blob&(1<<fastBits-1)]; e != 0 {
		r.pos += int(e >> 8)
		return byte(e), nil
	}
	for _, l := range s.long {
		if uint32(blob)&(1<<l.len-1) == l.code {
			r.pos += int(l.len)
			return l.sym, nil
		}
	}
	return 0, errors.New("hulsken: unknown Huffman code")
}

// message: zero-run decode, zero-fills the rest.
func (s *Scratch) message(r *bitReader, serLen int, zeroSym byte, counterBits uint, version int) ([]byte, int, error) {
	if cap(s.msg) < serLen {
		s.msg = make([]byte, serLen)
	}
	msg := s.msg[:serLen]
	end := 8 * len(r.b)
	zCode, zLen := s.code[zeroSym], uint(max(s.size[zeroSym], 1))
	if !s.inUse[zeroSym] {
		zCode, zLen = 0, 1
	}
	counterMask := uint64(1)<<counterBits - 1
	fast := &s.fast
	n := 0
	for n < serLen && r.pos < end {
		// Plain symbols: 5 x 11 bits fit a peek.
		blob, got, pos := r.peek(), 0, r.pos
		for ; got < 5 && n < serLen && pos < end; got++ {
			e := fast[blob&(1<<fastBits-1)]
			if e == 0 || byte(e) == zeroSym {
				break
			}
			msg[n] = byte(e)
			n++
			pos += int(e >> 8)
			blob >>= e >> 8
		}
		r.pos = pos
		if got > 0 {
			continue
		}
		sym, err := s.symbol(r)
		if err != nil {
			return nil, 0, err
		}
		if sym != zeroSym {
			msg[n] = sym
			n++
			continue
		}
		cnt := r.peek() & counterMask
		r.pos += int(counterBits)
		if cnt == 0 {
			msg[n] = sym // escaped zero-run symbol
			n++
			continue
		}
		run := func() int {
			if version == 2 {
				return int(cnt) + 1
			}
			return int(cnt)
		}
		if n+run() < serLen && r.pos < end {
			// Following zero-run codes extend the counter.
			for {
				blob := r.peek()
				if uint32(blob)&(1<<zLen-1) != zCode {
					break
				}
				cnt = cnt<<counterBits | blob>>zLen&counterMask
				r.pos += int(zLen + counterBits)
				if cnt > 1<<31 || n+run() >= serLen || r.pos >= end {
					break
				}
			}
		}
		k := min(run(), serLen-n)
		clear(msg[n : n+k])
		n += k
	}
	clear(msg[n:])
	return msg, n, nil
}

// bitReader: LSB-first; past-end reads as 0.
type bitReader struct {
	b   []byte
	pos int
}

// peek returns >=57 bits from pos.
func (r *bitReader) peek() uint64 {
	i := r.pos >> 3
	if i+8 <= len(r.b) {
		return binary.LittleEndian.Uint64(r.b[i:]) >> (r.pos & 7)
	}
	return r.peekTail()
}

// peekTail: peek near the end, kept out of line.
func (r *bitReader) peekTail() uint64 {
	i := r.pos >> 3
	var v uint64
	for k := 0; k < 8 && i+k < len(r.b); k++ {
		v |= uint64(r.b[i+k]) << (8 * k)
	}
	return v >> (r.pos & 7)
}
