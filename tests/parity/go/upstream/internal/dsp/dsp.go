// Package dsp holds the sample-level stages of a decode: the inverse
// LeGall 5/3 wavelet transform, YCoCg to RGB and RGB to YCbCr. Each hot
// loop (idwtRow, liftOdd, liftEven, YCoCgToRGBA, YCoCgToRGB16,
// PlanesToRGB16, YCoCgToSamples, PlanesToSamples) has a Go reference and
// NEON/AVX2 kernels in dsp_arm64.s/dsp_amd64.s (dsp_generic.go on other
// arches and under -tags purego); RGBAToYCbCr stays Go-only.
package dsp

import "image/color"

// RowScratch holds idwtRow's per-row working set (hx, dx, ex from
// format.md's row derivation), sized for the widest row this decoder
// makes. Callers own one (e.g. in their pooled decode scratch) and pass
// it to Inverse53, so a row pass never allocates.
type RowScratch struct{ hx, dx, ex [512]int16 }

// Inverse53 inverts one level of the 2D LeGall 5/3 transform in place.
// buf is 2n x 2n with the quadrants LL | HL over LH | HH; tmp is scratch
// of the same size, rs is the row pass's scratch (see RowScratch). Rows
// go first, then columns, each with the lifting in docs/format.md:
// int32 arithmetic, every output stored as int16 before later steps
// read it.
func Inverse53(buf, tmp []int16, n int, rs *RowScratch) {
	w := 2 * n
	buf, tmp = buf[:w*w], tmp[:w*w]
	for y := 0; y < w; y++ {
		row := buf[y*w : (y+1)*w]
		idwtRow(tmp[y*w:(y+1)*w], row[:n], row[n:], rs)
	}
	// Columns: same lift, transposed direction.
	L := func(k int) []int16 { return tmp[k*w : (k+1)*w] }
	H := func(k int) []int16 { return tmp[(n+k)*w : (n+k+1)*w] }
	out := func(k int) []int16 { return buf[k*w : (k+1)*w] }
	for k := 0; k < n-1; k++ {
		liftOdd(out(2*k+1), L(k), H(k), H(k+1))
	}
	liftOdd(out(2*n-1), L(n-1), H(n-1), H(n-1))
	liftEven(out(0), H(0), out(1), out(1))
	for k := 1; k < n; k++ {
		liftEven(out(2*k), H(k), out(2*k-1), out(2*k+1))
	}
}

// idwtRowGo: 1D inverse lift, one row.
func idwtRowGo(out, l, h []int16) {
	n := len(l)
	h, out = h[:n], out[:2*n]
	dc := int32(int16(int32(l[0]) - (int32(h[0])+int32(h[1])+2)>>2))
	out[0] = int16(int32(h[0]) + dc)
	for k := 1; k < n-1; k++ {
		hk := h[k : k+2 : k+2]
		o := out[2*k-1 : 2*k+1 : 2*k+1]
		dn := int32(int16(int32(l[k]) - (int32(hk[0])+int32(hk[1])+2)>>2))
		o[0] = int16(dc)
		o[1] = int16(int32(hk[0]) + (dc+dn)>>1)
		dc = dn
	}
	dn := int32(int16(int32(l[n-1]) - (int32(h[n-1])+1)>>1))
	out[2*n-3] = int16(dc)
	out[2*n-2] = int16(int32(h[n-1]) + (dc+dn)>>1)
	out[2*n-1] = int16(dn)
}

// liftOddGo: out = l - ((h0+h1+2)>>2).
func liftOddGo(out, l, h0, h1 []int16) {
	l, h0, h1 = l[:len(out)], h0[:len(out)], h1[:len(out)]
	for i := range out {
		out[i] = int16(int32(l[i]) - (int32(h0[i])+int32(h1[i])+2)>>2)
	}
}

// liftEvenGo: out = h + ((o0+o1)>>1).
func liftEvenGo(out, h, o0, o1 []int16) {
	h, o0, o1 = h[:len(out)], o0[:len(out)], o1[:len(out)]
	for i := range out {
		out[i] = int16(int32(h[i]) + (int32(o0[i])+int32(o1[i]))>>1)
	}
}

// ycocgGo: the transform, docs/format.md#colour-transform.
func ycocgGo(dst []byte, y, co, cg []int16, shift uint, yAbs bool) {
	n := len(y)
	dst, co, cg = dst[:4*n], co[:n], cg[:n]
	shift &= 15
	for i, yv := range y {
		if yAbs {
			m := yv >> 15
			yv = (yv ^ m - m) & 0x7FFF
		}
		c, g := co[i], cg[i]
		t := yv - g>>1
		G := t + g
		B := t - c>>1
		R := B + c
		d := dst[4*i : 4*i+4 : 4*i+4]
		d[0] = clamp(R >> shift)
		d[1] = clamp(G >> shift)
		d[2] = clamp(B >> shift)
		d[3] = 255
	}
}

// ycocgRGB16Go converts len(y) pixels to signed, unshifted, unclamped
// R, G, B int16 triples: the parity test's oracle comparison needs the
// values before YCoCgToRGBA's 8-bit shift and clamp.
func ycocgRGB16Go(dst, y, co, cg []int16, yAbs bool) {
	n := len(y)
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	for i, yv := range y {
		if yAbs {
			m := yv >> 15
			yv = (yv ^ m - m) & 0x7FFF
		}
		c, g := co[i], cg[i]
		t := yv - g>>1
		G := t + g
		B := t - c>>1
		R := B + c
		dst[3*i], dst[3*i+1], dst[3*i+2] = R, G, B
	}
}

// PlanesToRGBA is YCoCgToRGBA for files whose colour planes are
// already R, G, B (docs/format.md, Colour transform): shift and clamp
// only. yAbs is ignored.
func PlanesToRGBA(dst []byte, r, g, b []int16, shift uint, _ bool) {
	n := len(r)
	dst, g, b = dst[:4*n], g[:n], b[:n]
	shift &= 15
	for i := range r {
		d := dst[4*i : 4*i+4 : 4*i+4]
		d[0], d[1], d[2], d[3] = clamp(r[i]>>shift), clamp(g[i]>>shift), clamp(b[i]>>shift), 255
	}
}

// planesRGB16Go is ycocgRGB16Go for R, G, B planes: it interleaves
// them unchanged. yAbs is ignored.
func planesRGB16Go(dst, r, g, b []int16, _ bool) {
	n := len(r)
	dst, g, b = dst[:3*n], g[:n], b[:n]
	for i := range r {
		dst[3*i], dst[3*i+1], dst[3*i+2] = r[i], g[i], b[i]
	}
}

// ycocgSamplesGo: YCoCgToRGB16, each value clamped to [0, hi].
func ycocgSamplesGo(dst, y, co, cg []int16, hi int16, yAbs bool) {
	n := len(y)
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	for i, yv := range y {
		if yAbs {
			m := yv >> 15
			yv = (yv ^ m - m) & 0x7FFF
		}
		c, g := co[i], cg[i]
		t := yv - g>>1
		G := t + g
		B := t - c>>1
		R := B + c
		dst[3*i], dst[3*i+1], dst[3*i+2] = min(max(R, 0), hi), min(max(G, 0), hi), min(max(B, 0), hi)
	}
}

// planesSamplesGo: PlanesToRGB16, clamped to [0, hi].
func planesSamplesGo(dst, r, g, b []int16, hi int16, _ bool) {
	n := len(r)
	dst, g, b = dst[:3*n], g[:n], b[:n]
	for i := range r {
		dst[3*i], dst[3*i+1], dst[3*i+2] = min(max(r[i], 0), hi), min(max(g[i], 0), hi), min(max(b[i], 0), hi)
	}
}

func clamp(v int16) byte {
	if uint16(v) > 255 {
		return byte(^(v >> 15)) // 0 below, 255 above
	}
	return byte(v)
}

// RGBAToYCbCr converts len(y) RGBA pixels with color.RGBToYCbCr
// (full-range BT.601).
func RGBAToYCbCr(y, cb, cr, rgba []byte) {
	n := len(y)
	cb, cr, rgba = cb[:n], cr[:n], rgba[:4*n]
	for i := range y {
		y[i], cb[i], cr[i] = color.RGBToYCbCr(rgba[4*i], rgba[4*i+1], rgba[4*i+2])
	}
}

// truncGo: |v| &^ mask, +o if nonzero, sign restored; int16 wraparound.
func truncGo(row []int16, mask, o int16) {
	for i, v := range row {
		a := max(v, -v) &^ mask
		if a != 0 {
			a += o
		}
		if v < 0 {
			a = -a
		}
		row[i] = a
	}
}
