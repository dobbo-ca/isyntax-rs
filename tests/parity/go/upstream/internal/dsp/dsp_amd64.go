//go:build !purego

package dsp

// lanes: AVX2's widest signed-16 width.
const lanes = 16

// useAVX2: false forces the Go fallback.
var useAVX2 = !noSIMD && hasAVX2()

// Variables so ISYNTAX_NOSIMD can select the Go kernels.
var (
	liftOdd         = pickKernel(liftOddSIMD, liftOddGo)
	liftEven        = pickKernel(liftEvenSIMD, liftEvenGo)
	YCoCgToRGBA     = pickKernel(ycocgSIMD, ycocgGo)
	YCoCgToRGB16    = pickKernel(ycocgRGB16SIMD, ycocgRGB16Go)
	PlanesToRGB16   = pickKernel(planesRGB16SIMD, planesRGB16Go)
	YCoCgToSamples  = pickKernel(ycocgSamplesSIMD, ycocgSamplesGo)
	PlanesToSamples = pickKernel(planesSamplesSIMD, planesSamplesGo)
	TruncRow        = pickKernel(truncSIMD, truncGo)
)

func idwtRow(out, l, h []int16, s *RowScratch) {
	n := len(l)
	if !useAVX2 || n < lanes || n+1 > len(s.hx) {
		idwtRowGo(out, l, h)
		return
	}
	h, out = h[:n], out[:2*n]
	copy(s.hx[:n], h)
	s.hx[n] = h[n-1]
	liftOdd(s.dx[1:n+1], l, s.hx[:n], s.hx[1:n+1])
	s.dx[0] = s.dx[1]
	liftEven(s.ex[:n], h, s.dx[:n], s.dx[1:n+1])
	m := n &^ (lanes - 1)
	if m > 0 {
		interleaveAVX2(&out[0], &s.ex[0], &s.dx[1], m)
	}
	for k := m; k < n; k++ {
		out[2*k], out[2*k+1] = s.ex[k], s.dx[k+1]
	}
}

// AVX2 over full lanes, Go tail.
func liftOddSIMD(out, l, h0, h1 []int16) {
	n := len(out)
	l, h0, h1 = l[:n], h0[:n], h1[:n]
	if !useAVX2 {
		liftOddGo(out, l, h0, h1)
		return
	}
	m := n &^ (lanes - 1)
	if m > 0 {
		liftOddAVX2(&out[0], &l[0], &h0[0], &h1[0], m)
	}
	liftOddGo(out[m:], l[m:], h0[m:], h1[m:])
}

func liftEvenSIMD(out, h, o0, o1 []int16) {
	n := len(out)
	h, o0, o1 = h[:n], o0[:n], o1[:n]
	if !useAVX2 {
		liftEvenGo(out, h, o0, o1)
		return
	}
	m := n &^ (lanes - 1)
	if m > 0 {
		liftEvenAVX2(&out[0], &h[0], &o0[0], &o1[0], m)
	}
	liftEvenGo(out[m:], h[m:], o0[m:], o1[m:])
}

// Handles only signed Y, shift <=1.
func ycocgSIMD(dst []byte, y, co, cg []int16, shift uint, yAbs bool) {
	n := len(y)
	if !useAVX2 || yAbs || shift > 1 {
		ycocgGo(dst, y, co, cg, shift, yAbs)
		return
	}
	dst, co, cg = dst[:4*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		ycocgAVX2(&dst[0], &y[0], &co[0], &cg[0], m, shift == 1)
	}
	ycocgGo(dst[4*m:], y[m:], co[m:], cg[m:], shift, yAbs)
}

// Handles only signed Y.
func ycocgSamplesSIMD(dst, y, co, cg []int16, hi int16, yAbs bool) {
	n := len(y)
	if !useAVX2 || yAbs {
		ycocgSamplesGo(dst, y, co, cg, hi, yAbs)
		return
	}
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesAVX2(&dst[0], &y[0], &co[0], &cg[0], m, 0, hi, false)
	}
	ycocgSamplesGo(dst[3*m:], y[m:], co[m:], cg[m:], hi, yAbs)
}

func planesSamplesSIMD(dst, r, g, b []int16, hi int16, _ bool) {
	n := len(r)
	if !useAVX2 {
		planesSamplesGo(dst, r, g, b, hi, false)
		return
	}
	dst, g, b = dst[:3*n], g[:n], b[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesAVX2(&dst[0], &r[0], &g[0], &b[0], m, 0, hi, true)
	}
	planesSamplesGo(dst[3*m:], r[m:], g[m:], b[m:], hi, false)
}

// Handles only signed Y; lo, hi make the clamp a no-op.
func ycocgRGB16SIMD(dst, y, co, cg []int16, yAbs bool) {
	n := len(y)
	if !useAVX2 || yAbs {
		ycocgRGB16Go(dst, y, co, cg, yAbs)
		return
	}
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesAVX2(&dst[0], &y[0], &co[0], &cg[0], m, -32768, 32767, false)
	}
	ycocgRGB16Go(dst[3*m:], y[m:], co[m:], cg[m:], yAbs)
}

func planesRGB16SIMD(dst, r, g, b []int16, _ bool) {
	n := len(r)
	if !useAVX2 {
		planesRGB16Go(dst, r, g, b, false)
		return
	}
	dst, g, b = dst[:3*n], g[:n], b[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesAVX2(&dst[0], &r[0], &g[0], &b[0], m, -32768, 32767, true)
	}
	planesRGB16Go(dst[3*m:], r[m:], g[m:], b[m:], false)
}

func hasAVX2() bool {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(1<<27) == 0 || ecx1&(1<<28) == 0 { // OSXSAVE, AVX
		return false
	}
	if xcr0, _ := xgetbv(); xcr0&6 != 6 { // OS saves XMM and YMM state
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0 // AVX2
}

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

func xgetbv() (eax, edx uint32)

//go:noescape
func liftOddAVX2(out, l, h0, h1 *int16, n int)

//go:noescape
func liftEvenAVX2(out, h, o0, o1 *int16, n int)

//go:noescape
func interleaveAVX2(out, e, d *int16, n int)

//go:noescape
func ycocgAVX2(dst *byte, y, co, cg *int16, n int, shift1 bool)

//go:noescape
func samplesAVX2(dst, a, b, c *int16, n int, lo, hi int16, planes bool)

func truncSIMD(row []int16, mask, o int16) {
	if !useAVX2 {
		truncGo(row, mask, o)
		return
	}
	m := len(row) &^ (lanes - 1)
	if m > 0 {
		truncAVX2(&row[0], m, mask, o)
	}
	truncGo(row[m:], mask, o)
}

//go:noescape
func truncAVX2(p *int16, n int, mask, o int16)
