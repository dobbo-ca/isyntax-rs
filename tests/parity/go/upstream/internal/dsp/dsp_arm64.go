//go:build !purego

package dsp

// lanes: NEON's widest signed-16 width.
const lanes = 8

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
	if noSIMD || n < lanes || n+1 > len(s.hx) {
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
		interleaveNEON(&out[0], &s.ex[0], &s.dx[1], m)
	}
	for k := m; k < n; k++ {
		out[2*k], out[2*k+1] = s.ex[k], s.dx[k+1]
	}
}

// NEON over full lanes, Go tail.
func liftOddSIMD(out, l, h0, h1 []int16) {
	n := len(out)
	l, h0, h1 = l[:n], h0[:n], h1[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		liftOddNEON(&out[0], &l[0], &h0[0], &h1[0], m)
	}
	liftOddGo(out[m:], l[m:], h0[m:], h1[m:])
}

func liftEvenSIMD(out, h, o0, o1 []int16) {
	n := len(out)
	h, o0, o1 = h[:n], o0[:n], o1[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		liftEvenNEON(&out[0], &h[0], &o0[0], &o1[0], m)
	}
	liftEvenGo(out[m:], h[m:], o0[m:], o1[m:])
}

// Handles only signed Y, shift <=1.
func ycocgSIMD(dst []byte, y, co, cg []int16, shift uint, yAbs bool) {
	n := len(y)
	if yAbs || shift > 1 {
		ycocgGo(dst, y, co, cg, shift, yAbs)
		return
	}
	dst, co, cg = dst[:4*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		ycocgNEON(&dst[0], &y[0], &co[0], &cg[0], m, shift == 1)
	}
	ycocgGo(dst[4*m:], y[m:], co[m:], cg[m:], shift, yAbs)
}

// Handles only signed Y.
func ycocgSamplesSIMD(dst, y, co, cg []int16, hi int16, yAbs bool) {
	n := len(y)
	if yAbs {
		ycocgSamplesGo(dst, y, co, cg, hi, yAbs)
		return
	}
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesNEON(&dst[0], &y[0], &co[0], &cg[0], m, 0, hi, false)
	}
	ycocgSamplesGo(dst[3*m:], y[m:], co[m:], cg[m:], hi, yAbs)
}

func planesSamplesSIMD(dst, r, g, b []int16, hi int16, _ bool) {
	n := len(r)
	dst, g, b = dst[:3*n], g[:n], b[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesNEON(&dst[0], &r[0], &g[0], &b[0], m, 0, hi, true)
	}
	planesSamplesGo(dst[3*m:], r[m:], g[m:], b[m:], hi, false)
}

// Handles only signed Y; lo, hi make the clamp a no-op.
func ycocgRGB16SIMD(dst, y, co, cg []int16, yAbs bool) {
	n := len(y)
	if yAbs {
		ycocgRGB16Go(dst, y, co, cg, yAbs)
		return
	}
	dst, co, cg = dst[:3*n], co[:n], cg[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesNEON(&dst[0], &y[0], &co[0], &cg[0], m, -32768, 32767, false)
	}
	ycocgRGB16Go(dst[3*m:], y[m:], co[m:], cg[m:], yAbs)
}

func planesRGB16SIMD(dst, r, g, b []int16, _ bool) {
	n := len(r)
	dst, g, b = dst[:3*n], g[:n], b[:n]
	m := n &^ (lanes - 1)
	if m > 0 {
		samplesNEON(&dst[0], &r[0], &g[0], &b[0], m, -32768, 32767, true)
	}
	planesRGB16Go(dst[3*m:], r[m:], g[m:], b[m:], false)
}

//go:noescape
func liftOddNEON(out, l, h0, h1 *int16, n int)

//go:noescape
func liftEvenNEON(out, h, o0, o1 *int16, n int)

//go:noescape
func interleaveNEON(out, e, d *int16, n int)

//go:noescape
func ycocgNEON(dst *byte, y, co, cg *int16, n int, shift1 bool)

//go:noescape
func samplesNEON(dst, a, b, c *int16, n int, lo, hi int16, planes bool)

func truncSIMD(row []int16, mask, o int16) {
	m := len(row) &^ (lanes - 1)
	if m > 0 {
		truncNEON(&row[0], m, mask, o)
	}
	truncGo(row[m:], mask, o)
}

//go:noescape
func truncNEON(p *int16, n int, mask, o int16)
