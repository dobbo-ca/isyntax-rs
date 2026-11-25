//go:build (amd64 || arm64) && !purego

package dsp

import "os"

// noSIMD: force every kernel's Go version.
var noSIMD = os.Getenv("ISYNTAX_NOSIMD") != ""

func pickKernel[F any](simd, goKernel F) F {
	if noSIMD {
		return goKernel
	}
	return simd
}
