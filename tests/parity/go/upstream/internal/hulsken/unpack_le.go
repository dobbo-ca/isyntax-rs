//go:build !ppc64 && !s390x && !mips && !mips64

package hulsken

import "unsafe"

// put4 stores v's 16-bit lanes as r[0:4] in one write.
func put4(r []int16, v uint64) {
	_ = r[3]
	*(*uint64)(unsafe.Pointer(&r[0])) = v
}
