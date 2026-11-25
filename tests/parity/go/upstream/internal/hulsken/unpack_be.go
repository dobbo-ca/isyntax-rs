//go:build ppc64 || s390x || mips || mips64

package hulsken

// put4 stores v's 16-bit lanes, lane 0 lowest, as r[0:4].
func put4(r []int16, v uint64) {
	r[0], r[1], r[2], r[3] = int16(v), int16(v>>16), int16(v>>32), int16(v>>48)
}
