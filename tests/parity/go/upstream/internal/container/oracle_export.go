package container

// Not part of upstream: exported entries to two unexported parsers, for the parity oracle's `.b64`
// and `.ct` inputs (the seeds of FuzzDecodeBase64 and FuzzParseClusterTable).

// OracleDecodeBase64 is decodeBase64.
func OracleDecodeBase64(s string) ([]byte, error) { return decodeBase64(s) }

// OracleClusterTable is FuzzParseClusterTable's call: b against its two fixed templates; it returns
// the number of codeblocks.
func OracleClusterTable(b []byte) (int, error) {
	tmpl := []clusterTemplate{{rel: []relCoord{
		{x: 1, y: 2, colour: 0, scale: 1, coeff: 1},
		{x: 3, y: 4, colour: 1, scale: 1, coeff: 3},
	}}}
	cb, err := parseClusterTable(b, tmpl)
	return len(cb), err
}
