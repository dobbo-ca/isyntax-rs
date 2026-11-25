// The module path is under isyntax-go's so that this program may import its internal packages
// (fixture, container): Go allows that for any path that has the parent of `internal` as a prefix.
module github.com/dobbo-ca/isyntax-go/parity-oracle

go 1.27

require github.com/dobbo-ca/isyntax-go v0.0.0

replace github.com/dobbo-ca/isyntax-go => ../upstream
