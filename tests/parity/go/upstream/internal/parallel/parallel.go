// Package parallel spreads decode work over goroutines and hands a
// worker's panic back to the caller as an error.
package parallel

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// For calls fn(i) for every i in [0,n) from up to GOMAXPROCS goroutines,
// the caller's included. After the first error no new items start; For
// returns it once every worker has stopped.
func For(n int, fn func(i int) error) error {
	nw := max(1, min(runtime.GOMAXPROCS(0), n))
	var (
		next  atomic.Int64
		stop  atomic.Bool
		once  sync.Once
		first error
		wg    sync.WaitGroup
	)
	fail := func(err error) {
		once.Do(func() { first = err })
		stop.Store(true)
	}
	run := func() {
		// Else a panic kills the process.
		defer func() {
			if v := recover(); v != nil {
				fail(fmt.Errorf("isyntax: internal error: panic: %v", v))
			}
		}()
		for !stop.Load() {
			i := int(next.Add(1)) - 1
			if i >= n {
				return
			}
			if err := fn(i); err != nil {
				fail(err)
				return
			}
		}
	}
	wg.Add(nw - 1)
	for w := 1; w < nw; w++ {
		go func() {
			defer wg.Done()
			run()
		}()
	}
	run()
	wg.Wait()
	return first
}
