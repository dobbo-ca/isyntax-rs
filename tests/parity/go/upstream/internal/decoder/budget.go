package decoder

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
)

// budget: process-wide cache and scratch limit.
var budget struct {
	mu       sync.Mutex
	explicit bool  // limit was set by SetMemoryLimit
	limit    int64 // math.MaxInt64 for none
	used     int64
}

// ErrMemoryLimit is isyntax.ErrMemoryLimit.
var ErrMemoryLimit = errors.New("isyntax: memory limit held by other decodes (transient, retry later)")

var errTooBig = errors.New("isyntax: decode needs more than the memory limit")

// SetMemoryLimit is isyntax.SetMemoryLimit.
func SetMemoryLimit(n int64) int64 {
	if n <= 0 {
		n = math.MaxInt64
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	prev := limitNow()
	budget.explicit, budget.limit = true, n
	return prev
}

// limitNow: byte limit; budget.mu held.
func limitNow() int64 {
	if budget.explicit {
		return budget.limit
	}
	return defaultLimit()
}

// currentLimit: limitNow without the lock held.
func currentLimit() int64 {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return limitNow()
}

// reserve charges n bytes, or fails.
func reserve(n int64) error {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	l := limitNow()
	if n > l {
		return errTooBig
	}
	if budget.used > l-n {
		return ErrMemoryLimit
	}
	budget.used += n
	return nil
}

func release(n int64) {
	budget.mu.Lock()
	budget.used -= n
	budget.mu.Unlock()
}

// defaultLimit: half runtime limit, else cgroup.
func defaultLimit() int64 {
	if l := debug.SetMemoryLimit(-1); l < math.MaxInt64 {
		return max(1, l/2)
	}
	if runtime.GOOS == "linux" {
		if l := cgroupMax(); l > 0 {
			return max(1, l/2)
		}
	}
	return math.MaxInt64
}

// cgroupMax: the cgroup memory limit, cached.
var cgroupMax = sync.OnceValue(func() int64 {
	root := "/sys/fs/cgroup"
	if b, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if p, ok := strings.CutPrefix(line, "0::"); ok {
				if l := cgroupLimit(filepath.Join(root, p)); l > 0 {
					return l
				}
			}
		}
	}
	return cgroupLimit(root)
})

func cgroupLimit(dir string) int64 {
	for _, f := range []string{"memory.max", "memory/memory.limit_in_bytes"} {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			continue
		}
		// v2 says max; v1 near 2^63.
		n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err == nil && n > 0 && n < 1<<50 {
			return n
		}
	}
	return 0
}
