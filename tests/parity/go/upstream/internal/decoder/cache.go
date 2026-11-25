package decoder

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
)

// cache: sharded, budget-charged, computed once.
type cache struct {
	shards [nShards]shard
	max    int64 // bytes
	bytes  atomic.Int64
	clock  atomic.Uint64 // last-use stamps
	free   freeList
}

const nShards = 64

type ckey struct {
	kind    uint8 // kindCoef, kindRecon or an edge
	v       uint8 // recon variant, trunc.go
	s, x, y int32
}

const (
	kindCoef = iota
	kindRecon
	kindCoefEdge
	kindReconEdge
)

type centry struct {
	key        ckey
	done       chan struct{}
	ready      bool   // data set, listed; shard lock held
	used       uint64 // last use; shard lock held
	err        error
	data       []int16
	refs       int32   // pinned readers; shard lock held
	evicted    bool    // unlinked, awaiting refs==0; shard lock held
	uses       int32   // centre uses; shard lock held
	prev, next *centry // LRU ring; sentinel then newest
}

type shard struct {
	mu  sync.Mutex
	m   map[ckey]*centry
	lru centry // sentinel
}

// freeList: bounded recycling.
type freeList struct {
	mu     sync.Mutex
	bySize map[int][][]int16
}

const freeListCap = 8 // buffers kept per size

func (f *freeList) get(n int) []int16 {
	f.mu.Lock()
	defer f.mu.Unlock()
	lst := f.bySize[n]
	if len(lst) == 0 {
		return nil
	}
	data := lst[len(lst)-1]
	f.bySize[n] = lst[:len(lst)-1]
	return data
}

func (f *freeList) put(data []int16) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.bySize == nil {
		f.bySize = make(map[int][][]int16)
	}
	if len(f.bySize[len(data)]) < freeListCap {
		f.bySize[len(data)] = append(f.bySize[len(data)], data)
	}
}

func newCache(max int64) *cache {
	c := &cache{max: max, free: freeList{bySize: make(map[int][][]int16)}}
	for i := range c.shards {
		s := &c.shards[i]
		s.m = make(map[ckey]*centry)
		s.lru.prev, s.lru.next = &s.lru, &s.lru
	}
	return c
}

func (c *cache) shard(k ckey) *shard {
	h := uint32(k.x)*73856093 ^ uint32(k.y)*19349663 ^ uint32(k.s)*83492791 ^ uint32(k.kind) ^ uint32(k.v)<<4
	return &c.shards[h%nShards]
}

func (s *shard) unlink(e *centry) {
	e.prev.next, e.next.prev = e.next, e.prev
	e.prev, e.next = nil, nil
}

func (s *shard) pushFront(e *centry) {
	e.prev, e.next = &s.lru, s.lru.next
	e.prev.next, e.next.prev = e, e
}

// pin: get's release, alloc-free by design.
type pin struct {
	c *cache
	s *shard
	e *centry
}

// release unpins e; recycles if evicted.
func (p pin) release() {
	if p.e == nil {
		return
	}
	p.s.mu.Lock()
	p.e.refs--
	refs, evicted, data := p.e.refs, p.e.evicted, p.e.data
	p.s.mu.Unlock()
	if evicted && refs == 0 {
		p.c.free.put(data)
	}
}

// used: true at the nth use.
func (p pin) used(n int32) bool {
	p.s.mu.Lock()
	defer p.s.mu.Unlock()
	p.e.uses++
	return p.e.uses == n
}

// demote: evict this entry next.
func (p pin) demote() {
	p.s.mu.Lock()
	defer p.s.mu.Unlock()
	if p.e.evicted {
		return
	}
	p.s.unlink(p.e)
	p.e.used = 0
	p.e.prev, p.e.next = p.s.lru.prev, &p.s.lru
	p.e.prev.next, p.e.next.prev = p.e, p.e
}

// get returns k's pinned entry.
func (c *cache) get(ctx context.Context, k ckey, n int, fill func([]int16) error) ([]int16, pin, error) {
	s := c.shard(k)
	for {
		s.mu.Lock()
		e := s.m[k]
		if e != nil && e.ready {
			e.used = c.clock.Add(1)
			e.refs++
			s.unlink(e)
			s.pushFront(e)
			s.mu.Unlock()
			return e.data, pin{c, s, e}, nil
		}
		if e != nil {
			s.mu.Unlock()
			select {
			case <-e.done:
			case <-ctx.Done():
				return nil, pin{}, ctx.Err()
			}
			if e.err != nil {
				// Its creator gave up; we retry.
				if (errors.Is(e.err, context.Canceled) || errors.Is(e.err, context.DeadlineExceeded)) && ctx.Err() == nil {
					continue
				}
				return nil, pin{}, e.err
			}
			s.mu.Lock()
			if e.evicted {
				s.mu.Unlock()
				continue // recycled before we pinned it; refill
			}
			e.refs++
			s.mu.Unlock()
			return e.data, pin{c, s, e}, nil
		}
		e = &centry{key: k, done: make(chan struct{})}
		s.m[k] = e
		s.mu.Unlock()
		return c.fill(s, e, n, fill)
	}
}

func (c *cache) fill(s *shard, e *centry, n int, fill func([]int16) error) ([]int16, pin, error) {
	size := int64(2 * n)
	if err := c.reserve(size); err != nil {
		s.mu.Lock()
		delete(s.m, e.key)
		e.err = err
		s.mu.Unlock()
		close(e.done)
		return nil, pin{}, err
	}
	data := c.free.get(n)
	if data == nil {
		data = make([]int16, n)
	}
	err := fill(data)
	s.mu.Lock()
	if err != nil {
		delete(s.m, e.key)
		e.err = err
	} else {
		e.data, e.ready, e.used, e.refs = data, true, c.clock.Add(1), 1
		s.pushFront(e)
	}
	s.mu.Unlock()
	close(e.done)
	if err != nil {
		release(size)
		return nil, pin{}, err
	}
	c.bytes.Add(size)
	for c.bytes.Load() > c.max && c.evictOldest(e) {
	}
	return data, pin{c, s, e}, nil
}

// evictOldest drops LRU entry, excluding keep.
func (c *cache) evictOldest(keep *centry) bool {
	var oldest *shard
	var at uint64
	for i := range c.shards {
		s := &c.shards[i]
		s.mu.Lock()
		if t := s.lru.prev; t != &s.lru && t != keep && (oldest == nil || t.used < at) {
			oldest, at = s, t.used
		}
		s.mu.Unlock()
	}
	if oldest == nil {
		return false
	}
	oldest.mu.Lock()
	e := oldest.lru.prev
	if e == &oldest.lru || e == keep {
		oldest.mu.Unlock()
		return true // raced; the caller looks again
	}
	oldest.unlink(e)
	delete(oldest.m, e.key)
	e.evicted = true
	refs, data := e.refs, e.data
	oldest.mu.Unlock()
	n := int64(2 * len(data))
	c.bytes.Add(-n)
	release(n)
	if refs == 0 {
		c.free.put(data)
	}
	return true
}

// reserve charges n bytes, evicting entries.
func (c *cache) reserve(n int64) error {
	for {
		err := reserve(n)
		if !errors.Is(err, ErrMemoryLimit) || !c.evictOldest(nil) {
			return err
		}
	}
}

// drop releases an unreferenced Image's bytes.
func (c *cache) drop() {
	for c.evictOldest(nil) {
	}
}
