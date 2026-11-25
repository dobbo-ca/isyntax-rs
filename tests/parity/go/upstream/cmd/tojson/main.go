// Command tojson turns `go test -bench` text (or libisyntax's sweep,
// which prints the same line shape) on stdin into one JSON document,
// taking the median and range over repeated runs (`-count`).
//
//	Usage: go test -run '^$' -bench 'Sweep' -count 5 -cpu 1 . | \
//		go run ./cmd/tojson -sha $(git rev-parse --short=12 HEAD) > bench/results/<sha>-<goarch>.json
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

var (
	benchLine  = regexp.MustCompile(`^(Benchmark\S+?)(?:-(\d+))?\s+\d+\s+(.*)$`)
	header     = regexp.MustCompile(`^(goos|goarch|cpu): (.+)$`)
	metricKeys = map[string]string{
		"ms/tile":      "ms_per_tile",
		"tiles/s":      "tiles_per_s",
		"tiles/s/core": "tiles_per_s_per_core",
		"cpu-us/tile":  "cpu_us_per_tile",
		"maxrss-MiB":   "maxrss_mib",
	}
)

type key struct {
	name  string
	procs int
}

func main() {
	sha := flag.String("sha", "", "commit sha to embed")
	flag.Parse()

	metrics := map[key]map[string][]float64{}
	var order []key
	var goos, goarch, cpu string
	hdr := map[string]*string{"goos": &goos, "goarch": &goarch, "cpu": &cpu}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if hm := header.FindStringSubmatch(line); hm != nil {
			*hdr[hm[1]] = hm[2]
			continue
		}
		m := benchLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		procs, _ := strconv.Atoi(m[2])
		if procs == 0 {
			procs = 1 // GOMAXPROCS==1 prints no -N suffix
		}
		k := key{m[1], procs}
		if _, ok := metrics[k]; !ok {
			order = append(order, k)
			metrics[k] = map[string][]float64{}
		}
		f := strings.Fields(m[3])
		for i := 0; i+1 < len(f); i += 2 {
			if v, err := strconv.ParseFloat(f[i], 64); err == nil {
				metrics[k][f[i+1]] = append(metrics[k][f[i+1]], v)
			}
		}
	}
	if len(order) == 0 {
		fmt.Fprintln(os.Stderr, "tojson: no benchmark lines found on stdin")
		os.Exit(1)
	}
	var out []map[string]any
	for _, k := range order {
		r := map[string]any{"name": k.name, "procs": k.procs, "runs": len(metrics[k]["ns/op"]), "ns_per_op_median": median(metrics[k]["ns/op"])}
		for metricKey, jsonKey := range metricKeys {
			v := metrics[k][metricKey]
			if len(v) == 0 {
				continue
			}
			r[jsonKey] = median(v)
			r[jsonKey+"_min"] = minOf(v)
			r[jsonKey+"_max"] = maxOf(v)
		}
		out = append(out, r)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(map[string]any{"sha": *sha, "goos": goos, "goarch": goarch, "cpu": cpu, "go": runtime.Version(), "benchmarks": out})
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	n := len(v)
	if n%2 == 0 {
		return (v[n/2-1] + v[n/2]) / 2
	}
	return v[n/2]
}

func minOf(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func maxOf(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}
