// Command commentlint enforces the repo's comment-length rule: every
// comment group is 6 words or fewer and says why, not what. Exempt: Go
// doc comments on exported identifiers, package docs, and LICENSE.
//
//	Usage: go run ./cmd/commentlint [paths...]
package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const maxWords = 6

type violation struct {
	file string
	line int
	n    int
	text string
}

func main() {
	roots := os.Args[1:]
	if len(roots) == 0 {
		roots = []string{"."}
	}
	var violations []violation
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if d.Name() == ".git" {
					return filepath.SkipDir
				}
				return nil
			}
			switch filepath.Ext(path) {
			case ".go":
				vs, err := lintGo(path)
				if err != nil {
					return err
				}
				violations = append(violations, vs...)
			case ".s", ".sh", ".c":
				vs, err := lintLineComments(path)
				if err != nil {
					return err
				}
				violations = append(violations, vs...)
			}
			return nil
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	for _, v := range violations {
		fmt.Printf("%s:%d: %d words: %s\n", v.file, v.line, v.n, v.text)
	}
	fmt.Printf("%d over %d words\n", len(violations), maxWords)
	if len(violations) > 0 {
		os.Exit(1)
	}
}

func wordCount(s string) int {
	return len(strings.Fields(s))
}

// Directive syntax, e.g. //go:build, not prose.
var directive = regexp.MustCompile(`^//\S+:\S`)

func groupText(g *ast.CommentGroup) string {
	var b strings.Builder
	for _, c := range g.List {
		if directive.MatchString(c.Text) {
			continue
		}
		t := c.Text
		t = strings.TrimPrefix(t, "//")
		t = strings.TrimPrefix(t, "/*")
		t = strings.TrimSuffix(t, "*/")
		b.WriteString(strings.TrimSpace(t))
		b.WriteString(" ")
	}
	return strings.TrimSpace(b.String())
}

// Exempts package doc, exported decl docs.
func lintGo(path string) ([]violation, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	exempt := map[*ast.CommentGroup]bool{}
	if f.Doc != nil {
		exempt[f.Doc] = true
	}
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Doc != nil && d.Name.IsExported() {
				exempt[d.Doc] = true
			}
		case *ast.GenDecl:
			if d.Doc == nil {
				continue
			}
			exported := false
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					exported = exported || s.Name.IsExported()
				case *ast.ValueSpec:
					for _, n := range s.Names {
						exported = exported || n.IsExported()
					}
				}
			}
			if exported {
				exempt[d.Doc] = true
			}
			// Exported struct fields: API docs too.
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				st, ok := ts.Type.(*ast.StructType)
				if !ok {
					continue
				}
				for _, field := range st.Fields.List {
					if field.Doc == nil {
						continue
					}
					for _, n := range field.Names {
						if n.IsExported() {
							exempt[field.Doc] = true
						}
					}
				}
			}
		}
	}

	var violations []violation
	for _, g := range f.Comments {
		if exempt[g] {
			continue
		}
		text := groupText(g)
		if n := wordCount(text); n > maxWords {
			pos := fset.Position(g.Pos())
			violations = append(violations, violation{path, pos.Line, n, text})
		}
	}
	return violations, nil
}

var (
	slashComment = regexp.MustCompile(`^\s*//\s?(.*)$`)
	hashComment  = regexp.MustCompile(`^\s*#\s?(.*)$`)
)

// Groups full-line comments; # for .sh.
func lintLineComments(path string) ([]violation, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	lineRe := slashComment
	if filepath.Ext(path) == ".sh" {
		lineRe = hashComment
	}

	var violations []violation
	var group []string
	startLine := 0
	flush := func() {
		if len(group) == 0 {
			return
		}
		text := strings.Join(group, " ")
		if n := wordCount(text); n > maxWords {
			violations = append(violations, violation{path, startLine, n, text})
		}
		group = nil
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := sc.Text()
		if m := lineRe.FindStringSubmatch(raw); m != nil {
			if directive.MatchString(strings.TrimSpace(raw)) {
				flush()
				continue
			}
			if len(group) == 0 {
				startLine = line
			}
			group = append(group, m[1])
			continue
		}
		flush()
		// Trailing comment: its own group.
		if i := findInlineComment(raw, path); i >= 0 {
			text := strings.TrimSpace(raw[i+2:])
			if n := wordCount(text); n > maxWords {
				violations = append(violations, violation{path, line, n, text})
			}
		}
	}
	flush()
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return violations, nil
}

// Trailing "//" outside quotes; skips .sh.
func findInlineComment(line, path string) int {
	if filepath.Ext(path) == ".sh" {
		return -1
	}
	inStr := false
	var quote byte
	for i := 0; i < len(line); i++ {
		c := line[i]
		if inStr {
			if c == quote && (i == 0 || line[i-1] != '\\') {
				inStr = false
			}
			continue
		}
		switch c {
		case '"', '\'':
			inStr = true
			quote = c
		case '/':
			if i+1 < len(line) && line[i+1] == '/' {
				return i
			}
		}
	}
	return -1
}
