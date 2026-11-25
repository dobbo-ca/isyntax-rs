// Package container reads the iSyntax header: the XML up to the 0x04
// terminator, its base64 attributes, and either v1's block header table
// and seektable or iSyntax2's cluster templates and cluster table, into
// a codeblock index by tile. See docs/format.md, "File layout" and
// "Codeblock tables".
package container

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"image"
	"io"
	"strconv"
	"strings"
)

// ErrUnsupported is isyntax.ErrUnsupported.
var ErrUnsupported = errors.New("isyntax: unsupported file")

// Header is what the decoder needs from the XML header of the WSI image.
type Header struct {
	Major, Minor     int    // PIM_DP_UFS_INTERFACE_VERSION; Major < 100 is v1
	InterfaceVersion string // raw PIM_DP_UFS_INTERFACE_VERSION, trimmed
	Compressor       int    // 16 or 19
	BitsStored       int    // 8, 9, or (v1 only) 16
	OffsetX, OffsetY int
	PaddedW, PaddedH int
	Levels           int
	Width, Height    int
	GridW, GridH     int // level-0 tile grid
	MPPX, MPPY       float64
	ICC              []byte
	Label, Macro     []byte // JPEG
	Codeblocks       []Codeblock
	RGB              bool // planes R, G, B; no YCoCg
	Quality          int  // Quality=N of DICOM_DERIVATION_DESCRIPTION; -1 if absent

	// Scanner metadata, empty if the tag is absent.
	Barcode             string // PIM_DP_UFS_BARCODE, base64; sensitive
	ModelName           string // DICOM_MANUFACTURERS_MODEL_NAME
	DeviceSerial        string // DICOM_DEVICE_SERIAL_NUMBER
	AcquisitionDateTime string // DICOM_ACQUISITION_DATETIME

	// Best effort: zero/nil when absent or malformed.
	CalibrationStatus string   // (101D,100A)
	CalibrationDate   []string // (0018,1200) IStringArray tokens
	CalibrationTime   []string // (0018,1201) IStringArray tokens
	Rack, Slot        int      // (101D,1007), (101D,1008); -1 if absent

	// Envelopes is UFS_IMAGE_VALID_DATA_ENVELOPES, one polygon per
	// header object, vertices in level-0 pixels already shifted by
	// the pyramid pad, in file order (docs/format.md).
	Envelopes [][]image.Point
	// EnvelopeDefault: v1 without the tag.
	EnvelopeDefault bool
}

// Codeblock is one entry of the cluster table.
type Codeblock struct {
	X, Y          int64
	Colour, Scale int
	LL            bool
	Offset, Size  int64
}

// BlockSize is the codeblock side; tiles are twice that.
const BlockSize = 128

const (
	maxHeader    = 256 << 20
	maxCodeblock = 1 << 24
	maxLevels    = 16
)

// Parse reads and checks the header of the file r[0:size].
func Parse(ctx context.Context, r io.ReaderAt, size int64) (*Header, error) {
	xmlBytes, dataOffset, err := readXML(ctx, r, size)
	if err != nil {
		return nil, err
	}
	var root xObject
	if err := xml.Unmarshal(xmlBytes, &root); err != nil {
		return nil, fmt.Errorf("isyntax: XML header: %w", err)
	}
	h := &Header{Rack: -1, Slot: -1, Quality: -1}
	if a := root.attr(0x301D, 0x1001); a != nil {
		v := strings.TrimSpace(a.Value)
		h.InterfaceVersion = v
		maj, min, _ := strings.Cut(v, ".")
		h.Major, _ = strconv.Atoi(maj)
		h.Minor, _ = strconv.Atoi(min)
	}
	if a := root.attr(0x0008, 0x002A); a != nil {
		h.AcquisitionDateTime = strings.TrimSpace(a.Value)
	}
	if a := root.attr(0x0008, 0x1090); a != nil {
		h.ModelName = strings.TrimSpace(a.Value)
	}
	if a := root.attr(0x0018, 0x1000); a != nil {
		h.DeviceSerial = strings.TrimSpace(a.Value)
	}
	if a := root.attr(0x101D, 0x100A); a != nil {
		h.CalibrationStatus = strings.TrimSpace(a.Value)
	}
	if a := root.attr(0x0018, 0x1200); a != nil {
		h.CalibrationDate = quotedTokens(a.Value)
	}
	if a := root.attr(0x0018, 0x1201); a != nil {
		h.CalibrationTime = quotedTokens(a.Value)
	}
	h.Rack = uint16Attr(&root, 0x101D, 0x1007)
	h.Slot = uint16Attr(&root, 0x101D, 0x1008)
	if b := root.base64(0x301D, 0x1002); b != nil {
		h.Barcode = string(b)
	}
	var wsi *xObject
	if a := root.attr(0x301D, 0x1003); a != nil {
		for i := range a.Objects {
			img := &a.Objects[i]
			t := img.attr(0x301D, 0x1004)
			if t == nil {
				continue
			}
			switch strings.TrimSpace(t.Value) {
			case "WSI":
				wsi = img
			case "LABELIMAGE":
				h.Label = img.base64(0x301D, 0x1005)
			case "MACROIMAGE":
				h.Macro = img.base64(0x301D, 0x1005)
			}
		}
	}
	if wsi == nil {
		return nil, errors.New("isyntax: no WSI image in the header")
	}
	if err := h.parseWSI(wsi, r, dataOffset, size); err != nil {
		return nil, err
	}
	return h, nil
}

// quotedTokens splits IStringArray quoted tokens.
func quotedTokens(s string) []string {
	var out []string
	for {
		_, rest, ok := strings.Cut(s, `"`)
		if !ok {
			return out
		}
		tok, rest, ok := strings.Cut(rest, `"`)
		if !ok {
			return out
		}
		out = append(out, tok)
		s = rest
	}
}

// uint16Attr: -1 if absent or bad.
func uint16Attr(o *xObject, group, elem uint16) int {
	a := o.attr(group, elem)
	if a == nil {
		return -1
	}
	n, err := strconv.ParseUint(strings.TrimSpace(a.Value), 10, 16)
	if err != nil {
		return -1
	}
	return int(n)
}

// readXML: header bytes, offset after 0x04.
func readXML(ctx context.Context, r io.ReaderAt, size int64) ([]byte, int64, error) {
	var out []byte
	buf := make([]byte, 1<<20)
	for off := int64(0); off < size; off += int64(len(buf)) {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		n := int(min(int64(len(buf)), size-off))
		if _, err := r.ReadAt(buf[:n], off); err != nil && (err != io.EOF || off+int64(n) != size) {
			return nil, 0, fmt.Errorf("isyntax: reading the header: %w", err)
		}
		if i := bytes.IndexByte(buf[:n], 4); i >= 0 {
			return append(out, buf[:i]...), off + int64(i) + 1, nil
		}
		out = append(out, buf[:n]...)
		if len(out) > maxHeader {
			return nil, 0, errors.New("isyntax: XML header over 256 MiB")
		}
	}
	return nil, 0, errors.New("isyntax: no end of XML header (0x04) before the end of the file")
}

type xObject struct {
	Type  string  `xml:"ObjectType,attr"`
	Attrs []xAttr `xml:"Attribute"`
}

type xAttr struct {
	Group   string    `xml:"Group,attr"`
	Element string    `xml:"Element,attr"`
	Value   string    `xml:",chardata"`
	Objects []xObject `xml:"Array>DataObject"`
}

func (a *xAttr) is(group, elem uint16) bool {
	g, err1 := strconv.ParseUint(a.Group, 0, 16)
	e, err2 := strconv.ParseUint(a.Element, 0, 16)
	return err1 == nil && err2 == nil && uint16(g) == group && uint16(e) == elem
}

// attr: o's first match, direct only.
func (o *xObject) attr(group, elem uint16) *xAttr {
	for i := range o.Attrs {
		if o.Attrs[i].is(group, elem) {
			return &o.Attrs[i]
		}
	}
	return nil
}

// find: match for tag, any depth.
func (o *xObject) find(group, elem uint16) *xAttr {
	for i := range o.Attrs {
		a := &o.Attrs[i]
		if a.is(group, elem) {
			return a
		}
		for j := range a.Objects {
			if f := a.Objects[j].find(group, elem); f != nil {
				return f
			}
		}
	}
	return nil
}

// objects: o's array attribute's DataObjects.
func (o *xObject) objects(group, elem uint16) []xObject {
	if a := o.attr(group, elem); a != nil {
		return a.Objects
	}
	return nil
}

// base64: decoded value, nil if bad.
func (o *xObject) base64(group, elem uint16) []byte {
	if a := o.find(group, elem); a != nil {
		b, _ := decodeBase64(a.Value)
		return b
	}
	return nil
}

func ints(s string) ([]int, error) {
	f := strings.Fields(s)
	out := make([]int, len(f))
	for i, v := range f {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("isyntax: bad integer %q", v)
		}
		out[i] = n
	}
	return out, nil
}

// planeNames: colour dimension's 0x2008 value.
func planeNames(d *xObject) string {
	if t := d.attr(0x301D, 0x2005); t == nil || strings.TrimSpace(t.Value) != "colour component" {
		return ""
	}
	a := d.attr(0x301D, 0x2008)
	if a == nil {
		return ""
	}
	return strings.Join(strings.Fields(strings.ReplaceAll(a.Value, `"`, " ")), " ")
}

// dimRange is UFS_IMAGE_DIMENSION_RANGE "start step end".
type dimRange struct{ start, steps int }

// parseEnvelopes: vertex list per envelope, pad-shifted.
func parseEnvelopes(wsi *xObject, pad int) [][]image.Point {
	var out [][]image.Point
	for _, env := range wsi.objects(0x301D, 0x2023) {
		var poly []image.Point
		for _, vh := range env.objects(0x301D, 0x2024) {
			a := vh.attr(0x301D, 0x2025)
			if a == nil {
				continue
			}
			f := strings.Fields(a.Value)
			if len(f) != 2 {
				continue
			}
			x, errX := strconv.Atoi(f[0])
			y, errY := strconv.Atoi(f[1])
			if errX != nil || errY != nil {
				continue
			}
			poly = append(poly, image.Pt(x-pad, y-pad))
		}
		if len(poly) > 0 {
			out = append(out, poly)
		}
	}
	return out
}

func ranges(o *xObject) ([]dimRange, error) {
	var out []dimRange
	for _, d := range o.objects(0x301D, 0x200A) {
		a := d.attr(0x301D, 0x200B)
		if a == nil {
			return nil, errors.New("isyntax: dimension range without a value")
		}
		v, err := ints(a.Value)
		if err != nil {
			return nil, err
		}
		if len(v) != 3 {
			return nil, fmt.Errorf("isyntax: dimension range %q", a.Value)
		}
		step := v[1]
		if step == 0 {
			step = 1
		}
		out = append(out, dimRange{v[0], (v[2] + v[1] - v[0]) / step})
	}
	return out, nil
}

// quality: Quality=N gives N, else -1.
func quality(desc string) int {
	for _, f := range strings.Split(desc, "|") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(f), "Quality="); ok {
			if n, err := strconv.Atoi(v); err == nil && n >= 0 {
				return n
			}
		}
	}
	return -1
}

func (h *Header) parseWSI(wsi *xObject, rd io.ReaderAt, dataOffset, size int64) error {
	h.ICC = wsi.base64(0x0028, 0x2000)
	if a := wsi.attr(0x0008, 0x2111); a != nil {
		h.Quality = quality(a.Value)
	}

	gens := wsi.objects(0x301D, 0x2000)
	if len(gens) == 0 {
		return errors.New("isyntax: no UFS_IMAGE_GENERAL_HEADERS")
	}
	gen := &gens[0]
	for i, d := range gen.objects(0x301D, 0x2003) {
		switch names := planeNames(&d); names {
		case "R G B":
			h.RGB = true
		case "", "Y Co Cg":
			// YCoCg, the default colour transform.
		default:
			return fmt.Errorf("%w: plane names %q", ErrUnsupported, names)
		}
		a := d.attr(0x301D, 0x2007)
		if a == nil || i > 1 {
			continue
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(a.Value), 64)
		if err == nil && f > 0 {
			if i == 0 {
				h.MPPX = f
			} else {
				h.MPPY = f
			}
		}
	}
	r, err := ranges(gen)
	if err != nil {
		return err
	}
	if len(r) < 4 {
		return errors.New("isyntax: general header has under 4 dimension ranges")
	}
	h.OffsetX, h.PaddedW = r[0].start, r[0].steps
	h.OffsetY, h.PaddedH = r[1].start, r[1].steps
	h.Levels = r[3].steps
	if h.Levels < 1 || h.Levels > maxLevels {
		return fmt.Errorf("isyntax: %d levels", h.Levels)
	}
	pad := 3<<h.Levels - 3
	h.Width, h.Height = h.PaddedW-2*pad, h.PaddedH-2*pad
	if h.Width <= 0 || h.Height <= 0 || h.PaddedW > 1<<30 || h.PaddedH > 1<<30 {
		return fmt.Errorf("isyntax: image %dx%d", h.Width, h.Height)
	}
	span := BlockSize << h.Levels
	h.GridW = (h.PaddedW + span - 1) / span << (h.Levels - 1)
	h.GridH = (h.PaddedH + span - 1) / span << (h.Levels - 1)
	h.Envelopes = parseEnvelopes(wsi, pad)
	if len(h.Envelopes) == 0 {
		// v1 lacks the tag; default rectangle.
		h.EnvelopeDefault = true
		h.Envelopes = [][]image.Point{{
			{X: 0, Y: 0}, {X: h.Width, Y: 0}, {X: h.Width, Y: h.Height}, {X: 0, Y: h.Height},
		}}
	}

	// Block size, bits, compressor: first template.
	tmpls := wsi.objects(0x301D, 0x2009)
	if len(tmpls) == 0 {
		return errors.New("isyntax: no block header templates")
	}
	t0 := &tmpls[0]
	if a := t0.attr(0x0028, 0x0101); a != nil {
		h.BitsStored, _ = strconv.Atoi(strings.TrimSpace(a.Value))
	}
	if a := t0.attr(0x301D, 0x200F); a != nil {
		h.Compressor, _ = strconv.Atoi(strings.TrimSpace(a.Value))
	}
	if h.Compressor != 16 && h.Compressor != 19 {
		return fmt.Errorf("%w: compressor %d", ErrUnsupported, h.Compressor)
	}
	// v1 compressor 19: untested, no corpus.
	if h.Major < 100 && h.Compressor == 19 {
		return fmt.Errorf("%w: v1 compressor 19", ErrUnsupported)
	}
	// v1 also allows 16 bits (format.md).
	if h.BitsStored != 8 && h.BitsStored != 9 && (h.Major >= 100 || h.BitsStored != 16) {
		return fmt.Errorf("%w: %d bits stored", ErrUnsupported, h.BitsStored)
	}
	br, err := ranges(t0)
	if err != nil {
		return err
	}
	if len(br) < 2 || br[0].steps != BlockSize || br[1].steps != BlockSize {
		return fmt.Errorf("%w: codeblocks other than %dx%d", ErrUnsupported, BlockSize, BlockSize)
	}

	if h.Major < 100 {
		h.Codeblocks, err = h.parseBlockHeaderTable(wsi, rd, dataOffset, size)
		return err
	}

	var ctmpls []clusterTemplate
	for _, c := range wsi.objects(0x301D, 0x2016) {
		t, err := parseClusterTemplate(&c)
		if err != nil {
			return err
		}
		ctmpls = append(ctmpls, t)
	}
	a := wsi.attr(0x301D, 0x201F)
	if a == nil {
		return errors.New("isyntax: no cluster header table")
	}
	table, err := decodeBase64(a.Value)
	if err != nil {
		return fmt.Errorf("isyntax: cluster header table: %w", err)
	}
	h.Codeblocks, err = parseClusterTable(table, ctmpls)
	return err
}

// relCoord: codeblock's spot in its cluster.
type relCoord struct{ x, y, colour, scale, coeff int }

type clusterTemplate struct{ rel []relCoord }

func parseClusterTemplate(c *xObject) (clusterTemplate, error) {
	var t clusterTemplate
	r, err := ranges(c)
	if err != nil {
		return t, err
	}
	var base [5]int
	for i := 0; i < len(r) && i < 5; i++ {
		base[i] = r[i].start
	}
	var order []int
	if a := c.attr(0x301D, 0x2021); a != nil {
		if order, err = ints(a.Value); err != nil {
			return t, err
		}
	}
	if len(order) > 5 {
		return t, errors.New("isyntax: over 5 dimensions in a cluster")
	}
	for _, d := range order {
		if d < 0 || d > 4 {
			return t, fmt.Errorf("isyntax: cluster dimension %d", d)
		}
	}
	for _, b := range c.objects(0x301D, 0x200D) {
		var raw []int
		if a := b.attr(0x301D, 0x200E); a != nil {
			if raw, err = ints(a.Value); err != nil {
				return t, err
			}
		}
		v := base
		for k, d := range order {
			if k < len(raw) {
				v[d] += raw[k]
			}
		}
		t.rel = append(t.rel, relCoord{x: v[0], y: v[1], colour: v[2], scale: v[3], coeff: v[4]})
	}
	return t, nil
}

// parseClusterTable reads the decoded (301D,201F) value.
func parseClusterTable(b []byte, tmpls []clusterTemplate) ([]Codeblock, error) {
	if len(b) < 4 {
		return nil, errors.New("isyntax: cluster table under 4 bytes")
	}
	var out []Codeblock
	pos := 4
	short := errors.New("isyntax: cluster table truncated")
	type elem struct {
		group, elem uint16
		data        []byte
	}
	read := func(end int) (elem, error) {
		if pos+8 > end {
			return elem{}, short
		}
		e := elem{binary.LittleEndian.Uint16(b[pos:]), binary.LittleEndian.Uint16(b[pos+2:]), nil}
		n := int(binary.LittleEndian.Uint32(b[pos+4:]))
		if n < 0 || n > end-pos-8 {
			return elem{}, short
		}
		e.data = b[pos+8 : pos+8+n]
		pos += 8 + n
		return e, nil
	}
	for first := true; pos < len(b); first = false {
		item, err := read(len(b))
		if err != nil {
			return nil, err
		}
		if item.group != 0xFFFE || item.elem != 0xE000 {
			if first {
				return nil, errors.New("isyntax: cluster table does not start with an item")
			}
			break
		}
		end := pos
		pos -= len(item.data)
		var f [6][]byte
		for i := range f {
			e, err := read(end)
			if err != nil {
				return nil, err
			}
			f[i] = e.data
		}
		pos = end
		m := len(f[4]) / 4
		if len(f[0]) < 8 || len(f[1]) != 4 || len(f[2]) != 8 || len(f[3]) != 8 || len(f[5])/4 != m {
			return nil, errors.New("isyntax: malformed cluster table item")
		}
		cx := int64(int32(binary.LittleEndian.Uint32(f[0])))
		cy := int64(int32(binary.LittleEndian.Uint32(f[0][4:])))
		id := binary.LittleEndian.Uint32(f[1])
		if int(id) >= len(tmpls) || m > len(tmpls[id].rel) {
			return nil, fmt.Errorf("isyntax: cluster template %d with %d codeblocks missing", id, m)
		}
		dataOff := int64(binary.LittleEndian.Uint64(f[2]))
		for j := 0; j < m; j++ {
			rel := tmpls[id].rel[j]
			out = append(out, Codeblock{
				X: cx + int64(rel.x), Y: cy + int64(rel.y),
				Colour: rel.colour, Scale: rel.scale, LL: rel.coeff == 3,
				Offset: dataOff + int64(binary.LittleEndian.Uint32(f[4][4*j:])),
				Size:   int64(binary.LittleEndian.Uint32(f[5][4*j:])),
			})
		}
	}
	return out, nil
}

// seektableEntry: one (301D,2015) entry size.
const seektableEntry = 40

type seekEntry struct{ offset, size int64 }

// codeblockOrigin: shared by Index and blockID.
func codeblockOrigin(scale int, ll bool) int64 {
	off := int64(3<<scale - 2)
	if ll {
		off += int64(1) << scale
	}
	return off
}

// blockID: libisyntax's seektable index (L isyntax.c:3301-3330).
func (h *Header) blockID(tileCount []int, total int64, x, y int64, scale, colour int, ll bool) (int64, error) {
	if scale < 0 || scale >= h.Levels || colour < 0 || colour > 2 {
		return 0, fmt.Errorf("isyntax: codeblock at scale %d colour %d", scale, colour)
	}
	maxScale := scale
	if ll {
		maxScale++
	}
	var id int64
	for s := 0; s < maxScale; s++ {
		id += int64(tileCount[s])
	}
	off := codeblockOrigin(scale, ll)
	dx, dy := x-int64(h.OffsetX)-off, y-int64(h.OffsetY)-off
	span := int64(2 * BlockSize << scale)
	if dx < 0 || dy < 0 {
		return 0, fmt.Errorf("isyntax: codeblock at (%d, %d) scale %d outside the tile grid", x, y, scale)
	}
	stride := int64(h.GridW >> scale)
	id += dy/span*stride + dx/span + int64(colour)*total
	return id, nil
}

// readSeektable reads v1's seektable (docs/format.md).
func readSeektable(r io.ReaderAt, dataOffset int64, blockCount int) ([]seekEntry, error) {
	var tag [8]byte
	if _, err := r.ReadAt(tag[:], dataOffset); err != nil {
		return nil, fmt.Errorf("isyntax: reading the seektable: %w", err)
	}
	if binary.LittleEndian.Uint16(tag[:]) != 0x301D || binary.LittleEndian.Uint16(tag[2:]) != 0x2015 {
		return nil, fmt.Errorf("isyntax: no seektable at %d", dataOffset)
	}
	size := int32(binary.LittleEndian.Uint32(tag[4:]))
	if size < 0 {
		size = int32(blockCount) * seektableEntry
	}
	if size <= 0 || int(size)%seektableEntry != 0 {
		return nil, fmt.Errorf("isyntax: seektable of %d bytes", size)
	}
	buf := make([]byte, size)
	if _, err := r.ReadAt(buf, dataOffset+8); err != nil {
		return nil, fmt.Errorf("isyntax: reading the seektable: %w", err)
	}
	out := make([]seekEntry, len(buf)/seektableEntry)
	for i := range out {
		p := i * seektableEntry
		out[i] = seekEntry{
			offset: int64(binary.LittleEndian.Uint64(buf[p+16:])),
			size:   int64(binary.LittleEndian.Uint64(buf[p+32:])),
		}
	}
	return out, nil
}

// parseBlockHeaderTable reads v1's codeblock index (docs/format.md).
func (h *Header) parseBlockHeaderTable(wsi *xObject, r io.ReaderAt, dataOffset, size int64) ([]Codeblock, error) {
	a := wsi.attr(0x301D, 0x2014)
	if a == nil {
		return nil, errors.New("isyntax: no block header table")
	}
	table, err := decodeBase64(a.Value)
	if err != nil {
		return nil, fmt.Errorf("isyntax: block header table: %w", err)
	}
	if len(table) < 12 {
		return nil, errors.New("isyntax: block header table truncated")
	}
	headerSize := int(binary.LittleEndian.Uint32(table))
	seqSize := int(binary.LittleEndian.Uint32(table[8:]))
	var entrySize int
	partial := false
	switch seqSize {
	case 40:
		entrySize, partial = 48, true
	case 72:
		entrySize = 80
	default:
		return nil, fmt.Errorf("isyntax: block header entry of size %d", seqSize)
	}
	if headerSize < 0 || headerSize%entrySize != 0 || 4+headerSize > len(table) {
		return nil, errors.New("isyntax: block header table truncated")
	}
	count := headerSize / entrySize

	type raw struct {
		x, y          int64
		colour, scale int
		ll            bool
	}
	entries := make([]raw, count)
	out := make([]Codeblock, count)
	for i := range entries {
		p := 4 + i*entrySize
		x := int64(int32(binary.LittleEndian.Uint32(table[p+16:])))
		y := int64(int32(binary.LittleEndian.Uint32(table[p+20:])))
		colour := int(binary.LittleEndian.Uint32(table[p+24:]))
		scale := int(binary.LittleEndian.Uint32(table[p+28:]))
		ll := binary.LittleEndian.Uint32(table[p+32:]) == 0
		entries[i] = raw{x, y, colour, scale, ll}
		out[i] = Codeblock{X: x, Y: y, Colour: colour, Scale: scale, LL: ll}
		if !partial {
			out[i].Offset = int64(binary.LittleEndian.Uint64(table[p+44:]))
			out[i].Size = int64(binary.LittleEndian.Uint64(table[p+60:]))
		}
	}
	if !partial {
		return out, nil
	}

	seek, err := readSeektable(r, dataOffset, count)
	if err != nil {
		return nil, err
	}
	tileCount := make([]int, h.Levels)
	base := h.GridW * h.GridH
	var total int64
	for s := range tileCount {
		tileCount[s] = base >> (2 * s)
		total += int64(tileCount[s])
	}
	total += int64(tileCount[h.Levels-1]) // top scale's own LL plane
	for i, e := range entries {
		id, err := h.blockID(tileCount, total, e.x, e.y, e.scale, e.colour, e.ll)
		if err != nil {
			return nil, err
		}
		if id < 0 || id >= int64(len(seek)) {
			return nil, fmt.Errorf("isyntax: block id %d outside the %d-entry seektable", id, len(seek))
		}
		out[i].Offset, out[i].Size = seek[id].offset, seek[id].size
	}
	return out, nil
}

// decodeBase64: matches libisyntax's tolerant alphabet.
func decodeBase64(s string) ([]byte, error) {
	s = strings.TrimSuffix(s, "/")
	s = strings.TrimRight(s, "\r\n ")
	clean := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '+' || c == '/' || c == '=' {
			clean = append(clean, c)
		}
	}
	if len(clean) == 0 || len(clean)%4 != 0 {
		return nil, fmt.Errorf("isyntax: base64 of %d characters", len(clean))
	}
	for i := 0; i < len(clean); i += 4 {
		if bytes.IndexByte(clean[i:i+4], '=') >= 0 {
			clean = clean[:i+4]
			break
		}
	}
	out := make([]byte, base64.StdEncoding.DecodedLen(len(clean)))
	n, err := base64.StdEncoding.Decode(out, clean)
	if err != nil {
		return nil, fmt.Errorf("isyntax: base64: %w", err)
	}
	return out[:n], nil
}

// TileKey is a tile of the grid at a scale.
type TileKey struct{ Scale, X, Y int32 }

// Block is a codeblock's place in the file; Size 0 reads as zeros.
type Block struct {
	Offset int64
	Size   int32
}

// Tile holds a tile's codeblocks per colour. Only top-scale tiles have
// LL blocks. Exists is set by any colour-0 codeblock, as in libisyntax.
type Tile struct {
	H, LL  [3]Block
	Exists bool
}

// Index maps the codeblocks to tiles (docs/format.md "Codeblock
// position") and checks they lie within a file of fileSize bytes.
func (h *Header) Index(fileSize int64) (map[TileKey]*Tile, error) {
	tiles := make(map[TileKey]*Tile)
	for _, c := range h.Codeblocks {
		if c.Scale < 0 || c.Scale >= h.Levels || c.Colour < 0 || c.Colour > 2 {
			return nil, fmt.Errorf("isyntax: codeblock at scale %d colour %d", c.Scale, c.Colour)
		}
		if c.Offset < 0 || c.Size < 0 || c.Size > maxCodeblock || c.Offset > fileSize-c.Size {
			return nil, fmt.Errorf("isyntax: codeblock of %d bytes at %d outside the file", c.Size, c.Offset)
		}
		off := codeblockOrigin(c.Scale, c.LL)
		dx, dy := c.X-int64(h.OffsetX)-off, c.Y-int64(h.OffsetY)-off
		span := int64(2 * BlockSize << c.Scale)
		if dx < 0 || dy < 0 || dx/span >= int64(h.GridW>>c.Scale) || dy/span >= int64(h.GridH>>c.Scale) {
			return nil, fmt.Errorf("isyntax: codeblock at (%d, %d) scale %d outside the tile grid", c.X, c.Y, c.Scale)
		}
		k := TileKey{int32(c.Scale), int32(dx / span), int32(dy / span)}
		t := tiles[k]
		if t == nil {
			t = &Tile{}
			tiles[k] = t
		}
		b := Block{c.Offset, int32(c.Size)}
		if c.LL {
			t.LL[c.Colour] = b
		} else {
			t.H[c.Colour] = b
		}
		t.Exists = t.Exists || c.Colour == 0
	}
	return tiles, nil
}
