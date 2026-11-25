package fixture

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
)

// FileSpec describes a synthetic iSyntax file, v1 or iSyntax2. Version's
// major (before the first '.') under 100 selects the v1 layout: a block
// header table and seektable instead of cluster templates and table.
type FileSpec struct {
	Version          string // PIM_DP_UFS_INTERFACE_VERSION, default "100.5"
	Compressor       int    // 16 or 19
	BitsStored       int    // 8 or 9
	Levels           int
	Width, Height    int // image size without the wavelet padding
	OffsetX, OffsetY int
	MPP              float64
	// Exists reports whether a tile has codeblocks; nil means every tile.
	Exists func(scale, tx, ty int) bool
	// Coef returns a codeblock's 128x128 coefficients in raster order,
	// three bands (HL, LH, HH) for detail blocks and one for LL.
	Coef              func(scale, tx, ty, colour int, ll bool) []int16
	Label, Macro, ICC []byte
	// Extra is raw XML for root-level attributes, e.g. calibration tags.
	Extra string
}

// Block is where File put one codeblock.
type Block struct {
	Scale, TX, TY, Colour int
	LL                    bool
	Offset, Size          int64
}

const blockSize = 128

// Grid is the tile grid of docs/format.md "Levels and tiles".
func (s FileSpec) Grid() (w, h int) {
	pad := 3<<s.Levels - 3
	span := blockSize << s.Levels
	w = (s.Width + 2*pad + span - 1) / span << (s.Levels - 1)
	h = (s.Height + 2*pad + span - 1) / span << (s.Levels - 1)
	return w, h
}

// v1Major: version's major is under 100.
func v1Major(version string) bool {
	maj, _, _ := strings.Cut(version, ".")
	n, err := strconv.Atoi(maj)
	return err == nil && n < 100
}

// File builds the file and lists its codeblocks.
func File(s FileSpec) ([]byte, []Block) {
	if s.Version == "" {
		s.Version = "100.5"
	}
	if v1Major(s.Version) {
		return fileV1(s)
	}
	version := 2
	if s.Compressor == 16 {
		version = 1
	}
	pad := 3<<s.Levels - 3
	gw, gh := s.Grid()
	top := s.Levels - 1

	// One cluster per tile, plus LL.
	type cluster struct {
		x, y, scale, tmpl int
		blocks            [][]byte
		meta              []Block
	}
	var clusters []cluster
	for sc := top; sc >= 0; sc-- {
		for ty := 0; ty < gh>>sc; ty++ {
			for tx := 0; tx < gw>>sc; tx++ {
				if s.Exists != nil && !s.Exists(sc, tx, ty) {
					continue
				}
				c := cluster{
					x:     s.OffsetX + 3<<sc - 2 + tx*(2*blockSize<<sc),
					y:     s.OffsetY + 3<<sc - 2 + ty*(2*blockSize<<sc),
					scale: sc,
					tmpl:  2 * sc,
				}
				kinds := []bool{false}
				if sc == top {
					kinds = append(kinds, true)
					c.tmpl++
				}
				for _, ll := range kinds {
					for col := 0; col < 3; col++ {
						n := 3
						if ll {
							n = 1
						}
						p := HulskenParams{Version: version, ZeroSymbol: byte(col * 7), CounterBits: 3 + col}
						c.blocks = append(c.blocks, EncodeHulsken(s.Coef(sc, tx, ty, col, ll), n, blockSize, blockSize, p))
						c.meta = append(c.meta, Block{Scale: sc, TX: tx, TY: ty, Colour: col, LL: ll})
					}
				}
				clusters = append(clusters, c)
			}
		}
	}

	// Lay fixed-width table, fill offsets after.
	table := func(dataStart int64) ([]byte, []Block) {
		var t []byte
		var blocks []Block
		tag := func(group, elem uint16, size int) {
			t = binary.LittleEndian.AppendUint16(t, group)
			t = binary.LittleEndian.AppendUint16(t, elem)
			t = binary.LittleEndian.AppendUint32(t, uint32(size))
		}
		off := dataStart
		for _, c := range clusters {
			m := len(c.blocks)
			itemSize := 8 + 12 + 8 + 4 + 8 + 8 + 8 + 8 + 8 + 4*m + 8 + 4*m
			tag(0xFFFE, 0xE000, itemSize)
			tag(0x301D, 0x2020, 12)
			for _, v := range []int{c.x, c.y, c.scale} {
				t = binary.LittleEndian.AppendUint32(t, uint32(v))
			}
			tag(0x301D, 0x2019, 4)
			t = binary.LittleEndian.AppendUint32(t, uint32(c.tmpl))
			var total int64
			for _, b := range c.blocks {
				total += int64(len(b))
			}
			tag(0x301D, 0x2010, 8)
			t = binary.LittleEndian.AppendUint64(t, uint64(off))
			tag(0x301D, 0x2011, 8)
			t = binary.LittleEndian.AppendUint64(t, uint64(total))
			tag(0x301D, 0x201B, 4*m)
			rel := int64(0)
			for i, b := range c.blocks {
				t = binary.LittleEndian.AppendUint32(t, uint32(rel))
				mb := c.meta[i]
				mb.Offset, mb.Size = off+rel, int64(len(b))
				blocks = append(blocks, mb)
				rel += int64(len(b))
			}
			tag(0x301D, 0x201C, 4*m)
			for _, b := range c.blocks {
				t = binary.LittleEndian.AppendUint32(t, uint32(len(b)))
			}
			off += total
		}
		return append(binary.LittleEndian.AppendUint32(nil, uint32(len(t))), t...), blocks
	}

	xmlFor := func(tbl []byte) string {
		var b strings.Builder
		attr := func(name string, group, elem uint16, vr, value string) {
			fmt.Fprintf(&b, "<Attribute Name=%q Group=\"0x%04x\" Element=\"0x%04x\" PMSVR=%q>%s</Attribute>\n", name, group, elem, vr, value)
		}
		open := func(name string, group, elem uint16) {
			fmt.Fprintf(&b, "<Attribute Name=%q Group=\"0x%04x\" Element=\"0x%04x\" PMSVR=\"IDataObjectArray\">\n<Array>\n", name, group, elem)
		}
		closeArr := func() { b.WriteString("</Array>\n</Attribute>\n") }
		ranges := func(rs ...string) {
			open("UFS_IMAGE_DIMENSION_RANGES", 0x301D, 0x200A)
			for _, r := range rs {
				b.WriteString("<DataObject ObjectType=\"UFSImageDimensionRange\">\n")
				attr("UFS_IMAGE_DIMENSION_RANGE", 0x301D, 0x200B, "IUInt32Array", r)
				b.WriteString("</DataObject>\n")
			}
			closeArr()
		}
		b.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<DataObject ObjectType=\"DPUfsImport\">\n")
		attr("DICOM_MANUFACTURER", 0x0008, 0x0070, "IString", "PHILIPS")
		b.WriteString(s.Extra)
		open("PIM_DP_SCANNED_IMAGES", 0x301D, 0x1003)

		b.WriteString("<DataObject ObjectType=\"DPScannedImage\">\n")
		attr("PIM_DP_IMAGE_TYPE", 0x301D, 0x1004, "IString", "WSI")
		open("DP_COLOR_MANAGEMENT", 0x301D, 0x1013)
		b.WriteString("<DataObject ObjectType=\"DPColorManagement\">\n")
		attr("DICOM_ICCPROFILE", 0x0028, 0x2000, "IString", base64.StdEncoding.EncodeToString(s.ICC)+"/")
		b.WriteString("</DataObject>\n")
		closeArr()
		open("UFS_IMAGE_GENERAL_HEADERS", 0x301D, 0x2000)
		b.WriteString("<DataObject ObjectType=\"UFSImageGeneralHeader\">\n")
		open("UFS_IMAGE_DIMENSIONS", 0x301D, 0x2003)
		for i := 0; i < 5; i++ {
			b.WriteString("<DataObject ObjectType=\"UFSImageDimension\">\n")
			if i < 2 {
				attr("UFS_IMAGE_DIMENSION_SCALE_FACTOR", 0x301D, 0x2007, "IDouble", fmt.Sprint(s.MPP))
			}
			b.WriteString("</DataObject>\n")
		}
		closeArr()
		ranges(
			fmt.Sprintf("%d 1 %d", s.OffsetX, s.OffsetX+s.Width+2*pad-1),
			fmt.Sprintf("%d 1 %d", s.OffsetY, s.OffsetY+s.Height+2*pad-1),
			"0 1 2", fmt.Sprintf("0 1 %d", top), "0 1 3")
		b.WriteString("</DataObject>\n")
		closeArr()
		open("UFS_IMAGE_BLOCK_HEADER_TEMPLATES", 0x301D, 0x2009)
		for i, r := range []string{"0 2 254", "0 1 127"} {
			b.WriteString("<DataObject ObjectType=\"UFSImageBlockHeaderTemplate\">\n")
			attr("DICOM_BITS_STORED", 0x0028, 0x0101, "IUInt16", fmt.Sprint(s.BitsStored))
			ranges(r, r, fmt.Sprintf("%d 1 %d", i, i))
			attr("UFS_IMAGE_BLOCK_COMPRESSION_METHOD", 0x301D, 0x200F, "IString", fmt.Sprint(s.Compressor))
			b.WriteString("</DataObject>\n")
		}
		closeArr()
		// Template parity picks LL, field order.
		open("UFS_IMAGE_CLUSTER_HEADER_TEMPLATES", 0x301D, 0x2016)
		for sc := 0; sc <= top; sc++ {
			for _, withLL := range []bool{false, true} {
				b.WriteString("<DataObject ObjectType=\"UFSImageClusterHeaderTemplate\">\n")
				ranges("0 1 0", "0 1 0", "0 1 2", fmt.Sprintf("%d 1 %d", sc, sc), "1 1 3")
				open("UFS_IMAGE_BLOCK_HEADERS", 0x301D, 0x200D)
				kinds := []bool{false}
				if withLL {
					kinds = append(kinds, true)
				}
				fiveDims := withLL || sc%2 == 1
				for _, ll := range kinds {
					for col := 0; col < 3; col++ {
						d := 0
						w := 0
						if ll {
							d, w = 1<<sc, 2
						}
						coord := fmt.Sprintf("%d %d 0 %d", d, d, col)
						if fiveDims {
							coord = fmt.Sprintf("%d %d 0 %d %d", d, d, w, col)
						}
						b.WriteString("<DataObject ObjectType=\"UFSImageBlockHeader\">\n")
						attr("UFS_IMAGE_BLOCK_COORDINATE", 0x301D, 0x200E, "IUInt32Array", coord)
						attr("UFS_IMAGE_BLOCK_HEADER_TEMPLATE_ID", 0x301D, 0x2012, "IUInt32", "0")
						b.WriteString("</DataObject>\n")
					}
				}
				closeArr()
				order := "0 1 3 2"
				if fiveDims {
					order = "0 1 3 4 2"
				}
				attr("UFS_IMAGE_DIMENSIONS_IN_CLUSTER", 0x301D, 0x2021, "IUInt16Array", order)
				b.WriteString("</DataObject>\n")
			}
		}
		closeArr()
		attr("UFS_IMAGE_CLUSTER_HEADER_TABLE", 0x301D, 0x201F, "IString", wrap(base64.StdEncoding.EncodeToString(tbl), "\r\n"))
		b.WriteString("</DataObject>\n")

		for _, img := range []struct {
			kind string
			data []byte
		}{{"MACROIMAGE", s.Macro}, {"LABELIMAGE", s.Label}} {
			b.WriteString("<DataObject ObjectType=\"DPScannedImage\">\n")
			attr("PIM_DP_IMAGE_TYPE", 0x301D, 0x1004, "IString", img.kind)
			attr("PIM_DP_IMAGE_DATA", 0x301D, 0x1005, "IString", wrap(base64.StdEncoding.EncodeToString(img.data), "&#x0D;&#x0A;"))
			b.WriteString("</DataObject>\n")
		}
		closeArr()
		attr("PIM_DP_UFS_INTERFACE_VERSION", 0x301D, 0x1001, "IString", s.Version)
		b.WriteString("</DataObject>\n")
		return b.String()
	}

	// Offsets change values, not table size.
	tbl, _ := table(0)
	head := xmlFor(tbl) + "\x00\x00\r\n\x04" + "\xe0\x7f\x10\x00\xff\xff\xff\xff"
	tbl, blocks := table(int64(len(head)))
	out := []byte(xmlFor(tbl) + "\x00\x00\r\n\x04" + "\xe0\x7f\x10\x00\xff\xff\xff\xff")
	for _, c := range clusters {
		for _, b := range c.blocks {
			out = append(out, b...)
		}
	}
	return out, blocks
}

func wrap(s, sep string) string {
	var b strings.Builder
	for len(s) > 76 {
		b.WriteString(s[:76] + sep)
		s = s[76:]
	}
	b.WriteString(s)
	return b.String()
}
