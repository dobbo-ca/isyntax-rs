package fixture

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strings"
)

// fileV1 builds a v1 file.
func fileV1(s FileSpec) ([]byte, []Block) {
	pad := 3<<s.Levels - 3
	gw, gh := s.Grid()
	top := s.Levels - 1

	type entry struct {
		tx, ty, colour, scale int
		ll                    bool
		x, y                  int
		data                  []byte
	}
	var entries []entry
	for sc := top; sc >= 0; sc-- {
		for ty := 0; ty < gh>>sc; ty++ {
			for tx := 0; tx < gw>>sc; tx++ {
				if s.Exists != nil && !s.Exists(sc, tx, ty) {
					continue
				}
				kinds := []bool{false}
				if sc == top {
					kinds = append(kinds, true)
				}
				for _, ll := range kinds {
					off := 3<<sc - 2
					if ll {
						off += 1 << sc
					}
					x := s.OffsetX + off + tx*(2*blockSize<<sc)
					y := s.OffsetY + off + ty*(2*blockSize<<sc)
					for col := 0; col < 3; col++ {
						n := 3
						if ll {
							n = 1
						}
						p := HulskenParams{Version: 1, ZeroSymbol: byte(col * 7), CounterBits: 3 + col}
						data := EncodeHulsken(s.Coef(sc, tx, ty, col, ll), n, blockSize, blockSize, p)
						entries = append(entries, entry{tx, ty, col, sc, ll, x, y, data})
					}
				}
			}
		}
	}

	// blockID mirrors container.go's seektable index.
	tileCount := make([]int, s.Levels)
	base := gw * gh
	var total int64
	for i := range tileCount {
		tileCount[i] = base >> (2 * i)
		total += int64(tileCount[i])
	}
	total += int64(tileCount[top])
	blockID := func(e entry) int64 {
		maxScale := e.scale
		if e.ll {
			maxScale++
		}
		var id int64
		for i := 0; i < maxScale; i++ {
			id += int64(tileCount[i])
		}
		stride := int64(gw >> e.scale)
		id += int64(e.ty)*stride + int64(e.tx) + int64(e.colour)*total
		return id
	}

	var maxID int64
	for _, e := range entries {
		if id := blockID(e); id > maxID {
			maxID = id
		}
	}
	seektable := make([]byte, (maxID+1)*40)

	// Block header table: partial entries.
	var table []byte
	tag := func(g, e uint16, size int) {
		table = binary.LittleEndian.AppendUint16(table, g)
		table = binary.LittleEndian.AppendUint16(table, e)
		table = binary.LittleEndian.AppendUint32(table, uint32(size))
	}
	for _, e := range entries {
		tag(0xFFFE, 0xE000, 40)
		tag(0x301D, 0x200E, 20)
		for _, v := range []int{e.x, e.y, e.colour, e.scale} {
			table = binary.LittleEndian.AppendUint32(table, uint32(v))
		}
		coeff := uint32(1)
		if e.ll {
			coeff = 0
		}
		table = binary.LittleEndian.AppendUint32(table, coeff)
		tag(0x301D, 0x2012, 4)
		table = binary.LittleEndian.AppendUint32(table, 0)
	}
	tbl := append(binary.LittleEndian.AppendUint32(nil, uint32(len(table))), table...)

	head := []byte(xmlForV1(s, pad, top, tbl) + "\x00\x00\r\n\x04")
	seekTag := make([]byte, 8)
	binary.LittleEndian.PutUint16(seekTag, 0x301D)
	binary.LittleEndian.PutUint16(seekTag[2:], 0x2015)
	binary.LittleEndian.PutUint32(seekTag[4:], uint32(len(seektable)))

	out := append(head, seekTag...)
	dataStart := int64(len(out)) + int64(len(seektable))
	var blocks []Block
	for _, e := range entries {
		id := blockID(e)
		p := id * 40
		binary.LittleEndian.PutUint64(seektable[p+16:], uint64(dataStart))
		binary.LittleEndian.PutUint64(seektable[p+32:], uint64(len(e.data)))
		blocks = append(blocks, Block{Scale: e.scale, TX: e.tx, TY: e.ty, Colour: e.colour, LL: e.ll,
			Offset: dataStart, Size: int64(len(e.data))})
		dataStart += int64(len(e.data))
	}
	out = append(out, seektable...)
	for _, e := range entries {
		out = append(out, e.data...)
	}
	return out, blocks
}

// xmlForV1: v1's XML layout.
func xmlForV1(s FileSpec, pad, top int, tbl []byte) string {
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
	attr("UFS_IMAGE_BLOCK_HEADER_TABLE", 0x301D, 0x2014, "IString", wrap(base64.StdEncoding.EncodeToString(tbl), "\r\n"))
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
