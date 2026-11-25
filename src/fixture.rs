//! Synthetic iSyntax inputs for tests: Hulsken codeblocks and whole iSyntax2 and v1 files. Port of
//! `internal/fixture`. Nothing here comes from slide data.

use crate::testutil::base64_encode;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write;
use std::sync::Arc;

/// The encoder's free parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HulskenParams {
    /// 1 or 2, per compressor.
    pub version: u8,
    /// The symbol that starts a zero run.
    pub zero_symbol: u8,
    /// Bits per zero-run counter chunk, 1 to 8.
    pub counter_bits: usize,
}

struct BitWriter {
    buf: Vec<u8>,
    nbit: usize,
}

impl BitWriter {
    fn write(&mut self, v: u64, n: usize) {
        for i in 0..n {
            if self.nbit / 8 >= self.buf.len() {
                self.buf.push(0);
            }
            self.buf[self.nbit / 8] |= (((v >> i) & 1) as u8) << (self.nbit % 8);
            self.nbit += 1;
        }
    }
}

struct Token {
    sym: u8,
    /// A counter chunk, or -1 for none.
    chunk: i32,
}

fn emit_run(toks: &mut Vec<Token>, p: HulskenParams, r: usize) {
    let mut s = if p.version == 2 { r - 1 } else { r };
    let mut chunks = Vec::new();
    while s > 0 {
        chunks.push((s & ((1 << p.counter_bits) - 1)) as i32);
        s >>= p.counter_bits;
    }
    for c in chunks.into_iter().rev() {
        toks.push(Token {
            sym: p.zero_symbol,
            chunk: c,
        });
    }
}

fn literal(toks: &mut Vec<Token>, p: HulskenParams, b: u8) {
    // An escape for the zero symbol itself.
    let chunk = if b == p.zero_symbol { 0 } else { -1 };
    toks.push(Token { sym: b, chunk });
}

fn tokenize(data: &[u8], p: HulskenParams) -> Vec<Token> {
    let min_run = if p.version == 2 { 2 } else { 1 };
    let mut toks = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if data[i] != 0 {
            literal(&mut toks, p, data[i]);
            i += 1;
            continue;
        }
        let mut j = i;
        while j < data.len() && data[j] == 0 {
            j += 1;
        }
        let r = j - i;
        // Runs before an escape absorb it.
        let next_escape = j < data.len() && data[j] == p.zero_symbol && p.zero_symbol != 0;
        if p.zero_symbol == 0 && r < min_run {
            literal(&mut toks, p, 0);
        } else if p.zero_symbol != 0 && (r < min_run || (next_escape && r - 1 < min_run)) {
            for _ in 0..r {
                literal(&mut toks, p, 0);
            }
        } else if next_escape {
            emit_run(&mut toks, p, r - 1);
            literal(&mut toks, p, 0);
        } else {
            emit_run(&mut toks, p, r);
        }
        i = j;
    }
    toks
}

struct HNode {
    /// -1 for internal.
    sym: i32,
    left: usize,
    right: usize,
}

fn walk(
    nodes: &[HNode],
    n: usize,
    code: u32,
    depth: usize,
    codes: &mut [u32; 256],
    lens: &mut [usize; 256],
) -> bool {
    let node = &nodes[n];
    if node.sym >= 0 {
        codes[node.sym as usize] = code;
        lens[node.sym as usize] = depth;
        return depth <= 16;
    }
    let a = walk(nodes, node.left, code, depth + 1, codes, lens);
    let b = walk(
        nodes,
        node.right,
        code | (1 << depth),
        depth + 1,
        codes,
        lens,
    );
    a && b
}

/// Codes up to 16 bits deep; returns the nodes and the root.
fn huffman(freq: &[usize; 256]) -> ([u32; 256], [usize; 256], Vec<HNode>, usize) {
    let mut f = *freq;
    loop {
        let mut nodes: Vec<HNode> = Vec::new();
        let mut heap = BinaryHeap::new();
        for (s, &n) in f.iter().enumerate() {
            if n > 0 {
                nodes.push(HNode {
                    sym: s as i32,
                    left: 0,
                    right: 0,
                });
                heap.push(Reverse((n, s, nodes.len() - 1)));
            }
        }
        if heap.is_empty() {
            nodes.push(HNode {
                sym: 0,
                left: 0,
                right: 0,
            });
            heap.push(Reverse((1, 0, 0)));
        }
        let mut order = 256;
        while heap.len() > 1 {
            let Reverse((fa, _, a)) = heap.pop().unwrap();
            let Reverse((fb, _, b)) = heap.pop().unwrap();
            order += 1;
            nodes.push(HNode {
                sym: -1,
                left: a,
                right: b,
            });
            heap.push(Reverse((fa + fb, order, nodes.len() - 1)));
        }
        let root = heap.pop().unwrap().0.2;
        let (mut codes, mut lens) = ([0u32; 256], [0usize; 256]);
        if walk(&nodes, root, 0, 0, &mut codes, &mut lens) {
            return (codes, lens, nodes, root);
        }
        for v in f.iter_mut() {
            if *v > 0 {
                *v = *v / 2 + 1;
            }
        }
    }
}

/// Pre-order: internal then leaf bits.
fn write_tree(w: &mut BitWriter, nodes: &[HNode], n: usize) {
    let node = &nodes[n];
    if node.sym >= 0 {
        w.write(1, 1);
        w.write(node.sym as u64, 8);
        return;
    }
    w.write(0, 1);
    write_tree(w, nodes, node.left);
    write_tree(w, nodes, node.right);
}

/// Encodes `n` coefficient planes of `bw * bh` values in raster order as one codeblock, the
/// inverse of docs/format.md "Entropy coding".
pub(crate) fn encode_hulsken(
    coef: &[i16],
    n: usize,
    bw: usize,
    bh: usize,
    p: HulskenParams,
) -> Vec<u8> {
    let area = bw * bh;
    let plane_bytes = area / 8;
    // Sign-magnitude, in 4x4 element order.
    let mut sm = vec![0u16; n * area];
    for c in 0..n {
        for e in 0..area {
            let (a, w) = (e / 16, e % 16);
            let x = (a % (bw / 4)) * 4 + w % 4;
            let y = (a / (bw / 4)) * 4 + w / 4;
            let v = i32::from(coef[c * area + y * bw + x]);
            sm[c * area + e] = if v < 0 {
                (-v) as u16 | 0x8000
            } else {
                v as u16
            };
        }
    }
    // The value bit for mask bit b.
    let target = |b: usize| -> usize {
        if p.version == 1 {
            if b == 0 { 15 } else { b - 1 }
        } else {
            15 - b
        }
    };
    let mut masks = vec![0u16; n];
    for (c, m) in masks.iter_mut().enumerate() {
        for b in 0..16 {
            if (0..area).any(|e| (sm[c * area + e] >> target(b)) & 1 != 0) {
                *m |= 1 << b;
            }
        }
    }
    let plane = |c: usize, bit: usize| -> Vec<u8> {
        let mut out = vec![0u8; plane_bytes];
        for e in 0..area {
            out[e / 8] |= (((sm[c * area + e] >> bit) & 1) as u8) << (e % 8);
        }
        out
    };
    let mut data = Vec::new();
    if p.version == 1 {
        for (c, &m) in masks.iter().enumerate() {
            for b in 0..16 {
                if (m >> b) & 1 != 0 {
                    data.extend(plane(c, target(b)));
                }
            }
        }
        if !masks.iter().all(|&m| m == 0xFFFF) {
            for &m in &masks {
                data.extend_from_slice(&m.to_le_bytes());
            }
        }
    } else {
        for b in 0..16 {
            for (c, &m) in masks.iter().enumerate() {
                if (m >> b) & 1 != 0 {
                    data.extend(plane(c, target(b)));
                }
            }
        }
    }

    let mut out = Vec::new();
    if p.version == 1 {
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    } else {
        for &m in &masks {
            out.extend_from_slice(&m.to_le_bytes());
        }
    }
    out.push(p.zero_symbol);
    out.push(p.counter_bits as u8);
    let nbit = out.len() * 8;
    let mut w = BitWriter { buf: out, nbit };
    if p.version == 2 && !data.is_empty() {
        let agg = masks.iter().fold(0u16, |a, m| a | m);
        let ptr_bits = (usize::BITS - data.len().leading_zeros()) as usize - 1 + 5;
        for _ in 0..(agg.count_ones() as usize).saturating_sub(1) {
            w.write(0, ptr_bits); // the seek table, skipped by decoders
        }
    }

    let toks = tokenize(&data, p);
    let mut freq = [0usize; 256];
    for t in &toks {
        freq[t.sym as usize] += 1;
    }
    let (codes, lens, nodes, root) = huffman(&freq);
    write_tree(&mut w, &nodes, root);
    for t in &toks {
        let s = t.sym as usize;
        w.write(u64::from(codes[s]), lens[s].max(1));
        if t.chunk >= 0 {
            w.write(t.chunk as u64, p.counter_bits);
        }
    }
    w.buf
}

type ExistsFn = Arc<dyn Fn(usize, usize, usize) -> bool + Send + Sync>;
type CoefFn = Arc<dyn Fn(usize, usize, usize, usize, bool) -> Vec<i16> + Send + Sync>;

/// A synthetic iSyntax file, v1 or iSyntax2. A version whose major (before the first '.') is under
/// 100 selects the v1 layout: a block header table and seektable instead of cluster templates and
/// table.
#[derive(Clone)]
pub(crate) struct FileSpec {
    /// `PIM_DP_UFS_INTERFACE_VERSION`, default "100.5".
    pub version: String,
    /// 16 or 19.
    pub compressor: u32,
    /// 8 or 9.
    pub bits_stored: u32,
    pub levels: usize,
    /// The image size without the wavelet padding.
    pub width: usize,
    pub height: usize,
    pub offset_x: usize,
    pub offset_y: usize,
    pub mpp: f64,
    /// Whether a tile has codeblocks; `None` means every tile.
    pub exists: Option<ExistsFn>,
    /// A codeblock's 128x128 coefficients in raster order: three bands (HL, LH, HH) for detail
    /// blocks and one for LL.
    pub coef: CoefFn,
    pub label: Vec<u8>,
    pub macro_image: Vec<u8>,
    pub icc: Vec<u8>,
    /// Raw XML for root-level attributes, e.g. calibration tags.
    pub extra: String,
}

impl FileSpec {
    pub(crate) fn new(
        coef: impl Fn(usize, usize, usize, usize, bool) -> Vec<i16> + Send + Sync + 'static,
    ) -> FileSpec {
        FileSpec {
            version: String::new(),
            compressor: 19,
            bits_stored: 8,
            levels: 3,
            width: 1500,
            height: 900,
            offset_x: 0,
            offset_y: 0,
            mpp: 0.25,
            exists: None,
            coef: Arc::new(coef),
            label: Vec::new(),
            macro_image: Vec::new(),
            icc: Vec::new(),
            extra: String::new(),
        }
    }

    pub(crate) fn set_exists(
        &mut self,
        f: impl Fn(usize, usize, usize) -> bool + Send + Sync + 'static,
    ) {
        self.exists = Some(Arc::new(f));
    }

    pub(crate) fn set_coef(
        &mut self,
        f: impl Fn(usize, usize, usize, usize, bool) -> Vec<i16> + Send + Sync + 'static,
    ) {
        self.coef = Arc::new(f);
    }

    /// The tile grid of docs/format.md "Levels and tiles".
    pub(crate) fn grid(&self) -> (usize, usize) {
        let pad = (3usize << self.levels) - 3;
        let span = 128usize << self.levels;
        (
            (self.width + 2 * pad).div_ceil(span) << (self.levels - 1),
            (self.height + 2 * pad).div_ceil(span) << (self.levels - 1),
        )
    }

    fn has(&self, s: usize, x: usize, y: usize) -> bool {
        self.exists.as_ref().is_none_or(|f| f(s, x, y))
    }

    fn is_v1(&self) -> bool {
        let v = if self.version.is_empty() {
            "100.5"
        } else {
            &self.version
        };
        let maj = v.split('.').next().unwrap_or("");
        maj.parse::<i64>().is_ok_and(|n| n < 100)
    }

    fn version_str(&self) -> &str {
        if self.version.is_empty() {
            "100.5"
        } else {
            &self.version
        }
    }
}

/// Where `file` put one codeblock.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockInfo {
    pub scale: usize,
    pub tx: usize,
    pub ty: usize,
    pub colour: usize,
    pub ll: bool,
    pub offset: i64,
    pub size: i64,
}

fn wrap(s: &str, sep: &str) -> String {
    let mut out = String::new();
    let mut s = s;
    while s.len() > 76 {
        out.push_str(&s[..76]);
        out.push_str(sep);
        s = &s[76..];
    }
    out.push_str(s);
    out
}

/// The XML writer both layouts share.
struct Xml(String);

impl Xml {
    fn attr(&mut self, name: &str, group: u16, elem: u16, vr: &str, value: &str) {
        let _ = writeln!(
            self.0,
            "<Attribute Name={name:?} Group=\"0x{group:04x}\" Element=\"0x{elem:04x}\" PMSVR={vr:?}>{value}</Attribute>"
        );
    }

    fn open(&mut self, name: &str, group: u16, elem: u16) {
        let _ = writeln!(
            self.0,
            "<Attribute Name={name:?} Group=\"0x{group:04x}\" Element=\"0x{elem:04x}\" PMSVR=\"IDataObjectArray\">\n<Array>"
        );
    }

    fn close(&mut self) {
        self.0.push_str("</Array>\n</Attribute>\n");
    }

    fn ranges(&mut self, rs: &[String]) {
        self.open("UFS_IMAGE_DIMENSION_RANGES", 0x301D, 0x200A);
        for r in rs {
            self.0
                .push_str("<DataObject ObjectType=\"UFSImageDimensionRange\">\n");
            self.attr(
                "UFS_IMAGE_DIMENSION_RANGE",
                0x301D,
                0x200B,
                "IUInt32Array",
                r,
            );
            self.0.push_str("</DataObject>\n");
        }
        self.close();
    }

    /// The part of the header both layouts share, up to and including the block header templates.
    fn common(&mut self, s: &FileSpec) {
        let pad = (3usize << s.levels) - 3;
        let top = s.levels - 1;
        self.0.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<DataObject ObjectType=\"DPUfsImport\">\n");
        self.attr("DICOM_MANUFACTURER", 0x0008, 0x0070, "IString", "PHILIPS");
        self.0.push_str(&s.extra);
        self.open("PIM_DP_SCANNED_IMAGES", 0x301D, 0x1003);

        self.0
            .push_str("<DataObject ObjectType=\"DPScannedImage\">\n");
        self.attr("PIM_DP_IMAGE_TYPE", 0x301D, 0x1004, "IString", "WSI");
        self.open("DP_COLOR_MANAGEMENT", 0x301D, 0x1013);
        self.0
            .push_str("<DataObject ObjectType=\"DPColorManagement\">\n");
        self.attr(
            "DICOM_ICCPROFILE",
            0x0028,
            0x2000,
            "IString",
            &(base64_encode(&s.icc) + "/"),
        );
        self.0.push_str("</DataObject>\n");
        self.close();
        self.open("UFS_IMAGE_GENERAL_HEADERS", 0x301D, 0x2000);
        self.0
            .push_str("<DataObject ObjectType=\"UFSImageGeneralHeader\">\n");
        self.open("UFS_IMAGE_DIMENSIONS", 0x301D, 0x2003);
        for i in 0..5 {
            self.0
                .push_str("<DataObject ObjectType=\"UFSImageDimension\">\n");
            if i < 2 {
                self.attr(
                    "UFS_IMAGE_DIMENSION_SCALE_FACTOR",
                    0x301D,
                    0x2007,
                    "IDouble",
                    &s.mpp.to_string(),
                );
            }
            self.0.push_str("</DataObject>\n");
        }
        self.close();
        self.ranges(&[
            format!("{} 1 {}", s.offset_x, s.offset_x + s.width + 2 * pad - 1),
            format!("{} 1 {}", s.offset_y, s.offset_y + s.height + 2 * pad - 1),
            "0 1 2".to_string(),
            format!("0 1 {top}"),
            "0 1 3".to_string(),
        ]);
        self.0.push_str("</DataObject>\n");
        self.close();
        self.open("UFS_IMAGE_BLOCK_HEADER_TEMPLATES", 0x301D, 0x2009);
        for (i, r) in ["0 2 254", "0 1 127"].iter().enumerate() {
            self.0
                .push_str("<DataObject ObjectType=\"UFSImageBlockHeaderTemplate\">\n");
            self.attr(
                "DICOM_BITS_STORED",
                0x0028,
                0x0101,
                "IUInt16",
                &s.bits_stored.to_string(),
            );
            self.ranges(&[r.to_string(), r.to_string(), format!("{i} 1 {i}")]);
            self.attr(
                "UFS_IMAGE_BLOCK_COMPRESSION_METHOD",
                0x301D,
                0x200F,
                "IString",
                &s.compressor.to_string(),
            );
            self.0.push_str("</DataObject>\n");
        }
        self.close();
    }

    /// The label and macro images, the closing of the scanned images, the version.
    fn finish(&mut self, s: &FileSpec) {
        for (kind, data) in [("MACROIMAGE", &s.macro_image), ("LABELIMAGE", &s.label)] {
            self.0
                .push_str("<DataObject ObjectType=\"DPScannedImage\">\n");
            self.attr("PIM_DP_IMAGE_TYPE", 0x301D, 0x1004, "IString", kind);
            self.attr(
                "PIM_DP_IMAGE_DATA",
                0x301D,
                0x1005,
                "IString",
                &wrap(&base64_encode(data), "&#x0D;&#x0A;"),
            );
            self.0.push_str("</DataObject>\n");
        }
        self.close();
        self.attr(
            "PIM_DP_UFS_INTERFACE_VERSION",
            0x301D,
            0x1001,
            "IString",
            s.version_str(),
        );
        self.0.push_str("</DataObject>\n");
    }
}

struct Cluster {
    x: usize,
    y: usize,
    scale: usize,
    tmpl: usize,
    blocks: Vec<Vec<u8>>,
    meta: Vec<BlockInfo>,
}

fn params(version: u8, col: usize) -> HulskenParams {
    HulskenParams {
        version,
        zero_symbol: (col * 7) as u8,
        counter_bits: 3 + col,
    }
}

fn le32(t: &mut Vec<u8>, v: usize) {
    t.extend_from_slice(&(v as u32).to_le_bytes());
}

fn tag(t: &mut Vec<u8>, group: u16, elem: u16, size: usize) {
    t.extend_from_slice(&group.to_le_bytes());
    t.extend_from_slice(&elem.to_le_bytes());
    le32(t, size);
}

/// Builds the file and lists its codeblocks.
pub(crate) fn file(s: &FileSpec) -> (Vec<u8>, Vec<BlockInfo>) {
    if s.is_v1() {
        return file_v1(s);
    }
    let version = if s.compressor == 16 { 1 } else { 2 };
    let (gw, gh) = s.grid();
    let top = s.levels - 1;

    // One cluster per tile, plus LL.
    let mut clusters: Vec<Cluster> = Vec::new();
    for sc in (0..=top).rev() {
        for ty in 0..gh >> sc {
            for tx in 0..gw >> sc {
                if !s.has(sc, tx, ty) {
                    continue;
                }
                let mut c = Cluster {
                    x: s.offset_x + (3 << sc) - 2 + tx * ((2 * 128) << sc),
                    y: s.offset_y + (3 << sc) - 2 + ty * ((2 * 128) << sc),
                    scale: sc,
                    tmpl: 2 * sc,
                    blocks: Vec::new(),
                    meta: Vec::new(),
                };
                let mut kinds = vec![false];
                if sc == top {
                    kinds.push(true);
                    c.tmpl += 1;
                }
                for ll in kinds {
                    for col in 0..3 {
                        let n = if ll { 1 } else { 3 };
                        c.blocks.push(encode_hulsken(
                            &(s.coef)(sc, tx, ty, col, ll),
                            n,
                            128,
                            128,
                            params(version, col),
                        ));
                        c.meta.push(BlockInfo {
                            scale: sc,
                            tx,
                            ty,
                            colour: col,
                            ll,
                            offset: 0,
                            size: 0,
                        });
                    }
                }
                clusters.push(c);
            }
        }
    }

    // A fixed-width table: the offsets are filled in after.
    let table = |data_start: i64| -> (Vec<u8>, Vec<BlockInfo>) {
        let mut t = Vec::new();
        let mut blocks = Vec::new();
        let mut off = data_start;
        for c in &clusters {
            let m = c.blocks.len();
            let item_size = 8 + 12 + 8 + 4 + 8 + 8 + 8 + 8 + 8 + 4 * m + 8 + 4 * m;
            tag(&mut t, 0xFFFE, 0xE000, item_size);
            tag(&mut t, 0x301D, 0x2020, 12);
            for v in [c.x, c.y, c.scale] {
                le32(&mut t, v);
            }
            tag(&mut t, 0x301D, 0x2019, 4);
            le32(&mut t, c.tmpl);
            let total: i64 = c.blocks.iter().map(|b| b.len() as i64).sum();
            tag(&mut t, 0x301D, 0x2010, 8);
            t.extend_from_slice(&(off as u64).to_le_bytes());
            tag(&mut t, 0x301D, 0x2011, 8);
            t.extend_from_slice(&(total as u64).to_le_bytes());
            tag(&mut t, 0x301D, 0x201B, 4 * m);
            let mut rel = 0i64;
            for (b, meta) in c.blocks.iter().zip(&c.meta) {
                le32(&mut t, rel as usize);
                blocks.push(BlockInfo {
                    offset: off + rel,
                    size: b.len() as i64,
                    ..*meta
                });
                rel += b.len() as i64;
            }
            tag(&mut t, 0x301D, 0x201C, 4 * m);
            for b in &c.blocks {
                le32(&mut t, b.len());
            }
            off += total;
        }
        let mut out = (t.len() as u32).to_le_bytes().to_vec();
        out.extend(t);
        (out, blocks)
    };

    let xml_for = |tbl: &[u8]| -> String {
        let top = s.levels - 1;
        let mut x = Xml(String::new());
        x.common(s);
        // Template parity picks LL, field order.
        x.open("UFS_IMAGE_CLUSTER_HEADER_TEMPLATES", 0x301D, 0x2016);
        for sc in 0..=top {
            for with_ll in [false, true] {
                x.0.push_str("<DataObject ObjectType=\"UFSImageClusterHeaderTemplate\">\n");
                x.ranges(&[
                    "0 1 0".to_string(),
                    "0 1 0".to_string(),
                    "0 1 2".to_string(),
                    format!("{sc} 1 {sc}"),
                    "1 1 3".to_string(),
                ]);
                x.open("UFS_IMAGE_BLOCK_HEADERS", 0x301D, 0x200D);
                let kinds: &[bool] = if with_ll { &[false, true] } else { &[false] };
                let five_dims = with_ll || sc % 2 == 1;
                for &ll in kinds {
                    for col in 0..3 {
                        let (d, w) = if ll { (1usize << sc, 2) } else { (0, 0) };
                        let coord = if five_dims {
                            format!("{d} {d} 0 {w} {col}")
                        } else {
                            format!("{d} {d} 0 {col}")
                        };
                        x.0.push_str("<DataObject ObjectType=\"UFSImageBlockHeader\">\n");
                        x.attr(
                            "UFS_IMAGE_BLOCK_COORDINATE",
                            0x301D,
                            0x200E,
                            "IUInt32Array",
                            &coord,
                        );
                        x.attr(
                            "UFS_IMAGE_BLOCK_HEADER_TEMPLATE_ID",
                            0x301D,
                            0x2012,
                            "IUInt32",
                            "0",
                        );
                        x.0.push_str("</DataObject>\n");
                    }
                }
                x.close();
                let order = if five_dims { "0 1 3 4 2" } else { "0 1 3 2" };
                x.attr(
                    "UFS_IMAGE_DIMENSIONS_IN_CLUSTER",
                    0x301D,
                    0x2021,
                    "IUInt16Array",
                    order,
                );
                x.0.push_str("</DataObject>\n");
            }
        }
        x.close();
        x.attr(
            "UFS_IMAGE_CLUSTER_HEADER_TABLE",
            0x301D,
            0x201F,
            "IString",
            &wrap(&base64_encode(tbl), "\r\n"),
        );
        x.0.push_str("</DataObject>\n");
        x.finish(s);
        x.0
    };

    let tail: &[u8] = b"\x00\x00\r\n\x04\xe0\x7f\x10\x00\xff\xff\xff\xff";
    // Offsets change values, not the table's size.
    let (tbl, _) = table(0);
    let head_len = xml_for(&tbl).len() + tail.len();
    let (tbl, blocks) = table(head_len as i64);
    let mut out = xml_for(&tbl).into_bytes();
    out.extend_from_slice(tail);
    for c in &clusters {
        for b in &c.blocks {
            out.extend_from_slice(b);
        }
    }
    (out, blocks)
}

/// Builds a v1 file.
fn file_v1(s: &FileSpec) -> (Vec<u8>, Vec<BlockInfo>) {
    let (gw, gh) = s.grid();
    let top = s.levels - 1;

    struct Entry {
        tx: usize,
        ty: usize,
        colour: usize,
        scale: usize,
        ll: bool,
        x: usize,
        y: usize,
        data: Vec<u8>,
    }
    let mut entries: Vec<Entry> = Vec::new();
    for sc in (0..=top).rev() {
        for ty in 0..gh >> sc {
            for tx in 0..gw >> sc {
                if !s.has(sc, tx, ty) {
                    continue;
                }
                let kinds: &[bool] = if sc == top { &[false, true] } else { &[false] };
                for &ll in kinds {
                    let mut off = (3 << sc) - 2;
                    if ll {
                        off += 1 << sc;
                    }
                    let x = s.offset_x + off + tx * ((2 * 128) << sc);
                    let y = s.offset_y + off + ty * ((2 * 128) << sc);
                    for col in 0..3 {
                        let n = if ll { 1 } else { 3 };
                        let data = encode_hulsken(
                            &(s.coef)(sc, tx, ty, col, ll),
                            n,
                            128,
                            128,
                            params(1, col),
                        );
                        entries.push(Entry {
                            tx,
                            ty,
                            colour: col,
                            scale: sc,
                            ll,
                            x,
                            y,
                            data,
                        });
                    }
                }
            }
        }
    }

    // block_id mirrors container.rs's seektable index.
    let tile_count: Vec<usize> = (0..s.levels).map(|i| (gw * gh) >> (2 * i)).collect();
    let total: usize = tile_count.iter().sum::<usize>() + tile_count[top];
    let block_id = |e: &Entry| -> usize {
        let max_scale = if e.ll { e.scale + 1 } else { e.scale };
        let id: usize = tile_count.iter().take(max_scale).sum();
        let stride = gw >> e.scale;
        id + e.ty * stride + e.tx + e.colour * total
    };
    let max_id = entries.iter().map(&block_id).max().unwrap_or(0);
    let mut seektable = vec![0u8; (max_id + 1) * 40];

    // The block header table: partial entries.
    let mut table = Vec::new();
    for e in &entries {
        tag(&mut table, 0xFFFE, 0xE000, 40);
        tag(&mut table, 0x301D, 0x200E, 20);
        for v in [e.x, e.y, e.colour, e.scale] {
            le32(&mut table, v);
        }
        le32(&mut table, if e.ll { 0 } else { 1 });
        tag(&mut table, 0x301D, 0x2012, 4);
        le32(&mut table, 0);
    }
    let mut tbl = (table.len() as u32).to_le_bytes().to_vec();
    tbl.extend(table);

    let mut x = Xml(String::new());
    x.common(s);
    x.attr(
        "UFS_IMAGE_BLOCK_HEADER_TABLE",
        0x301D,
        0x2014,
        "IString",
        &wrap(&base64_encode(&tbl), "\r\n"),
    );
    x.0.push_str("</DataObject>\n");
    x.finish(s);
    let mut head = x.0.into_bytes();
    head.extend_from_slice(b"\x00\x00\r\n\x04");
    let mut seek_tag = Vec::new();
    tag(&mut seek_tag, 0x301D, 0x2015, seektable.len());

    let mut out = head;
    out.extend(seek_tag);
    let mut data_start = (out.len() + seektable.len()) as i64;
    let mut blocks = Vec::new();
    for e in &entries {
        let p = block_id(e) * 40;
        seektable[p + 16..p + 24].copy_from_slice(&(data_start as u64).to_le_bytes());
        seektable[p + 32..p + 40].copy_from_slice(&(e.data.len() as u64).to_le_bytes());
        blocks.push(BlockInfo {
            scale: e.scale,
            tx: e.tx,
            ty: e.ty,
            colour: e.colour,
            ll: e.ll,
            offset: data_start,
            size: e.data.len() as i64,
        });
        data_start += e.data.len() as i64;
    }
    out.extend(seektable);
    for e in &entries {
        out.extend_from_slice(&e.data);
    }
    (out, blocks)
}
