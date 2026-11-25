//! The iSyntax header: the XML up to the 0x04 terminator, its base64 attributes, and either v1's
//! block header table and seektable or iSyntax2's cluster templates and cluster table, into a
//! codeblock index by tile. Port of `internal/container`; see isyntax-go's docs/format.md, "File
//! layout" and "Codeblock tables".

use crate::budget::Acct;
use crate::error::{check_cancel, header as hdr};
use crate::types::{Point, ReadAt, read_full};
use crate::xml::{self, XObject};
use crate::{Error, Limits};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// The codeblock side; tiles are twice that.
pub(crate) const BLOCK_SIZE: i64 = 128;

const MAX_CODEBLOCK: i64 = 1 << 24;
const MAX_LEVELS: i64 = 16;
/// Bytes charged per codeblock found: the entry, and its place in the tile index.
const CODEBLOCK_COST: u64 = 48 + 160;

/// What the decoder needs from the XML header of the WSI image.
#[derive(Debug, Clone, Default)]
pub(crate) struct Header {
    /// `PIM_DP_UFS_INTERFACE_VERSION`'s major; under 100 is v1.
    pub major: i64,
    #[allow(dead_code)] // read by nothing, as in Go
    pub minor: i64,
    /// The raw `PIM_DP_UFS_INTERFACE_VERSION`, trimmed.
    pub interface_version: Arc<str>,
    /// 16 or 19.
    pub compressor: i64,
    /// 8, 9, or (v1 only) 16.
    pub bits_stored: i64,
    pub offset_x: i64,
    pub offset_y: i64,
    pub padded_w: i64,
    pub padded_h: i64,
    /// 1 to 16.
    pub levels: usize,
    pub width: i64,
    pub height: i64,
    /// The level-0 tile grid.
    pub grid_w: i64,
    pub grid_h: i64,
    pub mpp_x: f64,
    pub mpp_y: f64,
    /// The blobs, strings and polygons are behind `Arc` so that `Image::wide_grid` shares them.
    pub icc: Option<Arc<Vec<u8>>>,
    /// JPEG.
    pub label: Option<Arc<Vec<u8>>>,
    pub macro_image: Option<Arc<Vec<u8>>>,
    pub codeblocks: Vec<Codeblock>,
    /// Planes R, G, B; no YCoCg.
    pub rgb: bool,
    /// `Quality=N` of `DICOM_DERIVATION_DESCRIPTION`; -1 if absent.
    pub quality: i64,

    // Scanner metadata, empty if the tag is absent.
    /// `PIM_DP_UFS_BARCODE`, base64 decoded; sensitive.
    pub barcode: Arc<Vec<u8>>,
    pub model_name: Arc<str>,
    pub device_serial: Arc<str>,
    pub acquisition_date_time: Arc<str>,

    // Best effort: empty or -1 when absent or malformed.
    pub calibration_status: Arc<str>,
    pub calibration_date: Arc<[String]>,
    pub calibration_time: Arc<[String]>,
    pub rack: i64,
    pub slot: i64,

    /// `UFS_IMAGE_VALID_DATA_ENVELOPES`, one polygon per header object, vertices in level-0 pixels
    /// already shifted by the pyramid pad, in file order.
    pub envelopes: Arc<Vec<Vec<Point>>>,
    /// v1 without the tag.
    pub envelope_default: bool,
}

/// One entry of the cluster table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Codeblock {
    pub x: i64,
    pub y: i64,
    pub colour: i64,
    pub scale: i64,
    pub ll: bool,
    pub offset: i64,
    pub size: i64,
}

fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    v.try_reserve(n).map_err(|_| Error::OutOfMemory)
}

fn reserve_exact<T>(v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)
}

/// `strconv.Atoi` with its error ignored: 0 for a syntax error, the limit for an overflow.
fn atoi_lenient(s: &str) -> i64 {
    use std::num::IntErrorKind::{NegOverflow, PosOverflow};
    match s.parse::<i64>() {
        Ok(v) => v,
        Err(e) => match e.kind() {
            PosOverflow => i64::MAX,
            NegOverflow => i64::MIN,
            _ => 0,
        },
    }
}

fn le16(b: &[u8], i: usize) -> u16 {
    match b.get(i..i + 2) {
        Some(s) => u16::from_le_bytes([s[0], s[1]]),
        None => 0,
    }
}

fn le32(b: &[u8], i: usize) -> u32 {
    match b.get(i..i + 4) {
        Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0,
    }
}

fn le64(b: &[u8], i: usize) -> u64 {
    match b.get(i..i + 8) {
        Some(s) => {
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            u64::from_le_bytes(a)
        }
        None => 0,
    }
}

/// Reads and checks the header of the file `r[0..size]`.
pub(crate) fn parse(
    r: &dyn ReadAt,
    size: i64,
    limits: &Limits,
    cancel: &AtomicBool,
    acct: &mut Acct,
) -> Result<Header, Error> {
    let (xml_bytes, data_offset) = read_xml(r, size, limits, acct, cancel)?;
    let root = xml::parse(&xml_bytes, acct)?;
    drop(xml_bytes);
    let mut h = Header {
        rack: -1,
        slot: -1,
        quality: -1,
        ..Header::default()
    };
    if let Some(a) = root.attr(0x301D, 0x1001) {
        let v = a.value.trim();
        h.interface_version = v.into();
        let (maj, min) = v.split_once('.').unwrap_or((v, ""));
        h.major = atoi_lenient(maj);
        h.minor = atoi_lenient(min);
    }
    let text = |g: u16, e: u16| root.attr(g, e).map(|a| Arc::<str>::from(a.value.trim()));
    h.acquisition_date_time = text(0x0008, 0x002A).unwrap_or_default();
    h.model_name = text(0x0008, 0x1090).unwrap_or_default();
    h.device_serial = text(0x0018, 0x1000).unwrap_or_default();
    h.calibration_status = text(0x101D, 0x100A).unwrap_or_default();
    if let Some(a) = root.attr(0x0018, 0x1200) {
        h.calibration_date = quoted_tokens(&a.value, acct)?;
    }
    if let Some(a) = root.attr(0x0018, 0x1201) {
        h.calibration_time = quoted_tokens(&a.value, acct)?;
    }
    h.rack = uint16_attr(&root, 0x101D, 0x1007);
    h.slot = uint16_attr(&root, 0x101D, 0x1008);
    if let Some(b) = base64_attr(&root, 0x301D, 0x1002, acct)? {
        h.barcode = b;
    }
    let mut wsi: Option<&XObject> = None;
    if let Some(a) = root.attr(0x301D, 0x1003) {
        for img in &a.objects {
            let Some(t) = img.attr(0x301D, 0x1004) else {
                continue;
            };
            match t.value.trim() {
                "WSI" => wsi = Some(img),
                "LABELIMAGE" => h.label = base64_attr(img, 0x301D, 0x1005, acct)?,
                "MACROIMAGE" => h.macro_image = base64_attr(img, 0x301D, 0x1005, acct)?,
                _ => {}
            }
        }
    }
    let Some(wsi) = wsi else {
        return Err(hdr("no WSI image in the header"));
    };
    let mut cx = Ctx {
        rd: r,
        data_offset,
        size,
        limits,
        cancel,
        acct,
    };
    h.parse_wsi(wsi, &mut cx)?;
    Ok(h)
}

/// What `parse_wsi` reads besides the XML.
struct Ctx<'a, 'c> {
    rd: &'a dyn ReadAt,
    data_offset: i64,
    size: i64,
    limits: &'a Limits,
    cancel: &'a AtomicBool,
    acct: &'a mut Acct<'c>,
}

/// What a token costs besides its bytes.
const TOKEN_COST: usize = size_of::<String>();

/// Splits IStringArray quoted tokens, charging `acct` for each before it is allocated: `""` is two
/// bytes of input and a whole `String` here.
fn quoted_tokens(s: &str, acct: &mut Acct) -> Result<Arc<[String]>, Error> {
    let mut out: Vec<String> = Vec::new();
    let mut s = s;
    while let Some((_, rest)) = s.split_once('"')
        && let Some((tok, rest)) = rest.split_once('"')
    {
        acct.tick()?;
        acct.add((TOKEN_COST + tok.len()) as u64)?;
        let mut t = String::new();
        t.try_reserve_exact(tok.len())
            .map_err(|_| Error::OutOfMemory)?;
        t.push_str(tok);
        reserve(&mut out, 1)?;
        out.push(t);
        s = rest;
    }
    Ok(out.into())
}

/// -1 if absent or bad.
fn uint16_attr(o: &XObject, group: u16, elem: u16) -> i64 {
    let Some(a) = o.attr(group, elem) else {
        return -1;
    };
    let v = a.value.trim();
    if v.is_empty() || !v.bytes().all(|c| c.is_ascii_digit()) {
        return -1;
    }
    v.parse::<u16>().map_or(-1, i64::from)
}

/// The decoded attribute: `None` if it is absent or not base64 (as in Go); a limit is an error.
fn base64_attr(
    o: &XObject,
    group: u16,
    elem: u16,
    acct: &mut Acct,
) -> Result<Option<Arc<Vec<u8>>>, Error> {
    let Some(a) = o.find(group, elem) else {
        return Ok(None);
    };
    match decode_base64(&a.value, acct) {
        Ok(v) => Ok(Some(Arc::new(v))),
        Err(e @ (Error::AllocLimit | Error::OutOfMemory)) => Err(e),
        Err(_) => Ok(None),
    }
}

/// Makes room in `out` for `n` more bytes. A full `out` doubles (up to `max_cap`, or what `n`
/// needs), so the reallocations are few; the new capacity is charged, whole, before it is
/// reserved, which also covers the old buffer that is live while the new one is filled.
fn grow(out: &mut Vec<u8>, n: usize, max_cap: usize, acct: &mut Acct) -> Result<(), Error> {
    let need = out.len().checked_add(n).ok_or(Error::AllocLimit)?;
    if need > out.capacity() {
        let cap = (out.capacity().saturating_mul(2)).min(max_cap).max(need);
        acct.add(cap as u64)?;
        reserve_exact(out, cap - out.len())?;
    }
    Ok(())
}

/// The header bytes, and the offset after the 0x04 that ends them.
fn read_xml(
    r: &dyn ReadAt,
    size: i64,
    limits: &Limits,
    acct: &mut Acct,
    cancel: &AtomicBool,
) -> Result<(Vec<u8>, i64), Error> {
    const CHUNK: usize = 1 << 20;
    acct.add(CHUNK as u64)?;
    let mut buf = Vec::new();
    reserve(&mut buf, CHUNK)?;
    buf.resize(CHUNK, 0u8);
    let mut out: Vec<u8> = Vec::new();
    // The most `out` can need: `max_header` is checked after each chunk, so one chunk over it.
    let max_cap = usize::try_from(limits.max_header)
        .unwrap_or(usize::MAX)
        .saturating_add(CHUNK);
    let mut off: i64 = 0;
    while off < size {
        check_cancel(cancel)?;
        let n = (CHUNK as i64).min(size - off) as usize;
        // As in Go, a short read that ends at `size` is tolerated (the reader is shorter than
        // `size`); the bytes it did not fill stay as they were.
        match read_full(r, &mut buf[..n], off as u64) {
            Err(Error::Io(std::io::ErrorKind::UnexpectedEof)) if off + n as i64 == size => {}
            res => res?,
        }
        if let Some(i) = buf[..n].iter().position(|&b| b == 4) {
            // As in Go, the limit is tested only after a chunk without the terminator, so a
            // header under `max_header` plus one chunk opens.
            grow(&mut out, i, max_cap, acct)?;
            out.extend_from_slice(&buf[..i]);
            return Ok((out, off + i as i64 + 1));
        }
        grow(&mut out, n, max_cap, acct)?;
        out.extend_from_slice(&buf[..n]);
        if out.len() as u64 > limits.max_header {
            return Err(Error::HeaderLimit);
        }
        off += n as i64;
    }
    Err(hdr(
        "no end of XML header (0x04) before the end of the file",
    ))
}

/// The colour dimension's 0x2008 value, normalised.
fn plane_names(d: &XObject) -> String {
    match d.attr(0x301D, 0x2005) {
        Some(t) if t.value.trim() == "colour component" => {}
        _ => return String::new(),
    }
    let Some(a) = d.attr(0x301D, 0x2008) else {
        return String::new();
    };
    let mut out = String::new();
    for w in (a.value.split(|c: char| c == '"' || c.is_whitespace())).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

/// `strconv.Atoi` of each whitespace-separated field; the first `keep` are returned, so that the
/// result does not grow with the input (callers need at most one more than they accept).
fn ints(s: &str, keep: usize) -> Result<Vec<i64>, Error> {
    let mut out = Vec::new();
    for v in s.split_whitespace() {
        let n = v
            .parse::<i64>()
            .map_err(|_| hdr(format!("bad integer {v:?}")))?;
        if out.len() < keep {
            out.push(n);
        }
    }
    Ok(out)
}

/// `UFS_IMAGE_DIMENSION_RANGE` "start step end".
struct DimRange {
    start: i64,
    steps: i64,
}

/// A vertex list per envelope, shifted by the pad.
fn parse_envelopes(wsi: &XObject, pad: i64) -> Vec<Vec<Point>> {
    let mut out = Vec::new();
    for env in wsi.objects(0x301D, 0x2023) {
        let mut poly = Vec::new();
        for vh in env.objects(0x301D, 0x2024) {
            let Some(a) = vh.attr(0x301D, 0x2025) else {
                continue;
            };
            let mut f = a.value.split_whitespace();
            let (Some(x), Some(y), None) = (f.next(), f.next(), f.next()) else {
                continue;
            };
            let (Ok(x), Ok(y)) = (x.parse::<i64>(), y.parse::<i64>()) else {
                continue;
            };
            poly.push(Point {
                x: x.wrapping_sub(pad),
                y: y.wrapping_sub(pad),
            });
        }
        if !poly.is_empty() {
            out.push(poly);
        }
    }
    out
}

fn ranges(o: &XObject) -> Result<Vec<DimRange>, Error> {
    let mut out = Vec::new();
    for d in o.objects(0x301D, 0x200A) {
        let Some(a) = d.attr(0x301D, 0x200B) else {
            return Err(hdr("dimension range without a value"));
        };
        let v = ints(&a.value, 4)?;
        if v.len() != 3 {
            return Err(hdr(format!("dimension range {:?}", a.value)));
        }
        let step = if v[1] == 0 { 1 } else { v[1] };
        out.push(DimRange {
            start: v[0],
            steps: v[2]
                .wrapping_add(v[1])
                .wrapping_sub(v[0])
                .wrapping_div(step),
        });
    }
    Ok(out)
}

/// `Quality=N` gives N, else -1.
fn quality(desc: &str) -> i64 {
    for f in desc.split('|') {
        if let Some(v) = f.trim().strip_prefix("Quality=")
            && let Ok(n) = v.parse::<i64>()
            && n >= 0
        {
            return n;
        }
    }
    -1
}

impl Header {
    fn parse_wsi(&mut self, wsi: &XObject, cx: &mut Ctx) -> Result<(), Error> {
        self.icc = base64_attr(wsi, 0x0028, 0x2000, cx.acct)?;
        if let Some(a) = wsi.attr(0x0008, 0x2111) {
            self.quality = quality(&a.value);
        }

        let Some(gen_hdr) = wsi.objects(0x301D, 0x2000).first() else {
            return Err(hdr("no UFS_IMAGE_GENERAL_HEADERS"));
        };
        for (i, d) in gen_hdr.objects(0x301D, 0x2003).iter().enumerate() {
            match plane_names(d).as_str() {
                "R G B" => self.rgb = true,
                // YCoCg, the default colour transform.
                "" | "Y Co Cg" => {}
                names => {
                    return Err(Error::Unsupported(format!("plane names {names:?}")));
                }
            }
            let Some(a) = d.attr(0x301D, 0x2007) else {
                continue;
            };
            if i > 1 {
                continue;
            }
            if let Ok(f) = a.value.trim().parse::<f64>()
                && f > 0.0
            {
                if i == 0 {
                    self.mpp_x = f;
                } else {
                    self.mpp_y = f;
                }
            }
        }
        let r = ranges(gen_hdr)?;
        if r.len() < 4 {
            return Err(hdr("general header has under 4 dimension ranges"));
        }
        self.offset_x = r[0].start;
        self.padded_w = r[0].steps;
        self.offset_y = r[1].start;
        self.padded_h = r[1].steps;
        let levels = r[3].steps;
        if !(1..=MAX_LEVELS).contains(&levels) {
            return Err(hdr(format!("{levels} levels")));
        }
        self.levels = levels as usize;
        let pad = (3i64 << levels) - 3;
        self.width = self.padded_w.wrapping_sub(2 * pad);
        self.height = self.padded_h.wrapping_sub(2 * pad);
        if self.width <= 0 || self.height <= 0 || self.padded_w > 1 << 30 || self.padded_h > 1 << 30
        {
            return Err(hdr(format!("image {}x{}", self.width, self.height)));
        }
        if self.width as u64 * self.height as u64 > cx.limits.max_pixels {
            return Err(Error::PixelLimit);
        }
        let span = BLOCK_SIZE << levels;
        self.grid_w = ((self.padded_w + span - 1) / span) << (levels - 1);
        self.grid_h = ((self.padded_h + span - 1) / span) << (levels - 1);
        self.envelopes = Arc::new(parse_envelopes(wsi, pad));
        if self.envelopes.is_empty() {
            // v1 lacks the tag; a default rectangle.
            self.envelope_default = true;
            let (w, h) = (self.width, self.height);
            self.envelopes = Arc::new(vec![vec![
                Point { x: 0, y: 0 },
                Point { x: w, y: 0 },
                Point { x: w, y: h },
                Point { x: 0, y: h },
            ]]);
        }

        // Block size, bits, compressor: the first template.
        let Some(t0) = wsi.objects(0x301D, 0x2009).first() else {
            return Err(hdr("no block header templates"));
        };
        if let Some(a) = t0.attr(0x0028, 0x0101) {
            self.bits_stored = atoi_lenient(a.value.trim());
        }
        if let Some(a) = t0.attr(0x301D, 0x200F) {
            self.compressor = atoi_lenient(a.value.trim());
        }
        if self.compressor != 16 && self.compressor != 19 {
            return Err(Error::Unsupported(format!(
                "compressor {}",
                self.compressor
            )));
        }
        // v1 compressor 19: untested, no corpus.
        if self.major < 100 && self.compressor == 19 {
            return Err(Error::Unsupported("v1 compressor 19".into()));
        }
        // v1 also allows 16 bits (docs/format.md).
        if self.bits_stored != 8
            && self.bits_stored != 9
            && (self.major >= 100 || self.bits_stored != 16)
        {
            return Err(Error::Unsupported(format!(
                "{} bits stored",
                self.bits_stored
            )));
        }
        let br = ranges(t0)?;
        if br.len() < 2 || br[0].steps != BLOCK_SIZE || br[1].steps != BLOCK_SIZE {
            return Err(Error::Unsupported(format!(
                "codeblocks other than {BLOCK_SIZE}x{BLOCK_SIZE}"
            )));
        }

        if self.major < 100 {
            self.codeblocks = self.parse_block_header_table(wsi, cx)?;
            return Ok(());
        }

        let mut ctmpls = Vec::new();
        for c in wsi.objects(0x301D, 0x2016) {
            reserve(&mut ctmpls, 1)?;
            ctmpls.push(parse_cluster_template(c)?);
        }
        let Some(a) = wsi.attr(0x301D, 0x201F) else {
            return Err(hdr("no cluster header table"));
        };
        let table =
            decode_base64(&a.value, cx.acct).map_err(|e| table_error("cluster header table", e))?;
        self.codeblocks = parse_cluster_table(&table, &ctmpls, cx.acct)?;
        Ok(())
    }
}

/// An error's message without the `isyntax: ` prefix, to nest it in another.
fn strip(e: &Error) -> String {
    match e {
        Error::Header(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `e` of decoding the base64 of `what`: a limit or a cancel stays what it is, as a malformed
/// table is a header error.
fn table_error(what: &str, e: Error) -> Error {
    match e {
        Error::AllocLimit | Error::OutOfMemory | Error::Cancelled => e,
        other => hdr(format!("{what}: {}", strip(&other))),
    }
}

/// A codeblock's spot in its cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RelCoord {
    x: i64,
    y: i64,
    colour: i64,
    scale: i64,
    coeff: i64,
}

#[derive(Debug, Default)]
struct ClusterTemplate {
    rel: Vec<RelCoord>,
}

fn parse_cluster_template(c: &XObject) -> Result<ClusterTemplate, Error> {
    let mut t = ClusterTemplate::default();
    let r = ranges(c)?;
    let mut base = [0i64; 5];
    for (b, r) in base.iter_mut().zip(&r) {
        *b = r.start;
    }
    let order = match c.attr(0x301D, 0x2021) {
        Some(a) => ints(&a.value, 6)?,
        None => Vec::new(),
    };
    if order.len() > 5 {
        return Err(hdr("over 5 dimensions in a cluster"));
    }
    for &d in &order {
        if !(0..=4).contains(&d) {
            return Err(hdr(format!("cluster dimension {d}")));
        }
    }
    for b in c.objects(0x301D, 0x200D) {
        let raw = match b.attr(0x301D, 0x200E) {
            Some(a) => ints(&a.value, 5)?,
            None => Vec::new(),
        };
        let mut v = base;
        for (k, &d) in order.iter().enumerate() {
            if let Some(&x) = raw.get(k) {
                v[d as usize] = v[d as usize].wrapping_add(x);
            }
        }
        reserve(&mut t.rel, 1)?;
        t.rel.push(RelCoord {
            x: v[0],
            y: v[1],
            colour: v[2],
            scale: v[3],
            coeff: v[4],
        });
    }
    Ok(t)
}

/// The cluster-table fuzz target's entry (`fuzz_api`): `b` against two fixed templates, as Go's
/// `FuzzParseClusterTable`; the number of codeblocks it holds.
pub(crate) fn fuzz_cluster_table(b: &[u8], max_alloc: u64) -> Result<usize, Error> {
    let tmpl = [ClusterTemplate {
        rel: vec![
            RelCoord {
                x: 1,
                y: 2,
                colour: 0,
                scale: 1,
                coeff: 1,
            },
            RelCoord {
                x: 3,
                y: 4,
                colour: 1,
                scale: 1,
                coeff: 3,
            },
        ],
    }];
    parse_cluster_table(b, &tmpl, &mut Acct::new(max_alloc)).map(|v| v.len())
}

/// Reads the decoded (301D,201F) value.
fn parse_cluster_table(
    b: &[u8],
    tmpls: &[ClusterTemplate],
    acct: &mut Acct,
) -> Result<Vec<Codeblock>, Error> {
    if b.len() < 4 {
        return Err(hdr("cluster table under 4 bytes"));
    }
    let short = || hdr("cluster table truncated");
    // One element at `pos` that ends by `end`: its tag and the range of its data.
    let read = |pos: &mut usize, end: usize| -> Result<(u16, u16, usize, usize), Error> {
        if *pos + 8 > end {
            return Err(short());
        }
        let (g, e) = (le16(b, *pos), le16(b, *pos + 2));
        let n = le32(b, *pos + 4) as usize;
        if n > end - *pos - 8 {
            return Err(short());
        }
        let start = *pos + 8;
        *pos += 8 + n;
        Ok((g, e, start, start + n))
    };
    let mut out: Vec<Codeblock> = Vec::new();
    let mut pos = 4usize;
    let mut first = true;
    while pos < b.len() {
        acct.tick()?;
        let (group, elem, item_start, _) = read(&mut pos, b.len())?;
        if group != 0xFFFE || elem != 0xE000 {
            if first {
                return Err(hdr("cluster table does not start with an item"));
            }
            break;
        }
        first = false;
        let end = pos;
        pos = item_start;
        let mut f = [(0usize, 0usize); 6];
        for fi in &mut f {
            let (_, _, s, e) = read(&mut pos, end)?;
            *fi = (s, e);
        }
        pos = end;
        let len = |i: usize| f[i].1 - f[i].0;
        let m = len(4) / 4;
        if len(0) < 8 || len(1) != 4 || len(2) != 8 || len(3) != 8 || len(5) / 4 != m {
            return Err(hdr("malformed cluster table item"));
        }
        let cx = i64::from(le32(b, f[0].0) as i32);
        let cy = i64::from(le32(b, f[0].0 + 4) as i32);
        let id = le32(b, f[1].0) as usize;
        if id >= tmpls.len() || m > tmpls[id].rel.len() {
            return Err(hdr(format!(
                "cluster template {id} with {m} codeblocks missing"
            )));
        }
        let data_off = le64(b, f[2].0) as i64;
        acct.add(m as u64 * CODEBLOCK_COST)?;
        reserve(&mut out, m)?;
        for (j, rel) in tmpls[id].rel.iter().take(m).enumerate() {
            out.push(Codeblock {
                x: cx.wrapping_add(rel.x),
                y: cy.wrapping_add(rel.y),
                colour: rel.colour,
                scale: rel.scale,
                ll: rel.coeff == 3,
                offset: data_off.wrapping_add(i64::from(le32(b, f[4].0 + 4 * j))),
                size: i64::from(le32(b, f[5].0 + 4 * j)),
            });
        }
    }
    Ok(out)
}

/// One (301D,2015) entry size.
const SEEKTABLE_ENTRY: usize = 40;

struct SeekEntry {
    offset: i64,
    size: i64,
}

/// A codeblock's origin, shared by `index` and `block_id`.
fn codeblock_origin(scale: i64, ll: bool) -> i64 {
    let mut off = (3i64 << scale) - 2;
    if ll {
        off += 1i64 << scale;
    }
    off
}

impl Header {
    /// libisyntax's seektable index (isyntax.c:3301-3330).
    fn block_id(&self, tile_count: &[i64], total: i64, cb: &Codeblock) -> Result<i64, Error> {
        let (scale, colour) = (cb.scale, cb.colour);
        if scale < 0 || scale >= self.levels as i64 || !(0..=2).contains(&colour) {
            return Err(hdr(format!("codeblock at scale {scale} colour {colour}")));
        }
        let max_scale = if cb.ll { scale + 1 } else { scale };
        let mut id: i64 = tile_count.iter().take(max_scale as usize).sum();
        let off = codeblock_origin(scale, cb.ll);
        let dx = cb.x.wrapping_sub(self.offset_x).wrapping_sub(off);
        let dy = cb.y.wrapping_sub(self.offset_y).wrapping_sub(off);
        let span = (2 * BLOCK_SIZE) << scale;
        if dx < 0 || dy < 0 {
            return Err(hdr(format!(
                "codeblock at ({}, {}) scale {scale} outside the tile grid",
                cb.x, cb.y
            )));
        }
        let stride = self.grid_w >> scale;
        id = id
            .wrapping_add((dy / span).wrapping_mul(stride).wrapping_add(dx / span))
            .wrapping_add(colour.wrapping_mul(total));
        Ok(id)
    }

    /// v1's block header table (docs/format.md).
    fn parse_block_header_table(
        &self,
        wsi: &XObject,
        cx: &mut Ctx,
    ) -> Result<Vec<Codeblock>, Error> {
        let Some(a) = wsi.attr(0x301D, 0x2014) else {
            return Err(hdr("no block header table"));
        };
        let table =
            decode_base64(&a.value, cx.acct).map_err(|e| table_error("block header table", e))?;
        if table.len() < 12 {
            return Err(hdr("block header table truncated"));
        }
        let header_size = le32(&table, 0) as usize;
        let seq_size = le32(&table, 8);
        let (entry_size, partial) = match seq_size {
            40 => (48usize, true),
            72 => (80, false),
            _ => return Err(hdr(format!("block header entry of size {seq_size}"))),
        };
        if !header_size.is_multiple_of(entry_size) || 4 + header_size > table.len() {
            return Err(hdr("block header table truncated"));
        }
        let count = header_size / entry_size;
        cx.acct.add(count as u64 * CODEBLOCK_COST)?;
        let mut out: Vec<Codeblock> = Vec::new();
        reserve(&mut out, count)?;
        for i in 0..count {
            cx.acct.tick()?;
            let p = 4 + i * entry_size;
            let mut cb = Codeblock {
                x: i64::from(le32(&table, p + 16) as i32),
                y: i64::from(le32(&table, p + 20) as i32),
                colour: i64::from(le32(&table, p + 24)),
                scale: i64::from(le32(&table, p + 28)),
                ll: le32(&table, p + 32) == 0,
                offset: 0,
                size: 0,
            };
            if !partial {
                cb.offset = le64(&table, p + 44) as i64;
                cb.size = le64(&table, p + 60) as i64;
            }
            out.push(cb);
        }
        if !partial {
            return Ok(out);
        }

        let seek = read_seektable(cx, count)?;
        let base = self.grid_w * self.grid_h;
        let tile_count: Vec<i64> = (0..self.levels).map(|s| base >> (2 * s)).collect();
        let mut total: i64 = tile_count.iter().sum();
        total += tile_count[self.levels - 1]; // the top scale's own LL plane
        for cb in &mut out {
            cx.acct.tick()?;
            let id = self.block_id(&tile_count, total, cb)?;
            let Some(e) = usize::try_from(id).ok().and_then(|i| seek.get(i)) else {
                return Err(hdr(format!(
                    "block id {id} outside the {}-entry seektable",
                    seek.len()
                )));
            };
            cb.offset = e.offset;
            cb.size = e.size;
        }
        Ok(out)
    }

    /// Maps the codeblocks to tiles (docs/format.md "Codeblock position") and checks they lie
    /// within a file of `file_size` bytes.
    pub(crate) fn index(
        &self,
        file_size: i64,
        acct: &mut Acct,
    ) -> Result<HashMap<TileKey, Tile>, Error> {
        // At most one tile per codeblock: charge the table for that, then reserve it, so the map
        // never grows while it is built. The shrink after the build holds the big and the small
        // table at once; both are charged.
        let n = self.codeblocks.len();
        acct.add(table_bytes(n)?)?;
        let mut tiles: HashMap<TileKey, Tile> = HashMap::new();
        tiles.try_reserve(n).map_err(|_| Error::OutOfMemory)?;
        for c in &self.codeblocks {
            acct.tick()?;
            if c.scale < 0 || c.scale >= self.levels as i64 || !(0..=2).contains(&c.colour) {
                return Err(hdr(format!(
                    "codeblock at scale {} colour {}",
                    c.scale, c.colour
                )));
            }
            if c.offset < 0 || c.size < 0 || c.size > MAX_CODEBLOCK || c.offset > file_size - c.size
            {
                return Err(hdr(format!(
                    "codeblock of {} bytes at {} outside the file",
                    c.size, c.offset
                )));
            }
            let off = codeblock_origin(c.scale, c.ll);
            let dx = c.x.wrapping_sub(self.offset_x).wrapping_sub(off);
            let dy = c.y.wrapping_sub(self.offset_y).wrapping_sub(off);
            let span = (2 * BLOCK_SIZE) << c.scale;
            if dx < 0
                || dy < 0
                || dx / span >= self.grid_w >> c.scale
                || dy / span >= self.grid_h >> c.scale
            {
                return Err(hdr(format!(
                    "codeblock at ({}, {}) scale {} outside the tile grid",
                    c.x, c.y, c.scale
                )));
            }
            let k = TileKey {
                scale: c.scale as i32,
                x: (dx / span) as i32,
                y: (dy / span) as i32,
            };
            let t = tiles.entry(k).or_default();
            let b = Block {
                offset: c.offset,
                size: c.size as i32,
            };
            if c.ll {
                t.ll[c.colour as usize] = b;
            } else {
                t.h[c.colour as usize] = b;
            }
            t.exists = t.exists || c.colour == 0;
        }
        // A well-formed tile has 3 to 6 codeblocks: shrink the table to the tiles, so the image
        // keeps (and is charged for) that, not the worst case. shrink_to_fit cannot report an
        // allocation failure, but its size is charged and smaller than the reserved table. The smaller table is charged before it is
        // allocated, while the old one is still live.
        acct.add(table_bytes(tiles.len())?)?;
        tiles.shrink_to_fit();
        Ok(tiles)
    }
}

/// The bytes of a tile index table for `n` entries: one bucket (an entry and a control byte) per
/// 7/8 of the entries rounded up to a power of two.
fn table_bytes(n: usize) -> Result<u64, Error> {
    let buckets = (n.checked_mul(8))
        .map(|m| m.div_ceil(7).max(4))
        .and_then(usize::checked_next_power_of_two)
        .ok_or(Error::AllocLimit)?;
    (buckets.checked_mul(size_of::<(TileKey, Tile)>() + 1))
        .and_then(|b| b.checked_add(32))
        .map(|b| b as u64)
        .ok_or(Error::AllocLimit)
}

/// Reads v1's seektable (docs/format.md).
fn read_seektable(cx: &mut Ctx, block_count: usize) -> Result<Vec<SeekEntry>, Error> {
    let mut tag = [0u8; 8];
    read_full(cx.rd, &mut tag, cx.data_offset as u64)?;
    if le16(&tag, 0) != 0x301D || le16(&tag, 2) != 0x2015 {
        return Err(hdr(format!("no seektable at {}", cx.data_offset)));
    }
    let mut size = le32(&tag, 4) as i32;
    if size < 0 {
        size = (block_count as i32).wrapping_mul(SEEKTABLE_ENTRY as i32);
    }
    if size <= 0 || !(size as usize).is_multiple_of(SEEKTABLE_ENTRY) {
        return Err(hdr(format!("seektable of {size} bytes")));
    }
    let size = size as usize;
    if cx.data_offset + 8 + size as i64 > cx.size {
        return Err(Error::Io(std::io::ErrorKind::UnexpectedEof));
    }
    cx.acct.add(size as u64 * 2)?;
    let mut buf = Vec::new();
    reserve(&mut buf, size)?;
    buf.resize(size, 0u8);
    // In chunks, with the cancel flag checked between them.
    const CHUNK: usize = 1 << 20;
    for (i, c) in buf.chunks_mut(CHUNK).enumerate() {
        check_cancel(cx.cancel)?;
        read_full(cx.rd, c, (cx.data_offset + 8) as u64 + (i * CHUNK) as u64)?;
    }
    let mut out = Vec::new();
    reserve(&mut out, size / SEEKTABLE_ENTRY)?;
    for p in (0..size).step_by(SEEKTABLE_ENTRY) {
        out.push(SeekEntry {
            offset: le64(&buf, p + 16) as i64,
            size: le64(&buf, p + 32) as i64,
        });
    }
    Ok(out)
}

/// Decodes base64 with libisyntax's tolerant alphabet: characters outside it are skipped, one
/// trailing `/` and trailing whitespace are dropped, and decoding stops after the first quad
/// with padding.
pub(crate) fn decode_base64(s: &str, acct: &mut Acct) -> Result<Vec<u8>, Error> {
    let s = s.strip_suffix('/').unwrap_or(s);
    let s = s.trim_end_matches(['\r', '\n', ' ']);
    // The cleaned copy and the output are charged before they are allocated.
    acct.add(s.len() as u64 + (s.len() / 4 * 3) as u64)?;
    let mut clean: Vec<u8> = Vec::new();
    reserve(&mut clean, s.len())?;
    clean.extend(
        s.bytes()
            .filter(|&c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/' | b'=')),
    );
    if clean.is_empty() || !clean.len().is_multiple_of(4) {
        return Err(hdr(format!("base64 of {} characters", clean.len())));
    }
    if let Some(i) = clean.chunks(4).position(|q| q.contains(&b'=')) {
        clean.truncate(4 * i + 4);
    }
    let val = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let bad = || hdr("base64: illegal data");
    let mut out = Vec::new();
    reserve(&mut out, clean.len() / 4 * 3)?;
    let quads = clean.len() / 4;
    for (qi, q) in clean.as_chunks::<4>().0.iter().enumerate() {
        acct.tick()?;
        let pad = if qi + 1 == quads {
            q.iter().filter(|&&c| c == b'=').count()
        } else {
            0
        };
        // Padding is "xx==" or "xxx=" only.
        if pad > 2 || (pad > 0 && q[4 - pad..].iter().any(|&c| c != b'=')) {
            return Err(bad());
        }
        let mut n = 0u32;
        for &c in &q[..4 - pad] {
            n = (n << 6) | u32::from(val(c).ok_or_else(bad)?);
        }
        n <<= 6 * pad as u32;
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - pad]);
    }
    Ok(out)
}

/// A tile of the grid at a scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TileKey {
    pub scale: i32,
    pub x: i32,
    pub y: i32,
}

/// A codeblock's place in the file; size 0 reads as zeros.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Block {
    pub offset: i64,
    pub size: i32,
}

/// A tile's codeblocks per colour. Only top-scale tiles have LL blocks. `exists` is set by any
/// colour-0 codeblock, as in libisyntax.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Tile {
    pub h: [Block; 3],
    pub ll: [Block; 3],
    pub exists: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{self, FileSpec};
    use crate::testutil::{Rng, base64_encode};

    fn parse_vec(file: &[u8]) -> Result<Header, Error> {
        parse_with(file, &Limits::default(), &AtomicBool::new(false))
    }

    fn parse_with(file: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Header, Error> {
        let v = file.to_vec();
        let mut acct = Acct::with_cancel(limits.max_alloc, cancel);
        parse(&v, v.len() as i64, limits, cancel, &mut acct)
    }

    fn spec() -> FileSpec {
        let mut s = FileSpec::new(|s, x, y, c, ll| {
            let n = if ll { 1 } else { 3 };
            let mut v = vec![0i16; n * 128 * 128];
            v[x + y + c] = (s + 1) as i16;
            v
        });
        s.compressor = 19;
        s.bits_stored = 9;
        s.levels = 3;
        s.width = 1500;
        s.height = 900;
        s.offset_x = 7;
        s.offset_y = 11;
        s.mpp = 0.25;
        s.set_exists(|s, x, y| !(x + y + s).is_multiple_of(3));
        s.label = b"label jpeg".to_vec();
        s.macro_image = b"macro jpeg bytes".to_vec();
        s.icc = b"icc!".to_vec();
        s
    }

    fn v1_spec() -> FileSpec {
        let mut s = spec();
        s.version = "5.0".into();
        s.compressor = 16;
        s.bits_stored = 16;
        s
    }

    fn replace(hay: &[u8], from: &str, to: &str, all: bool) -> Vec<u8> {
        let (from, to) = (from.as_bytes(), to.as_bytes());
        let mut out = Vec::new();
        let mut i = 0;
        let mut done = false;
        while i < hay.len() {
            if !done && hay[i..].starts_with(from) {
                out.extend_from_slice(to);
                i += from.len();
                done = !all;
            } else {
                out.push(hay[i]);
                i += 1;
            }
        }
        out
    }

    #[test]
    fn base64() {
        for (input, want) in [
            ("aGVsbG8=", Some("hello")),
            ("aGVs\r\nbG8=", Some("hello")),
            ("aGVsbG8=/", Some("hello")),     // one trailing '/' dropped
            ("aGVsbG8=\r\n ", Some("hello")), // trailing whitespace
            ("aG Vs*bG8=", Some("hello")),    // characters outside the alphabet skipped
            ("aGVsbG8=aGVsbG8=", Some("hello")), // decoding stops at padding
            ("aGVsbG8", None),                // 7 characters
            ("", None),
            ("aGVsbA===", None), // three padding characters
            ("====", None),
            ("aG=s", None),
            ("a===", None),
            ("aA==", Some("h")),
        ] {
            let got = decode_base64(input, &mut Acct::new(u64::MAX));
            match want {
                Some(w) => assert_eq!(got.unwrap(), w.as_bytes(), "{input:?}"),
                None => assert!(got.is_err(), "{input:?}"),
            }
        }
        let data: Vec<u8> = (0..=255).collect();
        for n in 0..40 {
            assert_eq!(
                decode_base64(&base64_encode(&data[..n + 1]), &mut Acct::new(u64::MAX)).unwrap(),
                data[..n + 1]
            );
        }
    }

    #[test]
    fn base64_is_charged_and_cancellable() {
        let s = "AAAA".repeat(5000);
        let r = decode_base64(&s, &mut Acct::new(100));
        assert_eq!(r, Err(Error::AllocLimit), "charged before it allocates");
        let c = AtomicBool::new(true);
        let r = decode_base64(&s, &mut Acct::with_cancel(u64::MAX, &c));
        assert_eq!(r, Err(Error::Cancelled));
    }

    fn check_index(file: &[u8], blocks: &[fixture::BlockInfo], h: &Header) {
        let tiles = h
            .index(file.len() as i64, &mut Acct::new(u64::MAX))
            .unwrap();
        for b in blocks {
            let key = TileKey {
                scale: b.scale as i32,
                x: b.tx as i32,
                y: b.ty as i32,
            };
            let tile = tiles
                .get(&key)
                .unwrap_or_else(|| panic!("block {b:?}: no tile"));
            assert!(tile.exists, "block {b:?}");
            let got = if b.ll {
                tile.ll[b.colour]
            } else {
                tile.h[b.colour]
            };
            assert_eq!(
                (got.offset, i64::from(got.size)),
                (b.offset, b.size),
                "block {b:?}"
            );
        }
        let count: usize = tiles
            .values()
            .map(|t| t.h.iter().chain(&t.ll).filter(|b| b.size > 0).count())
            .sum();
        assert_eq!(count, blocks.len());
    }

    #[test]
    fn parse_iso() {
        let s = spec();
        let (file, blocks) = fixture::file(&s);
        let h = parse_vec(&file).unwrap();
        let (gw, gh) = s.grid();
        assert_eq!(
            (h.major, h.minor, h.compressor, h.bits_stored),
            (100, 5, 19, 9)
        );
        assert_eq!(
            (h.levels, h.width, h.height, h.offset_x, h.offset_y),
            (3, 1500, 900, 7, 11)
        );
        assert_eq!((h.grid_w, h.grid_h), (gw as i64, gh as i64));
        assert_eq!((h.mpp_x, h.mpp_y), (0.25, 0.25));
        assert_eq!(
            h.label.as_deref().map(Vec::as_slice),
            Some(&b"label jpeg"[..])
        );
        assert_eq!(
            h.macro_image.as_deref().map(Vec::as_slice),
            Some(&b"macro jpeg bytes"[..])
        );
        assert_eq!(h.icc.as_deref().map(Vec::as_slice), Some(&b"icc!"[..]));
        check_index(&file, &blocks, &h);
    }

    #[test]
    fn parse_v1() {
        for version in ["5.0", "5"] {
            let mut s = v1_spec();
            s.version = version.into();
            let (file, blocks) = fixture::file(&s);
            let h = parse_vec(&file).unwrap();
            let (gw, gh) = s.grid();
            assert_eq!((h.major, h.compressor, h.bits_stored), (5, 16, 16));
            assert_eq!(
                (h.levels, h.width, h.height, h.offset_x, h.offset_y),
                (3, 1500, 900, 7, 11)
            );
            assert_eq!((h.grid_w, h.grid_h), (gw as i64, gh as i64));
            check_index(&file, &blocks, &h);
        }
    }

    #[test]
    fn unsupported() {
        let mut s = spec();
        s.compressor = 16; // the encoder needs a known version
        let (file, _) = fixture::file(&s);
        let mut cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "compressor",
                replace(
                    &file,
                    "PMSVR=\"IString\">16<",
                    "PMSVR=\"IString\">17<",
                    true,
                ),
            ),
            (
                "16 bits",
                replace(&file, "PMSVR=\"IUInt16\">9<", "PMSVR=\"IUInt16\">16<", true),
            ),
            (
                // Plane names outside R,G,B and Y,Co,Cg.
                "plane names",
                replace(
                    &file,
                    "<DataObject ObjectType=\"UFSImageDimension\">\n</DataObject>\n",
                    "<DataObject ObjectType=\"UFSImageDimension\">\n\
                     <Attribute Name=\"UFS_IMAGE_DIMENSION_TYPE\" Group=\"0x301D\" Element=\"0x2005\" PMSVR=\"IString\">colour component</Attribute>\n\
                     <Attribute Name=\"UFS_IMAGE_DIMENSION_DISCRETE_VALUES_STRING\" Group=\"0x301D\" Element=\"0x2008\" PMSVR=\"IString\">&quot;X&quot; &quot;Y&quot; &quot;Z&quot;</Attribute>\n\
                     </DataObject>\n",
                    false,
                ),
            ),
        ];
        let mut v1 = spec();
        v1.version = "5.0".into();
        v1.compressor = 19;
        cases.push(("v1 compressor 19", fixture::file(&v1).0));
        for (name, file) in cases {
            let err = parse_vec(&file).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{name}: {err:?}");
        }
    }

    #[test]
    fn bad_files() {
        let (file, _) = fixture::file(&spec());
        let eot = file.iter().position(|&b| b == 4).unwrap();
        for (name, f) in [
            ("no EOT", file[..eot].to_vec()),
            ("empty", Vec::new()),
            ("not XML", b"hello\x04".to_vec()),
            ("truncated", file[..eot - 1000].to_vec()),
        ] {
            assert!(parse_vec(&f).is_err(), "{name}");
        }
        // Blocks past the end of the file fail the index.
        let h = parse_vec(&file).unwrap();
        assert!(h.index(eot as i64 + 100, &mut Acct::new(u64::MAX)).is_err());
    }

    /// The tile index is charged before it is built, against the same total as the parse: a limit
    /// just below the two together fails the index, one at them opens.
    #[test]
    fn index_shares_the_parse_total() {
        let (file, _) = fixture::file(&spec());
        let (size, lim, no) = (file.len() as i64, Limits::default(), AtomicBool::new(false));
        let run = |max: u64| {
            let mut a = Acct::new(max);
            let h = parse(&file, size, &lim, &no, &mut a)?;
            let parsed = a.used();
            let blocks = h.codeblocks.len();
            h.index(size, &mut a)
                .map(|t| (parsed, a.used(), t.len(), t.capacity(), blocks))
        };
        let (parsed, total, tiles, cap, blocks) = run(u64::MAX).unwrap();
        let entry = size_of::<(TileKey, Tile)>() as u64;
        assert!(total >= parsed + tiles as u64 * entry, "{parsed} {total}");
        // The kept table is sized for the tiles, not for one tile per codeblock.
        assert!(
            blocks >= 3 * tiles && cap <= 2 * tiles.max(8),
            "{tiles} {cap} {blocks}"
        );
        assert!(run(total).is_ok());
        assert_eq!(run(total - 1).err(), Some(Error::AllocLimit));
    }

    /// A header of many chunks charges the capacity of its buffer, which doubles, not each chunk.
    #[test]
    fn read_xml_grows_geometrically() {
        let mut v = vec![b'a'; 5 << 20];
        v.push(4);
        let lim = Limits {
            max_header: u64::MAX,
            ..Limits::default()
        };
        let no = AtomicBool::new(false);
        let mut a = Acct::new(u64::MAX);
        let (xml, _) = read_xml(&v, v.len() as i64, &lim, &mut a, &no).unwrap();
        assert_eq!(xml.len(), 5 << 20);
        // The chunk buffer, then capacities 1, 2, 4 and 8 MiB.
        assert_eq!(a.used(), (16 << 20) as u64);
        // Under the charge the open is refused before the buffer is grown.
        let r = read_xml(&v, v.len() as i64, &lim, &mut Acct::new(8 << 20), &no);
        assert_eq!(r.err(), Some(Error::AllocLimit));
    }

    #[test]
    fn size_over_the_data() {
        // A size one over the real length: the short header read at the end is tolerated, as in Go.
        let v = fixture::file(&spec()).0;
        assert!(
            parse(
                &v,
                v.len() as i64 + 1,
                &Limits::default(),
                &AtomicBool::new(false),
                &mut Acct::new(u64::MAX)
            )
            .is_ok()
        );
        // Without a terminator the tolerated read ends in the header error, not an Io error.
        let v = b"no terminator".to_vec();
        let r = parse(
            &v,
            v.len() as i64 + 1,
            &Limits::default(),
            &AtomicBool::new(false),
            &mut Acct::new(u64::MAX),
        );
        assert!(matches!(r, Err(Error::Header(_))), "{r:?}");
    }

    #[test]
    fn cancelled() {
        let (file, _) = fixture::file(&spec());
        let r = parse_with(&file, &Limits::default(), &AtomicBool::new(true));
        assert_eq!(r.unwrap_err(), Error::Cancelled);
    }

    #[test]
    fn limits() {
        let (file, _) = fixture::file(&spec());
        let small = Limits {
            max_header: 1000,
            ..Limits::default()
        };
        let no_cancel = AtomicBool::new(false);
        // Go tests the header limit after each chunk without the terminator only: a header of
        // 1000 bytes passes `max_header: 1000`, one of two chunks does not.
        assert!(parse_with(&file, &small, &no_cancel).is_ok());
        let long = Limits {
            max_header: 1 << 20,
            ..Limits::default()
        };
        let read = |n: usize, l: &Limits| {
            let mut v = vec![b'a'; n];
            v.push(4);
            read_xml(&v, v.len() as i64, l, &mut Acct::new(u64::MAX), &no_cancel).map(|x| x.0.len())
        };
        // The terminator in the second chunk: `max_header` + 1 is accepted, as in Go at 256 MiB + 1.
        assert_eq!(read((1 << 20) + 1, &long), Ok((1 << 20) + 1));
        // No terminator in the second chunk: the limit is tested after it.
        assert_eq!(read((2 << 20) + 5, &long), Err(Error::HeaderLimit));
        let tiny = Limits {
            max_alloc: 1 << 20,
            ..Limits::default()
        };
        assert_eq!(
            parse_with(&file, &tiny, &no_cancel).unwrap_err(),
            Error::AllocLimit
        );
        let few = Limits {
            max_pixels: 1000,
            ..Limits::default()
        };
        assert_eq!(
            parse_with(&file, &few, &no_cancel).unwrap_err(),
            Error::PixelLimit
        );
        // The DOM is charged too: a limit that fits the chunk and the header fails on the nodes.
        let eot = file.iter().position(|&b| b == 4).unwrap();
        let dom = Limits {
            max_alloc: (1 << 20) + eot as u64 + 1000,
            ..Limits::default()
        };
        assert_eq!(
            parse_with(&file, &dom, &no_cancel).unwrap_err(),
            Error::AllocLimit
        );
    }

    fn cluster_item() -> (Vec<u8>, Vec<ClusterTemplate>) {
        let tmpl = vec![ClusterTemplate {
            rel: vec![
                RelCoord {
                    x: 1,
                    y: 2,
                    colour: 0,
                    scale: 1,
                    coeff: 1,
                },
                RelCoord {
                    x: 3,
                    y: 4,
                    colour: 1,
                    scale: 1,
                    coeff: 3,
                },
            ],
        }];
        // Cluster (100, 200); data at 5000.
        let mut b = Vec::new();
        let le = |b: &mut Vec<u8>, v: u64, n: usize| b.extend_from_slice(&v.to_le_bytes()[..n]);
        let tag = |b: &mut Vec<u8>, g: u64, e: u64, size: u64| {
            le(b, g, 2);
            le(b, e, 2);
            le(b, size, 4);
        };
        tag(
            &mut b,
            0xFFFE,
            0xE000,
            8 + 8 + 8 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8,
        );
        tag(&mut b, 0x301D, 0x2020, 8);
        le(&mut b, 100, 4);
        le(&mut b, 200, 4);
        tag(&mut b, 0x301D, 0x2019, 4);
        le(&mut b, 0, 4);
        tag(&mut b, 0x301D, 0x2010, 8);
        le(&mut b, 5000, 8);
        tag(&mut b, 0x301D, 0x2011, 8);
        le(&mut b, 30, 8);
        tag(&mut b, 0x301D, 0x201B, 8);
        le(&mut b, 0, 4);
        le(&mut b, 10, 4);
        tag(&mut b, 0x301D, 0x201C, 8);
        le(&mut b, 10, 4);
        le(&mut b, 20, 4);
        let mut table = vec![b.len() as u8, (b.len() >> 8) as u8, 0, 0];
        table.extend(b);
        (table, tmpl)
    }

    #[test]
    fn cluster_table() {
        let (table, tmpl) = cluster_item();
        let acct = || Acct::new(1 << 30);
        let got = parse_cluster_table(&table, &tmpl, &mut acct()).unwrap();
        let want = vec![
            Codeblock {
                x: 101,
                y: 202,
                colour: 0,
                scale: 1,
                ll: false,
                offset: 5000,
                size: 10,
            },
            Codeblock {
                x: 103,
                y: 204,
                colour: 1,
                scale: 1,
                ll: true,
                offset: 5010,
                size: 20,
            },
        ];
        assert_eq!(got, want);
        for i in (1..table.len()).step_by(7) {
            if i > 4 {
                assert!(
                    parse_cluster_table(&table[..i], &tmpl, &mut acct()).is_err(),
                    "truncated at {i}"
                );
            }
        }
        let err = parse_cluster_table(&table, &[], &mut acct()).unwrap_err();
        assert!(err.to_string().contains("template"), "{err}");
        assert_eq!(
            parse_cluster_table(&table, &tmpl, &mut Acct::new(10)).unwrap_err(),
            Error::AllocLimit
        );
    }

    /// v1's full (80-byte) entries carry offset and size directly, no seektable (docs/format.md
    /// "v1: block header table and seektable").
    #[test]
    fn block_header_table_full() {
        let mut table = Vec::new();
        let le32 = |t: &mut Vec<u8>, v: u32| t.extend_from_slice(&v.to_le_bytes());
        let entry = |t: &mut Vec<u8>, x, y, colour, scale, coeff, offset: u64, size: u64| {
            for (g, e, n) in [(0xFFFEu16, 0xE000u16, 72u32), (0x301D, 0x200E, 20)] {
                t.extend_from_slice(&g.to_le_bytes());
                t.extend_from_slice(&e.to_le_bytes());
                le32(t, n);
            }
            for v in [x, y, colour, scale, coeff] {
                le32(t, v);
            }
            for (e, v) in [(0x2010u16, offset), (0x2011, size)] {
                t.extend_from_slice(&0x301Du16.to_le_bytes());
                t.extend_from_slice(&e.to_le_bytes());
                le32(t, 8);
                t.extend_from_slice(&v.to_le_bytes());
            }
            t.extend_from_slice(&0x301Du16.to_le_bytes());
            t.extend_from_slice(&0x2012u16.to_le_bytes());
            le32(t, 4);
            le32(t, 0);
        };
        entry(&mut table, 1, 1, 0, 0, 1, 5000, 10);
        entry(&mut table, 2, 2, 1, 0, 0, 5010, 20);
        let mut tbl = (table.len() as u32).to_le_bytes().to_vec();
        tbl.extend(table);

        let h = Header {
            levels: 1,
            ..Header::default()
        };
        let wsi = XObject {
            attrs: vec![xml::XAttr {
                id: Some((0x301D, 0x2014)),
                value: base64_encode(&tbl),
                objects: Vec::new(),
            }],
        };
        let rd: Vec<u8> = Vec::new();
        let mut acct = Acct::new(1 << 30);
        let mut cx = Ctx {
            rd: &rd,
            data_offset: 0,
            size: 0,
            limits: &Limits::default(),
            cancel: &AtomicBool::new(false),
            acct: &mut acct,
        };
        let got = h.parse_block_header_table(&wsi, &mut cx).unwrap();
        let want = vec![
            Codeblock {
                x: 1,
                y: 1,
                colour: 0,
                scale: 0,
                ll: false,
                offset: 5000,
                size: 10,
            },
            Codeblock {
                x: 2,
                y: 2,
                colour: 1,
                scale: 0,
                ll: true,
                offset: 5010,
                size: 20,
            },
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn plane_names_forms() {
        for (values, want) in [
            ("&quot;R&quot; &quot;G&quot; &quot;B&quot;", "R G B"), // an entity form seen in the wild
            (r#""R" "G" "B""#, "R G B"),
            (r#""Y" "Co" "Cg""#, "Y Co Cg"),
            (r#""X" "Y" "Z""#, "X Y Z"),
            ("", ""),
        ] {
            let x = format!(
                "<DataObject><Attribute Group=\"0x301D\" Element=\"0x2005\">colour component</Attribute>\
                 <Attribute Group=\"0x301D\" Element=\"0x2008\">{values}</Attribute></DataObject>"
            );
            let d = xml::parse(x.as_bytes(), &mut Acct::new(1 << 30)).unwrap();
            assert_eq!(plane_names(&d), want, "{values}");
        }
    }

    #[test]
    fn hostile_header_fields_are_charged_or_bounded() {
        // 100 empty tokens are 200 bytes of input and 2400 bytes of Strings.
        let toks = "\"\"".repeat(100);
        assert_eq!(
            quoted_tokens(&toks, &mut Acct::new(2000)).unwrap_err(),
            Error::AllocLimit
        );
        assert_eq!(
            quoted_tokens(&toks, &mut Acct::new(2500)).unwrap().len(),
            100
        );
        // The integers kept do not grow with the input; a bad one past them is still an error.
        let many = "0 ".repeat(10_000);
        assert_eq!(ints(&many, 4).unwrap(), vec![0; 4]);
        assert!(ints(&format!("{many}x"), 4).is_err());
    }

    #[test]
    fn calibration_fields() {
        let attr = |g: u16, e: u16, vr: &str, v: &str| {
            format!(
                "<Attribute Name=\"X\" Group=\"0x{g:04x}\" Element=\"0x{e:04x}\" PMSVR={vr:?}>{v}</Attribute>\n"
            )
        };
        type Case = (
            &'static str,
            String,
            &'static str,
            Vec<&'static str>,
            Vec<&'static str>,
            i64,
            i64,
        );
        let cases: Vec<Case> = vec![
            ("absent", String::new(), "", vec![], vec![], -1, -1),
            (
                "present",
                attr(0x101D, 0x100A, "IString", " OK ")
                    + &attr(0x0018, 0x1200, "IStringArray", "\"20190612\"")
                    + &attr(0x0018, 0x1201, "IStringArray", "\"101500.000000\"")
                    + &attr(0x101D, 0x1007, "IUInt16", "3")
                    + &attr(0x101D, 0x1008, "IUInt16", " 12 "),
                "OK",
                vec!["20190612"],
                vec!["101500.000000"],
                3,
                12,
            ),
            (
                "multi token",
                attr(0x0018, 0x1200, "IStringArray", "\"20190612\" \"20190701\"")
                    + &attr(0x0018, 0x1201, "IStringArray", "\"1\" \"2\" \"3\""),
                "",
                vec!["20190612", "20190701"],
                vec!["1", "2", "3"],
                -1,
                -1,
            ),
            (
                "malformed",
                attr(0x0018, 0x1200, "IStringArray", "20190612")
                    + &attr(0x0018, 0x1201, "IStringArray", "\"open")
                    + &attr(0x101D, 0x1007, "IUInt16", "70000")
                    + &attr(0x101D, 0x1008, "IUInt16", "-1"),
                "",
                vec![],
                vec![],
                -1,
                -1,
            ),
            (
                "malformed keeps valid tokens",
                attr(0x0018, 0x1200, "IStringArray", "\"a\" \"b")
                    + &attr(0x101D, 0x1007, "IUInt16", "x")
                    + &attr(0x101D, 0x1008, "IUInt16", "0"),
                "",
                vec!["a"],
                vec![],
                -1,
                0,
            ),
        ];
        for (name, extra, status, date, tm, rack, slot) in cases {
            let mut s = spec();
            s.extra = extra;
            let (file, _) = fixture::file(&s);
            let h = parse_vec(&file).unwrap();
            assert_eq!(&*h.calibration_status, status, "{name}");
            assert_eq!(h.calibration_date.to_vec(), date, "{name}");
            assert_eq!(h.calibration_time.to_vec(), tm, "{name}");
            assert_eq!((h.rack, h.slot), (rack, slot), "{name}");
            assert_eq!(&*h.interface_version, "100.5");
        }
    }

    #[test]
    fn interface_version_v1() {
        let (file, _) = fixture::file(&v1_spec());
        let h = parse_vec(&file).unwrap();
        assert_eq!((&*h.interface_version, h.rack), ("5.0", -1));
    }

    #[test]
    fn quality_values() {
        for (input, want) in [
            (
                "PHILIPS UFS B300 V1.1.1.1 | Quality=2 | DWT=LeGall53 | Compressor=Hulsken2",
                2,
            ),
            (
                "USS2iSyntax | Quality=1 | DWT=legall53sse | Compressor=hulsken",
                1,
            ),
            ("Quality=0", 0),
            ("", -1),
            ("DWT=LeGall53 | Quality=x", -1),
        ] {
            assert_eq!(quality(input), want, "{input:?}");
        }
    }

    /// Hostile headers error, not panic: byte edits and truncations of valid files.
    #[test]
    fn mutated_headers_do_not_panic() {
        let mut small = spec();
        small.width = 300;
        small.height = 200;
        small.levels = 2;
        for s in [small.clone(), {
            let mut v = small.clone();
            v.version = "5.0".into();
            v.compressor = 16;
            v.bits_stored = 16;
            v
        }] {
            let (file, _) = fixture::file(&s);
            let eot = file.iter().position(|&b| b == 4).unwrap();
            let mut rnd = Rng::new(7);
            for i in 0..400 {
                let mut f = file.clone();
                match i % 4 {
                    0 => f.truncate(rnd.below(f.len() as u64) as usize),
                    1 => {
                        for _ in 0..3 {
                            let at = rnd.below(eot as u64 + 64) as usize;
                            f[at] = rnd.next_u64() as u8;
                        }
                    }
                    2 => {
                        let at = rnd.below(eot as u64) as usize;
                        f[at] ^= 1 << rnd.below(8);
                    }
                    _ => {
                        // The binary table after the XML.
                        let at = eot + 1 + rnd.below((f.len() - eot - 1) as u64) as usize;
                        f[at] = rnd.next_u64() as u8;
                    }
                }
                if let Ok(h) = parse_vec(&f) {
                    let _ = h.index(f.len() as i64, &mut Acct::new(u64::MAX));
                }
            }
        }
    }
}
