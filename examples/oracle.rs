//! The Rust side of the parity check: `oracle <inputs> <outputs>` opens and decodes every input
//! like `tests/parity/go/oracle` and writes `<name>.out`: a line per operation, `<op> ok <digest>` or
//! `<op> err <class>`, and `<op> partial <digest>` after the error line of a tile decode (the
//! destination as the decoder left it). The class table is in `docs/parity.md`.

use isyntax::fuzz_api::{self, Hulsken};
use isyntax::{Error, Image, Limits, Rect, Subsample, YCbCr};
use std::fmt::Write as _;
use std::sync::atomic::AtomicBool;
use std::{env, fs, io};

fn class(e: &Error) -> &'static str {
    match e {
        Error::Cancelled => "cancelled",
        Error::Io(_) => "io",
        Error::Xml(_) => "xml",
        Error::Header(_) => "header",
        Error::Unsupported(_) => "unsupported",
        Error::Codeblock(_) => "codeblock",
        Error::MemoryLimit => "memory-limit",
        Error::AllocLimit => "alloc-limit",
        Error::PixelLimit => "pixel-limit",
        Error::HeaderLimit => "header-limit",
        Error::OutOfMemory => "out-of-memory",
        Error::InvalidArgument(_) => "invalid-argument",
        Error::Internal(_) => "internal",
    }
}

/// FNV-1a 64 and the length of the parts together: `digest` of the Go oracle.
fn digest(parts: &[&[u8]]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut n = 0;
    for p in parts {
        for &c in *p {
            h = (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3);
        }
        n += p.len();
    }
    format!("{h:016x} {n}")
}

fn le16(v: impl Iterator<Item = u16>) -> Vec<u8> {
    v.flat_map(u16::to_le_bytes).collect()
}

/// Go's `image.RGBA64.Pix` holds each sample big-endian.
fn be16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

fn hx(s: impl AsRef<[u8]>) -> String {
    s.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn hxs(v: &[String]) -> String {
    v.iter().map(hx).collect::<Vec<_>>().join(",")
}

/// The lines of one input.
#[derive(Default)]
struct Report {
    out: String,
    /// The regions run with one worker, so every region has a partial line.
    one_worker: bool,
    /// A region decode failed.
    region_err: bool,
}

impl Report {
    fn line(&mut self, s: String) {
        self.out.push_str(&s);
        self.out.push('\n');
    }

    fn res(&mut self, op: &str, r: Result<(), Error>, data: &[&[u8]]) {
        match r {
            Err(e) => self.line(format!("{op} err {}", class(&e))),
            Ok(()) => self.line(format!("{op} ok {}", digest(data))),
        }
    }

    /// `res` for a region decode, with the partial line when the level is one tile or one worker
    /// runs (see `regionRes` of the Go oracle).
    fn region_res(&mut self, single: bool, op: &str, r: Result<(), Error>, data: &[&[u8]]) {
        self.region_err |= r.is_err();
        if single || self.one_worker {
            self.tile_res(op, r, data);
        } else {
            self.res(op, r, data);
        }
    }

    fn tile_res(&mut self, op: &str, r: Result<(), Error>, data: &[&[u8]]) {
        let failed = r.is_err();
        self.res(op, r, data);
        if failed {
            self.line(format!("{op} partial {}", digest(data)));
        }
    }
}

fn rect(r: Rect) -> String {
    format!("{},{},{},{}", r.min_x, r.min_y, r.max_x, r.max_y)
}

fn info(r: &mut Report, im: &Image) {
    r.line(format!(
        "info levels={} tile={} bits={}",
        im.levels(),
        im.tile_size(),
        im.bits_stored()
    ));
    let (mx, my) = im.mpp();
    r.line(format!(
        "info mpp={:016x},{:016x}",
        mx.to_bits(),
        my.to_bits()
    ));
    let (l, m, i) = (im.label(), im.macro_image(), im.icc_profile());
    let d = |v: Option<&[u8]>| digest(&[v.unwrap_or_default()]);
    r.line(format!("info label={} macro={} icc={}", d(l), d(m), d(i)));
    r.line(format!(
        "info label-nil={} macro-nil={} icc-nil={}",
        l.is_none(),
        m.is_none(),
        i.is_none()
    ));
    r.line(format!(
        "info barcode={} model={} serial={} acq={} cal={} iface={}",
        hx(im.barcode()),
        hx(im.model_name()),
        hx(im.device_serial()),
        hx(im.acquisition_date_time()),
        hx(im.calibration_status()),
        hx(im.interface_version())
    ));
    let (rack, slot) = im.rack_slot();
    r.line(format!(
        "info dates={} times={} rack={} slot={}",
        hxs(im.date_of_last_calibration()),
        hxs(im.time_of_last_calibration()),
        rack.map_or(-1, i32::from),
        slot.map_or(-1, i32::from)
    ));
    for l in 0..im.levels() {
        let Some(lv) = im.level(l) else { continue };
        let mut env = String::new();
        for poly in im.envelopes(l) {
            for p in poly {
                let _ = write!(env, "{},{} ", p.x, p.y);
            }
            env.push_str("| ");
        }
        r.line(format!(
            "info level {l}: {}x{} tiles {}x{} ds {} bounds {} env {env}",
            lv.width,
            lv.height,
            lv.tiles_x,
            lv.tiles_y,
            lv.downsample,
            rect(im.source_bounds(l))
        ));
    }
}

/// No memory or pixel limit, as the Go oracle has none.
fn unlimited() -> Limits {
    Limits {
        max_alloc: u64::MAX,
        max_pixels: u64::MAX,
        ..Limits::default()
    }
}

fn tiles(r: &mut Report, im: &Image, tag: &str, cancel: &AtomicBool) {
    for l in 0..im.levels() {
        let Some(lv) = im.level(l) else { continue };
        let side = if l == im.levels() - 1 {
            128
        } else {
            im.tile_size()
        };
        let (tx, ty) = (lv.tiles_x as i64, lv.tiles_y as i64);
        for t in [
            (0, 0),
            (tx / 2, ty / 2),
            (tx - 1, ty - 1),
            (tx, ty),
            (-1, 0),
        ] {
            let mut d = vec![0xAAu8; 4 * side * side];
            let res = im.decode_tile_rgba(l, t.0, t.1, &mut d, 4 * side, cancel);
            r.tile_res(&format!("{tag}tile.{l}.{}.{}", t.0, t.1), res, &[&d]);
        }
        let t = (tx / 2, ty / 2);
        let mut d = vec![0xAAAAu16; 4 * side * side];
        let res = im.decode_tile_rgba64(l, t.0, t.1, &mut d, 4 * side, cancel);
        r.tile_res(&format!("{tag}tile64.{l}"), res, &[&be16(&d)]);
        for (name, s) in [("420", Subsample::S420), ("444", Subsample::S444)] {
            let side = side as i64;
            let Ok(mut y) = YCbCr::new(Rect::new(0, 0, side, side), s, &unlimited()) else {
                r.line(format!("{tag}tileycc{name}.{l} err out-of-memory"));
                continue;
            };
            let res = im.decode_tile_ycbcr(l, t.0, t.1, &mut y, cancel);
            r.tile_res(
                &format!("{tag}tileycc{name}.{l}"),
                res,
                &[&y.y, &y.cb, &y.cr],
            );
        }
    }
}

/// Every tile of every level as RGBA, a digest line each, for an image of at most 64 level-0 tiles:
/// `allTiles` of the Go oracle.
fn all_tiles(r: &mut Report, im: &Image, cancel: &AtomicBool) {
    if im.level(0).is_none_or(|lv| lv.tiles_x * lv.tiles_y > 64) {
        return;
    }
    for l in 0..im.levels() {
        let Some(lv) = im.level(l) else { continue };
        let side = if l == im.levels() - 1 {
            128
        } else {
            im.tile_size()
        };
        for y in 0..lv.tiles_y as i64 {
            for x in 0..lv.tiles_x as i64 {
                let mut d = vec![0u8; 4 * side * side];
                let res = im.decode_tile_rgba(l, x, y, &mut d, 4 * side, cancel);
                r.res(&format!("all.{l}.{x}.{y}"), res, &[&d]);
            }
        }
    }
}

fn regions(r: &mut Report, im: &Image, tag: &str, all: bool, cancel: &AtomicBool) {
    for l in 0..im.levels() {
        let Some(lv) = im.level(l) else { continue };
        let (w, h) = (lv.width as i64, lv.height as i64);
        let single = lv.tiles_x == 1 && lv.tiles_y == 1;
        let a = Rect::new(0, 0, w.min(400), h.min(300));
        let b = Rect::new(w - 100, h - 90, w + 60, h + 50);
        for (i, rc) in [a, b].into_iter().enumerate() {
            let op = |k: &str| format!("{tag}region{k}.{l}.{i}");
            let (dx, dy) = (rc.dx() as usize, rc.dy() as usize);
            let mut d = vec![0u8; 4 * dx * dy];
            let res = im.decode_region_rgba(l, rc, &mut d, 4 * dx, cancel);
            r.region_res(single, &op("rgba"), res, &[&d]);
            if i == 1 && !all {
                continue;
            }
            if i == 0 || all {
                let mut d64 = vec![0u16; 4 * dx * dy];
                let res = im.decode_region_rgba64(l, rc, &mut d64, 4 * dx, cancel);
                r.region_res(single, &op("rgba64"), res, &[&be16(&d64)]);
                let mut u = vec![0u16; 3 * dx * dy];
                let res = im.decode_region_samples(l, rc, &mut u, 3 * dx, cancel);
                r.region_res(single, &op("samples"), res, &[&le16(u.iter().copied())]);
                let mut s = vec![0i16; 3 * dx * dy];
                let res = im.decode_region_signed_samples(l, rc, &mut s, 3 * dx, cancel);
                r.region_res(
                    single,
                    &op("signed"),
                    res,
                    &[&le16(s.iter().map(|&v| v as u16))],
                );
                let rect0 = Rect::new(0, 0, dx as i64, dy as i64);
                if let Ok(mut y) = YCbCr::new(rect0, Subsample::S420, &unlimited()) {
                    let res = im.decode_region_ycbcr(l, rc, &mut y, cancel);
                    r.region_res(single, &op("ycc420"), res, &[&y.y, &y.cb, &y.cr]);
                }
            }
        }
        // TestSignedSamples' rectangles: one partly above and left of the level, one narrow and tall.
        if all {
            for (i, rc) in [Rect::new(-20, -10, 10, 10), Rect::new(250, 250, 263, 520)]
                .into_iter()
                .enumerate()
            {
                let op = |k: &str| format!("{tag}region{k}.{l}.{}", i + 2);
                let (dx, dy) = (rc.dx() as usize, rc.dy() as usize);
                let mut u = vec![0u16; 3 * dx * dy];
                let res = im.decode_region_samples(l, rc, &mut u, 3 * dx, cancel);
                r.region_res(single, &op("samples"), res, &[&le16(u.iter().copied())]);
                let mut s = vec![0i16; 3 * dx * dy];
                let res = im.decode_region_signed_samples(l, rc, &mut s, 3 * dx, cancel);
                let b = le16(s.iter().map(|&v| v as u16));
                r.region_res(single, &op("signed"), res, &[&b]);
            }
        }
    }
    // Bad arguments: a level out of range, a destination of the wrong size.
    let mut d = vec![0u8; 8 * 8 * 4];
    let res = im.decode_region_rgba(im.levels(), Rect::new(0, 0, 8, 8), &mut d, 32, cancel);
    r.res(&format!("{tag}bad.level"), res, &[]);
    let res = im.decode_region_rgba(0, Rect::new(0, 0, 9, 8), &mut d, 36, cancel);
    r.res(&format!("{tag}bad.size"), res, &[]);
    let mut u = vec![0u16; 24];
    let res = im.decode_region_samples(0, Rect::new(0, 0, 8, 8), &mut u, 24, cancel);
    r.res(&format!("{tag}bad.stride"), res, &[]);
    let mut d64 = vec![0u16; 4 * 9 * 10];
    let res = im.decode_region_rgba64(0, Rect::new(0, 0, 10, 10), &mut d64, 4 * 9, cancel);
    r.res(&format!("{tag}bad.size64"), res, &[]);
    let mut t64 = vec![0u16; 4 * 128 * 128];
    let res = im.decode_tile_rgba64(0, 0, 0, &mut t64, 4 * 128, cancel);
    r.res(&format!("{tag}bad.tile64"), res, &[]);
}

/// A `.blk` input, `[version-1, planes/3, size index][codeblock]`: the Hulsken decoder alone, as
/// `hulskenOps` of the Go oracle.
fn hulsken_ops(data: &[u8], hul: &mut Hulsken) -> String {
    let mut r = Report::default();
    let Some((&[v, n, bw], src)) = data.split_first_chunk::<3>() else {
        r.line("hulsken err header".into());
        return r.out;
    };
    let (v, n, bw) = (
        1 + (v & 1),
        if n & 1 == 0 { 1 } else { 3 },
        [4, 8, 16, 128][usize::from(bw & 3)],
    );
    match hul.decode(src, n, v, bw, u64::MAX) {
        Ok(c) => r.line(format!(
            "hulsken ok {}",
            digest(&[&le16(c.iter().map(|&x| x as u16))])
        )),
        Err(e) => r.line(format!("hulsken err {}", class(&e))),
    }
    r.out
}

/// Every operation on one input, as `ops` of the Go oracle.
fn ops(name: &str, data: &[u8], limits: &Limits, hul: &mut Hulsken) -> String {
    if name.ends_with(".blk") {
        return hulsken_ops(data, hul);
    }
    if name.ends_with(".b64") {
        let mut r = Report::default();
        match fuzz_api::decode_base64(&String::from_utf8_lossy(data), u64::MAX) {
            Ok(b) => r.line(format!("base64 ok {}", digest(&[&b]))),
            Err(e) => r.line(format!("base64 err {}", class(&e))),
        }
        return r.out;
    }
    if name.ends_with(".ct") {
        let mut r = Report::default();
        match fuzz_api::cluster_table(data, u64::MAX) {
            Ok(n) => r.line(format!("ct ok {n}")),
            Err(e) => r.line(format!("ct err {}", class(&e))),
        }
        return r.out;
    }
    let mut r = Report::default();
    let no = AtomicBool::new(false);
    let im = match Image::open(data.to_vec(), data.len() as u64, limits, &no) {
        Ok(im) => im,
        Err(e) => {
            r.res("open", Err(e), &[]);
            return r.out;
        }
    };
    r.line("open ok".into());
    info(&mut r, &im);
    tiles(&mut r, &im, "", &no);
    regions(&mut r, &im, "", true, &no);
    tiles(&mut r, &im, "warm.", &no);
    all_tiles(&mut r, &im, &no);
    let sdk = im.sdk_truncated();
    r.res("sdk", sdk.as_ref().map(|_| ()).map_err(Error::clone), &[]);
    if let Ok(sdk) = &sdk {
        regions(&mut r, sdk, "sdk.", false, &no);
    }
    let wide = im.wide_grid();
    for l in 0..wide.levels() {
        if let Some(lv) = wide.level(l) {
            r.line(format!(
                "wide level {l}: {}x{} tiles {}x{}",
                lv.width, lv.height, lv.tiles_x, lv.tiles_y
            ));
        }
    }
    regions(&mut r, &wide, "wide.", false, &no);
    // Where a region failed: the regions again with one worker on a fresh image, so the tiles
    // written before the error are the same on every run.
    if r.region_err {
        let one = Limits {
            max_threads: 1,
            ..*limits
        };
        if let Ok(im1) = Image::open(data.to_vec(), data.len() as u64, &one, &no) {
            r.one_worker = true;
            regions(&mut r, &im1, "w1.", true, &no);
            let sdk1 = im1.sdk_truncated();
            if let Ok(sdk1) = &sdk1 {
                regions(&mut r, sdk1, "w1.sdk.", false, &no);
            }
            regions(&mut r, &im1.wide_grid(), "w1.wide.", false, &no);
            r.one_worker = false;
        }
    }
    // A cancelled decode, on a fresh image.
    if let Ok(im2) = Image::open(data.to_vec(), data.len() as u64, limits, &no) {
        let yes = AtomicBool::new(true);
        let mut d = vec![0u8; 4 * 256 * 256];
        let res = im2.decode_tile_rgba(0, 0, 0, &mut d, 4 * 256, &yes);
        r.res("cancel", res, &[]);
    }
    // A reader size one short and one over the real length: the open reads the header only.
    for (d, name) in [(-1i64, "size-1"), (1, "size+1")] {
        let size = data.len().wrapping_add_signed(d as isize) as u64;
        let res = Image::open(data.to_vec(), size, limits, &no);
        r.res(name, res.map(|_| ()), &[]);
    }
    r.out
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: oracle <inputs> <outputs>");
        std::process::exit(2);
    };
    // No memory limit: the Go oracle runs with SetMemoryLimit(0). The pixel limit stays, which the
    // Go oracle's inputs never reach (gen drops the files that would).
    // ISYNTAX_ORACLE_THREADS=1 runs the whole pass with one thread (parity compares it with Go's
    // output of the default pass: the lines of that pass do not depend on scheduling).
    let mut limits = Limits {
        max_alloc: u64::MAX,
        ..Limits::default()
    };
    if let Some(n) = env::var("ISYNTAX_ORACLE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        limits.max_threads = n;
    }
    // One decoder for all `.blk` inputs (sorted by name), as `blkScratch` of the Go oracle.
    let mut hul = Hulsken::default();
    let mut names: Vec<String> = fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    fs::create_dir_all(output)?;
    for name in names {
        let data = fs::read(format!("{input}/{name}"))?;
        fs::write(
            format!("{output}/{name}.out"),
            ops(&name, &data, &limits, &mut hul),
        )?;
    }
    Ok(())
}
