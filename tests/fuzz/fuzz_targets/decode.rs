#![no_main]
use isyntax::{Decoder, Error, Image, Limits, Rect, Subsample, YCbCr};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[path = "../common.rs"]
mod common;
use common::{Flaky, limits};

static NO: AtomicBool = AtomicBool::new(false);
/// The previous input's image, kept with its bytes: its caches and budget carry over from input to
/// input, and its decodes must still equal those of a fresh image of the same file. Its seed gives
/// the limits it was opened with.
static PREV: Mutex<Option<(Vec<u8>, u64, Image)>> = Mutex::new(None);

/// The warm images (not `cold`) take their buffers from this one `Decoder`, kept across inputs, so
/// buffers move from the file of one input to that of the next; the warm-vs-cold check holds it
/// to what an image without one gives. Generous limits: what it holds is not charged to an image.
static DEC: OnceLock<Decoder> = OnceLock::new();

fn dec() -> &'static Decoder {
    DEC.get_or_init(|| {
        Decoder::new(&Limits {
            max_alloc: 128 << 20,
            ..Limits::default()
        })
    })
}

type Out = Result<Vec<u8>, Error>;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n.max(1)
    }

    /// A rectangle corner from the whole i64 range: the edges of it and of the 2^40 coordinate
    /// limit, or any value.
    fn wild(&mut self) -> i64 {
        const EDGE: [i64; 11] = [
            i64::MIN,
            i64::MIN + 1,
            i64::MAX - 1,
            i64::MAX,
            -(1 << 40),
            1 << 40,
            -(1 << 40) - 1,
            (1 << 40) + 1,
            -1,
            0,
            1,
        ];
        match self.below(4) {
            0 => self.next() as i64,
            _ => EDGE[self.below(EDGE.len() as u64) as usize],
        }
    }
}

/// The most elements `dst` allocates; a destination that needs more gets 16.
const MAX_DST: usize = 1 << 22;

/// A destination of `rows` rows of `row` elements, `fill`-filled, with a stride of `row` plus 0 to 4
/// elements of padding, and its stride. Sometimes it is 1 or 2 elements short of what the rows
/// need; a destination too big for `MAX_DST` is 16 elements. The flag is true when a decode of a
/// non-empty region into it must fail with `InvalidArgument`.
fn dst<T: Copy>(rng: &mut Rng, row: usize, rows: usize, fill: T) -> (Vec<T>, usize, bool) {
    let stride = row.saturating_add(if rng.below(2) == 0 {
        0
    } else {
        rng.below(5) as usize
    });
    let short = if rng.below(8) == 0 {
        1 + rng.below(2) as usize
    } else {
        0
    };
    let need = match rows {
        0 => Some(0),
        _ => (rows - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row)),
    };
    match need {
        Some(n) if n <= MAX_DST => (
            vec![fill; n.saturating_sub(short)],
            stride,
            short > 0 && n > 0,
        ),
        _ => (vec![fill; 16], stride, true),
    }
}

/// What a decode must leave of its destination: the padding between rows as it was, and
/// `InvalidArgument` if the destination was too small.
fn check_dst<T: PartialEq + Copy>(
    res: &Result<(), Error>,
    d: &[T],
    row: usize,
    stride: usize,
    fill: T,
    bad: bool,
) {
    assert!(
        stride <= row
            || d.chunks(stride)
                .all(|c| c.iter().skip(row).all(|&v| v == fill)),
        "the padding between rows was written"
    );
    if bad {
        assert!(
            matches!(res, Err(Error::InvalidArgument(_))),
            "a destination too small gave {:?}",
            res.as_ref().err()
        );
    }
}

/// The two sides of the region checks: a corner beyond 2^40 of a non-empty rectangle is always
/// `InvalidArgument` (checked before anything is read, so cancelling does not matter); with all
/// corners within it, a good level and destination, and no cancelling, it is never.
fn check_region(res: &Result<(), Error>, r: Rect, level_ok: bool, bad: bool, cancelled: bool) {
    if !level_ok || r.is_empty() {
        return;
    }
    let far = [r.min_x, r.min_y, r.max_x, r.max_y]
        .iter()
        .any(|c| c.unsigned_abs() > 1 << 40);
    let invalid = matches!(res, Err(Error::InvalidArgument(_)));
    if far {
        assert!(
            invalid,
            "a corner beyond 2^40 gave {:?}",
            res.as_ref().err()
        );
    } else if !bad && !cancelled {
        assert!(!invalid, "a valid region gave {:?}", res.as_ref().err());
    }
}

fn open(data: &[u8], seed: u64) -> Result<Image, Error> {
    Image::open(data.to_vec(), data.len() as u64, &limits(seed), &NO)
}

fn be16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// Up to four decodes of `seed`'s choosing on `im`, a view of it, or its wide grid, under a memory
/// limit of the seed's choosing. Some are cancelled part way: their output is not kept, but the
/// decodes after them run on the same image.
fn run(im: &Image, seed: u64) -> Vec<Out> {
    let mut rng = Rng(seed | 1);
    im.set_memory_limit(match rng.below(4) {
        0 => 256 << 20,
        _ => (1 + rng.below(64)) << 20,
    });
    let (sdk, wide);
    let im = match rng.below(3) {
        0 => im,
        1 => match im.sdk_truncated() {
            Ok(v) => {
                sdk = v;
                &sdk
            }
            Err(e) => return vec![Err(e)],
        },
        _ => {
            wide = im.wide_grid();
            &wide
        }
    };
    let mut out = Vec::new();
    for _ in 0..1 + rng.below(4) {
        let level = rng.below(im.levels() as u64 + 1) as usize;
        let lv = im.level(level).unwrap_or(isyntax::Level {
            width: 0,
            height: 0,
            tiles_x: 0,
            tiles_y: 0,
            downsample: 1,
        });
        let side = if level + 1 == im.levels() { 128 } else { 256 };
        let (tx, ty) = (
            rng.below(lv.tiles_x as u64 + 2) as i64 - 1,
            rng.below(lv.tiles_y as u64 + 2) as i64 - 1,
        );
        let x0 = rng.below(lv.width as u64 + 100) as i64 - 50;
        let y0 = rng.below(lv.height as u64 + 100) as i64 - 50;
        let r = Rect::new(
            x0,
            y0,
            x0 + rng.below(300) as i64,
            y0 + rng.below(300) as i64,
        );
        // Sometimes corners from the whole i64 range, with the destination small.
        let r = if rng.below(4) == 0 {
            Rect::new(rng.wild(), rng.wild(), rng.wild(), rng.wild())
        } else {
            r
        };
        let (dx, dy) = (r.dx() as usize, r.dy() as usize);
        let kind = rng.below(8);
        let (ox, oy) = (rng.below(7) as i64 - 3, rng.below(7) as i64 - 3);
        let sub = if rng.below(2) == 0 {
            Subsample::S420
        } else {
            Subsample::S444
        };
        // A YCbCr whose rectangle is 1 to 3 wider or taller than the request: InvalidArgument.
        let grow = if rng.below(8) == 0 {
            1 + rng.below(3) as i64
        } else {
            0
        };
        let (gx, gy) = if rng.below(2) == 0 {
            (grow, 0)
        } else {
            (0, grow)
        };
        let cancelled = rng.below(4) == 0;
        let delay = rng.below(2000);
        let cancel = AtomicBool::new(false);
        let region_ok = |res: &Result<(), Error>, bad: bool| {
            check_region(res, r, level < im.levels(), bad, cancelled);
        };
        let res: Out = std::thread::scope(|sc| {
            if cancelled {
                sc.spawn(|| {
                    for _ in 0..delay {
                        std::thread::yield_now();
                    }
                    cancel.store(true, Ordering::Relaxed);
                });
            }
            match kind {
                0 => {
                    let (mut d, st, bad) = dst(&mut rng, 4 * side, side, 0xAAu8);
                    let res = im.decode_tile_rgba(level, tx, ty, &mut d, st, &cancel);
                    check_dst(&res, &d, 4 * side, st, 0xAA, bad);
                    res.map(|_| d)
                }
                1 => {
                    let (mut d, st, bad) = dst(&mut rng, 4 * side, side, 0xAAAAu16);
                    let res = im.decode_tile_rgba64(level, tx, ty, &mut d, st, &cancel);
                    check_dst(&res, &d, 4 * side, st, 0xAAAA, bad);
                    res.map(|_| be16(&d))
                }
                2 => {
                    let t = Rect::new(ox, oy, ox + side as i64 + gx, oy + side as i64 + gy);
                    YCbCr::new(t, sub, &limits(seed)).and_then(|mut y| {
                        let res = im.decode_tile_ycbcr(level, tx, ty, &mut y, &cancel);
                        assert!(grow == 0 || matches!(res, Err(Error::InvalidArgument(_))));
                        res.map(|_| [y.y, y.cb, y.cr].concat())
                    })
                }
                3 => {
                    let (mut d, st, bad) = dst(&mut rng, 4usize.saturating_mul(dx), dy, 0xAAu8);
                    let res = im.decode_region_rgba(level, r, &mut d, st, &cancel);
                    check_dst(
                        &res,
                        &d,
                        4usize.saturating_mul(dx),
                        st,
                        0xAA,
                        bad && !r.is_empty(),
                    );
                    region_ok(&res, bad);
                    res.map(|_| d)
                }
                4 => {
                    let (mut d, st, bad) = dst(&mut rng, 3usize.saturating_mul(dx), dy, 0xAAAAu16);
                    let res = im.decode_region_samples(level, r, &mut d, st, &cancel);
                    check_dst(
                        &res,
                        &d,
                        3usize.saturating_mul(dx),
                        st,
                        0xAAAA,
                        bad && !r.is_empty(),
                    );
                    region_ok(&res, bad);
                    res.map(|_| be16(&d))
                }
                5 => {
                    let (mut d, st, bad) = dst(&mut rng, 3usize.saturating_mul(dx), dy, 0x2AAAi16);
                    let res = im.decode_region_signed_samples(level, r, &mut d, st, &cancel);
                    check_dst(
                        &res,
                        &d,
                        3usize.saturating_mul(dx),
                        st,
                        0x2AAA,
                        bad && !r.is_empty(),
                    );
                    region_ok(&res, bad);
                    res.map(|_| d.iter().flat_map(|v| v.to_be_bytes()).collect())
                }
                6 => {
                    let (mut d, st, bad) = dst(&mut rng, 4usize.saturating_mul(dx), dy, 0xAAAAu16);
                    let res = im.decode_region_rgba64(level, r, &mut d, st, &cancel);
                    check_dst(
                        &res,
                        &d,
                        4usize.saturating_mul(dx),
                        st,
                        0xAAAA,
                        bad && !r.is_empty(),
                    );
                    region_ok(&res, bad);
                    res.map(|_| be16(&d))
                }
                _ => {
                    let t = Rect::new(
                        ox,
                        oy,
                        ox.saturating_add(dx as i64).saturating_add(gx),
                        oy.saturating_add(dy as i64).saturating_add(gy),
                    );
                    YCbCr::new(t, sub, &limits(seed)).and_then(|mut y| {
                        let res = im.decode_region_ycbcr(level, r, &mut y, &cancel);
                        assert!(grow == 0 || matches!(res, Err(Error::InvalidArgument(_))));
                        res.map(|_| [y.y, y.cb, y.cr].concat())
                    })
                }
            }
        });
        if cancelled {
            if let Err(Error::Internal(e)) = &res {
                panic!("{e}");
            }
            out.push(Ok(Vec::new()));
        } else {
            out.push(res);
        }
    }
    out
}

/// A warm image gives what a fresh one gives: the same bytes, or the same kind of error.
fn same(warm: &[Out], cold: &[Out]) {
    assert_eq!(warm.len(), cold.len());
    for (w, c) in warm.iter().zip(cold) {
        for e in [w, c].into_iter().filter_map(|r| r.as_ref().err()) {
            assert!(!matches!(e, Error::Internal(_)), "{e}");
        }
        // Memory held by other decodes (or by a cache) is not the same on a warm and a cold image.
        if [w, c].iter().any(|r| matches!(r, Err(Error::MemoryLimit))) {
            continue;
        }
        match (w, c) {
            (Ok(a), Ok(b)) => assert!(a == b, "warm and cold decodes differ"),
            (Err(a), Err(b)) => assert_eq!(
                std::mem::discriminant(a),
                std::mem::discriminant(b),
                "{a} / {b}"
            ),
            _ => panic!("warm {:?}, cold {:?}", w.as_ref().err(), c.as_ref().err()),
        }
    }
}

// data = [seed: u64 LE][file]. The seed picks the limits, a memory limit and up to four decodes (tile or region,
// RGBA, RGBA64, YCbCr into planes at any small origin, samples, signed samples; level, tile and
// rectangle, in and out of range; some cancelled part way) on the image, its SDK view or its wide
// grid. They run in sequence on one image, so later ones read its cache, and
// the results must equal those of a fresh image of the file. Some rectangles have corners from the
// whole i64 range; destinations have padding between rows, which a decode must not write, and are
// sometimes 1 or 2 elements short (InvalidArgument); some YCbCr rectangles do not match the request
// (InvalidArgument). The same decodes then run on the image
// of the previous input (its cache and budget kept) against a fresh image of that file.
fuzz_target!(|data: &[u8]| {
    let Some((seed, file)) = data.split_first_chunk::<8>() else {
        return;
    };
    let seed = u64::from_le_bytes(*seed);
    let Ok(im) = open(file, seed) else { return };
    let im = im.with_decoder(dec());
    let cold = open(file, seed).expect("the second open of a file that opened");
    let cold_out = run(&cold, seed);
    same(&run(&im, seed), &cold_out);
    // The same decodes on an image whose reader fails or ends at an offset from the seed, which
    // go on to read the cache after an I/O error: every one that does not fail with an I/O error
    // gives what the fresh image gave.
    let flaky = Flaky {
        data: file.to_vec(),
        at: (seed.rotate_left(17) % (file.len() as u64 + 1)) as usize,
        short: seed >> 63 == 0,
    };
    if let Ok(fim) = Image::open(flaky, file.len() as u64, &limits(seed), &NO) {
        let fim = fim.with_decoder(dec());
        let (f, c): (Vec<_>, Vec<_>) = run(&fim, seed)
            .into_iter()
            .zip(cold_out)
            .filter(|(f, _)| !matches!(f, Err(Error::Io(_))))
            .unzip();
        same(&f, &c);
    }
    let mut prev = PREV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((bytes, pseed, pim)) = prev.as_ref() {
        let fresh = open(bytes, *pseed).expect("the second open of a file that opened");
        same(&run(pim, seed), &run(&fresh, seed));
    }
    *prev = Some((file.to_vec(), seed, im));
});
