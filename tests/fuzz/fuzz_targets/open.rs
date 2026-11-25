#![no_main]
use isyntax::{Error, Image};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

#[path = "../common.rs"]
mod common;
use common::{Flaky, limits, with_cancel};

// data = the file. Opens it (the header parse: XML, base64, cluster or block header table, tile
// index) and reads every accessor of the image; it must return an error or an image, never panic
// or allocate past the limits (the limits, chosen from the file's bytes, and -malloc_limit_mb are
// the backstop). It is also opened through a reader that fails or ends early at an offset taken
// from the file's own bytes, with a size that is off by up to two, and a tile is decoded from it.
// Both opens may be cancelled from another thread after a delay taken from the bytes.
fuzz_target!(|data: &[u8]| {
    let h = data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
        (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
    });
    let limits = limits(h);
    // A cancel after 0 to 2047 yields in a quarter of the runs.
    let delay = if h >> 42 & 3 == 0 {
        (h >> 44) % 2048
    } else {
        0
    };
    let flaky = Flaky {
        data: data.to_vec(),
        at: (h % (data.len() as u64 + 1)) as usize,
        short: h >> 32 & 1 == 0,
    };
    let size = (data.len() as u64 + (h >> 40) % 5).saturating_sub(2);
    match with_cancel(delay, |c| Image::open(flaky, size, &limits, c)) {
        Ok(im) => {
            let mut d = vec![0u8; 4 * 256 * 256];
            let r = im.decode_tile_rgba(0, 0, 0, &mut d, 4 * 256, &AtomicBool::new(false));
            assert!(!matches!(r, Err(Error::Internal(_))), "{r:?}");
        }
        Err(e) => assert!(!matches!(e, Error::Internal(_)), "{e}"),
    }
    let im = match with_cancel(delay, |c| {
        Image::open(data.to_vec(), data.len() as u64, &limits, c)
    }) {
        Ok(im) => im,
        Err(e) => {
            assert!(!matches!(e, Error::Internal(_)), "{e}");
            return;
        }
    };
    let _ = (
        im.label(),
        im.macro_image(),
        im.icc_profile(),
        im.mpp(),
        im.barcode(),
    );
    let _ = (
        im.model_name(),
        im.device_serial(),
        im.acquisition_date_time(),
    );
    let _ = (
        im.calibration_status(),
        im.date_of_last_calibration(),
        im.time_of_last_calibration(),
    );
    let _ = (
        im.rack_slot(),
        im.interface_version(),
        im.bits_stored(),
        im.tile_size(),
    );
    for l in 0..im.levels() {
        assert!(im.level(l).is_some());
        let _ = (im.source_bounds(l), im.envelopes(l));
    }
    assert!(im.level(im.levels()).is_none());
    let _ = im.envelopes(usize::MAX);
    let _ = im.wide_grid().level(0);
    let _ = im.sdk_truncated();
});
