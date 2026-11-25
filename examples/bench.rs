//! `bench <fixtures> <n> [threads]`: prints name, output bytes, ns per decode of level 0 of each
//! fixture, each time from a fresh `Image` (a cold cache) that shares one `Decoder` per fixture
//! (warm buffers), and the FNV-1a 64 of the pixels, as `oracle -bench` does for Go. `threads` is
//! `Limits::max_threads` (default: available parallelism). A fifth column is the minor page faults
//! per decode (0 off Linux). See `tests/parity/bench.sh`.

use isyntax::{Decoder, Image, Limits, Rect};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::{env, fs, io};

/// Minor page faults of this process so far: field 10 of `/proc/self/stat`, 0 if unreadable.
fn minflt() -> u64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // The command name is in parentheses and may hold spaces: count the fields after its end.
    let rest = stat
        .rfind(')')
        .and_then(|i| stat.get(i + 1..))
        .unwrap_or("");
    rest.split_whitespace()
        .nth(7)
        .and_then(|f| f.parse().ok())
        .unwrap_or(0)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (dir, n, threads) = match args.as_slice() {
        [_, dir, n] => (dir, n, None),
        [_, dir, n, t] => (dir, n, t.parse::<usize>().ok()),
        _ => {
            eprintln!("usage: bench <fixtures> <n> [threads]");
            std::process::exit(2);
        }
    };
    let n: u32 = n.parse().unwrap_or(1).max(1);
    let defaults = Limits::default();
    let limits = Limits {
        max_alloc: u64::MAX,
        max_threads: threads.unwrap_or(defaults.max_threads),
        ..defaults
    };
    let cancel = AtomicBool::new(false);
    let mut names: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let data = Arc::new(fs::read(format!("{dir}/{name}"))?);
        let mut dst = Vec::new();
        let decoder = Decoder::new(&limits);
        let mut one = || -> Result<(), isyntax::Error> {
            let im = Image::open(data.clone(), data.len() as u64, &limits, &cancel)?
                .with_decoder(&decoder);
            let lv = im
                .level(0)
                .ok_or_else(|| isyntax::Error::Internal("no level 0".into()))?;
            if dst.is_empty() {
                dst = vec![0u8; 4 * lv.width * lv.height];
            }
            let r = Rect::new(0, 0, lv.width as i64, lv.height as i64);
            im.decode_region_rgba(0, r, &mut dst, 4 * lv.width, &cancel)
        };
        if let Err(e) = one() {
            eprintln!("{name}: {e}");
            std::process::exit(1);
        }
        let faults = minflt();
        let t = Instant::now();
        for _ in 0..n {
            if let Err(e) = one() {
                eprintln!("{name}: {e}");
                std::process::exit(1);
            }
        }
        let ns = t.elapsed().as_nanos() / u128::from(n);
        let faults = minflt().saturating_sub(faults) / u64::from(n);
        let fnv = dst.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
            (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
        });
        println!("{name}\t{}\t{ns}\t{fnv:016x}\t{faults}", dst.len());
    }
    Ok(())
}
