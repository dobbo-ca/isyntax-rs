//! `ppm <in> <out>`: for every file of `<in>`, in sorted order, opens it and writes every tile of
//! every level, decoded as RGBA, as a binary P6 (alpha dropped) named
//! `<file name with . as _>.l<level>.<x>.<y>.ppm` to `<out>`. A file that does not open and a tile
//! that does not decode write nothing and are named on stderr. All files share one
//! `Decoder`. The Rust side of the quality gate (`.github/workflows/quality.yml`); the Go side is `tests/parity/go/oracle -ppm`.

use isyntax::{Decoder, Image, Limits};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::{env, fs, io};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: ppm <in> <out>");
        std::process::exit(2);
    };
    fs::create_dir_all(output)?;
    let mut names: Vec<String> = Vec::new();
    for e in fs::read_dir(input)? {
        let e = e?;
        if e.file_type()?.is_file() {
            names.push(e.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    // No memory or pixel limit, as the Go side has none.
    let limits = Limits {
        max_alloc: u64::MAX,
        max_pixels: u64::MAX,
        ..Limits::default()
    };
    // One Decoder for every file: the buffers of a file serve the next, so the exact gate also
    // checks reuse across files.
    let decoder = Decoder::new(&limits);
    let cancel = AtomicBool::new(false);
    let mut count = 0usize;
    for name in &names {
        let data = fs::read(Path::new(input).join(name))?;
        let size = data.len() as u64;
        let im = match Image::open(data, size, &limits, &cancel) {
            Ok(im) => im.with_decoder(&decoder),
            Err(e) => {
                eprintln!("{name}: open: {e}");
                continue;
            }
        };
        let stem = name.replace('.', "_");
        for l in 0..im.levels() {
            let Some(lv) = im.level(l) else { continue };
            let side = if l == im.levels() - 1 {
                128
            } else {
                im.tile_size()
            };
            let mut rgba = vec![0u8; 4 * side * side];
            for y in 0..lv.tiles_y {
                for x in 0..lv.tiles_x {
                    // A fresh tile as on the Go side, whatever a decode leaves untouched.
                    rgba.fill(0);
                    if let Err(e) =
                        im.decode_tile_rgba(l, x as i64, y as i64, &mut rgba, 4 * side, &cancel)
                    {
                        eprintln!("{name}: tile {l}.{x}.{y}: {e}");
                        continue;
                    }
                    let mut buf = format!("P6\n{side} {side}\n255\n").into_bytes();
                    buf.reserve(3 * side * side);
                    for px in rgba.as_chunks::<4>().0 {
                        buf.extend_from_slice(&px[..3]);
                    }
                    fs::write(
                        Path::new(output).join(format!("{stem}.l{l}.{x}.{y}.ppm")),
                        buf,
                    )?;
                    count += 1;
                }
            }
        }
    }
    println!("{count} tiles decoded");
    Ok(())
}
