//! `kernels`: times each SIMD kernel on each tier this CPU has and prints a markdown table: the
//! best of ROUNDS rounds, in ns per call, on the decoder's row lengths (`W`, `TS`) and a whole
//! tile plane (`TS * TS`), with deterministic xorshift data. Each call is the kernel and its
//! scalar tail, as the decoder runs it. The last column is the AVX-512 time over the AVX2 time.

use isyntax::fuzz_api::{Bench, TS, Tier, W, tiers};
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: u32 = 7;
/// Elements a round processes, so a round takes about a millisecond whatever the size.
const WORK: usize = 1 << 22;

type Kernel = fn(&mut Bench, Tier);

fn main() {
    let all = tiers();
    let kernels: [(&str, Kernel, [usize; 3]); 11] = [
        ("lift_odd", Bench::lift_odd, [W, TS, TS * TS]),
        ("lift_even", Bench::lift_even, [W, TS, TS * TS]),
        ("idwt_row", Bench::idwt_row, [W / 2, TS, TS * TS]),
        ("ycocg_to_rgba", Bench::ycocg_to_rgba, [W, TS, TS * TS]),
        ("ycocg_to_rgb16", Bench::ycocg_to_rgb16, [W, TS, TS * TS]),
        (
            "ycocg_to_samples",
            Bench::ycocg_to_samples,
            [W, TS, TS * TS],
        ),
        (
            "planes_to_samples",
            Bench::planes_to_samples,
            [W, TS, TS * TS],
        ),
        ("trunc_row", Bench::trunc_row, [W / 2, TS, TS * TS]),
        ("pack64", Bench::pack64, [W, TS, TS * TS]),
        ("rgba_to_ycbcr", Bench::rgba_to_ycbcr, [W, TS, TS * TS]),
        // The sizes are block areas (256 x 256 coefficients at most), 512 to 8192 byte positions.
        (
            "hulsken_unpack",
            Bench::unpack,
            [TS * TS / 16, TS * TS / 4, TS * TS],
        ),
    ];
    println!("| kernel | size | scalar ns | avx2 ns | avx512 ns | avx512/avx2 |");
    println!("|---|---|---|---|---|---|");
    for (name, run, sizes) in kernels {
        for n in sizes {
            let mut b = Bench::new(n);
            let ns = |want: &str, b: &mut Bench| -> Option<f64> {
                let t = all.iter().find(|t| t.name() == want)?;
                let iters = (WORK / n).max(1);
                let mut best = f64::MAX;
                for _ in 0..ROUNDS {
                    let start = Instant::now();
                    for _ in 0..iters {
                        run(black_box(&mut *b), *t);
                    }
                    best = best.min(start.elapsed().as_nanos() as f64 / iters as f64);
                }
                Some(best)
            };
            let (s, a2, a5) = (
                ns("scalar", &mut b),
                ns("avx2", &mut b),
                ns("avx512", &mut b),
            );
            // The unpack kernel needs GFNI as well: without it the avx512 column would be scalar.
            #[cfg(target_arch = "x86_64")]
            let a5 = a5.filter(|_| name != "hulsken_unpack" || is_x86_feature_detected!("gfni"));
            let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.0}"));
            let ratio = match (a2, a5) {
                (Some(a2), Some(a5)) => format!("{:.2}", a5 / a2),
                _ => "-".to_string(),
            };
            println!(
                "| {name} | {n} | {} | {} | {} | {ratio} |",
                show(s),
                show(a2),
                show(a5)
            );
        }
    }
}
