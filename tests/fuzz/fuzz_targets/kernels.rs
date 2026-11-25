#![no_main]
use isyntax::fuzz_api as k;
use libfuzzer_sys::fuzz_target;

/// The allocation limit of the helpers: far above what 255 + 15 elements need.
const MAX: u64 = 1 << 24;

fn check() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let t = k::tiers();
        // x86_64 CI runners have AVX2 (some AVX-512) and aarch64 runners have NEON: with only
        // the scalar tier this would fuzz scalar against scalar.
        assert!(t.len() > 1, "no SIMD tier: scalar against scalar");
        let names: Vec<_> = t.iter().map(|t| t.name()).collect();
        eprintln!("SIMD tiers fuzzed against scalar: {}", names.join(" "));
        #[cfg(target_arch = "x86_64")]
        eprintln!(
            "gfni: {} (without it the hulsken_unpack case fuzzes scalar against scalar)",
            is_x86_feature_detected!("gfni")
        );
    });
}

// data = [kernel][n][offsets a, b, c][e0, e1, e2][bytes]: the kernel (lift_odd, lift_even,
// idwt_row, ycocg_to_rgba, the four sample kernels, trunc_row) in every SIMD tier of this CPU
// (AVX2 and AVX-512 on x86_64, NEON on aarch64) equals its scalar reference, on operands that are sub-slices of one buffer starting 0 to 3
// elements into their 300-element spans (so not aligned to a vector), have different lengths (about n, from
// n / 4 to n + 15; bits 2 to 7 of each offset byte) and fill an output that starts 0 to 3
// elements into its buffer and is longer than all of them, with values cycled from the bytes.
// e0..e2 are the shift, the clamp, the mask and the rounding offset; the output offset is bits 3
// and 4 of the kernel byte. With bit 7 of the kernel byte set the kernel is instead pack64 (8 or
// 9 bits, bit 0 of e0 flips it; kernel % 4 of 0 or 1), rgba_to_ycbcr on `a`'s bytes as pixels (2),
// or the Hulsken unpack of a block with 24 to 68 coefficients a side, bits of e1 and the last
// byte as the planes present and e2 as the zeroed 16-byte groups (3).
fuzz_target!(|data: &[u8]| {
    check();
    let Some((&[kernel, n, oa, ob, oc, e0, e1, e2], bytes)) = data.split_first_chunk::<8>() else {
        return;
    };
    if bytes.len() < 2 {
        return;
    }
    let n = usize::from(n);
    const SPAN: usize = 300;
    let all: Vec<i16> = (0..3 * SPAN)
        .map(|i| {
            i16::from_le_bytes([
                bytes[(2 * i) % bytes.len()],
                bytes[(2 * i + 1) % bytes.len()],
            ])
        })
        .collect();
    let buf = |span: usize, off: u8| {
        let len = (n >> (off >> 6)) + usize::from((off >> 2) & 15);
        let start = span * SPAN + usize::from(off & 3);
        start..start + len
    };
    let (a, b, c) = (&all[buf(0, oa)], &all[buf(1, ob)], &all[buf(2, oc)]);
    let skip = usize::from((kernel >> 3) & 3);
    let wide = i16::from_le_bytes([e0, e1]);
    // Inputs of the kernels picked by bit 7 of the kernel byte: pixels from the bytes of `a`, the
    // side of a block of 24 to 68 (whole blocks of 64 positions and a tail) and its planes present.
    let rgba: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bw = 24 + 4 * (n % 12);
    let present = u16::from_le_bytes([e1, bytes[bytes.len() - 1]]);
    for t in k::tiers() {
        let r = if kernel & 0x80 != 0 {
            match kernel % 4 {
                0 => k::pack64(t, a, e0, skip, MAX),
                1 => k::pack64(t, a, e0 ^ 1, skip, MAX),
                2 => k::rgba_to_ycbcr(t, &rgba, skip, MAX),
                _ => k::unpack(t, bw, bytes, present, e2, skip, MAX),
            }
        } else {
            match kernel % 8 {
                0 => k::lift_odd(t, a, b, c, skip, MAX),
                1 => k::lift_even(t, a, b, c, skip, MAX),
                2 => k::idwt_row(t, a, b, skip, MAX),
                3 => k::ycocg_to_rgba(t, a, b, c, u32::from(e0), skip, MAX),
                4 => k::samples(t, a, b, c, wide & 0x7fff, e2, skip, MAX),
                5 => k::samples(t, a, b, c, wide & 0x7fff, e2 ^ 2, skip, MAX),
                6 => k::trunc_row(t, a, wide, i16::from_le_bytes([e2, bytes[0]]), skip, MAX),
                _ => k::ycocg_to_rgba(t, a, b, c, u32::from(e0) % 16, skip, MAX),
            }
        };
        if let Err(e) = r {
            panic!("{e}");
        }
    }
});
