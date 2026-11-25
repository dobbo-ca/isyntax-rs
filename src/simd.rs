//! SIMD tiers for the sample kernels of `dsp`: AVX-512 or AVX2 on x86_64, NEON on aarch64, picked
//! once. `ISYNTAX_SIMD=scalar` forces the scalar tier and `ISYNTAX_SIMD=avx2` the AVX2 tier. A
//! kernel without an AVX-512 version runs its AVX2 one on the AVX-512 tier. Each kernel here
//! handles the whole-lane prefix of its input and returns how many elements (or, for `idwt_row`,
//! row positions) it did; the caller finishes the tail with the scalar code, which is also the oracle in the tests.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tier {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Needs AVX2 too: a kernel without an AVX-512 version runs its AVX2 one.
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Every tier this CPU has, scalar first and the best last, whatever `ISYNTAX_SIMD` says.
pub(crate) fn available() -> Vec<Tier> {
    let mut v = vec![Tier::Scalar];
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        v.push(Tier::Avx2);
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            v.push(Tier::Avx512);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        v.push(Tier::Neon);
    }
    v
}

/// The tier every kernel runs on: the best this CPU has, or the one `ISYNTAX_SIMD` (`scalar` or
/// `avx2`) names if the CPU has it.
pub(crate) fn tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| {
        let all = available();
        let forced = match std::env::var_os("ISYNTAX_SIMD") {
            Some(v) if v == "scalar" => Some(Tier::Scalar),
            #[cfg(target_arch = "x86_64")]
            Some(v) if v == "avx2" => Some(Tier::Avx2),
            _ => None,
        };
        match forced {
            Some(t) if all.contains(&t) => t,
            Some(_) => Tier::Scalar,
            None => all.last().copied().unwrap_or(Tier::Scalar),
        }
    })
}

/// Defines `$name` as the kernel of tier `t`, or 0 (nothing done) on the scalar tier. `$hi` is the
/// module whose kernel the AVX-512 tier runs: `avx512`, or `avx2` for a kernel with no AVX-512 version.
macro_rules! dispatch {
    ($(fn $name:ident($($a:ident: $ty:ty),*) $hi:ident;)*) => {$(
        #[allow(unsafe_code, clippy::too_many_arguments)]
        pub(crate) fn $name(t: Tier, $($a: $ty),*) -> usize {
            #[cfg(target_arch = "x86_64")]
            if t == Tier::Avx2 {
                // SAFETY: the tier is Avx2 only if is_x86_feature_detected!("avx2") held, the
                // one feature these kernels enable; their loads and stores take arrays.
                return unsafe { avx2::$name($($a),*) };
            }
            #[cfg(target_arch = "x86_64")]
            if t == Tier::Avx512 {
                // SAFETY: the tier is Avx512 only if is_x86_feature_detected! held for avx2,
                // avx512f and avx512bw, the features the avx2 and avx512 kernels enable; their
                // loads and stores take arrays.
                return unsafe { $hi::$name($($a),*) };
            }
            #[cfg(target_arch = "aarch64")]
            if t == Tier::Neon {
                // SAFETY: the tier is Neon only if is_aarch64_feature_detected!("neon") held, the
                // one feature these kernels enable; their loads and stores take arrays.
                return unsafe { neon::$name($($a),*) };
            }
            let _ = t;
            0
        }
    )*};
}

dispatch! {
    fn lift_odd(out: &mut [i16], l: &[i16], h0: &[i16], h1: &[i16]) avx2;
    fn lift_even(out: &mut [i16], h: &[i16], o0: &[i16], o1: &[i16]) avx512;
    fn idwt_row(out: &mut [i16], l: &[i16], h: &[i16]) avx512;
    fn ycocg_to_rgba(dst: &mut [u8], y: &[i16], co: &[i16], cg: &[i16], shift: u32) avx512;
    fn samples(dst: &mut [i16], a: &[i16], b: &[i16], c: &[i16], lo: i16, hi: i16, planes: bool) avx512;
    fn trunc_row(row: &mut [i16], mask: i16, o: i16) avx2;
}

/// Like `dispatch!` for a kernel that has an AVX-512 version only: it runs on that tier and does
/// nothing (returns 0) on the others.
macro_rules! dispatch512 {
    ($(fn $name:ident($($a:ident: $ty:ty),*);)*) => {$(
        #[allow(unsafe_code)]
        pub(crate) fn $name(t: Tier, $($a: $ty),*) -> usize {
            #[cfg(target_arch = "x86_64")]
            if t == Tier::Avx512 {
                // SAFETY: the tier is Avx512 only if is_x86_feature_detected! held for avx2,
                // avx512f and avx512bw, the features this kernel enables; its loads and stores
                // take arrays.
                return unsafe { avx512::$name($($a),*) };
            }
            let _ = (t, $(&$a),*);
            0
        }
    )*};
}

dispatch512! {
    fn pack64(dst: &mut [u16], src: &[i16], bits: u32);
    fn rgba_to_ycbcr(y: &mut [u8], cb: &mut [u8], cr: &mut [u8], rgba: &[u8]);
}

/// The Hulsken bit-plane unpack (`hulsken::unpack`) for whole blocks of 64 byte positions, on
/// the AVX-512 tier of a CPU that also has GFNI (`ISYNTAX_SIMD=avx2` or `scalar` turn it off):
/// returns the positions done, 0 on any other tier. `lo` and `hi` say whether planes 0 to 7 and 8
/// to 15 have any data.
#[allow(unsafe_code)]
pub(crate) fn unpack(
    t: Tier,
    d: &mut [i16],
    bw: usize,
    blk: &[i32],
    p: &[&[u8]; 16],
    lo: bool,
    hi: bool,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if t == Tier::Avx512 && is_x86_feature_detected!("gfni") {
        // SAFETY: the tier is Avx512 only if is_x86_feature_detected! held for avx2, avx512f and
        // avx512bw, and gfni was just detected: the features this kernel enables. Its loads and
        // stores take arrays, and its stores into `d` are indexed.
        return unsafe { avx512::unpack(d, bw, blk, p, lo, hi) };
    }
    let _ = (t, d, bw, blk, p, lo, hi);
    0
}
