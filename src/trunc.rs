//! SDK default truncation: the Philips SDK SourceView drops detail bits of Q1 and Q2 files, and
//! `Image::sdk_truncated` decodes like it. Port of `internal/decoder/trunc.go`; see
//! isyntax-go's docs/format.md, "SDK default truncation".

use crate::container::Header;
use crate::dsp::trunc_row;
use crate::{Error, decoder};

/// The bits dropped per scale level and colour, and whether a nonzero value gets `2^(k-1) - 1`
/// back.
pub(crate) struct TruncTable {
    /// `[scale level][colour]` bits dropped.
    pub k: [[u8; 3]; 2],
    pub round: bool,
}

// Measured presets only: the Philips SDK SourceView default truncation.
pub(crate) static TRUNC_Q2B8: TruncTable = TruncTable {
    k: [[1, 4, 4], [0, 2, 2]],
    round: true,
};
pub(crate) static TRUNC_Q2B9: TruncTable = TruncTable {
    k: [[3, 3, 3], [2, 3, 3]],
    round: true,
};
pub(crate) static TRUNC_Q1_RGB: TruncTable = TruncTable {
    k: [[0, 4, 4], [0, 0, 0]],
    round: false,
};

/// Recon variant 1 (output) reads its parent as variant 2.
pub(crate) fn parent_variant(v: u8) -> u8 {
    if v == 1 { 2 } else { 0 }
}

/// The table the SDK would apply to this file; `None` is no truncation.
pub(crate) fn sdk_preset(h: &Header) -> Result<Option<&'static TruncTable>, Error> {
    let mut q = h.quality;
    if q < 0 && h.major >= 100 && h.bits_stored == 8 {
        q = 2; // the SDK reports Q2
    }
    let v2 = h.major >= 100;
    if q == 0 {
        return Ok(None);
    }
    if q == 2 && v2 && !h.rgb && h.bits_stored == 8 {
        return Ok(Some(&TRUNC_Q2B8));
    }
    if q == 2 && v2 && !h.rgb && h.bits_stored == 9 {
        return Ok(Some(&TRUNC_Q2B9));
    }
    if q == 1 && !v2 && h.rgb && h.bits_stored == 8 && h.levels == 1 {
        return Ok(Some(&TRUNC_Q1_RGB));
    }
    Err(Error::Unsupported(format!(
        "no measured SDK truncation for Quality={}, {} bits stored, interface {}",
        h.quality, h.bits_stored, h.interface_version
    )))
}

impl TruncTable {
    /// Truncates colour `c`'s detail quadrants of the `W x W` inverse-transform buffer at scale
    /// level `j`; the LL quadrant stays.
    pub(crate) fn apply(&self, buf: &mut [i16], j: usize, c: usize) {
        let k = self.k[j][c];
        if k == 0 {
            return;
        }
        let mask = (1i16 << k).wrapping_sub(1);
        let o = if self.round {
            (1i16 << (k - 1)).wrapping_sub(1)
        } else {
            0
        };
        for (r, row) in buf
            .as_chunks_mut::<{ decoder::W }>()
            .0
            .iter_mut()
            .enumerate()
        {
            let row: &mut [i16] = if r < decoder::Q {
                &mut row[decoder::Q..]
            } else {
                row
            };
            trunc_row(row, mask, o);
        }
    }
}
