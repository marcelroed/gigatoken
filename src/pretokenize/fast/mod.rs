//! Fast scalar pretokenizers, one submodule per pretokenization scheme.
//!
//! Each scheme implements an advance function that consumes exactly one
//! pretoken, wrapped in a thin iterator struct. The byte predicates and
//! SWAR scans below are shared; a new scheme (e.g. o200k) should slot in
//! as another submodule reusing these primitives where its character
//! classes line up.

pub(crate) mod cl100k_family;
pub(crate) mod mask;

pub mod cl100k;
pub mod deepseek_v3;
pub mod olmo3;
pub mod qwen2;
pub mod qwen3_5;
pub mod r50k;

pub use cl100k::FastCl100kPretokenizer;
pub use deepseek_v3::FastDeepSeekV3Pretokenizer;
pub use olmo3::FastOlmo3Pretokenizer;
pub use qwen2::FastQwen2Pretokenizer;
pub use qwen3_5::FastQwen35Pretokenizer;
pub use r50k::FastR50kPretokenizer;

use crate::pretokenize::SpanBatch;
use crate::pretokenize::unicode;

// -----------------------------------------------------------------------
// Shared chunked span pull for the mask-scanner pretokenizers
// -----------------------------------------------------------------------

/// The `PretokenSpans::fill_spans_keyed` body shared by every mask-scanner
/// pretokenizer (all of them wrap a `(bytes, MaskState)` pair). With a SIMD
/// scanner this is the two-phase chunk walker
/// ([`mask::MaskState::fill_spans_two_phase`]): boundary harvest into a
/// flat buffer, then a branch-free emission loop — the per-span refill
/// ladder and pack branches of the fused pull loop were the largest single
/// source of encode's discarded issue bandwidth. Without SIMD support it
/// pulls spans one at a time over `next_span`, fusing its
/// `#[inline(always)]` walker body into one tight loop. `#[inline(never)]`:
/// each monomorphization is its own out-of-line loop, keeping its register
/// allocation away from the (register-hungry) encode loop that calls it.
/// Routing this through `Iterator::next` instead measured ~23% of warm
/// encode time in un-inlined call overhead.
#[inline(never)]
pub(crate) fn fill_spans_keyed_mask<'a, S: mask::MaskScheme>(
    bytes: &'a [u8],
    state: &mut mask::MaskState,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if mask::simd_scanner_available() {
        return state.fill_spans_two_phase::<S>(bytes, batch, prefetch);
    }
    crate::pretokenize::fill_spans_keyed_with(
        || {
            let (start, end) = state.next_span::<S>(bytes)?;
            // SAFETY: next_span returns in-bounds span boundaries.
            Some(unsafe { bytes.get_unchecked(start..end) })
        },
        batch,
        prefetch,
    )
}

/// Implement [`crate::pretokenize::PretokenSpans`] for a mask-scanner
/// pretokenizer (any `{ bytes, state: MaskState }` struct) by delegating
/// to its scheme's [`fill_spans_keyed_mask`] monomorphization.
macro_rules! impl_mask_pretoken_spans {
    ($pretokenizer:ident, $scheme:ty) => {
        impl<'a> crate::pretokenize::PretokenSpans<'a> for $pretokenizer<'a> {
            #[inline]
            fn fill_spans_keyed(
                &mut self,
                batch: &mut crate::pretokenize::SpanBatch<'a>,
                prefetch: &impl Fn(u64),
            ) -> usize {
                crate::pretokenize::fast::fill_spans_keyed_mask::<$scheme>(
                    self.bytes,
                    &mut self.state,
                    batch,
                    prefetch,
                )
            }
        }
    };
}
pub(crate) use impl_mask_pretoken_spans;

// -----------------------------------------------------------------------
// Branchless byte predicates
// -----------------------------------------------------------------------

#[inline(always)]
pub(crate) fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(crate) fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(crate) fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

#[inline(always)]
pub(crate) unsafe fn decode_non_ascii(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

/// Decode one non-ASCII scalar from valid UTF-8. `bytes[pos]` must be a UTF-8
/// leading byte (>= 0xC2) with its full sequence in bounds. Returns the
/// codepoint and the sequence length in bytes.
#[inline(always)]
pub(crate) unsafe fn decode_cp(bytes: &[u8], pos: usize) -> (u32, usize) {
    unsafe {
        let b0 = *bytes.get_unchecked(pos) as u32;
        let b1 = (*bytes.get_unchecked(pos + 1) & 0x3F) as u32;
        if b0 < 0xE0 {
            return (((b0 & 0x1F) << 6) | b1, 2);
        }
        let b2 = (*bytes.get_unchecked(pos + 2) & 0x3F) as u32;
        if b0 < 0xF0 {
            return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
        }
        let b3 = (*bytes.get_unchecked(pos + 3) & 0x3F) as u32;
        (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3, 4)
    }
}

/// `[\r\n]*`: advance past a run of CR/LF bytes (trailing newlines after a
/// punctuation run in the cl100k-family schemes).
#[inline(always)]
pub(crate) fn scan_newlines(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if b == b'\r' || b == b'\n' {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// If the char at `pos` is a letter (`\p{L}` under the 4-way `CharClass`
/// classifier), return the offset just past it.
#[inline(always)]
pub(crate) fn letter_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if unicode::class_of(cp) == unicode::CharClass::Letter {
            return Some(pos + l);
        }
    }
    None
}

// -----------------------------------------------------------------------
// SWAR
// -----------------------------------------------------------------------

pub(crate) const HI: u64 = 0x8080_8080_8080_8080;

/// Returns the high bit set in each lane that is NOT an ASCII letter,
/// computed directly (rather than as the complement of a letter mask) so
/// the scan loop can branch on `!= 0` and reuse the value for `trailing_zeros`.
#[inline(always)]
pub(crate) fn swar64_letter_nonmask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HI
}

/// SWAR letter scan: advances `pos` past ASCII letters.
/// Returns the updated pos.
#[inline(always)]
pub(crate) fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    // SWAR: 8 bytes at a time
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let nonletter = swar64_letter_nonmask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    // Scalar tail
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// NEON letter scan: 16 bytes per iteration. Non-ASCII bytes (>= 0x80) fail
/// the `<= 'z'` check after case-folding, so they stop the run exactly like
/// non-letters; the caller's unicode continuation handles them.
///
/// NOT used by `scan_letters_from`: measured 0.83x of the SWAR scan on OWT.
/// The `vshrn`-based movemask needs a vector→GPR transfer whose latency sits
/// on the serial per-token chain, and typical letter runs (~4-6 bytes) fit in
/// one SWAR iteration anyway. Kept as a reference / benchmark baseline.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn neon_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    use std::arch::aarch64::*;
    let len = bytes.len();
    while pos + 16 <= len {
        unsafe {
            let v = vld1q_u8(bytes.as_ptr().add(pos));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            let ge_a = vcgeq_u8(lowered, vdupq_n_u8(b'a'));
            let le_z = vcleq_u8(lowered, vdupq_n_u8(b'z'));
            let nonletter = vmvnq_u8(vandq_u8(ge_a, le_z));
            // Narrowing movemask: 4 bits per lane, first set nibble = first
            // non-letter lane.
            let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(
                vreinterpretq_u16_u8(nonletter),
            )));
            if mask != 0 {
                return pos + (mask.trailing_zeros() >> 2) as usize;
            }
        }
        pos += 16;
    }
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_letter(b) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// -----------------------------------------------------------------------
// Shared run scans (`\p{L}+`, `\p{N}+`, `\p{N}{1,3}`, `[^\s\p{L}\p{N}]+`)
// -----------------------------------------------------------------------

/// `\p{N}{1,3}`: extend a number run that already matched `consumed` chars
/// to at most 3 chars total. Shared by the cl100k and olmo3 schemes.
#[inline(always)]
pub(crate) fn scan_numbers_max3(bytes: &[u8], mut pos: usize, mut consumed: u32) -> usize {
    let len = bytes.len();
    while consumed < 3 && pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_digit(b) {
            pos += 1;
            consumed += 1;
            continue;
        }
        if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                pos += l;
                consumed += 1;
                continue;
            }
        }
        break;
    }
    pos
}

#[inline(always)]
pub(crate) fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Letter {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_digits_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len && is_digit(unsafe { *bytes.get_unchecked(p) }) {
            p += 1;
        }
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Number {
                p += l;
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
pub(crate) fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 { break; }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) { return p; }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == unicode::CharClass::Other {
                p += l;
                continue;
            }
        }
        return p;
    }
}

