#![allow(unsafe_op_in_unsafe_fn)]
//! AVX-512 accelerated pretokenizer for the GPT-2 regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Uses AVX-512BW/VL byte-level comparisons with k-register masks for fast
//! scanning of character runs.

use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::unicode;

use std::arch::x86_64::*;

// ==========================================================================
// Public iterator
// ==========================================================================

pub struct Avx512PretokenizerIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Avx512PretokenizerIter<'a> {
    #[inline]
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            bytes: input,
            pos: 0,
        }
    }
}

impl<'a> Iterator for Avx512PretokenizerIter<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.bytes;
        let pos = self.pos;
        if pos >= bytes.len() {
            return None;
        }
        let end = unsafe { find_pretoken_end(bytes, pos) };
        self.pos = end;
        Some(Pretoken(unsafe { bytes.get_unchecked(pos..end) }))
    }

    #[inline]
    fn count(self) -> usize {
        count_pretokens_inner(self.bytes, self.pos)
    }
}

/// Two-cursor counting: splits input at a safe boundary and processes
/// both halves interleaved, allowing OoO execution to overlap latencies.
#[inline]
fn count_pretokens_inner(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    if len - start < 4096 {
        return count_simple(bytes, start);
    }

    // Find a safe split near the middle: a newline preceded by a non-ws byte
    let mid = start + (len - start) / 2;
    let split = find_split(bytes, mid);
    if split == 0 {
        return count_simple(bytes, start);
    }

    let mut p1 = start;
    let mut p2 = split;
    let mut count = 0usize;

    // Interleaved processing: two independent cursor chains
    while p1 < split && p2 < len {
        p1 = unsafe { find_pretoken_end(bytes, p1) };
        p2 = unsafe { find_pretoken_end(bytes, p2) };
        count += 2;
    }
    while p1 < split {
        p1 = unsafe { find_pretoken_end(bytes, p1) };
        count += 1;
    }
    while p2 < len {
        p2 = unsafe { find_pretoken_end(bytes, p2) };
        count += 1;
    }
    count
}

/// Simple single-cursor counting loop.
fn count_simple(bytes: &[u8], mut pos: usize) -> usize {
    let mut count = 0usize;
    while pos < bytes.len() {
        pos = unsafe { find_pretoken_end(bytes, pos) };
        count += 1;
    }
    count
}

/// Find a token boundary near `target` to split the input.
/// Returns a position where a new token is guaranteed to start,
/// or 0 if no safe split found.
fn find_split(bytes: &[u8], target: usize) -> usize {
    // Search forward for a newline followed by a non-whitespace byte
    let search_end = (target + 256).min(bytes.len());
    for i in target..search_end {
        if bytes[i] == b'\n' {
            let after = i + 1;
            if after < bytes.len() && bytes[after] != b' ' && bytes[after].wrapping_sub(9) >= 5 {
                return after;
            }
        }
    }
    // Search backward
    let search_start = target.saturating_sub(256);
    for i in (search_start..target).rev() {
        if bytes[i] == b'\n' {
            let after = i + 1;
            if after < bytes.len() && bytes[after] != b' ' && bytes[after].wrapping_sub(9) >= 5 {
                return after;
            }
        }
    }
    0
}

// ==========================================================================
// Dispatch — comparison-based for minimal data dependencies
// ==========================================================================

#[inline]
unsafe fn find_pretoken_end(bytes: &[u8], pos: usize) -> usize {
    let first = *bytes.get_unchecked(pos);

    // Hot path: ASCII letter (~40% of tokens start with a letter)
    if (first | 0x20).wrapping_sub(b'a') < 26 {
        return scan_full_letter_run(bytes, pos);
    }

    // Second hottest: space before content
    if first == b' ' {
        return space_start(bytes, pos);
    }

    if first >= 0x80 {
        return non_ascii_start(bytes, pos);
    }

    if first.wrapping_sub(b'0') < 10 {
        return scan_full_digit_run(bytes, pos);
    }

    if first == b'\'' {
        return contraction_or_other(bytes, pos);
    }

    if first.wrapping_sub(9) < 5 {
        return whitespace_boundary(bytes, pos);
    }

    scan_full_other_run(bytes, pos)
}

#[inline]
unsafe fn space_start(bytes: &[u8], pos: usize) -> usize {
    let next = pos + 1;
    if next >= bytes.len() {
        return next;
    }
    let b = *bytes.get_unchecked(next);

    if (b | 0x20).wrapping_sub(b'a') < 26 {
        return scan_full_letter_run(bytes, next);
    }
    if b.wrapping_sub(b'0') < 10 {
        return scan_full_digit_run(bytes, next);
    }
    if b == b' ' || b.wrapping_sub(9) < 5 {
        return whitespace_boundary(bytes, pos);
    }
    if b >= 0x80 {
        return space_then_non_ascii(bytes, pos, next);
    }
    scan_full_other_run(bytes, next)
}

#[cold]
unsafe fn space_then_non_ascii(bytes: &[u8], space_pos: usize, char_pos: usize) -> usize {
    let c = decode_char_unchecked(bytes, char_pos);
    if unicode::is_letter(c) {
        scan_letter_after_unicode(bytes, char_pos, c)
    } else if unicode::is_number(c) {
        scan_digit_after_unicode(bytes, char_pos, c)
    } else if unicode::is_whitespace(c) {
        whitespace_boundary(bytes, space_pos)
    } else {
        scan_other_after_unicode(bytes, char_pos, c)
    }
}

#[cold]
unsafe fn non_ascii_start(bytes: &[u8], pos: usize) -> usize {
    let c = decode_char_unchecked(bytes, pos);
    if unicode::is_letter(c) {
        scan_letter_after_unicode(bytes, pos, c)
    } else if unicode::is_number(c) {
        scan_digit_after_unicode(bytes, pos, c)
    } else if unicode::is_whitespace(c) {
        whitespace_boundary(bytes, pos)
    } else {
        scan_other_after_unicode(bytes, pos, c)
    }
}

// ==========================================================================
// Contractions
// ==========================================================================

#[inline]
unsafe fn contraction_or_other(bytes: &[u8], pos: usize) -> usize {
    let a = pos + 1;
    if a < bytes.len() {
        match *bytes.get_unchecked(a) {
            b's' | b'd' | b'm' | b't' => return a + 1,
            b'l' if a + 1 < bytes.len() && *bytes.get_unchecked(a + 1) == b'l' => return a + 2,
            b'v' if a + 1 < bytes.len() && *bytes.get_unchecked(a + 1) == b'e' => return a + 2,
            b'r' if a + 1 < bytes.len() && *bytes.get_unchecked(a + 1) == b'e' => return a + 2,
            _ => {}
        }
    }
    scan_full_other_run(bytes, pos)
}

// ==========================================================================
// Whitespace boundary: \s+(?!\S) | \s+
// ==========================================================================

#[inline]
unsafe fn whitespace_boundary(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start;

    while i < len {
        let b = *bytes.get_unchecked(i);
        if b == b' ' || b.wrapping_sub(9) < 5 {
            i += 1;
        } else if b >= 0x80 {
            let c = decode_char_unchecked(bytes, i);
            if unicode::is_whitespace(c) {
                i += c.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    if i >= len {
        return i;
    }

    let mut last_start = i - 1;
    while last_start > start && *bytes.get_unchecked(last_start) & 0xC0 == 0x80 {
        last_start -= 1;
    }

    if last_start <= start {
        return i;
    }

    last_start
}

// ==========================================================================
// SIMD scan primitives — 32-byte chunks
// ==========================================================================

#[inline]
unsafe fn scan_ascii_letters(bytes: &[u8], start: usize) -> usize {
    let ptr = bytes.as_ptr();
    let len = bytes.len();
    let mut i = start;

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let lower = _mm256_or_si256(chunk, _mm256_set1_epi8(0x20u8 as i8));
        let diff = _mm256_sub_epi8(lower, _mm256_set1_epi8(b'a' as i8));
        let mask: u32 = _mm256_cmplt_epu8_mask(diff, _mm256_set1_epi8(26));
        if mask != u32::MAX {
            return i + mask.trailing_ones() as usize;
        }
        i += 32;
    }

    while i < len && (*ptr.add(i) | 0x20).wrapping_sub(b'a') < 26 {
        i += 1;
    }
    i
}

#[inline]
unsafe fn scan_ascii_digits(bytes: &[u8], start: usize) -> usize {
    let ptr = bytes.as_ptr();
    let len = bytes.len();
    let mut i = start;

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let diff = _mm256_sub_epi8(chunk, _mm256_set1_epi8(b'0' as i8));
        let mask: u32 = _mm256_cmplt_epu8_mask(diff, _mm256_set1_epi8(10));
        if mask != u32::MAX {
            return i + mask.trailing_ones() as usize;
        }
        i += 32;
    }

    while i < len && (*ptr.add(i)).wrapping_sub(b'0') < 10 {
        i += 1;
    }
    i
}

#[inline]
unsafe fn scan_ascii_other(bytes: &[u8], start: usize) -> usize {
    let ptr = bytes.as_ptr();
    let len = bytes.len();
    let mut i = start;

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let lower = _mm256_or_si256(chunk, _mm256_set1_epi8(0x20u8 as i8));
        let diff_l = _mm256_sub_epi8(lower, _mm256_set1_epi8(b'a' as i8));
        let is_letter: u32 = _mm256_cmplt_epu8_mask(diff_l, _mm256_set1_epi8(26));
        let diff_d = _mm256_sub_epi8(chunk, _mm256_set1_epi8(b'0' as i8));
        let is_digit: u32 = _mm256_cmplt_epu8_mask(diff_d, _mm256_set1_epi8(10));
        let is_space: u32 = _mm256_cmpeq_epi8_mask(chunk, _mm256_set1_epi8(b' ' as i8));
        let diff_w = _mm256_sub_epi8(chunk, _mm256_set1_epi8(9));
        let is_ctrl_ws: u32 = _mm256_cmplt_epu8_mask(diff_w, _mm256_set1_epi8(5));
        let is_high: u32 = _mm256_movepi8_mask(chunk);
        let not_other = is_letter | is_digit | is_space | is_ctrl_ws | is_high;
        if not_other != 0 {
            return i + not_other.trailing_zeros() as usize;
        }
        i += 32;
    }

    while i < len {
        let b = *ptr.add(i);
        if b >= 0x80
            || (b | 0x20).wrapping_sub(b'a') < 26
            || b.wrapping_sub(b'0') < 10
            || b == b' '
            || b.wrapping_sub(9) < 5
        {
            break;
        }
        i += 1;
    }
    i
}

// ==========================================================================
// Full run scanners: ASCII SIMD + Unicode extension
// ==========================================================================

#[inline]
unsafe fn scan_full_letter_run(bytes: &[u8], start: usize) -> usize {
    let mut i = scan_ascii_letters(bytes, start);
    while i < bytes.len() && *bytes.get_unchecked(i) >= 0x80 {
        let c = decode_char_unchecked(bytes, i);
        if unicode::is_letter(c) {
            i += c.len_utf8();
            i = scan_ascii_letters(bytes, i);
        } else {
            break;
        }
    }
    i
}

#[inline]
unsafe fn scan_full_digit_run(bytes: &[u8], start: usize) -> usize {
    let mut i = scan_ascii_digits(bytes, start);
    while i < bytes.len() && *bytes.get_unchecked(i) >= 0x80 {
        let c = decode_char_unchecked(bytes, i);
        if unicode::is_number(c) {
            i += c.len_utf8();
            i = scan_ascii_digits(bytes, i);
        } else {
            break;
        }
    }
    i
}

#[inline]
unsafe fn scan_full_other_run(bytes: &[u8], start: usize) -> usize {
    let mut i = scan_ascii_other(bytes, start);
    while i < bytes.len() && *bytes.get_unchecked(i) >= 0x80 {
        let c = decode_char_unchecked(bytes, i);
        if unicode::is_other_complete(c) {
            i += c.len_utf8();
            i = scan_ascii_other(bytes, i);
        } else {
            break;
        }
    }
    i
}

#[cold]
unsafe fn scan_letter_after_unicode(bytes: &[u8], pos: usize, c: char) -> usize {
    scan_full_letter_run(bytes, pos + c.len_utf8())
}

#[cold]
unsafe fn scan_digit_after_unicode(bytes: &[u8], pos: usize, c: char) -> usize {
    scan_full_digit_run(bytes, pos + c.len_utf8())
}

#[cold]
unsafe fn scan_other_after_unicode(bytes: &[u8], pos: usize, c: char) -> usize {
    scan_full_other_run(bytes, pos + c.len_utf8())
}

// ==========================================================================
// Utility
// ==========================================================================

#[inline]
unsafe fn decode_char_unchecked(bytes: &[u8], pos: usize) -> char {
    std::str::from_utf8_unchecked(bytes.get_unchecked(pos..))
        .chars()
        .next()
        .unwrap_unchecked()
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::min;

    #[test]
    fn avx512_matches_state_machine() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let input = std::fs::read(data_dir.join("TinyStoriesV2-GPT4-valid.txt")).unwrap();
        let standard = crate::pretokenize::pretokenize_as_iter(&input);
        let avx512 = Avx512PretokenizerIter::new(&input);

        let mut count = 0usize;
        for (a, b) in standard.zip(avx512) {
            if a.0 != b.0 {
                let a_offset = a.0.as_ptr() as usize - input.as_ptr() as usize;
                let region_start = a_offset.saturating_sub(32);
                let region_end = min(input.len(), a_offset + 64);
                let context = String::from_utf8_lossy(&input[region_start..region_end]);
                panic!(
                    "Mismatch at token {count}: sm={:?} avx={:?}\nContext: {context:?}",
                    String::from_utf8_lossy(a.0),
                    String::from_utf8_lossy(b.0),
                );
            }
            count += 1;
        }
        eprintln!("All {count} tokens match.");
    }

    #[test]
    fn avx512_matches_state_machine_owt() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes =
            std::fs::read(data_dir.join("owt_train.txt")).expect("Could not read owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        let standard = crate::pretokenize::pretokenize_as_iter(input);
        let avx512 = Avx512PretokenizerIter::new(input);

        let mut count = 0usize;
        for (a, b) in standard.zip(avx512) {
            if a.0 != b.0 {
                let a_offset = a.0.as_ptr() as usize - input.as_ptr() as usize;
                let region_start = a_offset.saturating_sub(32);
                let region_end = min(input.len(), a_offset + 64);
                let context = String::from_utf8_lossy(&input[region_start..region_end]);
                panic!(
                    "Mismatch at token {count}: sm={:?} avx={:?}\nContext: {context:?}",
                    String::from_utf8_lossy(a.0),
                    String::from_utf8_lossy(b.0),
                );
            }
            count += 1;
        }
        eprintln!("All {count} OWT tokens match.");
    }

    #[test]
    fn avx512_count_matches() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes =
            std::fs::read(data_dir.join("owt_train.txt")).expect("Could not read owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        let sm_count = crate::pretokenize::pretokenize_as_iter(input).count();
        let avx_count = Avx512PretokenizerIter::new(input).count();
        assert_eq!(sm_count, avx_count, "Token counts differ");
    }
}
