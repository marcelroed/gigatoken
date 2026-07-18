//! Implement the regex
//! '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
//! using winnow parser combinators.
use crate::pretokenize::{Pretoken, unicode};
use std::cmp::min;

use eyre::Context;
use itertools::Itertools;
use rayon::prelude::*;
use winnow::Parser;
use winnow::combinator::{alt, iterator, trace};
use winnow::prelude::*;

// ---------------------------------------------------------------------------
// NEON scan utilities — find the end of a run of matching ASCII bytes,
// processing 16 bytes at a time with precise exit via trailing_zeros.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Return the byte index (relative to the chunk start) of the first lane
    /// where `mask` is non-zero, or 16 if all lanes are zero.
    #[inline(always)]
    unsafe fn first_nonzero_lane(mask: uint8x16_t) -> usize {
        unsafe {
            let lo = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(mask));
            if lo != 0 {
                return (lo.trailing_zeros() as usize) / 8;
            }
            let hi = vgetq_lane_u64::<1>(vreinterpretq_u64_u8(mask));
            if hi != 0 {
                return 8 + (hi.trailing_zeros() as usize) / 8;
            }
            16
        }
    }

    /// Scan ASCII letters: `(b | 0x20) - 'a' < 26`.
    #[inline]
    pub unsafe fn scan_ascii_letters(bytes: &[u8], start: usize) -> usize {
        unsafe { scan_generic(bytes, start, is_ascii_letter_mask) }
    }

    /// Scan ASCII digits: `b - '0' < 10`.
    #[inline]
    pub unsafe fn scan_ascii_digits(bytes: &[u8], start: usize) -> usize {
        unsafe { scan_generic(bytes, start, is_ascii_digit_mask) }
    }

    /// Scan ASCII "other" bytes: not letter, not digit, not whitespace, not >= 0x80.
    #[inline]
    pub unsafe fn scan_ascii_other(bytes: &[u8], start: usize) -> usize {
        unsafe { scan_generic(bytes, start, is_ascii_other_mask) }
    }

    /// Generic NEON scanner. `classify` returns a mask with 0xFF for matching bytes.
    /// Returns the index of the first non-matching byte.
    #[inline(always)]
    unsafe fn scan_generic(
        bytes: &[u8],
        start: usize,
        classify: unsafe fn(uint8x16_t) -> uint8x16_t,
    ) -> usize {
        let mut i = start;
        unsafe {
            while i + 16 <= bytes.len() {
                let chunk = vld1q_u8(bytes.as_ptr().add(i));
                let is_match = classify(chunk);
                let not_match = vmvnq_u8(is_match);
                let pos = first_nonzero_lane(not_match);
                if pos < 16 {
                    return i + pos;
                }
                i += 16;
            }
        }
        i // caller handles the scalar tail
    }

    #[inline(always)]
    unsafe fn is_ascii_letter_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let lower = vorrq_u8(chunk, vdupq_n_u8(0x20));
            let sub = vsubq_u8(lower, vdupq_n_u8(b'a'));
            vcgtq_u8(vdupq_n_u8(26), sub) // 0xFF if letter
        }
    }

    #[inline(always)]
    unsafe fn is_ascii_digit_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let sub = vsubq_u8(chunk, vdupq_n_u8(b'0'));
            vcgtq_u8(vdupq_n_u8(10), sub)
        }
    }

    #[inline(always)]
    unsafe fn is_ascii_other_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            // "other" = NOT letter AND NOT digit AND NOT whitespace AND NOT non-ASCII
            let is_letter = is_ascii_letter_mask(chunk);
            let is_digit = is_ascii_digit_mask(chunk);
            // Whitespace: 9-13 or 32
            let ws_sub = vsubq_u8(chunk, vdupq_n_u8(9));
            let is_ws_ctrl = vcgeq_u8(vdupq_n_u8(4), ws_sub); // 9..=13
            let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
            let is_ws = vorrq_u8(is_ws_ctrl, is_space);
            // Non-ASCII (>= 0x80)
            let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));
            // Other = none of the above
            let any_exclude = vorrq_u8(vorrq_u8(is_letter, is_digit), vorrq_u8(is_ws, is_high));
            vmvnq_u8(any_exclude) // 0xFF where "other"
        }
    }
}

// Portable fallbacks
#[cfg(not(target_arch = "aarch64"))]
mod neon {
    #[inline]
    pub unsafe fn scan_ascii_letters(bytes: &[u8], start: usize) -> usize {
        start // fall through to scalar immediately
    }
    #[inline]
    pub unsafe fn scan_ascii_digits(bytes: &[u8], start: usize) -> usize {
        start
    }
    #[inline]
    pub unsafe fn scan_ascii_other(bytes: &[u8], start: usize) -> usize {
        start
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn backtrack() -> winnow::error::ErrMode<winnow::error::ContextError> {
    winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new())
}

/// Decode one char from a non-ASCII position in a byte slice.
/// Caller guarantees `bytes[0] >= 0x80` and `bytes` is valid UTF-8.
#[inline(always)]
unsafe fn decode_non_ascii(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

// ---------------------------------------------------------------------------
// Individual parsers — each returns () and advances *input on success.
// ---------------------------------------------------------------------------

fn contraction(input: &mut &str) -> ModalResult<()> {
    let bytes = input.as_bytes();
    if bytes.first() != Some(&b'\'') {
        return Err(backtrack());
    }
    let advance = match bytes.get(1) {
        Some(b's' | b'd' | b'm' | b't') => 2,
        Some(b'l') if bytes.get(2) == Some(&b'l') => 3,
        Some(b'v') if bytes.get(2) == Some(&b'e') => 3,
        Some(b'r') if bytes.get(2) == Some(&b'e') => 3,
        _ => return Err(backtrack()),
    };
    *input = &input[advance..];
    Ok(())
}

fn letter_run(input: &mut &str) -> ModalResult<()> {
    let before = *input;
    let bytes = before.as_bytes();
    let mut i = if bytes.first() == Some(&b' ') { 1 } else { 0 };
    let letter_start = i;

    // NEON fast path (returns precise position of first non-ASCII-letter)
    i = unsafe { neon::scan_ascii_letters(bytes, i) };
    // Scalar tail for remaining < 16 bytes
    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        i += 1;
    }

    // Handle non-ASCII unicode letters, resuming fast scan after each one
    while i < bytes.len() && bytes[i] >= 0x80 {
        let c = unsafe { decode_non_ascii(&bytes[i..]) };
        if unicode::is_letter(c) {
            i += c.len_utf8();
            i = unsafe { neon::scan_ascii_letters(bytes, i) };
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else {
            break;
        }
    }

    if i == letter_start {
        return Err(backtrack());
    }
    *input = &before[i..];
    Ok(())
}

fn number_run(input: &mut &str) -> ModalResult<()> {
    let before = *input;
    let bytes = before.as_bytes();
    let mut i = if bytes.first() == Some(&b' ') { 1 } else { 0 };
    let start = i;

    i = unsafe { neon::scan_ascii_digits(bytes, i) };
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }

    while i < bytes.len() && bytes[i] >= 0x80 {
        let c = unsafe { decode_non_ascii(&bytes[i..]) };
        if unicode::is_number(c) {
            i += c.len_utf8();
            i = unsafe { neon::scan_ascii_digits(bytes, i) };
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            break;
        }
    }

    if i == start {
        return Err(backtrack());
    }
    *input = &before[i..];
    Ok(())
}

fn other_run(input: &mut &str) -> ModalResult<()> {
    let before = *input;
    let bytes = before.as_bytes();
    let mut i = if bytes.first() == Some(&b' ') { 1 } else { 0 };
    let start = i;

    i = unsafe { neon::scan_ascii_other(bytes, i) };
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            if !b.is_ascii_alphanumeric() && !b.is_ascii_whitespace() {
                i += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Handle non-ASCII "other" chars (unicode symbols, etc.)
    while i < bytes.len() && bytes[i] >= 0x80 {
        let c = unsafe { decode_non_ascii(&bytes[i..]) };
        if unicode::is_other_complete(c) {
            i += c.len_utf8();
            i = unsafe { neon::scan_ascii_other(bytes, i) };
            while i < bytes.len() {
                let b = bytes[i];
                if b < 0x80 && !b.is_ascii_alphanumeric() && !b.is_ascii_whitespace() {
                    i += 1;
                } else {
                    break;
                }
            }
        } else {
            break;
        }
    }

    if i == start {
        return Err(backtrack());
    }
    *input = &before[i..];
    Ok(())
}

fn whitespace_run(input: &mut &str) -> ModalResult<()> {
    let before = *input;
    let bytes = before.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[i..]) };
            if unicode::is_whitespace(c) {
                i += c.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if i == 0 {
        return Err(backtrack());
    }
    // At end of input the `(?!\S)` lookahead succeeds, so `\s+(?!\S)` consumes
    // the entire trailing whitespace run as one pretoken.
    if i >= bytes.len() {
        *input = &before[i..];
        return Ok(());
    }
    // Find start of last whitespace char
    let mut last_start = i - 1;
    while last_start > 0 && bytes[last_start] & 0xC0 == 0x80 {
        last_start -= 1;
    }
    // Need at least 2 ws chars (last_start > 0 means there's content before the last char)
    if last_start == 0 {
        return Err(backtrack());
    }
    // Leave last ws char for single_whitespace
    *input = &before[last_start..];
    Ok(())
}

fn single_whitespace(input: &mut &str) -> ModalResult<()> {
    let bytes = input.as_bytes();
    let b = *bytes.first().ok_or_else(backtrack)?;
    if b.is_ascii_whitespace() {
        *input = &input[1..];
        Ok(())
    } else if b >= 0x80 {
        let c = unsafe { decode_non_ascii(bytes) };
        if unicode::is_whitespace(c) {
            *input = &input[c.len_utf8()..];
            Ok(())
        } else {
            Err(backtrack())
        }
    } else {
        Err(backtrack())
    }
}

// ---------------------------------------------------------------------------
// Top-level pretoken parser — direct dispatch on first byte instead of alt().
// ---------------------------------------------------------------------------

fn pretoken<'a>(input: &mut &'a str) -> ModalResult<Pretoken<'a>> {
    let before = *input;
    let bytes = before.as_bytes();
    let first = *bytes.first().ok_or_else(backtrack)?;

    if first < 0x80 {
        if first.is_ascii_alphabetic() {
            letter_run(input)?;
        } else if first == b' ' {
            // Peek at second byte to avoid cascading alt() failures
            match bytes.get(1) {
                Some(&b) if b.is_ascii_alphabetic() => letter_run(input)?,
                Some(&b) if b.is_ascii_digit() => number_run(input)?,
                Some(&b) if !b.is_ascii_whitespace() && b < 0x80 => {
                    other_run(input)?
                }
                Some(&b) if b >= 0x80 => {
                    let c = unsafe { decode_non_ascii(&bytes[1..]) };
                    if unicode::is_letter(c) {
                        letter_run(input)?;
                    } else if unicode::is_number(c) {
                        number_run(input)?;
                    } else if unicode::is_whitespace(c) {
                        ws_or_single(input)?;
                    } else {
                        other_run(input)?;
                    }
                }
                _ => ws_or_single(input)?,
            }
        } else if first.is_ascii_digit() {
            number_run(input)?;
        } else if first == b'\'' {
            contraction(input).or_else(|_| other_run(input))?;
        } else if first.is_ascii_whitespace() {
            ws_or_single(input)?;
        } else {
            other_run(input)?;
        }
    } else {
        let c = unsafe { decode_non_ascii(bytes) };
        if unicode::is_letter(c) {
            letter_run(input)?;
        } else if unicode::is_number(c) {
            number_run(input)?;
        } else if unicode::is_whitespace(c) {
            ws_or_single(input)?;
        } else {
            other_run(input)?;
        }
    }

    let consumed = before.len() - input.len();
    Ok(Pretoken(&before.as_bytes()[..consumed]))
}

/// Try whitespace_run first (2+ ws chars with non-ws following), fall back to single.
#[inline]
fn ws_or_single(input: &mut &str) -> ModalResult<()> {
    whitespace_run(input).or_else(|_| single_whitespace(input))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct PretokenIterator<'a> {
    input: &'a [u8],
}

/// Parse the next pretoken from `input`, advancing the slice past the consumed bytes.
/// Returns `None` when `input` is empty.
pub fn pretoken_next<'a>(input: &mut &'a str) -> Option<Pretoken<'a>> {
    if input.is_empty() {
        return None;
    }
    pretoken(input).ok()
}

// The `impl FnMut` parser type is unnameable, so this signature can't be
// factored into a type alias.
#[allow(clippy::type_complexity)]
pub fn pretokens_iterator<'a>(
    input: &'a mut &'a str,
) -> winnow::combinator::ParserIterator<
    'a,
    impl FnMut(
        &mut &'a str,
    )
        -> std::result::Result<Pretoken<'a>, winnow::error::ErrMode<winnow::error::ContextError>>
    + 'a,
    &'a str,
    Pretoken<'a>,
    winnow::error::ErrMode<winnow::error::ContextError>,
> {
    iterator(input, pretoken)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trailing-whitespace and mixed-whitespace edge cases, checked against the
    /// reference GPT-2 regex. Catches the `\s+(?!\S)` end-of-input case: the
    /// lookahead succeeds at EOF, so a trailing run like "\n\n\n" must be a
    /// single pretoken, not one per character.
    #[test]
    fn whitespace_edge_cases_match_regex() {
        let re = fancy_regex::Regex::new(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .unwrap();
        let cases = [
            "\n\n\n",
            "\n\n\nhello",
            "a\n\n\nb",
            "hello  \n  world",
            "  \t\n ",
            "x  ",
            "x \n",
            " \n x",
            "\t\t\tword",
            "end\n\n",
            "end\n",
            " ",
            "\u{a0}\u{a0}",
            "word\u{2028}\u{2028}",
        ];
        for case in cases {
            let expected: Vec<&str> = re.find_iter(case).map(|m| m.unwrap().as_str()).collect();

            let mut input = case;
            let mut combinator: Vec<String> = Vec::new();
            while let Some(p) = pretoken_next(&mut input) {
                combinator.push(String::from_utf8(p.0.to_vec()).unwrap());
            }
            assert_eq!(combinator, expected, "combinator mismatch for {case:?}");

            let mut fast: Vec<String> = Vec::new();
            let mut it =
                crate::pretokenize::fast::FastR50kPretokenizer::new(case.as_bytes());
            for p in it {
                fast.push(String::from_utf8(p.as_ref().to_vec()).unwrap());
            }
            assert_eq!(fast, expected, "fast mismatch for {case:?}");

            let state_machine: Vec<String> =
                crate::pretokenize::PretokenizerIter::new(case.as_bytes())
                    .map(|p| String::from_utf8(p.0.to_vec()).unwrap())
                    .collect();
            assert_eq!(
                state_machine, expected,
                "state machine mismatch for {case:?}"
            );
        }
    }

    #[test]
    fn combinator_compare() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let input = std::fs::read_to_string(data_dir.join("TinyStoriesV2-GPT4-valid.txt")).unwrap();
        let input_bytes = input.as_bytes();
        let standard_iterator = crate::pretokenize::pretokenize_as_iter(input_bytes);
        let mut input_slice = input.as_str();
        let mut combinator_iterator = pretokens_iterator(&mut input_slice);
        for eorb in standard_iterator.zip_longest(&mut combinator_iterator) {
            match eorb {
                itertools::EitherOrBoth::Both(a, b) => {
                    if a.0 != b.0 {
                        eprintln!(
                            "Mismatch: {:?} != {:?}",
                            String::from_utf8_lossy(a.0),
                            String::from_utf8_lossy(b.0)
                        );

                        // Find text before and after the mismatch by comparing pointers from a.0 and input_bytes
                        let a_start = a.0.as_ptr() as usize;
                        let b_start = b.0.as_ptr() as usize;
                        let input_start = input_bytes.as_ptr() as usize;
                        let a_offset = a_start - input_start;

                        let region = &input_bytes
                            [a_offset.saturating_sub(32)..min(input_bytes.len(), a_offset + 32)];
                        eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                        panic!("combinator pretoken differs from standard pretokenizer");
                    }
                }
                itertools::EitherOrBoth::Left(a) => {
                    eprintln!("Left only: {:?}", String::from_utf8_lossy(a.0));

                    // Find text before and after the mismatch by comparing pointers from a.0 and input_bytes
                    let a_start = a.0.as_ptr() as usize;
                    let input_start = input_bytes.as_ptr() as usize;
                    let a_offset = a_start - input_start;

                    let region = &input_bytes
                        [a_offset.saturating_sub(32)..min(input_bytes.len(), a_offset + 32)];
                    eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                    panic!("standard pretokenizer produced an extra pretoken");
                }
                itertools::EitherOrBoth::Right(b) => {
                    eprintln!("Right only: {:?}", String::from_utf8_lossy(b.0));

                    // Find text before and after the mismatch by comparing pointers from b and input_bytes
                    let b_start = b.as_ptr() as usize;
                    let input_start = input_bytes.as_ptr() as usize;
                    let b_offset = b_start - input_start;

                    let region = &input_bytes
                        [b_offset.saturating_sub(32)..min(input_bytes.len(), b_offset + 32)];
                    eprintln!("Context: {:?}", String::from_utf8_lossy(region));

                    panic!("combinator pretokenizer produced an extra pretoken");
                }
            }
        }
    }
}
