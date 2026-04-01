//! Implement the regex
//! '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
//! using winnow parser combinators.
//!
//! Scalar (non-SIMD) version — kept for reference. The NEON-accelerated version
//! is in pretoken_combinator.rs.
use crate::pretokenize::{Pretoken, unicode};
use std::cmp::min;

use eyre::Context;
use itertools::Itertools;
use rayon::prelude::*;
use winnow::Parser;
use winnow::combinator::{alt, iterator, trace};
use winnow::prelude::*;

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
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphabetic() {
            i += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[i..]) };
            if unicode::is_letter(c) {
                i += c.len_utf8();
            } else {
                break;
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
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_digit() {
            i += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[i..]) };
            if unicode::is_number(c) {
                i += c.len_utf8();
            } else {
                break;
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
    // Need 2+ ws bytes consumed, non-ws must follow (not EOF)
    if i == 0 || i >= bytes.len() {
        return Err(backtrack());
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

fn other_run(input: &mut &str) -> ModalResult<()> {
    let before = *input;
    let bytes = before.as_bytes();
    let mut i = if bytes.first() == Some(&b' ') { 1 } else { 0 };
    let start = i;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            if !b.is_ascii_alphanumeric() && !b.is_ascii_whitespace() {
                i += 1;
            } else {
                break;
            }
        } else {
            let c = unsafe { decode_non_ascii(&bytes[i..]) };
            if unicode::is_other_complete(c) {
                i += c.len_utf8();
            } else {
                break;
            }
        }
    }
    if i == start {
        return Err(backtrack());
    }
    *input = &before[i..];
    Ok(())
}

fn pretoken<'a>(input: &mut &'a str) -> ModalResult<Pretoken<'a>> {
    alt((
        trace("letter_run", letter_run),
        trace("contraction", contraction),
        trace("number_run", number_run),
        trace("other_run", other_run),
        trace("whitespace_run", whitespace_run),
        trace("single_whitespace", single_whitespace),
    ))
    .take()
    .map(|s: &str| Pretoken(s.as_bytes()))
    .parse_next(input)
}

pub fn pretokens_iterator_scalar<'a>(
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
