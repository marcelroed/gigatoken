//! SIMD-accelerated pretokenizer.
//!
//! Implements the GPT-2 regex:
//!   '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!
//! Key optimizations:
//! - Inlined fast path for space+letter and letter patterns (~75% of pretokens)
//! - Table-based byte classification for remaining cases (no cascading if/else)
//! - Specialized NEON scanners for each character class
//! - Direct byte-level iteration (no &str conversion, no winnow overhead)

use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode;

// ---- Character class constants ----
// CO and CA are adjacent (0,1) so "other" = class < 2
const CO: u8 = 0; // Other
const CA: u8 = 1; // Apostrophe
const CL: u8 = 2; // Letter
const CN: u8 = 3; // Number
const CS: u8 = 4; // Space
const CW: u8 = 5; // Whitespace control
const CH: u8 = 7; // Non-ASCII

const fn build_class_table() -> [u8; 128] {
    let mut t = [CO; 128];
    let mut i = 0;
    while i < 26 {
        t[b'A' as usize + i] = CL;
        t[b'a' as usize + i] = CL;
        i += 1;
    }
    i = 0;
    while i < 10 {
        t[b'0' as usize + i] = CN;
        i += 1;
    }
    t[b' ' as usize] = CS;
    t[9] = CW;
    t[10] = CW;
    t[11] = CW;
    t[12] = CW;
    t[13] = CW;
    t[b'\'' as usize] = CA;
    t
}

static ASCII_CLASS: [u8; 128] = build_class_table();

#[inline(always)]
fn classify_byte(b: u8) -> u8 {
    if b < 0x80 {
        unsafe { *ASCII_CLASS.get_unchecked(b as usize) }
    } else {
        CH
    }
}

/// Check if byte is ASCII letter using branchless arithmetic.
#[inline(always)]
fn is_ascii_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
unsafe fn decode_char(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

// ---------------------------------------------------------------------------
// NEON module
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline(always)]
    pub unsafe fn first_nonzero_lane(mask: uint8x16_t) -> usize {
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

    #[inline(always)]
    unsafe fn is_letter_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let lower = vorrq_u8(chunk, vdupq_n_u8(0x20));
            let sub = vsubq_u8(lower, vdupq_n_u8(b'a'));
            vcgtq_u8(vdupq_n_u8(26), sub)
        }
    }

    #[inline(always)]
    unsafe fn is_digit_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let sub = vsubq_u8(chunk, vdupq_n_u8(b'0'));
            vcgtq_u8(vdupq_n_u8(10), sub)
        }
    }

    #[inline(always)]
    unsafe fn is_other_mask(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let is_letter = is_letter_mask(chunk);
            let is_digit = is_digit_mask(chunk);
            let ws_sub = vsubq_u8(chunk, vdupq_n_u8(9));
            let is_ws_ctrl = vcgeq_u8(vdupq_n_u8(4), ws_sub);
            let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
            let is_ws = vorrq_u8(is_ws_ctrl, is_space);
            let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));
            let any_exclude =
                vorrq_u8(vorrq_u8(is_letter, is_digit), vorrq_u8(is_ws, is_high));
            vmvnq_u8(any_exclude)
        }
    }

    /// Scan for end of ASCII letter run. Returns position of first non-letter.
    #[inline(always)]
    pub unsafe fn scan_letters(bytes: &[u8], start: usize) -> usize {
        unsafe {
            let mut i = start;
            while i + 16 <= bytes.len() {
                let chunk = vld1q_u8(bytes.as_ptr().add(i));
                let m = is_letter_mask(chunk);
                let nm = vmvnq_u8(m);
                let pos = first_nonzero_lane(nm);
                if pos < 16 {
                    return i + pos;
                }
                i += 16;
            }
            i
        }
    }

    #[inline(always)]
    pub unsafe fn scan_digits(bytes: &[u8], start: usize) -> usize {
        unsafe {
            let mut i = start;
            while i + 16 <= bytes.len() {
                let chunk = vld1q_u8(bytes.as_ptr().add(i));
                let m = is_digit_mask(chunk);
                let nm = vmvnq_u8(m);
                let pos = first_nonzero_lane(nm);
                if pos < 16 {
                    return i + pos;
                }
                i += 16;
            }
            i
        }
    }

    #[inline(always)]
    pub unsafe fn scan_other(bytes: &[u8], start: usize) -> usize {
        unsafe {
            let mut i = start;
            while i + 16 <= bytes.len() {
                let chunk = vld1q_u8(bytes.as_ptr().add(i));
                let m = is_other_mask(chunk);
                let nm = vmvnq_u8(m);
                let pos = first_nonzero_lane(nm);
                if pos < 16 {
                    return i + pos;
                }
                i += 16;
            }
            i
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod neon {
    #[inline(always)]
    pub unsafe fn scan_letters(_bytes: &[u8], start: usize) -> usize {
        start
    }
    #[inline(always)]
    pub unsafe fn scan_digits(_bytes: &[u8], start: usize) -> usize {
        start
    }
    #[inline(always)]
    pub unsafe fn scan_other(_bytes: &[u8], start: usize) -> usize {
        start
    }
}

// ---------------------------------------------------------------------------
// Run scanners with scalar tail + unicode continuation
// ---------------------------------------------------------------------------

/// SWAR (u64 arithmetic) letter detection: returns bitmask with high bit set
/// in each byte lane that is NOT an ASCII letter. Returns 0 if all 8 are letters.
#[inline(always)]
fn swar_non_letter_mask(word: u64) -> u64 {
    const HI: u64 = 0x8080_8080_8080_8080;
    // Lowercase everything, then check if in ['a','z']
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161) & HI;
    let le_z = (0x7A7A_7A7A_7A7A_7A7A | HI).wrapping_sub(lowered) & HI;
    // NOT letter = high bit not set in both checks
    !(ge_a & le_z) & HI
}

#[inline(always)]
fn scan_letter_run(bytes: &[u8], from: usize) -> usize {
    let mut i = from;

    // SWAR: check 8 bytes at a time with u64 arithmetic (fast for short words)
    while i + 8 <= bytes.len() {
        let word = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u64) };
        if word & 0x8080_8080_8080_8080 != 0 {
            return scan_letter_run_nonascii(bytes, i);
        }
        let not_letter = swar_non_letter_mask(word);
        if not_letter != 0 {
            return i + (not_letter.trailing_zeros() as usize) / 8;
        }
        i += 8;
    }

    // Padded SWAR for tail (0-7 remaining bytes)
    let remaining = bytes.len() - i;
    if remaining > 0 {
        let mut padded: u64 = 0; // zero bytes classify as non-letter
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(i),
                &mut padded as *mut u64 as *mut u8,
                remaining,
            );
        }
        if padded & 0x8080_8080_8080_8080 != 0 {
            return scan_letter_run_nonascii(bytes, i);
        }
        let not_letter = swar_non_letter_mask(padded);
        if not_letter != 0 {
            return i + ((not_letter.trailing_zeros() as usize) / 8).min(remaining);
        }
        return i + remaining;
    }
    i
}

/// Non-ASCII encountered during letter scan. Handle byte-by-byte with unicode.
#[cold]
#[inline(never)]
fn scan_letter_run_nonascii(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(i) };
        if is_ascii_letter(b) {
            i += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_char(bytes.get_unchecked(i..)) };
            if unicode::is_letter(c) {
                i += c.len_utf8();
                // Re-enter SWAR for subsequent ASCII
                return scan_letter_run(bytes, i);
            }
            break;
        } else {
            break;
        }
    }
    i
}

/// SWAR digit detection: high bit set in each non-digit byte lane.
#[inline(always)]
fn swar_non_digit_mask(word: u64) -> u64 {
    const HI: u64 = 0x8080_8080_8080_8080;
    let ge_0 = (word | HI).wrapping_sub(0x3030_3030_3030_3030) & HI;
    let le_9 = (0x3939_3939_3939_3939 | HI).wrapping_sub(word) & HI;
    !(ge_0 & le_9) & HI
}

#[inline(always)]
fn scan_digit_run(bytes: &[u8], from: usize) -> usize {
    let mut i = from;

    while i + 8 <= bytes.len() {
        let word = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u64) };
        if word & 0x8080_8080_8080_8080 != 0 {
            break;
        }
        let not_digit = swar_non_digit_mask(word);
        if not_digit != 0 {
            return i + (not_digit.trailing_zeros() as usize) / 8;
        }
        i += 8;
    }

    if i + 16 <= bytes.len() && unsafe { *bytes.get_unchecked(i) } < 0x80 {
        i = unsafe { neon::scan_digits(bytes, i) };
    }

    while i < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(i) };
        if b < 0x80 {
            if classify_byte(b) == CN {
                i += 1;
            } else {
                break;
            }
        } else {
            let c = unsafe { decode_char(bytes.get_unchecked(i..)) };
            if unicode::is_number(c) {
                i += c.len_utf8();
                continue;
            }
            break;
        }
    }
    i
}

/// SWAR "other" detection: high bit set in each byte that IS a letter, digit,
/// whitespace, or non-ASCII (i.e., NOT "other").
#[inline(always)]
fn swar_non_other_mask(word: u64) -> u64 {
    const HI: u64 = 0x8080_8080_8080_8080;
    // "other" = ASCII && NOT letter && NOT digit && NOT whitespace
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161) & HI;
    let le_z = (0x7A7A_7A7A_7A7A_7A7A | HI).wrapping_sub(lowered) & HI;
    let is_letter = ge_a & le_z & HI;

    let ge_0 = (word | HI).wrapping_sub(0x3030_3030_3030_3030) & HI;
    let le_9 = (0x3939_3939_3939_3939 | HI).wrapping_sub(word) & HI;
    let is_digit = ge_0 & le_9 & HI;

    // Whitespace: byte == 0x20 OR byte in [9,13]
    // For SWAR, check: (b - 9) <= 4 OR b == 0x20
    let ge_9 = (word | HI).wrapping_sub(0x0909_0909_0909_0909) & HI;
    let le_13 = (0x0D0D_0D0D_0D0D_0D0D | HI).wrapping_sub(word) & HI;
    let is_ws_ctrl = ge_9 & le_13 & HI;
    // Space: each byte XOR 0x20, check zero
    // For each byte: if byte == 0x20, (byte ^ 0x20) == 0
    // Detect zero bytes using the standard SWAR trick
    let xor_space = word ^ 0x2020_2020_2020_2020;
    let has_zero = (xor_space.wrapping_sub(0x0101_0101_0101_0101)) & !xor_space & HI;
    let is_space = has_zero;
    let is_ws = is_ws_ctrl | is_space;

    // "not other" = is_letter | is_digit | is_ws
    is_letter | is_digit | is_ws
}

#[inline(always)]
fn scan_other_run(bytes: &[u8], from: usize) -> usize {
    let mut i = from;

    while i + 8 <= bytes.len() {
        let word = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(i) as *const u64) };
        if word & 0x8080_8080_8080_8080 != 0 {
            break;
        }
        let not_other = swar_non_other_mask(word);
        if not_other != 0 {
            return i + (not_other.trailing_zeros() as usize) / 8;
        }
        i += 8;
    }

    if i + 16 <= bytes.len() && i < bytes.len() && unsafe { *bytes.get_unchecked(i) } < 0x80 {
        i = unsafe { neon::scan_other(bytes, i) };
    }

    while i < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(i) };
        if b < 0x80 {
            let cl = classify_byte(b);
            if cl < 2 {
                i += 1;
            } else {
                break;
            }
        } else {
            let c = unsafe { decode_char(bytes.get_unchecked(i..)) };
            if unicode::is_other_complete(c) {
                i += c.len_utf8();
                continue;
            }
            break;
        }
    }
    i
}

// ---------------------------------------------------------------------------
// Tight counting loop (avoids Iterator/Option/Pretoken overhead)
// ---------------------------------------------------------------------------

/// Count pretokens without constructing them. Same logic as next() but no
/// Option/Pretoken wrapping, yielding a tighter loop.
fn count_pretokens(bytes: &[u8], mut pos: usize) -> usize {
    let mut count = 0usize;
    while pos < bytes.len() {
        let first = unsafe { *bytes.get_unchecked(pos) };

        // Fast path: space + letter
        if first == b' ' {
            if pos + 1 < bytes.len() {
                let second = unsafe { *bytes.get_unchecked(pos + 1) };
                if is_ascii_letter(second) {
                    pos = scan_letter_run(bytes, pos + 2);
                    count += 1;
                    continue;
                }
                let class = classify_byte(second);
                pos = match class {
                    CN => scan_digit_run(bytes, pos + 2),
                    c if c < 2 => scan_other_run(bytes, pos + 2),
                    CS | CW => {
                        count += count_ws_pretokens(bytes, &mut pos);
                        continue;
                    }
                    _ => {
                        // CH — non-ASCII after space
                        let c = unsafe { decode_char(bytes.get_unchecked(pos + 1..)) };
                        let clen = c.len_utf8();
                        if unicode::is_letter(c) {
                            scan_letter_run(bytes, pos + 1 + clen)
                        } else if unicode::is_number(c) {
                            scan_digit_run(bytes, pos + 1 + clen)
                        } else if unicode::is_whitespace(c) {
                            count += count_ws_pretokens(bytes, &mut pos);
                            continue;
                        } else {
                            scan_other_run(bytes, pos + 1 + clen)
                        }
                    }
                };
                count += 1;
                continue;
            }
            pos += 1;
            count += 1;
            continue;
        }

        // Fast path: letter
        if is_ascii_letter(first) {
            pos = scan_letter_run(bytes, pos + 1);
            count += 1;
            continue;
        }

        // General dispatch
        let class = classify_byte(first);
        match class {
            CN => {
                pos = scan_digit_run(bytes, pos + 1);
            }
            CO => {
                pos = scan_other_run(bytes, pos + 1);
            }
            CW => {
                count += count_ws_pretokens(bytes, &mut pos);
                continue;
            }
            CA => {
                pos = match bytes.get(pos + 1) {
                    Some(b's' | b'd' | b'm' | b't') => pos + 2,
                    Some(b'l') if bytes.get(pos + 2) == Some(&b'l') => pos + 3,
                    Some(b'v') if bytes.get(pos + 2) == Some(&b'e') => pos + 3,
                    Some(b'r') if bytes.get(pos + 2) == Some(&b'e') => pos + 3,
                    _ => scan_other_run(bytes, pos + 1),
                };
            }
            _ => {
                // CH — non-ASCII
                let c = unsafe { decode_char(bytes.get_unchecked(pos..)) };
                let clen = c.len_utf8();
                if unicode::is_letter(c) {
                    pos = scan_letter_run(bytes, pos + clen);
                } else if unicode::is_number(c) {
                    pos = scan_digit_run(bytes, pos + clen);
                } else if unicode::is_whitespace(c) {
                    count += count_ws_pretokens(bytes, &mut pos);
                    continue;
                } else {
                    pos = scan_other_run(bytes, pos + clen);
                }
            }
        }
        count += 1;
    }
    count
}

/// Count the pretokens emitted from a whitespace run starting at *pos,
/// advancing *pos past all the whitespace.
#[inline]
fn count_ws_pretokens(bytes: &[u8], pos: &mut usize) -> usize {
    let start = *pos;
    let mut i = start;

    // Scan all consecutive whitespace
    while i < bytes.len() {
        let b = unsafe { *bytes.get_unchecked(i) };
        if b < 0x80 {
            let cl = classify_byte(b);
            if cl == CS || cl == CW {
                i += 1;
            } else {
                break;
            }
        } else {
            let c = unsafe { decode_char(bytes.get_unchecked(i..)) };
            if unicode::is_whitespace(c) {
                i += c.len_utf8();
            } else {
                break;
            }
        }
    }

    if i >= bytes.len() {
        // Whitespace at end of input: each char is a separate pretoken
        let mut count = 0;
        let mut j = start;
        while j < i {
            let b = unsafe { *bytes.get_unchecked(j) };
            if b >= 0x80 {
                j += unsafe { decode_char(bytes.get_unchecked(j..)) }.len_utf8();
            } else {
                j += 1;
            }
            count += 1;
        }
        *pos = i;
        return count;
    }

    // Find start of last ws char
    let mut last_ws_start = i - 1;
    while last_ws_start > start && unsafe { *bytes.get_unchecked(last_ws_start) } & 0xC0 == 0x80
    {
        last_ws_start -= 1;
    }

    if last_ws_start == start {
        // Single ws char
        let b = unsafe { *bytes.get_unchecked(start) };
        *pos = if b >= 0x80 {
            start + unsafe { decode_char(bytes.get_unchecked(start..)) }.len_utf8()
        } else {
            start + 1
        };
        return 1;
    }

    // 2+ ws chars: "all but last" is 1 pretoken, last ws char re-enters dispatch
    *pos = last_ws_start;
    1
}

// ---------------------------------------------------------------------------
// SimdPretokIter
// ---------------------------------------------------------------------------

pub struct SimdPretokIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SimdPretokIter<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> Iterator for SimdPretokIter<'a> {
    type Item = Pretoken<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.bytes;
        let start = self.pos;
        if start >= bytes.len() {
            return None;
        }

        let first = unsafe { *bytes.get_unchecked(start) };

        // ---- FAST PATH: space + ASCII letter (most common pattern) ----
        if first == b' ' {
            if start + 1 < bytes.len() {
                let second = unsafe { *bytes.get_unchecked(start + 1) };
                if is_ascii_letter(second) {
                    let end = scan_letter_run(bytes, start + 2);
                    self.pos = end;
                    return Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }));
                }
                // Space + non-letter: use table dispatch on second byte
                return self.handle_space_nonletter(start, second);
            }
            // Space at end of input
            self.pos = start + 1;
            return Some(Pretoken(unsafe {
                bytes.get_unchecked(start..start + 1)
            }));
        }

        // ---- FAST PATH: ASCII letter (standalone word) ----
        if is_ascii_letter(first) {
            let end = scan_letter_run(bytes, start + 1);
            self.pos = end;
            return Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }));
        }

        // ---- REMAINING CASES: digit, other, whitespace, apostrophe, non-ASCII ----
        let class = classify_byte(first);
        match class {
            CN => {
                let end = scan_digit_run(bytes, start + 1);
                self.pos = end;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }))
            }
            CO => {
                let end = scan_other_run(bytes, start + 1);
                self.pos = end;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }))
            }
            CW => self.handle_whitespace(start),
            CA => self.handle_apostrophe(start),
            _ => self.handle_non_ascii(start), // CH
        }
    }
}

impl<'a> SimdPretokIter<'a> {
    /// Handle space followed by non-letter byte.
    #[inline]
    fn handle_space_nonletter(&mut self, start: usize, second: u8) -> Option<Pretoken<'a>> {
        let bytes = self.bytes;
        let class = classify_byte(second);
        let end = match class {
            CN => scan_digit_run(bytes, start + 2),
            CO | CA => scan_other_run(bytes, start + 2),
            CS | CW => return self.handle_whitespace(start),
            _ => {
                // CH — non-ASCII after space
                let c = unsafe { decode_char(bytes.get_unchecked(start + 1..)) };
                let clen = c.len_utf8();
                if unicode::is_letter(c) {
                    scan_letter_run(bytes, start + 1 + clen)
                } else if unicode::is_number(c) {
                    scan_digit_run(bytes, start + 1 + clen)
                } else if unicode::is_whitespace(c) {
                    return self.handle_whitespace(start);
                } else {
                    scan_other_run(bytes, start + 1 + clen)
                }
            }
        };
        self.pos = end;
        Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }))
    }

    #[cold]
    #[inline(never)]
    fn handle_whitespace(&mut self, start: usize) -> Option<Pretoken<'a>> {
        let bytes = self.bytes;
        let mut i = start;

        while i < bytes.len() {
            let b = unsafe { *bytes.get_unchecked(i) };
            if b < 0x80 {
                let cl = classify_byte(b);
                if cl == CS || cl == CW {
                    i += 1;
                } else {
                    break;
                }
            } else {
                let c = unsafe { decode_char(bytes.get_unchecked(i..)) };
                if unicode::is_whitespace(c) {
                    i += c.len_utf8();
                } else {
                    break;
                }
            }
        }

        if i >= bytes.len() {
            // End of input: emit single ws char
            let b = unsafe { *bytes.get_unchecked(start) };
            self.pos = if b >= 0x80 {
                start + unsafe { decode_char(bytes.get_unchecked(start..)) }.len_utf8()
            } else {
                start + 1
            };
            return Some(Pretoken(unsafe {
                bytes.get_unchecked(start..self.pos)
            }));
        }

        // Find start of last ws char
        let mut last_ws_start = i - 1;
        while last_ws_start > start
            && unsafe { *bytes.get_unchecked(last_ws_start) } & 0xC0 == 0x80
        {
            last_ws_start -= 1;
        }

        if last_ws_start == start {
            // Only 1 ws char
            let b = unsafe { *bytes.get_unchecked(start) };
            self.pos = if b >= 0x80 {
                start + unsafe { decode_char(bytes.get_unchecked(start..)) }.len_utf8()
            } else {
                start + 1
            };
            return Some(Pretoken(unsafe {
                bytes.get_unchecked(start..self.pos)
            }));
        }

        // 2+ ws chars: emit all but last
        self.pos = last_ws_start;
        Some(Pretoken(unsafe {
            bytes.get_unchecked(start..last_ws_start)
        }))
    }

    #[cold]
    #[inline(never)]
    fn handle_apostrophe(&mut self, start: usize) -> Option<Pretoken<'a>> {
        let bytes = self.bytes;
        match bytes.get(start + 1) {
            Some(b's' | b'd' | b'm' | b't') => {
                self.pos = start + 2;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..start + 2) }))
            }
            Some(b'l') if bytes.get(start + 2) == Some(&b'l') => {
                self.pos = start + 3;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..start + 3) }))
            }
            Some(b'v') if bytes.get(start + 2) == Some(&b'e') => {
                self.pos = start + 3;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..start + 3) }))
            }
            Some(b'r') if bytes.get(start + 2) == Some(&b'e') => {
                self.pos = start + 3;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..start + 3) }))
            }
            _ => {
                let end = scan_other_run(bytes, start + 1);
                self.pos = end;
                Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }))
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn handle_non_ascii(&mut self, start: usize) -> Option<Pretoken<'a>> {
        let bytes = self.bytes;
        let c = unsafe { decode_char(bytes.get_unchecked(start..)) };
        let clen = c.len_utf8();

        let end = if unicode::is_letter(c) {
            scan_letter_run(bytes, start + clen)
        } else if unicode::is_number(c) {
            scan_digit_run(bytes, start + clen)
        } else if unicode::is_whitespace(c) {
            return self.handle_whitespace(start);
        } else {
            scan_other_run(bytes, start + clen)
        };

        self.pos = end;
        Some(Pretoken(unsafe { bytes.get_unchecked(start..end) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use std::cmp::min;

    #[test]
    fn count_matches_next() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let input =
            std::fs::read_to_string(data_dir.join("TinyStoriesV2-GPT4-valid.txt")).unwrap();
        let input_bytes = input.as_bytes();

        let next_count = SimdPretokIter::new(input_bytes).fold(0usize, |c, _| c + 1);
        let count_count = SimdPretokIter::new(input_bytes).count();
        assert_eq!(
            next_count, count_count,
            "count() = {count_count} but next()-based count = {next_count}"
        );
    }

    #[test]
    fn simd_matches_fast() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let input =
            std::fs::read_to_string(data_dir.join("TinyStoriesV2-GPT4-valid.txt")).unwrap();
        let input_bytes = input.as_bytes();

        let standard = crate::pretokenize::pretokenize_as_iter(input_bytes);
        let simd = SimdPretokIter::new(input_bytes);

        for eorb in standard.zip_longest(simd) {
            match eorb {
                itertools::EitherOrBoth::Both(a, b) => {
                    if a.0 != b.0 {
                        let a_offset =
                            a.0.as_ptr() as usize - input_bytes.as_ptr() as usize;
                        let region = &input_bytes[a_offset.saturating_sub(32)
                            ..min(input_bytes.len(), a_offset + 64)];
                        panic!(
                            "Mismatch at byte {a_offset}:\n  standard: {:?}\n  simd:     {:?}\n  context:  {:?}",
                            String::from_utf8_lossy(a.0),
                            String::from_utf8_lossy(b.0),
                            String::from_utf8_lossy(region),
                        );
                    }
                }
                itertools::EitherOrBoth::Left(a) => {
                    panic!("Standard extra: {:?}", String::from_utf8_lossy(a.0));
                }
                itertools::EitherOrBoth::Right(b) => {
                    panic!("SIMD extra: {:?}", String::from_utf8_lossy(b.0));
                }
            }
        }
    }
}
