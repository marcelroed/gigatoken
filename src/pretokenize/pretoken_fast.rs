//! Fast scalar pretokenizer for the GPT-2 regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Uses LUT dispatch + SWAR (u64) for letter runs + arithmetic predicates.
//! The hot path (space + letters / bare letters) is fully inlined in count().

use crate::pretokenize::{Pretoken, unicode};

// -----------------------------------------------------------------------
// Byte classification LUT
// -----------------------------------------------------------------------

const LETTER: u8 = 0;
const DIGIT: u8 = 1;
const SPACE: u8 = 2;
const WHITESPACE: u8 = 3;
const APOSTROPHE: u8 = 4;
const OTHER: u8 = 5;
const NON_ASCII: u8 = 6;

static CLASS: [u8; 256] = {
    let mut lut = [NON_ASCII; 256];
    let mut i: usize = 0;
    while i < 128 {
        lut[i] = if (i >= 0x41 && i <= 0x5A) || (i >= 0x61 && i <= 0x7A) {
            LETTER
        } else if i >= 0x30 && i <= 0x39 {
            DIGIT
        } else if i == 0x20 {
            SPACE
        } else if i >= 9 && i <= 13 {
            WHITESPACE
        } else if i == 0x27 {
            APOSTROPHE
        } else {
            OTHER
        };
        i += 1;
    }
    lut
};

// -----------------------------------------------------------------------
// Branchless byte predicates
// -----------------------------------------------------------------------

#[inline(always)]
fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline(always)]
fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

#[inline(always)]
unsafe fn decode_non_ascii(bytes: &[u8]) -> char {
    unsafe {
        std::str::from_utf8_unchecked(bytes)
            .chars()
            .next()
            .unwrap_unchecked()
    }
}

// -----------------------------------------------------------------------
// SWAR
// -----------------------------------------------------------------------

const HI: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
fn swar64_letter_mask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lowered);
    ge_a & le_z & HI
}

/// SWAR letter scan: advances `pos` past ASCII letters.
/// Returns the updated pos. Handles unicode letters via callback to struct method.
#[inline(always)]
fn swar_scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    // SWAR: 8 bytes at a time
    while pos + 8 <= len {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HI != 0 {
            break;
        }
        let mask = swar64_letter_mask(word);
        if mask != HI {
            return pos + (!mask & HI).to_le().trailing_zeros() as usize / 8;
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

// -----------------------------------------------------------------------
// FastPretokenizer
// -----------------------------------------------------------------------

pub struct FastPretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        loop {
            self.pos = swar_scan_letters(bytes, self.pos);
            // Unicode letters
            if self.pos < len && unsafe { *bytes.get_unchecked(self.pos) } >= 0x80 {
                let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                if unicode::is_letter(c) {
                    self.pos += c.len_utf8();
                    continue;
                }
            }
            return;
        }
    }

    #[inline(always)]
    fn scan_digits(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        loop {
            while self.pos < len && is_digit(unsafe { *bytes.get_unchecked(self.pos) }) {
                self.pos += 1;
            }
            if self.pos < len && unsafe { *bytes.get_unchecked(self.pos) } >= 0x80 {
                let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                if unicode::is_number(c) {
                    self.pos += c.len_utf8();
                    continue;
                }
            }
            return;
        }
    }

    #[inline(always)]
    fn scan_other(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        loop {
            while self.pos < len {
                let b = unsafe { *bytes.get_unchecked(self.pos) };
                if b >= 0x80 {
                    break;
                }
                if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                    return;
                }
                self.pos += 1;
            }
            if self.pos < len {
                let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                if unicode::is_other_complete(c) {
                    self.pos += c.len_utf8();
                    continue;
                }
            }
            return;
        }
    }

    #[inline(always)]
    fn advance_whitespace(&mut self, start: usize) {
        let bytes = self.bytes;
        let len = bytes.len();
        while self.pos < len {
            let b = unsafe { *bytes.get_unchecked(self.pos) };
            if is_ascii_ws(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                if unicode::is_whitespace(c) {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        if self.pos < len {
            let ws_bytes = self.pos - start;
            if ws_bytes >= 2 {
                let mut last = self.pos - 1;
                while last > start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                    last -= 1;
                }
                if last > start {
                    self.pos = last;
                }
            }
        }
    }

    /// Advance past one token. self.pos must be < self.bytes.len().
    #[inline(always)]
    fn advance(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        let start = self.pos;
        let b0 = unsafe { *bytes.get_unchecked(start) };

        match CLASS[b0 as usize] {
            LETTER => {
                self.pos = start + 1;
                self.scan_letters();
            }
            SPACE => {
                if start + 1 < len {
                    let b1 = unsafe { *bytes.get_unchecked(start + 1) };
                    if is_letter(b1) {
                        self.pos = start + 2;
                        self.scan_letters();
                    } else if is_digit(b1) {
                        self.pos = start + 2;
                        self.scan_digits();
                    } else if b1 >= 0x80 {
                        self.pos = start + 1;
                        let c = unsafe { decode_non_ascii(&bytes[self.pos..]) };
                        self.pos += c.len_utf8();
                        if unicode::is_letter(c) {
                            self.scan_letters();
                        } else if unicode::is_number(c) {
                            self.scan_digits();
                        } else if unicode::is_whitespace(c) {
                            self.advance_whitespace(start);
                        } else {
                            self.scan_other();
                        }
                    } else if is_ascii_ws(b1) {
                        self.pos = start + 1;
                        self.advance_whitespace(start);
                    } else {
                        self.pos = start + 2;
                        self.scan_other();
                    }
                } else {
                    self.pos = start + 1;
                }
            }
            DIGIT => {
                self.pos = start + 1;
                self.scan_digits();
            }
            APOSTROPHE => match bytes.get(start + 1) {
                Some(b's' | b'd' | b'm' | b't') => {
                    self.pos = start + 2;
                }
                Some(b'l') if bytes.get(start + 2) == Some(&b'l') => {
                    self.pos = start + 3;
                }
                Some(b'v') if bytes.get(start + 2) == Some(&b'e') => {
                    self.pos = start + 3;
                }
                Some(b'r') if bytes.get(start + 2) == Some(&b'e') => {
                    self.pos = start + 3;
                }
                _ => {
                    self.pos = start + 1;
                    self.scan_other();
                }
            },
            WHITESPACE => {
                self.pos = start + 1;
                self.advance_whitespace(start);
            }
            OTHER => {
                self.pos = start + 1;
                self.scan_other();
            }
            NON_ASCII => {
                let c = unsafe { decode_non_ascii(&bytes[start..]) };
                self.pos = start + c.len_utf8();
                if unicode::is_letter(c) {
                    self.scan_letters();
                } else if unicode::is_number(c) {
                    self.scan_digits();
                } else if unicode::is_whitespace(c) {
                    self.advance_whitespace(start);
                } else {
                    self.scan_other();
                }
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

impl<'a> Iterator for FastPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        self.advance();
        Some(Pretoken(&self.bytes[start..self.pos]))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        count_dual_cursor(self.bytes, self.pos)
    }
}

// -----------------------------------------------------------------------
// Free-function advance: (bytes, pos) → new_pos
// -----------------------------------------------------------------------

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    match CLASS[b0 as usize] {
        LETTER => {
            scan_letters_from(bytes, pos + 1)
        }
        SPACE => {
            if pos + 1 < len {
                let b1 = unsafe { *bytes.get_unchecked(pos + 1) };
                if is_letter(b1) {
                    scan_letters_from(bytes, pos + 2)
                } else if is_digit(b1) {
                    scan_digits_from(bytes, pos + 2)
                } else if b1 >= 0x80 {
                    let c = unsafe { decode_non_ascii(&bytes[pos + 1..]) };
                    let p = pos + 1 + c.len_utf8();
                    if unicode::is_letter(c) {
                        scan_letters_from(bytes, p)
                    } else if unicode::is_number(c) {
                        scan_digits_from(bytes, p)
                    } else if unicode::is_whitespace(c) {
                        advance_ws(bytes, p, pos)
                    } else {
                        scan_other_from(bytes, p)
                    }
                } else if is_ascii_ws(b1) {
                    advance_ws(bytes, pos + 1, pos)
                } else {
                    scan_other_from(bytes, pos + 2)
                }
            } else {
                pos + 1
            }
        }
        DIGIT => {
            scan_digits_from(bytes, pos + 1)
        }
        APOSTROPHE => match bytes.get(pos + 1) {
            Some(b's' | b'd' | b'm' | b't') => pos + 2,
            Some(b'l') if bytes.get(pos + 2) == Some(&b'l') => pos + 3,
            Some(b'v') if bytes.get(pos + 2) == Some(&b'e') => pos + 3,
            Some(b'r') if bytes.get(pos + 2) == Some(&b'e') => pos + 3,
            _ => scan_other_from(bytes, pos + 1),
        },
        WHITESPACE => {
            advance_ws(bytes, pos + 1, pos)
        }
        OTHER => {
            scan_other_from(bytes, pos + 1)
        }
        NON_ASCII => {
            let c = unsafe { decode_non_ascii(&bytes[pos..]) };
            let p = pos + c.len_utf8();
            if unicode::is_letter(c) {
                scan_letters_from(bytes, p)
            } else if unicode::is_number(c) {
                scan_digits_from(bytes, p)
            } else if unicode::is_whitespace(c) {
                advance_ws(bytes, p, pos)
            } else {
                scan_other_from(bytes, p)
            }
        }
        _ => unsafe { std::hint::unreachable_unchecked() },
    }
}

#[inline(always)]
fn scan_letters_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_letter(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
fn scan_digits_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len && is_digit(unsafe { *bytes.get_unchecked(p) }) {
            p += 1;
        }
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_number(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

#[inline(always)]
fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
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
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_other_complete(c) {
                p += c.len_utf8();
                continue;
            }
        }
        return p;
    }
}

/// Advance through whitespace. `scan_pos` is where to continue scanning,
/// `token_start` is where the token began (for the split-off-last-char logic).
#[inline(always)]
fn advance_ws(bytes: &[u8], scan_pos: usize, token_start: usize) -> usize {
    let len = bytes.len();
    let mut p = scan_pos;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if is_ascii_ws(b) {
            p += 1;
        } else if b >= 0x80 {
            let c = unsafe { decode_non_ascii(&bytes[p..]) };
            if unicode::is_whitespace(c) {
                p += c.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if p < len {
        let ws_bytes = p - token_start;
        if ws_bytes >= 2 {
            let mut last = p - 1;
            while last > token_start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > token_start {
                return last;
            }
        }
    }
    p
}

// -----------------------------------------------------------------------
// Dual-cursor count
// -----------------------------------------------------------------------

/// Find a safe split point: a position where a newline is followed by a
/// non-whitespace byte. This guarantees a token boundary.
fn find_split(bytes: &[u8], start: usize, target: usize) -> Option<usize> {
    // Search forward from target for \n followed by non-ws
    let mut i = target;
    while i + 1 < bytes.len() {
        if bytes[i] == b'\n' {
            let next = bytes[i + 1];
            if !is_ascii_ws(next) && next < 0x80 {
                return Some(i + 1);
            }
        }
        i += 1;
        // Don't search too far
        if i > target + 4096 {
            break;
        }
    }
    // Search backward from target
    let mut i = target;
    while i > start + 1 {
        if bytes[i - 1] == b'\n' {
            let next = bytes[i];
            if !is_ascii_ws(next) && next < 0x80 {
                return Some(i);
            }
        }
        i -= 1;
        if i + 4096 < target {
            break;
        }
    }
    None
}

fn count_dual_cursor(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    if start >= len {
        return 0;
    }

    // Try to split near the midpoint
    let mid = start + (len - start) / 2;
    let split = match find_split(bytes, start, mid) {
        Some(s) => s,
        None => {
            // No safe split found — fall back to single cursor
            let mut p = start;
            let mut n = 0usize;
            while p < len {
                p = advance_pos(bytes, p);
                n += 1;
            }
            return n;
        }
    };

    let mut p1 = start;
    let mut p2 = split;
    let mut count = 0usize;

    // Interleaved loop — two independent dependency chains
    while p1 < split && p2 < len {
        p1 = advance_pos(bytes, p1);
        p2 = advance_pos(bytes, p2);
        count += 2;
    }

    // Drain whichever cursor has remaining work
    while p1 < split {
        p1 = advance_pos(bytes, p1);
        count += 1;
    }
    while p2 < len {
        p2 = advance_pos(bytes, p2);
        count += 1;
    }

    count
}

// -----------------------------------------------------------------------
// Direct SWAR transition counting — no intermediate buffer
// -----------------------------------------------------------------------

/// SWAR: bit 7 set per byte where byte == val. Requires all bytes < 0x80.
#[inline(always)]
fn swar_cmpeq(word: u64, val: u8) -> u64 {
    let v = val as u64 * 0x0101_0101_0101_0101;
    ((word | HI).wrapping_sub(v)) & ((v | HI).wrapping_sub(word)) & HI
}

/// SWAR: bit 7 set per byte that is an ASCII digit [0-9].
#[inline(always)]
fn swar_digit_mask(word: u64) -> u64 {
    let ge = (word | HI).wrapping_sub(0x3030_3030_3030_3030);
    let le = 0xB9B9_B9B9_B9B9_B9B9_u64.wrapping_sub(word);
    ge & le & HI
}

/// SWAR: bit 7 set per byte that is ASCII whitespace (space or 9-13).
#[inline(always)]
fn swar_ws_mask(word: u64) -> u64 {
    let is_space = swar_cmpeq(word, b' ');
    let ge = (word | HI).wrapping_sub(0x0909_0909_0909_0909);
    let le = 0x8D8D_8D8D_8D8D_8D8D_u64.wrapping_sub(word);
    is_space | (ge & le & HI)
}

/// Count tokens using direct SWAR transition detection on raw bytes.
///
/// For each pair of adjacent bytes, determines whether they are in the same
/// merged character class (letter, digit, ws, other) using pure arithmetic —
/// no LUT, no classification buffer. Non-ASCII and special cases (contractions,
/// whitespace splits) are handled by a sequential fixup pass that only fires
/// at the rare transition points.
fn count_swar_transitions(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    if start >= len {
        return 0;
    }
    let bytes = &bytes[start..];
    let len = bytes.len();

    let mut total: usize = 1; // first byte always starts a token
    let mut pos: usize = 0;

    // ---- Main SWAR loop: 8 transitions per iteration ----
    while pos + 9 <= len {
        let w0 = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        let w1 = unsafe { (bytes.as_ptr().add(pos + 1) as *const u64).read_unaligned() };

        // If ANY of the 16 bytes is non-ASCII, fall back to scalar for this block.
        if (w0 | w1) & HI != 0 {
            // Process 8 transitions byte-by-byte
            for i in 0..8 {
                let b0 = bytes[pos + i];
                let b1 = bytes[pos + i + 1];
                total += scalar_transition(bytes, pos + i, b0, b1);
            }
            pos += 8;
            continue;
        }

        // All ASCII — classify both words into 4 merged classes:
        //   LETTER, DIGIT, WS (space+ws merged), OTHER (other+apostrophe merged)
        let l0 = swar64_letter_mask(w0);
        let l1 = swar64_letter_mask(w1);
        let d0 = swar_digit_mask(w0);
        let d1 = swar_digit_mask(w1);
        let ws0 = swar_ws_mask(w0);
        let ws1 = swar_ws_mask(w1);
        // "other" = everything that's not letter, digit, or ws (and not non-ASCII, already checked)
        let o0 = !(l0 | d0 | ws0) & HI;
        let o1 = !(l1 | d1 | ws1) & HI;

        // Same merged class: corresponding bytes are both in the same class.
        let same = (l0 & l1) | (d0 & d1) | (ws0 & ws1) | (o0 & o1);
        let transitions = !same & HI;
        total += transitions.count_ones() as usize;

        // Subtract space-absorbed transitions:
        // byte0 == ' ' (actual space, not tab/newline) AND transition AND byte1 is NOT ws
        let space0 = swar_cmpeq(w0, b' ');
        let absorbed = space0 & transitions & (!ws1 & HI);
        total -= absorbed.count_ones() as usize;

        pos += 8;
    }

    // ---- Scalar tail for remaining bytes ----
    while pos + 1 <= len {
        let b0 = bytes[pos];
        let b1 = if pos + 1 < len { bytes[pos + 1] } else { break };
        total += scalar_transition(bytes, pos, b0, b1);
        pos += 1;
    }

    // ---- Fixup: whitespace splits ----
    // A run of 2+ ws bytes followed by non-ws generates one extra token.
    {
        let mut i = 0;
        while i < len {
            let b = bytes[i];
            if is_ascii_ws(b) || (b >= 0x80 && is_byte_unicode_ws(bytes, i)) {
                let ws_start = i;
                while i < len
                    && (is_ascii_ws(bytes[i])
                        || (bytes[i] >= 0x80 && is_byte_unicode_ws(bytes, i)))
                {
                    i += 1;
                }
                let ws_len = i - ws_start;
                if i < len && ws_len >= 2 {
                    total += 1;
                }
            } else {
                i += 1;
            }
        }
    }

    // ---- Fixup: contractions ----
    // A contraction like 's at a token boundary merges APOS+LETTER into one token.
    // The SWAR counted OTHER→LETTER as a transition. If the contraction is NOT
    // followed by the same class (LETTER), subtract 1. If it IS followed by LETTER,
    // the subtraction is offset by an invisible boundary → no net change.
    {
        let mut i = 0;
        while i + 1 < len {
            if bytes[i] == b'\'' && is_contraction_at(bytes, i) {
                // Is the apostrophe at a token boundary? It is if the previous byte
                // is a different merged class (not OTHER/APOSTROPHE).
                let at_boundary = if i == 0 {
                    true
                } else {
                    let prev = bytes[i - 1];
                    // Previous byte is letter, digit, ws, or non-ASCII → boundary exists
                    is_letter(prev) || is_digit(prev) || is_ascii_ws(prev) || prev >= 0x80
                };
                // Was the apostrophe absorbed by a preceding space?
                let space_absorbed = i > 0 && bytes[i - 1] == b' ';

                if at_boundary && !space_absorbed {
                    let clen = contraction_len_at(bytes, i);
                    let after_is_letter =
                        i + clen < len && is_letter(bytes[i + clen]);
                    if !after_is_letter {
                        total -= 1;
                    }
                    i += clen;
                    continue;
                }
            }
            i += 1;
        }
    }

    total
}

/// Scalar transition check for a single pair of adjacent bytes.
/// Returns 1 if the pair represents a token boundary, 0 otherwise.
#[inline(always)]
fn scalar_transition(bytes: &[u8], pos: usize, b0: u8, b1: u8) -> usize {
    let c0 = merged_class(bytes, pos, b0);
    let c1 = merged_class(bytes, pos + 1, b1);
    if c0 == c1 {
        return 0;
    }
    // Transition exists. Check for space absorption.
    if b0 == b' ' && !is_ascii_ws(b1) && !(b1 >= 0x80 && is_byte_unicode_ws(bytes, pos + 1)) {
        return 0; // absorbed
    }
    1
}

/// Return the merged class of a byte: LETTER=0, DIGIT=1, WS=2, OTHER=3.
/// Non-ASCII bytes are resolved via unicode.
#[inline(always)]
fn merged_class(bytes: &[u8], pos: usize, b: u8) -> u8 {
    if b < 0x80 {
        if is_letter(b) { 0 }
        else if is_digit(b) { 1 }
        else if is_ascii_ws(b) { 2 }
        else { 3 } // other + apostrophe merged
    } else {
        // Non-ASCII: decode unicode character
        let mut start = pos;
        while start > 0 && bytes[start] & 0xC0 == 0x80 { start -= 1; }
        if bytes[start] < 0x80 { return 3; } // shouldn't happen
        if let Some(c) = decode_utf8_char_from(bytes, start) {
            if unicode::is_letter(c) { 0 }
            else if unicode::is_number(c) { 1 }
            else if unicode::is_whitespace(c) { 2 }
            else { 3 }
        } else {
            3
        }
    }
}

/// Check if the byte at `pos` belongs to a unicode whitespace character.
#[inline(always)]
fn is_byte_unicode_ws(bytes: &[u8], pos: usize) -> bool {
    merged_class(bytes, pos, bytes[pos]) == 2
}

/// Decode a UTF-8 character starting at `start` in `bytes`.
fn decode_utf8_char_from(bytes: &[u8], start: usize) -> Option<char> {
    if start >= bytes.len() { return None; }
    let b = bytes[start];
    let char_len = if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 };
    if start + char_len > bytes.len() { return None; }
    let mut cp = match char_len {
        2 => (b & 0x1F) as u32,
        3 => (b & 0x0F) as u32,
        4 => (b & 0x07) as u32,
        _ => return None,
    };
    for i in 1..char_len {
        cp = (cp << 6) | (bytes[start + i] & 0x3F) as u32;
    }
    char::from_u32(cp)
}

/// Check if position `pos` in `bytes` starts a valid contraction.
fn is_contraction_at(bytes: &[u8], pos: usize) -> bool {
    if pos >= bytes.len() || bytes[pos] != b'\'' { return false; }
    match bytes.get(pos + 1) {
        Some(b's' | b'd' | b'm' | b't') => true,
        Some(b'l') if bytes.get(pos + 2) == Some(&b'l') => true,
        Some(b'v') if bytes.get(pos + 2) == Some(&b'e') => true,
        Some(b'r') if bytes.get(pos + 2) == Some(&b'e') => true,
        _ => false,
    }
}

fn contraction_len_at(bytes: &[u8], pos: usize) -> usize {
    match bytes.get(pos + 1) {
        Some(b's' | b'd' | b'm' | b't') => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::pretokenize_as_iter;

    #[test]
    fn twopass_small_cases() {
        fn count(s: &str) -> usize {
            FastPretokenizer::new(s.as_bytes()).count()
        }
        fn count_onepass(s: &str) -> usize {
            let mut iter = FastPretokenizer::new(s.as_bytes());
            let mut n = 0;
            while iter.next().is_some() { n += 1; }
            n
        }
        let cases = [
            "hello",
            " hello",
            "hello world",
            " hello world",
            "  hello",
            "\nhello",
            "\n\nhello",
            "don't",
            "don't stop",
            "they'll go",
            "it's he'll",
            "'hello",
            " 'hello",
            "123 456",
            " 123",
            "hello, world!",
            "  \n  hello",
            "\n\n",
            "   ",
            " ",
            "",
            "café",
        ];
        for &case in &cases {
            let expected = count_onepass(case);
            let got = count(case);
            assert_eq!(expected, got, "Case {:?}: expected {expected}, got {got}", case);
        }
        eprintln!("All small cases pass.");
    }

    #[test]
    fn twopass_find_divergence() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");

        // Binary search for the first byte where counts diverge
        let max = 5_000_000.min(all_bytes.len());
        let counts_match = |end: usize| -> bool {
            let input = &all_bytes[..end];
            let mut iter = FastPretokenizer::new(input);
            let mut op = 0;
            while iter.next().is_some() { op += 1; }
            let tp = FastPretokenizer::new(input).count();
            op == tp
        };

        let mut lo = 0usize;
        let mut hi = max;
        // Find UTF-8-safe hi
        while hi > 0 && std::str::from_utf8(&all_bytes[..hi]).is_err() { hi -= 1; }

        if counts_match(hi) {
            eprintln!("No divergence found in {hi} bytes");
            return;
        }

        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            let mut m = mid;
            while m > 0 && std::str::from_utf8(&all_bytes[..m]).is_err() { m -= 1; }
            if m <= lo { lo = mid; continue; } // couldn't find UTF-8 boundary, skip
            if counts_match(m) { lo = m; } else { hi = m; }
        }
        eprintln!("First divergence at byte {hi}");

        // Show context
        let start_ctx = hi.saturating_sub(40);
        let end_ctx = (hi + 10).min(max);
        eprintln!("Context: {:?}", String::from_utf8_lossy(&all_bytes[start_ctx..end_ctx]));
        eprintln!("Classes: {:?}", all_bytes[start_ctx..end_ctx].iter().map(|&b| CLASS[b as usize]).collect::<Vec<_>>());

        for &size in &[hi, hi - 1, hi - 2] {
            let mut end = size;
            while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() { end -= 1; }
            let input = &all_bytes[..end];
            let mut iter = FastPretokenizer::new(input);
            let mut onepass = 0;
            while iter.next().is_some() { onepass += 1; }
            let twopass = FastPretokenizer::new(input).count();
            if onepass != twopass {
                eprintln!("DIVERGENCE at size {end}: one-pass={onepass} two-pass={twopass} diff={}", onepass as isize - twopass as isize);
                let start = end.saturating_sub(30);
                eprintln!("Tail: {:?}", String::from_utf8_lossy(&input[start..end]));
                eprintln!("Tail classes: {:?}", input[start..end].iter().map(|&b| CLASS[b as usize]).collect::<Vec<_>>());
                // Also check size-1
                let prev = &all_bytes[..end - 1];
                let mut iter2 = FastPretokenizer::new(prev);
                let mut op2 = 0;
                while iter2.next().is_some() { op2 += 1; }
                let tp2 = FastPretokenizer::new(prev).count();
                eprintln!("At size {}: one-pass={op2} two-pass={tp2}", end - 1);
                break;
            }
        }
    }

    #[test]
    fn twopass_count_matches_onepass() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        // One-pass count (known correct from fast_matches_state_machine_owt)
        let mut iter = FastPretokenizer::new(input);
        let mut onepass_count = 0;
        while iter.next().is_some() {
            onepass_count += 1;
        }

        // Two-pass count
        let twopass_count = FastPretokenizer::new(input).count();

        assert_eq!(
            onepass_count, twopass_count,
            "One-pass ({onepass_count}) != two-pass ({twopass_count})"
        );
        eprintln!("Both counts match: {onepass_count}");
    }

    #[test]
    fn fast_matches_state_machine_owt() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        let mut sm = pretokenize_as_iter(input);
        let mut fast = FastPretokenizer::new(input);
        let mut idx = 0usize;

        loop {
            match (sm.next(), fast.next()) {
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a.0, b.0,
                        "Mismatch at token {idx}: sm={:?} fast={:?}",
                        String::from_utf8_lossy(a.0),
                        String::from_utf8_lossy(b.0),
                    );
                }
                (None, None) => break,
                (Some(a), None) => panic!("SM extra at {idx}: {:?}", String::from_utf8_lossy(a.0)),
                (None, Some(b)) => {
                    panic!("Fast extra at {idx}: {:?}", String::from_utf8_lossy(b.0))
                }
            }
            idx += 1;
        }
        eprintln!("All {idx} tokens match.");
    }
}
