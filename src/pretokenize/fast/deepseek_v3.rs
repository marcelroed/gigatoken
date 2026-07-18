//! Fast scalar pretokenizer for the DeepSeek V3/V3.1/V4 scheme — a
//! `Sequence` of three `Split` (Isolated) pre-tokenizers applied in order:
//!
//! 1. `\p{N}{1,3}` — number runs, three chars at a time
//! 2. `[\u{4E00}-\u{9FA5}\u{3040}-\u{30FF}]+` — CJK ideograph / kana runs
//! 3. the main regex:
//!    `[ascii punct][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Sequence `Split`s re-split every piece produced by the previous stage,
//! and spans a regex does not match survive as their own pieces (the digit
//! pieces from stage 1, or the control-character gaps stage 3 skips).
//!
//! A single left-to-right pass reproduces the hierarchy by treating number
//! chars and CJK-range chars as hard piece boundaries for the main regex:
//! no match may cross one, and the `(?!\S)` lookahead succeeds at a
//! boundary exactly as it does at end of input (so a whitespace run ending
//! at a digit stays whole). Within a CJK piece the main regex still runs —
//! the ranges contain a few non-letters (U+309B/U+309C voicing marks are
//! `\p{S}`, U+30A0/U+30FB are `\p{P}`, U+3040 etc. are unassigned) — with
//! the piece edges as the region bounds. Each scan therefore carries a
//! `cjk_region` flag: a char belongs to the current region iff
//! `is_deepseek_cjk(cp) == cjk_region`.

use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, scan_newlines, scan_numbers_max3,
    swar_scan_letters,
};
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{DsCharClass, ds_class_of, is_deepseek_cjk};

pub struct FastDeepSeekV3Pretokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastDeepSeekV3Pretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FastDeepSeekV3Pretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        self.pos = advance_pos(self.bytes, start);
        Some(Pretoken(&self.bytes[start..self.pos]))
    }
}

// SAFETY: delegates to `fill_spans_keyed_with_buf`, which writes exactly
// the first `n` entries from live in-bounds spans of `self.bytes`.
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastDeepSeekV3Pretokenizer<'a> {
    /// Chunked pull with the cursor in a local across the whole chunk (the
    /// per-`next` store-load of `self.pos` costs real time in the encode
    /// loop's register-starved surroundings).
    #[inline(never)]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        let (bytes, len) = (self.bytes, self.bytes.len());
        let mut pos = self.pos;
        let n = crate::pretokenize::fill_spans_keyed_with_buf(
            bytes,
            || {
                if pos >= len {
                    return None;
                }
                let start = pos;
                // advance_pos returns an in-bounds end > start.
                pos = advance_pos(bytes, start);
                Some((start, pos))
            },
            batch,
            prefetch,
        );
        self.pos = pos;
        n
    }

    #[inline]
    fn remaining_bytes_hint(&self) -> Option<usize> {
        Some(self.bytes.len().saturating_sub(self.pos))
    }
}

/// If the char at `pos` is `\p{L}` or `\p{M}` within the region, return the
/// offset just past it.
#[inline(always)]
fn lm_end_at(bytes: &[u8], pos: usize, cjk_region: bool) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if b < 0x80 {
        if !cjk_region && is_letter(b) {
            return Some(pos + 1);
        }
        return None;
    }
    let (cp, l) = unsafe { decode_cp(bytes, pos) };
    if is_deepseek_cjk(cp) != cjk_region {
        return None;
    }
    match ds_class_of(cp) {
        DsCharClass::Letter | DsCharClass::Mark => Some(pos + l),
        _ => None,
    }
}

/// `[\p{L}\p{M}]+` from `pos`, bounded by the region.
#[inline(always)]
fn scan_lm_from(bytes: &[u8], pos: usize, cjk_region: bool) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        if !cjk_region {
            p = swar_scan_letters(bytes, p);
        }
        if p >= len || unsafe { *bytes.get_unchecked(p) } < 0x80 {
            return p; // ASCII non-letter (or any ASCII inside a CJK region)
        }
        let (cp, l) = unsafe { decode_cp(bytes, p) };
        if is_deepseek_cjk(cp) != cjk_region {
            return p;
        }
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => p += l,
            _ => return p,
        }
    }
}

/// `[\p{P}\p{S}]+` from `pos`, bounded by the region.
#[inline(always)]
fn scan_ps_from(bytes: &[u8], pos: usize, cjk_region: bool) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        if !cjk_region {
            while p < len {
                let b = unsafe { *bytes.get_unchecked(p) };
                if b >= 0x80 {
                    break;
                }
                if !b.is_ascii_punctuation() {
                    return p;
                }
                p += 1;
            }
        }
        if p >= len || unsafe { *bytes.get_unchecked(p) } < 0x80 {
            return p;
        }
        let (cp, l) = unsafe { decode_cp(bytes, p) };
        if is_deepseek_cjk(cp) != cjk_region || ds_class_of(cp) != DsCharClass::PunctSym {
            return p;
        }
        p += l;
    }
}

/// Whitespace-led token starting at `start`: `\s*[\r\n]+` | `\s+(?!\S)` |
/// `\s+`, in that priority, with the lookahead succeeding at a piece
/// boundary (number/CJK char) as well as at end of input. Main region only.
#[inline(always)]
fn ws_token_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    let mut at_boundary = false;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b < 0x80 {
            at_boundary = is_digit(b);
            break;
        } else {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if ds_class_of(cp) == DsCharClass::Whitespace {
                last_char_start = p;
                p += l;
            } else {
                at_boundary = ds_class_of(cp) == DsCharClass::Number || is_deepseek_cjk(cp);
                break;
            }
        }
    }
    if last_nl_end != 0 {
        return last_nl_end; // `\s*[\r\n]+`: through the last newline, even at EOS
    }
    if p >= len || at_boundary {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS / piece boundary
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// Unmatched-gap piece: chars the main regex cannot match (`Other` class not
/// prefixing a letter/mark run), emitted as one piece like HF's Isolated
/// split leaves them. `first_len` is the byte length of the char at `pos`,
/// which the caller already established starts a gap.
#[inline(always)]
fn scan_gap_from(bytes: &[u8], pos: usize, first_len: usize, cjk_region: bool) -> usize {
    let len = bytes.len();
    let mut p = pos + first_len;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        let (cp, l) = if b < 0x80 {
            (b as u32, 1)
        } else {
            unsafe { decode_cp(bytes, p) }
        };
        if is_deepseek_cjk(cp) != cjk_region || ds_class_of(cp) != DsCharClass::Other {
            return p;
        }
        // This char starts a `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` match,
        // so the gap ends before it.
        if lm_end_at(bytes, p + l, cjk_region).is_some() {
            return p;
        }
        p += l;
    }
    p
}

/// One main-regex token starting at `pos` in the main (non-CJK) region.
/// The char at `pos` is not a number char and not CJK.
#[inline(always)]
fn advance_main(bytes: &[u8], pos: usize) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `[\p{L}\p{M}]+` with empty prefix
    if is_letter(b0) {
        return scan_lm_from(bytes, pos + 1, false);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_lm_from(bytes, pos + 2, false); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // ws run whole before a digit piece
            }
            if b1.is_ascii_punctuation() {
                // ` ?[\p{P}\p{S}]+[\r\n]*`
                let p = scan_ps_from(bytes, pos + 2, false);
                return scan_newlines(bytes, p);
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            return pos + 1; // `\s+`: single space before an ASCII control
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        if is_deepseek_cjk(cp) {
            return pos + 1; // ws run whole before a CJK piece
        }
        let p1 = pos + 1 + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => scan_lm_from(bytes, p1, false),
            DsCharClass::Number => pos + 1,
            DsCharClass::Whitespace => ws_token_end(bytes, pos),
            DsCharClass::PunctSym => {
                let p = scan_ps_from(bytes, p1, false);
                scan_newlines(bytes, p)
            }
            DsCharClass::Other => pos + 1, // `\s+`: single space before a control
        }
    } else if b0 < 0x80 {
        if b0 == b'\r' || b0 == b'\n' {
            return ws_token_end(bytes, pos); // \r\n are excluded from prefixes
        }
        if is_ascii_ws(b0) {
            // \t \x0b \x0c may prefix a letter/mark run
            if let Some(e) = lm_end_at(bytes, pos + 1, false) {
                return scan_lm_from(bytes, e, false);
            }
            return ws_token_end(bytes, pos);
        }
        if b0.is_ascii_punctuation() {
            // `[ascii punct][A-Za-z]+` — ASCII letters only
            if let Some(&b1) = bytes.get(pos + 1)
                && is_letter(b1)
            {
                return swar_scan_letters(bytes, pos + 1);
            }
            let p = scan_ps_from(bytes, pos + 1, false);
            return scan_newlines(bytes, p);
        }
        // ASCII control: may prefix a letter/mark run, else starts a gap
        if let Some(e) = lm_end_at(bytes, pos + 1, false) {
            return scan_lm_from(bytes, e, false);
        }
        scan_gap_from(bytes, pos, 1, false)
    } else {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => scan_lm_from(bytes, p0, false),
            DsCharClass::Whitespace => {
                // Non-\r\n whitespace may prefix a letter/mark run
                if let Some(e) = lm_end_at(bytes, p0, false) {
                    return scan_lm_from(bytes, e, false);
                }
                ws_token_end(bytes, pos)
            }
            DsCharClass::PunctSym => {
                let p = scan_ps_from(bytes, p0, false);
                scan_newlines(bytes, p)
            }
            // `Other` (controls/format/unassigned); `Number` is unreachable
            // (the caller dispatched it to the digit rule).
            DsCharClass::Number | DsCharClass::Other => {
                if let Some(e) = lm_end_at(bytes, p0, false) {
                    return scan_lm_from(bytes, e, false);
                }
                scan_gap_from(bytes, pos, l, false)
            }
        }
    }
}

/// One main-regex token starting at `pos` inside a CJK piece. The char at
/// `pos` is in the CJK ranges. No whitespace, newlines, or ASCII exist in
/// the region, so only the letter/mark, punct/symbol, and gap rules apply.
#[inline(always)]
fn advance_cjk(bytes: &[u8], pos: usize) -> usize {
    let (cp, l) = unsafe { decode_cp(bytes, pos) };
    let p0 = pos + l;
    match ds_class_of(cp) {
        DsCharClass::Letter | DsCharClass::Mark => scan_lm_from(bytes, p0, true),
        DsCharClass::PunctSym => scan_ps_from(bytes, p0, true),
        _ => {
            // Unassigned (e.g. U+3040): may prefix a letter/mark run
            if let Some(e) = lm_end_at(bytes, p0, true) {
                return scan_lm_from(bytes, e, true);
            }
            scan_gap_from(bytes, pos, l, true)
        }
    }
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };
    if b0 < 0x80 {
        if is_digit(b0) {
            return scan_numbers_max3(bytes, pos + 1, 1); // `\p{N}{1,3}`
        }
        return advance_main(bytes, pos);
    }
    let (cp, l) = unsafe { decode_cp(bytes, pos) };
    if is_deepseek_cjk(cp) {
        return advance_cjk(bytes, pos);
    }
    if ds_class_of(cp) == DsCharClass::Number {
        return scan_numbers_max3(bytes, pos + l, 1);
    }
    advance_main(bytes, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// The three Split regexes verbatim (no possessive quantifiers, so they
    /// run directly under fancy-regex).
    const DS_SPLIT_1: &str = r"\p{N}{1,3}";
    const DS_SPLIT_2: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
    const DS_SPLIT_3: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

    /// HF `Split` with Isolated behavior: matches and the gaps between them.
    fn split_isolated<'s>(re: &fancy_regex::Regex, s: &'s str) -> Vec<&'s str> {
        let mut pieces = Vec::new();
        let mut last = 0;
        for m in re.find_iter(s) {
            let m = m.expect("regex match error");
            if m.start() > last {
                pieces.push(&s[last..m.start()]);
            }
            pieces.push(m.as_str());
            last = m.end();
        }
        if last < s.len() {
            pieces.push(&s[last..]);
        }
        pieces
    }

    /// Reference implementation: the three Splits applied in sequence, each
    /// re-splitting every piece from the previous stage.
    fn reference_tokens(s: &str) -> Vec<String> {
        let res = [
            fancy_regex::Regex::new(DS_SPLIT_1).unwrap(),
            fancy_regex::Regex::new(DS_SPLIT_2).unwrap(),
            fancy_regex::Regex::new(DS_SPLIT_3).unwrap(),
        ];
        let mut pieces: Vec<&str> = vec![s];
        for re in &res {
            pieces = pieces
                .iter()
                .flat_map(|p| split_isolated(re, p))
                .collect();
        }
        pieces.into_iter().map(str::to_owned).collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastDeepSeekV3Pretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    #[test]
    fn deepseek_small_cases() {
        let cases = [
            "hello",
            " hello",
            "hello world",
            "  hello",
            "\thello",
            "\nhello",
            "\n\n   hello",
            "!hello",
            "!!hello",
            "?!x",
            "don't",
            "DON'T",
            "they'LL go",
            "'sound",
            "'sé",
            "'ſ",
            "x'0",
            "123",
            "1234",
            "12345678",
            " 123",
            "  123",
            "a  1",
            "3rd",
            "3.14159",
            "abc123def",
            "abc 123 def",
            "١٢٣٤٥ ٦٧",
            "1٢3x",
            "hello, world!",
            "hi!\n\ndef",
            " !!!",
            " !word",
            "a-b",
            "...",
            "hello\n",
            "hello \nx",
            "hello  \n\n  ",
            "x \n\n ",
            "x  ",
            "  \n  hello",
            "\r\nhello",
            "a\r\n ",
            "a\n \n",
            "\n\n",
            "   ",
            " ",
            "",
            "café",
            " café",
            "cafe\u{301} de\u{301}composed",
            "\u{301}leading mark",
            " \u{301}abc",
            "e\u{301}f",
            "\u{a0}word",
            "voilà ¡hola!",
            "日本語テスト",
            " 日本語",
            " 日本語 abc",
            "漢字test",
            "x字y",
            "あ゛が",
            "\u{3040}あ",
            "ん・カナ",
            "ガード",
            "カタカナ゠ヷヸヹ",
            "一二三１２３",
            "＄￥",
            "a\u{0}b",
            "a\u{0}\u{0}b",
            "\u{0}",
            " \u{0}",
            "\u{ad}word",
            "a\u{200b}b",
            "tab\tsep\tvals",
            "\x0bword",
            "a\u{2028}b",
            "price: $5.99!",
            "@user #tag",
            "<|endoftext|>",
            "https://example.com/path?q=1",
        ];
        for case in cases {
            assert_eq!(
                fast_tokens(case),
                reference_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes,
    /// compared against the composed reference regexes.
    #[test]
    fn deepseek_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'Z', 'é', 'ß', 'Ж', 'ا', '한'],           // letters
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '+', '«', '¡', '€', '☃'],  // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
            &['一', '龥', 'あ', 'ゟ', '゠', 'ヿ', '゛', '\u{3040}', '・', 'ー'], // CJK ranges
            &['丂', '日', '本', 'カ', 'な'],                  // more CJK
        ];
        let mut rng = StdRng::seed_from_u64(0xDEE9_5EEC);
        for round in 0..2000 {
            let len = rng.random_range(1..40);
            let s: String = (0..len)
                .map(|_| {
                    let pool = pools.choose(&mut rng).unwrap();
                    *pool.choose(&mut rng).unwrap()
                })
                .collect();
            assert_eq!(
                fast_tokens(&s),
                reference_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }

    /// Load the first `max_bytes` of ~/data/owt_train.txt, truncated to a
    /// UTF-8 boundary (streamed; the full file is ~12 GB).
    fn load_owt_prefix(max_bytes: usize) -> Vec<u8> {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("Could not open ~/data/owt_train.txt");
        let mut buf = Vec::with_capacity(max_bytes);
        f.take(max_bytes as u64).read_to_end(&mut buf).unwrap();
        while !buf.is_empty() && std::str::from_utf8(&buf).is_err() {
            buf.pop();
        }
        buf
    }

    #[test]
    fn deepseek_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt_prefix(SIZE);
        let text = std::str::from_utf8(&input).unwrap();
        eprintln!(
            "Testing deepseek fast pretokenizer vs composed regexes on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );
        let fast: Vec<String> = fast_tokens(text);
        let reference = reference_tokens(text);
        for (i, (f, r)) in fast.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                f,
                r,
                "Mismatch at token {i}: fast {f:?} vs reference {r:?}, context: {:?}",
                &fast[i.saturating_sub(5)..(i + 5).min(fast.len())]
            );
        }
        assert_eq!(fast.len(), reference.len());
        eprintln!("All {} tokens match.", fast.len());
    }
}
