//! Fast pretokenizer for the Qwen3.5 regex — on aarch64 (NEON) and
//! x86_64 with AVX-512 (runtime-detected) a mask scanner
//! via the shared `cl100k_family::batch_masks` boundary algebra with the
//! mark-folding classifier (`unicode::class_of_marks_join`), so marks
//! join letter runs in-mask exactly as in the scalar `advance_pos`:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Differences from the Qwen2/Qwen3 scheme:
//! - `\p{M}` joins letter runs (`[\p{L}\p{M}]+` instead of `\p{L}+`), so a
//!   combining mark extends a word and a bare mark run is a word of its own
//! - `\p{M}` is excluded from the punctuation run (`[^\s\p{L}\p{M}\p{N}]+`),
//!   so a mark after punctuation terminates the run
//!
//! The optional one-char prefix `[^\r\n\p{L}\p{N}]?` is unchanged; a mark is
//! a valid prefix, but since marks are also in the run class the match span
//! is the same either way.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg(target_arch = "aarch64")]
use super::cl100k_family::batch_masks;
#[cfg(target_arch = "x86_64")]
use super::cl100k_family::batch_masks_x86;
use super::mask::{MaskScheme, MaskState};
use super::{decode_cp, is_ascii_ws, is_digit, is_letter, scan_newlines, swar_scan_letters};
use crate::pretokenize::unicode::{self, DsCharClass, ds_class_of};
use crate::pretokenize::Pretoken;

pub(crate) struct Qwen35Scheme;

impl MaskScheme for Qwen35Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::DsClassTable::get();
        batch_masks(bytes, scan, false, move |cp| ct.class_of_marks_join(cp))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::DsClassTable::get();
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan, false, move |cp| ct.class_of_marks_join(cp)) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512 detected at runtime),
/// iteration runs the shared cl100k-family mask scanner (see
/// `cl100k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastQwen35Pretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastQwen35Pretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, state: MaskState::new(pos) }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }
}

impl<'a> Iterator for FastQwen35Pretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<Qwen35Scheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastQwen35Pretokenizer, Qwen35Scheme);

/// If the char at `pos` is `\p{L}` or `\p{M}`, return the offset just past it.
#[inline(always)]
fn lm_end_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some(pos + 1);
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        if matches!(ds_class_of(cp), DsCharClass::Letter | DsCharClass::Mark) {
            return Some(pos + l);
        }
    }
    None
}

/// `[\p{L}\p{M}]+` from `pos`.
#[inline(always)]
fn scan_lm_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        p = swar_scan_letters(bytes, p);
        if p < len && unsafe { *bytes.get_unchecked(p) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(ds_class_of(cp), DsCharClass::Letter | DsCharClass::Mark) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// `[^\s\p{L}\p{M}\p{N}]+` from `pos` (punctuation, symbols, controls —
/// everything except letters, marks, numbers, and whitespace).
#[inline(always)]
fn scan_other_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 {
                break;
            }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                return p;
            }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(ds_class_of(cp), DsCharClass::PunctSym | DsCharClass::Other) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// Whitespace-led token starting at `start`, i.e. the alternatives
/// `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, in that priority.
/// Precondition: the letter-prefix (`[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`) and
/// space+punct (` ?[^\s\p{L}\p{M}\p{N}]+...`) alternatives were ruled out.
#[inline(always)]
fn ws_token_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if ds_class_of(cp) == DsCharClass::Whitespace {
                last_char_start = p;
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if last_nl_end != 0 {
        return last_nl_end; // `\s*[\r\n]+`: through the last newline, even at EOS
    }
    if p >= len {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
fn advance_pos(bytes: &[u8], pos: usize) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter — `[\p{L}\p{M}]+` with empty prefix
    if is_letter(b0) {
        return scan_lm_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_lm_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => return scan_lm_from(bytes, p1),
            DsCharClass::Whitespace => return ws_token_end(bytes, pos),
            DsCharClass::Number => return pos + 1,
            DsCharClass::PunctSym | DsCharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        match ds_class_of(cp) {
            DsCharClass::Letter | DsCharClass::Mark => return scan_lm_from(bytes, p0),
            DsCharClass::Number => return p0, // `\p{N}`: exactly one char
            // Any non-letter/mark/number char except \r\n may prefix a run
            class => {
                if let Some(p) = lm_end_at(bytes, p0) {
                    return scan_lm_from(bytes, p);
                }
                if class == DsCharClass::Whitespace {
                    return ws_token_end(bytes, pos);
                }
                let p = scan_other_from(bytes, p0);
                return scan_newlines(bytes, p);
            }
        }
    }

    // ASCII digit: `\p{N}` matches exactly one char
    if is_digit(b0) {
        return pos + 1;
    }

    // Apostrophe: case-insensitive contractions
    if b0 == b'\'' {
        match bytes.get(pos + 1).map(u8::to_ascii_lowercase) {
            Some(b's' | b'd' | b'm' | b't') => return pos + 2,
            Some(b'l') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'l') => {
                return pos + 3;
            }
            Some(b'v') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            Some(b'r') if bytes.get(pos + 2).map(u8::to_ascii_lowercase) == Some(b'e') => {
                return pos + 3;
            }
            _ => {}
        }
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's' under `(?i)`
        if bytes.get(pos + 1) == Some(&0xC5) && bytes.get(pos + 2) == Some(&0xBF) {
            return pos + 3;
        }
        // Not a contraction: `'` can still prefix a letter/mark run
        if let Some(p) = lm_end_at(bytes, pos + 1) {
            return scan_lm_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter/mark run
    if is_ascii_ws(b0) {
        if let Some(p) = lm_end_at(bytes, pos + 1) {
            return scan_lm_from(bytes, p);
        }
        return ws_token_end(bytes, pos);
    }

    // ASCII punctuation/symbol/control
    if let Some(p) = lm_end_at(bytes, pos + 1) {
        return scan_lm_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// The Qwen3.5 pattern verbatim — it contains no possessive quantifiers,
    /// so it runs directly under fancy-regex.
    const QWEN35_REF_REGEX: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(QWEN35_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastQwen35Pretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
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
    fn qwen35_small_cases() {
        let cases = [
            "hello",
            " hello",
            "hello world",
            "  hello",
            "   hello",
            "\thello",
            "\t\thello",
            "\nhello",
            "\n\nhello",
            "\n\n   hello",
            "!hello",
            "!!hello",
            "?!x",
            "don't",
            "DON'T",
            "they'LL go",
            "it'S he'Ll",
            "we'Ve THEY'RE",
            "'sound",
            "'lx",
            "'hello",
            " 'hello",
            " 's",
            "x'0",
            "123",
            "1234",
            "1234567",
            " 123",
            "  123",
            "3rd",
            "abc1234def",
            "hello, world!",
            "hi!\n\ndef",
            "hi !!\n\ndef",
            " !!!",
            "a-b",
            "a - b",
            "...",
            "hello\n",
            "hello \n",
            "hello \nx",
            "hello\n x",
            "hello  \n\n  ",
            "x \n\n ",
            "x  ",
            "x \t",
            "  \n  hello",
            "\r\nhello",
            "a\r\n",
            "a\r\n ",
            "a\n \n",
            "a \n \t",
            "\n\n",
            "\n\n\t",
            "   ",
            " ",
            "",
            "café",
            " café",
            "\u{a0}word",
            "voilà ¡hola!",
            "١٢٣٤٥",
            "1٢3x",
            "tab\tsep\tvals",
            "\x0bword",
            "a\u{2028}b",
            "a\u{2028}\n",
            "price: $5.99!",
            "'ſ",
            "it'ſ fine",
            "日本語のテキスト",
            " 日本語",
            // Mark-specific cases: `[\p{L}\p{M}]+` runs and marks excluded
            // from punctuation runs.
            "e\u{301}f",
            "cafe\u{301} de\u{301}composed",
            "\u{301}leading mark",
            "\u{301}\u{301}two marks",
            " \u{301}abc",
            "\t\u{301}abc",
            "!\u{301}",
            "!\u{301}!",
            "!!\u{301}x",
            "1\u{301}2",
            "'\u{301}s",
            "देवनागरी में परीक्षण",
            "अंग्रेज़ी",
            "టెస్ట్ తెలుగు",
            "עִבְרִית נִקּוּד",
            "الْعَرَبِيَّة",
            "a\u{20dd}b",
            " \u{20dd}",
            "\u{200b}\u{301}x",
        ];
        for case in cases {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes,
    /// compared against the reference regex.
    #[test]
    fn qwen35_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'Z', 'é', 'ß', 'Ж', 'ا', '한', '日'],      // letters
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        let mut rng = StdRng::seed_from_u64(0x93E3_5EEC);
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
                regex_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }

    #[test]
    fn qwen35_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt_prefix(SIZE);
        let text = std::str::from_utf8(&input).unwrap();
        eprintln!(
            "Testing qwen3.5 fast pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(QWEN35_REF_REGEX).unwrap();
        let mut fast_iter = FastQwen35Pretokenizer::new(&input);
        let mut re_iter = re.find_iter(text);
        let mut token_idx: usize = 0;
        let mut recent: Vec<(String, String)> = Vec::new();

        loop {
            match (fast_iter.next(), re_iter.next()) {
                (Some(fast_tok), Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    let fast_str = String::from_utf8_lossy(fast_tok.0);
                    let re_str = &text[re_match.start()..re_match.end()];
                    recent.push((fast_str.to_string(), re_str.to_string()));
                    if recent.len() > 10 {
                        recent.remove(0);
                    }
                    assert_eq!(
                        fast_str, re_str,
                        "Mismatch at token {token_idx} (byte ~{}).\n  fast:  {:?}\n  regex: {:?}\n  recent tokens: {:?}",
                        re_match.start(), fast_str, re_str, recent
                    );
                }
                (None, None) => break,
                (Some(fast_tok), None) => panic!(
                    "Fast produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                    String::from_utf8_lossy(fast_tok.0),
                    recent
                ),
                (None, Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    panic!(
                        "Regex produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        &text[re_match.start()..re_match.end()],
                        recent
                    );
                }
            }
            token_idx += 1;
        }
        eprintln!("All {token_idx} tokens match.");
    }
}
