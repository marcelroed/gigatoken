//! Fast pretokenizer for the Olmo 2/3 (dolma2) regex — on aarch64 (NEON)
//! and x86_64 with AVX-512 (runtime-detected) a mask scanner via the shared `cl100k_family::batch_masks` boundary algebra,
//! with the scalar `advance_pos` below as reference, no-SIMD fallback,
//! and bad-zone/tail executor:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! This is the Qwen2 scheme with cl100k's number rule: `\p{N}{1,3}` matches
//! runs of up to THREE number chars (Qwen2 matches exactly one). Everything
//! else — contractions, letter-run prefixes, the `\s*[\r\n]+` newline rule
//! outranking end-of-input whitespace — is identical to Qwen2.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg(target_arch = "aarch64")]
use super::cl100k_family::batch_masks;
#[cfg(target_arch = "x86_64")]
use super::cl100k_family::batch_masks_x86;
use super::mask::{MaskScheme, MaskState};
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, letter_end_at, scan_letters_from,
    scan_newlines, scan_numbers_max3, scan_other_from,
};
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{self, CharClass};

pub(crate) struct Olmo3Scheme;

impl MaskScheme for Olmo3Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::ClassTable::get();
        batch_masks(bytes, scan, true, move |cp| ct.class_of(cp))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // Class-table LazyLock resolved once per batch; the extended
        // path's per-char classify is then a bare slice index.
        let ct = unicode::ClassTable::get();
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan, true, move |cp| ct.class_of(cp)) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512 detected at runtime),
/// iteration runs the shared cl100k-family mask scanner (see
/// `cl100k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastOlmo3Pretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastOlmo3Pretokenizer<'a> {
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

impl<'a> Iterator for FastOlmo3Pretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<Olmo3Scheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastOlmo3Pretokenizer, Olmo3Scheme);

/// Whitespace-led token starting at `start`, i.e. the alternatives
/// `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, in that priority.
/// Precondition: the letter-prefix (`[^\r\n\p{L}\p{N}]?\p{L}+`) and
/// space+punct (` ?[^\s\p{L}\p{N}]+...`) alternatives were ruled out.
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
            if unicode::class_of(cp) == CharClass::Whitespace {
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

    // Hot path 1: ASCII letter — `\p{L}+` with empty prefix
    if is_letter(b0) {
        return scan_letters_from(bytes, pos + 1);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            return scan_letters_from(bytes, pos + 2); // " word"
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let p = scan_other_from(bytes, pos + 2);
            return scan_newlines(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        match unicode::class_of(cp) {
            CharClass::Letter => return scan_letters_from(bytes, p1),
            CharClass::Whitespace => return ws_token_end(bytes, pos),
            CharClass::Number => return pos + 1,
            CharClass::Other => {
                let p = scan_other_from(bytes, p1);
                return scan_newlines(bytes, p);
            }
        }
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        let class = unicode::class_of(cp);
        if class == CharClass::Letter {
            return scan_letters_from(bytes, p0);
        }
        if class == CharClass::Number {
            return scan_numbers_max3(bytes, p0, 1);
        }
        // Any non-letter/number char except \r\n may prefix a letter run
        if let Some(p) = letter_end_at(bytes, p0) {
            return scan_letters_from(bytes, p);
        }
        if class == CharClass::Whitespace {
            return ws_token_end(bytes, pos);
        }
        let p = scan_other_from(bytes, p0);
        return scan_newlines(bytes, p);
    }

    // ASCII digit: `\p{N}{1,3}`
    if is_digit(b0) {
        return scan_numbers_max3(bytes, pos + 1, 1);
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
        // Not a contraction: `'` can still prefix a letter run
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        let p = scan_other_from(bytes, pos + 1);
        return scan_newlines(bytes, p);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some(p) = letter_end_at(bytes, pos + 1) {
            return scan_letters_from(bytes, p);
        }
        return ws_token_end(bytes, pos);
    }

    // ASCII punctuation/symbol
    if let Some(p) = letter_end_at(bytes, pos + 1) {
        return scan_letters_from(bytes, p);
    }
    let p = scan_other_from(bytes, pos + 1);
    scan_newlines(bytes, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// The Olmo3/dolma2 pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const OLMO3_REF_REGEX: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(OLMO3_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastOlmo3Pretokenizer::new(s.as_bytes())
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
    fn olmo3_small_cases() {
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
            "12345678901",
            " 123",
            " 1234",
            "  123",
            "3rd",
            "abc1234def",
            "3.14159",
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
            " ١٢٣٤٥",
            "e\u{301}f",
            "日本語のテキスト",
            " 日本語",
            "1٢3x",
            "1٢34",
            "tab\tsep\tvals",
            "\x0bword",
            "a\u{2028}b",
            "a\u{2028}\n",
            "price: $5.99!",
            "'ſ",
            "it'ſ fine",
        ];
        for case in cases {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    #[test]
    fn olmo3_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt_prefix(SIZE);
        let text = std::str::from_utf8(&input).unwrap();
        eprintln!(
            "Testing olmo3 fast pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(OLMO3_REF_REGEX).unwrap();
        let mut fast_iter = FastOlmo3Pretokenizer::new(&input);
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

    /// Full-OWT (~12 GB) comparison against the reference regex, parallelized
    /// with rayon over ~32 MB chunks cut at newline boundaries. Splitting is
    /// safe because both sides tokenize the identical chunk. Run with:
    /// `cargo test --release olmo3_matches_regex_owt_full -- --ignored --nocapture`
    #[test]
    #[ignore = "reads the full ~12 GB OWT file; run explicitly in release mode"]
    fn olmo3_matches_regex_owt_full() {
        use rayon::prelude::*;

        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let file = std::fs::File::open(&path).expect("Could not open ~/data/owt_train.txt");
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let bytes: &[u8] = &mmap;

        const CHUNK: usize = 32 * 1024 * 1024;
        let mut boundaries = vec![0usize];
        while *boundaries.last().unwrap() < bytes.len() {
            let target = (*boundaries.last().unwrap() + CHUNK).min(bytes.len());
            let end = if target == bytes.len() {
                target
            } else {
                // Cut at the next newline (ASCII, so always a UTF-8 boundary).
                match memchr::memchr(b'\n', &bytes[target..]) {
                    Some(off) => target + off + 1,
                    None => bytes.len(),
                }
            };
            boundaries.push(end);
        }
        eprintln!(
            "Comparing olmo3 fast pretokenizer vs regex on {:.2} GB in {} chunks",
            bytes.len() as f64 / 1e9,
            boundaries.len() - 1
        );

        let total_tokens: usize = boundaries
            .par_windows(2)
            .map(|w| {
                let chunk = &bytes[w[0]..w[1]];
                let text = std::str::from_utf8(chunk).expect("chunk is not valid UTF-8");
                let re = fancy_regex::Regex::new(OLMO3_REF_REGEX).unwrap();
                let mut fast_iter = FastOlmo3Pretokenizer::new(chunk);
                let mut re_iter = re.find_iter(text);
                let mut count = 0usize;
                loop {
                    match (fast_iter.next(), re_iter.next()) {
                        (Some(fast_tok), Some(re_match)) => {
                            let re_match = re_match.expect("regex match error");
                            let fast_str = String::from_utf8_lossy(fast_tok.0);
                            let re_str = &text[re_match.start()..re_match.end()];
                            assert_eq!(
                                fast_str, re_str,
                                "Mismatch in chunk at byte {} (chunk offset {})",
                                w[0] + re_match.start(),
                                re_match.start()
                            );
                        }
                        (None, None) => break,
                        (fast, re) => panic!(
                            "Token count mismatch in chunk starting at byte {}: fast={:?} regex={:?}",
                            w[0],
                            fast.map(|t| String::from_utf8_lossy(t.0).into_owned()),
                            re.map(|m| m.unwrap().as_str().to_string()),
                        ),
                    }
                    count += 1;
                }
                count
            })
            .sum();
        eprintln!("All {total_tokens} tokens match across the full file.");
    }
}
