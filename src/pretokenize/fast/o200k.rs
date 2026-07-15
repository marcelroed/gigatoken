//! Fast pretokenizer for the o200k_base regex (GPT-4o, gpt-oss):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! See `o200k_family` for the shared scalar walker and mask-scanner
//! boundary algebra (`CONTRACTIONS = true`, `DIGITS3 = true`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family;
use crate::pretokenize::Pretoken;

pub(crate) struct O200kScheme;

impl MaskScheme for O200kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<true, true>(bytes, pos)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<true, true>(bytes, scan)
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastO200kPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastO200kPretokenizer<'a> {
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

impl<'a> Iterator for FastO200kPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<O200kScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastO200kPretokenizer, O200kScheme);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Read;

    /// The o200k pattern verbatim — no possessive quantifiers, so it runs
    /// directly under fancy-regex.
    const O200K_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(O200K_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastO200kPretokenizer::new(s.as_bytes())
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

    pub(crate) const SMALL_CASES: &[&str] = &[
        "hello",
        "Hello",
        "HELLO",
        "HeLLo",
        "camelCase",
        "PascalCase",
        "HTTPResponse",
        "HTTPresponse",
        "parseHTMLDocument",
        "XMLHttpRequest",
        "aB",
        "aBc",
        "ABc",
        "ABCdef GHIjkl",
        " hello",
        " Hello World",
        "hello world",
        "  hello",
        "\thello",
        "\tHello",
        "\nhello",
        "\n\nHello",
        "!hello",
        "!Hello",
        "!!hello",
        "?!x",
        "don't",
        "DON'T",
        "Don'T",
        "don'ts",
        "can'ts more",
        "they'LL go",
        "it'S he'Ll",
        "we'Ve THEY'RE",
        "x'll'd",
        "don't's",
        "o'clock",
        "don'x",
        "x'lm",
        "x'm'm",
        "'sound",
        "'Sound",
        "'lx",
        "'hello",
        " 'hello",
        " 's",
        " 'S",
        "x'0",
        "3's",
        "3'ts",
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
        "a/b",
        "a//b",
        "http://x.com/path",
        ".\n//x",
        "!\n/",
        "!\n/\n/x",
        "\n/",
        "//\n/",
        "x/\n",
        "x\n/",
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
        "Café",
        "CAFÉ",
        "cafÉ",
        " café",
        "\u{a0}word",
        "voilà ¡hola!",
        "ΑΒΓδε",
        "αβΓΔ",
        "Привет Мир",
        "ПРИВЕТ мир",
        "ẞßẞ",
        "ǅungla ǄUNGLA ǆungla",
        "١٢٣٤٥",
        "1٢3x",
        "١٢٣٤٥٦٧",
        "tab\tsep\tvals",
        "\x0bword",
        "a\u{2028}b",
        "a\u{2028}\n",
        "price: $5.99!",
        "'ſ",
        "x'ſ fine",
        "日本語のテキスト",
        " 日本語",
        "日本語ABC",
        "abc日本語Def",
        "e\u{301}f",
        "cafe\u{301} de\u{301}composed",
        "\u{301}leading mark",
        "\u{301}\u{301}two marks",
        " \u{301}abc",
        "\t\u{301}abc",
        "!\u{301}",
        "!\u{301}!",
        "!!\u{301}x",
        "!!\u{301}X",
        "1\u{301}2",
        "'\u{301}s",
        "A\u{301}B",
        "a\u{301}B",
        "x\u{301}'s",
        "деВНАгарІ",
        "देवनागरी में परीक्षण",
        "עִבְרִית נִקּוּד",
        "الْعَرَبِيَّة",
        "a\u{20dd}b",
        " \u{20dd}",
        "\u{200b}\u{301}x",
    ];

    #[test]
    fn o200k_small_cases() {
        for case in SMALL_CASES {
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
    fn o200k_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'],      // lower/caseless
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
            &['s', 't', 'm', 'd', 'l', 'v', 'r', 'e', 'S', 'T', 'L'], // suffix letters
        ];
        let mut rng = StdRng::seed_from_u64(0x93E3_5EED);
        for round in 0..3000 {
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
    fn o200k_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt_prefix(SIZE);
        let text = std::str::from_utf8(&input).unwrap();
        eprintln!(
            "Testing o200k fast pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(O200K_REF_REGEX).unwrap();
        let mut fast_iter = FastO200kPretokenizer::new(&input);
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
