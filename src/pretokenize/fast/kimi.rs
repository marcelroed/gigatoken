//! Fast pretokenizer for the Kimi regex (moonshotai Kimi-K2 family, from
//! `tokenization_kimi.py`):
//! `[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme with a leading `[\p{Han}]+` alternative, Han excluded
//! from the letter brackets, and no `/` in the absorbed punct tail. See
//! `o200k_family` (`CONTRACTIONS = true`, `DIGITS = DIGITS_3`,
//! `SLASH = false`, `HAN = true`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family::{self, DIGITS_3};
use crate::pretokenize::Pretoken;

pub(crate) struct KimiScheme;

impl MaskScheme for KimiScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<true, DIGITS_3, false, true>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<true, DIGITS_3, false, true>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, true, DIGITS_3, false, true>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastKimiPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastKimiPretokenizer<'a> {
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

impl<'a> Iterator for FastKimiPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<KimiScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastKimiPretokenizer, KimiScheme);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Read;

    /// The Kimi pattern verbatim (from `tokenization_kimi.py`; only greedy
    /// quantifiers, so it runs directly under fancy-regex, which shares
    /// regex-syntax's `&&` class intersection).
    const KIMI_REF_REGEX: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(KIMI_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastKimiPretokenizer::new(s.as_bytes())
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

    /// Han-specific cases on top of the shared o200k list: run splits at
    /// script edges, the `[\r\n]*`-only tail (no `/` absorption), Han
    /// numerals inside digit groups, and Han symbols inside punct runs.
    pub(crate) const HAN_CASES: &[&str] = &[
        "中文",
        "中文模型",
        " 中文",
        "  中文",
        "\t中文",
        "\n中文",
        "!中文",
        "中English文",
        "abc中文def",
        "ABC中文",
        "中文ABC",
        "中'se",
        "中's",
        "中'S x",
        "中文's",
        "日本語のテキスト漢字かな",
        "漢字とひらがな",
        "中。文",
        "中，文！",
        "中文123",
        "123中文",
        "1〇2",
        "〇〇",
        "中〇文",
        "1〇〇〇〇",
        "1234〇",
        " 〇",
        "〇1",
        "⼀⼁⼂",
        "!⼀x",
        " ⼀",
        "中⼀文",
        "a⼀b",
        "々中",
        "中々",
        "〆切",
        "㐀㿿",
        "𠀀𠀁",
        "中\u{16FF0}文",
        "!\u{16FF0}",
        ".\n//x",
        "!\n/",
        "!\n/\n/x",
        "\n/",
        "//\n/",
        "x/\n",
        "x\n/",
        "}\n///doc",
        "*/\n/**",
        "`\n//! bindings",
        "path/to/file",
        "http://x.com/path",
        "中\n文",
        "中\r\n文",
        "中 文",
        "中  文",
        "中\n\n文",
        "。\n中",
        "中。\n\n/x",
    ];

    #[test]
    fn kimi_small_cases() {
        for case in crate::pretokenize::fast::o200k::tests::SMALL_CASES {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
        for case in HAN_CASES {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes
    /// (including the Han classes), compared against the reference regex.
    #[test]
    fn kimi_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', 'ひ', 'カ'],   // lower/caseless (non-Han)
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers (non-Han)
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
            &['s', 't', 'm', 'd', 'l', 'v', 'r', 'e', 'S', 'T', 'L'], // suffix letters
            &['中', '文', '日', '本', '語', '々', '〆', '㐀', '𠀀'], // Han letters
            &['〇', '〡', '〢', '㆒'],                          // Han numerals (Nl)
            &['⼀', '⼁', '⺀', '\u{16FF0}'],                   // Han symbols/marks (So/Mc)
        ];
        let mut rng = StdRng::seed_from_u64(0x93E3_5EEC);
        for round in 0..6000 {
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
    fn kimi_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt_prefix(SIZE);
        let text = std::str::from_utf8(&input).unwrap();
        eprintln!(
            "Testing kimi fast pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(KIMI_REF_REGEX).unwrap();
        let mut fast_iter = FastKimiPretokenizer::new(&input);
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
