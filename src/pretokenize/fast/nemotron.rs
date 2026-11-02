//! Fast pretokenizer for the Nemotron-3 regex (nvidia Nemotron-3 family):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//! and its right-aligned digit variant, which replaces `\p{N}` with
//! `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}`.
//!
//! The o200k scheme without contraction suffixes. See `o200k_family`
//! (`CONTRACTIONS = false`, `DIGITS` = `DIGITS_1` or `DIGITS_3_RIGHT`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family::{self, DIGITS_1, DIGITS_3_RIGHT};
use crate::pretokenize::Pretoken;

pub(crate) struct NemotronScheme<const DIGITS: u8>;

impl<const DIGITS: u8> MaskScheme for NemotronScheme<DIGITS> {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, DIGITS, true, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, DIGITS, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, false, DIGITS, true, false>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`. Used through its two aliases.
pub struct NemotronPretokenizer<'a, const DIGITS: u8> {
    bytes: &'a [u8],
    state: MaskState,
}

/// Nemotron-3: every `\p{N}` char is its own token.
pub type FastNemotronPretokenizer<'a> = NemotronPretokenizer<'a, DIGITS_1>;

/// Nemotron with right-aligned digit groups: "1234567" is `1|234|567`.
pub type FastNemotronRightAlignedPretokenizer<'a> = NemotronPretokenizer<'a, DIGITS_3_RIGHT>;

impl<'a, const DIGITS: u8> NemotronPretokenizer<'a, DIGITS> {
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

impl<'a, const DIGITS: u8> Iterator for NemotronPretokenizer<'a, DIGITS> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<NemotronScheme<DIGITS>>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastNemotronPretokenizer, NemotronScheme<DIGITS_1>);
super::impl_mask_pretoken_spans!(
    FastNemotronRightAlignedPretokenizer,
    NemotronScheme<DIGITS_3_RIGHT>
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::{PretokenSpans, SpanBatch};
    use std::sync::OnceLock;

    /// The Nemotron pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const NEMOTRON_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// Tokens under the reference regex: Nemotron's, or with `right_aligned`
    /// its right-aligned digit variant.
    fn regex_tokens(s: &str, right_aligned: bool) -> Vec<String> {
        static RES: [OnceLock<fancy_regex::Regex>; 2] = [const { OnceLock::new() }; 2];
        let re = RES[right_aligned as usize].get_or_init(|| {
            let digits = if right_aligned {
                r"|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}|"
            } else {
                r"|\p{N}|"
            };
            fancy_regex::Regex::new(&NEMOTRON_REF_REGEX.replace(r"|\p{N}|", digits)).unwrap()
        });
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastNemotronPretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// Right-aligned tokens from the iterator, checked against the chunked
    /// span fill (the encode path).
    #[track_caller]
    fn right_aligned_tokens(s: &str) -> Vec<String> {
        let lossy = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        let mut scanner = FastNemotronRightAlignedPretokenizer::new(s.as_bytes());
        let mut batch = SpanBatch::new();
        let mut filled = Vec::new();
        loop {
            let n = scanner.fill_spans_keyed(&mut batch, &|_| {});
            if n == 0 {
                break;
            }
            // SAFETY: entries 0..n were written by the fill just above.
            filled.extend((0..n).map(|i| lossy(unsafe { batch.span(i) })));
        }
        let iterated: Vec<String> = FastNemotronRightAlignedPretokenizer::new(s.as_bytes())
            .map(|t| lossy(t.0))
            .collect();
        assert_eq!(filled, iterated, "span fill vs iterator on {s:?}");
        iterated
    }

    /// The o200k small-case list applies verbatim (contraction cases just
    /// tokenize differently, which the reference regex reflects).
    #[test]
    fn nemotron_small_cases() {
        for case in crate::pretokenize::fast::o200k::tests::SMALL_CASES {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case, false),
                "Mismatch on case {case:?}"
            );
            assert_eq!(
                right_aligned_tokens(case),
                regex_tokens(case, true),
                "Right-aligned mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes,
    /// compared against the reference regex.
    #[test]
    fn nemotron_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'],      // lower/caseless
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        let mut rng = StdRng::seed_from_u64(0x93E3_5EEE);
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
                regex_tokens(&s, false),
                "Mismatch on round {round}, case {s:?}"
            );
            assert_eq!(
                right_aligned_tokens(&s),
                regex_tokens(&s, true),
                "Right-aligned mismatch on round {round}, case {s:?}"
            );
        }
    }

    /// Right-aligned digit runs of every length at every offset against the
    /// 64-byte batch grid, and non-ASCII runs inside and across batches.
    #[test]
    fn right_aligned_digit_runs() {
        let mut cases = vec![];
        for n in 1..130 {
            for prefix in 0..70 {
                cases.push(format!("{} {} suffix", "a".repeat(prefix), "1".repeat(n)));
            }
        }
        let non_ascii = ["١٢٣٤٥٦٧", "１２３４５", "ⅧⅡⅣⅦⅤ", "½¼¾⅓⅔", "1٢３Ⅷ½4٥6"];
        for text in non_ascii {
            cases.push(text.to_string());
            cases.push(format!("{}{text}{}", "x".repeat(60), "y".repeat(70)));
        }
        for case in &cases {
            assert_eq!(
                right_aligned_tokens(case),
                regex_tokens(case, true),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// A 100k-digit run splits in linear time: only a run's first token
    /// counts the run.
    #[test]
    fn right_aligned_long_run() {
        for digit in ["1", "٢"] {
            let tokens = right_aligned_tokens(&digit.repeat(100_001));
            assert_eq!(tokens[0], digit.repeat(2));
            assert!(tokens[1..].iter().all(|t| *t == digit.repeat(3)));
        }
    }
}
