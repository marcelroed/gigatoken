//! Fast pretokenizer for the Nemotron-3 regex (nvidia Nemotron-3 family):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme without contraction suffixes and with single-char
//! `\p{N}` digit tokens. See `o200k_family` (`CONTRACTIONS = false`,
//! `DIGITS3 = false`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family;
use crate::pretokenize::Pretoken;

pub(crate) struct NemotronScheme;

impl MaskScheme for NemotronScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, false>(bytes, pos)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, false>(bytes, scan)
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastNemotronPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastNemotronPretokenizer<'a> {
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

impl<'a> Iterator for FastNemotronPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<NemotronScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastNemotronPretokenizer, NemotronScheme);

#[cfg(test)]
mod tests {
    use super::*;

    /// The Nemotron pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const NEMOTRON_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(NEMOTRON_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastNemotronPretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// The o200k small-case list applies verbatim (contraction cases just
    /// tokenize differently, which the reference regex reflects).
    #[test]
    fn nemotron_small_cases() {
        for case in crate::pretokenize::fast::o200k::tests::SMALL_CASES {
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
                regex_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }
}
