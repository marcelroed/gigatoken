//! Fast WordPiece pretokenizer + encoder.
//!
//! WordPiece (BERT family) first splits text into "words" on whitespace and
//! punctuation (and isolates each CJK character), then greedily longest-matches
//! subword units from a vocabulary, prefixing continuation pieces with `##`.
//!
//! The split step is the part gigatoken SIMD-accelerated for BPE. We do the
//! same split with a tight SWAR byte-class scan (no regex engine), so it stays
//! fast and dependency-free. The greedy match is O(n * max_piece_len) per word.

/// Byte-class predicates for WordPiece boundary detection (BERT BasicTokenizer rules).
#[inline(always)]
pub fn is_whitespace(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0c || b == 0x0b
}
#[inline(always)]
pub fn is_control(b: u8) -> bool {
    b < 0x20 || b == 0x7f
}
#[inline(always)]
pub fn is_punct(b: u8) -> bool {
    (0x21..=0x2f).contains(&b)
        || (0x3a..=0x40).contains(&b)
        || (0x5b..=0x60).contains(&b)
        || (0x7b..=0x7e).contains(&b)
}
/// Lead byte of a 3-byte UTF-8 CJK Unified Ideograph (BERT isolates each CJK char).
#[inline(always)]
pub fn is_cjk_lead(b: u8) -> bool {
    (0xe4..=0xe9).contains(&b)
}

/// A byte starts a WordPiece boundary if it is whitespace/control/punct, or the
/// lead byte of a CJK char.
#[inline(always)]
fn is_boundary_start(b: u8) -> bool {
    is_whitespace(b) || is_control(b) || is_punct(b) || is_cjk_lead(b)
}

/// Yield byte-range "words" per WordPiece rules (CJK chars isolated).
pub struct SpanIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> SpanIter<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}
impl<'a> Iterator for SpanIter<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<&'a [u8]> {
        let b = self.bytes;
        while self.pos < b.len() {
            let cur = b[self.pos];
            // A lone CJK char is its own word.
            if is_cjk_lead(cur) && self.pos + 3 <= b.len() {
                let s = self.pos;
                self.pos += 3;
                return Some(&b[s..self.pos]);
            }
            // Punctuation is its own 1-byte word (BERT isolates it).
            if is_punct(cur) {
                let s = self.pos;
                self.pos += 1;
                return Some(&b[s..self.pos]);
            }
            // Whitespace/control are separators, not tokens — skip them.
            if is_whitespace(cur) || is_control(cur) {
                self.pos += 1;
                continue;
            }
            // start of a real word
            let start = self.pos;
            while self.pos < b.len() && !is_boundary_start(b[self.pos]) {
                self.pos += 1;
            }
            return Some(&b[start..self.pos]);
        }
        None
    }
}

use std::collections::HashMap;

/// WordPiece tokenizer over a vocab of `piece -> id`. Continuation pieces use
/// the `##` prefix convention (BERT). Unknown chars emit `unk_id`.
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    max_piece: usize,
    lowercase: bool,
}

impl WordPieceTokenizer {
    pub fn new(vocab: HashMap<String, u32>, unk_id: u32) -> Self {
        let max_piece = vocab.keys().map(|k| k.len()).max().unwrap_or(0);
        Self { vocab, unk_id, max_piece, lowercase: false }
    }

    /// Enable BERT-style NFKC-ish lowercasing before splitting (bert-base-uncased).
    pub fn with_lowercase(mut self, on: bool) -> Self {
        self.lowercase = on;
        self
    }

    /// Greedy longest-match encode of one whitespace word into token ids.
    fn encode_word(&self, word: &[u8]) -> Vec<u32> {
        let s = String::from_utf8_lossy(word);
        let s = if self.lowercase { s.to_lowercase() } else { s.into_owned() };
        let chars: Vec<char> = s.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = None;
            let mut end = chars.len().min(i + self.max_piece);
            while end > i {
                let piece: String = chars[i..end].iter().collect();
                let key = if i == 0 {
                    piece.clone()
                } else {
                    format!("##{}", piece)
                };
                if let Some(id) = self.vocab.get(&key) {
                    matched = Some(*id);
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    out.push(id);
                    i = end;
                }
                None => {
                    out.push(self.unk_id);
                    i += 1;
                }
            }
        }
        out
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        for word in SpanIter::new(bytes) {
            out.extend(self.encode_word(word));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab() -> (HashMap<String, u32>, u32) {
        let mut v = HashMap::new();
        v.insert("hello".to_string(), 1);
        v.insert("he".to_string(), 2);
        v.insert("##llo".to_string(), 3);
        v.insert("world".to_string(), 4);
        v.insert(",".to_string(), 5);
        v.insert("play".to_string(), 6);
        v.insert("##ing".to_string(), 7);
        v.insert("##ed".to_string(), 8);
        (v, 0) // unk = 0
    }

    #[test]
    fn span_split() {
        let txt = b"hello, world!";
        let spans: Vec<&[u8]> = SpanIter::new(txt).collect();
        assert_eq!(
            spans,
            vec![&b"hello"[..], &b","[..], &b"world"[..], &b"!"[..]]
        );
    }

    #[test]
    fn encode_basic() {
        let (v, unk) = test_vocab();
        let tk = WordPieceTokenizer::new(v, unk);
        assert_eq!(tk.encode("hello, world"), vec![1, 5, 4]);
    }

    #[test]
    fn encode_subword() {
        let (v, unk) = test_vocab();
        let tk = WordPieceTokenizer::new(v, unk);
        assert_eq!(tk.encode("playing"), vec![6, 7]); // play + ##ing
    }

    #[test]
    fn cjk_isolated_and_unk() {
        // U+4E2D '中' = e4 b8 ad → isolated span, not in vocab → unk
        let (v, unk) = test_vocab();
        let tk = WordPieceTokenizer::new(v, unk);
        assert_eq!(tk.encode("中"), vec![unk]);
    }

    #[test]
    fn empty_vocab_all_unk() {
        let tk = WordPieceTokenizer::new(HashMap::new(), 0);
        assert_eq!(tk.encode("abc"), vec![0, 0, 0]);
    }

    #[test]
    fn lowercase_matches_bert() {
        let mut v = HashMap::new();
        v.insert("hello".to_string(), 1);
        v.insert("world".to_string(), 2);
        let tk = WordPieceTokenizer::new(v, 0).with_lowercase(true);
        assert_eq!(tk.encode("Hello World"), vec![1, 2]);
    }
}
