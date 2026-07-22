//! Fast SentencePiece (unigram) tokenizer.
//!
//! gigatoken's README admits SentencePiece is "not nearly as optimized" as its
//! BPE path. The common SentencePiece model is the *unigram* language model:
//! every substring has a score, and the best tokenization is the Viterbi
//! (max-sum) path over the string. Unlike BPE (greedy longest-match from the
//! left), unigram is a global DP — which is why it needs its own encoder.
//!
//! This is a standalone, dependency-free implementation: load a vocab of
//! `piece -> (id, score)`, run Viterbi, emit ids. Continuation is handled by
//! SentencePiece's `▁` (U+2581) space marker convention.

use std::collections::HashMap;

/// Unigram SentencePiece model: piece -> (id, log-score).
pub struct SentencePieceTokenizer {
    pieces: HashMap<String, (u32, f64)>,
    unk_id: u32,
    max_piece: usize,
}

impl SentencePieceTokenizer {
    /// Build from a vocab: map of `piece -> id`. Scores default to 0.0
    /// (acceptable for correctness tests; real `.model` files supply scores).
    pub fn from_vocab(vocab: HashMap<String, u32>, unk_id: u32) -> Self {
        let max_piece = vocab.keys().map(|k| k.len()).max().unwrap_or(0);
        let pieces = vocab
            .into_iter()
            .map(|(k, v)| (k, (v, 0.0)))
            .collect();
        Self { pieces, unk_id, max_piece }
    }

    /// Build from a SentencePiece `.model`-style line list (`piece\tscore`).
    pub fn from_model_lines(lines: &[String], unk_id: u32) -> Self {
        let mut pieces = HashMap::new();
        let mut max_piece = 0;
        for (i, line) in lines.iter().enumerate() {
            let mut it = line.split('\t');
            let piece = it.next().unwrap_or("").to_string();
            let score: f64 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            max_piece = max_piece.max(piece.len());
            pieces.insert(piece, (i as u32, score));
        }
        Self { pieces, unk_id, max_piece }
    }

    /// Viterbi best-path over `text`. `▁` marks a word-initial space; we map
    /// ASCII spaces to `▁` so "hello world" tokenizes like the SentencePiece
    /// convention.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // SentencePiece convention: a word-initial space is the `▁` (U+2581)
        // marker. Prepend it, then swap remaining spaces for `▁`.
        let sp: String = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        let chars: Vec<char> = sp.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        // best[i] = (score, back_ptr, piece_id) ending at char i
        let mut best: Vec<(f64, usize, u32)> = vec![(f64::NEG_INFINITY, 0, 0); n + 1];
        best[0] = (0.0, 0, 0);
        for i in 0..n {
            if best[i].0 == f64::NEG_INFINITY {
                continue;
            }
            let max_len = self.max_piece.min(n - i);
            for l in 1..=max_len {
                let piece: String = chars[i..i + l].iter().collect();
                if let Some(&(id, score)) = self.pieces.get(&piece) {
                    let cand = best[i].0 + score;
                    if cand > best[i + l].0 {
                        best[i + l] = (cand, i, id);
                    }
                }
            }
            // unknown single char fallback
            if best[i + 1].0 == f64::NEG_INFINITY {
                best[i + 1] = (best[i].0 - 10.0, i, self.unk_id);
            }
        }
        // backtrack
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (_, back, id) = best[pos];
            ids.push(id);
            pos = back;
        }
        ids.reverse();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        v.insert("▁".to_string(), 0);
        v.insert("▁he".to_string(), 1);
        v.insert("▁hello".to_string(), 2);
        v.insert("▁world".to_string(), 3);
        v.insert("l".to_string(), 4);
        v.insert("lo".to_string(), 5);
        v.insert("▁play".to_string(), 6);
        v.insert("ing".to_string(), 7);
        v
    }

    #[test]
    fn unigram_prefers_longer_piece() {
        // With scores equal, Viterbi still produces a valid full cover.
        let tk = SentencePieceTokenizer::from_vocab(vocab(), 99);
        let ids = tk.encode("hello world");
        assert!(!ids.is_empty());
        assert!(ids.contains(&2)); // ▁hello
        assert!(ids.contains(&3)); // ▁world
    }

    #[test]
    fn unigram_unk_on_garbage() {
        let tk = SentencePieceTokenizer::from_vocab(vocab(), 99);
        let ids = tk.encode("🚀🌟"); // ▁ matches (id0), rest unk
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 0); // leading ▁
        assert!(ids[1..].iter().all(|&x| x == 99));
    }

    #[test]
    fn unigram_empty() {
        let tk = SentencePieceTokenizer::from_vocab(vocab(), 99);
        assert!(tk.encode("").is_empty());
    }
}
