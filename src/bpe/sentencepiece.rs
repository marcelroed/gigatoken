use crate::bpe::bpe_merge_symbols_ranked;
use crate::token::TokenId;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::sync::Arc;

/// SentencePiece uses U+2581 (▁) as a space marker.
const SENTENCEPIECE_SPACE: char = '\u{2581}';

/// A tokenizer that mirrors SentencePiece BPE with `byte_fallback`.
///
/// This struct holds the immutable model data (vocab, merges, added tokens).
/// For encoding, create an [`Encoder`] which processes text through the
/// normalize → character init → BPE merge pipeline.
pub struct SentencePieceBPE {
    /// Merges with explicit rank: `(a, b) → (merged, rank)`.
    pub(crate) merges: HashMap<(TokenId, TokenId), (TokenId, u32), FxBuildHasher>,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    /// Maps byte sequences → token IDs (for character lookup).
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
    /// Token ID for each byte value (0x00–0xFF) via `<0xHH>` fallback tokens.
    pub(crate) byte_fallback_ids: [TokenId; 256],
    /// Special/added tokens that are matched in text before encoding.
    pub(crate) added_tokens: Vec<(String, TokenId)>,
}

impl std::fmt::Debug for SentencePieceBPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SentencePieceBPE {{ vocab_size: {}, merges_count: {} }}",
            self.vocab.len(),
            self.merges.len(),
        )
    }
}

impl SentencePieceBPE {
    /// Normalise input the way the Llama tokenizer does:
    /// prepend ▁ and replace all spaces with ▁.
    pub fn normalize(input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(input.len() + 3);
        out.push(SENTENCEPIECE_SPACE);
        for ch in input.chars() {
            if ch == ' ' {
                out.push(SENTENCEPIECE_SPACE);
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// Create an encoder for this model.
    pub fn encoder(&self) -> Encoder<'_> {
        Encoder { model: self }
    }

    /// Convenience: encode a single text (creates a temporary encoder).
    pub fn encode_raw(&self, input: &str) -> Vec<TokenId> {
        self.encoder().encode_raw(input)
    }

    /// Decode token IDs back to a UTF-8 string.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut raw = Vec::new();
        for &t in tokens {
            let idx: usize = t.into();
            if idx < self.vocab.len() {
                raw.extend_from_slice(&self.vocab[idx]);
            }
        }
        let text = String::from_utf8_lossy(&raw);
        let mut out: Vec<u8> = text.replace(SENTENCEPIECE_SPACE, " ").into_bytes();
        if out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }
}

/// An encoder that holds a reference to the model.
/// Create one per thread for parallel encoding.
pub struct Encoder<'a> {
    model: &'a SentencePieceBPE,
}

impl<'a> Encoder<'a> {
    /// Encode raw (un-normalized) text with added-token splitting.
    pub fn encode_raw(&mut self, input: &str) -> Vec<TokenId> {
        if self.model.added_tokens.is_empty() {
            let normalized = SentencePieceBPE::normalize(input);
            return self.encode_normalized(&normalized);
        }

        let mut result = Vec::new();
        let mut remaining = input;

        while !remaining.is_empty() {
            let mut best: Option<(usize, &str, TokenId)> = None;
            for (content, id) in &self.model.added_tokens {
                if let Some(pos) = remaining.find(content.as_str())
                    && (best.is_none() || pos < best.unwrap().0) {
                        best = Some((pos, content, *id));
                    }
            }

            match best {
                Some((pos, content, id)) => {
                    if pos > 0 {
                        let normalized = SentencePieceBPE::normalize(&remaining[..pos]);
                        result.extend(self.encode_normalized(&normalized));
                    }
                    result.push(id);
                    remaining = &remaining[pos + content.len()..];
                }
                None => {
                    let normalized = SentencePieceBPE::normalize(remaining);
                    result.extend(self.encode_normalized(&normalized));
                    break;
                }
            }
        }

        result
    }

    /// Encode already-normalized text: character init → BPE merge.
    pub fn encode_normalized(&mut self, input: &str) -> Vec<TokenId> {
        let mut symbols = Vec::new();
        for ch in input.chars() {
            let mut buf = [0u8; 4];
            let ch_bytes = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.model.vocab_inv.get(ch_bytes) {
                symbols.push(id);
            } else {
                for &b in ch_bytes {
                    symbols.push(self.model.byte_fallback_ids[b as usize]);
                }
            }
        }
        bpe_merge_symbols_ranked(&self.model.merges, &mut symbols);
        symbols
    }
}
