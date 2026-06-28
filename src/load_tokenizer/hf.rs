//! Load HuggingFace tokenizer.json files.
//!
//! Supports two styles:
//! - SentencePiece BPE with `byte_fallback=true` (e.g. Llama) → [`load_hf_sentencepiece`]
//! - ByteLevel BPE without byte_fallback (e.g. GPT-2) → [`load_hf_bpe`]

use crate::bpe::{self, SentencePieceBPE};
use crate::token::TokenId;
use eyre::{Context, Result, ensure};
use rustc_hash::FxBuildHasher;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// JSON schema (only the fields we need)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

// ---------------------------------------------------------------------------
// Token string → raw bytes conversion
// ---------------------------------------------------------------------------

/// Convert a HuggingFace vocab string to raw bytes.
///
/// - Byte-fallback tokens `<0xHH>` → the single byte.
/// - Everything else → its UTF-8 bytes (▁ is kept as-is).
fn token_str_to_bytes(s: &str) -> Vec<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }
    s.as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a HuggingFace `tokenizer.json` that uses SentencePiece-style BPE with
/// byte fallback (e.g. Llama 2 / TinyLlama).
///
/// Returns a [`SentencePieceBPE`] that preserves the original HF token IDs.
pub fn load_hf_sentencepiece(path: impl AsRef<Path>) -> Result<SentencePieceBPE> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let tj: TokenizerJson = sonic_rs::from_slice(&data)
        .with_context(|| format!("Failed to parse {}", path.display()))?;

    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        tj.model.byte_fallback,
        "Only byte_fallback tokenizers are supported"
    );

    let hf_vocab = &tj.model.vocab;
    let hf_merges = &tj.model.merges;

    // --- Build vocab (preserving original HF IDs) ----------------------------

    let max_id = hf_vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(hf_vocab.len(), FxBuildHasher);

    // Insert byte-fallback tokens first, then character tokens, so that
    // character tokens win in vocab_inv when both map to the same bytes.
    let mut byte_fallback_entries = Vec::new();
    let mut other_entries = Vec::new();
    for (tok_str, &id) in hf_vocab {
        if tok_str.len() == 6 && tok_str.starts_with("<0x") && tok_str.ends_with('>') {
            byte_fallback_entries.push((tok_str, id));
        } else {
            other_entries.push((tok_str, id));
        }
    }
    for (tok_str, id) in byte_fallback_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }
    for (tok_str, id) in other_entries {
        let bytes: Arc<[u8]> = token_str_to_bytes(tok_str).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    // --- Extract byte-fallback token IDs -------------------------------------

    let mut byte_fallback_ids = [TokenId::from(0u32); 256];
    for byte_val in 0u16..=255 {
        let key = format!("<0x{:02X}>", byte_val);
        let &id = hf_vocab
            .get(&key)
            .ok_or_else(|| eyre::eyre!("Missing byte fallback token {key}"))?;
        byte_fallback_ids[byte_val as usize] = TokenId::from(id);
    }

    // --- Build merge table (with explicit ranks) -----------------------------

    let mut merges: HashMap<(TokenId, TokenId), (TokenId, u32), FxBuildHasher> =
        HashMap::with_capacity_and_hasher(hf_merges.len(), FxBuildHasher);

    let hf_str_to_id = |s: &str| -> Option<TokenId> {
        let bytes = token_str_to_bytes(s);
        vocab_inv.get(bytes.as_slice()).copied()
    };

    for (rank, [str_a, str_b]) in hf_merges.iter().enumerate() {
        let id_a = match hf_str_to_id(str_a) {
            Some(id) => id,
            None => continue,
        };
        let id_b = match hf_str_to_id(str_b) {
            Some(id) => id,
            None => continue,
        };

        let merged_str = format!("{str_a}{str_b}");
        let id_merged = match hf_str_to_id(&merged_str) {
            Some(id) => id,
            None => continue,
        };

        merges.entry((id_a, id_b)).or_insert((id_merged, rank as u32));
    }

    // --- Extract added tokens (for splitting before encoding) ----------------

    let added_tokens: Vec<(String, TokenId)> = tj
        .added_tokens
        .iter()
        .filter(|t| t.special)
        .map(|t| (t.content.clone(), TokenId::from(t.id)))
        .collect();

    Ok(SentencePieceBPE {
        merges,
        vocab,
        vocab_inv,
        byte_fallback_ids,
        added_tokens,
    })
}

// ---------------------------------------------------------------------------
// GPT-2 / ByteLevel BPE loader
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping table.
/// Returns (byte_to_unicode, unicode_to_byte).
fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut b2u = ['\0'; 256];
    for &b in &allowed {
        b2u[b as usize] = b as char;
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if b2u[b as usize] == '\0' {
            b2u[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
    (b2u, u2b)
}

/// Decode a GPT-2 ByteLevel unicode string back to raw bytes.
fn unicode_to_bytes(s: &str, u2b: &HashMap<char, u8>) -> Vec<u8> {
    s.chars().map(|c| u2b[&c]).collect()
}

/// Load a HuggingFace `tokenizer.json` that uses ByteLevel BPE without
/// byte_fallback (e.g. GPT-2, RoBERTa).
///
/// Returns a [`bpe::tiktoken::Tokenizer`] with byte remapping.
pub fn load_hf_bpe(path: impl AsRef<Path>) -> Result<bpe::tiktoken::Tokenizer> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let tj: TokenizerJson = sonic_rs::from_slice(&data)
        .with_context(|| format!("Failed to parse {}", path.display()))?;

    ensure!(
        tj.model.model_type == "BPE",
        "Unsupported model type: {} (expected BPE)",
        tj.model.model_type
    );
    ensure!(
        !tj.model.byte_fallback,
        "byte_fallback tokenizers should use load_hf_sentencepiece instead"
    );

    let (_b2u, u2b) = build_byte_unicode_tables();

    // Build vocab sorted by ID — each entry is the raw bytes for that token
    let max_id = tj.model.vocab.values().max().copied().unwrap_or(0) as usize;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from(Vec::new().as_slice()); max_id + 1];
    let mut vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tj.model.vocab.len(), FxBuildHasher);
    for (tok_str, &id) in &tj.model.vocab {
        let bytes: Arc<[u8]> = unicode_to_bytes(tok_str, &u2b).into();
        vocab[id as usize] = bytes.clone();
        vocab_inv.insert(bytes, TokenId::from(id));
    }

    // Build merges from the merge list. Each merge "a b" means:
    // look up token IDs for "a" and "b", the merged token is vocab[concat(a,b)].
    let mut merges: HashMap<(TokenId, TokenId), TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tj.model.merges.len(), FxBuildHasher);
    for [str_a, str_b] in &tj.model.merges {
        let bytes_a = unicode_to_bytes(str_a, &u2b);
        let bytes_b = unicode_to_bytes(str_b, &u2b);
        let id_a = match vocab_inv.get(bytes_a.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let id_b = match vocab_inv.get(bytes_b.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        let mut merged_bytes = bytes_a;
        merged_bytes.extend_from_slice(&bytes_b);
        let id_merged = match vocab_inv.get(merged_bytes.as_slice()) {
            Some(&id) => id,
            None => continue,
        };
        merges.entry((id_a, id_b)).or_insert(id_merged);
    }

    let byte_remapping = bpe::ByteRemapping::from_byte_vocab(&vocab)?;

    Ok(bpe::tiktoken::Tokenizer::new(merges, vocab.into_iter().map(|a| a.to_vec()).collect(), byte_remapping))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_str_to_bytes() {
        assert_eq!(token_str_to_bytes("<0x00>"), vec![0x00]);
        assert_eq!(token_str_to_bytes("<0xFF>"), vec![0xFF]);
        assert_eq!(token_str_to_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(token_str_to_bytes("hello"), b"hello".to_vec());
        assert_eq!(token_str_to_bytes("▁the"), "▁the".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("▁"), "▁".as_bytes().to_vec());
        assert_eq!(token_str_to_bytes("<unk>"), b"<unk>".to_vec());
        assert_eq!(token_str_to_bytes("<s>"), b"<s>".to_vec());
    }

    #[test]
    fn test_load_tinyllama_sentencepiece() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/tinyllama_tokenizer.json"
        );
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        eprintln!("{:?}", tokenizer);
    }

    #[test]
    fn test_encode_hello_sentencepiece() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/tinyllama_tokenizer.json"
        );
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        let ids = tokenizer.encode_raw("Hello world");
        eprintln!("Encoded: {:?}", ids);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, b"Hello world");
    }

    #[test]
    fn test_load_gpt2_from_hf() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/gpt2_tokenizer.json"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let mut tokenizer = load_hf_bpe(path).unwrap();
        eprintln!("{:?}", tokenizer);

        // Encode and verify roundtrip
        let text = b"Hello, world! This is a test.";
        let pretokens = crate::pretokenize::pretokenize_as_iter(text);
        let mut token_ids: Vec<TokenId> = Vec::new();
        tokenizer.memoized_encode(pretokens, |tokens| {
            token_ids.extend_from_slice(tokens);
        });
        eprintln!("Encoded {} bytes -> {:?}", text.len(), token_ids);
        let decoded: Vec<u8> = tokenizer.decode(&token_ids).collect();
        assert_eq!(decoded, text);
    }
}
