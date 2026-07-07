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
    #[serde(default)]
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(default)]
    normalizer: Option<NormalizerJson>,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    normalizers: Vec<NormalizerJson>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreTokenizerJson>,
    #[serde(default)]
    pattern: Option<PatternJson>,
}

#[derive(Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
}

#[derive(Deserialize)]
struct Model {
    /// tokenizer.json files written before tokenizers 0.9 (e.g. the original
    /// GPT-2 upload) omit `model.type`; those are always BPE.
    #[serde(rename = "type", default = "legacy_bpe_type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(deserialize_with = "deserialize_merges")]
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
}

fn legacy_bpe_type() -> String {
    "BPE".to_string()
}

/// Merges appear as `["a", "b"]` arrays in current tokenizer.json files and
/// as `"a b"` strings in older ones; accept both.
fn deserialize_merges<'de, D>(deserializer: D) -> Result<Vec<[String; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Merge {
        Pair([String; 2]),
        Legacy(String),
    }
    let raw = Vec::<Merge>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|m| match m {
            Merge::Pair(pair) => Ok(pair),
            Merge::Legacy(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| {
                    serde::de::Error::custom(format!("invalid merge entry: {s:?}"))
                })?;
                Ok([a.to_string(), b.to_string()])
            }
        })
        .collect()
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

/// A tokenizer loaded from HuggingFace `tokenizer.json` data: the model's
/// `byte_fallback` flag decides which of the two supported styles applies.
pub enum HfTokenizer {
    Bpe(bpe::tiktoken::Tokenizer),
    SentencePiece(SentencePieceBPE),
}

fn parse_tokenizer_json(data: &[u8]) -> Result<TokenizerJson> {
    // Inline the deserializer's own message (offending field, position,
    // snippet): the first line is often all that surfaces in test summaries
    // and short tracebacks.
    sonic_rs::from_slice(data).map_err(|e| eyre::eyre!("Failed to parse tokenizer JSON: {e}"))
}

fn read_tokenizer_json(path: impl AsRef<Path>) -> Result<TokenizerJson> {
    let path = path.as_ref();
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    parse_tokenizer_json(&data).with_context(|| format!("Failed to parse {}", path.display()))
}

/// Load a tokenizer from in-memory `tokenizer.json` contents, choosing the
/// SentencePiece or ByteLevel BPE style from the model's `byte_fallback` flag.
pub fn load_hf_slice(data: &[u8]) -> Result<HfTokenizer> {
    let tj = parse_tokenizer_json(data)?;
    if tj.model.byte_fallback {
        Ok(HfTokenizer::SentencePiece(build_sentencepiece(&tj)?))
    } else {
        Ok(HfTokenizer::Bpe(build_bpe(&tj)?))
    }
}

/// Load a HuggingFace `tokenizer.json` that uses SentencePiece-style BPE with
/// byte fallback (e.g. Llama 2 / TinyLlama).
///
/// Returns a [`SentencePieceBPE`] that preserves the original HF token IDs.
pub fn load_hf_sentencepiece(path: impl AsRef<Path>) -> Result<SentencePieceBPE> {
    build_sentencepiece(&read_tokenizer_json(path)?)
}

fn build_sentencepiece(tj: &TokenizerJson) -> Result<SentencePieceBPE> {
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
// Normalizer detection
// ---------------------------------------------------------------------------

/// Determine whether the tokenizer's normalizer is NFC (the only kind we
/// support for ByteLevel BPE). Returns `true` for NFC, `false` for no
/// normalizer, and an error for anything else — silently skipping an unknown
/// normalizer would produce token IDs that diverge from HF.
fn detect_nfc_normalizer(normalizer: &Option<NormalizerJson>) -> Result<bool> {
    fn is_nfc(n: &NormalizerJson) -> Result<bool> {
        match n.kind.as_str() {
            "NFC" => Ok(true),
            "Sequence" => {
                let each: Vec<bool> = n.normalizers.iter().map(is_nfc).collect::<Result<_>>()?;
                Ok(each.into_iter().any(|b| b))
            }
            other => Err(eyre::eyre!("Unsupported normalizer type: {other}")),
        }
    }
    normalizer.as_ref().map_or(Ok(false), is_nfc)
}

// ---------------------------------------------------------------------------
// Pre-tokenizer detection
// ---------------------------------------------------------------------------

/// Determine the pretokenization scheme from a tokenizer.json `pre_tokenizer`.
///
/// Handles a bare `ByteLevel` (GPT-2 style, `use_regex: true`) and
/// `Sequence`s whose `Split` regexes (in order) form a known scheme —
/// either a single known regex or DeepSeek's digits/CJK/main triple.
fn detect_pretokenizer_type(
    pre_tokenizer: &Option<PreTokenizerJson>,
) -> Result<crate::pretokenize::PretokenizerType> {
    use crate::pretokenize::PretokenizerType;

    fn collect_split_regexes<'a>(pt: &'a PreTokenizerJson, out: &mut Vec<&'a str>) {
        if pt.kind == "Split"
            && let Some(PatternJson { regex: Some(re) }) = &pt.pattern
        {
            out.push(re);
        }
        for child in &pt.pretokenizers {
            collect_split_regexes(child, out);
        }
    }

    let Some(pt) = pre_tokenizer else {
        // No pre_tokenizer at all; keep the historical default.
        return Ok(PretokenizerType::GPT2);
    };
    let mut regexes = Vec::new();
    collect_split_regexes(pt, &mut regexes);
    if regexes.is_empty() {
        // ByteLevel with use_regex (the default) splits with the GPT-2 regex.
        if pt.kind == "ByteLevel" {
            return Ok(PretokenizerType::GPT2);
        }
        return Err(eyre::eyre!(
            "Unsupported pre_tokenizer type: {} (no Split regex found)",
            pt.kind
        ));
    }
    PretokenizerType::from_split_regexes(&regexes).ok_or_else(|| {
        eyre::eyre!("Unknown pre_tokenizer Split regexes, no fast pretokenizer for: {regexes:?}")
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
///
/// Byte-level vocab strings consist solely of table chars; a string with any
/// other char is stored raw (e.g. DeepSeek V4 keeps its special tokens
/// unencoded in `model.vocab`) and taken as literal UTF-8 content.
fn unicode_to_bytes(s: &str, u2b: &HashMap<char, u8>) -> Vec<u8> {
    if s.chars().all(|c| u2b.contains_key(&c)) {
        s.chars().map(|c| u2b[&c]).collect()
    } else {
        s.as_bytes().to_vec()
    }
}

/// Load a HuggingFace `tokenizer.json` that uses ByteLevel BPE without
/// byte_fallback (e.g. GPT-2, RoBERTa).
///
/// Returns a [`bpe::tiktoken::Tokenizer`] with byte remapping.
pub fn load_hf_bpe(path: impl AsRef<Path>) -> Result<bpe::tiktoken::Tokenizer> {
    build_bpe(&read_tokenizer_json(path)?)
}

fn build_bpe(tj: &TokenizerJson) -> Result<bpe::tiktoken::Tokenizer> {
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

    // Added tokens may live outside model.vocab (e.g. Qwen2's <|endoftext|>);
    // extend the vocab so their IDs decode to the literal content.
    for t in &tj.added_tokens {
        let id = t.id as usize;
        if id >= vocab.len() {
            vocab.resize(id + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[id].is_empty() {
            vocab[id] = t.content.as_bytes().into();
        }
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

    let mut tokenizer = bpe::tiktoken::Tokenizer::new(
        merges,
        vocab.into_iter().map(|a| a.to_vec()).collect(),
        byte_remapping,
    );
    tokenizer.set_pretokenizer_type(detect_pretokenizer_type(&tj.pre_tokenizer)?);
    tokenizer.set_normalize_nfc(detect_nfc_normalizer(&tj.normalizer)?);
    // All added tokens (special and non-special) are matched atomically in the
    // raw input by HF's AddedVocabulary; mirror that.
    tokenizer.set_added_tokens(
        tj.added_tokens
            .iter()
            .map(|t| (t.content.as_bytes().to_vec(), TokenId::from(t.id)))
            .collect(),
    );
    Ok(tokenizer)
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
    fn test_parse_legacy_model_without_type() {
        // Pre-tokenizers-0.9 files have no `model.type`; they must parse as BPE.
        let json = br#"{"model": {"vocab": {"a": 0}, "merges": []}}"#;
        let tj = parse_tokenizer_json(json).unwrap();
        assert_eq!(tj.model.model_type, "BPE");
    }

    #[test]
    fn test_parse_error_names_the_field() {
        // The first line of the error must carry the deserializer's detail,
        // not just a generic "failed to parse".
        let json = br#"{"model": {"type": "BPE", "merges": []}}"#;
        let err = match parse_tokenizer_json(json) {
            Ok(_) => panic!("expected a parse error"),
            Err(e) => e,
        };
        let first_line = err.to_string();
        assert!(first_line.contains("vocab"), "unhelpful error: {first_line}");
    }

    #[test]
    fn test_load_tinyllama_sentencepiece() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/tinyllama_tokenizer.json");
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        eprintln!("{:?}", tokenizer);
    }

    #[test]
    fn test_encode_hello_sentencepiece() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/tinyllama_tokenizer.json");
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let tokenizer = load_hf_sentencepiece(path).unwrap();
        let ids = tokenizer.encode_raw("Hello world");
        eprintln!("Encoded: {:?}", ids);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, b"Hello world");
    }

    #[test]
    fn test_load_gpt2_from_hf() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/gpt2_tokenizer.json");
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
