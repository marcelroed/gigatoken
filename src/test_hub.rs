//! Test-only lookup of HuggingFace-cached tokenizer files.
//!
//! Thin wrappers over `load_tokenizer::hub`'s pure-filesystem cache
//! resolution. Nothing is ever downloaded: tests skip when a file is absent,
//! except GPT-2, which falls back to the committed copy in tests/fixtures so
//! the core suite passes on a machine with no HF cache at all.

use std::path::PathBuf;

/// `filename` from a model repo's `main` snapshot in the local HF cache,
/// or None when the repo, ref, or file is not cached.
pub(crate) fn cached_hub_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    crate::load_tokenizer::hub::cached_hub_file(repo_id, filename, "main")
}

/// A model repo's tokenizer.json from the local HF cache.
pub(crate) fn hf_tokenizer_json(repo_id: &str) -> Option<PathBuf> {
    cached_hub_file(repo_id, "tokenizer.json")
}

/// GPT-2's tokenizer.json: the HF cache copy when present, else the
/// committed fixture (same vocab/merges/config; the fixture is a verbatim
/// copy of the openai-community/gpt2 file).
pub(crate) fn gpt2_tokenizer_json() -> PathBuf {
    hf_tokenizer_json("openai-community/gpt2").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpt2_tokenizer.json")
    })
}
