//! Test-only lookup of HuggingFace-cached tokenizer files.
//!
//! Mirrors the pure-filesystem cache resolution of `tests/hf_cache.py`:
//! HF_HUB_CACHE, then $HF_HOME/hub, then $XDG_CACHE_HOME/huggingface/hub,
//! then ~/.cache/huggingface/hub, with `refs/main` naming the snapshot.
//! Nothing is ever downloaded: tests skip when a file is absent, except
//! GPT-2, which falls back to the committed copy in tests/fixtures so the
//! core suite passes on a machine with no HF cache at all.

use std::path::PathBuf;

fn hub_cache_dir() -> PathBuf {
    let env = |key: &str| {
        std::env::var(key)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(hub_cache) = env("HF_HUB_CACHE") {
        return hub_cache;
    }
    let hf_home = env("HF_HOME").unwrap_or_else(|| {
        env("XDG_CACHE_HOME")
            .unwrap_or_else(|| std::env::home_dir().expect("home dir").join(".cache"))
            .join("huggingface")
    });
    hf_home.join("hub")
}

/// `filename` from a model repo's `main` snapshot in the local HF cache,
/// or None when the repo, ref, or file is not cached.
pub(crate) fn cached_hub_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    let repo_dir = hub_cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs/main")).ok()?;
    let path = repo_dir.join("snapshots").join(commit.trim()).join(filename);
    path.is_file().then_some(path)
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
