"""HuggingFace Hub file fetch — thin forwards to the Rust implementation.

The mechanics live in Rust (`src/load_tokenizer/hub.rs`) and mirror
`huggingface_hub.hf_hub_download` — same endpoint and URL layout, same token
discovery (HF_TOKEN env var, then the token file written by `hf auth login`),
same cache directory resolution — without requiring huggingface_hub,
tokenizers, or transformers to be installed. Files already present in the
standard HF cache are served with a pure-filesystem lookup (no network); on a
miss the file is downloaded straight into the shared HF cache, so later loads
(ours and huggingface_hub's) are served from it.
"""

from __future__ import annotations

from gigatoken.gigatoken_rs import get_hf_token, hub_file, looks_like_repo_id

# Filename suffixes of local tokenizer files (tokenizer.json contents and raw
# sentencepiece models — the formats `gigatoken._load.hf.to_tokenizer_json`
# reads from disk). A name ending in one of these is never treated as a Hub
# repo id, so a mistyped local path fails fast instead of hitting the network.
# Keep in sync with TOKENIZER_FILE_SUFFIXES in `src/load_tokenizer/hub.rs`.
TOKENIZER_FILE_SUFFIXES = (".json", ".model")

__all__ = [
    "TOKENIZER_FILE_SUFFIXES",
    "download_hub_file",
    "get_hf_token",
    "hub_file",
    "looks_like_repo_id",
]


def download_hub_file(repo_id: str, filename: str = "tokenizer.json", *, revision: str = "main") -> bytes:
    """Contents of `filename` from Hub repo `repo_id` at `revision`, served
    from the standard HF cache, downloading into it first when absent."""
    return hub_file(repo_id, filename, revision=revision).read_bytes()
