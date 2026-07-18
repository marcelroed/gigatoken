"""HuggingFace Hub files for tests and benches, without huggingface_hub.

A thin forward to the Rust cache-or-download (`src/load_tokenizer/hub.rs`,
exposed as `gigatoken.gigatoken_rs.hub_file`): cached files are found with a
pure-filesystem lookup against the standard HF cache; misses are downloaded —
passing the token from the standard HF discovery when one exists — into the
same cache layout huggingface_hub uses, so both stay interchangeable
consumers of one cache.
"""

from pathlib import Path

from gigatoken.gigatoken_rs import hub_file


def hf_file(repo_id: str, filename: str, repo_type: str = "model", revision: str = "main") -> Path:
    """Path of `filename` from `repo_id` in the standard HF cache, downloaded
    there on first use. `revision` is a branch/tag name or a commit hash."""
    return hub_file(repo_id, filename, repo_type=repo_type, revision=revision)
