"""Shared test fixtures. Downloads required data from HuggingFace into data/ on first use."""

import json
import shutil
import urllib.request
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_hf_file(repo_id: str, filename: str, local_name: str) -> Path:
    """Download a single file from a HuggingFace repo into DATA_DIR."""
    dest = DATA_DIR / local_name
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, dest)
    return dest


def _download_hf_tokenizer(repo_id: str, local_name: str) -> Path:
    """Download tokenizer.json from HF and normalize merges to array format.

    Newer HF tokenizer files store merges as strings ("▁ t") but the Rust
    parser expects arrays (["▁", "t"]). This converts on download.
    """
    dest = DATA_DIR / local_name
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    with open(cached) as f:
        data = json.load(f)
    merges = data.get("model", {}).get("merges", [])
    if merges and isinstance(merges[0], str):
        data["model"]["merges"] = [m.split(" ", 1) for m in merges]
    with open(dest, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return dest


def _download_url(url: str, local_name: str) -> Path:
    """Download a file from a URL into DATA_DIR."""
    dest = DATA_DIR / local_name
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Session-scoped fixtures (downloaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tinyllama_tokenizer_path() -> Path:
    """Path to TinyLlama tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tinyllama_tokenizer.json",
    )


@pytest.fixture(scope="session")
def gpt2_tokenizer_path() -> Path:
    """Path to GPT-2 tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "openai-community/gpt2",
        "gpt2_tokenizer.json",
    )


@pytest.fixture(scope="session")
def qwen2_tokenizer_path() -> Path:
    """Path to Qwen2 tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "Qwen/Qwen2-1.5B-Instruct",
        "qwen2_tokenizer.json",
    )


@pytest.fixture(scope="session")
def deepseek_v3_tokenizer_path() -> Path:
    """Path to DeepSeek V3 tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "deepseek-ai/DeepSeek-V3",
        "deepseek_v3_tokenizer.json",
    )


@pytest.fixture(scope="session")
def deepseek_v4_tokenizer_path() -> Path:
    """Path to DeepSeek V4 tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek_v4_tokenizer.json",
    )


@pytest.fixture(scope="session")
def olmo3_tokenizer_path() -> Path:
    """Path to Olmo3 (dolma2) tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "allenai/Olmo-3-1025-7B",
        "olmo3_tokenizer.json",
    )


# ---------------------------------------------------------------------------
# GPT-2 byte <-> unicode helpers (reused by test_bpe_train_compare, etc.)
# ---------------------------------------------------------------------------


def _build_gpt2_byte_unicode_tables():
    allowed = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    b2u = [None] * 256
    for b in allowed:
        b2u[b] = chr(b)
    n = 0
    for b in range(256):
        if b2u[b] is None:
            b2u[b] = chr(256 + n)
            n += 1
    u2b = {ch: i for i, ch in enumerate(b2u)}
    return b2u, u2b


GPT2_B2U, GPT2_U2B = _build_gpt2_byte_unicode_tables()


def gpt2_bytes_to_unicode(data: bytes) -> str:
    return "".join(GPT2_B2U[b] for b in data)


def gpt2_unicode_to_bytes(s: str) -> bytes:
    return bytes(GPT2_U2B[ch] for ch in s)


@pytest.fixture(scope="session")
def r50k_tiktoken_path() -> Path:
    """Path to r50k_base.tiktoken, downloaded from OpenAI if absent."""
    return _download_url(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "r50k_base.tiktoken",
    )
