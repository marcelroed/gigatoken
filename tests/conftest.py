"""Shared test fixtures. Downloads required data from HuggingFace into data/ on first use."""

import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from gigatoken._hf_compat import _gpt2_unicode_to_byte

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_hf_file(repo_id: str, filename: str, local_name: str, repo_type: str = "model") -> Path:
    """Download a single file from a HuggingFace repo into DATA_DIR."""
    dest = DATA_DIR / local_name
    if dest.exists():
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    shutil.copy2(cached, dest)
    return dest


def _download_hf_tokenizer(repo_id: str, local_name: str) -> Path:
    """Download a repo's tokenizer.json from HF into DATA_DIR, verbatim.

    (The Rust parser accepts both legacy `"a b"` string merges and array
    merges, so no normalization is needed.)
    """
    return _download_hf_file(repo_id, "tokenizer.json", local_name)


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
def tinyllama_spm_path() -> Path:
    """Path to TinyLlama's raw sentencepiece tokenizer.model."""
    return _download_hf_file(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tokenizer.model",
        "tinyllama_tokenizer.model",
    )


@pytest.fixture(scope="session")
def sp4096_spm_path() -> Path:
    """Path to the parameter-golf fineweb 4096 sentencepiece model (BPE with
    byte fallback, no dummy prefix, nmt_nfkc precompiled charsmap)."""
    return _download_hf_file(
        "Ryukijano/parameter-golf-sp4096",
        "tokenizers/fineweb_4096_bpe.model",
        "fineweb_4096_bpe.model",
        repo_type="dataset",
    )


@pytest.fixture(scope="session")
def phi3_tokenizer_path() -> Path:
    """Path to Phi-3-mini tokenizer.json (Llama-style with rstrip'd added
    tokens outside model.vocab)."""
    return _download_hf_tokenizer(
        "microsoft/Phi-3-mini-4k-instruct",
        "phi3_tokenizer.json",
    )


@pytest.fixture(scope="session")
def llama_legacy_tokenizer_path() -> Path:
    """Path to the original Llama tokenizer.json (added tokens with
    normalized=true, matched against normalizer output)."""
    return _download_hf_tokenizer(
        "hf-internal-testing/llama-tokenizer",
        "llama_legacy_tokenizer.json",
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
def qwen3_5_tokenizer_path() -> Path:
    """Path to Qwen3.5 tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "Qwen/Qwen3.5-9B",
        "qwen3_5_tokenizer.json",
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
def modernbert_tokenizer_path() -> Path:
    """Path to ModernBERT-base tokenizer.json, downloaded from HF if absent."""
    return _download_hf_tokenizer(
        "answerdotai/ModernBERT-base",
        "modernbert_tokenizer.json",
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
# The canonical Python copy of the table lives in gigatoken._hf_compat.
# ---------------------------------------------------------------------------

GPT2_U2B = _gpt2_unicode_to_byte()
GPT2_B2U = [None] * 256
for _ch, _b in GPT2_U2B.items():
    GPT2_B2U[_b] = _ch
del _ch, _b


def gpt2_bytes_to_unicode(data: bytes) -> str:
    return "".join(GPT2_B2U[b] for b in data)


def gpt2_unicode_to_bytes(s: str) -> bytes:
    return bytes(GPT2_U2B[ch] for ch in s)


@pytest.fixture(scope="session")
def dclm_docs() -> list[str]:
    """Curated ~20 MB of DCLM documents, selected from a shard that is
    downloaded into the HuggingFace cache on first use — see dclm_fixture.py."""
    import dclm_fixture

    return dclm_fixture.get_dclm_docs()


@pytest.fixture(scope="session")
def dclm_sample_path(dclm_docs, tmp_path_factory) -> Path:
    """The DCLM sample written to a session-temporary .jsonl.zst file."""
    import zstandard

    path = tmp_path_factory.mktemp("dclm") / "dclm_sample.jsonl.zst"
    with open(path, "wb") as fh:
        with zstandard.ZstdCompressor(level=3).stream_writer(fh) as writer:
            for text in dclm_docs:
                writer.write(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"))
                writer.write(b"\n")
    return path


@pytest.fixture(scope="session")
def r50k_tiktoken_path() -> Path:
    """Path to r50k_base.tiktoken, downloaded from OpenAI if absent."""
    return _download_url(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "r50k_base.tiktoken",
    )
