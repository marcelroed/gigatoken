"""Shared test fixtures. HuggingFace-hosted files are served straight from
the standard HF cache (~/.cache/huggingface by default) via a pure-filesystem
lookup, with cache misses downloaded by requests into the same layout (see
hf_cache.py) — huggingface_hub, tokenizers, and transformers are never
imported just to fetch a file, and nothing is copied into the repo."""

import json
import os
import urllib.request
from pathlib import Path

import pytest

from hf_cache import hf_file as _hf_file

from gigatoken._hf_compat import _gpt2_unicode_to_byte


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _hf_tokenizer_json(repo_id: str) -> Path:
    """Path of a repo's tokenizer.json in the standard HF cache, verbatim.

    (The Rust parser accepts both legacy `"a b"` string merges and array
    merges, so no normalization is needed.)
    """
    return _hf_file(repo_id, "tokenizer.json")


def _download_url(url: str, local_name: str) -> Path:
    """Download a non-HF file into the user cache (kept across sessions,
    nothing written to the repo)."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    dest = cache_home / "gigatoken-tests" / local_name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Session-scoped fixtures (downloaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tinyllama_tokenizer_path() -> Path:
    """Path to TinyLlama tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@pytest.fixture(scope="session")
def tinyllama_spm_path() -> Path:
    """Path to TinyLlama's raw sentencepiece tokenizer.model."""
    return _hf_file("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.model")


@pytest.fixture(scope="session")
def sp4096_spm_path() -> Path:
    """Path to the parameter-golf fineweb 4096 sentencepiece model (BPE with
    byte fallback, no dummy prefix, nmt_nfkc precompiled charsmap)."""
    return _hf_file(
        "Ryukijano/parameter-golf-sp4096",
        "tokenizers/fineweb_4096_bpe.model",
        repo_type="dataset",
    )


@pytest.fixture(scope="session")
def phi3_tokenizer_path() -> Path:
    """Path to Phi-3-mini tokenizer.json (Llama-style with rstrip'd added
    tokens outside model.vocab)."""
    return _hf_tokenizer_json("microsoft/Phi-3-mini-4k-instruct")


@pytest.fixture(scope="session")
def llama_legacy_tokenizer_path() -> Path:
    """Path to the original Llama tokenizer.json (added tokens with
    normalized=true, matched against normalizer output)."""
    return _hf_tokenizer_json("hf-internal-testing/llama-tokenizer")


@pytest.fixture(scope="session")
def gpt2_tokenizer_path() -> Path:
    """Path to GPT-2 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("openai-community/gpt2")


@pytest.fixture(scope="session")
def gpt2_hub_dir(gpt2_tokenizer_path) -> Path:
    """GPT-2 snapshot directory in the HF cache with every file
    AutoTokenizer.from_pretrained needs, prefetched — pinned to the cached
    tokenizer.json's revision so the snapshot is consistent — so transformers
    only ever reads locally and never downloads itself."""
    snapshot = gpt2_tokenizer_path.parent
    for name in ("tokenizer_config.json", "config.json", "vocab.json", "merges.txt"):
        _hf_file("openai-community/gpt2", name, revision=snapshot.name)
    return snapshot


@pytest.fixture(scope="session")
def qwen2_tokenizer_path() -> Path:
    """Path to Qwen2 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("Qwen/Qwen2-1.5B-Instruct")


@pytest.fixture(scope="session")
def qwen3_5_tokenizer_path() -> Path:
    """Path to Qwen3.5 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("Qwen/Qwen3.5-9B")


@pytest.fixture(scope="session")
def deepseek_v3_tokenizer_path() -> Path:
    """Path to DeepSeek V3 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("deepseek-ai/DeepSeek-V3")


@pytest.fixture(scope="session")
def deepseek_v4_tokenizer_path() -> Path:
    """Path to DeepSeek V4 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("deepseek-ai/DeepSeek-V4-Flash")


@pytest.fixture(scope="session")
def glm5_2_tokenizer_path() -> Path:
    """Path to GLM-5.2 tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("zai-org/GLM-5.2")


@pytest.fixture(scope="session")
def modernbert_tokenizer_path() -> Path:
    """Path to ModernBERT-base tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("answerdotai/ModernBERT-base")


@pytest.fixture(scope="session")
def olmo3_tokenizer_path() -> Path:
    """Path to Olmo3 (dolma2) tokenizer.json in the HF cache."""
    return _hf_tokenizer_json("allenai/Olmo-3-1025-7B")


@pytest.fixture(scope="session")
def nemotron_right_aligned_tokenizer_path(tmp_path_factory) -> Path:
    """Nemotron 3 with the right-aligned digit-group regex (issue #60), plus
    every 2- and 3-digit string as a token: Nemotron's own vocab has single
    digits only, so without them a grouping bug would not change the ids."""
    path = _hf_tokenizer_json("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    tok = json.loads(path.read_text())
    split = tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]
    right_aligned = split["Regex"].replace(
        r"|\p{N}|", r"|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}|"
    )
    assert right_aligned != split["Regex"]
    split["Regex"] = right_aligned
    model = tok["model"]
    next_id = 1 + max(*model["vocab"].values(), *(t["id"] for t in tok["added_tokens"]))
    for n in (2, 3):
        for i in range(10**n):
            digits = str(i).zfill(n)
            model["vocab"][digits] = next_id
            model["merges"].append([digits[:-1], digits[-1]])
            next_id += 1
    out = tmp_path_factory.mktemp("nemotron_right_aligned") / "tokenizer.json"
    out.write_text(json.dumps(tok, ensure_ascii=False))
    return out


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


def tiktoken_vocab_path(encoding: str) -> Path:
    """Path to a published .tiktoken vocabulary (r50k_base, cl100k_base,
    o200k_base, ...), downloaded from OpenAI if absent."""
    return _download_url(
        f"https://openaipublic.blob.core.windows.net/encodings/{encoding}.tiktoken",
        f"{encoding}.tiktoken",
    )


@pytest.fixture(scope="session")
def r50k_tiktoken_path() -> Path:
    """Path to r50k_base.tiktoken, downloaded from OpenAI if absent."""
    return tiktoken_vocab_path("r50k_base")
