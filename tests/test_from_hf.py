"""Test that SentencePieceTokenizer.from_hf produces identical token IDs to HuggingFace.

The SentencePieceTokenizer preserves original HF token IDs, uses character-level
initial tokenization with byte fallback, and applies BPE merges — exactly
matching the HuggingFace tokenizers pipeline.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from gigatoken.gigatoken_rs import SentencePieceTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OWT_PATH = DATA_DIR / "owt_train.txt"
OWT_SIZE = 10_000_000  # ~10 MB


@pytest.fixture(scope="module")
def hf_tok(tinyllama_tokenizer_path):
    return Tokenizer.from_file(str(tinyllama_tokenizer_path))


@pytest.fixture(scope="module")
def gigatoken_tok(tinyllama_tokenizer_path):
    return SentencePieceTokenizer.from_hf(tinyllama_tokenizer_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_ids_match(hf_tok, gigatoken_tok, text: str):
    """Encode text with both tokenizers and compare token IDs directly."""
    hf_ids = hf_tok.encode(text).ids[1:]  # strip BOS
    gigatoken_ids = gigatoken_tok.encode(text).tolist()
    assert gigatoken_ids == hf_ids, (
        f"Mismatch for {text!r}:\n"
        f"  HF:    {hf_ids}\n"
        f"  gigatoken: {gigatoken_ids}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TEXTS = [
    "Hello",
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "",
    " ",
    "   leading and trailing spaces   ",
    "café résumé naïve",
    "emoji: 😒🌍🎉",
    "mixed: hello 世界 🌎",
    "Ñoño año",
    "def foo(x: int) -> int:\n    return x + 1\n",
    "import os\nos.path.join('a', 'b')",
    'SELECT * FROM users WHERE id = 1;',
    '{"key": "value", "num": 123, "arr": [1, 2, 3]}',
    "https://example.com/path?query=value&other=123#fragment",
    "\n",
    "\n\n\n",
    "\t\t",
    "a" * 500,
    "hello " * 100,
    "\r\n\r\n",
    "日本語テスト",
    "Привет мир",
    "مرحبا بالعالم",
]


@pytest.mark.parametrize("text", TEXTS, ids=lambda t: repr(t)[:50])
def test_encode_matches_hf(hf_tok, gigatoken_tok, text):
    _assert_ids_match(hf_tok, gigatoken_tok, text)


def test_decode_roundtrip(gigatoken_tok):
    text = "Hello world, this is a test."
    ids = gigatoken_tok.encode(text)
    decoded = gigatoken_tok.decode(ids)
    assert decoded == text.encode("utf-8")


def test_decode_with_byte_fallback(gigatoken_tok):
    text = "emoji: 🚀"
    ids = gigatoken_tok.encode(text)
    decoded = gigatoken_tok.decode(ids)
    assert decoded == text.encode("utf-8")


# ---------------------------------------------------------------------------
# Large-scale OWT test
# ---------------------------------------------------------------------------


def _load_owt_lines(max_bytes: int) -> list[str]:
    with open(OWT_PATH, "rb") as f:
        raw = f.read(max_bytes)
    last_newline = raw.rfind(b"\n")
    if last_newline != -1:
        raw = raw[: last_newline + 1]
    text = raw.decode("utf-8", errors="replace")
    return text.splitlines()


@pytest.mark.skipif(not OWT_PATH.exists(), reason="OWT data not available")
def test_owt_10mb(hf_tok, gigatoken_tok):
    """Compare token IDs on ~10 MB of OWT, line by line."""
    lines = _load_owt_lines(OWT_SIZE)
    non_empty = [l for l in lines if l]
    total_bytes = sum(len(l.encode("utf-8")) for l in non_empty)
    print(f"\nLoaded {len(non_empty)} lines ({total_bytes / 1e6:.1f} MB)")

    # HF: batch encode for parallelism (includes BOS, so strip it)
    hf_encodings = hf_tok.encode_batch(non_empty)
    hf_id_lists = [enc.ids[1:] for enc in hf_encodings]  # strip BOS

    mismatches = 0
    for i, line in enumerate(non_empty):
        gigatoken_ids = gigatoken_tok.encode(line).tolist()
        hf_ids = hf_id_lists[i]
        if gigatoken_ids != hf_ids:
            if mismatches < 5:
                print(
                    f"  Line {i}: {line[:60]!r}...\n"
                    f"    HF:    {hf_ids[:10]}... ({len(hf_ids)} tokens)\n"
                    f"    gigatoken: {gigatoken_ids[:10]}... ({len(gigatoken_ids)} tokens)"
                )
            mismatches += 1

    print(f"Results: {len(non_empty)} lines, {mismatches} mismatches")
    assert mismatches == 0, f"{mismatches}/{len(non_empty)} lines had different token IDs"
