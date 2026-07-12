"""Whole-document parallel encoding must be token-identical to a serial pass.

A single document above the chunk target is split at pretoken-safe
boundaries (see `pretokenize::safe_split_ranges`) and encoded across cores.
This runs for every tokenizer spec, so each pretokenization scheme (r50k,
cl100k-family, qwen2/olmo3 with NFC, deepseek) is checked end to end.
"""

import json
from pathlib import Path

import pytest

from gigatoken.gigatoken_rs import BPETokenizer

OWT_PATH = Path.home() / "data" / "owt_train.txt"
DOC_BYTES = 8 * 2**20  # several fragments at the 1 MiB minimum chunk size


@pytest.fixture(scope="session")
def big_text() -> str:
    if OWT_PATH.exists():
        with open(OWT_PATH, "rb") as f:
            raw = f.read(DOC_BYTES)
        return raw.decode("utf-8", errors="ignore")
    seed = (
        "Numbers like 1234567890123 and marks — dashes, 'contractions', "
        "punct...\n\n  whitespace runs\t and Ünïcodé letters. "
    )
    return seed * (DOC_BYTES // len(seed))


def test_whole_doc_parallel_matches_serial(gigatoken_tok, big_text, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text(big_text)
    parallel = gigatoken_tok.encode_files([str(path)])  # whole file = one doc
    serial = gigatoken_tok.encode(big_text)
    assert len(parallel) == 1
    assert parallel[0].tolist() == serial.tolist()


def test_single_doc_batch_matches_serial(gigatoken_tok, big_text):
    out = gigatoken_tok.encode_batch([big_text])
    assert len(out) == 1
    assert out[0].tolist() == gigatoken_tok.encode(big_text).tolist()


def test_space_added_token_chunked_matches_serial(
    gpt2_tokenizer_path, big_text, tmp_path
):
    """A tokenizer whose added tokens contain spaces must still split huge
    documents, with boundaries that never cut an added-token occurrence."""
    token_text = "<|multi word separator|>"
    token_id = 50257  # one past the GPT-2 vocab

    spec = json.loads(gpt2_tokenizer_path.read_text())
    spec["added_tokens"].append(
        {"id": token_id, "content": token_text, "special": True}
    )
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(spec, ensure_ascii=False))
    tok = BPETokenizer.from_hf(path)

    pieces = 64
    step = len(big_text) // pieces
    text = token_text.join(
        big_text[i * step : (i + 1) * step] for i in range(pieces)
    )

    serial = tok.encode(text)
    assert (serial == token_id).sum() == pieces - 1

    chunked = tok.encode_batch([text])
    assert len(chunked) == 1
    assert chunked[0].tolist() == serial.tolist()
