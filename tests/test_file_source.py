"""Test FileSource with various file formats: .txt, .jsonl, .jsonl.gz, .jsonl.zst"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from gigatoken import FileSource, JsonlFileSource, TextFileSource, train_bpe

CORPUS_LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "To be, or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "Once upon a time, there was a little girl named Lily.",
    "The sun was shining and the birds were singing.",
    "Tom and his friend went to the park to play.",
    "The stars came out at night and twinkled in the sky.",
] * 50  # Repeat for enough data


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def txt_file(tmp_dir):
    path = tmp_dir / "corpus.txt"
    path.write_text("<|endoftext|>".join(CORPUS_LINES))
    return path


@pytest.fixture(scope="module")
def jsonl_file(tmp_dir):
    path = tmp_dir / "corpus.jsonl"
    with open(path, "w") as f:
        for line in CORPUS_LINES:
            f.write(json.dumps({"text": line}) + "\n")
    return path


@pytest.fixture(scope="module")
def jsonl_gz_file(tmp_dir):
    path = tmp_dir / "corpus.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for line in CORPUS_LINES:
            f.write(json.dumps({"text": line}) + "\n")
    return path


@pytest.fixture(scope="module")
def jsonl_zst_file(tmp_dir):
    zstd = pytest.importorskip("zstandard")
    path = tmp_dir / "corpus.jsonl.zst"
    cctx = zstd.ZstdCompressor()
    data = "".join(json.dumps({"text": line}) + "\n" for line in CORPUS_LINES)
    with open(path, "wb") as f:
        f.write(cctx.compress(data.encode("utf-8")))
    return path


VOCAB_SIZE = 400


# ---------------------------------------------------------------------------
# Reference result: train once from JSONL, compare all others against it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_result(jsonl_file):
    """Train from plain JSONL — the reference all other formats must match."""
    source = JsonlFileSource([str(jsonl_file)], field="text")
    return train_bpe(source, VOCAB_SIZE, [])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_file_source_txt(txt_file):
    source = TextFileSource([str(txt_file)], separator=b"<|endoftext|>")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE
    assert len(merges) == VOCAB_SIZE - 256


def test_file_source_jsonl(reference_result):
    vocab, merges = reference_result
    assert len(vocab) == VOCAB_SIZE
    assert len(merges) == VOCAB_SIZE - 256


def test_file_source_jsonl_gz_matches_jsonl(jsonl_gz_file, reference_result):
    """Gzip-compressed JSONL must produce identical merges to plain JSONL."""
    source = JsonlFileSource([str(jsonl_gz_file)], field="text")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    _, ref_merges = reference_result
    assert merges == ref_merges


def test_file_source_jsonl_zst_matches_jsonl(jsonl_zst_file, reference_result):
    """Zstd-compressed JSONL must produce identical merges to plain JSONL."""
    source = JsonlFileSource([str(jsonl_zst_file)], field="text")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    _, ref_merges = reference_result
    assert merges == ref_merges


def test_file_source_jsonl_matches_bytes(jsonl_file, reference_result):
    """FileSource(jsonl) must produce identical merges to training on equivalent bytes."""
    _, ref_merges = reference_result

    # Build equivalent bytes input (all documents joined by separator)
    corpus_bytes = "<|endoftext|>".join(CORPUS_LINES).encode("utf-8")
    _, bytes_merges = train_bpe(corpus_bytes, VOCAB_SIZE, [])

    assert ref_merges == bytes_merges


def test_file_source_multi_file_matches_single(
    jsonl_file, jsonl_gz_file, jsonl_zst_file, reference_result
):
    """Multiple copies of the same data (different formats) must produce
    identical merges to a single copy — the word counts scale uniformly
    so merge order is preserved."""
    source = JsonlFileSource(
        [str(jsonl_file), str(jsonl_gz_file), str(jsonl_zst_file)],
        field="text",
    )
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    _, ref_merges = reference_result
    assert merges == ref_merges


def test_file_source_custom_field(tmp_dir, reference_result):
    """JSONL with a non-default field name must produce identical merges."""
    path = tmp_dir / "custom_field.jsonl"
    with open(path, "w") as f:
        for line in CORPUS_LINES:
            f.write(json.dumps({"content": line}) + "\n")

    source = JsonlFileSource([str(path)], field="content")
    _, merges = train_bpe(source, VOCAB_SIZE, [])
    _, ref_merges = reference_result
    assert merges == ref_merges


def test_file_source_repr():
    assert "2 files" in repr(JsonlFileSource(["a.jsonl", "b.jsonl"]))
    assert "2 files" in repr(TextFileSource(["a.txt", "b.txt"]))


def test_file_source_base_not_constructible():
    with pytest.raises(TypeError):
        FileSource(["a.txt"])
