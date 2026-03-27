"""Test FileSource with various file formats: .txt, .jsonl, .jsonl.gz, .jsonl.zst"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from jeton import FileSource, train_bpe

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
# Tests
# ---------------------------------------------------------------------------


def test_file_source_txt(txt_file):
    source = FileSource([str(txt_file)])
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE
    assert len(merges) == VOCAB_SIZE - 256


def test_file_source_jsonl(jsonl_file):
    source = FileSource([str(jsonl_file)], field="text")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE


def test_file_source_jsonl_gz(jsonl_gz_file):
    source = FileSource([str(jsonl_gz_file)], field="text")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE


def test_file_source_jsonl_zst(jsonl_zst_file):
    source = FileSource([str(jsonl_zst_file)], field="text")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE


def test_file_source_multi_file(txt_file, jsonl_file, jsonl_gz_file, jsonl_zst_file):
    """Mix of formats in a single FileSource."""
    source = FileSource(
        [str(txt_file), str(jsonl_file), str(jsonl_gz_file), str(jsonl_zst_file)],
        field="text",
    )
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE


def test_file_source_jsonl_matches_bytes(jsonl_file):
    """FileSource(jsonl) should produce the same vocab as training on equivalent bytes."""
    source = FileSource([str(jsonl_file)], field="text")
    vocab_fs, merges_fs = train_bpe(source, VOCAB_SIZE, [])

    # Build equivalent bytes input (all documents joined by separator)
    corpus_bytes = "<|endoftext|>".join(CORPUS_LINES).encode("utf-8")
    vocab_bytes, merges_bytes = train_bpe(corpus_bytes, VOCAB_SIZE, [])

    # Vocab sizes must match
    assert len(vocab_fs) == len(vocab_bytes) == VOCAB_SIZE

    # The merge sets should heavily overlap since both train on the same text
    fs_merges = {(a, b) for a, b in merges_fs}
    bytes_merges = {(a, b) for a, b in merges_bytes}
    overlap = len(fs_merges & bytes_merges) / max(len(fs_merges), 1)
    assert overlap >= 0.8, f"Only {overlap:.0%} merge overlap between FileSource and bytes"


def test_file_source_custom_field(tmp_dir):
    """JSONL with a non-default field name."""
    path = tmp_dir / "custom_field.jsonl"
    with open(path, "w") as f:
        for line in CORPUS_LINES:
            f.write(json.dumps({"content": line}) + "\n")

    source = FileSource([str(path)], field="content")
    vocab, merges = train_bpe(source, VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE


def test_file_source_repr():
    source = FileSource(["a.txt", "b.jsonl"], field="text")
    assert "2 files" in repr(source)
