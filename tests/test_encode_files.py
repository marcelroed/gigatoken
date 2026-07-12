"""encode_files / encode_batch: source types, parallel chunking, and parity.

Uses the Olmo3 tokenizer (no NFC normalization, so byte-exact roundtrips).
"""

import gzip
import json
from pathlib import Path

import pytest

from gigatoken import JsonlFileSource, TextFileSource
from gigatoken.gigatoken_rs import BPETokenizer

DOCS = [
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
] * 20


@pytest.fixture(scope="module")
def tok(olmo3_tokenizer_path) -> BPETokenizer:
    return BPETokenizer.from_hf(olmo3_tokenizer_path)


@pytest.fixture(scope="module")
def expected(tok):
    """Reference: per-document ids via single-doc encode."""
    return [tok.encode(d).tolist() for d in DOCS]


def _ids(out):
    return out.tolist()


# ---------------------------------------------------------------------------
# encode_batch
# ---------------------------------------------------------------------------


def test_encode_batch_str_and_bytes_match(tok, expected):
    assert _ids(tok.encode_batch(DOCS)) == expected
    assert _ids(tok.encode_batch([d.encode() for d in DOCS])) == expected


def test_encode_batch_empty(tok):
    assert len(tok.encode_batch([])) == 0


def test_encode_batch_rejects_paths(tok, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("hello")
    with pytest.raises(TypeError, match="encode_files"):
        tok.encode_batch([path])


def test_encode_batch_rejects_mixed_types(tok):
    with pytest.raises(TypeError, match="same type"):
        tok.encode_batch(["text", b"bytes"])


def test_encode_batch_repeated_invocation(tok, expected):
    """Pooled workers keep their caches across calls; results must not drift."""
    for _ in range(3):
        assert _ids(tok.encode_batch(DOCS)) == expected


def test_encode_batch_awkward_input(tok, expected):
    """An awkward Array of strings or bytestrings encodes straight from its
    flat buffers, matching the list inputs."""
    import awkward as ak

    assert _ids(tok.encode_batch(ak.Array(DOCS))) == expected
    assert _ids(tok.encode_batch(ak.Array([d.encode() for d in DOCS]))) == expected


def test_encode_batch_awkward_rejects_non_strings(tok):
    import awkward as ak

    with pytest.raises(TypeError, match="strings or bytestrings"):
        tok.encode_batch(ak.Array([[1, 2], [3]]))


# ---------------------------------------------------------------------------
# encode_files
# ---------------------------------------------------------------------------


def test_jsonl_source(tok, expected, tmp_path):
    path = tmp_path / "docs.jsonl"
    with open(path, "w") as f:
        for d in DOCS:
            f.write(json.dumps({"text": d}) + "\n")
    assert _ids(tok.encode_files(JsonlFileSource([path]))) == expected


def test_jsonl_gzip_matches_plain(tok, expected, tmp_path):
    path = tmp_path / "docs.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for d in DOCS:
            f.write(json.dumps({"text": d}) + "\n")
    assert _ids(tok.encode_files(JsonlFileSource([path]))) == expected


def test_jsonl_custom_field(tok, expected, tmp_path):
    path = tmp_path / "docs.jsonl"
    with open(path, "w") as f:
        for d in DOCS:
            f.write(json.dumps({"content": d}) + "\n")
    assert _ids(tok.encode_files(JsonlFileSource([path], field="content"))) == expected


def test_text_source_with_separator(tok, expected, tmp_path):
    path = tmp_path / "docs.txt"
    path.write_text("<|sep|>".join(DOCS))
    got = tok.encode_files(TextFileSource([path], separator=b"<|sep|>"))
    assert _ids(got) == expected


def test_text_source_one_doc_per_file(tok, tmp_path):
    paths = []
    for i, d in enumerate(DOCS[:5]):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(d)
        paths.append(p)
    got = tok.encode_files(TextFileSource(paths))
    assert _ids(got) == [tok.encode(d).tolist() for d in DOCS[:5]]


def test_bare_path_list_defaults(tok, expected, tmp_path):
    """A bare list of paths auto-detects: .txt → one doc per file."""
    paths = []
    for i, d in enumerate(DOCS[:3]):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(d)
        paths.append(p)
    assert _ids(tok.encode_files(paths)) == expected[:3]


def test_bare_single_path_jsonl(tok, expected, tmp_path):
    """A single bare .jsonl path auto-detects JSONL with field 'text'."""
    path = tmp_path / "docs.jsonl"
    with open(path, "w") as f:
        for d in DOCS:
            f.write(json.dumps({"text": d}) + "\n")
    assert _ids(tok.encode_files(path)) == expected


def test_multiple_files_preserve_order(tok, expected, tmp_path):
    third = len(DOCS) // 3
    parts = [DOCS[:third], DOCS[third : 2 * third], DOCS[2 * third :]]
    paths = []
    for i, part in enumerate(parts):
        p = tmp_path / f"part_{i}.jsonl"
        with open(p, "w") as f:
            for d in part:
                f.write(json.dumps({"text": d}) + "\n")
        paths.append(p)
    assert _ids(tok.encode_files(JsonlFileSource(paths))) == expected


def test_empty_text_file(tok, tmp_path):
    """One doc per file holds even for an empty file: one empty array."""
    path = tmp_path / "empty.txt"
    path.write_text("")
    got = tok.encode_files(TextFileSource([path]))
    assert len(got) == 1
    assert got[0].tolist() == []


def test_missing_file_raises(tok, tmp_path):
    with pytest.raises(OSError, match="nope.txt"):
        tok.encode_files([tmp_path / "nope.txt"])


def test_single_huge_doc_split_matches_serial(tok, tmp_path):
    """One document far above the chunk target is split at pretoken-safe
    boundaries and encoded in parallel — ids must match a serial pass."""
    text = "".join(DOCS) * 500 + "digits 1234567890123 <|endoftext|>\n\n tail"
    assert len(text) > 4 * 2**20
    path = tmp_path / "huge.txt"
    path.write_text(text)

    serial = tok.encode(text).tolist()
    from_file = tok.encode_files([path])  # whole file = one document
    assert len(from_file) == 1
    assert from_file[0].tolist() == serial
    from_batch = tok.encode_batch([text])  # single-doc batch splits too
    assert from_batch[0].tolist() == serial


def test_large_jsonl_parallel_chunking(tok, tmp_path):
    """A single file well above the chunking threshold (several parallel
    chunks) must produce the same ids per document as encode_batch."""
    docs = DOCS * 200  # ~1.8 MB of JSONL, > MIN_CHUNK_BYTES
    path = tmp_path / "big.jsonl"
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    got = tok.encode_files(JsonlFileSource([path]))
    assert _ids(got) == _ids(tok.encode_batch(docs))
