"""Convert a raw sentencepiece `.model` file to HuggingFace tokenizer.json
contents using only the standard library.

Mirrors `transformers.convert_slow_tokenizer.SpmConverter` for BPE models
with byte fallback, with two deliberate differences that make the result
match raw sentencepiece more closely:

- `remove_extra_whitespaces` strips *leading* whitespace too (transformers
  emits `Strip(right)` only, diverging from sentencepiece on inputs with
  leading whitespace).
- No Metaspace pre-tokenizer: sentencepiece BPE merges across the whole
  sentence, so pre-splitting at ▁ would break whitespace-run pieces like
  Llama's "▁▁▁▁". Spaces are handled in the normalizer instead, exactly like
  HF's own legacy Llama conversion.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any, Iterator

# sentencepiece_model.proto field numbers.
_MODEL_PIECES = 1
_MODEL_TRAINER_SPEC = 2
_MODEL_NORMALIZER_SPEC = 3
_PIECE_PIECE = 1
_PIECE_SCORE = 2
_PIECE_TYPE = 3
_TRAINER_MODEL_TYPE = 3
_TRAINER_TREAT_WHITESPACE_AS_SUFFIX = 24
_TRAINER_BYTE_FALLBACK = 35
_TRAINER_UNK_PIECE = 45
_NORM_PRECOMPILED_CHARSMAP = 2
_NORM_ADD_DUMMY_PREFIX = 3
_NORM_REMOVE_EXTRA_WHITESPACES = 4
_NORM_ESCAPE_WHITESPACES = 5
# SentencePiece.Type values.
_TYPE_NORMAL, _TYPE_UNKNOWN, _TYPE_CONTROL, _TYPE_USER_DEFINED, _TYPE_UNUSED, _TYPE_BYTE = 1, 2, 3, 4, 5, 6


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes) -> Iterator[tuple[int, int | bytes]]:
    """Yield `(field_number, value)` for each field in a protobuf message.

    Varints come out as ints; length-delimited fields (strings, bytes,
    sub-messages) as bytes; fixed 32/64-bit fields as bytes.
    """
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        value: int | bytes
        if wire == 0:
            value, i = _read_varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            length, i = _read_varint(buf, i)
            value, i = buf[i : i + length], i + length
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"not a sentencepiece model: unsupported protobuf wire type {wire}")
        yield field, value


def _scalars(buf: bytes) -> dict[int, int | bytes]:
    """Last-value-wins view of a message's fields (protobuf semantics)."""
    return dict(_fields(buf))


def sentencepiece_to_tokenizer_json(data: bytes) -> str:
    """tokenizer.json contents equivalent to a sentencepiece `.model`.

    Only BPE models (`model_type: BPE`) with byte fallback are supported —
    the same class the Rust SentencePiece backend implements.
    """
    pieces: list[tuple[str, float, int]] = []  # (piece, score, type)
    trainer: dict[int, int | bytes] = {}
    normspec: dict[int, int | bytes] = {}
    for field, value in _fields(data):
        if field == _MODEL_PIECES:
            assert isinstance(value, bytes)
            piece = _scalars(value)
            content = piece.get(_PIECE_PIECE, b"")
            assert isinstance(content, bytes)
            raw_score = piece.get(_PIECE_SCORE, b"\x00\x00\x00\x00")
            assert isinstance(raw_score, bytes)
            piece_type = piece.get(_PIECE_TYPE, _TYPE_NORMAL)
            assert isinstance(piece_type, int)
            (score,) = struct.unpack("<f", raw_score)
            pieces.append((content.decode("utf-8"), score, piece_type))
        elif field == _MODEL_TRAINER_SPEC:
            assert isinstance(value, bytes)
            trainer = _scalars(value)
        elif field == _MODEL_NORMALIZER_SPEC:
            assert isinstance(value, bytes)
            normspec = _scalars(value)

    if not pieces:
        raise ValueError("not a sentencepiece model: no pieces found")
    model_type = trainer.get(_TRAINER_MODEL_TYPE, 1)
    if model_type != 2:
        kind = {1: "unigram", 2: "BPE", 3: "word", 4: "char"}.get(model_type, model_type)
        raise ValueError(f"only BPE sentencepiece models are supported, got model_type {kind!r}")
    if not trainer.get(_TRAINER_BYTE_FALLBACK, 0):
        raise ValueError("only byte_fallback sentencepiece models are supported")
    if not normspec.get(_NORM_ESCAPE_WHITESPACES, 1):
        raise ValueError("sentencepiece models with escape_whitespaces=false are not supported")
    if trainer.get(_TRAINER_TREAT_WHITESPACE_AS_SUFFIX, 0):
        raise ValueError("sentencepiece models with treat_whitespace_as_suffix are not supported")

    vocab = {piece: i for i, (piece, _, _) in enumerate(pieces)}
    scores = {piece: score for piece, score, _ in pieces}
    unk_piece = trainer.get(_TRAINER_UNK_PIECE, b"<unk>")
    assert isinstance(unk_piece, bytes)

    normalizers: list[dict[str, Any]] = []
    charsmap = normspec.get(_NORM_PRECOMPILED_CHARSMAP, b"")
    assert isinstance(charsmap, bytes)
    if charsmap:
        normalizers.append({"type": "Precompiled", "precompiled_charsmap": base64.b64encode(charsmap).decode("ascii")})
    if normspec.get(_NORM_REMOVE_EXTRA_WHITESPACES, 1):
        normalizers.append({"type": "Strip", "strip_left": True, "strip_right": True})
        normalizers.append({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "})
    if normspec.get(_NORM_ADD_DUMMY_PREFIX, 1):
        normalizers.append({"type": "Prepend", "prepend": "▁"})
    normalizers.append({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})

    added_tokens = [
        {
            "id": i,
            "content": piece,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": piece_type == _TYPE_CONTROL,
        }
        for i, (piece, _, piece_type) in enumerate(pieces)
        if piece_type in (_TYPE_CONTROL, _TYPE_USER_DEFINED)
    ]

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": unk_piece.decode("utf-8"),
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": _generate_merges(vocab, scores),
        },
    }
    return json.dumps(tokenizer_json, ensure_ascii=False)


def _generate_merges(vocab: dict[str, int], scores: dict[str, float]) -> list[tuple[str, str]]:
    """BPE merges recovered from the vocab, exactly like transformers'
    `generate_merges` with `vocab_scores`: every in-vocab split of a piece is
    a merge, ranked by the merged piece's *score* (descending) — sentencepiece
    BPE merges by score, and manually added pieces (e.g. Llama 2's whitespace
    runs, score 0) sort behind trained merges, unlike an ID-based ranking."""
    merges: list[tuple[str, str, float]] = []
    for merged, score in scores.items():
        local = []
        for i in range(1, len(merged)):
            left, right = merged[:i], merged[i:]
            if left in vocab and right in vocab:
                local.append((left, right, score))
        local.sort(key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda x: (x[2], len(x[0]), len(x[1])), reverse=True)
    return [(left, right) for left, right, _ in merges]
