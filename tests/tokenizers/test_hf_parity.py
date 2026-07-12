"""Test that BPETokenizer.from_hf produces token IDs identical to HuggingFace
`tokenizers` for every tokenizer in TOKENIZER_SPECS (see conftest.py), so it
can substitute it entirely.

Covers the fast pretokenizers, added-token extraction, NFC normalization
(Qwen2), and the BPE merge itself. The large-scale test streams OWT and
encodes both sides in parallel: HF via encode_batch_fast (rayon), gigatoken via
encode_batch (rayon).

Environment knobs for the OWT test:
- OWT_MAX_BYTES: cap on bytes read (default 100 MB; 0 = the whole ~12 GB file)
- OWT_SLAB_BYTES: bytes per streamed comparison slab (default 64 MiB)
"""

import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import pytest

OWT_PATH = Path.home() / "data" / "owt_train.txt"
SLAB_BYTES = int(os.environ.get("OWT_SLAB_BYTES", 64 * 1024 * 1024))
_owt_max = int(os.environ.get("OWT_MAX_BYTES", 100 * 1024 * 1024))
OWT_MAX_BYTES = _owt_max or None  # 0 = full file
DOC_BYTES = 1024 * 1024  # target document size within a slab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_ids_match(hf_tok, gigatoken_tok, text: str):
    # add_special_tokens=False: gigatoken encodes raw text without template
    # wrapping, so HF post-processors that inject tokens (ModernBERT's
    # TemplateProcessing adds [CLS]/[SEP]) must be disabled. A no-op for the
    # other tokenizers, whose ByteLevel post-processors add nothing.
    hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
    gigatoken_ids = gigatoken_tok.encode(text.encode("utf-8")).tolist()
    assert gigatoken_ids == hf_ids, (
        f"Mismatch for {text!r}:\n  HF:    {hf_ids}\n  gigatoken: {gigatoken_ids}"
    )


# ---------------------------------------------------------------------------
# Small cases
# ---------------------------------------------------------------------------

TEXTS = [
    "Hello",
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "12 345 6789",
    "3.14159",
    "",
    " ",
    "   leading and trailing spaces   ",
    "don't they'LL we'Ve THEY'RE it'S",
    "café résumé naïve",
    "emoji: 😒🌍🎉",
    "mixed: hello 世界 🌎",
    "١٢٣٤٥ ٦٧",
    "def foo(x: int) -> int:\n    return x + 1\n",
    "import os\nos.path.join('a', 'b')",
    'SELECT * FROM users WHERE id = 1;',
    '{"key": "value", "num": 123, "arr": [1, 2, 3]}',
    "https://example.com/path?query=value&other=123#fragment",
    "\n",
    "\n\n\n",
    "\t\t",
    "hi!\n\ndef",
    "hello  \n\n  ",
    "x \n\n ",
    "a" * 500,
    "hello " * 100,
    "\r\n\r\n",
    "日本語テスト",
    "Привет мир",
    "مرحبا بالعالم",
    "price: $5.99!",
    # Non-NFC input (decomposed combining characters); exercises the NFC
    # normalizer on tokenizers that declare one.
    "café décomposed",
    "Ŷ (fitted Y)",
    "κάμιλος",
    "́leading combining",
]

# Texts exercising added/special tokens. Parity must hold for every
# tokenizer: strings that aren't added tokens for a given tokenizer are
# simply encoded as regular text on both sides.
SPECIAL_TEXTS = [
    "<|endoftext|>",
    "a<|endoftext|>b",
    "hello <|endoftext|> world",
    "<|endoftext|><|endoftext|>",
    "text<|endoftext|>\nmore text",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    pass",
    "<|im_start|>user\nhi<|im_end|>",
    "call |||PHONE_NUMBER||| or mail |||EMAIL_ADDRESS|||",
    "from |||IP_ADDRESS|||!",
    "<|extra_id_0|> and <|extra_id_10|>",
    "<|pad|><|endofprompt|>",
    "á<|endoftext|>é",
    "<｜begin▁of▁sentence｜>hello<｜end▁of▁sentence｜>",
    "a<｜▁pad▁｜>b",
    "<｜place▁holder▁no▁3｜> and <｜place▁holder▁no▁100｜>",
    "<|end▁of▁sentence|>",  # ASCII pipes: not the DeepSeek token
    "<｜end of sentence｜>",
    # Lookalikes that must NOT match an added token
    "<|endoftext",
    "endoftext|>",
    "<|endoftext |>",
    "<|ENDOFTEXT|>",
    "||PHONE_NUMBER|||",
    "|||PHONE_NUMBER||",
    "|||UNKNOWN_TAG|||",
    "< | endoftext | >",
    "|||||",
    "<<|endoftext|>>",
    "||||||PHONE_NUMBER||||||",
    "text ||||PHONE_NUMBER|||| text",
]


@pytest.mark.parametrize("text", TEXTS, ids=lambda t: repr(t)[:50])
def test_encode_matches_hf(hf_tok, gigatoken_tok, text):
    _assert_ids_match(hf_tok, gigatoken_tok, text)


@pytest.mark.parametrize("text", SPECIAL_TEXTS, ids=lambda t: repr(t)[:50])
def test_added_tokens_match_hf(hf_tok, gigatoken_tok, text):
    _assert_ids_match(hf_tok, gigatoken_tok, text)


def test_endoftext_id(spec, gigatoken_tok):
    ids = gigatoken_tok.encode(f"a{spec.eot_text}b".encode()).tolist()
    assert spec.eot_id in ids


@pytest.mark.parametrize("text", TEXTS + SPECIAL_TEXTS, ids=lambda t: repr(t)[:50])
def test_decode_roundtrip(spec, gigatoken_tok, text):
    ids = gigatoken_tok.encode(text.encode("utf-8"))
    # An NFC-normalizing tokenizer roundtrips to the normalized form,
    # exactly like HF.
    expected = unicodedata.normalize("NFC", text) if spec.normalizes_nfc else text
    assert gigatoken_tok.decode(ids) == expected.encode("utf-8")


def test_encode_batch_matches_encode(gigatoken_tok):
    docs = [t.encode("utf-8") for t in TEXTS + SPECIAL_TEXTS]
    batched = gigatoken_tok.encode_batch(docs)
    for doc, batch_ids in zip(docs, batched):
        assert gigatoken_tok.encode(doc).tolist() == batch_ids.tolist()


# ---------------------------------------------------------------------------
# Large-scale OWT test
# ---------------------------------------------------------------------------


def _iter_slabs(path: Path, max_bytes: int | None):
    """Stream the file as slabs cut at newline boundaries (newline kept)."""
    remaining = max_bytes if max_bytes is not None else float("inf")
    carry = b""
    with open(path, "rb") as f:
        while remaining > 0:
            chunk = f.read(int(min(SLAB_BYTES, remaining)))
            if not chunk:
                break
            remaining -= len(chunk)
            slab = carry + chunk
            cut = slab.rfind(b"\n")
            if cut == -1:
                carry = slab
                continue
            carry, slab = slab[cut + 1 :], slab[: cut + 1]
            yield slab
    # Trailing piece: trim a partial UTF-8 sequence (possible when max_bytes
    # cuts mid-char), then emit.
    while carry:
        try:
            carry.decode("utf-8")
            break
        except UnicodeDecodeError:
            carry = carry[:-1]
    if carry:
        yield carry


def _split_docs(slab: bytes) -> list[bytes]:
    """Split a slab into ~DOC_BYTES documents at newline boundaries."""
    docs = []
    start = 0
    while start < len(slab):
        end = min(start + DOC_BYTES, len(slab))
        if end < len(slab):
            nl = slab.find(b"\n", end)
            end = len(slab) if nl == -1 else nl + 1
        docs.append(slab[start:end])
        start = end
    return docs


def _first_diff(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@pytest.mark.skipif(not OWT_PATH.exists(), reason="OWT data not available")
def test_owt_matches_hf(hf_tok, gigatoken_tok):
    """Compare token IDs against HF on OWT (100 MB unless OWT_MAX_BYTES).

    Both sides are internally multithreaded: HF's encode_batch_fast and
    gigatoken's encode_batch each fan documents out over rayon.
    """
    total_bytes = 0
    total_docs = 0
    total_tokens = 0
    mismatches = 0
    t0 = time.time()

    for slab_idx, slab in enumerate(_iter_slabs(OWT_PATH, OWT_MAX_BYTES)):
        docs = _split_docs(slab)
        texts = [d.decode("utf-8") for d in docs]

        hf_encodings = hf_tok.encode_batch_fast(texts, add_special_tokens=False)
        gigatoken_arrays = gigatoken_tok.encode_batch(docs)

        for i, (enc, jt) in enumerate(zip(hf_encodings, gigatoken_arrays)):
            hf_ids = np.asarray(enc.ids, dtype=np.uint32)
            total_tokens += len(hf_ids)
            if not np.array_equal(hf_ids, jt):
                mismatches += 1
                if mismatches <= 5:
                    h, j = hf_ids.tolist(), jt.tolist()
                    d = _first_diff(h, j)
                    ctx_lo = max(0, d - 5)
                    print(
                        f"\nMISMATCH slab {slab_idx} doc {i} "
                        f"(byte ~{total_bytes + i * DOC_BYTES}):\n"
                        f"  lens: HF {len(h)} vs gigatoken {len(j)}, first diff at {d}\n"
                        f"  HF:    ...{h[ctx_lo:d + 5]}\n"
                        f"  gigatoken: ...{j[ctx_lo:d + 5]}\n"
                        f"  HF toks there: "
                        f"{[hf_tok.decode([t]) for t in h[ctx_lo:d + 5]]!r}"
                    )

        total_bytes += len(slab)
        total_docs += len(docs)
        elapsed = time.time() - t0
        print(
            f"slab {slab_idx}: {total_bytes / 1e9:.2f} GB, {total_docs} docs, "
            f"{total_tokens / 1e6:.0f}M tokens, {mismatches} mismatches, "
            f"{total_bytes / 1e6 / elapsed:.0f} MB/s",
            flush=True,
        )
        assert mismatches == 0, f"{mismatches} mismatching documents so far"

    print(
        f"\nDone: {total_bytes / 1e9:.2f} GB, {total_docs} docs, "
        f"{total_tokens} tokens, all identical to HF."
    )
    assert total_docs > 0
