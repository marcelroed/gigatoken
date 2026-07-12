"""Verify gigatoken encoding matches HuggingFace on curated DCLM data.

The corpus (~20 MB) is DCLM-baseline documents selected for tokenizer-hostile
content — CJK/RTL scripts, NFC-divergent text, emoji, control whitespace,
code, unbroken 80+ char tokens, 128 KB+ documents — from a shard downloaded
into the HuggingFace cache on first use (dclm_fixture.py).
"""

import re
import unicodedata

import awkward as ak
import pytest
from tokenizers import Tokenizer as HFTokenizer

import gigatoken
from gigatoken import JsonlFileSource

TOKENIZER_FIXTURES = [
    "tinyllama_tokenizer_path",  # SentencePiece backend
    "olmo3_tokenizer_path",  # byte-level BPE backend
    "deepseek_v3_tokenizer_path",  # byte-level BPE with NFC normalizer
]


def test_dclm_sample_is_diverse(dclm_docs):
    """The fixture holds ~20 MB and contains the edge cases it promises."""
    total = sum(len(d.encode("utf-8")) for d in dclm_docs)
    assert total >= 15_000_000
    assert len(dclm_docs) > 1_000
    assert any(not unicodedata.is_normalized("NFC", d) for d in dclm_docs)
    assert any(re.search(r"[一-鿿]", d) for d in dclm_docs)  # CJK
    assert any(re.search(r"[Ѐ-ӿ]", d) for d in dclm_docs)  # Cyrillic
    assert any(re.search(r"[؀-ۿ]", d) for d in dclm_docs)  # Arabic
    assert any(re.search(r"[\U0001f300-\U0001faff]", d) for d in dclm_docs)  # emoji
    assert any("\t" in d or "\r" in d for d in dclm_docs)
    assert any(re.search(r"\S{80,}", d) for d in dclm_docs)
    assert any(len(d.encode("utf-8")) >= 131_072 for d in dclm_docs)
    assert any(len(d.encode("utf-8")) < 200 for d in dclm_docs)


@pytest.mark.parametrize("tok_fixture", TOKENIZER_FIXTURES)
def test_encode_dclm_matches_hf(tok_fixture, request, dclm_docs):
    """Exact token-ID parity with HuggingFace over the full DCLM sample."""
    path = request.getfixturevalue(tok_fixture)
    hf_tok = HFTokenizer.from_file(str(path))
    tok = gigatoken.Tokenizer(path)

    hf_ids = [e.ids for e in hf_tok.encode_batch(dclm_docs, add_special_tokens=False)]
    got = ak.to_list(tok.encode_batch(dclm_docs))

    mismatches = 0
    for i, (ours, theirs) in enumerate(zip(got, hf_ids)):
        if ours == theirs:
            continue
        mismatches += 1
        if mismatches > 5:
            continue
        for j in range(min(len(ours), len(theirs))):
            if ours[j] != theirs[j]:
                ctx = bytes(tok.decode(ours[max(0, j - 3) : j]))
                print(f"\n  Doc {i}: first diff at token {j}, gigatoken={ours[j]}, hf={theirs[j]}, context=...{ctx!r}")
                break
        else:
            print(f"\n  Doc {i}: length differs gigatoken={len(ours)}, hf={len(theirs)}")

    assert mismatches == 0, f"{mismatches}/{len(dclm_docs)} documents differ"


def test_encode_batch_matches_single_docs(olmo3_tokenizer_path, dclm_docs):
    """Parallel batch encoding agrees with one-doc-at-a-time encoding."""
    tok = gigatoken.Tokenizer(olmo3_tokenizer_path)
    batch = ak.to_list(tok.encode_batch(dclm_docs))
    for doc, row in zip(dclm_docs, batch):
        assert tok.encode(doc).tolist() == row


def test_encode_files_dclm_fixture(olmo3_tokenizer_path, dclm_sample_path, dclm_docs):
    """encode_files on the fixture .jsonl.zst matches encode_batch on its docs."""
    tok = gigatoken.Tokenizer(olmo3_tokenizer_path)
    from_file = ak.to_list(tok.encode_files(JsonlFileSource([dclm_sample_path])))
    assert len(from_file) == len(dclm_docs)
    assert from_file == ak.to_list(tok.encode_batch(dclm_docs))


def test_decode_roundtrip_dclm(olmo3_tokenizer_path, dclm_docs):
    """Byte-level BPE without normalizer must roundtrip every document."""
    tok = gigatoken.Tokenizer(olmo3_tokenizer_path)
    for doc in dclm_docs:
        assert bytes(tok.decode(tok.encode(doc))) == doc.encode("utf-8")
