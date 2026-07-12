"""Fixtures for end-to-end tokenizer parity tests against HuggingFace.

Every test in this directory is parametrized over TOKENIZER_SPECS via the
session-scoped `spec` fixture, so adding a tokenizer here adds it to the
whole suite (small-string parity, added-token handling, decode roundtrip,
and the large-scale OWT comparison).

A spec's `name` must have a matching `<name>_tokenizer_path` fixture in
tests/conftest.py that downloads the tokenizer.json.
"""

from dataclasses import dataclass

import pytest
from tokenizers import Tokenizer

from gigatoken.gigatoken_rs import BPETokenizer


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    eot_text: str  # the end-of-text special token
    eot_id: int
    normalizes_nfc: bool = False  # tokenizer.json declares an NFC normalizer


TOKENIZER_SPECS = {
    s.name: s
    for s in [
        TokenizerSpec(name="gpt2", eot_text="<|endoftext|>", eot_id=50256),
        TokenizerSpec(name="olmo3", eot_text="<|endoftext|>", eot_id=100257),
        TokenizerSpec(
            name="qwen2",
            eot_text="<|endoftext|>",
            eot_id=151643,
            normalizes_nfc=True,
        ),
        TokenizerSpec(
            name="qwen3_5",
            eot_text="<|endoftext|>",
            eot_id=248044,
            normalizes_nfc=True,
        ),
        TokenizerSpec(
            name="modernbert",
            eot_text="<|endoftext|>",
            eot_id=50279,
            normalizes_nfc=True,
        ),
        TokenizerSpec(name="deepseek_v3", eot_text="<｜end▁of▁sentence｜>", eot_id=1),
        TokenizerSpec(name="deepseek_v4", eot_text="<｜end▁of▁sentence｜>", eot_id=1),
    ]
}


@pytest.fixture(scope="session", params=sorted(TOKENIZER_SPECS))
def spec(request) -> TokenizerSpec:
    return TOKENIZER_SPECS[request.param]


@pytest.fixture(scope="session")
def tokenizer_path(spec, request):
    return request.getfixturevalue(f"{spec.name}_tokenizer_path")


@pytest.fixture(scope="session")
def hf_tok(tokenizer_path):
    return Tokenizer.from_file(str(tokenizer_path))


@pytest.fixture(scope="session")
def gigatoken_tok(tokenizer_path):
    return BPETokenizer.from_hf(tokenizer_path)
