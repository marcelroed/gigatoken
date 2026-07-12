"""Tests for the gigatoken.TiktokenCompat wrapper.

gigatoken.Tokenizer(source).as_tiktoken() adapts a gigatoken Tokenizer to the
`tiktoken.Encoding` API so it can replace a tiktoken encoding in existing
code. Parity is asserted against the real r50k_base encoding.
"""

import pytest
import tiktoken

import gigatoken

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "   leading and trailing spaces   ",
    "café résumé naïve",
    "emoji: 🚀🌍🎉",
    "def foo(x: int) -> int:\n    return x + 1\n",
    "日本語テスト",
    "",
]

ENDOFTEXT = "<|endoftext|>"
SPECIAL_TEXT = f"Hello world.{ENDOFTEXT}Next document."


@pytest.fixture(scope="module")
def r50k_ref() -> tiktoken.Encoding:
    return tiktoken.get_encoding("r50k_base")


@pytest.fixture(scope="module")
def r50k_compat(r50k_tiktoken_path) -> gigatoken.TiktokenCompat:
    return gigatoken.Tokenizer.from_tiktoken(r50k_tiktoken_path).as_tiktoken()


# The part of the tiktoken.Encoding surface that TiktokenCompat implements.
# Interface parity is asserted against the real class.
CORE_API = [
    "encode",
    "encode_ordinary",
    "encode_batch",
    "encode_ordinary_batch",
    "encode_single_token",
    "decode",
    "decode_bytes",
    "decode_batch",
    "decode_bytes_batch",
    "decode_single_token_bytes",
    "decode_tokens_bytes",
    "token_byte_values",
    "special_tokens_set",
    "eot_token",
    "n_vocab",
    "max_token_value",
    "name",
]


def test_tiktoken_compat_interface(r50k_ref, r50k_compat):
    for name in CORE_API:
        assert hasattr(r50k_ref, name), f"{name} not in reference API"
        assert hasattr(r50k_compat, name), f"{name} missing from TiktokenCompat"
        if callable(getattr(r50k_ref, name)):
            assert callable(getattr(r50k_compat, name)), f"{name} should be callable"


@pytest.mark.parametrize("text", TEXTS, ids=lambda t: repr(t)[:40])
def test_encode_matches(r50k_ref, r50k_compat, text):
    assert r50k_compat.encode(text) == r50k_ref.encode(text)
    assert r50k_compat.encode_ordinary(text) == r50k_ref.encode_ordinary(text)


def test_encode_batch_matches(r50k_ref, r50k_compat):
    assert r50k_compat.encode_batch(TEXTS) == r50k_ref.encode_batch(TEXTS)
    assert r50k_compat.encode_ordinary_batch(TEXTS) == r50k_ref.encode_ordinary_batch(TEXTS)


def test_encode_allowed_special(r50k_ref, r50k_compat):
    expected = r50k_ref.encode(SPECIAL_TEXT, allowed_special="all")
    assert r50k_compat.encode(SPECIAL_TEXT, allowed_special="all") == expected
    assert r50k_compat.encode(SPECIAL_TEXT, allowed_special={ENDOFTEXT}) == expected
    assert r50k_compat.encode_batch([SPECIAL_TEXT], allowed_special="all") == [expected]
    assert r50k_ref.eot_token in expected


def test_encode_disallowed_special_raises(r50k_ref, r50k_compat):
    """tiktoken's default rejects special tokens found in the text."""
    for ref in (r50k_ref, r50k_compat):
        with pytest.raises(ValueError, match="disallowed special token"):
            ref.encode(SPECIAL_TEXT)
        with pytest.raises(ValueError, match="disallowed special token"):
            ref.encode(SPECIAL_TEXT, allowed_special=set(), disallowed_special={ENDOFTEXT})
        with pytest.raises(ValueError, match="disallowed special token"):
            ref.encode_batch([SPECIAL_TEXT])


def test_encode_special_as_ordinary_raises_loudly(r50k_compat):
    """tiktoken would encode the special as plain text here; the gigatoken
    backend cannot, and must raise rather than silently diverge."""
    with pytest.raises(NotImplementedError):
        r50k_compat.encode_ordinary(SPECIAL_TEXT)
    with pytest.raises(NotImplementedError):
        r50k_compat.encode_ordinary_batch([SPECIAL_TEXT])
    with pytest.raises(NotImplementedError):
        r50k_compat.encode(SPECIAL_TEXT, disallowed_special=())


def test_decode_matches(r50k_ref, r50k_compat):
    for text in TEXTS:
        tokens = r50k_ref.encode(text)
        assert r50k_compat.decode(tokens) == r50k_ref.decode(tokens) == text
        assert r50k_compat.decode_bytes(tokens) == r50k_ref.decode_bytes(tokens)
        assert r50k_compat.decode_tokens_bytes(tokens) == r50k_ref.decode_tokens_bytes(tokens)
    batch = [r50k_ref.encode(t) for t in TEXTS]
    assert r50k_compat.decode_batch(batch) == r50k_ref.decode_batch(batch)
    assert r50k_compat.decode_bytes_batch(batch) == r50k_ref.decode_bytes_batch(batch)


def test_single_token_conversions_match(r50k_ref, r50k_compat):
    for token in [0, 1000, 50255]:
        piece = r50k_ref.decode_single_token_bytes(token)
        assert r50k_compat.decode_single_token_bytes(token) == piece
        assert r50k_compat.encode_single_token(piece) == r50k_ref.encode_single_token(piece) == token
    assert r50k_compat.encode_single_token(ENDOFTEXT) == r50k_ref.encode_single_token(ENDOFTEXT)
    with pytest.raises(KeyError):
        r50k_compat.encode_single_token(b"not a single token, definitely")


def test_vocab_and_specials_match(r50k_ref, r50k_compat):
    assert r50k_compat.n_vocab == r50k_ref.n_vocab
    assert r50k_compat.max_token_value == r50k_ref.max_token_value
    assert r50k_compat.eot_token == r50k_ref.eot_token
    assert r50k_compat.special_tokens_set == r50k_ref.special_tokens_set
    assert r50k_compat.is_special_token(r50k_ref.eot_token)
    assert not r50k_compat.is_special_token(0)
    assert sorted(r50k_compat.token_byte_values()) == sorted(r50k_ref.token_byte_values())


def test_as_tiktoken_from_hf_source(gpt2_tokenizer_path, r50k_ref):
    """GPT-2's tokenizer.json is the same vocabulary as r50k_base, so a
    tokenizer.json-loaded compat matches the real tiktoken encoding too."""
    compat = gigatoken.Tokenizer(gpt2_tokenizer_path).as_tiktoken()
    assert compat.special_tokens_set == {ENDOFTEXT}
    assert compat.eot_token == r50k_ref.eot_token
    for text in TEXTS:
        assert compat.encode(text) == r50k_ref.encode(text)
    assert compat.encode(SPECIAL_TEXT, allowed_special="all") == r50k_ref.encode(SPECIAL_TEXT, allowed_special="all")
    with pytest.raises(ValueError, match="disallowed special token"):
        compat.encode(SPECIAL_TEXT)


def test_tiktoken_compat_requires_gigatoken_tokenizer(r50k_ref):
    with pytest.raises(TypeError, match="as_tiktoken"):
        gigatoken.TiktokenCompat(r50k_ref)
