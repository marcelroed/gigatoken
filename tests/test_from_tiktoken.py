import pytest
import tiktoken

from gigatoken.gigatoken_rs import BPETokenizer


@pytest.fixture
def r50k(r50k_tiktoken_path) -> tuple[tiktoken.Encoding, BPETokenizer]:
    """Return (tiktoken_encoding, BPETokenizer) pair for r50k_base."""
    tt = tiktoken.get_encoding("r50k_base")
    bpe = BPETokenizer.from_tiktoken(r50k_tiktoken_path)
    return tt, bpe


def _assert_same(tt_enc, bpe_tok, text: str):
    expected = tt_enc.encode(text)
    actual = bpe_tok.encode(text.encode("utf-8")).tolist()
    assert actual == expected, f"Mismatch for {text!r}:\n  tiktoken: {expected}\n  gigatoken:    {actual}"


SIMPLE_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "",
    " ",
    "   leading and trailing spaces   ",
]

UNICODE_STRINGS = [
    "café résumé naïve",
    "日本語テスト",
    "emoji: 🚀🌍🎉",
    "mixed: hello 世界 🌎",
    "Ñoño año",
]

CODE_STRINGS = [
    "def foo(x: int) -> int:\n    return x + 1\n",
    "import os\nos.path.join('a', 'b')",
    "if __name__ == '__main__':\n    print('hello')",
    "SELECT * FROM users WHERE id = 1;",
]

EDGE_CASE_STRINGS = [
    "\n",
    "\n\n\n",
    "\t\t",
    "a" * 1000,
    "hello " * 200,
    "a\x00b\x01c",
    "\r\n\r\n",
]

PARAGRAPHS = [
    (
        "The Rust programming language helps you write faster, more reliable software. "
        "High-level ergonomics and low-level control are often at odds in programming language design; "
        "Rust challenges that conflict. Through balancing powerful technical capacity and a great developer "
        "experience, Rust gives you the option to control low-level details (such as memory usage) without "
        "all the hassle traditionally associated with such control."
    ),
    ("```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n"),
]


@pytest.mark.parametrize("text", SIMPLE_STRINGS, ids=lambda t: repr(t)[:40])
def test_simple(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", UNICODE_STRINGS, ids=lambda t: repr(t)[:40])
def test_unicode(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", CODE_STRINGS, ids=lambda t: repr(t)[:40])
def test_code(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", EDGE_CASE_STRINGS, ids=lambda t: repr(t)[:40])
def test_edge_cases(r50k, text):
    _assert_same(*r50k, text)


@pytest.mark.parametrize("text", PARAGRAPHS, ids=lambda t: repr(t)[:40])
def test_paragraphs(r50k, text):
    _assert_same(*r50k, text)


def test_roundtrip_token_count(r50k):
    """Token counts should match between tiktoken and gigatoken."""
    tt, bpe = r50k
    text = "Here is a moderately long sentence with some numbers 42 and symbols @#$%."
    assert len(tt.encode(text)) == len(bpe.encode(text.encode("utf-8")))


def test_single_characters(r50k):
    """Every printable ASCII character should tokenize identically."""
    tt, bpe = r50k
    for c in (chr(i) for i in range(32, 127)):
        _assert_same(tt, bpe, c)


def test_whitespace_variations(r50k):
    tt, bpe = r50k
    for text in ["a b", "a  b", "a   b", "a\tb", "a\nb", "a\r\nb", "a \n b"]:
        _assert_same(tt, bpe, text)


def test_repeated_tokens(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "the " * 500)
    _assert_same(tt, bpe, "aaaa" * 250)


def test_mixed_scripts(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "Hello мир 世界 مرحبا")
    _assert_same(tt, bpe, "price: €100 or ¥10000")


def test_json_like(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, '{"key": "value", "num": 123, "arr": [1, 2, 3]}')


def test_url_like(r50k):
    tt, bpe = r50k
    _assert_same(tt, bpe, "https://example.com/path?query=value&other=123#fragment")


def test_endoftext_added_token(r50k):
    """<|endoftext|> encodes to id 50256 and round-trips, matching both
    tiktoken with specials allowed and the tokenizer.json-loaded GPT-2."""
    tt, bpe = r50k
    text = "Hello world.<|endoftext|>Next document."
    expected = tt.encode(text, allowed_special="all")
    actual = bpe.encode(text.encode("utf-8")).tolist()
    assert actual == expected
    assert 50256 in actual
    assert bpe.decode(actual) == text.encode("utf-8")


def test_multiline_code(r50k):
    tt, bpe = r50k
    code = """class Foo:
    def __init__(self, x):
        self.x = x

    def bar(self):
        return self.x * 2
"""
    _assert_same(tt, bpe, code)
