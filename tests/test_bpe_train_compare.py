"""Compare BPE training: Gigatoken vs HuggingFace tokenizers library.

Trains both implementations on the same corpus and compares:
- Vocabulary size and structure
- Merge sequences (overlap and ordering)
- Encoding output and roundtrip correctness
"""

import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from gigatoken import train_bpe

from conftest import GPT2_B2U, gpt2_bytes_to_unicode as bytes_to_unicode


# ---------------------------------------------------------------------------
# Training corpus (~120 KB after repetition)
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The rain in Spain stays mainly in the plain.",
    "To be, or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "In the beginning was the Word, and the Word was with God.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The only thing we have to fear is fear itself.",
    "I think, therefore I am.",
    "That which does not kill us makes us stronger.",
    "To infinity and beyond!",
    "May the Force be with you.",
    "Elementary, my dear Watson.",
    "Houston, we have a problem.",
    "I'll be back.",
    "Once upon a time, there was a little girl named Lily. She lived in a big house with her family.",
    "One day, Lily found a beautiful butterfly in the garden. The butterfly had bright colors.",
    "The sun was shining and the birds were singing. It was a perfect day for an adventure.",
    "Tom and his friend went to the park to play. They had so much fun on the swings.",
    "The little dog ran across the yard, chasing after the ball. He was very happy.",
    "Mama said it was time for dinner. The children washed their hands and sat at the table.",
    "The stars came out at night and twinkled in the sky. The moon was big and round.",
    "def hello():\n    print('Hello, world!')\n    return 42\n",
    "for i in range(100):\n    total += items[i] * weights[i]\n",
    "class MyClass:\n    def __init__(self, value: int):\n        self.value = value\n",
    "The year 2024 saw 1,234,567 transactions worth $89.99 each.",
    "Temperature: -40 degrees equals -40 Fahrenheit, which is very cold indeed.",
    "Email: user@example.com, Phone: +1 (555) 123-4567",
    "https://www.example.com/path/to/resource?key=value&other=123#section",
    "The cafe served creme brulee and the diners loved it.",
    "CamelCase and snake_case and kebab-case and SCREAMING_SNAKE_CASE",
    "The the the the the the the the the the",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
]

CORPUS = "\n".join(_SENTENCES * 100)
CORPUS_BYTES = CORPUS.encode("utf-8")
VOCAB_SIZE = 500


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_hf(corpus: str, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([corpus], trainer=trainer, length=1)
    tok.decoder = decoders.ByteLevel()
    return tok


def _gigatoken_to_hf(vocab: dict, merges: list) -> Tokenizer:
    """Convert Gigatoken training output into an HF Tokenizer."""
    hf_vocab = {bytes_to_unicode(v): k for k, v in vocab.items()}
    hf_merges = [(bytes_to_unicode(a), bytes_to_unicode(b)) for a, b in merges]
    tok = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


# ---------------------------------------------------------------------------
# Module-scoped fixtures (train once, reuse across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gigatoken_result():
    return train_bpe(CORPUS_BYTES, VOCAB_SIZE, [])


@pytest.fixture(scope="module")
def hf_tokenizer():
    return _train_hf(CORPUS, VOCAB_SIZE)


@pytest.fixture(scope="module")
def gigatoken_as_hf(gigatoken_result):
    vocab, merges = gigatoken_result
    return _gigatoken_to_hf(vocab, merges)


# ---------------------------------------------------------------------------
# Helpers to extract HF merges
# ---------------------------------------------------------------------------


def _hf_merges(hf_tokenizer: Tokenizer) -> list[str]:
    raw = json.loads(hf_tokenizer.to_str())["model"]["merges"]
    # HF JSON may store merges as ["a", "b"] lists or "a b" strings
    if raw and isinstance(raw[0], list):
        return [f"{a} {b}" for a, b in raw]
    return raw


# ---------------------------------------------------------------------------
# Tests: vocabulary structure
# ---------------------------------------------------------------------------


def test_vocab_size(gigatoken_result, hf_tokenizer):
    vocab, _ = gigatoken_result
    assert len(vocab) == VOCAB_SIZE
    assert hf_tokenizer.get_vocab_size() == VOCAB_SIZE


def test_base_vocab_preserved(gigatoken_result, hf_tokenizer):
    """First 256 tokens should be single bytes in both."""
    vocab, _ = gigatoken_result
    for i in range(256):
        assert vocab[i] == bytes([i])

    hf_vocab = hf_tokenizer.get_vocab()
    for b in range(256):
        assert GPT2_B2U[b] in hf_vocab


def test_merges_count(gigatoken_result, hf_tokenizer):
    _, merges = gigatoken_result
    expected = VOCAB_SIZE - 256
    assert len(merges) == expected

    hf_merges = _hf_merges(hf_tokenizer)
    assert len(hf_merges) == expected


# ---------------------------------------------------------------------------
# Tests: merge comparison
# ---------------------------------------------------------------------------


def test_merges_identical(gigatoken_result, hf_tokenizer):
    """With HF-compatible tie-breaking, all merges should match exactly."""
    _, gigatoken_merges = gigatoken_result
    hf_merges = _hf_merges(hf_tokenizer)

    gigatoken_unicode = [
        f"{bytes_to_unicode(a)} {bytes_to_unicode(b)}" for a, b in gigatoken_merges
    ]

    first_divergence = None
    for i, (jm, hm) in enumerate(zip(gigatoken_unicode, hf_merges)):
        if jm != hm:
            first_divergence = i
            break

    if first_divergence is not None:
        j = gigatoken_unicode[first_divergence]
        h = hf_merges[first_divergence]
        pytest.fail(
            f"Merge {first_divergence} differs: gigatoken={j!r}, hf={h!r}"
        )

    assert len(gigatoken_unicode) == len(hf_merges)


# ---------------------------------------------------------------------------
# Tests: encoding with the trained tokenizers
# ---------------------------------------------------------------------------

_TEST_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def foo(x): return x + 1",
    "1234567890",
    "Once upon a time there was a little girl.",
    "The children washed their hands and sat at the table.",
]


def test_gigatoken_tokenizer_roundtrip(gigatoken_as_hf):
    """Gigatoken-trained tokenizer encodes and decodes correctly."""
    for text in _TEST_TEXTS:
        encoded = gigatoken_as_hf.encode(text)
        decoded = gigatoken_as_hf.decode(encoded.ids)
        assert decoded == text, f"Roundtrip failed for {text!r}: got {decoded!r}"


def test_hf_tokenizer_roundtrip(hf_tokenizer):
    """HF-trained tokenizer encodes and decodes correctly."""
    for text in _TEST_TEXTS:
        encoded = hf_tokenizer.encode(text)
        decoded = hf_tokenizer.decode(encoded.ids)
        assert decoded == text, f"Roundtrip failed for {text!r}: got {decoded!r}"


def test_encoding_identical(gigatoken_as_hf, hf_tokenizer):
    """With identical merges, both tokenizers should split text into the same tokens.

    The raw token IDs differ (Gigatoken uses byte-value IDs, HF uses unicode-codepoint-
    rank IDs), but the tokenization boundaries and token content must be identical.
    """
    for text in _TEST_TEXTS:
        gigatoken_enc = gigatoken_as_hf.encode(text)
        hf_enc = hf_tokenizer.encode(text)
        gigatoken_tokens = gigatoken_enc.tokens
        hf_tokens = hf_enc.tokens
        assert gigatoken_tokens == hf_tokens, (
            f"Tokenization mismatch for {text!r}:\n"
            f"  gigatoken: {gigatoken_tokens}\n"
            f"  hf:    {hf_tokens}"
        )


def test_vocab_tokens_are_valid(gigatoken_result):
    """Every merged token should be the concatenation of its merge components."""
    vocab, merges = gigatoken_result
    for i, (left, right) in enumerate(merges):
        token_id = 256 + i  # byte tokens 0-255, then merges
        expected = left + right
        assert vocab[token_id] == expected, (
            f"Merge {i}: {left!r} + {right!r} should be {expected!r}, "
            f"but vocab[{token_id}] = {vocab[token_id]!r}"
        )
