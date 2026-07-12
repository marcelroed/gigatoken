#!/usr/bin/env python3
"""Benchmark: Gigatoken vs HuggingFace tokenizers for BPE training and encoding.

Run with: uv run python tests/bench_train_encode.py
"""

import json
import shutil
import statistics
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from gigatoken import train_bpe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOKENIZER_JSON = DATA_DIR / "tinyllama_tokenizer.json"


def _ensure_tokenizer():
    """Download TinyLlama tokenizer from HF if not already in data/."""
    if TOKENIZER_JSON.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.json")
    with open(cached) as f:
        data = json.load(f)
    merges = data.get("model", {}).get("merges", [])
    if merges and isinstance(merges[0], str):
        data["model"]["merges"] = [m.split(" ", 1) for m in merges]
    with open(TOKENIZER_JSON, "w") as f:
        json.dump(data, f, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Corpus generation
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
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The only thing we have to fear is fear itself.",
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
    "CamelCase and snake_case and kebab-case and SCREAMING_SNAKE_CASE",
]


def generate_corpus(target_kb: int) -> str:
    """Generate a corpus of approximately target_kb kilobytes."""
    base = "\n".join(_SENTENCES)
    base_size = len(base.encode("utf-8"))
    repeats = max(1, (target_kb * 1024) // base_size)
    return "\n".join(_SENTENCES * repeats)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def timed(fn, n_runs: int = 3, warmup: int = 0):
    """Run fn() n_runs times, return (median_seconds, all_times)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), times, result


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------


def bench_training(corpus_size_kb: int = 1024, vocab_size: int = 2000, n_runs: int = 3):
    corpus = generate_corpus(corpus_size_kb)
    corpus_bytes = corpus.encode("utf-8")
    actual_mb = len(corpus_bytes) / 1e6

    print("=" * 65)
    print(" BPE Training Benchmark")
    print("=" * 65)
    print(f" Corpus size:     {actual_mb:.1f} MB")
    print(f" Vocab size:      {vocab_size}")
    print(f" Runs:            {n_runs}")
    print()

    # -- Gigatoken --
    def train_gigatoken():
        return train_bpe(corpus_bytes, vocab_size, [])

    gigatoken_median, gigatoken_times, _ = timed(train_gigatoken, n_runs=n_runs)

    # -- HF tokenizers --
    def train_hf():
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True
        )
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator([corpus], trainer=trainer, length=1)
        return tok

    hf_median, hf_times, _ = timed(train_hf, n_runs=n_runs)

    speedup = hf_median / gigatoken_median if gigatoken_median > 0 else float("inf")

    print(f" {'Implementation':<20} {'Median (s)':>10} {'Min (s)':>10} {'Speedup':>10}")
    print(f" {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f" {'HF tokenizers':<20} {hf_median:>10.3f} {min(hf_times):>10.3f} {'1.0x':>10}"
    )
    print(
        f" {'Gigatoken':<20} {gigatoken_median:>10.3f} {min(gigatoken_times):>10.3f} {speedup:>9.1f}x"
    )
    print()
    return gigatoken_median, hf_median


# ---------------------------------------------------------------------------
# Encoding benchmark
# ---------------------------------------------------------------------------


def bench_encoding(text_size_kb: int = 100, n_runs: int = 5):
    _ensure_tokenizer()
    if not TOKENIZER_JSON.exists():
        print(f"Skipping encoding benchmark: {TOKENIZER_JSON} not found")
        return

    print("=" * 65)
    print(" Encoding Benchmark (TinyLlama tokenizer)")
    print("=" * 65)

    # Generate test text
    corpus = generate_corpus(text_size_kb)
    lines = [line for line in corpus.split("\n") if line.strip()]
    total_bytes = sum(len(l.encode("utf-8")) for l in lines)
    total_mb = total_bytes / 1e6
    print(f" Text size:       {total_mb:.2f} MB ({len(lines)} lines)")
    print(f" Runs:            {n_runs}")
    print()

    # -- Load tokenizers --
    # Gigatoken: SentencePieceTokenizer
    from gigatoken.gigatoken_rs import SentencePieceTokenizer

    gigatoken_tok = SentencePieceTokenizer.from_hf(TOKENIZER_JSON)

    # HF: load from file (fast Rust backend, no BOS via encode without special tokens)
    hf_tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    # -- Sequential encoding --
    def encode_gigatoken():
        for line in lines:
            gigatoken_tok.encode(line)

    def encode_hf():
        for line in lines:
            hf_tok.encode(line, add_special_tokens=False)

    def encode_hf_batch():
        hf_tok.encode_batch(lines, add_special_tokens=False)

    gigatoken_median, gigatoken_times, _ = timed(encode_gigatoken, n_runs=n_runs, warmup=1)
    hf_median, hf_times, _ = timed(encode_hf, n_runs=n_runs, warmup=1)
    hf_batch_median, hf_batch_times, _ = timed(encode_hf_batch, n_runs=n_runs, warmup=1)

    def throughput(t):
        return total_mb / t if t > 0 else float("inf")

    speedup_seq = hf_median / gigatoken_median if gigatoken_median > 0 else float("inf")
    speedup_batch = hf_batch_median / gigatoken_median if gigatoken_median > 0 else float("inf")

    print(
        f" {'Implementation':<20} {'Median (s)':>10} {'MB/s':>10} {'vs HF seq':>10}"
    )
    print(f" {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f" {'HF sequential':<20} {hf_median:>10.3f} {throughput(hf_median):>10.1f} {'1.0x':>10}"
    )
    print(
        f" {'HF batch':<20} {hf_batch_median:>10.3f} {throughput(hf_batch_median):>10.1f} {hf_median / hf_batch_median:>9.1f}x"
    )
    print(
        f" {'Gigatoken sequential':<20} {gigatoken_median:>10.3f} {throughput(gigatoken_median):>10.1f} {speedup_seq:>9.1f}x"
    )
    print()

    # -- Verify outputs match --
    sample = lines[:10]
    mismatches = 0
    for text in sample:
        gigatoken_ids = gigatoken_tok.encode(text).tolist()
        hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
        if gigatoken_ids != hf_ids:
            mismatches += 1
            if mismatches <= 3:
                print(
                    f"  Mismatch: {text[:50]!r}..."
                    f"\n    Gigatoken: {gigatoken_ids[:8]}... ({len(gigatoken_ids)} tokens)"
                    f"\n    HF:    {hf_ids[:8]}... ({len(hf_ids)} tokens)"
                )
    if mismatches == 0:
        print(f" Encoding output verified: {len(sample)} samples match exactly.")
    else:
        print(f" WARNING: {mismatches}/{len(sample)} samples differ.")
    print()


# ---------------------------------------------------------------------------
# Training + encoding on same trained tokenizer
# ---------------------------------------------------------------------------


def bench_trained_encoding(corpus_size_kb: int = 256, vocab_size: int = 1000, n_runs: int = 5):
    """Train with both, convert to HF format, compare encoding speed."""

    # Byte-unicode tables for conversion
    def bytes_to_unicode(data: bytes) -> str:
        allowed = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        b2u: list[str] = [""] * 256
        for b in allowed:
            b2u[b] = chr(b)
        n = 0
        for b in range(256):
            if b2u[b] == "":
                b2u[b] = chr(256 + n)
                n += 1
        return "".join(b2u[b] for b in data)

    corpus = generate_corpus(corpus_size_kb)
    corpus_bytes = corpus.encode("utf-8")
    actual_mb = len(corpus_bytes) / 1e6

    print("=" * 65)
    print(" Trained Tokenizer Encoding Comparison")
    print("=" * 65)
    print(f" Training corpus:  {actual_mb:.1f} MB, vocab_size={vocab_size}")
    print()

    # Train Gigatoken
    print(" Training with Gigatoken...", flush=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(corpus_bytes, vocab_size, [])
    gigatoken_train_t = time.perf_counter() - start
    print(f"   done in {gigatoken_train_t:.2f}s")

    # Train HF
    print(" Training with HF tokenizers...", flush=True)
    start = time.perf_counter()
    hf_tok = Tokenizer(models.BPE())
    hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    hf_trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    hf_tok.train_from_iterator([corpus], trainer=hf_trainer, length=1)
    hf_tok.decoder = decoders.ByteLevel()
    hf_train_t = time.perf_counter() - start
    print(f"   done in {hf_train_t:.2f}s")

    # Convert Gigatoken to HF tokenizer
    hf_vocab = {bytes_to_unicode(v): k for k, v in vocab.items()}
    hf_merges = [(bytes_to_unicode(a), bytes_to_unicode(b)) for a, b in merges]
    gigatoken_hf_tok = Tokenizer(models.BPE(vocab=hf_vocab, merges=hf_merges))
    gigatoken_hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    gigatoken_hf_tok.decoder = decoders.ByteLevel()

    # Encode test texts
    test_texts = [line for line in corpus.split("\n") if line.strip()][:500]
    total_chars = sum(len(t) for t in test_texts)

    def encode_with(tok):
        for t in test_texts:
            tok.encode(t)

    gigatoken_enc_median, _, _ = timed(lambda: encode_with(gigatoken_hf_tok), n_runs=n_runs, warmup=1)
    hf_enc_median, _, _ = timed(lambda: encode_with(hf_tok), n_runs=n_runs, warmup=1)

    # Compare compression
    gigatoken_total_tokens = sum(len(gigatoken_hf_tok.encode(t).ids) for t in test_texts)
    hf_total_tokens = sum(len(hf_tok.encode(t).ids) for t in test_texts)

    print()
    print(f" {'Metric':<30} {'Gigatoken-trained':>15} {'HF-trained':>15}")
    print(f" {'-'*30} {'-'*15} {'-'*15}")
    print(f" {'Total tokens (500 lines)':<30} {gigatoken_total_tokens:>15,} {hf_total_tokens:>15,}")
    print(f" {'Chars / token':<30} {total_chars/gigatoken_total_tokens:>15.2f} {total_chars/hf_total_tokens:>15.2f}")
    print(f" {'Encode time (s)':<30} {gigatoken_enc_median:>15.4f} {hf_enc_median:>15.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    bench_training(corpus_size_kb=1024, vocab_size=2000, n_runs=3)
    bench_encoding(text_size_kb=100, n_runs=5)
    bench_trained_encoding(corpus_size_kb=256, vocab_size=1000, n_runs=3)


if __name__ == "__main__":
    main()
