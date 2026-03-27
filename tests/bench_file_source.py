#!/usr/bin/env python3
"""Benchmark: single-threaded pretokenization of a .jsonl.zst file.

Jeton (Rust, single-threaded) vs HuggingFace tokenizers ByteLevel pretokenizer.

Run with: uv run python tests/bench_file_source.py
"""

import json
import os
import time

import zstandard

DATA_PATH = os.path.expanduser("~/data/dclm-baseline/shard_00000000_processed.jsonl.zst")
FIELD = "text"


def load_documents(path: str, field: str, max_docs: int | None = None) -> list[str]:
    """Decompress and parse JSONL, return list of text documents."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        text_data = reader.read().decode("utf-8")

    docs = []
    for line in text_data.split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        t = obj.get(field)
        if t:
            docs.append(t)
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def bench_jeton_single_thread(docs: list[str], n_runs: int = 3):
    """Pretokenize all documents with Jeton on a single thread."""
    from jeton.jeton_rs import pretokenizer

    times = []
    total_pretokens = 0
    for run in range(n_runs):
        count = 0
        start = time.perf_counter()
        for doc in docs:
            for _tok in pretokenizer(doc.encode("utf-8")):
                count += 1
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_pretokens = count

    return times, total_pretokens


def bench_hf_single_thread(docs: list[str], n_runs: int = 3):
    """Pretokenize all documents with HF tokenizers ByteLevel on a single thread."""
    from tokenizers.pre_tokenizers import ByteLevel

    pretok = ByteLevel(add_prefix_space=False, use_regex=True)

    times = []
    total_pretokens = 0
    for run in range(n_runs):
        count = 0
        start = time.perf_counter()
        for doc in docs:
            tokens = pretok.pre_tokenize_str(doc)
            count += len(tokens)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_pretokens = count

    return times, total_pretokens


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return

    print(f"Loading {DATA_PATH}...")
    start = time.perf_counter()
    docs = load_documents(DATA_PATH, FIELD)
    load_time = time.perf_counter() - start
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_mb = total_bytes / 1e6

    print(f"  {len(docs)} documents, {total_mb:.1f} MB, loaded in {load_time:.1f}s")
    print()

    n_runs = 3

    print("=" * 65)
    print(" Single-Thread Pretokenization Benchmark")
    print("=" * 65)
    print(f" Data:  {len(docs)} docs, {total_mb:.1f} MB")
    print(f" Runs:  {n_runs}")
    print()

    jeton_times, jeton_tokens = bench_jeton_single_thread(docs, n_runs)
    hf_times, hf_tokens = bench_hf_single_thread(docs, n_runs)

    jeton_best = min(jeton_times)
    hf_best = min(hf_times)
    speedup = hf_best / jeton_best if jeton_best > 0 else float("inf")

    print(f" {'Implementation':<25} {'Best (s)':>10} {'MB/s':>10} {'Pretokens':>12} {'Speedup':>10}")
    print(f" {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    print(f" {'HF ByteLevel':<25} {hf_best:>10.3f} {total_mb/hf_best:>10.1f} {hf_tokens:>12,} {'1.0x':>10}")
    print(f" {'Jeton pretokenizer':<25} {jeton_best:>10.3f} {total_mb/jeton_best:>10.1f} {jeton_tokens:>12,} {speedup:>9.1f}x")
    print()

    if jeton_tokens != hf_tokens:
        print(f" NOTE: token counts differ (jeton={jeton_tokens:,}, hf={hf_tokens:,})")
    else:
        print(f" Token counts match: {jeton_tokens:,}")


if __name__ == "__main__":
    main()
