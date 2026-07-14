"""The batch encodes must compose with Python multiprocessing: inside a
worker process they auto-detect the situation and take the sequential Rust
paths, which never touch the process-global rayon pool. The fork test is the
critical regression: a rayon pool built before os.fork has no worker threads
in the child, so the parallel path would wait forever there.

Output identity between parallel=True and parallel=False is asserted for
every batch entry point on both backends.
"""

import multiprocessing as mp
import random
import sys

import awkward as ak
import numpy as np
import pytest
from pytest import fixture

import gigatoken
from gigatoken._parallel import in_worker_process, resolve_parallel

# Enough text that the parallel path genuinely fans out (chunks are at least
# 1 MB and fan-out needs more than one chunk), so the fork test would really
# deadlock without the sequential path.
_TOTAL_BYTES = 6 << 20


def _make_texts(n_docs: int = 96, total_bytes: int = _TOTAL_BYTES) -> list[str]:
    rng = random.Random(0)
    words = ["the", "quick", "brown", "fox", "jumps", "höher", "tokenizer", "42", "  space", "\n"]
    doc_words = total_bytes // (n_docs * 6)
    return [" ".join(rng.choices(words, k=doc_words)) for _ in range(n_docs)]


@fixture(scope="module")
def texts() -> list[str]:
    return _make_texts()


@fixture(scope="module")
def gpt2(gpt2_tokenizer_path) -> gigatoken.Tokenizer:
    return gigatoken.Tokenizer(gpt2_tokenizer_path)


@fixture(scope="module")
def tinyllama(tinyllama_tokenizer_path) -> gigatoken.Tokenizer:
    return gigatoken.Tokenizer(tinyllama_tokenizer_path)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_main_process_resolves_parallel():
    assert not in_worker_process()
    assert resolve_parallel(None) is True
    assert resolve_parallel(False) is False
    assert resolve_parallel(True) is True


def _spawn_probe() -> bool:
    from gigatoken._parallel import in_worker_process

    return in_worker_process()


def test_spawn_worker_detected():
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        assert pool.apply(_spawn_probe) is True


# ---------------------------------------------------------------------------
# parallel=True and parallel=False are output-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["gpt2", "tinyllama"])
def test_sequential_matches_parallel_batch(backend, request, texts):
    tok = request.getfixturevalue(backend)
    par = tok.encode_batch(texts, parallel=True)
    seq = tok.encode_batch(texts, parallel=False)
    assert ak.to_list(seq) == ak.to_list(par)


@pytest.mark.parametrize("backend", ["gpt2", "tinyllama"])
def test_sequential_matches_parallel_padded(backend, request, texts):
    tok = request.getfixturevalue(backend)
    par_ids, par_lens = tok.encode_batch_padded(texts, pad_id=0, max_length=256, truncate=True, pad_to_max_length=True)
    seq_ids, seq_lens = tok.encode_batch_padded(
        texts, pad_id=0, max_length=256, truncate=True, pad_to_max_length=True, parallel=False
    )
    np.testing.assert_array_equal(seq_ids, par_ids)
    np.testing.assert_array_equal(seq_lens, par_lens)


@pytest.mark.parametrize("backend", ["gpt2", "tinyllama"])
def test_sequential_matches_parallel_files(backend, request, texts, tmp_path):
    tok = request.getfixturevalue(backend)
    paths = []
    for i in range(3):
        p = tmp_path / f"docs_{i}.txt"
        p.write_text("\n\n".join(texts[i::3]), encoding="utf-8")
        paths.append(p)
    source = gigatoken.TextFileSource(paths, separator=b"\n\n")
    par = tok.encode_files(source, parallel=True)
    seq = tok.encode_files(source, parallel=False)
    assert ak.to_list(seq) == ak.to_list(par)


def test_tiktoken_compat_num_threads_one(gpt2, texts):
    compat = gpt2.as_tiktoken()
    assert compat.encode_batch(texts[:8], num_threads=1) == compat.encode_batch(texts[:8])


# ---------------------------------------------------------------------------
# The real scenario: fork-method workers after the parent built the pool
# ---------------------------------------------------------------------------

# Module global inherited by fork children; set by the test in the parent.
_FORK_TOK: gigatoken.Tokenizer | None = None


def _encode_in_fork_child(texts: list[str]) -> list[list[int]]:
    # Runs in a fork worker: _FORK_TOK and the parent's already-built rayon
    # pool are inherited, but the pool's threads are not. Auto-detection
    # must route encode_batch to the sequential path or this never returns.
    assert in_worker_process()
    assert _FORK_TOK is not None
    return ak.to_list(_FORK_TOK.encode_batch(texts))


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
def test_fork_pool_after_parent_warmup(gpt2, texts):
    global _FORK_TOK
    _FORK_TOK = gpt2
    try:
        # Parallel encode in the parent first: builds the rayon pool, so the
        # fork children inherit a pool whose threads do not exist for them.
        expected = ak.to_list(gpt2.encode_batch(texts))
        ctx = mp.get_context("fork")
        with ctx.Pool(2) as pool:
            halves = [texts[: len(texts) // 2], texts[len(texts) // 2 :]]
            results = [pool.apply_async(_encode_in_fork_child, (half,)) for half in halves]
            # A generous timeout turns a would-be deadlock into a failure.
            got = [ids for r in results for ids in r.get(timeout=180)]
        assert got == expected
    finally:
        _FORK_TOK = None
