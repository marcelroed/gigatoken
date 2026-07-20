"""Single-library tokenizer throughput on one file.

Each invocation measures exactly one library (--library {hf,tiktoken,gigatoken})
so every measurement starts from a fresh process: no shared thread pools,
allocator state, or warm caches between libraries. Interleave repeated
invocations externally and take the min per library (sweep.py does both and
records into benchmarks/results.json via results.py).

Tokenizers are named by HF repo id for every library, so measurements of the
same tokenizer line up across libraries. tiktoken only runs repos its own
registry supports natively: the repo must belong to an OpenAI org and its
basename must resolve through tiktoken.model's MODEL_TO_ENCODING /
MODEL_PREFIX_TO_ENCODING tables to an installed encoding. Vocabs that could
merely be converted into a tiktoken.Encoding (Qwen, Llama, ...) are refused;
`--tiktoken-support [REPO ...]` prints the resolution (or the whole
registry) and exits. --local-file makes hf/gigatoken load from an
already-resolved local tokenizer file while still recording the repo id.

The whole file is read into a Python bytes object before the timer starts;
only the batch-encode call is timed. All three libraries run with parallelism
enabled. hf and tiktoken only parallelize across documents, so their input is
split on --separator (before timing); gigatoken is handed the raw un-split
bytes object as a single document — both its BPE and SentencePiece backends
chunk oversized documents internally at provably safe boundaries
(src/batch.rs). Because of that, gigatoken also encodes the separator
strings themselves (as special tokens), so its token count differs from the
others by roughly one token per document.

Usage:
    uv run benchmarks/compare/measure.py --library gigatoken \
        --tokenizer openai-community/gpt2 --file ~/data/owt_train.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import results

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

NATIVE_TIKTOKEN_OWNERS = {"openai", "openai-community"}


def native_encoding(repo_id: str) -> str | None:
    """The tiktoken encoding name for `repo_id`, or None if unsupported."""
    import tiktoken
    import tiktoken.model

    owner, _, model = repo_id.rpartition("/")
    if owner.lower() not in NATIVE_TIKTOKEN_OWNERS:
        return None
    model = model.lower()
    encoding = tiktoken.model.MODEL_TO_ENCODING.get(model)
    if encoding is None:
        matches = [
            enc
            for prefix, enc in tiktoken.model.MODEL_PREFIX_TO_ENCODING.items()
            if model.startswith(prefix)
        ]
        encoding = matches[0] if matches else None
    if encoding is not None and encoding in tiktoken.list_encoding_names():
        return encoding
    return None


def print_tiktoken_support(repos: list[str]) -> None:
    import tiktoken
    import tiktoken.model

    if repos:
        for repo in repos:
            print(f"{repo} -> {native_encoding(repo) or '-'}")
        return
    print(f"tiktoken {getattr(tiktoken, '__version__', '?')} encodings: {', '.join(tiktoken.list_encoding_names())}")
    for model, enc in sorted(tiktoken.model.MODEL_TO_ENCODING.items()):
        print(f"{model} -> {enc}")
    for prefix, enc in sorted(tiktoken.model.MODEL_PREFIX_TO_ENCODING.items()):
        print(f"{prefix}* -> {enc}")


def load_hf(name: str):
    from tokenizers import Tokenizer

    if os.path.exists(name):
        return Tokenizer.from_file(name)
    return Tokenizer.from_pretrained(name)


def load_tiktoken(name: str):
    import tiktoken

    encoding = native_encoding(name)
    if encoding is None:
        raise SystemExit(f"{name} is not natively supported by tiktoken (see --tiktoken-support)")
    return tiktoken.get_encoding(encoding)


def load_gigatoken(name: str):
    import gigatoken

    return gigatoken.Tokenizer(name)


def encode_hf(tokenizer, docs: list[str]):
    encode_batch = getattr(tokenizer, "encode_batch_fast", tokenizer.encode_batch)
    return encode_batch(docs)


def encode_tiktoken(tokenizer, docs: list[str]):
    # Sliced: the list-of-Python-int output costs ~36 bytes per token, so on
    # multi-GB inputs one full-batch call would need tens of GB. Counting and
    # freeing per slice keeps the timed region honest (the allocation cost is
    # still paid) while bounding the footprint.
    step = 25_000
    total = 0
    for i in range(0, len(docs), step):
        rows = tokenizer.encode_ordinary_batch(docs[i : i + step], num_threads=os.cpu_count())
        total += sum(map(len, rows))
    return total


def encode_gigatoken(tokenizer, docs: list[str]):
    return tokenizer.encode_batch(docs, parallel=True)


def count_gigatoken(result) -> int:
    import awkward as ak

    return int(ak.sum(ak.num(result)))


LIBRARIES = {
    "hf": (load_hf, encode_hf, lambda encodings: sum(len(e.ids) for e in encodings)),
    "tiktoken": (load_tiktoken, encode_tiktoken, lambda total: total),
    "gigatoken": (load_gigatoken, encode_gigatoken, count_gigatoken),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", choices=sorted(LIBRARIES))
    parser.add_argument("--tokenizer", help="HF repo id, e.g. openai-community/gpt2")
    parser.add_argument("--local-file", default=None, help="already-resolved local tokenizer file for hf/gigatoken to load; the recorded tokenizer name stays --tokenizer")
    parser.add_argument("--file")
    parser.add_argument("--separator", default="<|endoftext|>", help="document separator; empty string encodes the file as a single document")
    parser.add_argument("--max-mb", type=float, default=None, help="truncate the file to roughly this many MB (at a UTF-8 character boundary)")
    parser.add_argument("--tiktoken-support", nargs="*", default=None, metavar="REPO", help="print tiktoken's native resolution for REPOs (or the whole registry) and exit")
    args = parser.parse_args()

    if args.tiktoken_support is not None:
        print_tiktoken_support(args.tiktoken_support)
        return
    for required in ("library", "tokenizer", "file"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required")

    load, encode, count = LIBRARIES[args.library]
    source = args.tokenizer if args.library == "tiktoken" else (args.local_file or args.tokenizer)
    tokenizer = load(source)
    if args.library == "gigatoken":
        import awkward  # noqa: F401  (imported here so it isn't timed)

    # Everything below happens before the timer: the file is fully in memory
    # as bytes (and for hf/tiktoken decoded and split), so timing covers only
    # tokenization.
    data = open(os.path.expanduser(args.file), "rb").read()
    if args.max_mb is not None:
        budget = min(int(args.max_mb * 1e6), len(data))
        while budget < len(data) and data[budget] & 0xC0 == 0x80:  # UTF-8 boundary
            budget += 1
        data = data[:budget]
    if args.library == "gigatoken":
        inputs = [data]
    else:
        text = data.decode("utf-8")
        inputs = text.split(args.separator) if args.separator else [text]
    n_docs = len(inputs)
    n_bytes = len(data)
    if args.library != "gigatoken":
        del data, text  # keep peak memory down on multi-GB files

    start = time.perf_counter()
    encoded = encode(tokenizer, inputs)
    elapsed = time.perf_counter() - start
    n_tokens = count(encoded)

    result = {
        "library": args.library,
        "tokenizer": args.tokenizer,
        "file": args.file,
        "cpu": results.cpu_label(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_mb": args.max_mb,
        "docs": n_docs,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "time_s": round(elapsed, 4),
        "mb_per_s": round(n_bytes / 1e6 / elapsed, 2),
        "mtokens_per_s": round(n_tokens / 1e6 / elapsed, 3),
    }
    if args.library == "tiktoken":
        result["encoding"] = tokenizer.name
    print(json.dumps(result))
    print(
        f"{args.library:>9}: {result['mb_per_s']:>8.2f} MB/s  "
        f"{result['mtokens_per_s']:>7.3f} Mtok/s  "
        f"({n_bytes / 1e6:.1f} MB, {n_docs} docs, {n_tokens} tokens, {elapsed:.3f}s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
