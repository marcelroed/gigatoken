"""Sequential cross-library sweep over the project's benchmark tokenizers.

Candidate repos are deduplicated by their tokenizer definition in the HF
cache: for tokenizer.json the hash covers the model, normalizer, and
pre_tokenizer sections (so repos differing only in added special tokens or
chat metadata — Llama-3.1 vs 3.2, DeepSeek-V3 vs R1 — count as shared);
tokenizer.model / tiktoken.model files are hashed whole. Each shared group
is benchmarked once under the first-listed repo, with the rest recorded as
"shared_with". For each unique tokenizer the sweep runs --rounds interleaved
rounds of measure.py — gigatoken on the whole file, hf on the first --hf-mb
MB (only when the repo ships a tokenizer.json), tiktoken on the first
--tiktoken-mb MB (only for natively supported repos, see
`measure.py --tiktoken-support`) — one fresh process per measurement,
strictly sequentially, never in parallel. Each round is merged into the
single results file (cpu > tokenizer > dataset > implementation) as it
completes, via results.py's best-round-by-gigatoken rule.

`--scan-families` instead scans every model in the local HF cache for repos
sharing a benchmarked row's tokenizer (same digest) and writes the coverage
map benchmarks/families.json used by `results.py render`, then exits.

Usage:
    uv run benchmarks/compare/sweep.py --file ~/data/owt_train.txt
    uv run benchmarks/compare/sweep.py --only openai-community/gpt2 --rounds 1
    uv run benchmarks/compare/sweep.py --scan-families
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import measure
import results

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(HERE, "measure.py")
DEFAULT_FAMILIES = results.DEFAULT_FAMILIES
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tiktoken.model")
HUB_CACHE = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))

# Candidate repos, deduplicated at runtime. Order matters: the first repo of
# a shared-tokenizer group becomes the group's representative.
REPOS = [
    # BPE / tokenizer.json families
    "openai-community/gpt2",
    "answerdotai/ModernBERT-base",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V4-Flash",
    "zai-org/GLM-4.7",
    "zai-org/GLM-5.2",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "allenai/Olmo-3-1025-7B",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "tencent/Hy3",
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
    # SentencePiece families
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "codellama/CodeLlama-7b-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "unsloth/gemma-2b",
    "google/gemma-3-4b-it",
    "google/gemma-4-E4B-it",
]


def cached_file(repo: str, filename: str) -> str | None:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename)
    return path if isinstance(path, str) else None


def tokenizer_digest(path: str) -> str:
    """Identity of the tokenizer proper: for tokenizer.json only the parts
    that determine how ordinary text is encoded (vocab/merges, normalizer,
    pre_tokenizer), so repos differing in added special tokens still count
    as the same tokenizer; other formats are hashed whole."""
    with open(path, "rb") as f:
        raw = f.read()
    if os.path.basename(path) == "tokenizer.json":
        cfg = json.loads(raw)
        core = {key: cfg.get(key) for key in ("model", "normalizer", "pre_tokenizer")}
        raw = json.dumps(core, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def repo_digest(repo: str) -> str | None:
    for fname in TOKENIZER_FILES:
        path = cached_file(repo, fname)
        if path is not None:
            try:
                return tokenizer_digest(path)
            except (OSError, ValueError, json.JSONDecodeError):
                return None
    return None


def tokenizer_groups(repos: list[str]) -> list[dict]:
    groups: list[dict] = []
    by_digest: dict[str, dict] = {}
    for repo in repos:
        for fname in TOKENIZER_FILES:
            path = cached_file(repo, fname)
            if path is not None:
                break
        else:
            print(f"skip {repo}: no tokenizer file in the HF cache", file=sys.stderr)
            continue
        digest = tokenizer_digest(path)
        if digest in by_digest:
            by_digest[digest]["shared_with"].append(repo)
        else:
            group = {"repo": repo, "hf_json": fname == "tokenizer.json", "local": path, "shared_with": []}
            by_digest[digest] = group
            groups.append(group)
    return groups


def scan_families(results_path: str, out_path: str) -> None:
    """Map each benchmarked tokenizer to every cached repo sharing its digest."""
    stored = results.load_results(results_path)
    benchmarked = sorted({repo for tokenizers in stored.values() for repo in tokenizers})
    digests = {repo: repo_digest(repo) for repo in benchmarked}
    for repo in (r for r, d in digests.items() if d is None):
        print(f"warning: no digestible tokenizer file for {repo}", file=sys.stderr)

    by_digest: dict[str, list[str]] = {d: [] for d in digests.values() if d is not None}
    cached = [
        entry[len("models--") :].replace("--", "/")
        for entry in sorted(os.listdir(HUB_CACHE))
        if entry.startswith("models--")
    ]
    for repo in cached:
        digest = repo_digest(repo)
        if digest in by_digest:
            by_digest[digest].append(repo)

    families = {
        repo: sorted(by_digest[digest], key=str.lower)
        for repo, digest in digests.items()
        if digest is not None
    }
    with open(out_path, "w") as f:
        json.dump(families, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"scanned {len(cached)} cached repos; wrote coverage for {len(families)} tokenizers to {out_path}")


def run_one(lib: str, repo: str, file: str, max_mb: float | None, local_file: str | None) -> dict | None:
    cmd = [sys.executable, MEASURE, "--library", lib, "--tokenizer", repo, "--file", file]
    if max_mb is not None:
        cmd += ["--max-mb", str(max_mb)]
    if local_file is not None:
        cmd += ["--local-file", local_file]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        print(f"    FAILED: {lib} on {repo} (exit {proc.returncode}); skipping this library", file=sys.stderr)
        return None
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="~/data/owt_train.txt")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hf-mb", type=float, default=100)
    parser.add_argument("--tiktoken-mb", type=float, default=1000)
    parser.add_argument("--results", default=results.DEFAULT_RESULTS)
    parser.add_argument("--only", nargs="*", default=None, help="benchmark only these repos instead of the built-in list")
    parser.add_argument("--scan-families", action="store_true", help="regenerate the families.json coverage map from the HF cache and exit")
    args = parser.parse_args()

    if args.scan_families:
        scan_families(args.results, DEFAULT_FAMILIES)
        return

    file = os.path.expanduser(args.file)
    repos = args.only if args.only else REPOS
    groups = tokenizer_groups(repos)
    print(f"{len(groups)} unique tokenizers from {len(repos)} candidate repos", file=sys.stderr)

    stored = results.load_results(args.results)
    for i, group in enumerate(groups):
        repo = group["repo"]
        plan = [("gigatoken", None, None)]
        if group["hf_json"]:
            plan.append(("hf", args.hf_mb, group["local"]))
        if measure.native_encoding(repo) is not None:
            plan.append(("tiktoken", args.tiktoken_mb, None))
        shared = f"  (= {', '.join(group['shared_with'])})" if group["shared_with"] else ""
        print(f"[{i + 1}/{len(groups)}] {repo}{shared}: {', '.join(lib for lib, _, _ in plan)}", file=sys.stderr)

        failed: set[str] = set()
        for rnd in range(args.rounds):
            records = []
            for lib, max_mb, local in plan:
                if lib in failed:
                    continue
                rec = run_one(lib, repo, file, max_mb, local)
                if rec is None:
                    failed.add(lib)
                    continue
                if group["shared_with"]:
                    rec["shared_with"] = group["shared_with"]
                records.append(rec)
            round_wrote = False
            for cset in results.comparison_sets(records):
                outcome, wrote = results.merge_set(stored, cset)
                round_wrote |= wrote
                print(f"    round {rnd + 1}: {outcome}", file=sys.stderr)
            if round_wrote:
                results.save_results(args.results, stored)
    print(f"sweep done -> {args.results}", file=sys.stderr)


if __name__ == "__main__":
    main()
