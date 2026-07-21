"""Best-results store and README table for the throughput benchmarks.

Stdlib-only (run with `uv run --no-project`). Two subcommands:

    uv run --no-project benchmarks/compare/results.py merge run.jsonl [...]
    uv run --no-project benchmarks/compare/results.py render

merge folds measurement JSONL (from measure.py, files or stdin) into
benchmarks/results.json, nested cpu -> tokenizer (HF repo id) -> dataset
(file basename) -> implementation {gigatoken, hf, tiktoken}. Within one
input batch, consecutive records on the same cpu/tokenizer/file with each
library appearing at most once form a *comparison set* — one interleaved
round. A set replaces the stored group only if its gigatoken throughput
beats the stored group's, so every stored group is one coherent round — the
best one, as judged by gigatoken — never a mix of libraries from different
rounds. Records without a "cpu" field are attributed to --cpu, which
defaults to this machine.

render rewrites the block between <!-- benchmarks:start --> and
<!-- benchmarks:end --> in the repo README (appending a section when the
markers are absent): one collapsible table per (cpu, dataset) with family
display names, throughput per implementation, and gigatoken's speedup vs
HF/tiktoken, plus hand-curated coverage footnotes (COVERS below, written
from the `sweep.py --scan-families` data in benchmarks/families.json).
Rows without a gigatoken measurement are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
DEFAULT_RESULTS = os.path.join(BENCH_DIR, "results.json")
DEFAULT_FAMILIES = os.path.join(BENCH_DIR, "families.json")
DEFAULT_README = os.path.join(os.path.dirname(BENCH_DIR), "README.md")
START, END = "<!-- benchmarks:start -->", "<!-- benchmarks:end -->"


def cpu_label() -> str:
    """Identify the CPU this benchmark ran on, e.g. "Apple M4 Max (16 cores)"."""
    import platform

    name = None
    system = platform.system()
    if system == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        name = out.stdout.strip() or None
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    name = name or platform.processor() or platform.machine() or "unknown"
    return f"{name} ({os.cpu_count()} cores)"


# --- merge -----------------------------------------------------------------


def read_records(origin: str, lines, cpu_fallback: str) -> list[dict]:
    records = []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{origin}:{lineno}: invalid JSON: {e}")
        if not isinstance(rec, dict) or not all(k in rec for k in ("library", "tokenizer", "file")):
            raise SystemExit(f"{origin}:{lineno}: not a measure.py record (needs library/tokenizer/file)")
        rec.setdefault("cpu", cpu_fallback)
        records.append(rec)
    return records


def fits(cur: dict, rec: dict) -> bool:
    return rec["library"] not in cur["libs"] and (rec["cpu"], rec["tokenizer"], rec["file"]) == (
        cur["cpu"],
        cur["tokenizer"],
        cur["file"],
    )


def comparison_sets(records: list[dict]) -> list[dict]:
    sets: list[dict] = []
    for rec in records:
        if not sets or not fits(sets[-1], rec):
            sets.append({"cpu": rec["cpu"], "tokenizer": rec["tokenizer"], "file": rec["file"], "libs": {}})
        sets[-1]["libs"][rec["library"]] = rec
    return sets


def stored_entry(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("library", "cpu")}


def merge_set(results: dict, cset: dict) -> tuple[str, bool]:
    dataset = os.path.basename(cset["file"])
    by_dataset = results.setdefault(cset["cpu"], {}).setdefault(cset["tokenizer"], {})
    stored = by_dataset.get(dataset)
    new_g = cset["libs"].get("gigatoken", {}).get("mb_per_s")
    old_g = (stored or {}).get("gigatoken", {}).get("mb_per_s")
    if stored is not None:
        if new_g is None:
            return "kept (set has no gigatoken measurement to judge by)", False
        if old_g is not None and new_g <= old_g:
            return f"kept (gigatoken {new_g} <= stored {old_g} MB/s)", False
    by_dataset[dataset] = {lib: stored_entry(rec) for lib, rec in sorted(cset["libs"].items())}
    if stored is None:
        return f"stored ({', '.join(sorted(cset['libs']))})", True
    if old_g is None:
        return f"updated (stored group had no gigatoken; now {new_g} MB/s)", True
    return f"updated (gigatoken {new_g} > {old_g} MB/s)", True


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}: invalid JSON: {e}")


def save_results(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")


def cmd_merge(args) -> None:
    cpu = args.cpu or cpu_label()
    batches = []
    if args.jsonl:
        for path in args.jsonl:
            with open(path) as f:
                batches.append(read_records(path, f, cpu))
    else:
        batches.append(read_records("<stdin>", sys.stdin, cpu))

    results = load_results(args.results)
    changed = 0
    for records in batches:
        for cset in comparison_sets(records):
            outcome, wrote = merge_set(results, cset)
            changed += wrote
            print(f"{cset['cpu']} / {cset['tokenizer']} / {os.path.basename(cset['file'])}: {outcome}")
    if changed:
        save_results(args.results, results)
    print(f"{changed} group(s) written to {args.results}" if changed else f"{args.results} unchanged")


# --- render ----------------------------------------------------------------

# Family display name per benchmarked repo; rows fall back to the repo id.
DISPLAY = {
    "openai-community/gpt2": "GPT-2",
    "answerdotai/ModernBERT-base": "ModernBERT",
    "meta-llama/Llama-3.1-8B": "Llama 3 / 3.1 / 3.2",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen 2 / 2.5",
    "Qwen/Qwen3-8B": "Qwen 3",
    "Qwen/Qwen3.5-9B": "Qwen 3.5 / 3.6",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3 / R1 / V4",
    "zai-org/GLM-4.7": "GLM 4",
    "zai-org/GLM-5.2": "GLM 5",
    "openai/gpt-oss-20b": "GPT-OSS",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron 3",
    "allenai/Olmo-3-1025-7B": "OLMo 2 / 3",
    "moonshotai/Kimi-K2-Instruct": "Kimi K2",
    "tencent/Hy3": "Hunyuan 3",
    "microsoft/phi-4": "Phi-4",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama / Phi-3 (Llama 2)",
    "codellama/CodeLlama-7b-hf": "CodeLlama",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
    "unsloth/gemma-2b": "Gemma 1",
    "google/gemma-3-4b-it": "Gemma 3",
    "google/gemma-4-E4B-it": "Gemma 4",
}

# Hand-curated coverage per row, written from the verified scan data in
# benchmarks/families.json (regenerate the raw data with
# `sweep.py --scan-families` and update these lines when it changes). Only
# rows whose coverage goes beyond their display name are listed.
COVERS = {
    "meta-llama/Llama-3.1-8B": "Llama 3 / 3.1 / 3.2, DeepSeek-R1-Distill-Llama, Hermes 3, Saiga, and other Llama-3 finetunes",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3, Llama-3.1-Nemotron-Nano-VL, SmolLM3, Kanana 1.5, jina-embeddings-v5, Ultravox",
    "Qwen/Qwen2-1.5B-Instruct": "Qwen 2 and 2.5 (incl. Coder and VL), Qwen3-Coder, Qwen3-VL, DeepSeek-R1 Qwen distills, MiMo V2.5, MiniCPM-o 2.6, InternVL3",
    "Qwen/Qwen3-8B": "Qwen 3 (incl. Embedding and Reranker), Qwen2.5-Omni, Qwen3-VL-Embedding, MiMo V2.5 Pro, jina-reranker-m0, pplx-embed, MOSS-TTS, Zeta",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3 / V3.1 / V3.2, R1, V4 Flash and Pro, DeepSeek-VL2",
    "zai-org/GLM-4.7": "GLM 4.1V, 4.5, and 4.7",
    "zai-org/GLM-5.2": "GLM 5 / 5.2 and GLM-4.7-Flash",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron 3 Nano, Super, and Ultra",
    "moonshotai/Kimi-K2-Instruct": "Kimi K2 / K2.5 / K2.6 / K2.7, Kimi-Linear, Kimi-VL, Moonlight",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini and Phi-4-multimodal",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama, Phi-3-mini, Phi-3.5-mini and Phi-3.5-vision (the Llama 2 vocab)",
    "google/gemma-3-4b-it": "Gemma 3 (270M–27B) and EmbeddingGemma",
    "google/gemma-4-E4B-it": "Gemma 4 (dense, MoE, and E-series) and DiffusionGemma",
}


# README table order (results.json sorts CPU keys alphabetically); unlisted
# CPUs sort last.
CPU_ORDER = [
    "AMD EPYC 9565 72-Core Processor (288 cores)",
    "Apple M4 Max (16 cores)",
    "AMD Ryzen 7 9800X3D 8-Core Processor (16 cores)",
]

# Hand-curated table headings where the raw cpu_label() is misleading (the
# EPYC's 288 comes from os.cpu_count() counting SMT threads over two sockets).
CPU_DISPLAY = {
    "AMD EPYC 9565 72-Core Processor (288 cores)": "AMD EPYC 9565 72-Core Processor x 2 sockets (144 cores)",
}


def fmt_speed(mb_per_s: float | None) -> str:
    if mb_per_s is None:
        return "—"
    if mb_per_s >= 1000:
        return f"{mb_per_s / 1000:.2f} GB/s"
    return f"{mb_per_s:.1f} MB/s"


def fmt_ratio(giga: float | None, other: float | None) -> str:
    if giga is None or not other:
        return "—"
    ratio = giga / other
    return f"{ratio:,.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def render_table(cpu: str, dataset: str, tokenizers: dict) -> str | None:
    rows = []
    corpus_bytes = 0
    for repo, by_dataset in tokenizers.items():
        group = by_dataset.get(dataset)
        if group is None or "gigatoken" not in group:
            continue
        corpus_bytes = max(corpus_bytes, group["gigatoken"].get("bytes", 0))
        speeds = {lib: rec.get("mb_per_s") for lib, rec in group.items()}
        rows.append((repo, speeds))
    if not rows:
        return None
    rows.sort(key=lambda row: row[1].get("gigatoken") or 0, reverse=True)

    size = f" ({corpus_bytes / 1e9:.1f} GB)" if corpus_bytes else ""
    lines = [
        "<details>",
        f"<summary><b>Encoding throughput on {dataset}{size} — {CPU_DISPLAY.get(cpu, cpu)}</b></summary>",
        "",
        "| Tokenizer | gigatoken | HF tokenizers | tiktoken | vs HF | vs tiktoken |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for repo, speeds in rows:
        giga, hf, tik = speeds.get("gigatoken"), speeds.get("hf"), speeds.get("tiktoken")
        lines.append(
            f"| {DISPLAY.get(repo, repo)} | {fmt_speed(giga)} | {fmt_speed(hf)} | {fmt_speed(tik)} "
            f"| {fmt_ratio(giga, hf)} | {fmt_ratio(giga, tik)} |"
        )
    lines += ["", "</details>"]
    return "\n".join(lines)


def render_notes(repos: set[str]) -> str:
    """The shared methodology/coverage spoiler placed beneath the tables."""
    coverage_notes = [f"- **{DISPLAY.get(repo, repo)}** — {COVERS[repo]}" for repo in COVERS if repo in repos]
    lines = [
        "<details>",
        "<summary><b>Benchmark details</b></summary>",
        "",
        "Best of 3 interleaved rounds, one fresh process per measurement, all libraries with parallelism enabled.",
        "Gigatoken encodes the whole file un-split, and is thus doing more work than the other tokenizers to find the split boundaries and automatically parallelize.",
        "HuggingFace tokenizers (`encode_batch_fast`) gets the first 100 MB and tiktoken (`encode_ordinary_batch`) the first 1 GB, both presplit on `<|endoftext|>`.",
        "This is fair because neither of the compared tokenizers do caching, meaning the speed is roughly uniform throughout processing.",
        "Tiktoken rows are currently only filled in for tokenizers with official support.",
        "",
        "The slowest rows are the SentencePiece-based tokenizers, which are only somewhat optimized in Gigatoken.",
    ]
    if coverage_notes:
        lines += [
            "",
            "Each row is one distinct tokenizer (identical vocab/merges/pretokenizer), measured on a representative repo.",
            "If you don't see your tokenizer here, it's likely based on some existing one.",
            "For instance:",
            "",
        ] + coverage_notes
    lines += ["", "</details>"]
    return "\n".join(lines)


def cmd_render(args) -> None:
    results = load_results(args.results)
    if not results:
        raise SystemExit(f"{args.results}: no results to render")

    def cpu_rank(cpu: str) -> tuple[int, str]:
        return (CPU_ORDER.index(cpu) if cpu in CPU_ORDER else len(CPU_ORDER), cpu)

    tables = []
    for cpu, tokenizers in sorted(results.items(), key=lambda kv: cpu_rank(kv[0])):
        datasets = sorted({ds for by_dataset in tokenizers.values() for ds in by_dataset})
        for dataset in datasets:
            table = render_table(cpu, dataset, tokenizers)
            if table is not None:
                tables.append(table)
    tabled_repos = {
        repo
        for tokenizers in results.values()
        for repo, by_dataset in tokenizers.items()
        if any("gigatoken" in group for group in by_dataset.values())
    }
    block = "\n".join([START, "## Benchmarks", ""] + tables + [render_notes(tabled_repos), END])

    with open(args.readme) as f:
        readme = f.read()
    start, end = readme.find(START), readme.find(END)
    if start == -1 and end == -1:
        readme = readme.rstrip("\n") + "\n\n" + block + "\n"
    elif start != -1 and end > start:
        readme = readme[:start] + block + readme[end + len(END):]
    else:
        raise SystemExit(f"{args.readme}: malformed benchmark markers ({START} / {END})")
    with open(args.readme, "w") as f:
        f.write(readme)
    print(f"wrote {len(tables)} table(s) to {args.readme}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="fold measurement JSONL into the results file")
    merge.add_argument("jsonl", nargs="*", help="JSONL measurement files; stdin if omitted")
    merge.add_argument("--results", default=DEFAULT_RESULTS, help="results JSON file to update (default: %(default)s)")
    merge.add_argument("--cpu", default=None, help="CPU label for records that lack one (default: this machine)")
    merge.set_defaults(func=cmd_merge)

    render = sub.add_parser("render", help="rewrite the README benchmark table from the results file")
    render.add_argument("--results", default=DEFAULT_RESULTS)
    render.add_argument("--readme", default=DEFAULT_README)
    render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
