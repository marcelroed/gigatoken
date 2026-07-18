#!/usr/bin/env python3
"""Interleaved A/B(/N) benchmark harness for the encode_st bench binary.

Methodology (carried over from the prior campaign, profiling/campaign_report.md):

- Strictly sequential: one bench process at a time, never concurrent with
  anything else heavy. The harness refuses to run under high load average.
- Fresh process per sample; variants interleaved round-robin (A B A B ...)
  so slow machine drift hits every variant equally.
- Token-count identity gate: every sample's token count must match across
  ALL variants — a mismatch is a correctness bug, not a performance result.
- Reports per-variant min/median/mean and pairwise ratios of means, plus a
  Welch t statistic as a significance hint (not a substitute for judgment).

The bench binary prints one line per pass on stderr:
  "pass 0: 22834020 tokens in 21.40s — 0.47 GB/s (467 MB/s)"
Throughput is parsed from the integer MB/s figure (full precision source:
size_gb / elapsed is recomputed from the parsed seconds when available).

Usage:
  ab_bench.py --variant main=bin/encode_st_main --variant new=bin/encode_st_x \\
              [--samples 7] [--encode-mb 10000] [--passes 1] [--json-out results.json]

Env passthrough: ENCODE_TOKENIZER is forwarded if set in the environment.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

PASS_RE = re.compile(
    r"pass (\d+): (\d+) tokens in ([0-9.]+)s — ([0-9.]+) GB/s \((\d+) MB/s\)"
)


def load_avg_ok(max_load: float) -> bool:
    """Refuse to measure when the machine is busy (1-min load avg per CPU)."""
    try:
        load1, _, _ = os.getloadavg()
        ncpu = os.cpu_count() or 1
        return load1 / ncpu < max_load
    except OSError:
        return True


def run_sample(binary: str, encode_mb: int, passes: int) -> tuple[int, float]:
    """One fresh-process sample. Returns (token_count, throughput_MBps)."""
    env = dict(os.environ)
    env["ENCODE_MB"] = str(encode_mb)
    if passes > 1:
        env["ENCODE_PASSES"] = str(passes)
    proc = subprocess.run(
        [binary], env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{binary} exited {proc.returncode}")
    samples = PASS_RE.findall(proc.stderr)
    if not samples:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"no pass lines parsed from {binary}")
    # Use the LAST pass (with passes=1 that's the only one; with warm passes
    # the last is the fully-warm steady state).
    _p, tokens, secs, _gbps, mbps = samples[-1]
    tokens_i = int(tokens)
    # Recompute throughput from raw seconds at full precision (the printed
    # integer MB/s is rounded); input size = encode_mb * 1e6 bytes.
    thr = (encode_mb * 1_000_000) / float(secs) / 1e6
    return tokens_i, thr


def welch_t(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return None
    return (statistics.mean(a) - statistics.mean(b)) / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="LABEL=BINARY",
        help="variant label and path to bench binary (repeatable)",
    )
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--encode-mb", type=int, default=10000)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument(
        "--max-load",
        type=float,
        default=0.35,
        help="max 1-min load average per CPU before refusing to run",
    )
    ap.add_argument("--json-out", metavar="PATH")
    ap.add_argument(
        "--expect-tokens",
        type=int,
        default=None,
        help="assert this exact token count on every sample",
    )
    args = ap.parse_args()

    variants = []
    for spec in args.variant:
        label, _, path = spec.partition("=")
        if not label or not path:
            ap.error(f"bad --variant spec: {spec!r}")
        if not os.path.isfile(path):
            ap.error(f"binary not found: {path}")
        variants.append((label, os.path.abspath(path)))

    if not load_avg_ok(args.max_load):
        print("machine under load; refusing to measure", file=sys.stderr)
        return 2

    results: dict[str, list[float]] = {label: [] for label, _ in variants}
    token_counts: dict[str, set[int]] = {label: set() for label, _ in variants}
    t_start = time.time()
    for rnd in range(args.samples):
        for label, binary in variants:
            tokens, thr = run_sample(binary, args.encode_mb, args.passes)
            results[label].append(thr)
            token_counts[label].add(tokens)
            print(
                f"round {rnd + 1}/{args.samples}  {label:<16} {thr:8.1f} MB/s  ({tokens} tokens)",
                flush=True,
            )

    # Identity gate
    all_counts = set().union(*token_counts.values())
    if len(all_counts) != 1:
        print("\nTOKEN IDENTITY FAILURE across variants:", file=sys.stderr)
        for label in token_counts:
            print(f"  {label}: {sorted(token_counts[label])}", file=sys.stderr)
        return 3
    the_count = all_counts.pop()
    if args.expect_tokens is not None and the_count != args.expect_tokens:
        print(
            f"\nTOKEN COUNT MISMATCH: got {the_count}, expected {args.expect_tokens}",
            file=sys.stderr,
        )
        return 3

    print(f"\ntokens: {the_count}  (identical across all variants)")
    print(f"{'variant':<16} {'min':>9} {'median':>9} {'mean':>9} {'stdev':>7}")
    for label, _ in variants:
        r = results[label]
        print(
            f"{label:<16} {min(r):9.1f} {statistics.median(r):9.1f} "
            f"{statistics.mean(r):9.1f} {statistics.stdev(r) if len(r) > 1 else 0:7.1f}"
        )
    print("\npairwise (mean ratio, Welch t vs first variant):")
    base_label = variants[0][0]
    base = results[base_label]
    for label, _ in variants[1:]:
        r = results[label]
        ratio = statistics.mean(r) / statistics.mean(base)
        t = welch_t(r, base)
        t_s = f"{t:+.2f}" if t is not None else "n/a"
        verdict = "WIN" if ratio > 1.0 else "loss"
        print(f"  {label} vs {base_label}: {ratio:.4f}x  (t={t_s})  {verdict}")

    if args.json_out:
        payload = {
            "encode_mb": args.encode_mb,
            "passes": args.passes,
            "samples": args.samples,
            "tokens": the_count,
            "wall_s": time.time() - t_start,
            "variants": {label: results[label] for label, _ in variants},
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nraw samples -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
