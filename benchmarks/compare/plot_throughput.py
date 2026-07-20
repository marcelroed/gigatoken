"""Bar charts (one PDF per corpus+tokenizer) from compare_throughput.py output.

Reads one or more .jsonl result files, groups rows by (file, tokenizer), and
draws one bar per library: bar height is the best (min-time) round, round dots
show the spread. The y-axis is MB/s on a log scale — the libraries sit three
orders of magnitude apart, so linear bars would leave two of them invisible;
every bar carries an explicit value label instead.

Usage:
    uv run benchmarks/plot_throughput.py results/*.jsonl --out-dir plots
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import defaultdict

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 (blue, green, magenta) of the validated palette; the
# same library keeps the same hue in every plot.
LIBRARY_ORDER = ["hf", "tiktoken", "gigatoken"]  # slowest to fastest
LIBRARY_LABELS = {"gigatoken": "gigatoken", "tiktoken": "tiktoken", "hf": "HF tokenizers"}
LIBRARY_COLORS = {"gigatoken": "#2a78d6", "tiktoken": "#008300", "hf": "#e87ba4"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#1a1a19", "#5f5e56", "#e4e3dc"


NICE_NAMES = {
    "gpt2": "GPT-2",
    "qwen2_tokenizer.json": "Qwen2",
    "qwen3_5_tokenizer.json": "Qwen 3.5",
    "deepseek_v3_tokenizer.json": "DeepSeek-V3",
    "glm5_2_tokenizer.json": "GLM-5.2",
    "gptoss_tokenizer.json": "gpt-oss",
    "o200k_harmony": "gpt-oss",  # tiktoken's native encoding for the same tokenizer
    "nemotron_tokenizer.json": "Nemotron",
    "olmo3_tokenizer.json": "OLMo-3",
}


def nice_name(tokenizer: str) -> str:
    base = os.path.basename(tokenizer)
    return NICE_NAMES.get(base, base.removesuffix("_tokenizer.json").removesuffix(".json"))


def fmt_mbs(v: float) -> str:
    return f"{v / 1000:.2f} GB/s" if v >= 1000 else f"{v:.1f} MB/s"


def cpu_label() -> str:
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    else:
        chip = platform.processor() or platform.machine()
    return f"{chip}, {os.cpu_count()} threads"


def fmt_duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 100 else f"{seconds:.1f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help=".jsonl files from compare_throughput.py")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--linear", action="store_true", help="linear y axis instead of log (small bars vanish; labels carry the values)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"])
    parser.add_argument("--no-slice-note", action="store_true", help="omit the asterisk and footnote on libraries measured on a slice of the file")
    parser.add_argument("--no-dots", action="store_true", help="omit the per-round dots")
    parser.add_argument("--horizontal", action="store_true", help="sideways bars, fastest library on top")
    parser.add_argument("--xmax", type=float, default=None, help="fixed upper limit for the value axis (horizontal+linear)")
    parser.add_argument(
        "--hf-arrow",
        nargs="?",
        const="Already implemented in Rust!",
        default=None,
        help="annotate the HF bar with an arrow and this text (horizontal charts only)",
    )
    args = parser.parse_args()

    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in args.results:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = (os.path.basename(row["file"]), nice_name(row["tokenizer"]))
                groups[key][row["library"]].append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    for (corpus, tok_name), by_lib in sorted(groups.items()):
        libs = [lib for lib in LIBRARY_ORDER if lib in by_lib]
        best = {lib: max(by_lib[lib], key=lambda r: r["mb_per_s"]) for lib in libs}
        full_bytes = max(r["bytes"] for rows in by_lib.values() for r in rows)

        fig, ax = plt.subplots(figsize=(7.2, 2.9) if args.horizontal else (5.4, 3.6))
        xs = range(len(libs))
        heights = [best[lib]["mb_per_s"] for lib in libs]
        colors = [LIBRARY_COLORS[lib] for lib in libs]
        if args.horizontal:
            # barh puts y=0 at the bottom, so LIBRARY_ORDER (slowest first)
            # lands the fastest library on top.
            ax.barh(xs, heights, height=0.55, color=colors, zorder=3)
        else:
            ax.bar(xs, heights, width=0.55, color=colors, zorder=3)
        value_max = args.xmax if (args.horizontal and args.linear and args.xmax) else max(heights) * 1.28
        for x, lib in zip(xs, libs):
            if not args.no_dots:
                rounds = [r["mb_per_s"] for r in by_lib[lib]]
                pos = ([x] * len(rounds), rounds) if not args.horizontal else (rounds, [x] * len(rounds))
                ax.plot(*pos, "o", ms=4, mfc="white", mec=TEXT_PRIMARY, mew=0.8, zorder=4)
            value = best[lib]["mb_per_s"]
            # A bar ending near the axis limit gets its label inside the bar
            # so it can't spill past the figure edge.
            inside = args.horizontal and args.linear and value > 0.8 * value_max
            ax.annotate(
                fmt_mbs(value),
                (value, x) if args.horizontal else (x, value),
                xytext=((-8, 0) if inside else (8, 0)) if args.horizontal else (0, 10),
                textcoords="offset points",
                ha=("right" if inside else "left") if args.horizontal else "center",
                va="center" if args.horizontal else "baseline",
                color="white" if inside else TEXT_PRIMARY,
                fontsize=10,
                fontweight="bold",
                zorder=5,
            )

        labels = []
        footnotes = []
        for lib in libs:
            label = LIBRARY_LABELS[lib]
            if best[lib]["bytes"] < full_bytes and not args.no_slice_note:
                label += "*"
                projected = full_bytes / 1e6 / best[lib]["mb_per_s"]
                footnotes.append(
                    f"*measured on the first {best[lib]['bytes'] / 1e6:.0f} MB; "
                    f"full file projected {fmt_duration(projected)}"
                )
            labels.append(label)
        if "tiktoken" not in by_lib:
            footnotes.append("tiktoken omitted: not expressible as a tiktoken Encoding")

        value_axis_label = "Throughput (MB/s)" if args.linear else "Throughput (MB/s, log scale)"
        if args.horizontal:
            ax.set_yticks(list(xs), labels, color=TEXT_PRIMARY, fontsize=10)
            ax.set_xlabel(value_axis_label, color=TEXT_PRIMARY)
            if args.linear:
                ax.set_xlim(0, args.xmax if args.xmax else max(heights) * 1.28)
            else:
                ax.set_xscale("log")
                ax.set_xlim(right=max(heights) * 8)
        else:
            ax.set_xticks(list(xs), labels, color=TEXT_PRIMARY, fontsize=10)
            ax.set_ylabel(value_axis_label, color=TEXT_PRIMARY)
            if args.linear:
                ax.set_ylim(0, max(heights) * 1.18)
            else:
                ax.set_yscale("log")
                ax.set_ylim(top=max(heights) * 4)
        ax.set_title(
            f"Tokenizer throughput — {tok_name} on {corpus} ({full_bytes / 1e9:.0f} GB)\n{cpu_label()}",
            fontsize=11,
            color=TEXT_PRIMARY,
            loc="left",
        )
        if args.hf_arrow and args.horizontal and "hf" in libs:
            hf_y = libs.index("hf")
            ax.annotate(
                args.hf_arrow,
                xy=(0.04 * value_max, hf_y + 0.3),
                xytext=(0.3 * value_max, hf_y + 0.62),
                ha="left",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=10,
                arrowprops={"arrowstyle": "->", "color": TEXT_PRIMARY, "lw": 1.2, "connectionstyle": "arc3,rad=0.25"},
                zorder=5,
            )
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(axis="x" if args.horizontal else "y", which="major", color=GRID, lw=0.8, zorder=0)
        keep_spine = "left" if args.horizontal else "bottom"
        for spine in ax.spines:
            ax.spines[spine].set_visible(spine == keep_spine)
        ax.spines[keep_spine].set_color(GRID)
        if footnotes:
            fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=7.5, color=TEXT_SECONDARY)

        out = os.path.join(args.out_dir, f"throughput_{corpus.removesuffix('.txt')}_{tok_name.lower().replace(' ', '').replace('.', '_')}.{args.format}")
        fig.tight_layout(rect=(0, 0.03 * len(footnotes), 1, 1))
        fig.savefig(out)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
