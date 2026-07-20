"""The `gigatoken` command-line interface.

Currently a single subcommand: `gigatoken bench` measures encode throughput
on a set of files, optionally comparing against (and validating token ids
with) the HuggingFace `tokenizers` library.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    import typer
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("the gigatoken CLI requires typer; install it with `pip install 'gigatoken[cli]'`") from e

if TYPE_CHECKING:
    from gigatoken import Tokenizer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="gigatoken command-line tools.")


@app.callback()
def _main() -> None:
    # With a single @app.command(), typer would otherwise promote it to the
    # top level; this keeps the `gigatoken bench ...` subcommand form.
    pass


# Tokens dropped from the end of a byte-truncated document during --validate:
# cutting the text mid-document can change the final merges, so the tail is
# not expected to match.
_TRUNCATION_GUARD_TOKENS = 100

_SIZE_UNITS = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}


def _parse_size(text: str) -> int | None:
    """Parse a decimal byte size like '100MB', '2.5GB', or '1000000';
    'none' means no limit."""
    if text.strip().lower() in ("none", "unlimited"):
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*", text.lower())
    if match is None:
        raise typer.BadParameter(f"cannot parse size {text!r}; expected something like 100MB (or 'none')")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2)])


def _cpu_info() -> str:
    """The benchmark machine's CPU as 'name, N cores', plus ', M sockets'
    when there is more than one socket."""
    name: str | None = None
    cores: int | None = None
    sockets: int | None = None
    system = platform.system()
    if system == "Darwin":

        def _sysctl(key: str) -> str | None:
            try:
                return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip() or None
            except (OSError, subprocess.CalledProcessError):
                return None

        name = _sysctl("machdep.cpu.brand_string")
        if (value := _sysctl("hw.physicalcpu")) is not None and value.isdigit():
            cores = int(value)
        if (value := _sysctl("hw.packages")) is not None and value.isdigit():
            sockets = int(value)
    elif system == "Linux":
        # Within each processor block "physical id" precedes "core id", so
        # (socket, core) pairs count physical cores across sockets.
        physical_ids: set[str] = set()
        socket_core_ids: set[tuple[str, str]] = set()
        physical_id = ""
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name" and name is None:
                    name = value
                elif key == "physical id":
                    physical_id = value
                    physical_ids.add(value)
                elif key == "core id":
                    socket_core_ids.add((physical_id, value))
        except OSError:
            pass
        cores = len(socket_core_ids) or None
        sockets = len(physical_ids) or None
    name = name or platform.processor() or platform.machine() or "unknown CPU"
    cores = cores or os.cpu_count()
    parts = [name]
    if cores is not None:
        parts.append(f"{cores} core" + ("s" if cores != 1 else ""))
    if sockets is not None and sockets > 1:
        parts.append(f"{sockets} sockets")
    return ", ".join(parts)


def _available_memory_bytes() -> int | None:
    """A best-effort estimate of the memory currently available without
    swapping, or None where it cannot be determined."""
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif system == "Darwin":
        # Free + inactive + purgeable + speculative pages are all reclaimable
        # on demand, matching what Linux reports as MemAvailable.
        try:
            page_size = int(subprocess.run(["sysctl", "-n", "hw.pagesize"], capture_output=True, text=True, check=True).stdout)
            pages = 0
            for line in subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
                    pages += int(value.strip().rstrip("."))
            return pages * page_size
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    return None


def _warn_if_memory_tight(total_bytes: int, comparing: bool, limit_bytes: int | None) -> None:
    """Warn when the default in-memory benchmark looks unlikely to fit: the
    raw file bytes plus roughly as much again for the encoded token ids,
    plus per-document bytes/str/id copies of the comparison subset."""
    available = _available_memory_bytes()
    if available is None:
        return
    needed = 2 * total_bytes
    if comparing:
        needed += 3 * (min(limit_bytes, total_bytes) if limit_bytes is not None else total_bytes)
    if needed > available:
        typer.secho(
            f"warning: this run may need ~{needed / 1e9:.1f} GB of memory but only ~{available / 1e9:.1f} GB looks available; "
            "consider --stream-from-disk to avoid holding the input files in memory",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _load_tokenizer(spec: str) -> Tokenizer:
    from gigatoken import Tokenizer

    if spec.endswith(".tiktoken"):
        return Tokenizer.from_tiktoken(spec)
    if spec.endswith(".model"):
        return Tokenizer.from_sentencepiece(spec)
    return Tokenizer(spec)


def _split_docs(raws: Iterable[bytes], separator: str | None) -> Iterator[bytes]:
    """One document per input, or the separator-split (separator dropped,
    empty documents skipped) pieces of each input in order, as raw bytes.
    Lazy, so a byte-limited consumer never touches the tail inputs."""
    for raw in raws:
        if separator is None:
            yield raw
        else:
            yield from (piece for piece in raw.split(separator.encode("utf-8")) if piece)


def _subset_docs(docs: Iterable[bytes], limit_bytes: int | None) -> tuple[list[bytes], bool]:
    """The prefix of `docs` totalling at most `limit_bytes`, byte-truncating
    the final document (at a valid UTF-8 boundary) to fill the budget.
    Returns the subset and whether its last document was truncated
    mid-document."""
    if limit_bytes is None:
        return list(docs), False
    subset: list[bytes] = []
    used = 0
    for doc in docs:
        if used + len(doc) <= limit_bytes:
            subset.append(doc)
            used += len(doc)
        else:
            room = limit_bytes - used
            # Re-encoding after errors="ignore" drops a partial trailing
            # UTF-8 character so both tokenizers see identical content.
            truncated = doc[:room].decode("utf-8", errors="ignore").encode("utf-8")
            if truncated:
                subset.append(truncated)
                return subset, True
            return subset, False
    return subset, False


def _label(name: str) -> str:
    """A right-aligned row label, styled after padding so the ANSI codes
    don't count against the field width."""
    return typer.style(f"{name:>9}", fg=typer.colors.CYAN, bold=True)


def _report(name: str, seconds: float, n_bytes: int, n_tokens: int) -> None:
    mb_per_s = typer.style(f"{n_bytes / 1e6 / seconds:8.2f}", fg=typer.colors.GREEN, bold=True)
    mtok_per_s = typer.style(f"{n_tokens / 1e6 / seconds:7.2f}", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        f"{_label(name)}: {seconds:8.3f} s | {n_bytes / 1e6:10.2f} MB at {mb_per_s} MB/s | {n_tokens / 1e6:8.2f} Mtok at {mtok_per_s} Mtok/s"
    )


@app.command()
def bench(
    tokenizer: str = typer.Argument(..., help="tokenizer.json path or directory, HuggingFace repo id, .tiktoken file, or sentencepiece .model file"),
    files: list[Path] = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="UTF-8 text files to encode"),
    stream_from_disk: bool = typer.Option(False, "--stream-from-disk", help="stream the files from disk inside the timed region instead of reading them into memory up front, so disk IO counts toward the measurement"),
    compare_to: Optional[str] = typer.Option(None, help="also benchmark another library on the same data; only 'hf' is supported"),
    comparison_limit: str = typer.Option("100MB", help="cap the bytes fed to the comparison tokenizer, e.g. 100MB; 'none' compares on everything"),
    validate: bool = typer.Option(False, "--validate", help="check that HuggingFace token ids match gigatoken's on the comparison subset (implies --compare-to hf)"),
    doc_separator: Optional[str] = typer.Option(None, "--doc-separator", help='document separator to split the files on, e.g. "<|endoftext|>"; whole files are single documents otherwise'),
) -> None:
    """Measure the time to encode FILES with TOKENIZER."""
    if validate and compare_to is None:
        compare_to = "hf"
    if compare_to not in (None, "hf"):
        raise typer.BadParameter("only --compare-to hf is supported")
    limit_bytes = _parse_size(comparison_limit)

    typer.echo(f"{_label('cpu')}: {_cpu_info()}")

    import awkward as ak

    from gigatoken import BytesSource, TextFileSource

    gt_tokenizer = _load_tokenizer(tokenizer)

    # gigatoken pass. A separator is handed to gigatoken along with the whole
    # files (documents are split inside Rust, during the encode itself) —
    # never pre-split into per-document objects here, which is several times
    # slower. By default the files are read into memory before timing,
    # excluding disk IO from the measurement; with --stream-from-disk they
    # are instead read inside Rust via encode_files, so timing includes disk
    # IO. Byte counts are whole-file bytes, separators included.
    gt_bytes = sum(file.stat().st_size for file in files)
    if stream_from_disk:
        # Bare paths keep their extension-based format detection (.jsonl).
        source = list(files) if doc_separator is None else TextFileSource(list(files), separator=doc_separator)
        start = time.perf_counter()
        encoded = gt_tokenizer.encode_files(source)
        gt_seconds = time.perf_counter() - start
    else:
        _warn_if_memory_tight(gt_bytes, comparing=compare_to is not None, limit_bytes=limit_bytes)
        raws = [file.read_bytes() for file in files]
        start = time.perf_counter()
        encoded = gt_tokenizer.encode_batch(BytesSource(raws, separator=doc_separator))
        gt_seconds = time.perf_counter() - start
    _report("gigatoken", gt_seconds, gt_bytes, int(ak.count(encoded)))

    if compare_to != "hf":
        return

    # Comparison pass: HuggingFace tokenizers on the (possibly size-capped)
    # same documents, always in memory.
    hf_json = gt_tokenizer._hf_json
    if hf_json is None:
        typer.secho("error: --compare-to hf needs a tokenizer with a HuggingFace configuration (.tiktoken files are not supported)", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ModuleNotFoundError:
        typer.secho("error: --compare-to hf requires the `tokenizers` package (pip install tokenizers)", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    hf_tokenizer = HFTokenizer.from_str(hf_json.decode("utf-8") if isinstance(hf_json, bytes) else hf_json)

    # The comparison needs one Python object per document (tokenizers takes
    # a str per document, and validation compares per-document ids), so the
    # files are split in Python. In the default in-memory mode that happens
    # outside the timed region, mirroring the gigatoken pass; with
    # --stream-from-disk gigatoken's timing included reading and splitting
    # the files, so to keep the comparison fair hf's timed region covers
    # reading, splitting, and decoding too (lazily, so at most the
    # comparison subset is read from disk).
    if stream_from_disk:
        start = time.perf_counter()
        subset, last_truncated = _subset_docs(_split_docs((file.read_bytes() for file in files), doc_separator), limit_bytes)
        hf_docs = [doc.decode("utf-8") for doc in subset]  # tokenizers only accepts str
        hf_encodings = hf_tokenizer.encode_batch(hf_docs, add_special_tokens=False)
        hf_seconds = time.perf_counter() - start
    else:
        subset, last_truncated = _subset_docs(_split_docs(raws, doc_separator), limit_bytes)
        hf_docs = [doc.decode("utf-8") for doc in subset]  # tokenizers only accepts str
        start = time.perf_counter()
        hf_encodings = hf_tokenizer.encode_batch(hf_docs, add_special_tokens=False)
        hf_seconds = time.perf_counter() - start
    if not subset:
        typer.secho("error: --comparison-limit is smaller than the first document; nothing to compare", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    subset_bytes = sum(len(doc) for doc in subset)
    hf_ids = [encoding.ids for encoding in hf_encodings]
    _report("hf", hf_seconds, subset_bytes, sum(len(ids) for ids in hf_ids))
    speedup = (gt_bytes / gt_seconds) / (subset_bytes / hf_seconds)
    speedup_text = typer.style(f"{speedup:.2f}x", fg=typer.colors.GREEN if speedup >= 1 else typer.colors.RED, bold=True)
    typer.echo(f"gigatoken is {speedup_text} {'faster' if speedup >= 1 else 'slower'} than hf")

    if not validate:
        return
    gt_ids = gt_tokenizer.encode_batch_list(subset)
    for index, (gt_doc, hf_doc) in enumerate(zip(gt_ids, hf_ids)):
        if last_truncated and index == len(subset) - 1:
            # The final document was cut at a byte limit, which can change
            # its last merges (or split a UTF-8 character), so only the
            # common prefix minus a guard is expected to match.
            compare_to_length = min(len(gt_doc), len(hf_doc)) - _TRUNCATION_GUARD_TOKENS
            gt_doc, hf_doc = gt_doc[:compare_to_length], hf_doc[:compare_to_length]
            if compare_to_length <= 0:
                continue
        if gt_doc != hf_doc:
            mismatch = next((i for i, (a, b) in enumerate(zip(gt_doc, hf_doc)) if a != b), min(len(gt_doc), len(hf_doc)))
            typer.secho(
                f"validation FAILED: document {index}: first mismatch at token {mismatch} "
                f"(gigatoken {gt_doc[mismatch : mismatch + 5]}... vs hf {hf_doc[mismatch : mismatch + 5]}..., "
                f"lengths {len(gt_doc)} vs {len(hf_doc)})",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            raise typer.Exit(1)
    guard_note = f" (last {_TRUNCATION_GUARD_TOKENS} tokens of the truncated final document ignored)" if last_truncated else ""
    typer.echo(typer.style(f"validation OK: {len(subset)} documents match", fg=typer.colors.GREEN) + guard_note)


if __name__ == "__main__":
    app()
