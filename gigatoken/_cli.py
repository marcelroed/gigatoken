"""The `gigatoken` command-line interface.

Currently a single subcommand: `gigatoken bench` measures encode throughput
on a set of files, optionally comparing against (and validating token ids
with) the HuggingFace `tokenizers` library.
"""

from __future__ import annotations

import re
import time
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


def _parse_size(text: str) -> int:
    """Parse a decimal byte size like '100MB', '2.5GB', or '1000000'."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*", text.lower())
    if match is None:
        raise typer.BadParameter(f"cannot parse size {text!r}; expected something like 100MB")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2)])


def _load_tokenizer(spec: str) -> Tokenizer:
    from gigatoken import Tokenizer

    if spec.endswith(".tiktoken"):
        return Tokenizer.from_tiktoken(spec)
    if spec.endswith(".model"):
        return Tokenizer.from_sentencepiece(spec)
    return Tokenizer(spec)


def _read_docs(files: list[Path], separator: str | None) -> list[bytes]:
    """One document per file, or the separator-split (separator dropped,
    empty documents skipped) pieces of each file in order, as raw bytes."""
    docs: list[bytes] = []
    for file in files:
        raw = file.read_bytes()
        if separator is None:
            docs.append(raw)
        else:
            docs.extend(piece for piece in raw.split(separator.encode("utf-8")) if piece)
    return docs


def _subset_docs(docs: list[bytes], limit_bytes: int | None) -> tuple[list[bytes], bool]:
    """The prefix of `docs` totalling at most `limit_bytes`, byte-truncating
    the final document (at a valid UTF-8 boundary) to fill the budget.
    Returns the subset and whether its last document was truncated
    mid-document."""
    if limit_bytes is None:
        return docs, False
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


def _report(name: str, seconds: float, n_bytes: int, n_tokens: int) -> None:
    typer.echo(
        f"{name:>9}: {seconds:8.3f} s | {n_bytes / 1e6:10.2f} MB at {n_bytes / 1e6 / seconds:8.2f} MB/s | {n_tokens / 1e6:8.2f} Mtok at {n_tokens / 1e6 / seconds:7.2f} Mtok/s"
    )


@app.command()
def bench(
    tokenizer: str = typer.Argument(..., help="tokenizer.json path or directory, HuggingFace repo id, .tiktoken file, or sentencepiece .model file"),
    files: list[Path] = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="UTF-8 text files to encode"),
    in_memory: bool = typer.Option(False, "--in-memory", help="read the files into memory before timing, excluding disk IO from the measurement"),
    compare_to: Optional[str] = typer.Option(None, help="also benchmark another library on the same data (in memory); only 'hf' is supported"),
    comparison_limit: Optional[str] = typer.Option(None, help="cap the bytes fed to the comparison tokenizer, e.g. 100MB"),
    validate: bool = typer.Option(False, "--validate", help="check that HuggingFace token ids match gigatoken's on the comparison subset (implies --compare-to hf)"),
    separator: Optional[str] = typer.Option(None, help='document separator to split the files on, e.g. "<|endoftext|>"; whole files are single documents otherwise'),
) -> None:
    """Measure the time to encode FILES with TOKENIZER."""
    if validate and compare_to is None:
        compare_to = "hf"
    if compare_to not in (None, "hf"):
        raise typer.BadParameter("only --compare-to hf is supported")
    if comparison_limit is not None and compare_to is None:
        raise typer.BadParameter("--comparison-limit requires --compare-to hf (or --validate)")
    limit_bytes = _parse_size(comparison_limit) if comparison_limit is not None else None

    import awkward as ak

    gt_tokenizer = _load_tokenizer(tokenizer)

    # gigatoken pass. Without --in-memory and without a separator the files
    # are read inside Rust via encode_files, so timing includes disk IO.
    docs: list[bytes] | None = None
    if in_memory:
        docs = _read_docs(files, separator)
        gt_bytes = sum(len(doc) for doc in docs)
        start = time.perf_counter()
        encoded = gt_tokenizer.encode_batch(docs)
        gt_seconds = time.perf_counter() - start
    elif separator is None:
        gt_bytes = sum(file.stat().st_size for file in files)
        start = time.perf_counter()
        encoded = gt_tokenizer.encode_files(list(files))
        gt_seconds = time.perf_counter() - start
    else:
        start = time.perf_counter()
        docs = _read_docs(files, separator)
        encoded = gt_tokenizer.encode_batch(docs)
        gt_seconds = time.perf_counter() - start
        gt_bytes = sum(len(doc) for doc in docs)
    _report("gigatoken", gt_seconds, gt_bytes, int(ak.count(encoded)))

    if compare_to != "hf":
        return

    # Comparison pass: HuggingFace tokenizers on the (possibly size-capped)
    # same documents, always in memory.
    hf_json = gt_tokenizer._hf_json
    if hf_json is None:
        typer.echo("error: --compare-to hf needs a tokenizer with a HuggingFace configuration (.tiktoken files are not supported)", err=True)
        raise typer.Exit(1)
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ModuleNotFoundError:
        typer.echo("error: --compare-to hf requires the `tokenizers` package (pip install tokenizers)", err=True)
        raise typer.Exit(1)
    hf_tokenizer = HFTokenizer.from_str(hf_json.decode("utf-8") if isinstance(hf_json, bytes) else hf_json)

    if docs is None:
        docs = _read_docs(files, separator)
    subset, last_truncated = _subset_docs(docs, limit_bytes)
    if not subset:
        typer.echo("error: --comparison-limit is smaller than the first document; nothing to compare", err=True)
        raise typer.Exit(1)
    subset_bytes = sum(len(doc) for doc in subset)
    hf_docs = [doc.decode("utf-8") for doc in subset]  # tokenizers only accepts str

    start = time.perf_counter()
    hf_encodings = hf_tokenizer.encode_batch(hf_docs, add_special_tokens=False)
    hf_seconds = time.perf_counter() - start
    hf_ids = [encoding.ids for encoding in hf_encodings]
    _report("hf", hf_seconds, subset_bytes, sum(len(ids) for ids in hf_ids))
    typer.echo(f"gigatoken is {(gt_bytes / gt_seconds) / (subset_bytes / hf_seconds):.2f}x faster than hf (by MB/s)")

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
            typer.echo(
                f"validation FAILED: document {index}: first mismatch at token {mismatch} "
                f"(gigatoken {gt_doc[mismatch : mismatch + 5]}... vs hf {hf_doc[mismatch : mismatch + 5]}..., "
                f"lengths {len(gt_doc)} vs {len(hf_doc)})",
                err=True,
            )
            raise typer.Exit(1)
    guard_note = f" (last {_TRUNCATION_GUARD_TOKENS} tokens of the truncated final document ignored)" if last_truncated else ""
    typer.echo(f"validation OK: {len(subset)} documents match{guard_note}")


if __name__ == "__main__":
    app()
