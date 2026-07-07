"""Curated DCLM test corpus: ~20 MB of diverse, edge-case-heavy documents.

Streams the head of one DCLM-baseline shard from HuggingFace (range requests
via HfFileSystem, no full-shard download), classifies each document into
edge-case categories (CJK, RTL/other scripts, NFC-divergent text, emoji,
control whitespace, code, giant unbroken tokens, ...), fills a byte quota per
category, and caches the selection at data/dclm_sample.jsonl.zst. The source
revision, scan window, and selection rules are fixed, so every machine builds
the identical fixture.

Build manually with: uv run python tests/dclm_fixture.py
"""

import io
import json
import re
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXTURE_PATH = DATA_DIR / "dclm_sample.jsonl.zst"

HF_REPO = "datasets/mlfoundations/dclm-baseline-1.0"
HF_REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
HF_SHARD = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"

TARGET_BYTES = 20_000_000  # total selection size (UTF-8 bytes of text)
SCAN_LIMIT_BYTES = 500_000_000  # stop scanning the shard after this much text
MAX_DOC_BYTES = 262_144  # skip anything larger outright

# Codepoint classes that make text hard for tokenizers.
_CJK_RE = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")
_OTHER_SCRIPT_RE = re.compile(r"[Ͱ-ϿЀ-ӿ԰-֏֐-׿؀-ۿऀ-ॿ฀-๿]")
_EMOJI_RE = re.compile(r"[←-⇿☀-➿⬀-⯿\U0001f000-\U0001faff]")
_COMBINING_RE = re.compile(r"[̀-ͯ᪰-᫿⃐-⃿︠-︯]")
_ODD_WS_RE = re.compile(r"[\t\r\x0b\f\x85\xa0​-‍  　﻿]| {8,}|\n{4,}")
_LONG_TOKEN_RE = re.compile(r"\S{80,}")
_CODE_MARK_RE = re.compile(r"[{};]|\n[ \t]+\S|</\w|&\w+;")
_DIGIT_RE = re.compile(r"[0-9]")


def _count(pattern: re.Pattern, text: str, at_least: int) -> bool:
    """True if `pattern` matches at least `at_least` times in `text`."""
    n = 0
    for _ in pattern.finditer(text):
        n += 1
        if n >= at_least:
            return True
    return False


# (category, byte quota). A document lands in the first category it matches
# that still has quota left; order therefore puts the rarest classes first.
QUOTAS: list[tuple[str, int]] = [
    ("nfc_divergent", 1_500_000),
    ("other_script", 2_000_000),
    ("cjk", 2_000_000),
    ("combining", 1_000_000),
    ("emoji", 1_500_000),
    ("odd_whitespace", 2_000_000),
    ("long_token", 1_500_000),
    ("code", 2_500_000),
    ("digit_heavy", 1_500_000),
    ("huge", 1_500_000),
    ("tiny", 500_000),
    ("general", 2_500_000),
]


def _classify(text: str, nbytes: int) -> list[str]:
    """All categories a document qualifies for, in QUOTAS priority order."""
    cats = []
    if not text.isascii():
        if not unicodedata.is_normalized("NFC", text):
            cats.append("nfc_divergent")
        if _count(_OTHER_SCRIPT_RE, text, 50):
            cats.append("other_script")
        if _count(_CJK_RE, text, 50):
            cats.append("cjk")
        if _count(_COMBINING_RE, text, 10):
            cats.append("combining")
        if _count(_EMOJI_RE, text, 5):
            cats.append("emoji")
    if _count(_ODD_WS_RE, text, 5):
        cats.append("odd_whitespace")
    if _LONG_TOKEN_RE.search(text):
        cats.append("long_token")
    if _count(_CODE_MARK_RE, text, max(20, nbytes // 200)):
        cats.append("code")
    if len(_DIGIT_RE.findall(text)) * 6 > len(text):
        cats.append("digit_heavy")
    if nbytes >= 131_072:
        cats.append("huge")
    if nbytes < 200:
        cats.append("tiny")
    cats.append("general")
    return cats


def _iter_shard_texts():
    """Yield document texts from the pinned DCLM shard, streamed from HF."""
    import zstandard
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"{HF_REPO}@{HF_REVISION}/{HF_SHARD}"
    with fs.open(path, "rb", block_size=16 * 2**20) as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                text = json.loads(line).get("text")
                if text:
                    yield text


def build_dclm_sample(dest: Path = FIXTURE_PATH, log=lambda msg: None) -> Path:
    import hashlib

    import zstandard

    order = [cat for cat, _ in QUOTAS]
    remaining = dict(QUOTAS)
    selected: list[tuple[str, str]] = []  # (category, text)
    spill: list[str] = []  # overflow docs kept to top up unfilled quotas
    spill_bytes = 0
    seen: set[bytes] = set()
    scanned = 0
    total = 0

    for text in _iter_shard_texts():
        nbytes = len(text.encode("utf-8"))
        scanned += nbytes
        if scanned >= SCAN_LIMIT_BYTES or total >= TARGET_BYTES:
            break
        if nbytes > MAX_DOC_BYTES:
            continue
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        for cat in _classify(text, nbytes):
            if remaining[cat] > 0:
                selected.append((cat, text))
                remaining[cat] -= nbytes
                total += nbytes
                break
        else:
            if spill_bytes < 8_000_000:
                spill.append(text)
                spill_bytes += nbytes
        if scanned % (2**26) < nbytes:
            log(f"scanned {scanned / 1e6:.0f} MB, selected {total / 1e6:.1f} MB")

    # Quotas the scan window couldn't fill are topped up from ordinary
    # overflow docs so the fixture still reaches ~20 MB.
    for text in spill:
        if total >= TARGET_BYTES:
            break
        selected.append(("general", text))
        total += len(text.encode("utf-8"))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        with zstandard.ZstdCompressor(level=10).stream_writer(fh) as writer:
            for cat, text in selected:
                writer.write(json.dumps({"text": text, "category": cat}, ensure_ascii=False).encode("utf-8"))
                writer.write(b"\n")
    tmp.replace(dest)

    counts: dict[str, int] = {cat: 0 for cat in order}
    for cat, _ in selected:
        counts[cat] += 1
    quota = dict(QUOTAS)
    log(f"selected {len(selected)} docs, {total / 1e6:.1f} MB (scanned {scanned / 1e6:.0f} MB)")
    for cat in order:
        log(f"  {cat}: {counts[cat]} docs, {(quota[cat] - remaining[cat]) / 1e6:.2f}/{quota[cat] / 1e6:.1f} MB")
    return dest


def ensure_dclm_sample(dest: Path = FIXTURE_PATH) -> Path:
    """Return the fixture path, building it from HuggingFace on first use."""
    if not dest.exists():
        build_dclm_sample(dest, log=lambda msg: print(f"[dclm_fixture] {msg}", file=sys.stderr))
    return dest


def load_dclm_texts(path: Path = FIXTURE_PATH) -> list[str]:
    import zstandard

    texts = []
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE_PATH
    build_dclm_sample(dest, log=print)
