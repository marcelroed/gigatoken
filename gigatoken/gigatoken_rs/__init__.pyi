from os import PathLike
from pathlib import Path
from typing import Literal

import awkward as ak
import numpy as np
import numpy.typing as npt

class FileSource:
    """Base class for file sources; construct TextFileSource, JsonlFileSource,
    or ParquetFileSource."""

    def __repr__(self) -> str: ...

class TextFileSource(FileSource):
    """Plain-text files. With `separator` (str separators are encoded to
    UTF-8 bytes), documents are the pieces between separator occurrences;
    without one, each file is a single document."""

    def __init__(
        self,
        paths: list[str | Path | PathLike[str]],
        separator: bytes | str | None = None,
    ) -> None: ...

class JsonlFileSource(FileSource):
    """JSON Lines files: one document per line, text taken from `field`."""

    def __init__(
        self,
        paths: list[str | Path | PathLike[str]],
        field: str = "text",
    ) -> None: ...

class ParquetFileSource(FileSource):
    """Parquet files: one document per row, text taken from `column` (a
    string or binary column). Null rows become empty documents, so results
    stay row-aligned with the table."""

    def __init__(
        self,
        paths: list[str | Path | PathLike[str]],
        column: str = "text",
    ) -> None: ...

class BytesSource:
    """In-memory bytes for encode_batch, the buffer analog of TextFileSource:
    documents are the pieces between separator occurrences (empty pieces
    skipped), or one document per buffer without a separator. Buffers are
    borrowed and split inside the parallel encode — pass whole buffers plus
    a separator rather than pre-splitting."""

    def __init__(
        self,
        data: bytes | list[bytes],
        separator: bytes | str | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

def train_bpe(
    in_data: bytes | Path | str | FileSource,
    vocab_size: int,
    special_tokens: list[str],
    tie_breaking: str = "huggingface",
    separator: bytes | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: ...

class PadTruncate:
    """How encode_batch_padded assembles rows into a rectangular matrix; see
    src/bindings/padding.rs for the semantics. Frozen: fields are validated
    at construction and read-only afterwards."""

    def __init__(
        self,
        pad_id: int,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
        truncate: bool = False,
        pad_left: bool = False,
        truncate_left: bool = False,
        prefix: list[int] = [],
        suffix: list[int] = [],
    ) -> None: ...

    pad_id: int
    max_length: int | None
    pad_to_max_length: bool
    truncate: bool
    pad_left: bool
    truncate_left: bool
    prefix: list[int]
    suffix: list[int]
    def __repr__(self) -> str: ...

class BPETokenizer:
    def encode(self, input: str | bytes) -> npt.NDArray[np.uint32]: ...
    def encode_batch(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool = True,
    ) -> ak.Array:
        """parallel=False encodes on the calling thread (identical output),
        never touching the process-global thread pool — for multiprocessing
        workers; gigatoken.Tokenizer passes it automatically."""
    def encode_batch_list(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool = True,
    ) -> list[list[int]]:
        """encode_batch returned as plain Python lists assembled in Rust;
        same inputs and `parallel` keyword as encode_batch."""
    def _encode_batch_list_compat(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        options: _WrapTruncate,
        *,
        parallel: bool = True,
    ) -> list[list[int]]:
        """Non-public entrypoint for the compat wrappers: encode_batch_list
        with row assembly options (see _WrapTruncate)."""
    def encode_batch_padded(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        options: PadTruncate,
        *,
        parallel: bool = True,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """encode_batch as one padded (rows x width) matrix plus each row's
        real (unpadded) length; prefer the keyword-argument wrapper
        gigatoken.Tokenizer.encode_batch_padded."""
    def encode_files(
        self,
        source: FileSource | str | Path | PathLike[str] | list[str | Path | PathLike[str]],
        *,
        parallel: bool = True,
    ) -> ak.Array: ...
    def decode(self, tokens: list[int] | npt.NDArray[np.integer] | ak.Array) -> bytes:
        """Integer numpy arrays are borrowed (uint32) or converted in one
        pass (other dtypes); sequences are extracted per element."""
    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary: one greater than the largest token ID,
        including added tokens."""
    @property
    def vocab(self) -> dict[int, bytes]:
        """The vocabulary as a freshly built dict mapping token ID to token
        bytes, in ID order, including added tokens."""
    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        """The merge rules as a freshly built list of `(left, right)` byte
        pairs in merge-priority order."""
    @staticmethod
    def from_tiktoken(path: str | Path) -> "BPETokenizer": ...
    @staticmethod
    def from_tiktoken_model(model_path: str | Path, config_path: str | Path, pretokenizer: str) -> "BPETokenizer":
        """Load from a tiktoken rank file plus a tokenizer_config.json
        carrying the special tokens — the layout of repos that ship no
        tokenizer.json (e.g. the moonshotai Kimi line) — with the named
        pretokenizer scheme ("gpt2", "gpt4", "qwen2", "qwen35", "olmo3",
        "deepseek_v3", "o200k", "nemotron", or "kimi")."""
    @staticmethod
    def from_hf(path: str | Path) -> "BPETokenizer": ...
    def __repr__(self) -> str: ...

class SentencePieceTokenizer:
    @staticmethod
    def from_hf(path: str | Path) -> "SentencePieceTokenizer": ...
    def encode(self, input: str | bytes) -> npt.NDArray[np.uint32]: ...
    def encode_batch(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool = True,
    ) -> ak.Array:
        """parallel=False encodes on the calling thread (identical output),
        never touching the process-global thread pool — for multiprocessing
        workers; gigatoken.Tokenizer passes it automatically."""
    def encode_batch_padded(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        options: PadTruncate,
        *,
        parallel: bool = True,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """encode_batch as one padded (rows x width) matrix plus each row's
        real (unpadded) length; prefer the keyword-argument wrapper
        gigatoken.Tokenizer.encode_batch_padded."""
    def encode_batch_list(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool = True,
    ) -> list[list[int]]:
        """encode_batch returned as plain Python lists assembled in Rust;
        same inputs and `parallel` keyword as encode_batch."""
    def _encode_batch_list_compat(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        options: _WrapTruncate,
        *,
        parallel: bool = True,
    ) -> list[list[int]]:
        """Non-public entrypoint for the compat wrappers: encode_batch_list
        with row assembly options (see _WrapTruncate)."""
    def encode_no_normalize(self, text: str) -> npt.NDArray[np.uint32]: ...
    def encode_files(
        self,
        source: FileSource | str | Path | PathLike[str] | list[str | Path | PathLike[str]],
        *,
        parallel: bool = True,
    ) -> ak.Array: ...
    def decode(self, tokens: list[int] | npt.NDArray[np.integer] | ak.Array) -> bytes:
        """Integer numpy arrays are borrowed (uint32) or converted in one
        pass (other dtypes); sequences are extracted per element."""
    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary: one greater than the largest token ID,
        including added tokens."""
    @property
    def vocab(self) -> dict[int, bytes]:
        """The vocabulary as a freshly built dict mapping token ID to token
        bytes, in ID order, including added tokens."""
    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        """The merge rules as a freshly built list of `(left, right)` byte
        pairs in merge-priority order."""
    def __repr__(self) -> str: ...

def load_hf_json(
    data: str | bytes,
) -> BPETokenizer | SentencePieceTokenizer:
    """Load a tokenizer from in-memory tokenizer.json contents; the model's
    byte_fallback flag selects SentencePieceTokenizer vs BPETokenizer."""

def hub_file(
    repo_id: str,
    filename: str = "tokenizer.json",
    *,
    repo_type: Literal["model", "dataset", "space"] = "model",
    revision: str = "main",
) -> Path:
    """Path of `filename` from Hub repo `repo_id` at `revision`, served from
    the standard HF cache, downloading into it first when absent. Raises
    FileNotFoundError on HTTP 404 and PermissionError on 401/403."""

def looks_like_repo_id(name: str) -> bool:
    """Whether `name` is shaped like a HuggingFace Hub repo id: `org/name` or
    a bare legacy repo name, and not an obvious local tokenizer-file path."""

def get_hf_token() -> str | None:
    """The HuggingFace access token, discovered like huggingface_hub does it:
    HF_TOKEN (or legacy HUGGING_FACE_HUB_TOKEN), then the token file
    (HF_TOKEN_PATH, default $HF_HOME/token)."""

class SpecialTokenFound(Exception):
    """Raised by _encode_batch_list_compat when a `forbid` pattern occurs in
    a document; args[0] holds the sorted indices of every matched pattern."""

class _SubstringMatcher:
    """Compiled multi-pattern substring matcher (Aho-Corasick), built once
    and reused across calls — plain substring containment, exactly like
    `pattern in text` for each pattern. Carried by _WrapTruncate's `forbid`
    to scan documents inside the encode call."""

    def __init__(self, patterns: list[str]) -> None: ...
    def present(self, text: str) -> list[int]:
        """Sorted indices of the patterns that occur in `text`. Runs with
        the GIL released."""
    def __len__(self) -> int: ...

class _WrapTruncate:
    """How _encode_batch_list_compat assembles each row, plus its fused
    forbidden-specials scan — the compat wrappers' per-call options, bundled
    like PadTruncate. Frozen: fields are validated at construction.
    `max_tokens` caps the encoded ids per row (from the left when
    `truncate_left`), not counting `prefix`/`suffix`; `forbid` patterns
    raise SpecialTokenFound when found in any document."""

    def __init__(
        self,
        *,
        prefix: list[int] = [],
        suffix: list[int] = [],
        max_tokens: int | None = None,
        truncate_left: bool = False,
        forbid: _SubstringMatcher | None = None,
    ) -> None: ...

class PretokenizerIter:
    def __iter__(self) -> "PretokenizerIter": ...
    def __next__(self) -> bytes: ...

def pretokenizer(text: bytes) -> PretokenizerIter: ...
def pretokenized_counts(
    text: bytes,
    separator: bytes | None = None,
) -> list[tuple[bytes, int]]: ...
