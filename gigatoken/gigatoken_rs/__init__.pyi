from os import PathLike
from pathlib import Path

import awkward as ak
import numpy as np
import numpy.typing as npt

class FileSource:
    """Base class for file sources; construct TextFileSource or JsonlFileSource."""

    def __repr__(self) -> str: ...

class TextFileSource(FileSource):
    """Plain-text files. With `separator`, documents are the pieces between
    separator occurrences; without one, each file is a single document."""

    def __init__(
        self,
        paths: list[str | Path | PathLike[str]],
        separator: bytes | None = None,
    ) -> None: ...

class JsonlFileSource(FileSource):
    """JSON Lines files: one document per line, text taken from `field`."""

    def __init__(
        self,
        paths: list[str | Path | PathLike[str]],
        field: str = "text",
    ) -> None: ...

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
    def __new__(cls) -> "BPETokenizer": ...
    def encode(self, input: str | bytes) -> npt.NDArray[np.uint32]: ...
    def encode_batch(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        *,
        parallel: bool = True,
    ) -> ak.Array:
        """parallel=False encodes on the calling thread (identical output),
        never touching the process-global thread pool — for multiprocessing
        workers; gigatoken.Tokenizer passes it automatically."""
    def encode_batch_padded(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
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
    def decode(self, tokens: list[int] | npt.NDArray[np.uint32] | ak.Array) -> bytes: ...
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
    def from_hf(path: str | Path) -> "BPETokenizer": ...
    def __repr__(self) -> str: ...

class SentencePieceTokenizer:
    @staticmethod
    def from_hf(path: str | Path) -> "SentencePieceTokenizer": ...
    def encode(self, input: str | bytes) -> npt.NDArray[np.uint32]: ...
    def encode_batch(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        *,
        parallel: bool = True,
    ) -> ak.Array:
        """parallel=False encodes on the calling thread (identical output),
        never touching the process-global thread pool — for multiprocessing
        workers; gigatoken.Tokenizer passes it automatically."""
    def encode_batch_padded(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        options: PadTruncate,
        *,
        parallel: bool = True,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """encode_batch as one padded (rows x width) matrix plus each row's
        real (unpadded) length; prefer the keyword-argument wrapper
        gigatoken.Tokenizer.encode_batch_padded."""
    def encode_no_normalize(self, text: str) -> npt.NDArray[np.uint32]: ...
    def encode_files(
        self,
        source: FileSource | str | Path | PathLike[str] | list[str | Path | PathLike[str]],
        *,
        parallel: bool = True,
    ) -> ak.Array: ...
    def decode(self, tokens: list[int] | npt.NDArray[np.uint32] | ak.Array) -> bytes: ...
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

class PretokenizerIter:
    def __iter__(self) -> "PretokenizerIter": ...
    def __next__(self) -> bytes: ...

def pretokenizer(text: bytes) -> PretokenizerIter: ...
def pretokenized_counts(
    text: bytes,
    separator: bytes | None = None,
) -> list[tuple[bytes, int]]: ...
