"""Unified high-level Tokenizer wrapping the Rust backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gigatok._load.hf import to_tokenizer_json
from gigatok.gigatok_rs import BPETokenizer, SentencePieceTokenizer, load_hf_json

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    import awkward as ak
    import numpy as np
    import numpy.typing as npt

    from gigatok._load.hf import HFTokenizerLike
    from gigatok.gigatok_rs import FileSource

_BACKEND_TYPES = (BPETokenizer, SentencePieceTokenizer)


class Tokenizer:
    """A tokenizer in one of the standard formats supported by the library.

    Construct it from a path to a HuggingFace tokenizer.json (or a directory
    containing one), from a HuggingFace Hub repo id like
    "openai-community/gpt2" (downloaded directly; neither transformers,
    tokenizers, nor huggingface_hub needs to be installed), from an
    already-initialized HuggingFace tokenizer (a `tokenizers.Tokenizer` or a
    `transformers` tokenizer, fast or slow), or from an existing Rust backend
    instance. The right backend — byte-level BPE or SentencePiece BPE with
    byte fallback — is chosen automatically from the tokenizer's
    configuration.
    """

    def __init__(
        self,
        tokenizer: str | Path | PathLike[str] | Tokenizer | BPETokenizer | SentencePieceTokenizer | HFTokenizerLike,
    ) -> None:
        if isinstance(tokenizer, Tokenizer):
            self._backend = tokenizer._backend
        elif isinstance(tokenizer, _BACKEND_TYPES):
            self._backend = tokenizer
        else:
            self._backend = load_hf_json(to_tokenizer_json(tokenizer))

    @classmethod
    def from_json(cls, data: str | bytes) -> "Tokenizer":
        """Load from in-memory tokenizer.json contents."""
        return cls(load_hf_json(data))

    @classmethod
    def from_tiktoken(cls, path: str | Path) -> "Tokenizer":
        """Load from a .tiktoken vocabulary file."""
        return cls(BPETokenizer.from_tiktoken(path))

    @classmethod
    def from_sentencepiece(cls, source: str | Path | bytes) -> "Tokenizer":
        """Load from a raw sentencepiece .model file (path or contents).

        Supports BPE models with byte fallback; the .model's normalizer spec
        (precompiled charsmap, extra-whitespace removal, dummy prefix) is
        honored. Neither sentencepiece nor protobuf needs to be installed."""
        from pathlib import Path as _Path

        from gigatok._load.sentencepiece import sentencepiece_to_tokenizer_json

        data = source if isinstance(source, bytes) else _Path(source).read_bytes()
        return cls(load_hf_json(sentencepiece_to_tokenizer_json(data)))

    @property
    def backend(self) -> BPETokenizer | SentencePieceTokenizer:
        """The underlying Rust tokenizer (BPETokenizer or SentencePieceTokenizer)."""
        return self._backend

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary: one greater than the largest token ID,
        including added tokens."""
        return self._backend.vocab_size

    @property
    def vocab(self) -> dict[int, bytes]:
        """The vocabulary as a freshly built dict mapping token ID to token
        bytes, in ID order, including added tokens."""
        return self._backend.vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        """The merge rules as a freshly built list of `(left, right)` byte
        pairs in merge-priority order."""
        return self._backend.merges

    def encode(self, input: str | bytes) -> npt.NDArray[np.uint32]:
        return self._backend.encode(input)

    def encode_batch(self, inputs: list[str] | list[bytes] | ak.Array) -> ak.Array:
        return self._backend.encode_batch(inputs)

    def encode_files(
        self,
        source: FileSource | str | Path | PathLike[str] | list[str | Path | PathLike[str]],
    ) -> ak.Array:
        return self._backend.encode_files(source)

    def decode(self, tokens: list[int] | npt.NDArray[np.uint32] | ak.Array) -> bytes:
        return self._backend.decode(tokens)

    def __getattr__(self, name: str) -> Any:
        # Backend-specific extras (e.g. SentencePiece's encode_no_normalize).
        if name == "_backend":
            raise AttributeError(name)
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return f"Tokenizer({self._backend!r})"
