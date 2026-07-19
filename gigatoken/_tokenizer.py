"""Unified high-level Tokenizer wrapping the Rust backends."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from gigatoken._load.hf import capture_named_special_tokens, to_tokenizer_json, try_load_from_config
from gigatoken._parallel import resolve_parallel
from gigatoken.gigatoken_rs import BPETokenizer, PadTruncate, SentencePieceTokenizer, load_hf_json

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    import awkward as ak
    import numpy as np
    import numpy.typing as npt

    from gigatoken._hf_compat import HFCompat
    from gigatoken._load.hf import HFTokenizerLike
    from gigatoken._tiktoken_compat import TiktokenCompat
    from gigatoken.gigatoken_rs import BytesSource, FileSource, _WrapTruncate

_BACKEND_TYPES = (BPETokenizer, SentencePieceTokenizer)

_TIKTOKEN_ENDOFTEXT = "<|endoftext|>"


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

    For drop-in use in code written against another tokenizer API, wrap it
    with `as_hf()` (transformers fast-tokenizer API) or `as_tiktoken()`
    (tiktoken.Encoding API), e.g.
    `hf_compatible = gigatoken.Tokenizer(hf_tokenizer).as_hf()`.
    """

    def __init__(
        self,
        tokenizer: str | Path | PathLike[str] | Tokenizer | BPETokenizer | SentencePieceTokenizer | HFTokenizerLike,
    ) -> None:
        # Source metadata kept for the as_hf()/as_tiktoken() adapters: the
        # tokenizer.json contents when loaded from one, the named special
        # tokens when the source is a transformers tokenizer, and the special
        # tokens when loaded from a .tiktoken file.
        self._hf_json: str | bytes | None = None
        self._hf_config_cache: dict[str, Any] | None = None
        self._named_specials: dict[str, str | list[str]] = {}
        self._tiktoken_specials: dict[str, int] | None = None
        if isinstance(tokenizer, Tokenizer):
            self._backend = tokenizer._backend
            self._hf_json = tokenizer._hf_json
            self._hf_config_cache = tokenizer._hf_config_cache
            self._named_specials = tokenizer._named_specials
            self._tiktoken_specials = tokenizer._tiktoken_specials
        elif isinstance(tokenizer, _BACKEND_TYPES):
            self._backend = tokenizer
        else:
            if isinstance(tokenizer, (str, os.PathLike)):
                # Config-driven dispatch: tokenizer_config.json names the
                # tokenizer implementation (tokenizer_class/auto_map); repos
                # of a registered line (e.g. Kimi/Moonlight) ship no
                # tokenizer.json and load from their line's vocab file.
                loaded = try_load_from_config(tokenizer)
                if loaded is not None:
                    self._backend, self._tiktoken_specials = loaded
                    return
            data = to_tokenizer_json(tokenizer)
            self._backend = load_hf_json(data)
            self._hf_json = data
            if not isinstance(tokenizer, (str, os.PathLike)):
                self._named_specials = capture_named_special_tokens(tokenizer)

    @classmethod
    def from_json(cls, data: str | bytes) -> "Tokenizer":
        """Load from in-memory tokenizer.json contents."""
        tokenizer = cls(load_hf_json(data))
        tokenizer._hf_json = data
        return tokenizer

    @classmethod
    def from_tiktoken(cls, path: str | Path) -> "Tokenizer":
        """Load from a .tiktoken vocabulary file."""
        tokenizer = cls(BPETokenizer.from_tiktoken(path))
        # The Rust loader registers <|endoftext|> right after the mergeable
        # ranks (see src/load_tokenizer/tiktoken.rs), which are contiguous.
        tokenizer._tiktoken_specials = {_TIKTOKEN_ENDOFTEXT: tokenizer.vocab_size - 1}
        return tokenizer

    @classmethod
    def from_sentencepiece(cls, source: str | Path | bytes) -> "Tokenizer":
        """Load from a raw sentencepiece .model file (path or contents).

        Supports BPE models with byte fallback; the .model's normalizer spec
        (precompiled charsmap, extra-whitespace removal, dummy prefix) is
        honored. Neither sentencepiece nor protobuf needs to be installed."""
        from pathlib import Path as _Path

        from gigatoken._load.sentencepiece import sentencepiece_to_tokenizer_json

        data = source if isinstance(source, bytes) else _Path(source).read_bytes()
        return cls.from_json(sentencepiece_to_tokenizer_json(data))

    def as_hf(self) -> HFCompat:
        """Wrap this tokenizer in a `gigatoken.HFCompat`, a drop-in for the
        HuggingFace `transformers` fast-tokenizer API."""
        from gigatoken._hf_compat import HFCompat

        return HFCompat(self)

    def as_tiktoken(self) -> TiktokenCompat:
        """Wrap this tokenizer in a `gigatoken.TiktokenCompat`, a drop-in for
        the `tiktoken.Encoding` API."""
        from gigatoken._tiktoken_compat import TiktokenCompat

        return TiktokenCompat(self)

    def _hf_config(self) -> dict[str, Any]:
        """The parsed tokenizer.json this tokenizer was loaded from."""
        if self._hf_json is None:
            raise ValueError(
                "this Tokenizer was not loaded from a HuggingFace tokenizer.json "
                "(e.g. it wraps a raw backend or a .tiktoken file), so its "
                "HuggingFace-side configuration is unavailable"
            )
        if self._hf_config_cache is None:
            self._hf_config_cache = json.loads(self._hf_json)
        return self._hf_config_cache

    def _special_tokens(self) -> dict[str, int]:
        """Special-token string -> id, from whichever source metadata exists
        (empty for tokenizers wrapping a raw backend)."""
        if self._tiktoken_specials is not None:
            return dict(self._tiktoken_specials)
        if self._hf_json is not None:
            added = self._hf_config().get("added_tokens") or []
            return {str(t["content"]): int(t["id"]) for t in added if t.get("special")}
        return {}

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

    def encode_batch(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool | None = None,
    ) -> ak.Array:
        """Encode a batch of documents; see the backend's encode_batch.

        `parallel=None` (the default) auto-detects: batches encode in
        parallel on the process-global thread pool, except inside a
        multiprocessing worker (or forked child), where everything runs on
        the calling thread so worker processes compose instead of
        oversubscribing — or, after a fork, deadlocking. Pass True/False to
        override. Output is identical either way."""
        return self._backend.encode_batch(inputs, parallel=resolve_parallel(parallel))

    def encode_batch_padded(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        pad_id: int,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
        truncate: bool = False,
        pad_left: bool = False,
        truncate_left: bool = False,
        prefix: list[int] | None = None,
        suffix: list[int] | None = None,
        *,
        parallel: bool | None = None,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """encode_batch assembled in Rust into one padded (rows x width)
        uint32 matrix, plus each row's real (unpadded) length; see
        gigatoken_rs.BPETokenizer.encode_batch_padded for the semantics and
        encode_batch here for `parallel`."""
        options = PadTruncate(
            pad_id,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            truncate=truncate,
            pad_left=pad_left,
            truncate_left=truncate_left,
            prefix=prefix or [],
            suffix=suffix or [],
        )
        return self._backend.encode_batch_padded(inputs, options, parallel=resolve_parallel(parallel))

    def encode_batch_list(
        self,
        inputs: list[str] | list[bytes] | BytesSource | ak.Array,
        *,
        parallel: bool | None = None,
    ) -> list[list[int]]:
        """encode_batch returned as plain Python lists, one list of token ids
        per document, assembled in Rust — for callers that need lists rather
        than the awkward Array. Same inputs as encode_batch, and see there
        for `parallel`."""
        return self._backend.encode_batch_list(inputs, parallel=resolve_parallel(parallel))

    def _encode_batch_list_compat(
        self,
        inputs: list[str] | list[bytes] | ak.Array,
        options: _WrapTruncate,
        *,
        parallel: bool | None = None,
    ) -> list[list[int]]:
        """Non-public entrypoint for the compat wrappers: encode_batch_list
        with row assembly options (`options` is a gigatoken_rs._WrapTruncate
        — prefix/suffix wrapping, max_tokens truncation, and the fused
        forbidden-specials scan)."""
        return self._backend._encode_batch_list_compat(inputs, options, parallel=resolve_parallel(parallel))

    def encode_files(
        self,
        source: FileSource | str | Path | PathLike[str] | list[str | Path | PathLike[str]],
        *,
        parallel: bool | None = None,
    ) -> ak.Array:
        """Encode all documents from files; see the backend's encode_files
        and encode_batch here for `parallel`."""
        return self._backend.encode_files(source, parallel=resolve_parallel(parallel))

    def decode(self, tokens: list[int] | npt.NDArray[np.uint32] | ak.Array) -> bytes:
        if type(tokens).__module__.partition(".")[0] == "awkward":
            # An awkward row converts to a numpy view (zero-copy when
            # contiguous), which the backend borrows directly; iterating it
            # element by element would build a Python int per token. Checked
            # by module rather than a `layout` attribute sniff, which would
            # also match torch.Tensor (whose .layout is the strided/sparse
            # kind) and send e.g. CUDA tensors into np.asarray.
            import numpy as np

            tokens = np.asarray(tokens)
        return self._backend.decode(tokens)

    def __getattr__(self, name: str) -> Any:
        # Backend-specific extras (e.g. SentencePiece's encode_no_normalize).
        if name == "_backend":
            raise AttributeError(name)
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return f"Tokenizer({self._backend!r})"
