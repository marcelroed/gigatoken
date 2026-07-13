"""HFCompat: a gigatoken-backed drop-in for the HuggingFace `transformers`
fast-tokenizer API (PreTrainedTokenizerFast / TokenizersBackend)."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload

from gigatoken._load.hf import NAMED_SPECIAL_TOKEN_ATTRS as _NAMED_SPECIAL_ATTRS
from gigatoken._tokenizer import Tokenizer

if TYPE_CHECKING:
    import awkward as ak
    import numpy.typing as npt

_SENTENCEPIECE_SPACE = "▁"


@functools.cache
def _gpt2_unicode_to_byte() -> dict[str, int]:
    """The GPT-2 ByteLevel char -> byte table (inverse of bytes_to_unicode)."""
    allowed = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    b2u = {b: chr(b) for b in allowed}
    n = 0
    for b in range(256):
        if b not in b2u:
            b2u[b] = chr(256 + n)
            n += 1
    return {ch: b for b, ch in b2u.items()}


def _template_special_ids(pp: dict[str, Any] | None) -> tuple[list[int], list[int]]:
    """Resolve a post_processor into (prefix_ids, suffix_ids) added around a
    single sequence, e.g. Llama's TemplateProcessing BOS. Raises for
    post-processors whose effect on ids cannot be reproduced."""
    if not pp:
        return [], []
    kind = pp.get("type")
    if kind == "ByteLevel":
        # only sets offsets/trim behavior; adds no tokens
        return [], []
    if kind == "Sequence":
        prefix: list[int] = []
        suffix: list[int] = []
        for sub in pp.get("processors", []):
            pre, suf = _template_special_ids(sub)
            prefix = pre + prefix
            suffix = suffix + suf
        return prefix, suffix
    if kind == "TemplateProcessing":
        specials = pp.get("special_tokens", {})
        prefix, suffix = [], []
        seen_sequence = False
        for item in pp.get("single", []):
            if "Sequence" in item:
                seen_sequence = True
            elif "SpecialToken" in item:
                ids = specials[item["SpecialToken"]["id"]]["ids"]
                (suffix if seen_sequence else prefix).extend(ids)
            else:
                raise ValueError(f"unsupported TemplateProcessing item: {item}")
        return prefix, suffix
    raise ValueError(f"unsupported post_processor type: {kind}")


class BatchEncoding(dict):
    """Minimal stand-in for transformers.BatchEncoding: a dict whose keys
    (input_ids, attention_mask) are also attributes."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None


class HFCompat:
    """Adapt a `gigatoken.Tokenizer` to the `transformers` fast-tokenizer API
    (TokenizersBackend / PreTrainedTokenizerFast), so it can replace the
    original tokenizer in existing code:
    `__call__`/`encode`/`decode`/`batch_decode`/`tokenize`/`convert_*`
    plus the vocab and special-token accessors.

    Obtain one with `gigatoken.Tokenizer(source).as_hf()`, where `source` is
    anything `gigatoken.Tokenizer` accepts — for a drop-in replacement of an
    existing HuggingFace tokenizer:
    `hf_tokenizer_compatible = gigatoken.Tokenizer(hf_tokenizer).as_hf()`.
    Named special-token attributes (eos_token, ...) are copied from the
    source tokenizer when it has them; otherwise they are None, like a
    TokenizersBackend built from a bare tokenizer_object.

    `return_tensors="np"` and `"pt"` are supported and hand the backend's
    buffers straight to numpy/torch instead of building Python lists:
    input_ids come back as uint32 ("np") or int32 ("pt", a zero-copy
    bit-cast that torch accepts as an index dtype), not transformers' int64.
    Padding and truncation are supported too — padded batches are assembled
    into their final matrix inside the Rust backend in a single pass — with
    one deviation: truncation requires an explicit max_length, since there
    is no model_max_length to fall back to. Sequence pairs, pre-tokenized
    input, and other return_tensors values raise instead of silently
    diverging.
    """

    model_input_names: list[str] = ["input_ids", "attention_mask"]
    is_fast: bool = True
    padding_side: str = "right"
    truncation_side: str = "right"

    # Set in __init__ from the source tokenizer (see _NAMED_SPECIAL_ATTRS).
    bos_token: str | None
    eos_token: str | None
    unk_token: str | None
    sep_token: str | None
    pad_token: str | None
    cls_token: str | None
    mask_token: str | None
    additional_special_tokens: list[str]

    # Derived properties attached at module bottom via _named_special_id_property.
    bos_token_id: int | None
    eos_token_id: int | None
    unk_token_id: int | None
    sep_token_id: int | None
    pad_token_id: int | None
    cls_token_id: int | None
    mask_token_id: int | None

    def __init__(self, tokenizer: Tokenizer) -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"HFCompat wraps a gigatoken.Tokenizer, not {type(tokenizer).__name__!r}; "
                "construct one first, e.g. gigatoken.Tokenizer(hf_tokenizer).as_hf()"
            )
        self._tokenizer = tokenizer
        config = tokenizer._hf_config()
        added = config.get("added_tokens") or []
        self._model_vocab: dict[str, int] = {str(tok): int(i) for tok, i in config["model"]["vocab"].items()}
        self._added_tokens: dict[str, int] = {str(t["content"]): int(t["id"]) for t in added}
        self._special_ids = frozenset(int(t["id"]) for t in added if t.get("special"))
        self._id_to_token: dict[int, str] = {i: tok for tok, i in self._model_vocab.items()}
        self._id_to_token.update((i, tok) for tok, i in self._added_tokens.items())
        self._vocab_len = len(self._model_vocab.keys() | self._added_tokens.keys())
        self._byte_fallback = bool(config["model"].get("byte_fallback"))
        try:
            self._prefix_ids, self._suffix_ids = _template_special_ids(config.get("post_processor"))
            self._post_processor_error = None
        except ValueError as e:
            # Only encoding with add_special_tokens=True needs the
            # post-processor; defer the failure until then.
            self._prefix_ids, self._suffix_ids = [], []
            self._post_processor_error = str(e)

        named = tokenizer._named_specials
        for attr in _NAMED_SPECIAL_ATTRS:
            token = named.get(attr)
            setattr(self, attr, str(token) if token is not None else None)
        extra = named.get("additional_special_tokens") or []
        self.additional_special_tokens = [str(t) for t in extra]

    @property
    def tokenizer(self) -> Tokenizer:
        """The underlying gigatoken Tokenizer (numpy/awkward-native API)."""
        return self._tokenizer

    # -- encoding -----------------------------------------------------------

    def _check_call_args(
        self,
        text_pair: str | None,
        padding: bool | str,
        truncation: bool | str | None,
        max_length: int | None,
        return_tensors: str | None,
        kwargs: dict[str, Any],
    ) -> None:
        if text_pair is not None:
            raise ValueError("gigatoken.HFCompat does not support sequence pairs")
        if padding not in (False, True, "do_not_pad", "longest", "max_length"):
            raise NotImplementedError(f"unsupported padding strategy {padding!r}")
        if truncation in ("only_first", "only_second"):
            raise ValueError("gigatoken.HFCompat does not support sequence pairs")
        if truncation not in (None, False, True, "do_not_truncate", "longest_first"):
            raise NotImplementedError(f"unsupported truncation strategy {truncation!r}")
        truncate = truncation in (True, "longest_first")
        if truncate and max_length is None:
            raise ValueError("truncation requires an explicit max_length; gigatoken.HFCompat has no model_max_length to fall back to")
        if padding == "max_length" and max_length is None:
            raise ValueError('padding="max_length" requires max_length')
        if max_length is not None and not truncate and padding != "max_length":
            raise ValueError('max_length has no effect without truncation or padding="max_length"')
        if return_tensors is not None and return_tensors not in ("np", "pt"):
            raise NotImplementedError(f'gigatoken.HFCompat supports return_tensors="np" and "pt" only, got {return_tensors!r}')
        if kwargs.get("is_split_into_words"):
            raise NotImplementedError("gigatoken.HFCompat does not support pre-tokenized input")
        if self.padding_side not in ("right", "left"):
            raise ValueError(f'padding_side must be "right" or "left", got {self.padding_side!r}')
        if self.truncation_side not in ("right", "left"):
            raise ValueError(f'truncation_side must be "right" or "left", got {self.truncation_side!r}')

    def _check_post_processor(self) -> None:
        if self._post_processor_error is not None:
            raise ValueError(f"cannot add special tokens: {self._post_processor_error}; pass add_special_tokens=False and add them yourself if needed")

    def _encode_ids(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = self._tokenizer.encode(text).tolist()
        if add_special_tokens:
            self._check_post_processor()
            if self._prefix_ids or self._suffix_ids:
                ids = self._prefix_ids + ids + self._suffix_ids
        return ids

    def _specials(self, add_special_tokens: bool) -> tuple[list[int], list[int]]:
        """(prefix_ids, suffix_ids) the post-processor would add, or empty."""
        if not add_special_tokens:
            return [], []
        self._check_post_processor()
        return self._prefix_ids, self._suffix_ids

    def _content_cap(self, max_length: int, prefix: list[int], suffix: list[int]) -> int:
        """Tokens each sequence may keep once the specials claim their share
        of max_length (matching how HF post-processors count)."""
        cap = max_length - len(prefix) - len(suffix)
        if cap < 0:
            raise ValueError(f"max_length={max_length} leaves no room for the {len(prefix) + len(suffix)} special tokens added per sequence")
        return cap

    def _truncate_rows(self, rows: ak.Array, cap: int) -> ak.Array:
        if self.truncation_side == "left":
            return rows[:, :0] if cap == 0 else rows[:, -cap:]
        return rows[:, :cap]

    def __call__(
        self,
        text: str | Iterable[str] | None = None,
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        self._check_call_args(text_pair, padding, truncation, max_length, return_tensors, kwargs)
        if text is None:
            raise ValueError("text must be provided")
        with_mask = return_attention_mask is None or return_attention_mask
        truncate = truncation in (True, "longest_first")
        if padding in (True, "longest", "max_length"):
            return self._padded_encoding(
                text,
                add_special_tokens,
                pad_to_max_length=padding == "max_length",
                truncate=truncate,
                max_length=max_length,
                return_tensors=return_tensors,
                with_mask=with_mask,
            )
        prefix, suffix = self._specials(add_special_tokens)
        cap = self._content_cap(max_length, prefix, suffix) if truncate else None
        if return_tensors is not None:
            return self._tensor_encoding(text, prefix, suffix, cap, return_tensors, with_mask)
        if isinstance(text, str):
            ids = self._tokenizer.encode(text)
            if cap is not None and len(ids) > cap:
                ids = ids[len(ids) - cap :] if self.truncation_side == "left" else ids[:cap]
            ids = prefix + ids.tolist() + suffix
            out = BatchEncoding(input_ids=ids)
            if with_mask:
                out["attention_mask"] = [1] * len(ids)
            return out
        import awkward as ak

        rows = self._tokenizer.encode_batch(list(text))
        if cap is not None:
            rows = self._truncate_rows(rows, cap)
        rows = ak.to_list(rows)
        if prefix or suffix:
            rows = [prefix + row + suffix for row in rows]
        out = BatchEncoding(input_ids=rows)
        if with_mask:
            out["attention_mask"] = [[1] * len(row) for row in rows]
        return out

    def _padded_encoding(
        self,
        text: str | Iterable[str],
        add_special_tokens: bool,
        pad_to_max_length: bool,
        truncate: bool,
        max_length: int | None,
        return_tensors: str | None,
        with_mask: bool,
    ) -> BatchEncoding:
        """__call__ with padding: the Rust backend encodes the batch and
        assembles the padded (rows x width) matrix in one parallel pass."""
        prefix, suffix = self._specials(add_special_tokens)
        pad_id = self.pad_token_id
        if pad_id is None:
            raise ValueError("asking to pad but the tokenizer has no pad token; set pad_token (e.g. tok.pad_token = tok.eos_token) first")
        single = isinstance(text, str)
        matrix, lengths = self._tokenizer.encode_batch_padded(
            [text] if single else list(text),
            pad_id=pad_id,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            truncate=truncate,
            pad_left=self.padding_side == "left",
            truncate_left=self.truncation_side == "left",
            prefix=prefix,
            suffix=suffix,
        )
        return self._matrix_encoding(matrix, lengths, single, return_tensors, with_mask)

    def _tensor_encoding(
        self,
        text: str | Iterable[str],
        prefix: list[int],
        suffix: list[int],
        cap: int | None,
        return_tensors: str,
        with_mask: bool,
    ) -> BatchEncoding:
        """__call__ with return_tensors="np"/"pt" and no padding: a
        (batch, len) tensor built on the backend's own buffers, with no
        per-token Python iteration."""
        import numpy as np

        if isinstance(text, str):
            ids = self._tokenizer.encode(text)
            if cap is not None and len(ids) > cap:
                ids = ids[len(ids) - cap :] if self.truncation_side == "left" else ids[:cap]
            if prefix or suffix:
                ids = np.concatenate([np.asarray(prefix, dtype=ids.dtype), ids, np.asarray(suffix, dtype=ids.dtype)])
            matrix = ids.reshape(1, -1)
        else:
            matrix = self._batch_matrix(list(text), prefix, suffix, cap)
        return self._matrix_encoding(matrix, None, False, return_tensors, with_mask)

    def _matrix_encoding(
        self,
        matrix: npt.NDArray[Any],
        lengths: npt.NDArray[Any] | None,
        single: bool,
        return_tensors: str | None,
        with_mask: bool,
    ) -> BatchEncoding:
        """Wrap a (rows x width) id matrix as a BatchEncoding; `lengths` gives
        each row's real (unpadded) length, or None when nothing is padded."""
        import numpy as np

        mask = None
        if with_mask:
            if lengths is None:
                mask = np.ones(matrix.shape, dtype=np.int64)
            else:
                positions = np.arange(matrix.shape[1], dtype=np.int64)
                if self.padding_side == "left":
                    mask = (positions >= matrix.shape[1] - lengths[:, None]).astype(np.int64)
                else:
                    mask = (positions < lengths[:, None]).astype(np.int64)
        out = BatchEncoding()
        if return_tensors is None:
            out["input_ids"] = matrix[0].tolist() if single else matrix.tolist()
            if mask is not None:
                out["attention_mask"] = mask[0].tolist() if single else mask.tolist()
            return out
        if return_tensors == "np":
            out["input_ids"] = matrix
            if mask is not None:
                out["attention_mask"] = mask
            return out
        import torch

        # Bit-cast uint32 -> int32: the numpy view is free (same item size,
        # ids never reach 2^31), and torch accepts int32 — but not uint32 —
        # as an index dtype.
        if matrix.dtype == np.uint32:
            matrix = matrix.view(np.int32)
        out["input_ids"] = torch.from_numpy(matrix)
        if mask is not None:
            out["attention_mask"] = torch.from_numpy(mask)
        return out

    def _batch_matrix(self, texts: list[str], prefix: list[int], suffix: list[int], cap: int | None) -> npt.NDArray[Any]:
        """Encode a batch and reshape the flat awkward buffer to (batch, len)
        without copying; specials are broadcast in as whole columns."""
        import awkward as ak
        import numpy as np

        rows = self._tokenizer.encode_batch(texts)
        if cap is not None:
            rows = self._truncate_rows(rows, cap)
        counts = np.asarray(ak.num(rows))
        if counts.size == 0:
            return np.empty((0, 0), dtype=np.uint32)
        width = int(counts[0])
        if (counts != width).any():
            raise ValueError(
                "Unable to create tensor: the encoded sequences have different lengths; "
                "pass padding=True (or encode without return_tensors) instead"
            )
        matrix = ak.to_numpy(ak.flatten(rows)).reshape(counts.size, width)
        if prefix or suffix:
            parts = [matrix]
            if prefix:
                parts.insert(0, np.broadcast_to(np.asarray(prefix, dtype=matrix.dtype), (matrix.shape[0], len(prefix))))
            if suffix:
                parts.append(np.broadcast_to(np.asarray(suffix, dtype=matrix.dtype), (matrix.shape[0], len(suffix))))
            matrix = np.hstack(parts)
        return matrix

    def encode(
        self,
        text: str,
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if not isinstance(text, str):
            raise NotImplementedError("gigatoken.HFCompat.encode takes a single string; it does not support pre-tokenized input")
        out = self(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=False,
            **kwargs,
        )
        return out["input_ids"]

    def tokenize(self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs: Any) -> list[str | None]:
        if pair is not None:
            raise ValueError("gigatoken.HFCompat does not support sequence pairs")
        return self.convert_ids_to_tokens(self._encode_ids(text, add_special_tokens))

    # -- decoding -----------------------------------------------------------

    @overload
    def decode(self, token_ids: int | Iterable[int], skip_special_tokens: bool = False, **kwargs: Any) -> str: ...
    @overload
    def decode(self, token_ids: Iterable[Iterable[int]], skip_special_tokens: bool = False, **kwargs: Any) -> list[str]: ...
    def decode(
        self,
        token_ids: int | Iterable[int] | Iterable[Iterable[int]],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str | list[str]:
        tolist = getattr(token_ids, "tolist", None)
        if callable(tolist):
            token_ids = tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        seq: list[Any] = list(token_ids)
        if seq and isinstance(seq[0], list):
            return self.batch_decode(seq, skip_special_tokens=skip_special_tokens, **kwargs)
        ids = [int(i) for i in seq]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        return self._tokenizer.decode(ids).decode("utf-8", errors="replace")

    def batch_decode(self, sequences: Iterable[Iterable[int]], skip_special_tokens: bool = False, **kwargs: Any) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in sequences]

    # -- token/id conversions ----------------------------------------------

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str | None: ...
    @overload
    def convert_ids_to_tokens(self, ids: Iterable[int], skip_special_tokens: bool = False) -> list[str | None]: ...
    def convert_ids_to_tokens(self, ids: int | Iterable[int], skip_special_tokens: bool = False) -> str | None | list[str | None]:
        if isinstance(ids, int):
            return self._id_to_token.get(ids)
        if skip_special_tokens:
            ids = [i for i in ids if int(i) not in self._special_ids]
        return [self._id_to_token.get(int(i)) for i in ids]

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int | None: ...
    @overload
    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> list[int | None]: ...
    def convert_tokens_to_ids(self, tokens: str | Iterable[str]) -> int | None | list[int | None]:
        if isinstance(tokens, str):
            return self._token_to_id(tokens)
        return [self._token_to_id(t) for t in tokens]

    def _token_to_id(self, token: str) -> int | None:
        id_ = self._token_to_id_no_unk(token)
        if id_ is None and self.unk_token is not None:
            id_ = self._token_to_id_no_unk(self.unk_token)
        return id_

    def _token_to_id_no_unk(self, token: str) -> int | None:
        id_ = self._added_tokens.get(token)
        return id_ if id_ is not None else self._model_vocab.get(token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Mirror the backend decoder: ByteLevel unicode->byte remapping for
        byte-level BPE, metaspace/byte-fallback handling for SentencePiece."""
        if self._byte_fallback:
            raw = bytearray()
            for token in tokens:
                if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                    raw.append(int(token[3:5], 16))
                else:
                    raw += token.replace(_SENTENCEPIECE_SPACE, " ").encode("utf-8")
            text = raw.decode("utf-8", errors="replace")
            return text[1:] if text.startswith(" ") else text
        u2b = _gpt2_unicode_to_byte()
        raw = bytearray()
        for token in tokens:
            for ch in token:
                b = u2b.get(ch)
                if b is None:
                    raw += ch.encode("utf-8")
                else:
                    raw.append(b)
        return raw.decode("utf-8", errors="replace")

    # -- vocab and special tokens --------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Size of the base vocabulary (without added tokens)."""
        return len(self._model_vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    def get_vocab(self) -> dict[str, int]:
        return {**self._model_vocab, **self._added_tokens}

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_tokens)

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        return dict(self._added_tokens)

    @property
    def added_tokens_decoder(self) -> dict[int, str]:
        return {i: tok for tok, i in self._added_tokens.items()}

    def __len__(self) -> int:
        return self._vocab_len

    @property
    def special_tokens_map(self) -> dict[str, str | list[str]]:
        out = {attr: getattr(self, attr) for attr in _NAMED_SPECIAL_ATTRS if getattr(self, attr) is not None}
        if self.additional_special_tokens:
            out["additional_special_tokens"] = list(self.additional_special_tokens)
        return out

    @property
    def all_special_tokens(self) -> list[str]:
        seen = dict.fromkeys([getattr(self, attr) for attr in _NAMED_SPECIAL_ATTRS if getattr(self, attr) is not None] + list(self.additional_special_tokens))
        return list(seen)

    @property
    def all_special_ids(self) -> list[int | None]:
        return [self._token_to_id_no_unk(t) for t in self.all_special_tokens]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise ValueError("gigatoken.HFCompat does not support sequence pairs")
        self._check_post_processor()
        return len(self._prefix_ids) + len(self._suffix_ids)

    def __repr__(self) -> str:
        return f"HFCompat({self._tokenizer!r})"


def _named_special_id_property(attr: str) -> property:
    def get(self: HFCompat) -> int | None:
        token = getattr(self, attr)
        return None if token is None else self._token_to_id_no_unk(token)

    get.__name__ = attr + "_id"
    return property(get)


# bos_token_id, eos_token_id, unk_token_id, sep_token_id, pad_token_id,
# cls_token_id, mask_token_id — same derived accessors as transformers.
for _attr in _NAMED_SPECIAL_ATTRS:
    setattr(HFCompat, _attr + "_id", _named_special_id_property(_attr))
del _attr
