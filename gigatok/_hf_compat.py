"""HFCompat: a gigatok-backed drop-in for the HuggingFace `transformers`
fast-tokenizer API (PreTrainedTokenizerFast / TokenizersBackend)."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload

from gigatok._load.hf import to_tokenizer_json
from gigatok._tokenizer import Tokenizer

if TYPE_CHECKING:
    from gigatok._load.hf import TokenizerJsonSource

_NAMED_SPECIAL_ATTRS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)

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
    """Wrap a HuggingFace tokenizer as a gigatok-accelerated object following
    the `transformers` fast-tokenizer API (TokenizersBackend /
    PreTrainedTokenizerFast), so it can replace the original in existing
    code: `__call__`/`encode`/`decode`/`batch_decode`/`tokenize`/`convert_*`
    plus the vocab and special-token accessors.

    Accepts the same sources as `gigatok.Tokenizer`: a path to a
    tokenizer.json, a HuggingFace Hub repo id, a `tokenizers.Tokenizer`, or a
    `transformers` tokenizer (fast or slow). Named special-token attributes
    (eos_token, ...) are
    copied from the source tokenizer when it has them; otherwise they are
    None, like a TokenizersBackend built from a bare tokenizer_object.

    Padding, truncation, sequence pairs, and return_tensors are not
    supported and raise instead of silently diverging.
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

    def __init__(self, tokenizer: TokenizerJsonSource) -> None:
        data = to_tokenizer_json(tokenizer)
        self._tokenizer = Tokenizer.from_json(data)
        config = json.loads(data)
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

        for attr in _NAMED_SPECIAL_ATTRS:
            token = getattr(tokenizer, attr, None)
            setattr(self, attr, str(token) if token is not None else None)
        extra = getattr(tokenizer, "additional_special_tokens", None) or []
        self.additional_special_tokens = [str(t) for t in extra]

    @property
    def tokenizer(self) -> Tokenizer:
        """The underlying gigatok Tokenizer (numpy/awkward-native API)."""
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
            raise ValueError("gigatok.HFCompat does not support sequence pairs")
        if padding:
            raise NotImplementedError("gigatok.HFCompat does not support padding")
        if truncation or max_length is not None:
            raise NotImplementedError("gigatok.HFCompat does not support truncation")
        if return_tensors is not None:
            raise NotImplementedError("gigatok.HFCompat does not support return_tensors")
        if kwargs.get("is_split_into_words"):
            raise NotImplementedError("gigatok.HFCompat does not support pre-tokenized input")

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
        if isinstance(text, str):
            ids = self._encode_ids(text, add_special_tokens)
            out = BatchEncoding(input_ids=ids)
            if with_mask:
                out["attention_mask"] = [1] * len(ids)
            return out
        import awkward as ak

        rows = ak.to_list(self._tokenizer.encode_batch(list(text)))
        if add_special_tokens:
            self._check_post_processor()
            if self._prefix_ids or self._suffix_ids:
                rows = [self._prefix_ids + row + self._suffix_ids for row in rows]
        out = BatchEncoding(input_ids=rows)
        if with_mask:
            out["attention_mask"] = [[1] * len(row) for row in rows]
        return out

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
        self._check_call_args(text_pair, padding, truncation, max_length, None, kwargs)
        return self._encode_ids(text, add_special_tokens)

    def tokenize(self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs: Any) -> list[str | None]:
        if pair is not None:
            raise ValueError("gigatok.HFCompat does not support sequence pairs")
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
            raise ValueError("gigatok.HFCompat does not support sequence pairs")
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
