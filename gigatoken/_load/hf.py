"""Loading tokenizer configurations from HuggingFace sources.

Nothing here imports `transformers` or `tokenizers` at module level; those
packages are only touched when the caller hands us one of their objects (in
which case they are necessarily already installed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

from gigatoken._load.hub import download_hub_file, looks_like_repo_id

if TYPE_CHECKING:
    import tokenizers
    import transformers

HFTokenizerLike: TypeAlias = "tokenizers.Tokenizer | transformers.PreTrainedTokenizerBase"
TokenizerJsonSource: TypeAlias = "str | os.PathLike[str] | HFTokenizerLike"

NAMED_SPECIAL_TOKEN_ATTRS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)


def capture_named_special_tokens(source: object) -> dict[str, str | list[str]]:
    """Copy the named special-token attributes (bos_token, eos_token, ...,
    additional_special_tokens) off a `transformers` tokenizer. Sources that
    don't carry them (paths, bare `tokenizers.Tokenizer`s) yield an empty
    dict, like a TokenizersBackend built from a bare tokenizer_object."""
    out: dict[str, str | list[str]] = {}
    for attr in NAMED_SPECIAL_TOKEN_ATTRS:
        token = getattr(source, attr, None)
        if token is not None:
            out[attr] = str(token)
    extra = getattr(source, "additional_special_tokens", None) or []
    if extra:
        out["additional_special_tokens"] = [str(t) for t in extra]
    return out


def load_hf_tokenizer(pretrained_model_name_or_path: str) -> transformers.PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    return cast("transformers.PreTrainedTokenizerBase", tokenizer)


def to_tokenizer_json(source: TokenizerJsonSource) -> str | bytes:
    """Resolve `source` to the contents of a HuggingFace tokenizer.json.

    Accepts a path to a tokenizer.json file (or a directory containing one),
    a HuggingFace Hub repo id like "openai-community/gpt2" (downloaded with
    the standard HF token discovery; huggingface_hub, tokenizers, and
    transformers are not required), a `tokenizers.Tokenizer`, or a
    `transformers` tokenizer — fast ones (TokenizersBackend) through their
    backend, slow ones by converting with `transformers.convert_slow_tokenizer`.
    """
    if isinstance(source, (str, os.PathLike)):
        path = Path(cast("str | os.PathLike[str]", source))
        if path.is_dir():
            path = path / "tokenizer.json"
        if path.is_file():
            # Suffix dispatch; keep gigatoken._load.hub.TOKENIZER_FILE_SUFFIXES
            # in sync so these names are never mistaken for Hub repo ids.
            if path.suffix == ".model":
                # A raw sentencepiece model rather than a tokenizer.json.
                from gigatoken._load.sentencepiece import sentencepiece_to_tokenizer_json

                return sentencepiece_to_tokenizer_json(path.read_bytes())
            return path.read_bytes()
        if isinstance(source, str) and looks_like_repo_id(source):
            return download_hub_file(source)
        raise FileNotFoundError(f"no file or directory at {path}, and {str(source)!r} does not look like a HuggingFace Hub repo id")

    root_module = type(source).__module__.split(".")[0]

    # tokenizers.Tokenizer (or anything else that serializes itself the same way)
    to_str = getattr(source, "to_str", None)
    if callable(to_str) and root_module == "tokenizers":
        return to_str()

    # transformers fast tokenizer: backed by a tokenizers.Tokenizer
    backend = getattr(source, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        return backend.to_str()

    # transformers slow tokenizer: convert to a tokenizers.Tokenizer first
    if root_module == "transformers":
        from transformers.convert_slow_tokenizer import convert_slow_tokenizer

        return convert_slow_tokenizer(source).to_str()

    raise TypeError(f"cannot extract a tokenizer.json from {type(source).__name__!r}; expected a path, a tokenizers.Tokenizer, or a transformers tokenizer")
