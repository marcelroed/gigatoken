"""Loading tokenizer configurations from HuggingFace sources.

Nothing here imports `transformers` or `tokenizers` at module level; those
packages are only touched when the caller hands us one of their objects (in
which case they are necessarily already installed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, cast

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


class TokenizerLine(NamedTuple):
    """A tokenizer line whose repos ship no tokenizer.json, identified by the
    tokenizer_config.json `tokenizer_class` and remote-code `auto_map` module.
    The split regex of such a line lives only in its remote code — no shipped
    file declares it — so each entry names the matching pretokenizer scheme
    implemented in Rust (`PretokenizerType`)."""

    tokenizer_class: str
    auto_map_modules: tuple[str, ...]
    vocab_file: str
    pretokenizer: str


# The registered lines. Add an entry (and, if needed, a Rust scheme) to
# support a new non-tokenizer.json tokenizer family.
TOKENIZER_LINES = (
    # moonshotai Kimi/Moonlight (K2/K2.5/K2.6/K2.7, Linear, VL, Moonlight):
    # one shared rank file, per-repo specials, Kimi split regex.
    TokenizerLine(
        tokenizer_class="TikTokenTokenizer",
        auto_map_modules=("tokenization_kimi", "tokenization_moonshot"),
        vocab_file="tiktoken.model",
        pretokenizer="kimi",
    ),
)


def _match_tokenizer_line(config: dict[str, object]) -> TokenizerLine | None:
    """The registered line a tokenizer_config.json identifies, or None.

    Matches on `tokenizer_class`, and when the config carries a remote-code
    `auto_map`, also on its module name — the class name alone is just the
    remote class's chosen name, which another org could reuse for a
    different regex."""
    auto_map = config.get("auto_map")
    entry = auto_map.get("AutoTokenizer") if isinstance(auto_map, dict) else None
    if isinstance(entry, (list, tuple)):
        entry = next((e for e in entry if e), None)
    module = str(entry).split(".")[0] if entry else None
    for line in TOKENIZER_LINES:
        if config.get("tokenizer_class") != line.tokenizer_class:
            continue
        if module is not None and module not in line.auto_map_modules:
            continue
        return line
    return None


def try_load_from_config(source: str | os.PathLike[str]) -> "tuple[object, dict[str, int]] | None":
    """Config-driven dispatch: resolve `source`'s tokenizer_config.json and,
    when it identifies a registered [`TokenizerLine`], load that line's vocab
    file with its pretokenizer scheme and the config's special tokens.

    Accepts a Hub repo id, a directory, or a path to a registered vocab file
    itself (e.g. a tiktoken.model, with the config as its sibling). Returns
    `(BPETokenizer, special_tokens)`, or None when the config is absent or
    identifies no registered line — then the tokenizer.json path applies."""
    import json

    from gigatoken.gigatoken_rs import BPETokenizer, hub_file

    path = Path(cast("str | os.PathLike[str]", source))
    is_repo = False
    if path.is_file() and any(path.name == line.vocab_file for line in TOKENIZER_LINES):
        config = path.parent / "tokenizer_config.json"
    elif path.is_dir():
        config = path / "tokenizer_config.json"
    elif isinstance(source, str) and looks_like_repo_id(source):
        is_repo = True
        try:
            config = hub_file(source, "tokenizer_config.json")
        except Exception:
            # No config in the repo (or it is unreachable): nothing to
            # dispatch on; let the tokenizer.json path report the failure.
            return None
    else:
        return None
    if not config.is_file():
        return None
    config_json = json.loads(config.read_bytes())
    line = _match_tokenizer_line(config_json)
    if line is None:
        return None
    if is_repo:
        model = hub_file(cast(str, source), line.vocab_file)
    elif path.is_dir():
        model = path / line.vocab_file
    elif path.name == line.vocab_file:
        model = path
    else:
        # A vocab-file path of one line whose sibling config identifies
        # another: trust the config and load its line's vocab file.
        model = path.parent / line.vocab_file
    if not model.is_file():
        return None
    specials = {str(t["content"]): int(i) for i, t in (config_json.get("added_tokens_decoder") or {}).items()}
    return BPETokenizer.from_tiktoken_model(model, config, line.pretokenizer), specials


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
