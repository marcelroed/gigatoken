"""Unit tests for the tokenizer_config.json-driven load dispatch.

End-to-end loading of a registered line (repo id, directory, and vocab-file
path) is covered by test_kimi.py; here the dispatch decisions themselves.
"""

import json

import pytest

from gigatoken._load.hf import TOKENIZER_LINES, _match_tokenizer_line, try_load_from_config

KIMI_CONFIG = {
    "tokenizer_class": "TikTokenTokenizer",
    "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
}


def test_matches_registered_line():
    line = _match_tokenizer_line(KIMI_CONFIG)
    assert line is not None
    assert line.vocab_file == "tiktoken.model"
    assert line.pretokenizer == "kimi"


def test_matches_all_registered_auto_map_modules():
    for line in TOKENIZER_LINES:
        for module in line.auto_map_modules:
            config = {
                "tokenizer_class": line.tokenizer_class,
                "auto_map": {"AutoTokenizer": [f"{module}.{line.tokenizer_class}", None]},
            }
            assert _match_tokenizer_line(config) == line


def test_matches_by_class_without_auto_map():
    assert _match_tokenizer_line({"tokenizer_class": "TikTokenTokenizer"}) is not None


def test_rejects_class_name_collision():
    """Another org's remote-code class reusing the name TikTokenTokenizer
    (different module, so possibly a different regex) must not match."""
    config = {
        "tokenizer_class": "TikTokenTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_other.TikTokenTokenizer", None]},
    }
    assert _match_tokenizer_line(config) is None


def test_rejects_unregistered_class():
    assert _match_tokenizer_line({"tokenizer_class": "PreTrainedTokenizerFast"}) is None
    assert _match_tokenizer_line({}) is None


def test_auto_map_string_form():
    """auto_map values may be a plain string instead of a [fast, slow] pair."""
    config = {
        "tokenizer_class": "TikTokenTokenizer",
        "auto_map": {"AutoTokenizer": "tokenization_kimi.TikTokenTokenizer"},
    }
    assert _match_tokenizer_line(config) is not None


def test_dispatch_returns_none_for_unregistered_dir(tmp_path):
    """A directory whose config identifies no registered line falls through
    to the tokenizer.json path (None), without touching any vocab file."""
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    assert try_load_from_config(tmp_path) is None


def test_dispatch_returns_none_without_config(tmp_path):
    assert try_load_from_config(tmp_path) is None
    assert try_load_from_config(tmp_path / "missing") is None


def test_unknown_scheme_rejected():
    from gigatoken.gigatoken_rs import BPETokenizer

    with pytest.raises(ValueError, match="unknown pretokenizer scheme"):
        BPETokenizer.from_tiktoken_model("x", "y", "not-a-scheme")
