import sys

import pytest

from gigatoken import Tokenizer
from gigatoken._load.hf import load_hf_tokenizer
from gigatoken._load.hub import get_hf_token, looks_like_repo_id


@pytest.mark.parametrize("name_or_path", ["openai-community/gpt2", "Qwen/Qwen2-1.5B-Instruct"])
def test_load_hf_tokenizer(name_or_path: str):
    tokenizer = load_hf_tokenizer(name_or_path)
    print(tokenizer)
    assert tokenizer is not None


@pytest.mark.parametrize("repo_id", ["openai-community/gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
def test_tokenizer_from_repo_id(repo_id: str):
    tokenizer = Tokenizer(repo_id)
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == b"Hello, world!"


def test_tokenizer_from_repo_id_without_huggingface_hub(monkeypatch, gpt2_tokenizer_path):
    """The Hub fallback must work with huggingface_hub not installed. The
    urllib request is served from the local fixture, so no network is hit."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # makes its import raise

    data = gpt2_tokenizer_path.read_bytes()
    opened = []

    class _FakeResponse:
        def read(self):
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeOpener:
        def open(self, req):
            opened.append(req.full_url)
            return _FakeResponse()

    monkeypatch.setattr("gigatoken._load.hub.urllib.request.build_opener", lambda *handlers: _FakeOpener())
    tokenizer = Tokenizer("openai-community/gpt2")
    assert len(opened) == 1
    assert opened[0].endswith("/openai-community/gpt2/resolve/main/tokenizer.json")
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == b"Hello, world!"


def test_missing_local_path_raises():
    with pytest.raises(FileNotFoundError):
        Tokenizer("no/such/path/tokenizer.json")


def test_looks_like_repo_id():
    assert looks_like_repo_id("gpt2")
    assert looks_like_repo_id("openai-community/gpt2")
    assert looks_like_repo_id("Qwen/Qwen3.5-9B")
    assert not looks_like_repo_id("data/tokenizers/gpt2.json")
    assert not looks_like_repo_id("./gpt2")
    assert not looks_like_repo_id("/abs/path")
    assert not looks_like_repo_id("gpt2_tokenizer.json")
    assert not looks_like_repo_id("subdir/tokenizer.model")


def test_get_hf_token_discovery(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token_file = tmp_path / "token"
    token_file.write_text("hf_filetoken\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))
    assert get_hf_token() == "hf_filetoken"

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    assert get_hf_token() == "hf_legacy"

    monkeypatch.setenv("HF_TOKEN", "hf_envtoken")
    assert get_hf_token() == "hf_envtoken"
