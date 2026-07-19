import http.server
import threading

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


@pytest.fixture
def hub_server(gpt2_tokenizer_path):
    """Loopback stand-in for the Hub: `resolve/` answers with x-repo-commit
    and a redirect to a `/cdn/` path serving the gpt2 fixture, like the real
    endpoint does for LFS files. Yields (endpoint, commit, requests) where
    requests collects (path, authorization_header) per request."""
    data = gpt2_tokenizer_path.read_bytes()
    commit = "a" * 40
    requests: list[tuple[str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/openai-community/gpt2/resolve/main/tokenizer_config.json":
                # Non-LFS files answer directly with 200 + x-repo-commit, no
                # CDN redirect. The config-first dispatch fetches this before
                # tokenizer.json; a plain class name routes to the json path.
                body = b'{"tokenizer_class": "GPT2Tokenizer"}'
                self.send_response(200)
                self.send_header("x-repo-commit", commit)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/openai-community/gpt2/resolve/main/tokenizer.json":
                self.send_response(302)
                self.send_header("x-repo-commit", commit)
                self.send_header("Location", "/cdn/tokenizer.json")
                self.end_headers()
            elif self.path == "/cdn/tokenizer.json":
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", commit, requests
    finally:
        server.shutdown()
        server.server_close()


def test_tokenizer_from_repo_id_downloads_into_cache(monkeypatch, tmp_path, hub_server):
    """On a cache miss the files (the tokenizer_config.json dispatch probe,
    then tokenizer.json) are fetched from the endpoint (no huggingface_hub
    involved) and land in the standard HF cache layout under the commit from
    x-repo-commit; the token travels only to the resolve URL, not to the
    redirect target; a reload is served from the cache with no request.
    HF_HOME points at an empty directory so the fast path misses."""
    endpoint, commit, requests = hub_server
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken")

    tokenizer = Tokenizer("openai-community/gpt2")
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == b"Hello, world!"
    assert requests == [
        ("/openai-community/gpt2/resolve/main/tokenizer_config.json", "Bearer hf_testtoken"),
        ("/openai-community/gpt2/resolve/main/tokenizer.json", "Bearer hf_testtoken"),
        ("/cdn/tokenizer.json", None),
    ]

    repo_dir = tmp_path / "hub" / "models--openai-community--gpt2"
    assert (repo_dir / "snapshots" / commit / "tokenizer.json").is_file()
    assert (repo_dir / "refs" / "main").read_text() == commit

    tokenizer = Tokenizer("openai-community/gpt2")
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == b"Hello, world!"
    assert len(requests) == 3, "second load must be served from the cache"


def test_tokenizer_from_repo_id_cache_fast_path(monkeypatch, tmp_path, gpt2_tokenizer_path):
    """A file already in the standard HF cache layout is served by the pure
    filesystem lookup: no network request (the endpoint points at a closed
    port, so any request would error out)."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")  # discard port: nothing listens
    commit = "0" * 40
    repo_dir = tmp_path / "hub" / "models--openai-community--gpt2"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(commit)
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_bytes(gpt2_tokenizer_path.read_bytes())
    (snapshot / "tokenizer_config.json").write_text('{"tokenizer_class": "GPT2Tokenizer"}')

    tokenizer = Tokenizer("openai-community/gpt2")
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
