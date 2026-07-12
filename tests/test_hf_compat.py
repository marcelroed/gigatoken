"""Tests for the unified gigatoken.Tokenizer and the gigatoken.HFCompat wrapper.

gigatoken.Tokenizer loads standard-format tokenizers (paths or already
initialized HuggingFace tokenizer objects) and picks the right Rust backend.
gigatoken.Tokenizer(source).as_hf() adapts it to the `transformers`
fast-tokenizer API so it can replace a HuggingFace tokenizer in existing
code.
"""

import json

import awkward as ak
import numpy as np
import pytest
from tokenizers import Tokenizer as HFTokenizer

import gigatoken
from gigatoken.gigatoken_rs import BPETokenizer, SentencePieceTokenizer

TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "   leading and trailing spaces   ",
    "café résumé naïve",
    "emoji: 😒🌍🎉",
    "def foo(x: int) -> int:\n    return x + 1\n",
    "日本語テスト",
]


@pytest.fixture(scope="module")
def gpt2_hf(gpt2_tokenizer_path):
    return HFTokenizer.from_file(str(gpt2_tokenizer_path))


@pytest.fixture(scope="module")
def tinyllama_hf(tinyllama_tokenizer_path):
    return HFTokenizer.from_file(str(tinyllama_tokenizer_path))


# ---------------------------------------------------------------------------
# gigatoken.Tokenizer: unified loading and backend dispatch
# ---------------------------------------------------------------------------


def test_tokenizer_from_path_dispatches_bpe(gpt2_tokenizer_path, gpt2_hf):
    tok = gigatoken.Tokenizer(gpt2_tokenizer_path)
    assert isinstance(tok.backend, BPETokenizer)
    for text in TEXTS:
        assert tok.encode(text).tolist() == gpt2_hf.encode(text).ids


def test_tokenizer_from_path_dispatches_sentencepiece(tinyllama_tokenizer_path, tinyllama_hf):
    tok = gigatoken.Tokenizer(tinyllama_tokenizer_path)
    assert isinstance(tok.backend, SentencePieceTokenizer)
    for text in TEXTS:
        expected = tinyllama_hf.encode(text, add_special_tokens=False).ids
        assert tok.encode(text).tolist() == expected


def test_tokenizer_from_tokenizers_object(gpt2_hf):
    """Load from an already initialized tokenizers.Tokenizer."""
    tok = gigatoken.Tokenizer(gpt2_hf)
    assert isinstance(tok.backend, BPETokenizer)
    text = "Hello world, this is a test."
    assert tok.encode(text).tolist() == gpt2_hf.encode(text).ids


def test_tokenizer_from_transformers_fast():
    """Load from a transformers fast tokenizer (TokenizersBackend)."""
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert hf.is_fast
    tok = gigatoken.Tokenizer(hf)
    assert isinstance(tok.backend, BPETokenizer)
    text = "Hello world, this is a test."
    assert tok.encode(text).tolist() == hf.encode(text)


def test_tokenizer_from_json_with_legacy_string_merges(gpt2_tokenizer_path, gpt2_hf):
    """Older tokenizer.json files store merges as "a b" strings."""
    with open(gpt2_tokenizer_path) as f:
        config = json.load(f)
    config["model"]["merges"] = [f"{a} {b}" for a, b in config["model"]["merges"]]
    tok = gigatoken.Tokenizer.from_json(json.dumps(config, ensure_ascii=False))
    text = "Hello world, this is a test."
    assert tok.encode(text).tolist() == gpt2_hf.encode(text).ids


def test_tokenizer_from_directory(tmp_path, gpt2_tokenizer_path, gpt2_hf):
    """A directory containing tokenizer.json also works."""
    (tmp_path / "tokenizer.json").write_bytes(gpt2_tokenizer_path.read_bytes())
    tok = gigatoken.Tokenizer(tmp_path)
    assert tok.encode("Hello").tolist() == gpt2_hf.encode("Hello").ids


def test_tokenizer_rejects_unknown_object():
    with pytest.raises(TypeError):
        gigatoken.Tokenizer(object())


# ---------------------------------------------------------------------------
# gigatoken.Tokenizer: single API across both backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["gpt2_tokenizer_path", "tinyllama_tokenizer_path"])
def test_unified_api(fixture, request):
    tok = gigatoken.Tokenizer(request.getfixturevalue(fixture))

    ids = tok.encode(TEXTS[0])
    assert isinstance(ids, np.ndarray) and ids.dtype == np.uint32

    batch = tok.encode_batch(TEXTS)
    assert isinstance(batch, ak.Array)
    assert len(batch) == len(TEXTS)
    for row, text in zip(batch, TEXTS):
        assert ak.to_list(row) == tok.encode(text).tolist()

    # awkward Array input works for both backends too
    batch_ak = tok.encode_batch(ak.Array(TEXTS))
    assert ak.to_list(batch_ak) == ak.to_list(batch)

    decoded = tok.decode(tok.encode(TEXTS[0]))
    assert decoded == TEXTS[0].encode("utf-8")


@pytest.mark.parametrize("fixture", ["gpt2_tokenizer_path", "tinyllama_tokenizer_path"])
def test_unified_encode_files(fixture, request, tmp_path):
    tok = gigatoken.Tokenizer(request.getfixturevalue(fixture))
    path = tmp_path / "doc.txt"
    path.write_text(TEXTS[1])
    result = tok.encode_files(path)
    assert len(result) == 1
    assert ak.to_list(result[0]) == tok.encode(TEXTS[1]).tolist()


# ---------------------------------------------------------------------------
# gigatoken.HFCompat: transformers fast-tokenizer (TokenizersBackend) API
# ---------------------------------------------------------------------------
#
# gpt2_ref is a fully configured transformers tokenizer (named special
# tokens from the hub config); tinyllama_ref is a TokenizersBackend built
# from a bare tokenizer.json, the same source HFCompat sees — together they
# cover both backends and both special-token situations.


@pytest.fixture(scope="module")
def gpt2_ref():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")


@pytest.fixture(scope="module")
def gpt2_compat(gpt2_ref):
    return gigatoken.Tokenizer(gpt2_ref).as_hf()


@pytest.fixture(scope="module")
def tinyllama_ref(tinyllama_hf):
    transformers = pytest.importorskip("transformers")
    return transformers.TokenizersBackend(tokenizer_object=tinyllama_hf)


@pytest.fixture(scope="module")
def tinyllama_compat(tinyllama_tokenizer_path):
    return gigatoken.Tokenizer(tinyllama_tokenizer_path).as_hf()


# The part of the PreTrainedTokenizerFast / TokenizersBackend surface that
# HFCompat implements. Interface parity is asserted against the real class.
CORE_API = [
    "__call__",
    "encode",
    "decode",
    "batch_decode",
    "tokenize",
    "convert_ids_to_tokens",
    "convert_tokens_to_ids",
    "convert_tokens_to_string",
    "get_vocab",
    "get_added_vocab",
    "vocab",
    "vocab_size",
    "__len__",
    "added_tokens_encoder",
    "added_tokens_decoder",
    "num_special_tokens_to_add",
    "special_tokens_map",
    "all_special_tokens",
    "all_special_ids",
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "bos_token_id",
    "eos_token_id",
    "unk_token_id",
    "pad_token_id",
    "model_input_names",
    "is_fast",
    "padding_side",
    "truncation_side",
]


def test_hf_compat_interface(gpt2_ref, gpt2_compat):
    for name in CORE_API:
        assert hasattr(gpt2_ref, name), f"{name} not in reference API"
        assert hasattr(gpt2_compat, name), f"{name} missing from HFCompat"
        if callable(getattr(gpt2_ref, name)):
            assert callable(getattr(gpt2_compat, name)), f"{name} should be callable"


@pytest.mark.parametrize("text", TEXTS, ids=lambda t: repr(t)[:40])
@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_encode_matches(pair, text, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    for add_special_tokens in (True, False):
        assert ours.encode(text, add_special_tokens=add_special_tokens) == ref.encode(
            text, add_special_tokens=add_special_tokens
        )
    assert ours.tokenize(text) == ref.tokenize(text)
    assert dict(ours(text)) == dict(ref(text))


@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_call_batch(pair, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    ref_out = ref(TEXTS)
    our_out = ours(TEXTS)
    assert our_out["input_ids"] == ref_out["input_ids"]
    assert our_out["attention_mask"] == ref_out["attention_mask"]
    # BatchEncoding-style attribute access
    assert our_out.input_ids == our_out["input_ids"]


@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_decode_matches(pair, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    for text in TEXTS:
        ids = ref.encode(text)
        for skip in (False, True):
            assert ours.decode(ids, skip_special_tokens=skip) == ref.decode(ids, skip_special_tokens=skip)
    ids = ref.encode(TEXTS[0])
    assert ours.decode(ids[0]) == ref.decode(ids[0])  # single int
    assert ours.decode(np.array(ids)) == ref.decode(np.array(ids))
    assert ours.decode([ids, ids]) == ref.decode([ids, ids])  # nested -> list[str]
    assert ours.decode([]) == ref.decode([]) == ""
    batch = [ref.encode(t) for t in TEXTS]
    assert ours.batch_decode(batch) == ref.batch_decode(batch)
    assert ours.batch_decode(batch, skip_special_tokens=True) == ref.batch_decode(batch, skip_special_tokens=True)


@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_conversions_match(pair, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    for text in TEXTS:
        tokens = ref.tokenize(text)
        ids = ref.convert_tokens_to_ids(tokens)
        assert ours.convert_tokens_to_ids(tokens) == ids
        assert ours.convert_ids_to_tokens(ids) == ref.convert_ids_to_tokens(ids)
        assert ours.convert_ids_to_tokens(ids[0]) == ref.convert_ids_to_tokens(ids[0])
        assert ours.convert_tokens_to_string(tokens) == ref.convert_tokens_to_string(tokens)
    # unknown tokens: unk fallback when the tokenizer has one, None otherwise
    assert ours.convert_tokens_to_ids("not-a-real-token") == ref.convert_tokens_to_ids("not-a-real-token")
    assert ours.convert_tokens_to_ids(["not-a-real-token"]) == ref.convert_tokens_to_ids(["not-a-real-token"])


def test_hf_compat_convert_tokens_to_string_byte_fallback(tinyllama_ref, tinyllama_compat):
    tokens = ["▁", "<0xF0>", "<0x9F>", "<0x9A>", "<0x80>"]  # ▁ + utf-8 bytes of 🚀
    assert tinyllama_compat.convert_tokens_to_string(tokens) == tinyllama_ref.convert_tokens_to_string(tokens) == "🚀"
    tokens = ["<s>", "▁Hello"]
    assert tinyllama_compat.convert_tokens_to_string(tokens) == tinyllama_ref.convert_tokens_to_string(tokens)


@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_vocab_matches(pair, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    assert ours.vocab_size == ref.vocab_size
    assert len(ours) == len(ref)
    assert ours.get_vocab() == ref.get_vocab()
    assert ours.get_added_vocab() == ref.get_added_vocab()
    assert ours.added_tokens_encoder == ref.added_tokens_encoder
    assert ours.added_tokens_decoder == {k: str(v) for k, v in ref.added_tokens_decoder.items()}


@pytest.mark.parametrize("pair", ["gpt2", "tinyllama"])
def test_hf_compat_special_tokens_match(pair, request):
    ref = request.getfixturevalue(f"{pair}_ref")
    ours = request.getfixturevalue(f"{pair}_compat")
    for attr in ["bos_token", "eos_token", "unk_token", "pad_token"]:
        ref_val = getattr(ref, attr)
        assert getattr(ours, attr) == (str(ref_val) if ref_val is not None else None)
        assert getattr(ours, f"{attr}_id") == getattr(ref, f"{attr}_id")
    assert ours.all_special_tokens == [str(t) for t in ref.all_special_tokens]
    assert ours.all_special_ids == ref.all_special_ids
    assert ours.special_tokens_map == {k: str(v) for k, v in ref.special_tokens_map.items()}
    assert ours.num_special_tokens_to_add() == ref.num_special_tokens_to_add()


def test_hf_compat_template_specials(tinyllama_ref, tinyllama_compat):
    """TinyLlama's TemplateProcessing adds BOS: reproduced, not skipped."""
    assert tinyllama_compat.encode("Hello world") == tinyllama_ref.encode("Hello world")
    assert tinyllama_compat.encode("Hello world")[0] == 1  # <s>
    assert tinyllama_compat.tokenize("Hello world", add_special_tokens=True) == tinyllama_ref.tokenize(
        "Hello world", add_special_tokens=True
    )
    assert tinyllama_compat.num_special_tokens_to_add() == 1


def test_hf_compat_unsupported_features_raise(gpt2_compat):
    with pytest.raises(NotImplementedError):
        gpt2_compat("Hello", padding=True)
    with pytest.raises(NotImplementedError):
        gpt2_compat("Hello", truncation=True)
    with pytest.raises(NotImplementedError):
        gpt2_compat("Hello", return_tensors="pt")
    with pytest.raises(ValueError):
        gpt2_compat.encode("Hello", text_pair="world")


def test_as_hf_accepts_all_source_kinds(gpt2_tokenizer_path, gpt2_hf, gpt2_ref):
    """Path, tokenizers.Tokenizer, and transformers tokenizer all load."""
    text = "Hello world, this is a test."
    expected = gpt2_ref.encode(text)
    for source in [gpt2_tokenizer_path, gpt2_hf, gpt2_ref]:
        compat = gigatoken.Tokenizer(source).as_hf()
        assert compat.encode(text) == expected
        assert compat.decode(expected) == text


def test_hf_compat_requires_gigatoken_tokenizer(gpt2_ref):
    """HFCompat no longer loads sources itself; wrap with Tokenizer first."""
    with pytest.raises(TypeError, match="as_hf"):
        gigatoken.HFCompat(gpt2_ref)


def test_as_hf_unavailable_without_hf_config(r50k_tiktoken_path):
    """A tokenizer loaded from a .tiktoken file has no tokenizer.json to
    reconstruct the HF-side configuration from."""
    tok = gigatoken.Tokenizer.from_tiktoken(r50k_tiktoken_path)
    with pytest.raises(ValueError, match="tokenizer.json"):
        tok.as_hf()
