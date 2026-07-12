"""SentencePiece backend correctness.

Two references:
- HuggingFace `tokenizers` for tokenizer.json loading (normalizer sequence,
  Metaspace pre-tokenizer, added-token lstrip/rstrip/normalized semantics).
- Raw `sentencepiece` for `Tokenizer.from_sentencepiece`, via golden token
  IDs captured from `SentencePieceProcessor.encode` (sentencepiece itself is
  not a test dependency).
"""

import itertools

import pytest
from tokenizers import AddedToken, Regex
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers.models import BPE

from gigatoken import Tokenizer

TEXTS = [
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "double  space",
    "many     spaces between",
    "\ttab\tseparated\t",
    "line\nbreaks\nhere\n",
    "",
    " ",
    "   ",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers 1234567890 and digits 3.14159",
    "Unicode: café naïve résumé",
    "CJK: 日本語のテキスト 中文文本 한국어",
    "Emoji: 🎉🚀 👩‍👩‍👧‍👦",
    "NFKC: ﬁligature ½ ㎒ Ⅷ ｆｕｌｌｗｉｄｔｈ",
    "Mixed   whitespace \t\n mess",
    "null byte \x00 inside",
    "'quotes' \"double\" `backticks`",
    "http://example.com/path?query=1&other=2",
    "def f(x):\n    return x * 2\n",
    " " * 20 + "indented",
    "word nbsp",
    "zero​width",
    "<s>special</s> <unk> markup",
    "<s> after bos",
    "ends with special <s>",
    "▁literal metaspace in input",
    "a" * 300,
]


def _assert_parity(hf_tok: HFTokenizer, gigatoken_tok: Tokenizer, texts=TEXTS):
    for text in texts:
        hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
        gigatoken_ids = gigatoken_tok.encode(text).tolist()
        assert gigatoken_ids == hf_ids, f"Mismatch for {text!r}:\n  HF:      {hf_ids}\n  gigatoken: {gigatoken_ids}"


# ---------------------------------------------------------------------------
# tokenizer.json parity with HF tokenizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_fixture",
    ["tinyllama_tokenizer_path", "phi3_tokenizer_path", "llama_legacy_tokenizer_path"],
)
def test_hf_json_parity(path_fixture, request):
    """Llama-style jsons: Prepend/Replace normalizer, plus Phi-3's rstrip'd
    added tokens and the legacy Llama normalized=true added tokens."""
    path = request.getfixturevalue(path_fixture)
    hf_tok = HFTokenizer.from_file(str(path))
    gigatoken_tok = Tokenizer(path)
    _assert_parity(hf_tok, gigatoken_tok)


def test_hf_json_decode_parity(tinyllama_tokenizer_path):
    hf_tok = HFTokenizer.from_file(str(tinyllama_tokenizer_path))
    gigatoken_tok = Tokenizer(tinyllama_tokenizer_path)
    for text in TEXTS:
        ids = hf_tok.encode(text, add_special_tokens=False).ids
        assert gigatoken_tok.decode(ids).decode("utf-8", "replace") == hf_tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Metaspace / normalizer feature matrix (synthetic tokenizers, no downloads)
# ---------------------------------------------------------------------------

_VOCAB = {
    "<unk>": 0,
    "<s>": 1,
    **{f"<0x{b:02X}>": 2 + b for b in range(256)},
    "▁": 258,
    "H": 259,
    "i": 260,
    "Hi": 261,
    "▁Hi": 262,
    "▁▁": 263,
}
_MERGES = [("H", "i"), ("▁", "Hi"), ("▁", "▁")]

_MATRIX_TEXTS = [
    "Hi Hi",
    " Hi",
    "Hi  Hi",
    "Hi<s>Hi",
    "<s>Hi",
    "Hi <s> Hi",
    "▁Hi",
    "",
    " ",
]


@pytest.mark.parametrize(
    "scheme,split,lstrip,rstrip,normalized",
    list(itertools.product(["never", "always", "first"], [True, False], [False, True], [False, True], [False, True])),
)
def test_metaspace_added_token_matrix(scheme, split, lstrip, rstrip, normalized):
    hf_tok = HFTokenizer(BPE(_VOCAB, _MERGES, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    hf_tok.add_tokens([AddedToken("<s>", normalized=normalized, special=True, lstrip=lstrip, rstrip=rstrip)])
    hf_tok.normalizer = normalizers.Sequence(
        [
            normalizers.Strip(left=False, right=True),
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
    )
    hf_tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=scheme, split=split)
    gigatoken_tok = Tokenizer(hf_tok)
    _assert_parity(hf_tok, gigatoken_tok, _MATRIX_TEXTS)


def test_prepend_replace_normalizer_matches_hf():
    """Llama-2 style: Prepend+Replace in the normalizer, no pre-tokenizer."""
    hf_tok = HFTokenizer(BPE(_VOCAB, _MERGES, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    hf_tok.add_tokens([AddedToken("<s>", normalized=False, special=True)])
    hf_tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    gigatoken_tok = Tokenizer(hf_tok)
    _assert_parity(hf_tok, gigatoken_tok, _MATRIX_TEXTS)


def test_unsupported_normalizer_errors():
    """Unknown normalizers must fail at load, not silently diverge."""
    hf_tok = HFTokenizer(BPE(_VOCAB, _MERGES, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    hf_tok.normalizer = normalizers.NFKD()
    with pytest.raises(Exception, match="NFKD"):
        Tokenizer(hf_tok)


# ---------------------------------------------------------------------------
# from_sentencepiece vs raw sentencepiece (golden IDs)
# ---------------------------------------------------------------------------

# SentencePieceProcessor.encode outputs, captured with sentencepiece 0.2.1.
SP4096_GOLDEN = {
    "Hello, world!": [4053, 465, 4015, 4034, 952, 4081],
    "double  space": [4022, 273, 890, 1860],
    "\ttab and\nnewline": [4013, 393, 290, 569, 882],
    " leading": [294, 335, 280],
    "trailing ": [4013, 364, 3982],
    "NFKC: ﬁ ½ ｗｉｄｅ": [4057, 4054, 4078, 4041, 4059, 277, 4016, 4011, 4044, 230, 133, 136, 4048, 2334],
    "CJK 日本語": [4041, 4073, 4078, 4011, 234, 155, 169, 234, 160, 176, 236, 174, 162],
    "null\x00byte": [4017, 874, 4, 1625, 468],
    "code:\n    indented block": [4023, 1502, 4059, 753, 303, 285, 3382],
    "": [],
    " ": [],
}

TINYLLAMA_GOLDEN = {
    "Hello, world!": [15043, 29892, 3186, 29991],
    "double  space": [3765, 29871, 2913],
    "\ttab and\nnewline": [29871, 12, 3891, 322, 13, 1482, 1220],
    " leading": [29871, 8236],
    "trailing ": [25053, 29871],
    "CJK 日本語": [315, 29967, 29968, 29871, 30325, 30346, 30968],
    "null\x00byte": [1870, 3, 10389],
    # Llama 2's manually added whitespace pieces (▁▁ etc., score 0) must rank
    # behind trained merges — this catches ID-ranked (rather than
    # score-ranked) merge recovery.
    "code:\n    indented block": [775, 29901, 13, 1678, 1399, 14927, 2908],
    "": [],
    " ": [259],
}


def test_from_sentencepiece_sp4096(sp4096_spm_path):
    tok = Tokenizer.from_sentencepiece(sp4096_spm_path)
    for text, expected in SP4096_GOLDEN.items():
        assert tok.encode(text).tolist() == expected, f"mismatch for {text!r}"


def test_from_sentencepiece_tinyllama(tinyllama_spm_path):
    tok = Tokenizer.from_sentencepiece(tinyllama_spm_path)
    for text, expected in TINYLLAMA_GOLDEN.items():
        assert tok.encode(text).tolist() == expected, f"mismatch for {text!r}"


def test_model_path_routing(sp4096_spm_path):
    """A .model path passed to the Tokenizer constructor loads as sentencepiece."""
    direct = Tokenizer(sp4096_spm_path)
    explicit = Tokenizer.from_sentencepiece(sp4096_spm_path)
    for text in TEXTS:
        assert direct.encode(text).tolist() == explicit.encode(text).tolist()


def test_from_sentencepiece_rejects_non_bpe():
    with pytest.raises(ValueError, match="sentencepiece"):
        Tokenizer.from_sentencepiece(b"\x00\x01not a real model")
