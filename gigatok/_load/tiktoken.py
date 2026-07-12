# URL sources: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

from typing import TypedDict

from tiktoken.load import load_tiktoken_bpe

ENDOFTEXT = "<|endoftext|>"

r50k_pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class EncodingParams(TypedDict):
    """Constructor kwargs for tiktoken.Encoding."""

    name: str
    explicit_n_vocab: int
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int]


def r50k_base() -> EncodingParams:
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        expected_hash="306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    )
    return {
        "name": "r50k_base",
        "explicit_n_vocab": 50257,
        "pat_str": r50k_pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }
