from pathlib import Path

from rich import get_console
from rich.pretty import Pretty, pprint
from rich.table import Table

from gigatoken import train_bpe


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


# shared_path = Path("../../data/owt_train.txt")
shared_path = Path("/Users/marcel/merged_text_frequency.parquet")
vocab_size = 128_000


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import GPT2_B2U as B2U, GPT2_U2B as U2B


def encode_bytes(data: bytes) -> str:
    """Bytes -> printable GPT-2-style string."""
    return "".join(B2U[b] for b in data)


def decode_to_bytes(s: str) -> bytes:
    """Printable string -> original bytes."""
    try:
        return bytes(U2B[ch] for ch in s)
    except KeyError as e:
        cp = ord(next(iter(e.args[0])))
        raise ValueError(f"Unknown code point U+{cp:04X}") from None


def save_gigatoken(result):
    vocab, merges = result
    vocab = {encode_bytes(v): k for k, v in vocab.items()}
    merges = [(encode_bytes(a), encode_bytes(b)) for a, b in merges]
    d = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    with (Path(__file__).parent / f"gigatoken_tokenization_{vocab_size}.json").open("w") as f:
        import json

        json.dump(d, f, indent=2, ensure_ascii=False)


def build_gigatoken_tokenizer():
    # bytes = shared_path.read_bytes()
    result = train_bpe(shared_path, vocab_size=vocab_size, special_tokens=[])
    save_gigatoken(result)
    return result


def build_hf_tokenizer():
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    split = shared_path.read_text().split("<|endoftext|>")
    tokenizer.train_from_iterator(
        split,
        trainer=trainer,
        length=len(split),
    )
    tokenizer.save(str(Path(__file__).parent / f"hf_tokenizer_ts_valid_{vocab_size}.json"))
    return tokenizer


def build_compare():
    gigatoken_result = build_gigatoken_tokenizer()
    hf_result = build_hf_tokenizer()
    breakpoint()
    table = Table()
    table.add_row(
        Pretty(gigatoken_result[0]),
        Pretty({v: k for k, v in sorted(hf_result.get_vocab().items(), key=lambda x: x[1])}),
    )
    get_console().print(table)


if __name__ == "__main__":
    # build_compare()
    build_gigatoken_tokenizer()
