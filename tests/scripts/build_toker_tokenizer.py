from pathlib import Path

from rich import get_console
from rich.pretty import Pretty, pprint
from rich.table import Table

from toker import train_bpe


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


# shared_path = Path("../../data/owt_train.txt")
shared_path = Path("/Users/marcel/merged_text_frequency.parquet")
vocab_size = 128_000


# Build the same byte -> unicode table used in your Rust code
def _build_tables():
    # Bytes that stay as themselves (printable, single-codepoint chars)
    allowed = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))

    # Table indexed by original byte value -> printable char
    b2u = [None] * 256
    for b in allowed:
        b2u[b] = chr(b)

    # The “missing” bytes map to U+0100, U+0101, ... in ascending byte order
    n = 0
    for b in range(256):
        if b2u[b] is None:
            b2u[b] = chr(256 + n)  # 256 == 0x0100
            n += 1

    # Inverse mapping: printable char -> original byte
    u2b = {ch: i for i, ch in enumerate(b2u)}
    return b2u, u2b


B2U, U2B = _build_tables()


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


def save_toker(result):
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
    with (Path(__file__).parent / f"toker_tokenization_{vocab_size}.json").open(
        "w"
    ) as f:
        import json

        json.dump(d, f, indent=2, ensure_ascii=False)


def build_toker_tokenizer():
    # bytes = shared_path.read_bytes()
    result = train_bpe(shared_path, vocab_size=vocab_size, special_tokens=[])
    save_toker(result)
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
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
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
    tokenizer.save(
        str(Path(__file__).parent / f"hf_tokenizer_ts_valid_{vocab_size}.json")
    )
    return tokenizer


def build_compare():
    toker_result = build_toker_tokenizer()
    hf_result = build_hf_tokenizer()
    breakpoint()
    table = Table()
    table.add_row(
        Pretty(toker_result[0]),
        Pretty(
            {v: k for k, v in sorted(hf_result.get_vocab().items(), key=lambda x: x[1])}
        ),
    )
    get_console().print(table)


if __name__ == "__main__":
    # build_compare()
    build_toker_tokenizer()
