from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel

# from toker.toker_rs import PretokenizerIterator
from toker.toker_rs import pretokenizer
from tqdm import tqdm


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


def decode_to_text(s: str, errors: str = "strict") -> str:
    """Printable string -> original UTF-8 text."""
    return decode_to_bytes(s).decode("utf-8", errors=errors)


def test_pretokenizer():
    print("Starting test")
    path = Path("../../data/owt_valid.txt")
    file_loaded = path.read_text()
    print("Starting pre_tokenize")
    hf_iterator = ByteLevel(add_prefix_space=False).pre_tokenize_str(file_loaded)
    it = pretokenizer(file_loaded.encode("utf-8"))
    print("Starting loop")
    for i, (pretoken, (hf_pretoken, position)) in enumerate(
        zip(it, tqdm(hf_iterator), strict=True)
    ):
        hf_pretoken = decode_to_bytes(hf_pretoken)
        assert pretoken == hf_pretoken, f"{pretoken} != {hf_pretoken} at {position}"
    # assert tokens == [
    #     "Hello",
    #     ",",
    #     "world",
    #     "!",
    #     "This",
    #     "is",
    #     "a",
    #     "test",
    #     ".",
    #     "Let's",
    #     "see",
    #     "how",
    #     "it",
    #     "works",
    #     ".",
    # ]

    # Now test with the tokenizers library
    # pretokenizer = PreTokenizer.custom(lambda i: (i, tokens))
    # tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel())
    # tokenizer.pre_tokenizer = pretokenizer

    # output = tokenizer.encode("Hello, world! This is a test. Let's see how it works.")
    # assert output.tokens == tokens


def test_pretokenizer_speed():
    path = Path("../../data/TinyStoriesV2-GPT4-train.txt")
    for tok in tqdm(pretokenizer(path.read_bytes())):
        pass
