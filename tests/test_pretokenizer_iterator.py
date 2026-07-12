from pathlib import Path

import pytest
from tokenizers.pre_tokenizers import ByteLevel

# from gigatoken.gigatoken_rs import PretokenizerIterator
from gigatoken.gigatoken_rs import pretokenizer
from tqdm import tqdm


from conftest import gpt2_unicode_to_bytes as decode_to_bytes


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_pretokenizer():
    print("Starting test")
    path = DATA_DIR / "owt_valid.txt"
    if not path.exists():
        pytest.skip("owt_valid.txt not available in data/")
    file_loaded = path.read_text()
    print("Starting pre_tokenize")
    hf_iterator = ByteLevel(add_prefix_space=False).pre_tokenize_str(file_loaded)
    it = pretokenizer(file_loaded.encode("utf-8"))
    print("Starting loop")
    for i, (pretoken, (hf_pretoken, position)) in enumerate(zip(it, tqdm(hf_iterator), strict=True)):
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
    path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    if not path.exists():
        pytest.skip("TinyStoriesV2-GPT4-train.txt not available in data/")
    for tok in tqdm(pretokenizer(path.read_bytes())):
        pass
