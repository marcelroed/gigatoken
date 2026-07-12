import tiktoken
from pytest import fixture

from gigatoken.gigatoken_rs import BPETokenizer


@fixture
def tiktoken_r50k():
    return tiktoken.get_encoding("r50k_base")


@fixture
def gigatoken_r50k(r50k_tiktoken_path):
    return BPETokenizer.from_tiktoken(r50k_tiktoken_path)


def test_use_gigatoken_model(gigatoken_r50k):
    print(gigatoken_r50k)
    print(gigatoken_r50k.encode(b"Here's a test string"))
