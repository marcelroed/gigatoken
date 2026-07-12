import pytest

# from gigatoken import Tokenizer — high-level Tokenizer API not yet implemented


@pytest.mark.skip(reason="High-level Tokenizer API not yet implemented")
def test_load_tiktoken():
    tokenizer = Tokenizer.from_tiktoken(tiktoken_name="r50k_base")


@pytest.mark.skip(reason="High-level Tokenizer API not yet implemented")
def test_load_hf():
    tokenizer = Tokenizer.from_hf(hf_name="openai-community/gpt2")
    assert tokenizer.vocab_size == 50257


@pytest.mark.skip(reason="High-level Tokenizer API not yet implemented")
def test_tokenizer_python_pipeline():
    tokenizer = (
        Tokenizer.from_tiktoken(tiktoken_name="r50k_base")
        .build_pipeline()
        .source_python_generator()
        .sink_python_generator()  # Defaults to
        .sink_
    )
    set_of_texts = [
        b"Hello, world!",
        b"This is a test with a longer sequence.",
        b"Tokenization is fun!",
        b"Let's see how it works.",
    ]

    tokenized_generator = tokenizer.run(set_of_texts)
    for tokens in tokenized_generator:
        print(tokens)
