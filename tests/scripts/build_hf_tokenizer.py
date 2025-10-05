from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def build_hf_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    # tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    split = (
        Path("../../data/TinyStoriesV2-GPT4-train.txt")
        .read_text()
        .split("<|endoftext|>")
    )
    tokenizer.train_from_iterator(
        split,
        trainer=trainer,
        length=len(split),
    )
    tokenizer.save(str(Path(__file__).parent / "hf_tokenizer.json"))


def build_hf_tokenizer_from_parquet():
    import polars as pl

    df = pl.scan_parquet("~/merged_text_frequency.parquet")
    length = int(df.select(pl.len()).collect().item())
    print(f"Length: {length}")
    texts = df.select(pl.col("text"))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    batch_size = 1024 * 1024

    tokenizer.train_from_iterator(
        (
            item
            for row in (
                texts.slice(i * batch_size, batch_size).collect()
                for i in range(ceildiv(length, batch_size))
            )
            for item in row[:, 0].to_list()
            if item is not None
        ),
        trainer=trainer,
        length=length,
    )


if __name__ == "__main__":
    build_hf_tokenizer_from_parquet()
