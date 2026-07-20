# Gigatoken

<div align="center">

~300-1000x faster than HuggingFace's tokenizers, drop-in replacement.

*Tokenize your text data at GB/s!*

![GPT-2 Speedup](https://raw.githubusercontent.com/marcelroed/gigatoken/main/assets/throughput_owt_train_gpt-2.svg)

Note that both HF tokenizers and tiktoken are already running multithreaded Rust!
</div>

## What is Gigatoken?
Gigatoken is the fastest tokenizer for language modeling.
It supports a wide range of CPU hardware, and nearly all commonly used tokenizers.

## Installation
```bash
pip install gigatoken
```

## Usage
Gigatoken can be used with its own API, or in compatibility mode with HuggingFace Tokenizers or Tiktoken.

### Compatibility Mode (Easiest)
```python
import gigatoken as gt

# Minimum change from existing HuggingFace tokenizers usage (compatibility mode)
hf_tokenizer = ...
tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()

# tokenizer can be used in the same contexts as hf_tokenizer
tokens = tokenizer.encode_batch(["This is a test string", "And here is another"])
```

A substantial amount of effort has been put into making sure the outputs match exactly with what you would get with HuggingFace Tokenizers in this setting, but this is at a non-negligible cost to performance.
You can still expect way faster performance across the board, but not quite the 1000x you will get with the Gigatoken API.

### Gigatoken API (Fastest)
```python
import gigatoken as gt

tokenizer = gt.Tokenizer("Qwen/Qwen3-8B")  # Accepts HF model names
file_source = gt.TextFileSource(["owt_train.txt"], separator=b"<|endoftext|>")
tokens = tokenizer.encode_files(file_source)
```

Using the Gigatoken API lets the Rust implementation read data directly, and skips as much overhead as possible while allowing for maximum parallelism.
Keep in mind that passing Python data structures through this API still incurs the overhead of reading from Python.


## FAQ
### Q: Did you just way over-optimize for a specific CPU and tokenizer? How is it so fast?
No, I way over-optimized for every combination of these!
The results are very consistent across CPUs (modern x86 and ARM), and across specific tokenizers.

The major improvements are in optimizing heavily an implementation that usually is outsourced to a Regex engine (pretokenization) using SIMD and other tricks, as well as heavily optimizing caching of pretoken mappings (if a word has been seen before, look it up its encoded tokens efficiently).
In addition, interactions with Python are minimized, and threads are minimally interacting with each other.


### Q: How can I quickly check if my tokenizer is supported?
You can try it out without installing anything! The following command will validate and time tokenization for a given HuggingFace model repo: 

```bash
# Download your data
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz  # Just an example!
gunzip owt_train.txt.gz
```

```bash
uvx --with tokenizers gigatoken bench 'openai-community/gpt2' owt_train.txt \
    --validate --doc-separator "<|endoftext|>"
```
```bash
      cpu: Apple M4 Max, 16 cores
gigatoken:    1.432 s |   11920.51 MB at  8327.05 MB/s |  2701.65 Mtok at 1887.23 Mtok/s
       hf:   16.250 s |     100.00 MB at     6.15 MB/s |    22.76 Mtok at    1.40 Mtok/s
gigatoken is 1353.13x faster than hf
validation OK: 20401 documents match
```

```bash
      cpu: AMD EPYC 9565 72-Core Processor, 144 cores, 2 sockets
gigatoken:    0.584 s |   11920.51 MB at 20412.35 MB/s |  2701.65 Mtok at 4626.23 Mtok/s
       hf:    3.738 s |     100.00 MB at    26.75 MB/s |    22.76 Mtok at    6.09 Mtok/s
gigatoken is 763.08x faster than hf
validation OK: 20401 documents match
```
This example uses the train sample from [this dataset](https://huggingface.co/datasets/stanford-cs336/owt-sample).
You can see help for these flags with `uvx gigatoken bench --help`.
You might need to run these twice on macOS to get a good reading, since the first run will always perform a security scan.

At the rates we see on the EPYC CPU, you could tokenize the [entirety of Common Crawl](https://arxiv.org/pdf/2211.04325) (often considered to be the entire internet, 130 trillion tokens) in just under 8 hours!


### Q: I've found a mismatch/slow use-case, is this expected?
Most likely not! Despite reasonably wide testing I don't have every use-case on hand, so please report anything you find in a [GitHub Issue](https://github.com/marcelroed/gigatoken/issues) so I can address it as soon as possible.


<!--
## How does Gigatoken work?

Gigatoken came from a few observations:
* A majority of the time in current tokenizers is spent on pretokenization -> 

Gigatoken implements pretokenizers that run at >2GB/s/thread

Gigatoken is faster than other libraries due to algorithmic and systems changes.
One key benefit of using Gigatoken is that it replaces the regex expression used by almost all tokenizers to do pre-tokenization with a custom implementation of the exact same method.
This is a serious bottleneck for other implementations, and is a big part in this library's breakneck speeds.
Additionally, Gigatoken uses concurrent data structures to use multiprocessing in more places.
\* All reference speeds in this section are measured on an M4 Pro CPU
-->

<!-- benchmarks:start -->
## Benchmarks

<details>
<summary><b>Encoding throughput on owt_train.txt (11.9 GB) — Apple M4 Max (16 cores)</b></summary>

Best of 3 interleaved rounds, one fresh process per measurement, all libraries with
parallelism enabled. gigatoken encodes the whole file un-split; HuggingFace
`tokenizers` (`encode_batch_fast`) gets the first 100 MB and tiktoken
(`encode_ordinary_batch`) the first 1 GB, both presplit on `<|endoftext|>`.
tiktoken rows exist only for tokenizers with official support.

| Tokenizer | gigatoken | HF tokenizers | tiktoken | vs HF | vs tiktoken |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 8.79 GB/s | 6.9 MB/s | 62.8 MB/s | 1,268× | 140× |
| Nemotron 3 | 7.82 GB/s | 10.9 MB/s | — | 715× | — |
| Phi-4 | 7.76 GB/s | 7.7 MB/s | — | 1,012× | — |
| Llama 3 / 3.1 / 3.2 | 7.60 GB/s | 11.2 MB/s | — | 676× | — |
| OLMo 2 / 3 | 7.56 GB/s | 5.8 MB/s | — | 1,299× | — |
| Llama 3.3 | 7.50 GB/s | 15.7 MB/s | — | 479× | — |
| Phi-4-mini | 6.97 GB/s | 7.2 MB/s | — | 964× | — |
| Kimi K2 | 6.88 GB/s | — | — | — | — |
| Llama 4 | 6.81 GB/s | 11.6 MB/s | — | 590× | — |
| Qwen 2 / 2.5 | 6.37 GB/s | 5.8 MB/s | — | 1,105× | — |
| Qwen 3 | 6.36 GB/s | 6.9 MB/s | — | 918× | — |
| Qwen 3.5 / 3.6 | 6.31 GB/s | 6.3 MB/s | — | 994× | — |
| GPT-OSS | 6.20 GB/s | 20.2 MB/s | 87.2 MB/s | 306× | 71× |
| GLM 4 | 6.17 GB/s | 15.8 MB/s | — | 392× | — |
| DeepSeek V3 / R1 / V4 | 5.68 GB/s | 7.2 MB/s | — | 788× | — |
| GLM 5 | 5.55 GB/s | 12.2 MB/s | — | 456× | — |
| ModernBERT | 2.64 GB/s | 5.8 MB/s | — | 452× | — |
| Mistral 7B v0.3 | 1.99 GB/s | 95.1 MB/s | — | 21× | — |
| Gemma 4 | 1.82 GB/s | 85.2 MB/s | — | 21× | — |
| CodeLlama | 1.73 GB/s | 80.2 MB/s | — | 22× | — |
| TinyLlama / Phi-3 (Llama 2) | 1.69 GB/s | 80.1 MB/s | — | 21× | — |
| Gemma 1 | 1.42 GB/s | 85.7 MB/s | — | 17× | — |
| Gemma 3 | 1.38 GB/s | 82.2 MB/s | — | 17× | — |

The slowest rows are the SentencePiece-based tokenizers (Mistral 7B and below),
which remain more expensive to encode than byte-level BPE even with gigatoken's
internal SP parallelism; ModernBERT is byte-level BPE with a heavier
pretokenizer than the GPT-2 family.

Each row is one distinct tokenizer (identical vocab/merges/pretokenizer), measured
on a representative repo. Rows whose tokenizer is shared beyond their own name
(verified by matching tokenizer definitions across the local HF model cache) cover:

- **Nemotron 3** — Nemotron 3 Nano, Super, and Ultra
- **Llama 3 / 3.1 / 3.2** — Llama 3 / 3.1 / 3.2, DeepSeek-R1-Distill-Llama, Hermes 3, Saiga, and other Llama-3 finetunes
- **Llama 3.3** — Llama 3.3, Llama-3.1-Nemotron-Nano-VL, SmolLM3, Kanana 1.5, jina-embeddings-v5, Ultravox
- **Phi-4-mini** — Phi-4-mini and Phi-4-multimodal
- **Kimi K2** — Kimi K2 / K2.5 / K2.6 / K2.7, Kimi-Linear, Kimi-VL, Moonlight
- **Qwen 2 / 2.5** — Qwen 2 and 2.5 (incl. Coder and VL), Qwen3-Coder, Qwen3-VL, DeepSeek-R1 Qwen distills, MiMo V2.5, MiniCPM-o 2.6, InternVL3
- **Qwen 3** — Qwen 3 (incl. Embedding and Reranker), Qwen2.5-Omni, Qwen3-VL-Embedding, MiMo V2.5 Pro, jina-reranker-m0, pplx-embed, MOSS-TTS, Zeta
- **GLM 4** — GLM 4.1V, 4.5, and 4.7
- **DeepSeek V3 / R1 / V4** — DeepSeek V3 / V3.1 / V3.2, R1, V4 Flash and Pro, DeepSeek-VL2
- **GLM 5** — GLM 5 / 5.2 and GLM-4.7-Flash
- **Gemma 4** — Gemma 4 (dense, MoE, and E-series) and DiffusionGemma
- **TinyLlama / Phi-3 (Llama 2)** — TinyLlama, Phi-3-mini, Phi-3.5-mini and Phi-3.5-vision (the Llama 2 vocab)
- **Gemma 3** — Gemma 3 (270M–27B) and EmbeddingGemma

</details>
<!-- benchmarks:end -->

## Citation
If you use Gigatoken in your research, please cite it as:

```bibtex
@software{roed2026gigatoken,
  author = {Marcel R{\o}d},
  title = {{G}igatoken: SIMD and Cache Hierarchies for 1000x Faster Byte-Pair Encoding Tokenization on Modern CPUs},
  url = {https://github.com/marcelroed/gigatoken},
  year = {2026},
}
```

---

<details>
<summary>AI Use Disclosure</summary>
A majority of this code base was crafted by hand without any use of AI (which can be seen from the project's Git history).
In the final stages of the project, AI was used to assist:

* Implementing the user-facing API
* Widening of compatibility, for instance generalizing and porting the pretokenizer implementations to support more tokenizers, less interesting features like padding/truncation/unicode normalization
* Porting SIMD strategies between AVX512/AVX2/NEON
* Final profiling stages and the last ~4x worth of performance from eliminating branching and improving the pretoken cache hierarchy
* Refactoring and code reuse
</details>
