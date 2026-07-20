# Gigatoken

<div align="center">

~300-1000x faster than HuggingFace's tokenizers, drop-in replacement.

*Tokenize your text data at GB/s!*

![GPT-2 Speedup](https://raw.githubusercontent.com/marcelroed/gigatoken/main/assets/throughput_owt_train_gpt-2.svg)

Keep in mind that both HF tokenizers and tiktoken are already running multithreaded Rust!
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
    --in-memory --validate --comparison-limit 100MB \
    --separator "<|endoftext|>"
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
Keep in mind that you might need to run twice on macOS to get a good reading since the first run will always perform a security scan.

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
