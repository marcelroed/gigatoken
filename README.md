# Gigatoken

<div align="center">

400-1600x faster than HuggingFace's tokenizers, drop-in replacement.

*Tokenize your text data at GB/s!*

![GPT-2 Speedup](assets/throughput_owt_train_gpt-2.svg)

Keep in mind that both HF tokenizers and tiktoken are already implemented in Rust!
</div>

## What is Gigatoken?
Gigatoken is the fastest tokenizer for language modeling.
It supports a wide range of CPU hardware, and nearly all commonly used tokenizers.

## Installation
```
pip install gigatoken
```

## Usage
Gigatoken can be used with its own API, or in compatibility mode with HuggingFace Tokenizers or Tiktoken.

### Compatibility Mode (Easiest)
```
import gigatoken as gt

# Minimum change from existing HuggingFace tokenizers usage (compatibility mode)
hf_tokenizer = ...
tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()

# tokenizer can be used in the same contexts as hf_tokenizer
tokens = tokenizer.encode_batch(["This is a test string", "And here is another"])
```

A substantial amount of effort has been put into making sure the outputs match exactly with what you would get with HuggingFace Tokenizers in this setting, but this is at a non-negligible cost to performance.
Below are some graphs of performance emasurements for typical 

### Gigatoken API (Fastest)
```
import gigatoken as gt

tokenizer = gt.Tokenizer("Qwen/Qwen3-8B")  # Accepts HF model names
file_source = gt.TextFileSource(["owt_train.txt"], separator="<|endoftext|>")
tokens = tokenizer.encode_files(file_source)
```

Using the Gigatoken API lets the Rust implementation read data directly, and skips as much overhead as possible while allowing for maximum parallelism.
Keep in mind that passing Python data structures through this API still incurs the overhead of reading from Python.

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

<details>
<summary>AI Use Disclosure</summary>
A majority of this code base was crafted by hand without any use of AI (which can be seen from the project's Git history).
In the final stages of the project, AI was used to assist:

* Implementing the user-facing API
* Widening of compatibility, for instance generalizing and porting the pretokenizer implementations to support more tokenizers, less interesting features like padding/truncation/unicode normalization
* Porting SIMD strategies between AVX512/AVX2/NEON
* Final profiling stages and the last ~4x worth of performance from eliminating branch prediction and improving the pretoken cache hierarchy
* Refactoring and code reuse
</details>
