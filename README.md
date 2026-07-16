# Gigatoken

<div align="center">

400-1600x faster than HuggingFace's tokenizers, drop-in replacement.

*Tokenize your text data at GB/s!*

![GPT-2 Speedup](assets/throughput_owt_train_gpt-2.svg)

Keep in mind that both HF tokenizers and tiktoken are already running multithreaded Rust!
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
tokens = tokenizer.encode_batch(["This is a tes runningt string", "And here is another"])
```

A substantial amount of effort has been put into making sure the outputs match exactly with what you would get with HuggingFace Tokenizers in this setting, but this is at a non-negligible cost to performance.
You can still expect way faster performance across the board, but not quite the 1000x you will get with the Gigatoken API.

### Gigatoken API (Fastest)
```
import gigatoken as gt

tokenizer = gt.Tokenizer("Qwen/Qwen3-8B")  # Accepts HF model names
file_source = gt.TextFileSource(["owt_train.txt"], separator=b"<|endoftext|>")
tokens = tokenizer.encode_files(file_source)
```

Using the Gigatoken API lets the Rust implementation read data directly, and skips as much overhead as possible while allowing for maximum parallelism.
Keep in mind that passing Python data structures through this API still incurs the overhead of reading from Python.


## FAQ
### Q: Did you just way over-optimize for a specific CPU and tokenizer?
No, I way over-optimized for every combination of these!
The results are very consistent across CPUs (modern x86 and ARM), and across specific tokenizers.


### Q: How can I quickly check if my tokenizer is supported?
You can try it out without installing anything! The following command will validate and time tokenization for a given HuggingFace model repo: 
```bash
$ uvx gigatoken bench 'openai-community/gpt2' ~/data/owt_train.txt --in-memory --validate --comparison-limit 100MB
```
```bash
gigatoken:    1.316 s |   11920.51 MB at  9059.61 MB/s |  2704.05 Mtok at 2055.08 Mtok/s
       hf:   25.385 s |     100.00 MB at     3.94 MB/s |    22.72 Mtok at    0.90 Mtok/s
gigatoken is 2299.75x faster than hf (by MB/s)
validation OK: 1 documents match
```
You can see help for these flags with `uvx gigatoken bench --help`.

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

---

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
