//! Whole-file parallel encode benchmark. For ByteLevel BPE tokenizers the
//! entire input is ONE document handed to the library's parallel encode path
//! (`encode_docs_ragged`) — the same chunking policy, pretoken-safe
//! splitting, and persistent worker pool as `BPETokenizer.encode_batch` /
//! `encode_files` — so this measures gigatoken's own parallelism, not a
//! bench-local split. SentencePiece-style tokenizers never split a document
//! (merges can span it), so for those the input is split into ~1 MB
//! newline-bounded documents first and handed to `sp_encode_docs_ragged`,
//! which parallelizes across documents.
//!
//! Run with: cargo bench --bench encode                 (full OWT)
//!           ENCODE_MB=500 cargo bench --bench encode
//!           TOKENIZER_JSON=data/qwen3_5_tokenizer.json cargo bench --bench encode
//!           TOKENIZER_JSON=meta-llama/Llama-3.1-8B cargo bench --bench encode
//!             (a HuggingFace repo id: resolved via the standard HF cache,
//!              downloaded into it on a miss)

use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use gigatoken_rs::load_tokenizer::hub;
use gigatoken_rs::{WorkerPool, encode_docs_ragged, sp_encode_docs_ragged};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

mod common;

/// Split at the first newline after each `target`-byte stride (documents for
/// the SP path; a newline is always a char boundary).
fn split_docs(text: &str, target: usize) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut docs = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = (start + target).min(bytes.len());
        if end < bytes.len() {
            end = match memchr::memchr(b'\n', &bytes[end..]) {
                Some(i) => end + i + 1,
                None => bytes.len(),
            };
        }
        docs.push(&text[start..end]);
        start = end;
    }
    docs
}

fn main() {
    common::allow_thp();
    let tokenizer_json =
        std::env::var("TOKENIZER_JSON").unwrap_or_else(|_| "data/gpt2_tokenizer.json".to_string());
    let tokenizer_path = {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);
        if !path.exists() && hub::looks_like_repo_id(&tokenizer_json) {
            hub::hub_file(&tokenizer_json, "tokenizer.json", "main")
                .expect("Could not fetch tokenizer.json from the HuggingFace Hub")
        } else {
            path
        }
    };
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let data = std::fs::read(&tokenizer_path).expect("Could not read tokenizer.json");
    let tokenizer = load_hf_slice(&data).expect("Could not load tokenizer");

    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;

    let workers = WorkerPool::new();
    let (n_tokens, elapsed) = match &tokenizer {
        HfTokenizer::Bpe(tokenizer) => {
            eprintln!(
                "Encoding (ByteLevel BPE, 1 document, {} threads)...",
                rayon::current_num_threads()
            );
            let start = Instant::now();
            let (ids, lens) = encode_docs_ragged(&workers, tokenizer, &[&input]);
            black_box((&ids, lens));
            (ids.len(), start.elapsed().as_secs_f64())
        }
        HfTokenizer::SentencePiece(tokenizer) => {
            // Trailing partial UTF-8 char (from the ENCODE_MB byte cap) is
            // dropped; SP encoding takes str input.
            let text = match std::str::from_utf8(&input) {
                Ok(text) => text,
                Err(e) => std::str::from_utf8(&input[..e.valid_up_to()]).unwrap(),
            };
            let docs = split_docs(text, 1 << 20);
            eprintln!(
                "Encoding (SentencePiece, {} documents, {} threads)...",
                docs.len(),
                rayon::current_num_threads()
            );
            let start = Instant::now();
            let (ids, lens) = sp_encode_docs_ragged(tokenizer, &docs);
            black_box((&ids, lens));
            (ids.len(), start.elapsed().as_secs_f64())
        }
    };
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{n_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
