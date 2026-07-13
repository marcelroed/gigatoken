//! Whole-document multithreaded encode benchmark, mirroring
//! `BPETokenizer.encode_files` on a single plain-text file: the entire input
//! is ONE document handed to the library's parallel encode path
//! (`encode_docs_ragged`), which splits it at pretoken-safe boundaries
//! (token-identical to a serial pass), encodes with a pooled worker per
//! thread, and gathers one flat id buffer. Five rounds, each with a fresh
//! worker pool so every round is a cold-cache sample.
//!
//! Run with: cargo bench --bench encode_doc              (2 GB default)
//!           ENCODE_MB=500 cargo bench --bench encode_doc
//!           TOKENIZER_JSON=data/qwen3_5_tokenizer.json cargo bench --bench encode_doc

use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::{WorkerPool, encode_docs_ragged};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

mod common;

const DEFAULT_MB: usize = 2000;

fn main() {
    let tokenizer_json = std::env::var("TOKENIZER_JSON")
        .unwrap_or_else(|_| "data/olmo3_tokenizer.json".to_string());
    let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load tokenizer");

    let input = common::load_owt_input(Some(DEFAULT_MB));
    let size_mb = input.len() as f64 / 1e6;
    eprintln!("1 document, {} threads\n", rayon::current_num_threads());

    for round in 0..5 {
        // Fresh worker pool every round: the pool retains one forked
        // tokenizer (and its pretoken caches) per rayon thread, so reusing
        // it would measure warm-cache reruns of the same input rather than
        // realistic first-pass encoding. Each round is an independent
        // cold-cache sample.
        let workers = WorkerPool::new();
        let t0 = Instant::now();
        let (flat, lens) = encode_docs_ragged(&workers, &tokenizer, &[&input]);
        black_box((&flat, lens));
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "round {round}: {} tokens in {elapsed:.3}s — {:.0} MB/s",
            flat.len(),
            size_mb / elapsed
        );
    }
}
