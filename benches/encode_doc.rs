//! Whole-document multithreaded encode benchmark, mirroring
//! `BPETokenizer.encode_files` on a single plain-text file: the entire input
//! is ONE document handed to the library's parallel encode path
//! (`encode_docs_ragged`), which splits it at pretoken-safe boundaries
//! (token-identical to a serial pass), encodes with a persistent worker
//! pool, and gathers one flat id buffer.
//!
//! One round per process by default: a first pass over a fresh dataset is
//! the workload that matters, and everything a second in-process round
//! reuses — worker pretoken caches, allocator arenas, faulted pages, grown
//! buffer capacities — makes later rounds unrealistically fast. Restart the
//! binary to collect independent samples; ENCODE_ROUNDS>1 remains available
//! for cache-behavior experiments.
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
    common::allow_thp();
    let tokenizer_json = std::env::var("TOKENIZER_JSON")
        .unwrap_or_else(|_| "data/olmo3_tokenizer.json".to_string());
    let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&tokenizer_json);
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load tokenizer");

    let input = common::load_owt_input(Some(DEFAULT_MB));
    let size_mb = input.len() as f64 / 1e6;
    eprintln!("1 document, {} threads\n", rayon::current_num_threads());

    let rounds: usize = std::env::var("ENCODE_ROUNDS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    for round in 0..rounds {
        // Fresh worker pool per round so extra rounds at least start with
        // cold pretoken caches (the pool retains one forked tokenizer per
        // rayon thread).
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
