//! Measure pretoken-cache memory growth while encoding OWT.
//!
//! Encodes ~/data/owt_train.txt in one streaming pass with a single
//! (non-forked) tokenizer and prints cache statistics at doubling byte
//! checkpoints, so cache size vs. input size can be fit and extrapolated.
//!
//! Usage: cargo run --release --example cache_memory [tokenizer.json]

use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let tokenizer_path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json")
    });
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load tokenizer");

    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    let file = std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap failed") };
    let bytes: &[u8] = &mmap;
    eprintln!("Input: {:.2} GB", bytes.len() as f64 / 1e9);

    // Doubling checkpoints from 16 MB up to the full file.
    let mut checkpoints: Vec<usize> = Vec::new();
    let mut c: usize = 16 << 20;
    while c < bytes.len() {
        checkpoints.push(c);
        c *= 2;
    }
    checkpoints.push(bytes.len());

    println!(
        "{:>14} {:>14} {:>12} {:>12} {:>10} {:>10} {:>14} {:>12} {:>12} {:>8}",
        "bytes", "tokens", "short_len", "short_cap", "long_len", "long_cap", "long_key_bytes",
        "arena_len", "arena_cap", "sec"
    );

    let mut prev = 0usize;
    let mut total_tokens: u64 = 0;
    let start = Instant::now();
    for &cp in &checkpoints {
        // Advance the cut to a newline boundary so we never split a pretoken
        // other than a whitespace run.
        let mut end = cp.min(bytes.len());
        while end < bytes.len() && bytes[end] != b'\n' {
            end += 1;
        }
        if end < bytes.len() {
            end += 1;
        }
        if end <= prev {
            continue;
        }
        tokenizer.encode_with_added_tokens(&bytes[prev..end], |toks| {
            total_tokens += toks.len() as u64;
        });
        prev = end;
        let (sl, sc, ll, lc, lkb, al, ac) = tokenizer.cache_mem_stats();
        println!(
            "{:>14} {:>14} {:>12} {:>12} {:>10} {:>10} {:>14} {:>12} {:>12} {:>8.1}",
            end, total_tokens, sl, sc, ll, lc, lkb, al, ac,
            start.elapsed().as_secs_f64()
        );
    }
}
