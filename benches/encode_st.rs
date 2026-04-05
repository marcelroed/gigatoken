use jeton_rs::encode::tiktoken_radix;
use jeton_rs::load_tokenizer::hf::load_hf_bpe;
use jeton_rs::pretokenize::pretoken_fast::FastPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json");
    eprintln!("Loading GPT-2 tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load GPT-2 tokenizer");

    let limit_mb: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());

    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();
    let input = std::fs::read(&owt_path).expect("Could not read ~/data/owt_train.txt");
    eprintln!("Read {:.2} GB in {:.1}s", input.len() as f64 / 1e9, t0.elapsed().as_secs_f64());

    let input = if let Some(mb) = limit_mb {
        let limit = mb * 1_000_000;
        let end = input[..limit.min(input.len())].iter().rposition(|&b| b == b'\n').unwrap_or(limit.min(input.len()));
        eprintln!("Truncating to {} MB ({} bytes)", mb, end);
        &input[..end]
    } else {
        &input[..]
    };
    let size_gb = input.len() as f64 / 1e9;

    let text = unsafe { std::str::from_utf8_unchecked(input) };
    let lines: Vec<&[u8]> = text.lines().map(|l| l.as_bytes()).collect();
    eprintln!("{} lines\n", lines.len());

    // --- Batch encoder (run first to avoid warm-cache bias) ---
    let tokenizer2 = load_hf_bpe(&tokenizer_path).expect("Could not load GPT-2 tokenizer");
    eprintln!("Encoding (single-threaded, batch)...");
    let start = Instant::now();
    let (tokens_batch, _boundaries) = tiktoken_radix::encode_lines(&lines, &tokenizer2);
    let total_tokens_batch = tokens_batch.len();
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("{total_tokens_batch} tokens in {elapsed:.2}s — {:.2} GB/s ({:.0} MB/s)",
        size_gb / elapsed, size_gb / elapsed * 1000.0);

    // --- Memoized (materialize output for fair comparison) ---
    eprintln!("\nEncoding (single-threaded, memoized)...");
    let start = Instant::now();
    let mut total_tokens: usize = 0;
    let mut output_memoized: Vec<u32> = Vec::with_capacity(input.len() / 4);
    for &line in &lines {
        for arc in tokenizer.memoized_encode(FastPretokenizer::new(line)) {
            total_tokens += arc.len();
            for &t in arc.iter() {
                output_memoized.push(t.0);
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("{total_tokens} tokens in {elapsed:.2}s — {:.2} GB/s ({:.0} MB/s)",
        size_gb / elapsed, size_gb / elapsed * 1000.0);

    assert_eq!(total_tokens, total_tokens_batch,
        "Token count mismatch! memoized={total_tokens} batch={total_tokens_batch}");

    // Verify ALL tokens match
    for i in 0..total_tokens {
        if output_memoized[i] != tokens_batch[i].0 {
            panic!("Token mismatch at position {i}/{total_tokens}: memoized={} batch={}",
                output_memoized[i], tokens_batch[i].0);
        }
    }
    eprintln!("\nToken counts and values match.");
}
