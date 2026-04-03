use jeton_rs::load_tokenizer::hf::load_hf_bpe;
use jeton_rs::pretokenize::pretoken_fast::FastPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json");
    eprintln!("Loading GPT-2 tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load GPT-2 tokenizer");

    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();
    let input = std::fs::read(&owt_path).expect("Could not read ~/data/owt_train.txt");
    let size_gb = input.len() as f64 / 1e9;
    eprintln!("Read {:.2} GB in {:.1}s", size_gb, t0.elapsed().as_secs_f64());

    let text = unsafe { std::str::from_utf8_unchecked(&input) };
    let lines: Vec<&[u8]> = text.lines().map(|l| l.as_bytes()).collect();
    eprintln!("{} lines\n", lines.len());

    eprintln!("Encoding (single-threaded)...");
    let start = Instant::now();
    let mut total_tokens: usize = 0;
    for &line in &lines {
        for arc in tokenizer.memoized_encode(FastPretokenizer::new(line)) {
            total_tokens += arc.len();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!("{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)", throughput_gb * 1000.0);
}
