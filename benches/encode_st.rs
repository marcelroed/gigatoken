use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::pretokenize::FastR50kPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

mod common;
fn main() {
    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json");
    eprintln!("Loading GPT-2 tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load GPT-2 tokenizer");

    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;
    // Encode the whole buffer in one pass (matches real usage; the pretokenizer
    // handles newlines itself, so pre-splitting into lines is unnecessary).
    let buf: &[u8] = &input;

    eprintln!("Encoding (single-threaded)...");
    let start = Instant::now();
    let mut total_tokens: usize = 0;
    tokenizer.memoized_encode(FastR50kPretokenizer::new(buf), |tokens| {
        total_tokens += tokens.len();
    });
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
