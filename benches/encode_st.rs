use jeton_rs::load_tokenizer::hf::load_hf_bpe;
use jeton_rs::pretokenize::pretoken_fast::FastPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

/// Load the benchmark input from `~/data/owt_train.txt`.
///
/// Optionally caps the input for fast profiling iterations. Set ENCODE_MB to
/// the desired number of megabytes (e.g. ENCODE_MB=500). When set, only that
/// many bytes are read from disk (so the read does not dominate a profile of
/// the encode loop).
fn load_input() -> Vec<u8> {
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();

    let input = match std::env::var("ENCODE_MB") {
        Ok(mb) => {
            use std::io::Read;
            let max_bytes = mb.trim().parse::<usize>().expect("ENCODE_MB must be an integer") * 1_000_000;
            let file = std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
            let mut data = Vec::with_capacity(max_bytes);
            file.take(max_bytes as u64).read_to_end(&mut data).expect("read failed");
            // Back up to a UTF-8 character boundary.
            let mut end = data.len();
            while end > 0 && std::str::from_utf8(&data[..end]).is_err() {
                end -= 1;
            }
            data.truncate(end);
            eprintln!("Capped input to {} MB (ENCODE_MB={mb})", data.len() / 1_000_000);
            data
        }
        Err(_) => std::fs::read(&owt_path).expect("Could not read ~/data/owt_train.txt"),
    };

    let size_gb = input.len() as f64 / 1e9;
    eprintln!("Read {:.2} GB in {:.1}s", size_gb, t0.elapsed().as_secs_f64());
    input
}

fn main() {
    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json");
    eprintln!("Loading GPT-2 tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load GPT-2 tokenizer");

    let input = load_input();
    let size_gb = input.len() as f64 / 1e9;
    // Encode the whole buffer in one pass (matches real usage; the pretokenizer
    // handles newlines itself, so pre-splitting into lines is unnecessary).
    let buf: &[u8] = &input;

    eprintln!("Encoding (single-threaded)...");
    let start = Instant::now();
    let mut total_tokens: usize = 0;
    tokenizer.memoized_encode(FastPretokenizer::new(buf), |tokens| {
        total_tokens += tokens.len();
    });
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
