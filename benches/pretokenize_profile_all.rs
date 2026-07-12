//! Per-scheme variant of `pretokenize_profile`: same single-pass loop over
//! OWT with every yielded pretoken black_boxed, scheme selected via the
//! SCHEME env var (r50k | cl100k | olmo3 | qwen2 | qwen3_5 | deepseek_v3).
//! r50k is also what ByteLevel tokenizers like ModernBERT resolve to. Used
//! for interleaved A/B runs of the mask-scanner schemes.

use gigatoken_rs::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer,
};
use std::hint::black_box;
use std::time::Instant;

mod common;

macro_rules! drive {
    ($ty:ty, $buf:expr) => {{
        let mut total_tokens: usize = 0;
        let mut iter = <$ty>::new($buf);
        while let Some(pretoken) = iter.next() {
            black_box(pretoken);
            total_tokens += 1;
        }
        total_tokens
    }};
}

fn main() {
    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;
    let buf: &[u8] = &input;
    let scheme = std::env::var("SCHEME").unwrap_or_else(|_| "r50k".to_string());

    eprintln!("Pretokenizing ({scheme}, single-threaded, whole buffer)...");
    let start = Instant::now();
    let total_tokens = match scheme.as_str() {
        "r50k" => drive!(FastR50kPretokenizer, buf),
        "cl100k" => drive!(FastCl100kPretokenizer, buf),
        "olmo3" => drive!(FastOlmo3Pretokenizer, buf),
        "qwen2" => drive!(FastQwen2Pretokenizer, buf),
        "qwen3_5" => drive!(FastQwen35Pretokenizer, buf),
        "deepseek_v3" => drive!(FastDeepSeekV3Pretokenizer, buf),
        other => panic!("unknown SCHEME {other:?}"),
    };
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
