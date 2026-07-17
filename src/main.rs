#![feature(portable_simd)]

mod bpe;
mod bpe_train;
mod input;
mod load_tokenizer;
mod pretokenize;
#[cfg(test)]
mod test_hub;
mod token;

use input::MmappedFile;
use input::Resource;
use input::jsonl::JsonLinesSlice;
use std::path::PathBuf;

#[allow(deprecated)]
pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    let tokenizer_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/scripts/tinyllama_tokenizer.json"
        ))
    });

    let input_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        std::env::home_dir()
            .unwrap()
            .join("data/dclm-baseline/shard_00000000_processed.jsonl")
    });

    let field = args.get(3).map(|s| s.as_str()).unwrap_or("text");

    eprintln!("Tokenizer:  {}", tokenizer_path.display());
    eprintln!("Input:      {}", input_path.display());
    eprintln!("Field:      {}", field);

    // Load tokenizer
    let tokenizer = load_tokenizer::hf::load_hf_sentencepiece(&tokenizer_path)
        .expect("Failed to load tokenizer");
    eprintln!("Loaded tokenizer: {:?}", tokenizer);

    // Mmap input file
    let mmap = MmappedFile::open(&input_path).expect("Failed to open input file");
    let bytes = mmap.as_bytes();
    eprintln!("Input size: {:.1} MB", bytes.len() as f64 / 1e6);

    // Parse JSONL to get documents
    let start = std::time::Instant::now();
    let docs: Vec<_> = JsonLinesSlice::new(bytes, field)
        .map(|doc| {
            let b = doc.as_ref();
            // Convert to string for encoding
            unsafe { std::str::from_utf8_unchecked(b) }.to_string()
        })
        .collect();
    let parse_time = start.elapsed();
    let total_chars: usize = docs.iter().map(|d| d.len()).sum();
    let total_text_mb = total_chars as f64 / 1e6;
    eprintln!(
        "Parsed {} docs ({:.1} MB text) in {:.2}s",
        docs.len(),
        total_text_mb,
        parse_time.as_secs_f64()
    );
    eprintln!();

    // Load r50k tokenizer for GPT-2 style encoding
    let r50k_path = args.get(4).map(PathBuf::from).unwrap_or_else(|| {
        std::env::home_dir()
            .unwrap()
            .join("data/tokenizers/r50k_base.tiktoken")
    });
    let mut r50k =
        load_tokenizer::tiktoken::load_tiktoken(&r50k_path).expect("Failed to load r50k tokenizer");
    eprintln!("Loaded r50k: {:?}", r50k);

    // Encode with r50k (GPT-2 style: pretokenize + memoized BPE)
    let start = std::time::Instant::now();
    let mut total_tokens: usize = 0;
    for doc in &docs {
        r50k.memoized_encode(pretokenize::pretokenize_as_iter(doc.as_bytes()), |tokens| {
            total_tokens += tokens.len();
        });
    }
    let elapsed = start.elapsed();
    eprintln!(
        "r50k encode: {} tokens in {:.2}s ({:.1} MB/s)",
        total_tokens,
        elapsed.as_secs_f64(),
        total_text_mb / elapsed.as_secs_f64(),
    );
}
