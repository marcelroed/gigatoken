#![feature(test)]
#![feature(portable_simd)]

use indicatif::ProgressIterator;

use crate::token::TokenId;

// use clap::

mod bpe;
mod bpe_train;
mod input;
mod load_tokenizer;
mod output;
mod pretokenize;
pub(crate) mod simd;
mod token;
pub(crate) mod unicode_tables;

pub fn main() {
    // Get args (path to file, vocab size)
    // let args: Vec<String> = std::env::args().collect();
    // if args.len() != 3 {
    //     eprintln!("Usage: {} <input_file> <vocab_size>", args[0]);
    //     std::process::exit(1);
    // }

    // let input_file = &args[1];
    // let vocab_size: usize = args[2].parse().expect("Invalid vocab size");
    // let file = std::fs::File::open(input_file).unwrap();
    // let bytes_memmapped = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    // let bpe_result = bpe_train::train_bpe(
    //     bpe_train::PretokenizeableSpec::Bytes(bytes_memmapped.as_ref()),
    //     vocab_size,
    //     vec![],
    // );
    // eprintln!("BPE result: {}", bpe_result.vocab.len());

    // let bpe_result = bpe_train::train_bpe(
    //     bpe_train::PretokenizeableSpec::Parquet(input_file.into()),
    //     vocab_size,
    //     vec![],
    // );

    // Tokenize a file using a tiktoken tokenizer
    let data_dir = std::env::home_dir().unwrap().join("data");
    let dir = data_dir.join("tokenizers/r50k_base.tiktoken");
    let mut tokenizer = load_tokenizer::tiktoken::load_tiktoken(dir).unwrap();
    // Memmap the file and treat it as a slice of bytes
    // let path = data_dir.join("TinyStoriesV2-GPT4-train.txt");
    let path = data_dir.join("TinyStoriesV2-GPT4-train.txt");

    // We process in chunks, then compute the total number of tokens we need to write (as well as offsets to write them at).
    const PARALLEL_CHUNK_READ_SIZE: usize = 1024 * 1024 * 1024; // 1GB read each iteration

    let file = std::fs::File::open(path).unwrap();
    let bytes_memmapped = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    // for window in bytes_memmapped.as_ref().windows(PARALLEL_CHUNK_READ_SIZE) {
    let pretoken_iter = pretokenize::pretokenize_as_iter(bytes_memmapped.as_ref());
    // let mut pretoken_iter = pretokenize::pretoken
    //     std::str::from_utf8_unchecked(bytes_memmapped.as_ref())
    // });
    let token_ids = tokenizer.memoized_encode(pretoken_iter);
    // let token_ids = tokenizer.memoized_encode(&mut pretoken_iter);
    let mut out: Vec<u32> = Vec::with_capacity(bytes_memmapped.len() / 3);
    let start_time = std::time::Instant::now();
    // let bar = ProgressBar::new(bytes_memmapped.len() as u64).with_style(
    //     indicatif::ProgressStyle::default_bar()
    //         .template("[{elapsed_precise}] ({per_sec}) [{wide_bar}] {pos}/{len} ({eta})")
    //         .unwrap(),
    // );
    // let out: Vec<TokenId> = token_ids
    //     .map(|pretoken_toks| pretoken_toks.to_vec())
    //     .flatten()
    //     .collect();
    for token_ids in token_ids {
        // bar.inc(token_ids.len() as u64);
        out.extend(unsafe { std::mem::transmute::<&[TokenId], &[u32]>(token_ids.as_ref()) });
    }
    // let out = unsafe { std::mem::transmute::<Vec<TokenId>, Vec<u32>>(out) };
    let end_time = std::time::Instant::now();
    println!(
        "Tokenized {} bytes into {} tokens in {:?} ({:.3} MB/s), with {} total pretokens",
        bytes_memmapped.len(),
        out.len(),
        end_time - start_time,
        bytes_memmapped.len() as f64 / (end_time - start_time).as_secs_f64() / 1024.0 / 1024.0,
        tokenizer.pretoken_cache_size(),
    );
}
