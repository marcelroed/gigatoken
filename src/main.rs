use indicatif::ProgressIterator;

// use clap::

mod bpe;
mod bpe_train;
mod input;
mod pretokenize;
mod token;

pub fn main() {
    // Get args (path to file, vocab size)
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_file> <vocab_size>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let vocab_size: usize = args[2].parse().expect("Invalid vocab size");
    let file = std::fs::File::open(input_file).unwrap();
    let bytes_memmapped = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let bpe_result = bpe_train::train_bpe(
        bpe_train::PretokenizeableSpec::Bytes(bytes_memmapped.as_ref()),
        vocab_size,
        vec![],
    );
    eprintln!("BPE result: {}", bpe_result.vocab.len());

    // let bpe_result = bpe_train::train_bpe(
    //     bpe_train::PretokenizeableSpec::Parquet(input_file.into()),
    //     vocab_size,
    //     vec![],
    // );

    // Tokenize a file using a tiktoken tokenizer
    // let mut tokenizer =
    //     bpe::load_tiktoken("/Users/marcel/data/tokenizers/r50k_base.tiktoken").unwrap();
    // // Memmap the file and treat it as a slice of bytes
    // let path = "/Users/marcel/data/TinyStoriesV2-GPT4-train.txt";
    // let file = std::fs::File::open(path).unwrap();
    // let bytes_memmapped = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    // let pretoken_iter = pretokenize::pretokenize_as_iter(bytes_memmapped.as_ref());
    // let token_ids = tokenizer.memoized_encode(pretoken_iter);
    // let mut out: Vec<u32> = vec![];
    // let start_time = std::time::Instant::now();
    // // let bar = ProgressBar::new(bytes_memmapped.len() as u64).with_style(
    // //     indicatif::ProgressStyle::default_bar()
    // //         .template("[{elapsed_precise}] ({per_sec}) [{wide_bar}] {pos}/{len} ({eta})")
    // //         .unwrap(),
    // // );
    // for token_ids in token_ids {
    //     // bar.inc(token_ids.len() as u64);
    //     out.extend(token_ids.as_ref());
    // }
    // let end_time = std::time::Instant::now();
    // println!(
    //     "Tokenized {} bytes into {} tokens in {:?}",
    //     bytes_memmapped.len(),
    //     out.len(),
    //     end_time - start_time
    // );
}
