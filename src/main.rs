mod bpe_train;
mod pretokenize;

pub fn main() {
    // Get args (path to file, vocab size)
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_file> <vocab_size>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let vocab_size: usize = args[2].parse().expect("Invalid vocab size");

    let bpe_result = bpe_train::train_bpe(
        bpe_train::PretokenizeableSpec::Parquet(input_file.into()),
        vocab_size,
        vec![],
    );
}
