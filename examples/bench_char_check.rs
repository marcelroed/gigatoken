use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use itertools::Itertools;

pub fn main() {
    // let file = std::fs::File::open("../../data/TinyStoriesV2-GPT4-train.txt").unwrap();
    let file = std::fs::File::open("../../data/owt_train.txt").unwrap();

    let memmapped = unsafe { memmap2::Mmap::map(&file).unwrap() };

    let text = unsafe { std::str::from_utf8_unchecked(&memmapped) };

    let start = Instant::now();

    let all_chars = [const { AtomicBool::new(false) }; char::MAX as usize];

    use rayon::prelude::*;
    // const N_THREADS: usize = 8; // You can set this to rayon::current_num_threads() if desired
    let n_threads = rayon::current_num_threads();

    let text_len = text.len();
    let chunk_size = (text_len + n_threads - 1) / n_threads;
    // To avoid splitting a character in the middle of a multi-byte UTF-8 sequence,
    // we'll chunk by byte and correct the boundaries.
    let bytes = text.as_bytes();

    // Use the chunks_at_utf8_boundaries function from utils.rs
    use toker_rs::utils::chunks_at_utf8_boundaries;

    let boundaries = if bytes.len() > 100_000 {
        chunks_at_utf8_boundaries(bytes, n_threads)
    } else {
        vec![0, bytes.len()] // Default to no parallelism for short inputs
    };

    (boundaries)
        .into_iter()
        .tuple_windows()
        .par_bridge()
        .for_each(|(chunk_start, chunk_end)| {
            let chunk_str =
                unsafe { std::str::from_utf8_unchecked(&bytes[chunk_start..chunk_end]) };

            for c in chunk_str.chars() {
                all_chars[c as usize].store(true, Ordering::Relaxed);
            }
        });

    println!("Time taken: {:?}", start.elapsed());
    println!(
        "Number of unique chars: {:?}",
        all_chars
            .iter()
            .filter(|b| b.load(Ordering::Relaxed))
            .count()
    );
}
