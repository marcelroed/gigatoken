// We can do a pass over the data to check for potential hash collisions.

use std::{hash::BuildHasher, time::Instant};

use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use gigatoken_rs::utils::chunks_at_utf8_boundaries;
use voracious_radix_sort::RadixSort;

pub fn main() {
    // let file = std::fs::File::open("../../data/TinyStoriesV2-GPT4-train.txt").unwrap();
    let file = std::fs::File::open("../../data/owt_train.txt").unwrap();
    let memmapped = unsafe { memmap2::Mmap::map(&file).unwrap() };

    let text = unsafe { std::str::from_utf8_unchecked(&memmapped) };

    let start = Instant::now();

    let boundaries = chunks_at_utf8_boundaries(text.as_bytes(), rayon::current_num_threads());

    let mut all_hashes = boundaries
        .into_iter()
        .tuple_windows()
        .par_bridge()
        .map(|(start, end)| {
            let mut local_hashes = vec![];
            let hasher = FxBuildHasher;
            let chunk = &text.as_bytes()[start..end];
            // let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk) };

            for word in chunk.split(|&b| b == b' ') {
                let hash = hasher.hash_one(word);
                local_hashes.push(hash);
            }
            local_hashes
        })
        .reduce(
            std::vec::Vec::new,
            |mut acc, hashes| {
                acc.extend(hashes);
                acc
            },
        );
    println!("Got hashes in: {:?}", start.elapsed());
    println!("Number of hashes: {:?}", all_hashes.len());

    let start = Instant::now();
    all_hashes.voracious_mt_sort(rayon::current_num_threads());
    println!("Time taken to sort: {:?}", start.elapsed());

    // Finally deduplicate the sorted hashes
    let deduped_hashes: Vec<u64> = all_hashes.into_iter().dedup().collect(); // dedup needs sorted input, which all_hashes is after sort

    // Count the number of unique hashes
    let num_unique = deduped_hashes.len();
    println!("Number of unique hashes: {}", num_unique);
    println!("Time taken to deduplicate: {:?}", start.elapsed());
}
