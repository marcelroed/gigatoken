use std::collections::HashMap;
use std::time::Instant;

use rustc_hash::FxBuildHasher;

fn main() {
    let file = std::fs::File::open("../../data/TinyStoriesV2-GPT4-train.txt").unwrap();

    let memmapped = unsafe { memmap2::Mmap::map(&file).unwrap() };

    let text = unsafe { std::str::from_utf8_unchecked(&memmapped) };

    let start = Instant::now();

    // let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut counts: HashMap<&[u8], usize, FxBuildHasher> =
        HashMap::with_hasher(FxBuildHasher);

    // Count words in the file
    // I have tested, and this is faster than looping explicitly and checking for space bytes.
    for word in text.as_bytes().split(|&b| b == b' ') {
        *counts.entry(word).or_default() += 1;
    }

    println!("Time taken: {:?}", start.elapsed());
    println!("Number of words: {:?}", counts.len());
}
