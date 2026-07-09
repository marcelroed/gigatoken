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
    let chunk_size = text_len.div_ceil(n_threads);
    // To avoid splitting a character in the middle of a multi-byte UTF-8 sequence,
    // we'll chunk by byte and correct the boundaries.
    let bytes = text.as_bytes();

    // Use the chunks_at_utf8_boundaries function from utils.rs
    use gigatok_rs::utils::chunks_at_utf8_boundaries;

    let boundaries = if bytes.len() > 100_000 {
        chunks_at_utf8_boundaries(bytes, n_threads)
    } else {
        vec![0, bytes.len()] // Default to no parallelism for short inputs
    };

    (boundaries.clone())
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

    let elapsed = start.elapsed();
    let mb = memmapped.len() as f64 / 1e6;
    println!("--- std chars() ---");
    println!("Time taken: {elapsed:?}");
    println!("{:.0} MB/s", mb / elapsed.as_secs_f64());
    println!(
        "Number of unique chars: {:?}",
        all_chars
            .iter()
            .filter(|b| b.load(Ordering::Relaxed))
            .count()
    );

    // --- simdutf version ---
    let all_chars_simd = [const { AtomicBool::new(false) }; char::MAX as usize];

    let start2 = Instant::now();

    // Process in small sub-chunks that fit in L2 cache (~128 KB per core on M4).
    // Input sub-chunk of 32 KB expands to at most 128 KB of UTF-32 output.
    const SUB_CHUNK: usize = 32 * 1024;

    (boundaries)
        .into_iter()
        .tuple_windows()
        .par_bridge()
        .for_each(|(chunk_start, chunk_end)| {
            let thread_chunk = &bytes[chunk_start..chunk_end];
            // Pre-allocate one small reusable buffer per thread.
            let mut dst = vec![0u32; SUB_CHUNK]; // worst case: all ASCII, 1:1

            let mut offset = 0;
            while offset < thread_chunk.len() {
                let mut end = (offset + SUB_CHUNK).min(thread_chunk.len());
                // Align to UTF-8 boundary.
                while end < thread_chunk.len()
                    && thread_chunk[end] & 0b1100_0000 == 0b1000_0000
                {
                    end += 1;
                }
                let sub = &thread_chunk[offset..end];
                let dst_needed = simdutf::utf32_length_from_utf8(sub);
                if dst_needed > dst.len() {
                    dst.resize(dst_needed, 0);
                }
                let written = unsafe {
                    simdutf::convert_valid_utf8_to_utf32(sub.as_ptr(), sub.len(), dst.as_mut_ptr())
                };
                for &cp in &dst[..written] {
                    all_chars_simd[cp as usize].store(true, Ordering::Relaxed);
                }
                offset = end;
            }
        });

    let elapsed2 = start2.elapsed();
    println!("\n--- simdutf utf8->utf32 ---");
    println!("Time taken: {elapsed2:?}");
    println!("{:.0} MB/s", mb / elapsed2.as_secs_f64());
    println!(
        "Number of unique chars: {:?}",
        all_chars_simd
            .iter()
            .filter(|b| b.load(Ordering::Relaxed))
            .count()
    );
}
