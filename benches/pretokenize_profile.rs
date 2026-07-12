//! Profiling target for the `FastR50kPretokenizer` hot loop in isolation: a plain
//! single-pass `main` (no criterion, no BPE encode) that `black_box`es every
//! yielded pretoken slice, so the slice production can't be optimized away.

use gigatoken_rs::pretokenize::FastR50kPretokenizer;
use std::hint::black_box;
use std::time::Instant;

mod common;
fn main() {
    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;

    // Feed the entire buffer to one pretokenizer in a single pass — this matches
    // the real encode path (`pretokenize_as_iter(text.as_bytes())`), which does
    // not pre-split on newlines.
    let buf: &[u8] = &input;

    eprintln!("Pretokenizing (fast_scalar, single-threaded, whole buffer)...");
    let start = Instant::now();
    let mut total_tokens: usize = 0;
    // Hand each real pretoken slice to black_box so the bounds computation can't
    // be optimized down to a counter.
    let mut iter = FastR50kPretokenizer::new(buf);
    for pretoken in iter {
        black_box(pretoken);
        total_tokens += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gb = size_gb / elapsed;

    eprintln!(
        "{total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
        throughput_gb * 1000.0
    );
}
