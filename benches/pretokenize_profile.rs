//! Profiling target for the `FastR50kPretokenizer` hot loop in isolation: a plain
//! single-pass `main` (no criterion, no BPE encode) that `black_box`es every
//! yielded pretoken slice, so the slice production can't be optimized away.

use gigatok_rs::pretokenize::FastR50kPretokenizer;
use std::hint::black_box;
use std::time::Instant;

/// Load the benchmark input from `data/owt_train.txt`.
///
/// Optionally caps the input for fast profiling iterations. Set ENCODE_MB to
/// the desired number of megabytes (e.g. ENCODE_MB=500). When set, only that
/// many bytes are read from disk (so the read does not dominate a profile of
/// the pretokenize loop).
fn load_input() -> Vec<u8> {
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");
    eprintln!("Reading {owt_path:?}...");
    let t0 = Instant::now();

    let input = match std::env::var("ENCODE_MB") {
        Ok(mb) => {
            use std::io::Read;
            let max_bytes = mb
                .trim()
                .parse::<usize>()
                .expect("ENCODE_MB must be an integer")
                * 1_000_000;
            let file = std::fs::File::open(&owt_path).expect("Could not open ~/data/owt_train.txt");
            let mut data = Vec::with_capacity(max_bytes);
            file.take(max_bytes as u64)
                .read_to_end(&mut data)
                .expect("read failed");
            // Back up to a UTF-8 character boundary.
            let mut end = data.len();
            while end > 0 && std::str::from_utf8(&data[..end]).is_err() {
                end -= 1;
            }
            data.truncate(end);
            eprintln!(
                "Capped input to {} MB (ENCODE_MB={mb})",
                data.len() / 1_000_000
            );
            data
        }
        Err(_) => std::fs::read(&owt_path).expect("Could not read ~/data/owt_train.txt"),
    };

    let size_gb = input.len() as f64 / 1e9;
    eprintln!(
        "Read {:.2} GB in {:.1}s",
        size_gb,
        t0.elapsed().as_secs_f64()
    );
    input
}

fn main() {
    let input = load_input();
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
