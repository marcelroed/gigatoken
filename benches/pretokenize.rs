use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
use gigatoken_rs::pretokenize::pretoken_avx512::Avx512PretokenizerIter;
use gigatoken_rs::pretokenize::{
    pretoken_combinator::pretokens_iterator, FastCl100kPretokenizer, FastQwen2Pretokenizer,
    FastQwen35Pretokenizer, FastR50kPretokenizer, PretokenizerIter,
};
use gigatoken_rs::pretokenize::pretoken_simd::SimdPretokIter;
use std::hint::black_box;

const TARGET_BENCH_SIZE: usize = 100_000_000; // ~100 MB

/// Load OWT data, truncated to a UTF-8-safe boundary near `max_bytes`.
fn load_owt(max_bytes: usize) -> Vec<u8> {
    let data_dir = std::env::home_dir().unwrap().join("data");
    let all_bytes =
        std::fs::read(data_dir.join("owt_train.txt")).expect("Could not read ~/data/owt_train.txt");
    let mut end = max_bytes.min(all_bytes.len());
    // Back up to a UTF-8 character boundary
    while end > 0 && !std::str::from_utf8(&all_bytes[..end]).is_ok() {
        end -= 1;
    }
    all_bytes[..end].to_vec()
}

fn pretokenize_benches(c: &mut Criterion) {
    let input = load_owt(TARGET_BENCH_SIZE);
    let input_len = input.len() as u64;
    eprintln!("Benchmark input size: {:.1} MB", input_len as f64 / 1e6);

    let mut group = c.benchmark_group("pretokenize");
    group.throughput(Throughput::Bytes(input_len));
    group.sample_size(10);

    group.bench_function("state_machine", |b| {
        b.iter(|| {
            let count = PretokenizerIter::new(&input).count();
            black_box(count);
        });
    });

    group.bench_function("winnow", |b| {
        b.iter(|| {
            let mut input_str = unsafe { std::str::from_utf8_unchecked(&input) };
            let count = pretokens_iterator(&mut input_str).count();
            black_box(count);
        });
    });

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vl"
    ))]
    group.bench_function("avx512", |b| {
        b.iter(|| {
            let mut iter = Avx512PretokenizerIter::new(&input);
            let mut count = 0;
            while iter.next().is_some() {
                count += 1;
            }
            black_box(count);
        });
    });
    group.bench_function("simd", |b| {
        b.iter(|| {
            let count = SimdPretokIter::new(&input).count();
            black_box(count);
        });
    });

    group.bench_function("fast_scalar", |b| {
        b.iter(|| {
            let mut iter = FastR50kPretokenizer::new(&input);
            let mut count = 0;
            while iter.next().is_some() {
                count += 1;
            }
            black_box(count);
        });
    });

    group.bench_function("cl100k_fast_scalar", |b| {
        b.iter(|| {
            let mut iter = FastCl100kPretokenizer::new(&input);
            let mut count = 0;
            while iter.next().is_some() {
                count += 1;
            }
            black_box(count);
        });
    });

    group.bench_function("qwen2_fast_scalar", |b| {
        b.iter(|| {
            let mut iter = FastQwen2Pretokenizer::new(&input);
            let mut count = 0;
            while iter.next().is_some() {
                count += 1;
            }
            black_box(count);
        });
    });

    group.bench_function("qwen3_5_fast_scalar", |b| {
        b.iter(|| {
            let mut iter = FastQwen35Pretokenizer::new(&input);
            let mut count = 0;
            while iter.next().is_some() {
                count += 1;
            }
            black_box(count);
        });
    });

    let re = fancy_regex::Regex::new(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )
    .unwrap();

    group.bench_function("regex", |b| {
        b.iter(|| {
            let text = unsafe { std::str::from_utf8_unchecked(&input) };
            let count = re.find_iter(text).count();
            black_box(count);
        });
    });

    // Backtracking-compatible equivalent of the possessive cl100k pattern
    let re_cl100k = fancy_regex::Regex::new(
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s+",
    )
    .unwrap();

    group.bench_function("cl100k_regex", |b| {
        b.iter(|| {
            let text = unsafe { std::str::from_utf8_unchecked(&input) };
            let count = re_cl100k.find_iter(text).count();
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(benches, pretokenize_benches);
criterion_main!(benches);
