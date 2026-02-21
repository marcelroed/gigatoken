use criterion::{Criterion, criterion_group, criterion_main};
use icu::properties::{CodePointMapDataBorrowed, props::EnumeratedProperty};
use regex::bytes;
use std::hint::black_box;
use toker_rs::pretokenize::{PretokenizerIter, pretoken_combinator::pretokens_iterator};

use rand::{self, Rng};

fn state_machine_pretokenize(input: &[u8]) -> Vec<&[u8]> {
    let mut iter = PretokenizerIter::new(input);
    let mut v = vec![];
    iter.map(|pretoken| pretoken).for_each(|pretoken| {
        v.push(pretoken.0);
    });
    v
}

fn winnow_pretokenize(input: &[u8]) -> Vec<&[u8]> {
    let mut iter = pretokens_iterator(unsafe { std::str::from_utf8_unchecked(input) });
    let mut v = vec![];
    iter.for_each(|pretoken| {
        v.push(pretoken.0);
    });
    // assert!(iter.finish().is_ok());
    v
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
    let mut group = c.benchmark_group("unicode_classify");

    // let chars_input: Vec<char> = rand::rng()
    //     .sample_iter::<char, _>(rand::distr::StandardUniform)
    //     .take(4096)
    //     .collect();
    let data_dir = std::env::home_dir().unwrap().join("data");
    let ts_input = std::fs::read_to_string(data_dir.join("TinyStoriesV2-GPT4-valid.txt")).unwrap();
    let bytes_input = ts_input.as_bytes();

    group.bench_with_input("winnow", bytes_input, |b, bytes: &[u8]| {
        b.iter(|| {
            let mut pretokens = winnow_pretokenize(bytes);
            let total_len = black_box(pretokens).len();
            black_box(total_len);
        });
    });
    group.bench_with_input("state machine", bytes_input, |b, bytes: &[u8]| {
        b.iter(|| {
            let mut pretokens = state_machine_pretokenize(bytes);
            let total_len = black_box(pretokens).len();
            black_box(total_len);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
