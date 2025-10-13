use criterion::{Criterion, criterion_group, criterion_main};
use icu::properties::{CodePointMapDataBorrowed, props::EnumeratedProperty};
use std::hint::black_box;

use rand::{self, Rng};

pub fn fibonacci(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let c = a + b;
        a = b;
        b = c;
    }
    a
}

// Removed dependency since icu is ~95% faster
// use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};
// pub fn unicode_properties_classify(c: char) -> bool {
//     c.general_category_group() == GeneralCategoryGroup::Letter
// }

pub fn icu4x_classify(c: char) -> bool {
    icu::properties::props::GeneralCategoryGroup::Letter
        .contains(icu::properties::props::GeneralCategory::for_char(c))
}

pub fn icu4x_classify_table(c: char) -> bool {
    let gc: CodePointMapDataBorrowed<icu::properties::props::GeneralCategory> =
        icu::properties::CodePointMapData::new();
    icu::properties::props::GeneralCategoryGroup::Letter.contains(gc.get(c))
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
    let mut group = c.benchmark_group("unicode_classify");

    let chars_input: Vec<char> = rand::rng()
        .sample_iter::<char, _>(rand::distr::StandardUniform)
        .take(4096)
        .collect();
    group.bench_with_input("icu4x", chars_input.as_slice(), |b, chars: &[char]| {
        b.iter(|| {
            for c in chars {
                icu4x_classify(*c);
            }
        });
    });
    // group.bench_with_input(
    //     "unicode_properties",
    //     chars_input.as_slice(),
    //     |b, chars: &[char]| {
    //         b.iter(|| {
    //             for c in chars {
    //                 unicode_properties_classify(*c);
    //             }
    //         });
    //     },
    // );
    group.bench_with_input(
        "icu4x table",
        chars_input.as_slice(),
        |b, chars: &[char]| {
            b.iter(|| {
                for c in chars {
                    icu4x_classify_table(*c);
                }
            });
        },
    );

    // c.bench_function("unicode classify letter", |b| {
    //     b.iter_batched(
    //         || {},
    //         |chars: Vec<char>| {
    //             for c in chars {
    //                 unicode_classify(c);
    //             }
    //         },
    //         criterion::BatchSize::SmallInput,
    //     )
    // });
    // c.bench_function("unicode pretokenize")
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
