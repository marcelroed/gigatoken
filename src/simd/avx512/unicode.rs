/// Look up a set of indices in a table using vpgatherdd from AVX512VL/AVX512F.
use core::arch::x86_64::_mm512_i32gather_epi32 as vpgatherdd;
use core::arch::x86_64::_mm512_permutex2var_epi8 as vpermi2b;
use std::simd as s;

unsafe fn vpgather_table_lookup(table: &[i32], offsets: s::i32x16) -> s::i32x16 {
    unsafe { vpgatherdd::<4>(offsets.into(), table.as_ptr()) }.into()
}

unsafe fn vpermi2b_table_lookup(table1: s::u8x64, table2: s::u8x64, indices: s::u8x64) -> s::u8x64 {
    unsafe { vpermi2b(table1.into(), indices.into(), table2.into()) }.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use test::Bencher;

    #[bench]
    fn bench_vpgather_table_lookup(b: &mut Bencher) {
        // Figure out how many elements we can look up per second in a table of 8192 elements.
        let mut table = [0; 8192];

        use rand::{Rng, RngExt};
        rand::rng().fill(&mut table);

        /// Create a large set of random offsets
        let all_offsets = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..8192))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(all_offsets.len());
            for offsets in all_offsets.chunks(16) {
                let gathered =
                    unsafe { vpgather_table_lookup(&table, offsets.try_into().unwrap_unchecked()) };
                output.extend(gathered.to_array());
            }
            std::hint::black_box(output);
        });
    }

    #[bench]
    fn bench_vpermi2b_table_lookup(b: &mut Bencher) {
        // Figure out how many elements we can look up per second in a table of 8192 elements.
        let mut table1 = [0; 64];
        let mut table2 = [0; 64];
        use rand::{Rng, RngExt};
        rand::rng().fill(&mut table1);
        rand::rng().fill(&mut table2);

        let table1 = s::u8x64::from_array(table1);
        let table2 = s::u8x64::from_array(table2);
        /// Create a large set of random offsets
        let all_offsets: Vec<u8> = (0..(1024 * 1024 * 1024))
            .map(|_| rand::rng().random_range(0..128))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(all_offsets.len());
            for offsets in all_offsets.chunks(64) {
                let gathered = unsafe {
                    vpermi2b_table_lookup(table1, table2, offsets.try_into().unwrap_unchecked())
                };
                output.extend(gathered.to_array());
            }
            std::hint::black_box(output);
        });
    }
}
