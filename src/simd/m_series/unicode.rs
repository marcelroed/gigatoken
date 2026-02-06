//! Look up a set of indices in a table using scalar gather lookups (fastest on M-series).
//! This module constructs a table of general category lookups for all Unicode characters.
//! Hopefully, a small subset of this table will be needed and can fit in cache.

use super::table::ScalarGatherTable;



// mod tests {
//     use super::*;
//     extern crate test;
//     use test::Bencher;

//     #[bench]
//     fn bench_vpgather_table_lookup(b: &mut Bencher) {
//         // Figure out how many elements we can look up per second in a table of 8192 elements.
//         let mut table = [0; 8192];

//         use rand::Rng;
//         rand::rng().fill(&mut table);

//         /// Create a large set of random offsets
//         let all_offsets = (0..(1024 * 1024))
//             .map(|_| rand::rng().random_range(0..8192))
//             .collect::<Vec<_>>();

//         b.iter(|| {
//             let mut output = Vec::with_capacity(all_offsets.len());
//             for offsets in all_offsets.chunks(16) {
//                 let gathered =
//                     unsafe { vpgather_table_lookup(&table, offsets.try_into().unwrap_unchecked()) };
//                 output.extend(gathered.to_array());
//             }
//             std::hint::black_box(output);
//         });
//     }

//     #[bench]
//     fn bench_vpermi2b_table_lookup(b: &mut Bencher) {
//         // Figure out how many elements we can look up per second in a table of 8192 elements.
//         let mut table1 = [0; 64];
//         let mut table2 = [0; 64];
//         use rand::Rng;
//         rand::rng().fill(&mut table1);
//         rand::rng().fill(&mut table2);

//         let table1 = s::u8x64::from_array(table1);
//         let table2 = s::u8x64::from_array(table2);
//         /// Create a large set of random offsets
//         let all_offsets: Vec<u8> = (0..(1024 * 1024 * 1024))
//             .map(|_| rand::rng().random_range(0..128))
//             .collect::<Vec<_>>();

//         b.iter(|| {
//             let mut output = Vec::with_capacity(all_offsets.len());
//             for offsets in all_offsets.chunks(64) {
//                 let gathered = unsafe {
//                     vpermi2b_table_lookup(table1, table2, offsets.try_into().unwrap_unchecked())
//                 };
//                 output.extend(gathered.to_array());
//             }
//             std::hint::black_box(output);
//         });
//     }
// }
