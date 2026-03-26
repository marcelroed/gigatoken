use std::arch::aarch64::{uint8x16x4_t, vdupq_n_u8, vgetq_lane_u8, vqtbx4q_u8, vsubq_u8};
use std::{mem::transmute, simd as s};

pub trait TbxTable {
    unsafe fn tbx_lookup(self, indices: s::u8x16) -> s::u8x16;
}

impl TbxTable for [u8; 64] {
    unsafe fn tbx_lookup(self, indices: s::u8x16) -> s::u8x16 {
        unsafe {
            let table: uint8x16x4_t = transmute(self);

            // Call tbx instruction
            let results = vqtbx4q_u8(indices.into(), table, indices.into());

            results.into()
        }
    }
}

impl TbxTable for [u8; 128] {
    unsafe fn tbx_lookup(self, indices: s::u8x16) -> s::u8x16 {
        unsafe {
            let offset = vdupq_n_u8(64);
            let idx_high = vsubq_u8(indices.into(), offset);

            let tables: [[u8; 64]; 2] = transmute(self);
            let table_high: uint8x16x4_t = transmute(tables[1]);
            let table_low: uint8x16x4_t = transmute(tables[0]);

            let zeros = vdupq_n_u8(0);
            let res_high = vqtbx4q_u8(zeros, table_high, idx_high);
            let result = vqtbx4q_u8(res_high, table_low, indices.into());
            result.into()
        }
    }
}

pub trait ScalarGatherTable {
    type Element: s::SimdElement;
    unsafe fn scalar_gather_lookup(self, indices: s::u16x8) -> s::Simd<Self::Element, 8>;
}

// Using loads directly to the vector register (ld1.b)

// impl ScalarGatherTable for &[u8] {
//     type Element = u8;
//     #[unsafe(no_mangle)]
//     unsafe fn scalar_gather_lookup(self, indices: s::u16x8) -> s::Simd<Self::Element, 8> {
//         unsafe {
//             let base = self.as_ptr();
//             let v0 = *base.add(indices[0] as usize);
//             let v1 = *base.add(indices[1] as usize);
//             let v2 = *base.add(indices[2] as usize);
//             let v3 = *base.add(indices[3] as usize);
//             let v4 = *base.add(indices[4] as usize);
//             let v5 = *base.add(indices[5] as usize);
//             let v6 = *base.add(indices[6] as usize);
//             let v7 = *base.add(indices[7] as usize);
//             s::Simd::from_array([v0, v1, v2, v3, v4, v5, v6, v7])
//         }
//     }
// }

// Using only scalar register loads (ldrb)

impl ScalarGatherTable for &[u8] {
    type Element = u8;
    // #[unsafe(no_mangle)]
    unsafe fn scalar_gather_lookup(self, indices: s::u16x8) -> s::Simd<Self::Element, 8> {
        unsafe {
            let base = self.as_ptr();

            // 1. Load as u64 to force usage of Integer registers (x-regs)
            //    Since we are shifting, the compiler will use ldrb + lsl
            let b0 = *base.add(indices[0] as usize) as u64;
            let b1 = *base.add(indices[1] as usize) as u64;
            let b2 = *base.add(indices[2] as usize) as u64;
            let b3 = *base.add(indices[3] as usize) as u64;
            let b4 = *base.add(indices[4] as usize) as u64;
            let b5 = *base.add(indices[5] as usize) as u64;
            let b6 = *base.add(indices[6] as usize) as u64;
            let b7 = *base.add(indices[7] as usize) as u64;

            // 2. Combine into one 64-bit integer
            //    This utilizes the M4's massive superscalar integer ALUs.
            //    Little-endian structure: index 0 is LSB.
            let packed = b0
                | (b1 << 8)
                | (b2 << 16)
                | (b3 << 24)
                | (b4 << 32)
                | (b5 << 40)
                | (b6 << 48)
                | (b7 << 56);

            // 3. Move from GPR to Vector Unit (fmov d0, xN) - 1 cycle latency
            transmute(packed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngExt};
    extern crate test;
    use test::{Bencher, black_box};

    #[test]
    fn test_tbx_lookup() {
        let mut table = [0_u8; 128];
        rand::rng().fill(&mut table);

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..128))
            .collect::<Vec<_>>();

        let mut gathered = vec![];
        for indices_chunk in indices.chunks(16) {
            let gathered_chunk =
                unsafe { table.tbx_lookup(indices_chunk.try_into().unwrap_unchecked()) };
            gathered.extend(gathered_chunk.to_array());
        }

        let expected = indices
            .iter()
            .map(|&i| table[i as usize])
            .collect::<Vec<_>>();

        assert_eq!(gathered, expected);
    }

    #[test]
    fn test_scalar_lookup() {
        let mut table = vec![0_u8; 8192];
        rand::rng().fill_bytes(&mut table);

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..8192))
            .collect::<Vec<_>>();

        let mut gathered = vec![];
        for indices_chunk in indices.chunks(8) {
            let gathered_chunk =
                unsafe { table.scalar_gather_lookup(indices_chunk.try_into().unwrap_unchecked()) };
            gathered.extend(gathered_chunk.to_array());
        }

        let expected = indices
            .iter()
            .map(|&i| table[i as usize])
            .collect::<Vec<_>>();

        assert_eq!(gathered, expected);
    }

    #[bench]
    fn bench_scalar_lookup(b: &mut Bencher) {
        // 223 µs/iter = 4.48 GB/s
        const TABLE_SIZE: u16 = 65_535;
        let mut table = vec![0_u8; TABLE_SIZE as usize];
        rand::rng().fill_bytes(&mut table);

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..TABLE_SIZE))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(indices.len());
            for indices_chunk in indices.chunks(8) {
                let gathered_chunk = unsafe {
                    table.scalar_gather_lookup(indices_chunk.try_into().unwrap_unchecked())
                };
                output.extend(gathered_chunk.to_array());
            }
            black_box(&mut output);
        });
    }

    #[bench]
    fn bench_regular_lookup(b: &mut Bencher) {
        // 529 µs/iter = 1.89 GB/s
        const TABLE_SIZE: u16 = 65_535;
        // const TABLE_SIZE: u16 = 1024;
        let mut table = vec![0_u8; TABLE_SIZE as usize];
        rand::rng().fill_bytes(&mut table);

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..TABLE_SIZE))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(indices.len());
            for &index in indices.iter() {
                let gathered = unsafe { table.get_unchecked(index as usize) };
                output.push(gathered);
            }
            black_box(&mut output);
        });
    }

    #[bench]
    fn bench_tbx_lookup(b: &mut Bencher) {
        // 161 µs/iter = 6.20 GB/s
        const TABLE_SIZE: u8 = 128;
        let mut table = [0_u8; TABLE_SIZE as usize];
        rand::rng().fill(&mut table);

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..TABLE_SIZE))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(indices.len());
            for indices_chunk in indices.chunks(16) {
                let gathered_chunk =
                    unsafe { table.tbx_lookup(indices_chunk.try_into().unwrap_unchecked()) };
                output.extend(gathered_chunk.to_array());
            }
            black_box(&mut output);
        });
    }
}
