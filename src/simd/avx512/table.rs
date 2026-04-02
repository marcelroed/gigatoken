use std::arch::x86_64::{
    __m512i, _mm512_cvtepi32_epi8, _mm512_i32gather_epi32, _mm512_mask_blend_epi8,
    _mm512_movepi8_mask, _mm512_permutex2var_epi8,
};
use std::{mem::transmute, simd as s};

pub trait VPermTable {
    unsafe fn vperm_lookup(self, indices: s::u8x64) -> s::u8x64;
}

impl VPermTable for [u8; 128] {
    unsafe fn vperm_lookup(self, indices: s::u8x64) -> s::u8x64 {
        unsafe {
            let tables: [s::u8x64; 2] = transmute(self);
            // Call vpermi2b
            let results =
                _mm512_permutex2var_epi8(tables[0].into(), indices.into(), tables[1].into());
            results.into()
        }
    }
}

impl VPermTable for [u8; 256] {
    unsafe fn vperm_lookup(self, indices: s::u8x64) -> s::u8x64 {
        unsafe {
            let tables: [[u8; 128]; 2] = transmute(self);
            let mask = _mm512_movepi8_mask(indices.into());
            let first_lookup = tables[0].vperm_lookup(indices);
            let second_lookup = tables[1].vperm_lookup(indices);

            let final_lookup =
                _mm512_mask_blend_epi8(mask, first_lookup.into(), second_lookup.into());
            final_lookup.into()
        }
    }
}

pub unsafe fn vperm_table_lookup(indices: s::u8x64, table: impl VPermTable) -> s::u8x64 {
    unsafe { table.vperm_lookup(indices) }
}

pub trait VPGatherTable {
    type Element: s::SimdElement;
    unsafe fn vpgather_lookup(self, indices: s::i32x16) -> s::Simd<Self::Element, 16>;
}

impl VPGatherTable for &[u8] {
    type Element = u8;
    unsafe fn vpgather_lookup(self, indices: s::i32x16) -> s::Simd<Self::Element, 16> {
        unsafe {
            let i32_results = _mm512_i32gather_epi32(indices.into(), transmute(self.as_ptr()), 1);
            let u8_results = _mm512_cvtepi32_epi8(i32_results);
            u8_results.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, Fill, RngExt};
    extern crate test;
    use test::{Bencher, black_box};

    #[test]
    fn test_vpgather_lookup() {
        let mut table = vec![0_u8; 8192];
        rand::rng().fill(table.as_mut_slice());

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..8192))
            .collect::<Vec<_>>();

        let mut gathered = vec![];
        for indices_chunk in indices.chunks(16) {
            let gathered_chunk =
                unsafe { table.vpgather_lookup(indices_chunk.try_into().unwrap_unchecked()) };
            gathered.extend(gathered_chunk.to_array());
        }

        let expected = indices
            .iter()
            .map(|&i| table[i as usize])
            .collect::<Vec<_>>();

        assert_eq!(gathered, expected);
    }

    #[bench]
    fn bench_vpgather_lookup(b: &mut Bencher) {
        let mut table = vec![0_u8; 131_072];
        rand::rng().fill(table.as_mut_slice());

        let indices = (0..(1024 * 1024))
            .map(|_| rand::rng().random_range(0..131_072))
            .collect::<Vec<_>>();

        b.iter(|| {
            let mut output = Vec::with_capacity(indices.len());
            for indices_chunk in indices.chunks(16) {
                let gathered_chunk =
                    unsafe { table.vpgather_lookup(indices_chunk.try_into().unwrap_unchecked()) };
                output.extend(gathered_chunk.to_array());
            }
            black_box(&mut output);
            output.clear();
        });
    }
}
