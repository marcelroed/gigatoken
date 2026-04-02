// use icu::locale::preferences::extensions::unicode::keywords;
// '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

extern crate test;

use core::arch::x86_64::{__m512i, _mm512_permutex2var_epi8};
use std::simd::cmp::SimdPartialEq;
use std::{mem::transmute, simd as s};

// SIMD ASCII methods

// Fits in a u4
#[repr(u8)]
pub enum CharacterClass {
    Other = 0b0000,
    Number = 0b0001,
    Whitespace = 0b0010,
    Space = 0b0011,
    Apostrophe = 0b0101,
    Letter = 0b1000,
    L = 0b1001,
    V = 0b1010,
    E = 0b1011,
    R = 0b1101,
    Sdmt = 0b1100,
}

// First, build the mapping table at compile time
const fn build_ascii_class_table() -> [s::u8x64; 2] {
    let mut table = [0u8; 128];

    let mut i = 0;
    while i < 128 {
        let b = i as u8;
        table[i] = match b {
            b'0'..=b'9' => CharacterClass::Number as u8,
            b'A'..=b'Z' | b'a'..=b'z' => CharacterClass::Letter as u8,
            0x09..=0x13 => CharacterClass::Whitespace as u8,
            b' ' => CharacterClass::Space as u8,
            _ => CharacterClass::Other as u8,
        };
        i += 1;
    }

    // Overrides for special cases
    table[b'l' as usize] = CharacterClass::L as u8;
    table[b'v' as usize] = CharacterClass::V as u8;
    table[b'e' as usize] = CharacterClass::E as u8;
    table[b'\'' as usize] = CharacterClass::Apostrophe as u8;
    table[b's' as usize] = CharacterClass::Sdmt as u8;
    table[b'd' as usize] = CharacterClass::Sdmt as u8;
    table[b'm' as usize] = CharacterClass::Sdmt as u8;
    table[b't' as usize] = CharacterClass::Sdmt as u8;

    unsafe { transmute(table) }
}

const ASCII_CLASS_TABLES: [s::u8x64; 2] = build_ascii_class_table();

unsafe fn ascii_classify(bytes: s::u8x64) -> s::u8x64 {
    unsafe {
        let classes = _mm512_permutex2var_epi8(
            ASCII_CLASS_TABLES[0].into(),
            bytes.into(),
            ASCII_CLASS_TABLES[1].into(),
        );
        classes.into()
    }
}

// SIMD DFA code

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use super::*;
    use rand::{Rng, Fill, RngExt};
    use test::Bencher;

    #[test]
    fn test_ascii_classify() {
        let bytes = s::u8x64::from_array([
            b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'o',
            b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z', b'0', b'1', b'2',
            b'3', b'4', b'5', b'6', b'7', b'8', b'9', b' ', b'\t', b'\n', b'\r', b'\'', b's', b'd',
            b'm', b't', b'a', b'b', b'c', b'd', b'e', b'!', b'g', 158_u8, b'i', b'j', b'k', b'l',
            b'm', b'n', b'o', b'p', b'q', b'r', b's', b't',
        ]);
        let classes = unsafe { ascii_classify(bytes) };

        // Print the characters and their classes
        for i in 0..64 {
            let char = bytes[i] as char;
            let class = classes[i];
            let expected_class = match char as u8 {
                b's' | b'd' | b'm' | b't' => CharacterClass::Sdmt as u8,
                b'l' => CharacterClass::L as u8,
                b'v' => CharacterClass::V as u8,
                b'e' => CharacterClass::E as u8,
                b'a'..=b'z' | b'A'..=b'Z' => CharacterClass::Letter as u8,
                b'0'..=b'9' => CharacterClass::Number as u8,
                b' ' => CharacterClass::Space as u8,
                b'\t' => CharacterClass::Whitespace as u8,
                b'\n' => CharacterClass::Whitespace as u8,
                b'\r' => CharacterClass::Whitespace as u8,
                b'\'' => CharacterClass::Apostrophe as u8,
                _ => CharacterClass::Other as u8,
            };
            assert_eq!(class, expected_class);
            println!("{}: {} (expected {})", char, class, expected_class);
        }

        // assert_eq!(
        //     classes,
        //     s::u8x64::from_array([CharacterClass::Letter as u8; 26])
        // );
    }

    #[bench]
    fn bench_ascii_classify(b: &mut Bencher) {
        // Initialize a random vector of 1GB of during setup
        let mut bytes = vec![0_u8; 1_000_000_000];
        rand::rng().fill(bytes.as_mut_slice());
        for byte in &mut bytes {
            // Set the first bit to 0 in order to make the bytes ASCII
            *byte &= 0b0111_1111;
        }

        let mut output = Vec::with_capacity(bytes.len());

        b.iter(|| {
            // Iterate through chunks of 64 bytes at a time
            for (i, chunk) in bytes.chunks(64).enumerate() {
                let chunk_simd =
                    s::u8x64::from_array(unsafe { chunk.try_into().unwrap_unchecked() });
                let classes = unsafe { ascii_classify(chunk_simd) };
                output.extend(classes.to_array());
            }
            black_box(output.len());
            output.clear();
        });
    }
}
