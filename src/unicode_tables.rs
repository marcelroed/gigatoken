use icu::collections::codepointtrie::{CodePointTrie, TrieType, TrieValue};
use std::{
    collections::HashMap,
    fmt,
    ops::{Range, RangeInclusive},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnicodeClassError;

impl fmt::Display for UnicodeClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid Unicode class")
    }
}

#[zerovec::make_ule(UnicodeClassULE)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum UnicodeClass {
    Letter = 0,
    Number = 1,
    Separator = 2,
    Other = 3,
}

impl From<char> for UnicodeClass {
    fn from(cp: char) -> Self {
        if cp.is_alphabetic() {
            UnicodeClass::Letter
        } else if cp.is_numeric() {
            UnicodeClass::Number
        } else if cp.is_whitespace() {
            UnicodeClass::Separator
        } else {
            UnicodeClass::Other
        }
    }
}

impl TrieValue for UnicodeClass {
    type TryFromU32Error = UnicodeClassError;

    fn try_from_u32(i: u32) -> Result<Self, Self::TryFromU32Error> {
        match i {
            0 => Ok(UnicodeClass::Letter),
            1 => Ok(UnicodeClass::Number),
            2 => Ok(UnicodeClass::Separator),
            3 => Ok(UnicodeClass::Other),
            _ => Err(UnicodeClassError),
        }
    }

    fn to_u32(self) -> u32 {
        match self {
            UnicodeClass::Letter => 0,
            UnicodeClass::Number => 1,
            UnicodeClass::Separator => 2,
            UnicodeClass::Other => 3,
        }
    }
}

/// Skips surrogates
fn unicode_range() -> RangeInclusive<char> {
    char::MIN..=char::MAX
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Codepoint {
    cp: char,
}

impl Codepoint {
    pub fn get_bits(self) -> Vec<bool> {
        // Encode as UTF-8 bytes, then return the bit sequence
        let mut buf = [0u8; 4];
        let bytes = self.cp.encode_utf8(&mut buf).as_bytes();

        let mut bits = Vec::with_capacity(bytes.len() * 8);

        // Collect all bits into a Vec<bool>
        for byte in bytes {
            for i in (0..8).rev() {
                bits.push((byte >> i) & 1 == 1);
            }
        }

        bits
    }
}

impl From<char> for Codepoint {
    fn from(cp: char) -> Self {
        Self { cp }
    }
}

fn build_unicode_tables(
    values_by_codepoint: &[UnicodeClass],
) -> CodePointTrie<'static, UnicodeClass> {
    use icu_codepointtrie_builder::CodePointTrieBuilder;
    let mut builder =
        CodePointTrieBuilder::new(UnicodeClass::Other, UnicodeClass::Other, TrieType::Small);
    for (cp, value) in values_by_codepoint.iter().enumerate() {
        builder.set_value(cp as u32, *value);
    }
    builder.build()
}

fn build_charset(text: &str) -> Vec<char> {
    let mut charset = vec![false; char::MAX as usize];
    for cp in text.chars() {
        charset[cp as usize] = true;
    }
    charset
        .into_iter()
        .enumerate()
        .filter_map(|(cp, b)| if b { char::from_u32(cp as u32) } else { None })
        .collect()
}

fn table_from_text(text: &str) -> CodePointTrie<'static, UnicodeClass> {
    let start_time = std::time::Instant::now();
    let charset = build_charset(text);
    let charset_time = start_time.elapsed();
    eprintln!("Time to build charset: {:?}", charset_time);
    // eprintln!("Charset: {:#?}, size: {}", charset, charset.len());
    let values_by_codepoint = charset
        .into_iter()
        .map(|cp| UnicodeClass::from(cp))
        .collect::<Vec<_>>();
    let cpt = build_unicode_tables(&values_by_codepoint[..]);
    let cpt_time = start_time.elapsed() - charset_time;
    eprintln!("Time to build table: {:?}", cpt_time);
    cpt
}

#[cfg(test)]
mod tests {
    use super::*;
    use icu::properties::CodePointMapDataBorrowed;
    use rand::{Rng, RngExt};
    extern crate test;
    use std::{
        collections::{BTreeSet, HashSet},
        fs::File,
    };
    use test::{Bencher, black_box};

    #[test]
    fn test_make_tree() {
        use memmap2::Mmap;
        // let file = File::open("/Users/marcel/data/TinyStoriesV2-GPT4-train.txt").unwrap();
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file = File::open(data_dir.join("owt_train.txt")).unwrap();
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let text = unsafe { str::from_utf8_unchecked(&mmap) };
        let table = table_from_text(text);
        // let gc: CodePointMapDataBorrowed<icu::properties::props::GeneralCategory> =
        //     icu::properties::CodePointMapData::new();
        eprintln!("Table: {:?}", table);
        // eprintln!()
        black_box(table);
    }

    // #[test]
    // fn test_tbx_lookup() {

    // }
    #[bench]
    fn bench_file_unicode_list(b: &mut Bencher) {
        // Read the file and build a set of all the codepoints that appear
        // Use memmap to read the file
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file = File::open(data_dir.join("owt_train.txt")).unwrap();

        use memmap2::Mmap;
        use std::io::Read;

        // Safety: read-only usage, file is not mutated.
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let data = &mmap[..];
        let text = unsafe { str::from_utf8_unchecked(data) };
        eprintln!("Starting");

        // For demonstration, count the number of bytes (or use `data` as needed in further benchmarks)
        b.iter(|| {
            // let mut set = BTreeSet::new();
            let mut charset = vec![false; char::MAX as usize];
            for cp in text.chars() {
                charset[cp as usize] = true;
            }
            black_box(charset);
            eprintln!("Done");
        });
    }

    #[bench]
    fn bench_scalar_lookup(b: &mut Bencher) {
        // 223 µs/iter = 4.48 GB/s
        // const TABLE_SIZE: u16 = 65_535;
        // let mut table = vec![0_u8; TABLE_SIZE as usize];
        // rand::rng().fill_bytes(&mut table);

        // let indices = (0..(1024 * 1024))
        //     .map(|_| rand::rng().random_range(0..TABLE_SIZE))
        //     .collect::<Vec<_>>();

        // b.iter(|| {
        //     let mut output = Vec::with_capacity(indices.len());
        //     for indices_chunk in indices.chunks(8) {
        //         let gathered_chunk = unsafe {
        //             table.scalar_gather_lookup(indices_chunk.try_into().unwrap_unchecked())
        //         };
        //         output.extend(gathered_chunk.to_array());
        //     }
        //     black_box(&mut output);
        // });
    }
}
