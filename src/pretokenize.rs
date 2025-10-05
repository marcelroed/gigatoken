use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::{cmp::min, io::BufRead};
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

#[derive(Clone, Debug)]
pub enum PretokenizerState {
    Start,   // Not matched anything yet
    Nonchar, // Matched some non-alphanumeric and non-whitespace characters, continue until something matching
    Apostrophe,
    AsciiSpace,
    Whitespace(u8),
    Letter,
    Number,
    Save,   // Save the current token and start a new one
    Finish, // Ran out of tokens
}

struct UTF8Iterator<'a> {
    bytes: &'a [u8],
    pos: usize,
}

enum StartResult {
    Apostrophe,
    Letter,
    Number,
    AsciiSpace,
    Whitespace(u8),
    Nonchar,
}

enum WhitespaceResult {
    AsciiSpace,
    Whitespace(u8),
    Neither,
}

enum ApostropheResult {
    Matched,
    NotMatched,
}

struct OutOfBytesError {}

impl<'a> UTF8Iterator<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn replace_bytes<'b>(&self, bytes: &'b [u8]) -> UTF8Iterator<'b> {
        UTF8Iterator {
            bytes,
            pos: self.pos,
        }
    }

    /// Returns the next codepoint as a char (u32) and its length in bytes.
    /// We need the length to rewind if it needs to be reprocessed.
    fn next_codepoint_and_length(&mut self) -> Option<(char, usize)> {
        let cp = unsafe { str::from_utf8_unchecked(&self.bytes[self.pos..]) }
            .chars()
            .next()?;
        let len = cp.len_utf8();
        self.pos += len;
        Some((cp, len))
    }

    pub fn start_check(&mut self) -> Result<StartResult, OutOfBytesError> {
        if self.pos >= self.bytes.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        if byte.is_ascii() {
            self.pos += 1;
            Ok(match byte {
                b'A'..=b'Z' | b'a'..=b'z' => StartResult::Letter,
                b' ' => StartResult::AsciiSpace,
                9..=13 => StartResult::Whitespace(1),
                b'0'..=b'9' => StartResult::Number,
                b'\'' => StartResult::Apostrophe,
                _ => StartResult::Nonchar,
            })
        } else {
            let (next_codepoint, len) =
                self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
            Ok(match next_codepoint.general_category_group() {
                GeneralCategoryGroup::Letter => StartResult::Letter,
                GeneralCategoryGroup::Number => StartResult::Number,
                GeneralCategoryGroup::Separator => StartResult::Whitespace(len as u8),
                _ => StartResult::Nonchar,
            })
        }
    }

    fn whitespace_check(&mut self) -> Result<WhitespaceResult, OutOfBytesError> {
        if self.pos >= self.bytes.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        if byte.is_ascii() {
            Ok(match byte {
                b' ' => {
                    self.pos += 1;
                    WhitespaceResult::AsciiSpace
                }
                9..=13 => {
                    self.pos += 1;
                    WhitespaceResult::Whitespace(1)
                }
                _ => WhitespaceResult::Neither,
            })
        } else {
            let (next_codepoint, len) =
                self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
            Ok(match next_codepoint.general_category_group() {
                GeneralCategoryGroup::Separator => WhitespaceResult::Whitespace(len as u8),
                _ => {
                    self.pos -= len;
                    WhitespaceResult::Neither
                }
            })
        }
    }

    fn letter_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'A'..=b'Z' | b'a'..=b'z' => {
                        self.pos += 1;
                    }
                    _ => {
                        return Ok(());
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                if next_codepoint.general_category_group() != GeneralCategoryGroup::Letter {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }

    fn number_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'0'..=b'9' => {
                        self.pos += 1;
                    }
                    _ => {
                        return Ok(());
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                if next_codepoint.general_category_group() != GeneralCategoryGroup::Number {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }
    fn other_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b' ' | 9..=13 => {
                        // Matches anything (not apostrophe though)
                        return Ok(());
                    }
                    _ => {
                        self.pos += 1;
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                if matches!(
                    next_codepoint.general_category_group(),
                    GeneralCategoryGroup::Letter
                        | GeneralCategoryGroup::Number
                        | GeneralCategoryGroup::Separator
                ) {
                    self.pos -= len;
                    return Ok(()); // We matched a letter or number, so we stop here
                }
            }
        }
    }
    pub fn apostrophe_check(&mut self) -> Result<ApostropheResult, OutOfBytesError> {
        if self.pos >= self.bytes.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        match byte {
            b's' | b'd' | b'm' | b't' => {
                self.pos += 1;
                Ok(ApostropheResult::Matched)
            }
            b'l' | b'v' | b'r' => {
                if self.pos + 1 >= self.bytes.len() {
                    return Err(OutOfBytesError {});
                }
                let next_byte = self.bytes[self.pos + 1];
                match (byte, next_byte) {
                    (b'l', b'l') | (b'v', b'e') | (b'r', b'e') => {
                        self.pos += 2;
                        Ok(ApostropheResult::Matched)
                    }
                    _ => Ok(ApostropheResult::NotMatched),
                }
            }
            _ => Ok(ApostropheResult::NotMatched),
        }
    }
}

fn find_boundaries(bytes: &[u8]) -> Vec<usize> {
    fn advance_to_boundary(input: &[u8]) -> usize {
        for (i, b) in input.iter().enumerate() {
            if *b == b'>' {
                return i + 1;
            }
        }
        panic!("No boundary found in input");
    }

    let n_threads = rayon::current_num_threads();
    eprintln!("Using {n_threads} threads for pretokenization");
    let chunk_size = bytes.len().div_ceil(n_threads);
    let mut boundaries: Vec<usize> = (0..=n_threads)
        .map(|i| min(i * chunk_size, bytes.len()))
        .collect();
    for b in boundaries[1..n_threads].iter_mut() {
        *b += advance_to_boundary(&bytes[*b..]);
    }
    boundaries
}

pub fn pretokenize_par(bytes: &[u8]) -> HashMap<&[u8], usize, rustc_hash::FxBuildHasher> {
    let start_time = std::time::Instant::now();
    let boundaries = find_boundaries(bytes);
    let merged_counts = boundaries
        .par_windows(2)
        .map(|window| {
            let start = window[0];
            let end = window[1];
            pretokenize_count(&bytes[start..end])
        })
        .reduce(
            || HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            |mut acc, counts| {
                if acc.is_empty() {
                    return counts;
                }

                for (k, v) in counts {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            },
        );

    let time_elapsed = start_time.elapsed();
    eprintln!("Pretokenization took {time_elapsed:?}");

    merged_counts
}

/// Return counts of all pretokens.
pub fn pretokenize_count(bytes: &[u8]) -> HashMap<&[u8], usize, rustc_hash::FxBuildHasher> {
    let iter = pretokenize_as_iter(bytes);

    let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
    iter.for_each(|token| {
        hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
    });
    hashmap
}

pub fn count_pretokens<'a>(
    pretoken_iter: impl Iterator<Item = &'a [u8]>,
) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
    let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
    pretoken_iter.for_each(|token| {
        hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
    });
    hashmap
}

pub fn count_pretokens_weighted<'a>(
    pretoken_weight_iter: impl Iterator<Item = (&'a [u8], usize)>,
) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
    let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
    pretoken_weight_iter.for_each(|(token, weight)| {
        hashmap
            .entry(token)
            .and_modify(|e| *e += weight)
            .or_insert(weight);
    });
    hashmap
}

pub fn pretokenize_doc_iterable<'a>(
    docs: impl Iterator<Item = &'a [u8]>,
) -> impl Iterator<Item = &'a [u8]> {
    docs.flat_map(|doc| pretokenize_as_iter(doc.as_ref()))
}

pub fn pretokenize_with_endoftext(
    bytes: &[u8],
) -> HashMap<&[u8], usize, rustc_hash::FxBuildHasher> {
    let string = unsafe { std::str::from_utf8_unchecked(bytes) };

    let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});

    string
        .split("<|endoftext|>")
        .flat_map(|part| pretokenize_as_iter(part.as_bytes()))
        .for_each(|token| {
            hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
        });

    hashmap
}

pub struct PretokenizerIter<'a> {
    iter: UTF8Iterator<'a>,
    starting: usize,
    state: PretokenizerState,
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let (state_after, new_pretoken) = loop {
            self.state = match self.state {
                PretokenizerState::Start => match self.iter.start_check() {
                    Ok(StartResult::Apostrophe) => {
                        if self.starting == self.iter.pos - 1 {
                            PretokenizerState::Apostrophe
                        } else {
                            // Only treat as apostrophe if we don't have a preceding space
                            PretokenizerState::Nonchar
                        }
                    }
                    Ok(StartResult::Letter) => PretokenizerState::Letter,
                    Ok(StartResult::Number) => PretokenizerState::Number,
                    Ok(StartResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(StartResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(StartResult::Nonchar) => PretokenizerState::Nonchar,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Save => {
                    let saved_tokens = &self.iter.bytes[self.starting..self.iter.pos];
                    self.starting = self.iter.pos;
                    break (PretokenizerState::Start, saved_tokens);
                }
                PretokenizerState::Apostrophe => match self.iter.apostrophe_check() {
                    Ok(ApostropheResult::Matched) => PretokenizerState::Save,
                    Ok(ApostropheResult::NotMatched) => PretokenizerState::Nonchar,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Nonchar => match self.iter.other_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Letter => match self.iter.letter_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Number => match self.iter.number_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Whitespace(prev_wslen) => match self.iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token =
                            &self.iter.bytes[self.starting..self.iter.pos - (prev_wslen as usize)];
                        self.starting = self.iter.pos - (prev_wslen as usize);
                        if saved_token.is_empty() {
                            PretokenizerState::Save
                        } else {
                            break (PretokenizerState::Save, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::AsciiSpace => match self.iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token = &self.iter.bytes[self.starting..self.iter.pos - 1];
                        if saved_token.is_empty() {
                            self.starting = self.iter.pos - 1;
                            PretokenizerState::Start
                        } else {
                            self.starting = self.iter.pos - 1;
                            break (PretokenizerState::Start, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => {
                        let saved_token = &self.iter.bytes[self.starting..self.iter.pos];
                        self.starting = self.iter.pos;
                        break (PretokenizerState::Finish, saved_token);
                    }
                },
                PretokenizerState::Finish => {
                    let saved_token = &self.iter.bytes[self.starting..self.iter.pos];
                    self.starting = self.iter.pos;
                    break (PretokenizerState::Finish, saved_token);
                }
            }
        };
        self.state = state_after;
        if new_pretoken.is_empty() {
            return None;
        }
        Some(new_pretoken)
    }
}

impl<'a> PretokenizerIter<'a> {
    pub fn replace_bytes<'b>(&self, bytes: &'b [u8]) -> PretokenizerIter<'b> {
        let iter = self.iter.replace_bytes(bytes);
        PretokenizerIter {
            iter,
            starting: self.starting,
            state: self.state.clone(),
        }
    }
}

impl PretokenizerIter<'static> {
    // If we contain a 'static, assume it's a dummy
    pub fn py_next<'a>(&mut self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        let mut py_self = self.replace_bytes(bytes);
        let result = py_self.next();
        *self = py_self.replace_bytes(&[]);
        result
    }
}

pub fn pretokenize_as_iter<'a>(bytes: &'a [u8]) -> PretokenizerIter<'a> {
    PretokenizerIter {
        iter: UTF8Iterator::new(bytes),
        starting: 0,
        state: PretokenizerState::Start,
    }
}

#[cfg(test)]
mod test {
    use indicatif::ProgressIterator;
    use itertools::Itertools;
    use onig::Regex;
    use rand::prelude::*;
    use std::fs;

    use super::*;

    #[test]
    fn test_pretokenizer_ts_timing() {
        let file_bytes = fs::read(
            "/home/marcel/projects/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        )
        .unwrap();

        let pretokenized_counts = pretokenize_par(&file_bytes);
        eprintln!("Pretokenized {} tokens", pretokenized_counts.len());
        // eprintln!("Pretokenized counts: {:?}", pretokenized_counts);
        // Print counts sorted by frequency
    }

    #[test]
    fn test_pretokenizer_matches_regex() {
        let re = Regex::new(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            // onig::RegexOptions::REGEX_OPTION_NONE,
            // onig::Syntax::oniguruma(),
        )
        .unwrap();
        // let re = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        //     .unwrap();
        let input = fs::read(
            "/home/marcel/projects/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        )
        .unwrap();

        // let input = input[..10_000_000].to_vec();

        // let pretokens = pretokenize_as_list(&input);
        // let mut last_match: Option<(usize, usize)> = None;
        for _ in (0..100).progress() {
            let mut previous_tokens: Vec<(String, String)> = vec![];

            const WINDOW_SIZE: usize = 1_000_000;
            let start = rand::rng().random_range(0..input.len() - WINDOW_SIZE);
            let input = input[start..start + WINDOW_SIZE].to_vec();
            let pretokens_iterator = pretokenize_as_iter(&input);
            let re_iterator = re.find_iter(str::from_utf8(&input).unwrap());
            for (token_idx, eorb) in pretokens_iterator.zip_longest(re_iterator).enumerate() {
                let (token, (start, end)) = match eorb {
                    itertools::EitherOrBoth::Both(first, second) => (first, second),
                    itertools::EitherOrBoth::Left(first) => panic!(
                        "No match found for token {token_idx} at bytes {first:?}, {:?}, {:?}",
                        str::from_utf8(&input[input.len().saturating_sub(10)..]).unwrap(),
                        &previous_tokens[previous_tokens.len().saturating_sub(10)..]
                    ),
                    itertools::EitherOrBoth::Right(second) => {
                        panic!("No token found for match {token_idx} at byte {second:?}")
                    }
                };
                // last_match = Some((start, end));
                // let (&token, (start, end)) = eorb.both().unwrap();
                let token_str = String::from_utf8_lossy(token).into_owned();
                let match_str = String::from_utf8_lossy(&input[start..end]).into_owned();
                previous_tokens.push((token_str.clone(), match_str.clone()));
                // if pretokens.len() > 1000 {
                //     pretokens.truncate(1000);
                // }
                assert_eq!(
                    token_str,
                    match_str,
                    "Token {token_idx} (byte {start}) does not match regex, see last few {:?}\n Byte representation: {:02X?}{:02X?}\nExtended{:02X?}",
                    &previous_tokens[previous_tokens.len().saturating_sub(10)..],
                    token,
                    &input[start..end],
                    &input[start.saturating_sub(5)..end + 5]
                );
            }
        }
    }

    #[test]
    fn test_pretokenizer_ts() {
        let file_bytes = fs::read("../../data/TinyStoriesV2-GPT4-train.txt").unwrap();

        // let pretokenized_counts = pretokenize_par(&file_bytes);
        let pretokenized_counts = pretokenize_as_iter(&file_bytes)
            .progress_count(644752805)
            .counts();
        eprintln!("Pretokenized {} tokens", pretokenized_counts.len());
        // eprintln!("Pretokenized counts: {:?}", pretokenized_counts);
        // Print counts sorted by frequency
        let mut sorted_counts: Vec<_> = pretokenized_counts.iter().collect();
        sorted_counts.sort_by_key(|&(_, &v)| v);
        sorted_counts.reverse();
        for &(&token, &count) in sorted_counts.iter().take(100) {
            eprintln!("{1}: {0}", String::from_utf8_lossy(token), count);
        }
    }

    #[test]
    fn test_pretokenizer_no_count_ts() {
        let file_bytes = fs::read(
            "/home/marcel/projects/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        )
        .unwrap();

        // let pretokenized_counts = pretokenize_par(&file_bytes);
        let pretokens = pretokenize_as_iter(&file_bytes)
            .progress_count(644752805)
            .count();
        eprintln!("Pretokenized {} tokens", pretokens);
    }

    /// Make sure the total number of pretokens matches Python regex
    #[test]
    fn test_pretokenizer_ts_length() {
        let file_bytes = fs::read(
            "/home/marcel/projects/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        )
        .unwrap();

        let pretokens_count = pretokenize_as_iter(&file_bytes).count();
        eprintln!("Pretokenized {pretokens_count} tokens");
        // Check that the total length of all tokens is equal to the input length
        assert_eq!(
            pretokens_count, 544752805,
            "Total number of pretokens does not match expected count"
        );
    }

    #[test]
    fn test_pretokenizer_owt_length() {
        let file_bytes =
            fs::read("/home/marcel/projects/spring2024-assignment1-basics/data/owt_train.txt")
                .unwrap();

        let pretokens_count = pretokenize_as_iter(&file_bytes).count();
        eprintln!("Pretokenized {pretokens_count} tokens");
        // Check that the total length of all tokens is equal to the input length
        // assert_eq!(pretokens_count, 544752805, "Total number of pretokens does not match expected count");
        assert_eq!(
            pretokens_count, 123456789,
            "Total number of pretokens does not match expected count"
        );
    }

    #[test]
    fn minimal_tokenization() {
        let input = vec![0x74, 0x75, 0x72, 0x65, 0x73, 0x2E, 0xC2, 0xA0, 0x0A, 0x4F];
        let pretokens: Vec<_> = pretokenize_as_iter(&input).collect();
        let re =
            Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
                .unwrap();

        for (&token, (start, end)) in pretokens
            .iter()
            .zip(re.find_iter(str::from_utf8(&input).unwrap()))
        {
            let token_str = String::from_utf8_lossy(token).into_owned();
            let match_str = String::from_utf8_lossy(&input[start..end]).into_owned();
            assert_eq!(token_str, match_str, "Token does not match regex");
        }
    }
}
