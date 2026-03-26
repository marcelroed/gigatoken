//! The pretokenizer is responsible for taking a single document and producing an iterator of
//! pretokens.
use crate::bpe_train::PretokenizeableSpec;
use crate::input::DocRef;
pub(crate) use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::pretokenize_traits::{
    ParallelMergeCounts, ParallelPretokenCountable, PretokenCountable,
};
use itertools::Itertools;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashMap;

mod options;
mod pretoken;
mod pretoken_chunks;
pub mod pretoken_combinator;
mod pretokenize_traits;
mod simd;
mod unicode;

pub use options::PretokenizerType;

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

pub struct UTF8Iterator<'a> {
    bytes: DocRef<'a>,
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

pub(crate) struct OutOfBytesError {}

impl<'a> UTF8Iterator<'a> {
    pub(crate) fn new(doc: DocRef<'a>) -> Self {
        Self { bytes: doc, pos: 0 }
    }

    pub fn replace_bytes<'b>(&self, bytes: &'b [u8]) -> UTF8Iterator<'b> {
        UTF8Iterator {
            bytes: bytes.into(),
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

    // #[inline(never)]
    fn start_check(&mut self) -> Result<StartResult, OutOfBytesError> {
        if self.pos >= self.bytes.0.len() {
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
            let gc = unicode::get_general_category(next_codepoint);
            Ok(if unicode::is_gc_letter(gc) {
                StartResult::Letter
            } else if unicode::is_gc_number(gc) {
                StartResult::Number
            } else if unicode::is_gc_separator(gc) {
                StartResult::Whitespace(len as u8)
            } else {
                StartResult::Nonchar
            })
        }
    }

    // #[inline(never)]
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
            Ok(if unicode::is_separator(next_codepoint) {
                WhitespaceResult::Whitespace(len as u8)
            } else {
                self.pos -= len;
                WhitespaceResult::Neither
            })
        }
    }

    // #[inline(never)]
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
                if !unicode::is_letter(next_codepoint) {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }

    // #[inline(never)]
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
                if !unicode::is_number(next_codepoint) {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }
    // #[inline(never)]
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
                let gc = unicode::get_general_category(next_codepoint);
                if unicode::is_gc_letter(gc)
                    || unicode::is_gc_number(gc)
                    || unicode::is_gc_separator(gc)
                {
                    self.pos -= len;
                    return Ok(()); // We matched a letter or number, so we stop here
                }
            }
        }
    }
    // #[inline(never)]
    fn apostrophe_check(&mut self) -> Result<ApostropheResult, OutOfBytesError> {
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

pub fn find_boundaries(bytes: &[u8]) -> Vec<usize> {
    fn advance_to_boundary(input: &[u8]) -> usize {
        for (i, (first, second)) in input.iter().tuple_windows().enumerate() {
            if matches!((first, second), (b'.', b' ')) {
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

pub fn pretokenize_par_bytes(
    bytes: &[u8],
) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let start_time = std::time::Instant::now();
    let boundaries = find_boundaries(bytes);
    let merged_counts = boundaries
        .par_windows(2)
        .map(|window| {
            let start = window[0];
            let end = window[1];
            pretokenize_count(&bytes[start..end])
        })
        .par_merge_counts();

    let time_elapsed = start_time.elapsed();
    eprintln!("Pretokenization took {time_elapsed:?}");

    merged_counts
    // merged_counts
    //     .into_iter()
    //     .map(|(k, v)| (k.to_owned(), v))
    //     .collect()
}

pub fn pretokenize_par(
    pretokenizeable: PretokenizeableSpec,
) -> HashMap<Pretoken, usize, rustc_hash::FxBuildHasher> {
    match pretokenizeable {
        PretokenizeableSpec::Bytes(s) => pretokenize_par_bytes(s),
        #[cfg(feature = "parquet")]
        PretokenizeableSpec::Parquet(path) => pretokenize_par_parquet(&path),
    }
}

// Only when the "parquet" feature is enabled
#[cfg(feature = "parquet")]
pub fn pretokenize_par_parquet(
    parquet_path: &Path,
) -> HashMap<Vec<u8>, usize, rustc_hash::FxBuildHasher> {
    use indicatif::{ProgressBar, ProgressIterator};
    use polars::prelude::*;
    let parquet_path = PlPath::Local(Arc::from(parquet_path.to_owned()));

    let df = LazyFrame::scan_parquet(parquet_path.clone(), ScanArgsParquet::default()).unwrap();

    let length = df.select([len()]).collect().unwrap();
    let length_value = length.get(0).unwrap();
    let length_value = length_value.first().unwrap();
    let length_value = match length_value {
        AnyValue::UInt32(v) => *v,
        _ => panic!("Unexpected length value type"),
    };

    eprintln!("Dataframe length: {:?}", length_value);

    let n_chunks = rayon::current_num_threads();
    let chunk_size = (length_value as usize).div_ceil(n_chunks);
    let total_counts = (0..n_chunks)
        .par_bridge()
        .map(|i| {
            let df =
                LazyFrame::scan_parquet(parquet_path.clone(), ScanArgsParquet::default())
                    .unwrap();
            let mut thread_counts = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
            let start = i * chunk_size;
            let end = min((i + 1) * chunk_size, length_value as usize);
            let m_chunks = 1024;
            let inner_chunk_size = (end - start).div_ceil(1024);
            for j in (0..m_chunks).progress_with(if i == 0 {
                ProgressBar::new(m_chunks as u64)
                    .with_finish(indicatif::ProgressFinish::AndLeave)
                    .with_style(
                        indicatif::ProgressStyle::default_bar()
                            .template(
                                "Pretokenizing and counting [{elapsed_precise}/{duration_precise}, ({per_sec})] [{wide_bar}] {pos}/{len} ({eta_precise} remaining)",
                            )
                            .unwrap(),
                    )
            } else {
                ProgressBar::hidden()
            }) {
                let inner_start = start + j * inner_chunk_size;
                let inner_end = min(start + (j + 1) * inner_chunk_size, end);
                let chunk = df.clone().slice(inner_start as i64, (inner_end - inner_start) as u32);
                let loaded = chunk.collect().unwrap();

                let col = loaded.column("text").unwrap();
                let strings = col.str().expect("Didn't find strings");
                let freqs = loaded.column("frequency").unwrap();
                let freqs = freqs.i64().expect("Didn't find frequencies");


                strings.iter().zip(freqs.iter()).flat_map(|(s, f)| match (s, f) {
                    (Some(s), Some(f)) => Some((s.as_bytes(), f as usize)),
                    (Some(s), None) => Some((s.as_bytes(), 1)),
                    _ => None,
                }).for_each(|(s, f)| {
                    pretokenize_as_iter(s).for_each(|pretoken| {
                        thread_counts
                            .entry(pretoken.to_owned())
                            .and_modify(|e| *e += f)
                            .or_insert(f);
                    })
                });
            }
            thread_counts
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

    total_counts
}

/// Return counts of all pretokens.
pub fn pretokenize_count(bytes: &[u8]) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let string = unsafe { std::str::from_utf8_unchecked(bytes) };
    string
        .split("<|endoftext|>")
        .flat_map(|s| pretokenize_as_iter(s.as_bytes().into()))
        .pretoken_count()
}

// pub fn count_pretokens<'a>(
//     pretoken_iter: impl Iterator<Item = &'a [u8]>,
// ) -> HashMap<&'a [u8], usize, rustc_hash::FxBuildHasher> {
//     let mut hashmap = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
//     pretoken_iter.for_each(|token| {
//         hashmap.entry(token).and_modify(|e| *e += 1).or_insert(1);
//     });
//     hashmap
// }

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
) -> impl Iterator<Item = Pretoken<'a>> {
    docs.flat_map(|doc| pretokenize_as_iter(doc.into()))
}

pub fn pretokenize_with_endoftext(
    bytes: &[u8],
) -> HashMap<Pretoken<'_>, usize, rustc_hash::FxBuildHasher> {
    let string = unsafe { std::str::from_utf8_unchecked(bytes) };

    let mut hashmap = HashMap::default();

    string
        .split("<|endoftext|>")
        .flat_map(|part| pretokenize_as_iter(part.as_bytes().into()))
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

impl<'a> PretokenizerIter<'a> {
    pub fn new(input: &'a [u8]) -> PretokenizerIter<'a> {
        PretokenizerIter {
            iter: UTF8Iterator::new(input.into()),
            starting: 0,
            state: PretokenizerState::Start,
        }
    }
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = Pretoken<'a>;

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
        Some(Pretoken(new_pretoken))
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
    // If we contain a 'static, assume it's a dummy.
    // This is needed only for PyO3 bindings.
    pub fn py_next<'a>(&mut self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        let mut py_self = self.replace_bytes(bytes);
        let result = py_self.next();
        *self = py_self.replace_bytes(&[]);
        Some(result?.0)
    }
}

pub fn pretokenize_as_iter(bytes: &[u8]) -> PretokenizerIter<'_> {
    PretokenizerIter {
        iter: UTF8Iterator::new(bytes.into()),
        starting: 0,
        state: PretokenizerState::Start,
    }
}

struct Pretokenizer {
    special_tokens: Vec<(Vec<u8>, u32)>, // Split on these tokens, keep them in the stream
}

impl Pretokenizer {
    pub fn new(special_tokens: Vec<(Vec<u8>, u32)>) -> Self {
        Pretokenizer { special_tokens }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use std::fs;

    use super::*;

    const GPT2_REGEX: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    /// Load the first `max_bytes` of ~/data/owt_train.txt, truncated to a UTF-8 boundary.
    fn load_owt(max_bytes: usize) -> Vec<u8> {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes =
            fs::read(data_dir.join("owt_train.txt")).expect("Could not read ~/data/owt_train.txt");
        let mut end = max_bytes.min(all_bytes.len());
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        all_bytes[..end].to_vec()
    }

    /// Compare the state-machine pretokenizer against the GPT-2 reference regex
    /// on ~5 MB of OWT data, token by token.
    #[test]
    fn test_pretokenizer_matches_regex_owt() {
        const SIZE: usize = 5_000_000;
        let input = load_owt(SIZE);
        eprintln!(
            "Testing pretokenizer vs regex on {:.1} MB of OWT",
            input.len() as f64 / 1e6
        );

        let re = fancy_regex::Regex::new(GPT2_REGEX).unwrap();
        let text = std::str::from_utf8(&input).unwrap();

        let mut sm_iter = pretokenize_as_iter(&input);
        let mut re_iter = re.find_iter(text);
        let mut token_idx: usize = 0;
        let mut recent: Vec<(String, String)> = Vec::new();

        loop {
            match (sm_iter.next(), re_iter.next()) {
                (Some(sm_tok), Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    let sm_str = String::from_utf8_lossy(sm_tok.0);
                    let re_str = &text[re_match.start()..re_match.end()];
                    recent.push((sm_str.to_string(), re_str.to_string()));
                    if recent.len() > 10 {
                        recent.remove(0);
                    }
                    assert_eq!(
                        sm_str, re_str,
                        "Mismatch at token {token_idx} (byte ~{}).\n  state machine: {:?}\n  regex:         {:?}\n  recent tokens: {:?}",
                        re_match.start(), sm_str, re_str, recent
                    );
                }
                (None, None) => break,
                (Some(sm_tok), None) => {
                    panic!(
                        "State machine produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        String::from_utf8_lossy(sm_tok.0),
                        recent
                    );
                }
                (None, Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    panic!(
                        "Regex produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        &text[re_match.start()..re_match.end()],
                        recent
                    );
                }
            }
            token_idx += 1;
        }
        eprintln!("All {token_idx} tokens match.");
    }

    #[test]
    fn test_pretokenizer_ts() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file_bytes = fs::read(data_dir.join("TinyStoriesV2-GPT4-train.txt")).unwrap();

        let pretokenized_counts = pretokenize_as_iter(&file_bytes).counts();
        eprintln!("Pretokenized {} unique tokens", pretokenized_counts.len());

        let mut sorted_counts: Vec<_> = pretokenized_counts.iter().collect();
        sorted_counts.sort_by_key(|&(_, &v)| v);
        sorted_counts.reverse();
        for &(&token, &count) in sorted_counts.iter().take(100) {
            eprintln!("{1}: {0}", String::from_utf8_lossy(&token), count);
        }
    }

    #[test]
    fn test_pretokenizer_owt_length() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let file_bytes = fs::read(data_dir.join("owt_train.txt")).unwrap();

        let pretokens_count = pretokenize_as_iter(&file_bytes).count();
        eprintln!("Pretokenized {pretokens_count} tokens");
    }
}
