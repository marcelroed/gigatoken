//! Pretokenization: split documents into pretokens following a tokenizer's
//! pretokenization regex.
//!
//! The production implementations live in `fast` (one submodule per scheme:
//! `fast::r50k` for GPT-2, `fast::cl100k` for GPT-4, ...), selected via
//! [`PretokenizerType`]. The state-machine, combinator, and SIMD variants are
//! kept as references and benchmark baselines.
//!
//! The main entry points are:
//! - `pretokenize_as_iter`: iterate pretokens of a `&[u8]` (r50k scheme)
//! - `PretokenizerType::pretokenize`: iterate pretokens of any scheme
//! - `Pretokenize` trait: `doc.pretokens()` on any `&[u8]`
//! - `pretokenize_par_bytes`: parallel pretokenization with document splitting and counting

pub(crate) use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::pretokenize_traits::{
    ParallelMergeCounts, PretokenCountable,
};
use crate::input::Resource;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod fast;
mod options;
mod pretoken;
// Leftover single-import module; `simd::unicode` only exists behind these
// compile-time features (see src/simd/mod.rs).
#[cfg(any(
    target_feature = "avx512vbmi",
    all(target_arch = "aarch64", target_feature = "neon")
))]
mod pretoken_chunks;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw", target_feature = "avx512vl"))]
pub mod pretoken_avx512;
pub mod pretoken_combinator;
pub mod pretoken_state_machine;
pub(crate) mod pretokenize_traits;
mod unicode;
pub mod pretoken_simd;

pub use fast::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer,
};
pub use options::{FastPretokenizerDispatch, PretokenizerType};
pub use pretoken_state_machine::PretokenizerIter;

/// Default document separator used in common training corpora.
pub const DEFAULT_SEPARATOR: &[u8] = b"<|endoftext|>";

/// Iterate the pretokens of `bytes` using the production (r50k) pretokenizer.
#[inline]
pub fn pretokenize_as_iter(bytes: &[u8]) -> FastR50kPretokenizer<'_> {
    FastR50kPretokenizer::new(bytes)
}

// ---------------------------------------------------------------------------
// Pretokenize trait — Layer 3
// ---------------------------------------------------------------------------

/// Anything that can be split into a stream of pretokens.
pub trait Pretokenize {
    fn pretokens(&self) -> FastR50kPretokenizer<'_>;
}

impl Pretokenize for [u8] {
    fn pretokens(&self) -> FastR50kPretokenizer<'_> {
        pretokenize_as_iter(self)
    }
}

// ---------------------------------------------------------------------------
// Pretoken-safe document splitting
// ---------------------------------------------------------------------------

/// Split `bytes` into ranges of roughly `target` bytes whose boundaries are
/// pretoken boundaries under every supported pretokenization scheme, so
/// encoding the ranges independently and concatenating the token streams is
/// identical to encoding `bytes` in one pass.
///
/// A boundary sits on a space that is preceded by an ASCII alphanumeric and
/// followed by an ASCII letter ("…word word…"). No scheme's pretoken can
/// cross such a point: whitespace only attaches to adjacent pretokens as a
/// single *leading* space of a following word (` ?\p{L}+` and friends), and
/// the only trailing attachments are `[\r\n]*`, which cannot contain a
/// space. Letter/digit runs cannot contain a space either, and the
/// all-whitespace rules (`\s+(?!\S)`, `\s*[\r\n]+`, …) never see a run that
/// crosses the boundary because the preceding byte is alphanumeric. The
/// three ASCII bytes also cannot sit inside a multi-byte UTF-8 character.
///
/// `added_tokens` are the byte sequences matched atomically *before*
/// pretokenization (see `Tokenizer::encode_with_added_tokens`); a candidate
/// boundary is rejected when an occurrence of one straddles it, since the
/// halves would otherwise be BPE-encoded as plain text. Only tokens that
/// contain a space can ever straddle a boundary (every boundary sits on a
/// space byte), so for typical vocabularies the check costs nothing. If no
/// occurrence crosses a boundary, greedy leftmost-longest matching restarted
/// there reproduces the single-pass matches: the matcher's only state is its
/// scan position, and no match can carry it across the boundary.
pub fn safe_split_ranges(
    bytes: &[u8],
    target: usize,
    added_tokens: &[&[u8]],
) -> Vec<std::ops::Range<usize>> {
    let blockers: Vec<memchr::memmem::Finder> = added_tokens
        .iter()
        .filter(|t| t.contains(&b' '))
        .map(|t| memchr::memmem::Finder::new(t))
        .collect();
    let max_blocker = blockers.iter().map(|f| f.needle().len()).max().unwrap_or(0);
    // Whether an added-token occurrence spans the cut between `p - 1` and
    // `p`. Such an occurrence must start within `max_blocker - 1` bytes
    // before `p`, so searching a window of that radius is exhaustive.
    let cuts_added_token = |p: usize| -> bool {
        let lo = p.saturating_sub(max_blocker.saturating_sub(1));
        let hi = (p + max_blocker.saturating_sub(1)).min(bytes.len());
        blockers.iter().any(|f| {
            f.find_iter(&bytes[lo..hi])
                .any(|s| lo + s < p && lo + s + f.needle().len() > p)
        })
    };
    let len = bytes.len();
    let target = target.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    'chunks: while start < len {
        let mut probe = start + target;
        while probe + 1 < len {
            if bytes[probe] == b' '
                && bytes[probe - 1].is_ascii_alphanumeric()
                && bytes[probe + 1].is_ascii_alphabetic()
                && !(max_blocker > 0 && cuts_added_token(probe))
            {
                out.push(start..probe);
                start = probe;
                continue 'chunks;
            }
            probe += 1;
        }
        out.push(start..len);
        break;
    }
    if out.is_empty() {
        out.push(0..0);
    }
    out
}

// ---------------------------------------------------------------------------
// Parallel pretokenization with document splitting
// ---------------------------------------------------------------------------

/// Pretokenize `bytes` in parallel, splitting documents on `separator`.
/// Returns a map of pretoken → count.
pub fn pretokenize_par_bytes<'a>(
    bytes: &'a [u8],
    separator: &'a [u8],
) -> HashMap<Pretoken<'a>, usize, rustc_hash::FxBuildHasher> {
    let start_time = std::time::Instant::now();
    let n_threads = rayon::current_num_threads();
    eprintln!("Using {n_threads} threads for pretokenization");

    let chunks = bytes.par_document_chunks(separator, n_threads);

    let merged_counts = chunks
        .into_par_iter()
        .map(|doc_iter| {
            doc_iter
                .flat_map(|doc| doc.pretokens())
                .pretoken_count()
        })
        .par_merge_counts();

    let time_elapsed = start_time.elapsed();
    eprintln!("Pretokenization took {time_elapsed:?}");

    merged_counts
}

// Only when the "parquet" feature is enabled
#[cfg(feature = "parquet")]
pub fn pretokenize_par_parquet(
    parquet_path: &std::path::Path,
) -> HashMap<Vec<u8>, usize, rustc_hash::FxBuildHasher> {
    use indicatif::{ProgressBar, ProgressIterator};
    use polars::prelude::*;
    use std::cmp::min;
    let parquet_path = PlRefPath::try_from_path(parquet_path).unwrap();

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
                            .entry(pretoken.0.to_owned())
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

    /// `safe_split_ranges` must produce boundaries that no pretoken crosses,
    /// for every supported scheme: pretokenizing the ranges independently and
    /// concatenating must equal pretokenizing the whole input in one pass.
    #[test]
    fn test_safe_split_ranges_pretoken_equivalent() {
        let input = load_owt(2_000_000);

        let ranges = safe_split_ranges(&input, 10_000, &[]);
        assert!(ranges.len() > 100, "expected many splits, got {}", ranges.len());
        // Ranges must cover the input contiguously.
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, input.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        fn collect<'a, I: Iterator<Item = Pretoken<'a>>>(it: I) -> Vec<&'a [u8]> {
            it.map(|p| p.0).collect()
        }

        macro_rules! check_scheme {
            ($name:literal, $ctor:path) => {
                let whole = collect($ctor(&input));
                let split: Vec<&[u8]> = ranges
                    .iter()
                    .flat_map(|r| collect($ctor(&input[r.clone()])))
                    .collect();
                assert_eq!(whole, split, "scheme {} differs across safe splits", $name);
            };
        }
        check_scheme!("r50k", FastR50kPretokenizer::new);
        check_scheme!("cl100k", FastCl100kPretokenizer::new);
        check_scheme!("qwen2", FastQwen2Pretokenizer::new);
        check_scheme!("qwen3_5", FastQwen35Pretokenizer::new);
        check_scheme!("olmo3", FastOlmo3Pretokenizer::new);
        check_scheme!("deepseek_v3", FastDeepSeekV3Pretokenizer::new);
    }

    /// Boundaries must never cut an occurrence of a space-containing added
    /// token, while splitting still proceeds elsewhere in the document.
    #[test]
    fn test_safe_split_ranges_avoids_added_tokens() {
        let special: &[u8] = b"<|multi word special|>";
        // Deterministic LCG so word lengths vary and split probes hit the
        // special token at every possible phase.
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as usize
        };
        let words: [&[u8]; 5] = [b"alpha ", b"be ", b"gamma7 ", b"x ", b"delta "];
        let mut input = Vec::new();
        for _ in 0..4000 {
            input.extend_from_slice(words[next() % words.len()]);
            if next() % 9 == 0 {
                input.extend_from_slice(special);
            }
        }

        let ranges = safe_split_ranges(&input, 300, &[special]);
        assert!(ranges.len() > 50, "expected many splits, got {}", ranges.len());
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, input.len());
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        let occurrences: Vec<usize> =
            memchr::memmem::find_iter(&input, special).collect();
        assert!(!occurrences.is_empty());
        let cuts_occurrence = |p: usize| {
            occurrences.iter().any(|&s| s < p && s + special.len() > p)
        };
        for r in &ranges[1..] {
            assert!(!cuts_occurrence(r.start), "boundary {} cuts an occurrence", r.start);
        }

        // The input must actually tempt the splitter: without the
        // added-token check, some boundary lands inside an occurrence.
        let unaware = safe_split_ranges(&input, 300, &[]);
        assert!(
            unaware[1..].iter().any(|r| cuts_occurrence(r.start)),
            "test input never places a naive boundary inside the special token"
        );
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
