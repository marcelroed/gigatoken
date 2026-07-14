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
// Batched pretoken pulling (the encode loop's input interface)
// ---------------------------------------------------------------------------

/// Chunk size of [`PretokenSpans::fill_spans_keyed`] — the live entries of
/// one [`SpanBatch`] fill.
pub const PRETOKEN_CHUNK: usize = 256;

/// Both 64-bit halves of the per-length pack mask, in scalar ALU ops. A
/// u128 `MAX >> s` lowers to a multi-instruction sequence and the 16-entry
/// table this replaces put a dependent L1 load (2.43% of process) on the
/// `n → key → store` chain; per-half variable shifts are single 1-cycle
/// ops, so the halves cost two independent 3-deep chains and no load port.
/// The length tag (`n << 120`) touches only the high half and is OR'd in
/// by the caller.
#[inline(always)]
pub(crate) fn pack_mask_halves(n: usize) -> (u64, u64) {
    debug_assert!((1..=15).contains(&n));
    let s = (n * 8) as u32;
    let lo = if n < 8 { u64::MAX >> (64u32.wrapping_sub(s) & 63) } else { u64::MAX };
    let hi = if n > 8 { u64::MAX >> (128u32.wrapping_sub(s) & 63) } else { 0 };
    (lo, hi)
}

/// Pack a pretoken of ≤ 15 bytes into a `u128` cache key: bytes in the low
/// 15 lanes, length in the top byte (so keys of different lengths never
/// collide, and a real key is never 0). Returns `None` for longer
/// pretokens, which use the slice-keyed fallback map.
///
/// The common path is a single unaligned 16-byte load followed by a mask,
/// avoiding both a variable-length `memcpy` and per-byte branching. The
/// load is only taken when it cannot cross a page boundary, so it can
/// never touch an unmapped page; the rare near-boundary case falls back to
/// a plain copy. Both paths produce the identical key.
#[inline(always)]
pub(crate) fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    if n == 0 {
        // Empty pretokens (possible through the public API, never from a
        // pretokenizer) pack to key 0, which the short table reserves as
        // its empty sentinel — the encode loop routes key 0 to the long
        // map. Also keeps the read below from touching a zero-length
        // slice's dangling pointer.
        return Some(0);
    }
    let p = bytes.as_ptr();
    let low = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: the offset within the (≥ 4096-byte) page is ≤ 4096 - 16,
        // so a 16-byte read stays inside the page holding `p`, which is
        // mapped because `p` points to at least one valid byte.
        let v = unsafe { (p as *const u128).read_unaligned() };
        let (mask_lo, mask_hi) = pack_mask_halves(n);
        ((v as u64 & mask_lo) as u128) | ((((v >> 64) as u64 & mask_hi) as u128) << 64)
    } else {
        // Rare: `p` is within 16 bytes of a page boundary. Gather with a
        // plain copy (≤ 15 bytes) — correctness over speed on this cold
        // path. Lanes past `n` stay zero, so no mask is needed.
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        u128::from_le_bytes(lanes)
    };
    Some(low | ((n as u128) << 120))
}

/// Hash of a packed pretoken key. Quality is noncritical for correctness
/// (the table compares full keys), but every consumer — the fill loops,
/// `ShortPretokenCache::grow`, `encode_bytes_cached` — must derive it
/// through this one function: the two implementations below produce
/// different values and may never mix in one process image. Both map key 0
/// to hash 0, which the fill loops' long-pretoken route stores.
#[inline(always)]
pub(crate) fn pretoken_key_hash(key: u128) -> u64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
    {
        // Hardware CRC32: two 3-cycle ops replace the 5-op multiply fold.
        // Linear over GF(2), so the low bits (the table index) see every
        // key bit; 32 bits suffice for any table under 2^32 slots.
        use core::arch::aarch64::__crc32d;
        // SAFETY: gated on the `crc` target feature at compile time.
        unsafe { __crc32d(__crc32d(0, key as u64), (key >> 64) as u64) as u64 }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "crc")))]
    {
        // One folded multiply, the cheapest mix whose low bits still see
        // every key bit.
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut h = (lo ^ hi.rotate_right(25)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
        h
    }
}

/// One batch slot: a pretoken span with its packed cache key and key hash,
/// as one 32-byte record (half a cache line, never straddling one thanks
/// to the alignment).
///
/// AoS instead of the previous three parallel arrays for dataflow on both
/// sides: the fill loops store one record with two `stp`s on a single
/// store stream (the parallel arrays cost 4 store µops across 3 streams —
/// the split u128 key store alone was two), and the probe loop's per-`i`
/// `(key, hash)` read touches one cache line instead of three.
///
/// `meta` carries the field the consumer needs next, keyed on `key`:
/// - `key != 0` (short pretoken, ≤ 15 bytes): `meta` is the full 64-bit
///   key hash. The span length rides in the key's top byte, so `ptr` +
///   `key >> 120` reconstructs the span on the (rare) slow path.
/// - `key == 0` (long pretoken, or an empty span through the public
///   adapter): `meta` is the span length in bytes. The hash is not stored:
///   the long route never probes the short table, and
///   `pretoken_key_hash(0) == 0` is what the old layout recorded anyway.
///   Prefetching `meta` as if it were a hash touches an arbitrary
///   (masked, in-bounds) table line — harmless, long pretokens are rare.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct BatchEntry {
    pub key: u128,
    pub ptr: *const u8,
    pub meta: u64,
}

const _: () = assert!(std::mem::size_of::<BatchEntry>() == 32);

impl BatchEntry {
    /// Span length, independent of the short/long route.
    #[inline(always)]
    pub fn span_len(&self) -> usize {
        if self.key != 0 { (self.key >> 120) as usize } else { self.meta as usize }
    }
}

/// Readable slack entries past a full chunk, so the emit loop's
/// prefetch-ahead `entries[i + D].meta` load needs no index clamp (a
/// per-pretoken `add + cmp + csel` in the hottest loop). Slack entries
/// are never written by a fill; prefetching a stale or zero `meta`
/// requests an arbitrary masked (in-bounds) table line — harmless.
pub const SPAN_BATCH_SLACK: usize = 16;

/// One chunk of pretoken spans with their packed cache keys (0 = longer
/// than 15 bytes, routed to the slice-keyed fallback map) and key hashes,
/// filled by [`PretokenSpans::fill_spans_keyed`]. See [`BatchEntry`] for
/// the record layout. Fills only ever write the first [`PRETOKEN_CHUNK`]
/// entries; the tail is prefetch slack (see [`SPAN_BATCH_SLACK`]).
pub struct SpanBatch<'a> {
    pub entries: [BatchEntry; PRETOKEN_CHUNK + SPAN_BATCH_SLACK],
    /// The entries' `ptr`s borrow the spans' backing storage.
    _spans: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> SpanBatch<'a> {
    pub fn new() -> Self {
        SpanBatch {
            entries: [BatchEntry { key: 0, ptr: std::ptr::null(), meta: 0 };
                PRETOKEN_CHUNK + SPAN_BATCH_SLACK],
            _spans: std::marker::PhantomData,
        }
    }

    /// Reconstruct entry `i`'s span.
    ///
    /// # Safety
    /// Entry `i` must have been written by the most recent fill (`i` below
    /// its returned count), so `ptr` still points at a live span of `'a`.
    #[inline(always)]
    pub unsafe fn span(&self, i: usize) -> &'a [u8] {
        let e = &self.entries[i];
        // SAFETY: per the contract, `ptr` points at `span_len()` live bytes.
        unsafe { std::slice::from_raw_parts(e.ptr, e.span_len()) }
    }
}

impl Default for SpanBatch<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// A source of pretoken spans, pulled a chunk at a time with their cache
/// keys derived on the way out.
///
/// `Tokenizer::memoized_encode` consumes pretokens through this instead of
/// `Iterator` for codegen reasons: pulling happens in a dedicated
/// out-of-line loop, so the pretokenizer state is register-allocated
/// across the whole chunk (inlined into the register-starved encode loop
/// it lives in stack slots, ~9 cycles/pretoken of spill traffic on Zen 2).
/// Key packing, hashing, and the cache-line prefetch ride along in the
/// same loop because the span walker is a serial dependency chain (IPC
/// ~1.7 standalone): the independent per-span key math fills its idle
/// issue slots nearly for free, where a separate pass paid for it in full.
pub trait PretokenSpans<'a> {
    /// Fill `batch` from the front with the next pretoken spans, calling
    /// `prefetch(hash)` for each. Returns how many were written; a short
    /// count (including 0) means the input is exhausted. (Recomputing the
    /// hash at the consumer instead of storing it here measured 4% slower
    /// end to end: the extra multiply sits on the probe loop's critical
    /// path, while this loop has store slots to spare.)
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize;
}

/// Shared body of the iterator-backed [`PretokenSpans`] implementations
/// (spans with no single backing buffer): pull spans from `next` and derive
/// each one's key, hash, and prefetch on the way out. `#[inline(always)]`
/// so each `#[inline(never)]` implementation fuses it with its span walker
/// into a single out-of-line loop (see the trait docs for why that fusion
/// matters). Sources that walk one backing slice use
/// [`fill_spans_keyed_with_buf`] instead.
#[inline(always)]
pub(crate) fn fill_spans_keyed_with<'a>(
    mut next: impl FnMut() -> Option<&'a [u8]>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    let mut n = 0;
    while n < PRETOKEN_CHUNK {
        let Some(span) = next() else { break };
        let (key, h) = match pack_pretoken_key(span) {
            Some(key) => (key, pretoken_key_hash(key)),
            None => (0, 0),
        };
        prefetch(h);
        // Long (and empty) spans record their length; short spans their
        // hash (see `BatchEntry::meta`).
        let meta = if key != 0 { h } else { span.len() as u64 };
        batch.entries[n] = BatchEntry { key, ptr: span.as_ptr(), meta };
        n += 1;
    }
    n
}

/// [`fill_spans_keyed_with`] for span sources that walk a single backing
/// slice, with `next` yielding `(start, end)` byte offsets into `bytes`.
/// Knowing the buffer removes the two data-dependent branches of
/// [`pack_pretoken_key`] from the per-span path:
///
/// - the `> 15` long-pretoken route becomes a select — for a long span the
///   16-byte load sits entirely inside the span, so it needs no guard, and
///   the select (not the mask clamp) provides the key-0 routing;
/// - the per-span page-boundary check (mispredict-prone: ~0.4% of spans on
///   4 KiB pages) becomes one buffer-end bound, hoisted per fill and false
///   only for short spans starting in the last 15 bytes of `bytes` — once
///   per input, so the branch predicts ~perfectly.
///
/// Empty spans cannot occur (`next` contract: `start < end`), so the key-0
/// route is exactly "longer than 15 bytes", as in the fallible packer.
#[inline(always)]
pub(crate) fn fill_spans_keyed_with_buf<'a>(
    bytes: &'a [u8],
    mut next: impl FnMut() -> Option<(usize, usize)>,
    batch: &mut SpanBatch<'a>,
    prefetch: &impl Fn(u64),
) -> usize {
    // A 16-byte load at `start` is in bounds iff `start < tail_lim`.
    let tail_lim = bytes.len().saturating_sub(15);
    let mut n = 0;
    while n < PRETOKEN_CHUNK {
        let Some((start, end)) = next() else { break };
        debug_assert!(start < end && end <= bytes.len());
        // SAFETY: `next` returns in-bounds span boundaries.
        let span = unsafe { bytes.get_unchecked(start..end) };
        let len = end - start;
        let long = len > 15;
        // `|` not `||`: one combined test, taken for all but the tail spans.
        let key = if long | (start < tail_lim) {
            // SAFETY: `start < tail_lim` puts `start + 16` within `bytes`;
            // a long span (len ≥ 16) contains its own first 16 bytes.
            let v = unsafe { (bytes.as_ptr().add(start) as *const u128).read_unaligned() };
            let m = len.min(15);
            let (mask_lo, mask_hi) = pack_mask_halves(m);
            let lo = (v as u64) & mask_lo;
            let hi = ((v >> 64) as u64 & mask_hi) | ((m as u64) << 56);
            let packed = (lo as u128) | ((hi as u128) << 64);
            if long { 0 } else { packed }
        } else {
            // Short span starting in the buffer's last 15 bytes: gather
            // with a plain copy, once per input. Lanes past `len` stay
            // zero, so no mask is needed.
            let mut lanes = [0u8; 16];
            lanes[..len].copy_from_slice(span);
            u128::from_le_bytes(lanes) | ((len as u128) << 120)
        };
        let h = pretoken_key_hash(key);
        prefetch(h);
        // Long spans record their length instead of the (unused) hash —
        // see `BatchEntry::meta`.
        let meta = if long { len as u64 } else { h };
        batch.entries[n] = BatchEntry { key, ptr: span.as_ptr(), meta };
        n += 1;
    }
    n
}

/// Adapter giving any pretoken iterator (reference pretokenizers, tests,
/// custom sources) the [`PretokenSpans`] interface. The `Fast*`
/// pretokenizers implement the trait directly over their walker state
/// instead (see `fast::fill_spans_keyed_mask`): routing them through
/// `Iterator::next` left the (large, `#[inline(always)]`) `next_span`
/// un-inlined behind a real call, costing ~23% of warm encode time.
pub struct SpanIter<I>(pub I);

impl<'a, I: Iterator<Item = Pretoken<'a>>> PretokenSpans<'a> for SpanIter<I> {
    #[inline(never)]
    fn fill_spans_keyed(&mut self, batch: &mut SpanBatch<'a>, prefetch: &impl Fn(u64)) -> usize {
        fill_spans_keyed_with(|| self.0.next().map(|p| p.0), batch, prefetch)
    }
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
        .map(memchr::memmem::Finder::new)
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

    /// Compare the production (fast r50k) pretokenizer against the GPT-2
    /// reference regex on ~5 MB of OWT data, token by token.
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

        let mut fast_iter = pretokenize_as_iter(&input);
        let mut re_iter = re.find_iter(text);
        let mut token_idx: usize = 0;
        let mut recent: Vec<(String, String)> = Vec::new();

        loop {
            match (fast_iter.next(), re_iter.next()) {
                (Some(fast_tok), Some(re_match)) => {
                    let re_match = re_match.expect("regex match error");
                    let fast_str = String::from_utf8_lossy(fast_tok.0);
                    let re_str = &text[re_match.start()..re_match.end()];
                    recent.push((fast_str.to_string(), re_str.to_string()));
                    if recent.len() > 10 {
                        recent.remove(0);
                    }
                    assert_eq!(
                        fast_str, re_str,
                        "Mismatch at token {token_idx} (byte ~{}).\n  fast:  {:?}\n  regex: {:?}\n  recent tokens: {:?}",
                        re_match.start(), fast_str, re_str, recent
                    );
                }
                (None, None) => break,
                (Some(fast_tok), None) => {
                    panic!(
                        "Fast pretokenizer produced extra token at index {token_idx}: {:?}\n  recent: {:?}",
                        String::from_utf8_lossy(fast_tok.0),
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

#[cfg(test)]
mod span_source_tests {
    use super::*;

    fn check_source<'a>(
        mut src: impl PretokenSpans<'a>,
        reference: impl Iterator<Item = Pretoken<'a>>,
        scheme: &str,
    ) {
        let expected: Vec<&[u8]> = reference.map(|p| p.0).collect();
        let mut got: Vec<&[u8]> = Vec::new();
        let mut batch = SpanBatch::new();
        let prefetched = std::cell::Cell::new(0usize);
        loop {
            let n = src.fill_spans_keyed(&mut batch, &|_h| {
                prefetched.set(prefetched.get() + 1)
            });
            for i in 0..n {
                // SAFETY: i < n, the count just returned by the fill.
                let span = unsafe { batch.span(i) };
                let (want_key, want_hash) = match pack_pretoken_key(span) {
                    Some(key) => (key, pretoken_key_hash(key)),
                    None => (0, 0),
                };
                let e = &batch.entries[i];
                assert_eq!(e.key, want_key, "{scheme}: bad key for {span:?}");
                let want_meta = if want_key != 0 {
                    want_hash
                } else {
                    span.len() as u64
                };
                assert_eq!(e.meta, want_meta, "{scheme}: bad meta for {span:?}");
                got.push(span);
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
        assert_eq!(prefetched.get(), got.len(), "{scheme}: one prefetch per span");
        assert_eq!(
            got.len(),
            expected.len(),
            "{scheme}: span count mismatch"
        );
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                g, e,
                "{scheme}: span {i} diverged: {:?} vs {:?}",
                String::from_utf8_lossy(g),
                String::from_utf8_lossy(e)
            );
        }
    }

    /// Every scheme's chunked `fill_spans_keyed` must reproduce its
    /// iterator's spans exactly, with keys/hashes derived per the shared
    /// helpers, one prefetch per span — including chunk-boundary and
    /// end-of-input handling (buffer sizes straddle multiples of
    /// PRETOKEN_CHUNK spans).
    #[test]
    fn fill_spans_keyed_matches_iterator_all_schemes() {
        let pieces: &[&str] = &[
            "word", " word", "12", " 345678", "'ll", "'s", " ", "  ", "\n", "\r\n", "\t",
            "!", " ?!", "(", "caf\u{e9}", "\u{65e5}\u{672c}", "\u{1F389}", "\u{00A0}",
            "\u{2003}", "a", " I", "don't", "one two three four five", "\u{4e2d}\u{6587}",
            "\u{30ab}\u{30bf}", "123456", " longpretokenword", "supercalifragilistic",
        ];
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..60 {
            // Vary length so end-of-input lands at many chunk offsets.
            let target = 40 + round * 97;
            let mut buf = Vec::new();
            while buf.len() < target {
                buf.extend_from_slice(pieces[(rng() % pieces.len() as u64) as usize].as_bytes());
            }
            let b: &[u8] = &buf;
            check_source(FastR50kPretokenizer::new(b), FastR50kPretokenizer::new(b), "r50k");
            check_source(
                FastCl100kPretokenizer::new(b),
                FastCl100kPretokenizer::new(b),
                "cl100k",
            );
            check_source(FastQwen2Pretokenizer::new(b), FastQwen2Pretokenizer::new(b), "qwen2");
            check_source(
                FastQwen35Pretokenizer::new(b),
                FastQwen35Pretokenizer::new(b),
                "qwen3_5",
            );
            check_source(FastOlmo3Pretokenizer::new(b), FastOlmo3Pretokenizer::new(b), "olmo3");
            check_source(
                FastDeepSeekV3Pretokenizer::new(b),
                FastDeepSeekV3Pretokenizer::new(b),
                "deepseek_v3",
            );
            check_source(
                SpanIter(PretokenizerIter::new(b)),
                PretokenizerIter::new(b),
                "state_machine",
            );
            for pt in [
                PretokenizerType::GPT2,
                PretokenizerType::GPT4,
                PretokenizerType::Qwen2,
                PretokenizerType::Qwen35,
                PretokenizerType::Olmo3,
                PretokenizerType::DeepSeekV3,
            ] {
                check_source(pt.pretokenize(b), pt.pretokenize(b), "dispatch");
            }
        }
    }

    /// Long runs hit the two-phase walker's rare paths: a pretoken longer
    /// than its u16 offset window (direct single-span emit, both the
    /// clean-batch and bad-zone variants), a scalar overrun whose end
    /// leaves the window mid-fill (dropped and re-derived next fill), and
    /// maximum-density boundary batches straddling fill boundaries.
    #[test]
    fn fill_spans_keyed_long_runs_all_schemes() {
        let mut cases: Vec<Vec<u8>> = Vec::new();
        // > 64 KB letter run between ordinary words.
        let mut v = b"intro words ".to_vec();
        v.extend(std::iter::repeat_n(b'a', 70_000));
        v.extend_from_slice(b" tail words");
        cases.push(v);
        // > 64 KB space run: boundaries only at the run's edges.
        let mut v = b"x".to_vec();
        v.extend(std::iter::repeat_n(b' ', 70_000));
        v.extend_from_slice(b"y z");
        cases.push(v);
        // > 64 KB of 3-byte unicode whitespace: batch-edge straddles make
        // bad zones, and the scalar walk overruns the whole run.
        let mut v = b"hello world ".repeat(4);
        for _ in 0..25_000 {
            v.extend_from_slice("\u{2003}".as_bytes());
        }
        v.extend_from_slice(b"end");
        cases.push(v);
        // Monster token flush at end of input.
        cases.push(std::iter::repeat_n(b'a', 70_000).collect());
        // Single-byte token alternation (64 boundaries per batch), then a
        // digit run some schemes split every 3 chars.
        let mut v = b"a1".repeat(2_000);
        v.extend(std::iter::repeat_n(b'7', 40_000));
        v.extend_from_slice(b" end");
        cases.push(v);
        for case in &cases {
            let b: &[u8] = case;
            check_source(FastR50kPretokenizer::new(b), FastR50kPretokenizer::new(b), "r50k");
            check_source(
                FastCl100kPretokenizer::new(b),
                FastCl100kPretokenizer::new(b),
                "cl100k",
            );
            check_source(FastQwen2Pretokenizer::new(b), FastQwen2Pretokenizer::new(b), "qwen2");
            check_source(
                FastQwen35Pretokenizer::new(b),
                FastQwen35Pretokenizer::new(b),
                "qwen3_5",
            );
            check_source(FastOlmo3Pretokenizer::new(b), FastOlmo3Pretokenizer::new(b), "olmo3");
            check_source(
                FastDeepSeekV3Pretokenizer::new(b),
                FastDeepSeekV3Pretokenizer::new(b),
                "deepseek_v3",
            );
        }
    }
}
