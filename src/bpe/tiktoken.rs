use crate::bpe::pretoken_cache::ShortPretokenCache;
use crate::bpe::{
    ByteRemapping, MergeScratch, PairRankTable, SHORT_MERGE_MAX, bpe_merge_symbols_short_scalar,
    bpe_merge_symbols_table_with_scratch, bpe_merge_symbols_with_scratch, simple_bpe_merge,
};
#[cfg(target_arch = "aarch64")]
use crate::bpe::bpe_merge_symbols_short_neon;
use crate::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer, PRETOKEN_CHUNK,
    Pretoken, PretokenSpans, PretokenizerType, SpanBatch, pack_pretoken_key, pretoken_key_hash,
};
use crate::token::TokenId;
use eyre::Result;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Byte-level BPE tokenizer (tiktoken / GPT-2 style).
///
/// Initial symbols are individual bytes (0–255).  Merge priority is
/// determined by the merged token's vocab ID (lower = first), which
/// equals the merge rank for tiktoken vocabularies.
pub struct Tokenizer {
    // The model tables (merges, pair_ranks, vocab, vocab_inv) are immutable
    // after construction and shared across forks behind `Arc`: parallel
    // workers read the same few MB of tables instead of holding one deep
    // clone each, which keeps a single copy resident per cache/cluster on
    // the cold miss path and makes forking the tables O(1). The rare
    // mutation (`add_special_token`) goes through `Arc::make_mut`.
    pub(crate) merges: Arc<HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>>,
    /// Flat pair-rank tables replacing `merges` lookups on the miss path's
    /// merge loop; `None` for vocabularies whose IDs don't fit its packed
    /// keys (those keep probing `merges`).
    pair_ranks: Option<Arc<PairRankTable>>,
    pub(crate) vocab: Arc<Vec<Arc<[u8]>>>,
    pub(crate) vocab_inv: Arc<HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>>,
    pub(crate) byte_remapping: Option<ByteRemapping>,
    /// Append-only arena of encoded token IDs. Cache entries for encodings
    /// of 5+ tokens store `(offset, len)` slices into this vector; shorter
    /// encodings (well over 99% of hit occurrences) live inline in the
    /// cache entry and never touch it.
    token_arena: Vec<TokenId>,
    /// Pretoken cache for the common case (≤ 15 bytes, ~99.9% of
    /// pretokens). The key packs the bytes into the low 15 bytes and the
    /// length into the top byte of a `u128`, so lookups are a single
    /// inlined 128-bit compare instead of a `memcmp` call. See
    /// `pretoken_cache.rs` for why this is a custom prefetchable table
    /// rather than a `HashMap`.
    pretoken_cache: ShortPretokenCache,
    /// Fallback cache for pretokens longer than 15 bytes.
    pretoken_cache_long: HashMap<Box<[u8]>, (u32, u32), rustc_hash::FxBuildHasher>,
    /// Scratch buffers reused across cache-missing pretokens so the merge loop
    /// performs no per-pretoken allocations.
    merge_scratch: MergeScratch,
    symbol_scratch: Vec<TokenId>,
    /// Pretokenization scheme used by [`Self::encode_with_added_tokens`].
    pub(crate) pretokenizer_type: PretokenizerType,
    /// Added tokens (special and non-special), matched atomically in the raw
    /// input before pretokenization, like HuggingFace's AddedVocabulary.
    added_tokens: Vec<(Arc<[u8]>, TokenId)>,
    /// Leftmost-longest Aho-Corasick automaton over `added_tokens` contents
    /// (pattern index == `added_tokens` index). A prebuilt automaton keeps the
    /// scan fast even when an added token starts with a byte that is common in
    /// text (ModernBERT has 23 space-run added tokens, so a first-byte
    /// candidate scan would probe on every space). Clones share the automaton
    /// via its internal `Arc`.
    added_matcher: Option<aho_corasick::AhoCorasick>,
    /// Apply NFC normalization to non-added-token segments before
    /// pretokenization, like HuggingFace's `NFC` normalizer (e.g. Qwen2).
    normalize_nfc: bool,
}

/// NFC-normalize a segment if needed, using `buf` as scratch on the slow path.
///
/// ASCII and already-normalized segments are returned as-is. Invalid UTF-8 is
/// passed through unchanged (HF only ever sees `str`, so there is no parity
/// behavior to match).
fn nfc_segment<'a>(seg: &'a [u8], buf: &'a mut String) -> &'a [u8] {
    if seg.is_ascii() {
        return seg;
    }
    let Ok(s) = std::str::from_utf8(seg) else {
        return seg;
    };
    let nfc = icu::normalizer::ComposingNormalizer::new_nfc();
    if nfc.is_normalized(s) {
        return seg;
    }
    buf.clear();
    nfc.normalize_to(s, buf)
        .expect("writing to a String cannot fail");
    buf.as_bytes()
}

/// Cache-value packing (shared by the short-pretoken table and decode in
/// the encode loop). `val` low byte: token count in bits 0-6 plus a
/// "spilled" flag in bit 7. Inline values (1-4 tokens; only the first ID
/// must fit 24 bits — true of every real vocab) carry tokens 1-2 in `val`
/// bits 8-31 and 32-63 and tokens 3-4 in `ext`'s two u32 lanes; spilled
/// values carry the token-arena offset in `val`'s high 32 bits and leave
/// `ext` unused.
const VAL_SPILL: u64 = 0x80;

#[inline(always)]
fn pack_val_inline(symbols: &[TokenId]) -> Option<(u64, u64)> {
    match *symbols {
        [a] if a.0 < (1 << 24) => Some((1 | ((a.0 as u64) << 8), 0)),
        [a, b] if a.0 < (1 << 24) => {
            Some((2 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32), 0))
        }
        [a, b, c] if a.0 < (1 << 24) => Some((
            3 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32),
            c.0 as u64,
        )),
        [a, b, c, d] if a.0 < (1 << 24) => Some((
            4 | ((a.0 as u64) << 8) | ((b.0 as u64) << 32),
            c.0 as u64 | ((d.0 as u64) << 32),
        )),
        _ => None,
    }
}

/// View a `TokenId` slice as its underlying `u32`s (repr(transparent)),
/// so bulk emits are `extend_from_slice` memcpys instead of per-element
/// iterator writes.
#[inline(always)]
fn token_ids_as_u32s(toks: &[TokenId]) -> &[u32] {
    // SAFETY: TokenId is #[repr(transparent)] over u32.
    unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u32, toks.len()) }
}

/// Unpack an inline value's four token lanes (lanes past the count are
/// another key's leftovers; callers truncate by the count).
#[inline(always)]
fn unpack_val_lanes(val: u64, ext: u64) -> [u32; 4] {
    [
        (val >> 8) as u32 & 0xFF_FFFF,
        (val >> 32) as u32,
        ext as u32,
        (ext >> 32) as u32,
    ]
}

impl Tokenizer {
    pub fn new(
        merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect::<Vec<Arc<_>>>();
        let vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher> = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();
        let pair_ranks =
            PairRankTable::build(&merges, byte_remapping.as_ref(), vocab.len()).map(Arc::new);
        let mut token_arena = Vec::new();
        let pretoken_cache = Self::seeded_pretoken_cache(&vocab, &mut token_arena, 0);
        Tokenizer {
            merges: Arc::new(merges),
            pair_ranks,
            vocab_inv: Arc::new(vocab_inv),
            vocab: Arc::new(vocab),
            byte_remapping,
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
        }
    }

    /// A short-pretoken cache pre-seeded with every vocab entry of 1..=15
    /// bytes as its own single-token encoding. Any short pretoken that is a
    /// whole vocab word then hits the cache outright, so the miss path can
    /// skip the reverse-vocab probe: a short miss is guaranteed not to be a
    /// vocab word. Iterating IDs descending and keeping the first insert
    /// per key resolves duplicate byte strings to the highest ID — the same
    /// entry `vocab_inv` (built ascending, later inserts overwrite) returns.
    ///
    /// `min_slots` additionally floors the table size for a worker with a
    /// known workload (see [`Self::fork_sized`]); the table is built once
    /// at the max of the seed requirement and that floor, so seeding never
    /// grows it mid-way.
    fn seeded_pretoken_cache(
        vocab: &[Arc<[u8]>],
        token_arena: &mut Vec<TokenId>,
        min_slots: usize,
    ) -> ShortPretokenCache {
        let n_short = vocab
            .iter()
            .filter(|bytes| (1..=15).contains(&bytes.len()))
            .count();
        let mut cache = ShortPretokenCache::with_at_least(n_short, min_slots);
        for id in (0..vocab.len() as u32).rev() {
            let bytes = &vocab[id as usize];
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            if cache.get(key, h).is_none() {
                let (val, ext) = Self::pack_val(&[TokenId(id)], token_arena);
                cache.insert(key, h, val, ext);
            }
        }
        cache
    }

    /// Pack a cache value: inline when possible, else spilled to the arena.
    #[inline(always)]
    fn pack_val(symbols: &[TokenId], token_arena: &mut Vec<TokenId>) -> (u64, u64) {
        pack_val_inline(symbols).unwrap_or_else(|| {
            let offset = token_arena.len() as u64;
            token_arena.extend_from_slice(symbols);
            (VAL_SPILL | symbols.len() as u64 | (offset << 32), 0)
        })
    }

    /// Given a list of tokens in rank order (by merge order), reconstructs the
    /// merges map and returns a Tokenizer.
    ///
    /// This process is necessary to load some tokenizers found in tiktoken.
    pub fn from_ranks(vocab: Vec<Vec<u8>>) -> Result<Self> {
        let mut merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher> =
            HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = vocab
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Arc<[u8]>>>();
        let vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher> = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();

        for (token_idx, token_bytes) in vocab.iter().cloned().enumerate() {
            if token_bytes.len() < 2 {
                continue;
            }
            let byte_symbols: Vec<u8> = token_bytes
                .iter()
                .map(|b| vocab_inv.get(std::slice::from_ref(b)).unwrap().0 as u8)
                .collect();
            let tokenized = simple_bpe_merge(&merges, &byte_symbols);
            assert_eq!(tokenized.len(), 2);
            merges.insert((tokenized[0], tokenized[1]), TokenId::from(token_idx));
        }

        let byte_remapping = ByteRemapping::from_byte_vocab(&vocab)?;
        let pair_ranks =
            PairRankTable::build(&merges, byte_remapping.as_ref(), vocab.len()).map(Arc::new);
        let mut token_arena = Vec::new();
        let pretoken_cache = Self::seeded_pretoken_cache(&vocab, &mut token_arena, 0);
        Ok(Tokenizer {
            merges: Arc::new(merges),
            pair_ranks,
            byte_remapping,
            vocab: Arc::new(vocab),
            vocab_inv: Arc::new(vocab_inv),
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
        })
    }

    /// Create a new tokenizer sharing the same model data but with a
    /// freshly seeded cache (no encoded pretokens beyond the vocab seed).
    /// Useful for per-thread encoding in parallel.
    pub fn fork(&self) -> Self {
        self.fork_sized(0)
    }

    /// [`Self::fork`] with the caches pre-sized for a worker expected to
    /// encode roughly `expected_bytes` of input. On a cold parallel run a
    /// default-sized worker rehashes its pretoken table through 6-7
    /// doublings — random scatter writes into a fresh zeroed allocation
    /// each time, on every worker at once; sizing from the input share
    /// pays for the table exactly once. The estimates are capacity hints
    /// only: every structure still grows past them as needed, and the
    /// clamps keep tiny inputs at the default size. The short-table size
    /// is a floor passed through the vocab seeding, so the seed
    /// requirement and the workload estimate resolve to one table
    /// construction (whichever is larger).
    pub(crate) fn fork_sized(&self, expected_bytes: usize) -> Self {
        // Distinct short pretokens follow Heaps' law: ~1.3M at 1 GB and
        // ~5.5M at 10 GB of OWT-like text gives distinct(n) ≈ 3.45·n^0.62.
        // The previous linear rule (1 per 256 bytes) covered a share but
        // ~2x oversized it — a 10 GB/16-way worker got a 2^22-slot (128 MB)
        // table for ~1M entries, and the 16 concurrent 128 MB memsets of
        // the zeroed tables were most of a measured ~32 ms fork+seed ramp.
        // Size for the Heaps estimate at the table's 3/4 growth load with
        // 1.4x headroom (self-paced chunk handout lets a fast core encode
        // more than its even share; the margin holds a >2x-oversubscribed
        // worker under the growth threshold before the table would resize).
        // Still a capacity hint: the table grows past it at 3/4 load on
        // corpora more diverse than the OWT calibration. Clamped to 2^22
        // slots (128 MB) per worker as before.
        let distinct = 3.45 * (expected_bytes as f64).powf(0.62);
        let cache_slots = ((distinct * (4.0 / 3.0) * 1.4) as usize)
            .clamp(1 << 16, 1 << 22)
            .next_power_of_two();
        let arena_cap = (expected_bytes / 256).min(1 << 24);
        let long_cap = (expected_bytes / 8192).min(1 << 20);
        let mut token_arena = Vec::with_capacity(arena_cap);
        let pretoken_cache =
            Self::seeded_pretoken_cache(&self.vocab, &mut token_arena, cache_slots);
        Tokenizer {
            merges: Arc::clone(&self.merges),
            pair_ranks: self.pair_ranks.clone(),
            vocab: Arc::clone(&self.vocab),
            vocab_inv: Arc::clone(&self.vocab_inv),
            byte_remapping: self.byte_remapping.clone(),
            token_arena,
            pretoken_cache,
            pretoken_cache_long: HashMap::with_capacity_and_hasher(
                long_cap,
                rustc_hash::FxBuildHasher {},
            ),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: self.pretokenizer_type,
            added_tokens: self.added_tokens.clone(),
            added_matcher: self.added_matcher.clone(),
            normalize_nfc: self.normalize_nfc,
        }
    }

    pub fn set_pretokenizer_type(&mut self, pretokenizer_type: PretokenizerType) {
        self.pretokenizer_type = pretokenizer_type;
    }

    pub fn pretokenizer_type(&self) -> PretokenizerType {
        self.pretokenizer_type
    }

    /// Enable NFC normalization of non-added-token segments before
    /// pretokenization (HF `normalizer: {"type": "NFC"}`).
    pub fn set_normalize_nfc(&mut self, normalize_nfc: bool) {
        self.normalize_nfc = normalize_nfc;
    }

    /// Set the added tokens matched atomically by
    /// [`Self::encode_with_added_tokens`]. Empty contents are ignored.
    pub fn set_added_tokens(&mut self, added_tokens: Vec<(Vec<u8>, TokenId)>) {
        let mut added_tokens: Vec<(Arc<[u8]>, TokenId)> = added_tokens
            .into_iter()
            .filter(|(content, _)| !content.is_empty())
            .map(|(content, id)| (content.into(), id))
            .collect();
        added_tokens.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));
        self.added_matcher = (!added_tokens.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(added_tokens.iter().map(|(c, _)| c.as_ref()))
                .expect("added-token automaton construction cannot fail")
        });
        self.added_tokens = added_tokens;
    }

    /// Register one additional added token, extending the decode vocab when
    /// its id lies outside the base ranks (mirrors the out-of-vocab
    /// added-token handling in the HF loader).
    pub fn add_special_token(&mut self, content: Vec<u8>, id: TokenId) {
        let idx = id.0 as usize;
        // Loader-phase mutation of the shared model tables: `make_mut`
        // copies only when a fork holds the tables too (never during
        // loading, where this is called).
        let vocab = Arc::make_mut(&mut self.vocab);
        if idx >= vocab.len() {
            vocab.resize(idx + 1, Arc::from(Vec::new().as_slice()));
        }
        if vocab[idx].is_empty() {
            vocab[idx] = content.clone().into();
            Arc::make_mut(&mut self.vocab_inv).insert(vocab[idx].clone(), id);
            // Keep the vocab seed in sync (see `seeded_pretoken_cache`): a
            // short pretoken matching this content must resolve to `id`
            // without the miss path's reverse-vocab probe. An existing cache
            // entry (a previously encoded pretoken) keeps its encoding, as
            // it would have before seeding existed.
            if (1..=15).contains(&content.len())
                && let Some(key) = pack_pretoken_key(&content)
            {
                let h = pretoken_key_hash(key);
                if self.pretoken_cache.get(key, h).is_none() {
                    let (val, ext) = Self::pack_val(&[id], &mut self.token_arena);
                    self.pretoken_cache.insert(key, h, val, ext);
                }
            }
        }
        let mut added: Vec<(Vec<u8>, TokenId)> = self
            .added_tokens
            .iter()
            .map(|(c, i)| (c.to_vec(), *i))
            .collect();
        added.push((content, id));
        self.set_added_tokens(added);
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens (IDs with no assigned content count too).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Vocabulary entries as `(id, bytes)` pairs in ID order, including
    /// added tokens and skipping IDs with no assigned content.
    pub fn vocab_entries(&self) -> impl Iterator<Item = (u32, &[u8])> {
        super::vocab_entries(&self.vocab)
    }

    /// Merge rules as `(left, right)` byte pairs in merge-priority order
    /// (priority equals the merged token's ID for tiktoken vocabularies).
    pub fn merge_entries(&self) -> Vec<(&[u8], &[u8])> {
        let mut ranked: Vec<_> = self.merges.iter().collect();
        ranked.sort_unstable_by_key(|&(_, merged)| merged);
        ranked
            .into_iter()
            .map(|(&(a, b), _)| {
                (
                    self.vocab[a.0 as usize].as_ref(),
                    self.vocab[b.0 as usize].as_ref(),
                )
            })
            .collect()
    }

    /// Contents of the added tokens, for callers that split documents at
    /// byte-level boundaries (see `pretokenize::safe_split_ranges`): added
    /// tokens are matched atomically before pretokenization, so a split must
    /// never cut an occurrence in half.
    pub fn added_token_contents(&self) -> Vec<&[u8]> {
        self.added_tokens.iter().map(|(c, _)| c.as_ref()).collect()
    }

    /// Find the leftmost added-token occurrence at or after `from`, taking
    /// the longest token when several match at the same position. Returns
    /// `(start, end, id)`.
    fn find_added_token(&self, bytes: &[u8], from: usize) -> Option<(usize, usize, TokenId)> {
        let m = self.added_matcher.as_ref()?.find(&bytes[from..])?;
        let id = self.added_tokens[m.pattern().as_usize()].1;
        Some((from + m.start(), from + m.end(), id))
    }

    /// Encode raw text: split out added-token occurrences (emitted as their
    /// single token ID), pretokenize the segments between them with this
    /// tokenizer's pretokenization scheme, and BPE-encode each pretoken.
    /// This mirrors the full HuggingFace `tokenizers` encode pipeline.
    pub fn encode_with_added_tokens(&mut self, bytes: &[u8], mut f: impl FnMut(&[TokenId])) {
        let pretokenizer_type = self.pretokenizer_type;
        let normalize_nfc = self.normalize_nfc;
        let mut nfc_buf = String::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let (seg_end, added) = match self.find_added_token(bytes, pos) {
                Some((start, end, id)) => (start, Some((end, id))),
                None => (bytes.len(), None),
            };
            let segment = if normalize_nfc {
                nfc_segment(&bytes[pos..seg_end], &mut nfc_buf)
            } else {
                &bytes[pos..seg_end]
            };
            match pretokenizer_type {
                PretokenizerType::GPT2 => {
                    self.memoized_encode(FastR50kPretokenizer::new(segment), &mut f)
                }
                PretokenizerType::GPT4 => {
                    self.memoized_encode(FastCl100kPretokenizer::new(segment), &mut f)
                }
                PretokenizerType::Qwen2 => {
                    self.memoized_encode(FastQwen2Pretokenizer::new(segment), &mut f)
                }
                PretokenizerType::Qwen35 => {
                    self.memoized_encode(FastQwen35Pretokenizer::new(segment), &mut f)
                }
                PretokenizerType::Olmo3 => {
                    self.memoized_encode(FastOlmo3Pretokenizer::new(segment), &mut f)
                }
                PretokenizerType::DeepSeekV3 => {
                    self.memoized_encode(FastDeepSeekV3Pretokenizer::new(segment), &mut f)
                }
            }
            match added {
                Some((end, id)) => {
                    f(&[id]);
                    pos = end;
                }
                None => break,
            }
        }
    }

    /// Flat variant of [`Self::encode_with_added_tokens`]: the identical
    /// token stream appended to `out` as raw u32 ids, routed through
    /// [`Self::memoized_encode_flat`] so segment tokens land directly in
    /// the caller's buffer (the batch engine's per-chunk id buffer).
    pub fn encode_with_added_tokens_flat(&mut self, bytes: &[u8], out: &mut Vec<u32>) {
        let pretokenizer_type = self.pretokenizer_type;
        let normalize_nfc = self.normalize_nfc;
        let mut nfc_buf = String::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let (seg_end, added) = match self.find_added_token(bytes, pos) {
                Some((start, end, id)) => (start, Some((end, id))),
                None => (bytes.len(), None),
            };
            let segment = if normalize_nfc {
                nfc_segment(&bytes[pos..seg_end], &mut nfc_buf)
            } else {
                &bytes[pos..seg_end]
            };
            match pretokenizer_type {
                PretokenizerType::GPT2 => {
                    self.memoized_encode_flat(FastR50kPretokenizer::new(segment), out)
                }
                PretokenizerType::GPT4 => {
                    self.memoized_encode_flat(FastCl100kPretokenizer::new(segment), out)
                }
                PretokenizerType::Qwen2 => {
                    self.memoized_encode_flat(FastQwen2Pretokenizer::new(segment), out)
                }
                PretokenizerType::Qwen35 => {
                    self.memoized_encode_flat(FastQwen35Pretokenizer::new(segment), out)
                }
                PretokenizerType::Olmo3 => {
                    self.memoized_encode_flat(FastOlmo3Pretokenizer::new(segment), out)
                }
                PretokenizerType::DeepSeekV3 => {
                    self.memoized_encode_flat(FastDeepSeekV3Pretokenizer::new(segment), out)
                }
            }
            match added {
                Some((end, id)) => {
                    out.push(id.0);
                    pos = end;
                }
                None => break,
            }
        }
    }

    /// For each pretoken in the input iterator, looks up the string in the
    /// cache, and if not found, encodes it and inserts it into the cache.
    /// Calls `f` with the encoded token slice for each pretoken.
    ///
    /// A thin wrapper over the flat probe/emit machinery (see
    /// [`Self::memoized_encode_flat`], the path the batch engine and
    /// benches use): each chunk's tokens land in a reused L1-resident
    /// buffer with per-pretoken end offsets recorded on the side, then `f`
    /// receives one slice per pretoken.
    pub fn memoized_encode<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        mut f: impl FnMut(&[TokenId]),
    ) {
        let mut batch = SpanBatch::new();
        let mut out: Vec<u32> = Vec::new();
        let mut ends = [0usize; PRETOKEN_CHUNK];
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            out.clear();
            self.probe_emit_chunk(&batch, n, &mut out, |i, w| ends[i] = w);
            let mut start = 0;
            for &end in &ends[..n] {
                // SAFETY: TokenId is repr(transparent) over u32, and the
                // recorded ends partition `out` (0 <= start <= end <= len).
                f(unsafe {
                    std::slice::from_raw_parts(
                        out.as_ptr().add(start) as *const TokenId,
                        end - start,
                    )
                });
                start = end;
            }
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Flat variant of [`Self::memoized_encode`]: the identical token
    /// stream appended to `out` as raw u32 ids (bit-compatible with
    /// `TokenId`), with no per-pretoken delivery. This is the batch
    /// engine's output shape (`batch::encode_into` fills chunk id buffers),
    /// so the emit loop writes tokens straight into the final buffer.
    ///
    /// Runs in chunks of `PRETOKEN_CHUNK` pretokens through two phases —
    /// pull spans from the pretokenizer with keys/hashes derived and probe
    /// lines prefetched into L2 on the way out (out of line, fused with
    /// the span walker — see PretokenSpans), then probe and emit. The
    /// phase split keeps the walker's state register-allocated in one
    /// tight loop and gives every probe line a chunk of latency (hundreds
    /// of cycles, enough to cover DRAM) before its probe.
    pub fn memoized_encode_flat<'i>(
        &mut self,
        mut pretokens: impl PretokenSpans<'i>,
        out: &mut Vec<u32>,
    ) {
        let mut batch = SpanBatch::new();
        loop {
            let cache = &self.pretoken_cache;
            let n = pretokens.fill_spans_keyed(&mut batch, &|h| cache.prefetch_l2(h));
            if n == 0 {
                break;
            }
            self.probe_emit_chunk(&batch, n, out, |_, _| {});
            if n < PRETOKEN_CHUNK {
                break;
            }
        }
    }

    /// Probe-and-emit for one chunk: branchless flat emit with a single
    /// rare data-dependent branch per pretoken. Every iteration stores the
    /// probed value's four token lanes unconditionally at the write cursor
    /// and advances by the token count only when the fast predicate (pair
    /// hit ∧ inline value ∧ short key, ~99% of pretokens) holds; stores
    /// past the cursor are dead — overwritten by a later iteration or
    /// truncated by the final `set_len`. Everything else — probe walks
    /// past the home pair, arena spills, long pretokens, misses — takes
    /// the `#[cold]` slow path. `record(i, cursor)` runs once per pretoken
    /// (per-pretoken slicing in [`Self::memoized_encode`]; a no-op closure
    /// in the flat variant).
    ///
    /// Slack invariant: `out.capacity() >= cursor + 4 * (iterations
    /// left)`, established by the reserve below and re-established by the
    /// slow path after any reallocation, so the two 8-byte stores are
    /// always in bounds.
    #[inline(always)]
    fn probe_emit_chunk(
        &mut self,
        batch: &SpanBatch<'_>,
        n: usize,
        out: &mut Vec<u32>,
        mut record: impl FnMut(usize, usize),
    ) {
        // One check up front so `i`- and `pf`-indexing of the batch arrays
        // below is provably in bounds (removes two per-iteration compares).
        assert!(n <= PRETOKEN_CHUNK);
        if n == 0 {
            return;
        }
        out.reserve(4 * n);
        let mut w = out.len();
        // Loop-invariant raw cursors. The slow path's `&mut self` call is
        // the only thing that can move `out`'s buffer or the cache's slot
        // array, so both are refreshed there and nowhere else; without
        // these the compiler reloaded the Vec pointer, table base, and
        // mask from the stack on every iteration.
        let mut dst = out.as_mut_ptr();
        let mut table = self.pretoken_cache.probe_view();
        // Probe-stage prefetch: promote the pair's line L2 -> L1 a fixed
        // short distance ahead (the fill phase staged it into L2; D only
        // has to cover the L2 hit latency, a handful of iterations).
        const D: usize = 16;
        const _: () = assert!(D <= crate::pretokenize::SPAN_BATCH_SLACK);
        for i in 0..D.min(n) {
            table.prefetch(batch.entries[i].meta);
        }
        for i in 0..n {
            // Unclamped prefetch distance: the batch carries D slack
            // entries past a full chunk, so `i + D` always indexes into
            // the array and the clamp's per-pretoken add+cmp+csel (and
            // before that, an `i + D < n` compare+branch worth 3.7% of
            // encode) disappears — the load is one fixed-offset ldr off
            // the walking entry pointer. Tail iterations prefetch stale
            // or zero `meta`, and long entries a length, not a hash —
            // either way a masked, in-bounds table line: harmless.
            table.prefetch(batch.entries[i + D].meta);
            // One 32-byte entry: key + meta land in a single cache line
            // (the parallel-array layout walked three load streams here).
            let (key, h) = (batch.entries[i].key, batch.entries[i].meta);
            let (val, ext, found) = table.probe_pair(key, h);
            // `key != 0` folds the long-pretoken route in AND guards the
            // empty-slot sentinel (probe_pair matches key 0 against empty
            // slots); on !found the lanes below are another entry's, dead
            // because the cursor does not advance.
            let fast = found & (val & VAL_SPILL == 0) & (key != 0);
            // Lanes 1-2 packed into one u64 store, lanes 3-4 are `ext`
            // verbatim (little-endian lane order, like the raw key load in
            // `pack_pretoken_key`); the two u64 writes fuse into one 16 B
            // `stp`.
            let ab = ((val >> 8) & 0x00FF_FFFF) | (val & 0xFFFF_FFFF_0000_0000);
            // SAFETY: the slack invariant leaves >= 4 u32s past `w`.
            unsafe {
                let p = dst.add(w);
                (p as *mut u64).write_unaligned(ab);
                (p.add(2) as *mut u64).write_unaligned(ext);
            }
            w += if fast { (val & 0x7F) as usize } else { 0 };
            if !fast {
                // Cold: reconstruct the span from the entry. For key == 0
                // `h` is really the span length, but the slow path never
                // reads `h` on the long route (see probe_emit_slow), so it
                // passes through unfiltered — a select here got hoisted
                // into the hot loop as a per-pretoken cset.
                // SAFETY: entry `i` was written by this chunk's fill, so
                // `ptr` points at a live span of the input's lifetime.
                let bytes = unsafe { batch.span(i) };
                w = self.probe_emit_slow(bytes, key, h, out, w);
                dst = out.as_mut_ptr();
                table = self.pretoken_cache.probe_view();
            }
            record(i, w);
        }
        // SAFETY: w <= capacity by the slack invariant, and every element
        // below `w` was written (fast advances never skip lanes; the slow
        // path appends through Vec).
        unsafe { out.set_len(w) };
    }

    /// Everything [`Self::probe_emit_chunk`]'s fast predicate rejects.
    /// Appends this pretoken's tokens at cursor `w` and returns the new
    /// cursor, re-establishing the emit loop's slack invariant.
    ///
    /// `h` is only meaningful (and only read) when `key != 0`: the long
    /// route keys on `bytes` and passes literal zeros to the miss path.
    /// The emit loop relies on this and forwards the batch entry's `meta`
    /// (the span length when `key == 0`) without filtering it.
    #[cold]
    #[inline(never)]
    fn probe_emit_slow(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        out: &mut Vec<u32>,
        w: usize,
    ) -> usize {
        // SAFETY: elements below `w` are initialized and w <= capacity
        // (emit-loop invariant); Vec append methods need len in sync.
        unsafe { out.set_len(w) };
        if key != 0 {
            // A miss hands back the insert slot its walk found, so the
            // miss path's insert skips re-walking the (just-touched)
            // chain.
            match self.pretoken_cache.get_or_slot(key, h) {
                Ok((val, ext)) => {
                    let len = (val & 0x7F) as usize;
                    if val & VAL_SPILL == 0 {
                        out.extend_from_slice(&unpack_val_lanes(val, ext)[..len]);
                    } else {
                        let start = (val >> 32) as usize;
                        // SAFETY: recorded right after appending `len`
                        // tokens at `start`; the arena never shrinks.
                        let toks =
                            unsafe { self.token_arena.get_unchecked(start..start + len) };
                        out.extend_from_slice(token_ids_as_u32s(toks));
                    }
                }
                Err(slot) => self.encode_pretoken_miss(bytes, key, h, slot, out),
            }
        } else {
            // Long pretokens (> 15 bytes, rare) always spill to the arena;
            // their token counts can exceed the packed-value range, so
            // they bypass it entirely.
            match self.pretoken_cache_long.get(bytes) {
                Some(&(offset, len)) => {
                    let start = offset as usize;
                    // SAFETY: as above.
                    let toks = unsafe {
                        self.token_arena.get_unchecked(start..start + len as usize)
                    };
                    out.extend_from_slice(token_ids_as_u32s(toks));
                }
                None => self.encode_pretoken_miss(bytes, 0, 0, 0, out),
            }
        }
        out.reserve(4 * PRETOKEN_CHUNK);
        out.len()
    }

    /// Cache-miss path of the probe/emit loop: BPE-encode `bytes`, record
    /// it in the table `key` routes to (the short-pretoken table, or the
    /// long map when `key == 0`), and append its tokens to `out`. `slot`
    /// is the short-cache insert position reported by the failed
    /// `get_or_slot` probe (meaningful only when `key != 0`); nothing
    /// here touches the short cache before the insert, so it stays valid.
    #[inline(never)]
    fn encode_pretoken_miss(
        &mut self,
        bytes: &[u8],
        key: u128,
        h: u64,
        slot: usize,
        out: &mut Vec<u32>,
    ) {
        if key != 0 {
            // Short pretoken (≤ 15 bytes, the overwhelming majority of
            // misses). The cache is pre-seeded with every short vocab entry
            // (see `seeded_pretoken_cache`), so a miss here is never a whole
            // vocab word — no reverse-vocab probe, straight to byte symbols
            // and the merge loop, in a stack buffer instead of the `Vec`
            // scratch.
            let n = bytes.len();
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            match self.byte_remapping.as_ref() {
                Some(br) => {
                    for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                        *dst = br.mapping[b as usize];
                    }
                }
                None => {
                    for (dst, &b) in buf[..n].iter_mut().zip(bytes) {
                        *dst = TokenId(b as u32);
                    }
                }
            }
            let n = if n >= 2 {
                match self.pair_ranks.as_deref() {
                    #[cfg(target_arch = "aarch64")]
                    Some(table) => bpe_merge_symbols_short_neon(table, &mut buf, n),
                    #[cfg(not(target_arch = "aarch64"))]
                    Some(table) => {
                        bpe_merge_symbols_short_scalar(|a, b| table.rank(a, b), &mut buf, n)
                    }
                    None => bpe_merge_symbols_short_scalar(
                        |a, b| self.merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                        &mut buf,
                        n,
                    ),
                }
            } else {
                n
            };
            let symbols = &buf[..n];
            let (val, ext) = Self::pack_val(symbols, &mut self.token_arena);
            self.pretoken_cache.insert_at(slot, key, h, val, ext);
            out.extend_from_slice(token_ids_as_u32s(symbols));
        } else {
            // Long pretoken (> 15 bytes, rare). A pretoken that is a
            // complete vocab entry skips the merge loop entirely via one
            // reverse-vocab probe (short keys get this from the vocab seed;
            // long ones still need the probe).
            let symbols = &mut self.symbol_scratch;
            symbols.clear();
            if let Some(&tid) = self.vocab_inv.get(bytes) {
                symbols.push(tid);
            } else {
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(bytes.iter().map(|&b| br.mapping[b as usize])),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                match self.pair_ranks.as_deref() {
                    Some(table) => {
                        bpe_merge_symbols_table_with_scratch(table, symbols, &mut self.merge_scratch)
                    }
                    None => {
                        bpe_merge_symbols_with_scratch(&self.merges, symbols, &mut self.merge_scratch)
                    }
                }
            }
            let len = symbols.len() as u32;
            let offset = self.token_arena.len() as u32;
            self.token_arena.extend_from_slice(symbols);
            self.pretoken_cache_long.insert(bytes.into(), (offset, len));
            out.extend_from_slice(token_ids_as_u32s(symbols));
        }
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| self.vocab[token.0 as usize].as_ref())
            .copied()
    }

    /// Detailed cache stats for memory accounting (see examples/cache_memory.rs):
    /// (short_len, short_cap, long_len, long_cap, long_key_bytes, arena_len, arena_cap).
    pub fn cache_mem_stats(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let long_key_bytes: usize = self.pretoken_cache_long.keys().map(|k| k.len()).sum();
        (
            self.pretoken_cache.len(),
            self.pretoken_cache.capacity(),
            self.pretoken_cache_long.len(),
            self.pretoken_cache_long.capacity(),
            long_key_bytes,
            self.token_arena.len(),
            self.token_arena.capacity(),
        )
    }
}

impl Debug for Tokenizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("merges_count", &self.merges.len())
            .field("pair_ranks", &self.pair_ranks.is_some())
            .field("byte_remapping", &self.byte_remapping.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_tokenizer::tiktoken::load_tiktoken;
    use std::io::Read;

    /// Token-for-token differential of the cached encode path
    /// (`memoized_encode`: packed keys, open-addressing table, inline
    /// values, prefetch pipeline) against the uncached reference
    /// (`encode_pretoken`, plain BPE merge per pretoken) on real OWT.
    #[test]
    #[ignore]
    fn memoized_encode_matches_reference_owt() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/gpt2_tokenizer.json");
        let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("load GPT-2 tokenizer");

        let data_dir = std::env::home_dir().unwrap().join("data");
        let all = std::fs::read(data_dir.join("owt_train.txt")).expect("read OWT");
        let mut end = 50_000_000.min(all.len());
        while end > 0 && std::str::from_utf8(&all[..end]).is_err() {
            end -= 1;
        }
        let input = &all[..end];

        let mut cached: Vec<TokenId> = Vec::new();
        tokenizer
            .memoized_encode(crate::pretokenize::pretokenize_as_iter(input), |tokens| {
                cached.extend_from_slice(tokens)
            });

        // Uncached reference: remap bytes and run the plain merge loop.
        let encode_reference = |pretoken: Pretoken| -> Vec<TokenId> {
            let mut symbols: Vec<TokenId> = match tokenizer.byte_remapping.as_ref() {
                Some(br) => pretoken.iter().map(|&b| br.mapping[b as usize]).collect(),
                None => pretoken.iter().map(|&b| TokenId::from(b as u32)).collect(),
            };
            crate::bpe::bpe_merge_symbols(&tokenizer.merges, &mut symbols);
            symbols
        };
        let mut idx = 0usize;
        for (pi, pretoken) in crate::pretokenize::pretokenize_as_iter(input).enumerate() {
            let reference = encode_reference(pretoken);
            assert!(
                cached[idx..(idx + reference.len()).min(cached.len())] == reference[..],
                "pretoken {pi} ({:?}) diverged: cached {:?} vs reference {:?}",
                String::from_utf8_lossy(pretoken.0),
                &cached[idx..(idx + reference.len()).min(cached.len())],
                reference,
            );
            idx += reference.len();
        }
        assert_eq!(idx, cached.len(), "cached encode produced extra tokens");
        eprintln!("all {idx} tokens match on {} MB", input.len() / 1_000_000);
    }

    /// GPT-2 must take the PairRankTable fast path, with the table agreeing
    /// with the merges map, and the vocab seed must serve every short vocab
    /// word as its own ID (what the removed reverse-vocab probe returned).
    #[test]
    fn gpt2_pair_rank_table_and_vocab_seed() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        use crate::pretokenize::{pack_pretoken_key, pretoken_key_hash};
        let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/gpt2_tokenizer.json");
        let tokenizer = load_hf_bpe(&tokenizer_path).expect("load GPT-2 tokenizer");

        let table = tokenizer
            .pair_ranks
            .as_deref()
            .expect("GPT-2 must take the pair-rank fast path");
        for (&(a, b), &m) in tokenizer.merges.iter() {
            assert_eq!(table.rank(a, b), m.0, "pair ({}, {})", a.0, b.0);
        }
        // Dense negatives (byte × byte) and flat negatives must agree with
        // the map too.
        for a in (0..50257u32).step_by(97) {
            for b in (0..50257u32).step_by(89) {
                let expected = tokenizer
                    .merges
                    .get(&(TokenId(a), TokenId(b)))
                    .map_or(u32::MAX, |m| m.0);
                assert_eq!(table.rank(TokenId(a), TokenId(b)), expected, "pair ({a}, {b})");
            }
        }

        let mut seeded = 0usize;
        for (id, bytes) in tokenizer.vocab_entries() {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).unwrap();
            let (val, ext) = tokenizer
                .pretoken_cache
                .get(key, pretoken_key_hash(key))
                .expect("short vocab entry must be seeded");
            // Every real vocab ID is < 2^24, so the seed is inline: 1 token,
            // the entry's own ID (duplicates resolve to the highest ID, but
            // GPT-2 has none).
            assert_eq!(val, 1 | ((id as u64) << 8), "vocab entry {id}");
            assert_eq!(ext, 0, "vocab entry {id}");
            seeded += 1;
        }
        assert_eq!(tokenizer.pretoken_cache.len(), seeded);
        // A fork starts from the same seed, sharing the same table.
        let fork = tokenizer.fork();
        assert_eq!(fork.pretoken_cache.len(), seeded);
        assert!(fork.pair_ranks.is_some());
    }

    #[test]
    fn short_pretoken_cache_serves_repeated_pretokens() {
        use crate::pretokenize::{SpanIter, pack_pretoken_key, pretoken_key_hash};

        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        let mut tokenizer = Tokenizer::new(merges, vocab, None);
        let bytes = b"hello";

        let mut first = Vec::new();
        tokenizer.memoized_encode(SpanIter([Pretoken(bytes)].into_iter()), |tokens| {
            first.extend(tokens.iter().map(|token| token.0));
        });
        let expected: Vec<u32> = bytes.iter().map(|&byte| byte as u32).collect();
        assert_eq!(first, expected);

        // The 5-token encoding is too long to inline, so it spilled to the
        // arena, but the cache entry serves it either way.
        let key = pack_pretoken_key(bytes).unwrap();
        let h = pretoken_key_hash(key);
        assert!(tokenizer.pretoken_cache.get(key, h).is_some());

        let mut repeated = Vec::new();
        tokenizer.memoized_encode(SpanIter([Pretoken(bytes)].into_iter()), |tokens| {
            repeated.extend(tokens.iter().map(|token| token.0));
        });
        assert_eq!(repeated, first);
        // The 256 single-byte vocab entries are pre-seeded; "hello" is the
        // one entry the encodes added.
        assert_eq!(tokenizer.pretoken_cache.len(), 257);

        // The zero key marks empty slots in the short table, so an empty
        // pretoken (possible through the public API) must take the long-map
        // path.
        tokenizer.memoized_encode(SpanIter([Pretoken(b"")].into_iter()), |tokens| {
            assert!(tokens.is_empty());
        });
        assert!(tokenizer.pretoken_cache_long.contains_key(&b""[..]));
    }

    #[test]
    fn concrete_pretokenizer_dispatch_matches_enum_dispatch() {
        let schemes = [
            PretokenizerType::GPT2,
            PretokenizerType::GPT4,
            PretokenizerType::Qwen2,
            PretokenizerType::Qwen35,
            PretokenizerType::Olmo3,
            PretokenizerType::DeepSeekV3,
        ];
        let input = "Hello, 世界! café 12345\r\ncan't  stop".as_bytes();

        for scheme in schemes {
            let make_tokenizer = || {
                let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
                let vocab = (0..=u8::MAX).map(|byte| vec![byte]).collect();
                Tokenizer::new(merges, vocab, None)
            };

            let mut reference = make_tokenizer();
            let mut expected = Vec::new();
            reference.memoized_encode(scheme.pretokenize(input), |tokens| {
                expected.extend_from_slice(tokens);
            });

            let mut concrete = make_tokenizer();
            concrete.set_pretokenizer_type(scheme);
            let mut actual = Vec::new();
            concrete.encode_with_added_tokens(input, |tokens| {
                actual.extend_from_slice(tokens);
            });
            assert_eq!(actual, expected, "dispatch differs for {scheme:?}");
        }
    }

    #[test]
    fn test_merges_from_vocab() {
        use base64::prelude::*;
        let mut buf = String::new();
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        std::fs::File::open(tiktoken_path)
            .expect("Didn't find file")
            .read_to_string(&mut buf)
            .unwrap();
        let vocab: Vec<Vec<u8>> = buf
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let (base64_token, id_str) = line.split_once(' ').unwrap();
                let id = id_str.trim().parse::<u32>().unwrap();
                assert!(id == i as u32);
                
                BASE64_STANDARD.decode(base64_token).unwrap()
            })
            .collect();
        for (i, token) in vocab.iter().enumerate().skip(256).take(20) {
            eprintln!("{i}: {:?}", String::from_utf8_lossy(token));
        }
        let tokenizer = Tokenizer::from_ranks(vocab).unwrap();

        let merges_inv = tokenizer
            .merges
            .iter()
            .map(|((a, b), c)| (*c, (*a, *b)))
            .collect::<HashMap<TokenId, (TokenId, TokenId)>>();

        let decode_token = |token_id: TokenId| -> String {
            String::from_utf8_lossy(&tokenizer.vocab[token_id.0 as usize]).into_owned()
        };

        eprintln!("Merges:");
        for i in 256..=300 {
            let (a, b) = *merges_inv.get(&i.into()).unwrap();
            eprintln!(
                "Merge {i}: \"{}\" + \"{}\" -> \"{}\"",
                decode_token(a),
                decode_token(b),
                decode_token(i.into()),
            )
        }
    }

    #[test]
    fn basic_tokenization() {
        let text = "This is a test string. Please tokenize it!";
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        let mut tokenizer = load_tiktoken(tiktoken_path).expect("Failed to load tokenizer");
        let pretokenize_iter = crate::pretokenize::pretokenize_as_iter(text.as_bytes());
        let mut output = vec![];
        tokenizer.memoized_encode(pretokenize_iter, |tokens| {
            output.extend_from_slice(tokens);
        });
        assert!(tokenizer.byte_remapping.is_some());
        println!("Encoded: {:?}", output);
        let decoded = tokenizer.decode(&output).collect::<Vec<u8>>();
        println!("Decoded: {:?}", String::from_utf8_lossy(&decoded));
    }
}
