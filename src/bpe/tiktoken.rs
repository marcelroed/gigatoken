use crate::bpe::pretoken_cache::ShortPretokenCache;
use crate::bpe::{
    ByteRemapping, MergeScratch, PairRankTable, SHORT_MERGE_MAX, bpe_merge_symbols_by_rank,
    bpe_merge_symbols_short_scalar, bpe_merge_symbols_with_scratch, simple_bpe_merge,
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
    /// HF BPE `ignore_merges`: a pretoken whose whole byte string is a
    /// vocab entry encodes as that single ID without running the merge
    /// loop. Matters when the vocab has whole-word entries the merges
    /// would decompose differently (GLM-5.2 has ~97k such words); a plain
    /// merge walk diverges from HF on those.
    ignore_merges: bool,
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

/// Overwrite the short-cache entry of every added-token content of 1..=15
/// bytes that resolves in `vocab_inv` with that single ID, so a matching
/// pretoken encodes as the added token rather than its merge decomposition.
///
/// This function IS the cache-seed sync invariant: the short cache's
/// seed-level state is always "vocab seed, then these overwrites" — a pure
/// function of `(vocab, added_tokens)` — because both
/// [`Tokenizer::set_added_tokens`] (on the parent) and
/// [`Tokenizer::fork_sized`] (after a fork's fresh reseed) apply the
/// overwrites through this one body, so parent and forked workers always
/// agree on every short pretoken.
/// One piece of the added-token pipeline walk (see
/// [`Tokenizer::for_each_piece`]): a between-occurrences text segment to
/// pretokenize and encode, or an added token's ID to emit verbatim.
enum Piece<'a> {
    Segment(&'a [u8]),
    Added(TokenId),
}

fn apply_added_token_overwrites(
    added_tokens: &[(Arc<[u8]>, TokenId)],
    vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
    pretoken_cache: &mut ShortPretokenCache,
    token_arena: &mut Vec<TokenId>,
) {
    for (content, _) in added_tokens {
        if !(1..=15).contains(&content.len()) {
            continue;
        }
        let Some(&id) = vocab_inv.get(content) else {
            continue;
        };
        let key = pack_pretoken_key(content).expect("length checked <= 15");
        let h = pretoken_key_hash(key);
        let (val, ext) = Tokenizer::pack_val(&[id], token_arena);
        pretoken_cache.replace(key, h, val, ext);
    }
}

impl Tokenizer {
    pub fn new(
        merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect();
        Self::from_tables(merges, vocab, byte_remapping)
    }

    /// Shared construction tail ([`Self::new`] and [`Self::from_ranks`]):
    /// derive `vocab_inv` and the pair-rank table from the finished
    /// merges/vocab, seed the pretoken cache, and assemble the tokenizer
    /// with default pipeline settings (GPT-2 pretokenization, no added
    /// tokens, no NFC).
    fn from_tables(
        merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        vocab: Vec<Arc<[u8]>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher> = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();
        let pair_ranks =
            PairRankTable::build(&merges, byte_remapping.as_ref(), vocab.len()).map(Arc::new);
        let mut token_arena = Vec::new();
        let pretoken_cache = Self::seeded_pretoken_cache(
            &vocab,
            byte_remapping.as_ref(),
            pair_ranks.as_deref(),
            &merges,
            false,
            &vocab_inv,
            &mut token_arena,
            0,
        );
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
            ignore_merges: false,
        }
    }

    /// A short-pretoken cache pre-seeded with the BPE encoding of every
    /// vocab entry of 1..=15 bytes: precomputed miss results, computed by
    /// the same [`Self::merge_short`] the miss path runs, so a seeded
    /// value is bit-identical to what a cold miss on those bytes would
    /// have produced and cached. Any short pretoken that is a whole vocab
    /// word then hits the cache outright, so the miss path never sees one.
    ///
    /// Without `ignore_merges`, the seed value must be the MERGE RESULT,
    /// not the entry's own ID: BPE encode semantics (HF `tokenizers`
    /// without `ignore_merges`, this repo's merge loop, and the pre-cache
    /// baseline 0e27c71) produce a whole-word token only when the merge
    /// rules can derive it, and vocabs may contain merge-UNREACHABLE
    /// entries — qwen3_5 has ~200 (multi-char CJK phrases, " Jap\u{f3}n",
    /// …) that must encode as their merge decomposition. Seeding
    /// `bytes -> [own id]` was a measured divergence from HF (see
    /// `verify_vocab_seeded_cache_matches_merge_decomposition`). For
    /// merge-reachable entries — all of gpt2/olmo3/qwen2/deepseek_v3 —
    /// the merge result is the single own ID, as before. Duplicate byte
    /// strings encode identically (the merge sees only bytes), so the
    /// insert-if-absent dedup is purely a work-skip.
    ///
    /// WITH `ignore_merges` the rule flips: HF emits the vocab entry's own
    /// ID for any whole-pretoken vocab hit, so every seed value is
    /// `[vocab_inv[bytes]]` (`vocab_inv` also resolves duplicate byte
    /// strings to the one ID a lookup would find).
    ///
    /// `min_slots` additionally floors the table size for a worker with a
    /// known workload (see [`Self::fork_sized`]); the table is built once
    /// at the max of the seed requirement and that floor, so seeding never
    /// grows it mid-way. Values of 5+ tokens (only possible for
    /// merge-unreachable entries) spill into `token_arena` like any other
    /// miss.
    fn seeded_pretoken_cache(
        vocab: &[Arc<[u8]>],
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        token_arena: &mut Vec<TokenId>,
        min_slots: usize,
    ) -> ShortPretokenCache {
        let n_short = vocab
            .iter()
            .filter(|bytes| (1..=15).contains(&bytes.len()))
            .count();
        let mut cache = ShortPretokenCache::with_at_least(n_short, min_slots);
        let mut buf = [TokenId(0); SHORT_MERGE_MAX];
        for bytes in vocab {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            // Duplicate byte strings seed the same value (see the doc
            // above), so insertion order is irrelevant and the
            // insert-if-absent check only skips redundant merges.
            if cache.get_or_slot(key, h).is_err() {
                let n = Self::seed_symbols(
                    byte_remapping,
                    pair_ranks,
                    merges,
                    ignore_merges,
                    vocab_inv,
                    bytes,
                    &mut buf,
                );
                let (val, ext) = Self::pack_val(&buf[..n], token_arena);
                cache.insert(key, h, val, ext);
            }
        }
        cache
    }

    /// Seed-level encoding of one short vocab byte string under the
    /// current `ignore_merges` setting: the single `vocab_inv` ID when the
    /// flag is set (HF's whole-pretoken vocab hit), the merge
    /// decomposition otherwise. One body shared by
    /// [`Self::seeded_pretoken_cache`] and [`Self::set_ignore_merges`] so
    /// a fork's fresh reseed and the parent's in-place rewrite always
    /// agree.
    #[inline]
    fn seed_symbols(
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        ignore_merges: bool,
        vocab_inv: &HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        if ignore_merges {
            if let Some(&id) = vocab_inv.get(bytes) {
                buf[0] = id;
                return 1;
            }
        }
        Self::merge_short(byte_remapping, pair_ranks, merges, bytes, buf)
    }

    /// BPE-encode one short pretoken (1..=15 bytes) into `buf`, returning
    /// its token count: byte remapping, then the short merge loop. This is
    /// exactly the computation [`Self::encode_pretoken_miss`] performs for
    /// short keys — shared with [`Self::seeded_pretoken_cache`] so the
    /// vocab seed can never disagree with a cold miss.
    #[inline]
    fn merge_short(
        byte_remapping: Option<&ByteRemapping>,
        pair_ranks: Option<&PairRankTable>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        bytes: &[u8],
        buf: &mut [TokenId; SHORT_MERGE_MAX],
    ) -> usize {
        let n = bytes.len();
        debug_assert!((1..SHORT_MERGE_MAX).contains(&n));
        match byte_remapping {
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
        if n < 2 {
            return n;
        }
        match pair_ranks {
            #[cfg(target_arch = "aarch64")]
            Some(table) => bpe_merge_symbols_short_neon(table, buf, n),
            // x86-64 stays scalar ON PURPOSE: the AVX-512/AVX2 ports of the
            // min-rank scan (`bpe_merge_symbols_short_avx512/_avx2`, kept as
            // tested reference) measured ~1% SLOWER on cold encode_st (Zen 5,
            // gpt2, 100 MB and 1 GB OWT, interleaved min-of-5) — the x86
            // horizontal reduce is a 4-step dependent chain plus a
            // vector->GPR transfer on the serial merge chain, and the
            // `target_feature` boundary blocks inlining, while the scalar
            // scan's `rank < best` branches predict well on Zen 5. See
            // profiling/x86_port_plan.md §6.
            #[cfg(not(target_arch = "aarch64"))]
            Some(table) => bpe_merge_symbols_short_scalar(
                |a, b| table.rank(a, b),
                |a, b| table.prefetch_rank(a, b),
                buf,
                n,
            ),
            None => bpe_merge_symbols_short_scalar(
                |a, b| merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                |_, _| {},
                buf,
                n,
            ),
        }
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
        Ok(Self::from_tables(merges, vocab, byte_remapping))
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
        // Size for the Heaps estimate at the table's 3/4 growth load with
        // 1.4x headroom (self-paced chunk handout lets a fast core encode
        // more than its even share; the margin holds a >2x-oversubscribed
        // worker under the growth threshold before the table would resize).
        // Still a capacity hint: the table grows past it at 3/4 load on
        // corpora more diverse than the OWT calibration. Clamped to 2^22
        // slots (128 MB) per worker.
        let distinct = 3.45 * (expected_bytes as f64).powf(0.62);
        let cache_slots = ((distinct * (4.0 / 3.0) * 1.4) as usize)
            .clamp(1 << 16, 1 << 22)
            .next_power_of_two();
        let arena_cap = (expected_bytes / 256).min(1 << 24);
        let long_cap = (expected_bytes / 8192).min(1 << 20);
        let mut token_arena = Vec::with_capacity(arena_cap);
        let mut pretoken_cache = Self::seeded_pretoken_cache(
            &self.vocab,
            self.byte_remapping.as_ref(),
            self.pair_ranks.as_deref(),
            &self.merges,
            self.ignore_merges,
            &self.vocab_inv,
            &mut token_arena,
            cache_slots,
        );
        // The vocab seed above holds the plain seed encoding of every short
        // byte string (merge result, or own ID under `ignore_merges`);
        // re-apply the added-token `[id]` overwrites so the fork's cache
        // matches the parent's seed-level state (the shared function is the
        // sync invariant — see `apply_added_token_overwrites`).
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut pretoken_cache,
            &mut token_arena,
        );
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
            ignore_merges: self.ignore_merges,
        }
    }

    /// Loader-phase mutator: like every `Tokenizer` mutation, this must
    /// run before any `WorkerPool` forks workers from this tokenizer —
    /// already-forked workers keep the old state (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
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

    /// Enable HF BPE `ignore_merges` semantics: a pretoken whose whole
    /// byte string is a vocab entry encodes as that single ID, skipping
    /// the merge loop.
    ///
    /// Rewrites the vocab-seeded short-cache entries to the new flag's
    /// seed values (own ID vs merge decomposition — see
    /// [`Self::seed_symbols`]) and reasserts the added-token overwrites,
    /// so the cache stays a pure function of
    /// `(vocab, ignore_merges, added_tokens)` and matches what a fork's
    /// fresh reseed produces.
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// state (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
    pub fn set_ignore_merges(&mut self, ignore_merges: bool) {
        if self.ignore_merges == ignore_merges {
            return;
        }
        self.ignore_merges = ignore_merges;
        let mut buf = [TokenId(0); SHORT_MERGE_MAX];
        for bytes in self.vocab.iter() {
            if !(1..=15).contains(&bytes.len()) {
                continue;
            }
            let key = pack_pretoken_key(bytes).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            let n = Self::seed_symbols(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                ignore_merges,
                &self.vocab_inv,
                bytes,
                &mut buf,
            );
            let (val, ext) = Self::pack_val(&buf[..n], &mut self.token_arena);
            self.pretoken_cache.replace(key, h, val, ext);
        }
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut self.pretoken_cache,
            &mut self.token_arena,
        );
    }

    /// Set the added tokens matched atomically by
    /// [`Self::encode_with_added_tokens`]. Empty contents are ignored.
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// added-token set (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
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
        let outgoing = std::mem::replace(&mut self.added_tokens, added_tokens);
        // Restore the plain seed value for the outgoing set first (only
        // contents that resolve in `vocab_inv` were ever overwritten, so
        // this replaces existing entries and never inserts), then apply
        // the incoming overwrites through the shared sync-invariant body
        // (see `apply_added_token_overwrites`).
        for (content, _) in &outgoing {
            if !(1..=15).contains(&content.len()) || self.vocab_inv.get(content).is_none() {
                continue;
            }
            let key = pack_pretoken_key(content).expect("length checked <= 15");
            let h = pretoken_key_hash(key);
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = Self::seed_symbols(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                self.ignore_merges,
                &self.vocab_inv,
                content,
                &mut buf,
            );
            let (val, ext) = Self::pack_val(&buf[..n], &mut self.token_arena);
            self.pretoken_cache.replace(key, h, val, ext);
        }
        apply_added_token_overwrites(
            &self.added_tokens,
            &self.vocab_inv,
            &mut self.pretoken_cache,
            &mut self.token_arena,
        );
    }

    /// Register one additional added token, extending the decode vocab when
    /// its id lies outside the base ranks (mirrors the out-of-vocab
    /// added-token handling in the HF loader).
    ///
    /// Loader-phase mutator: must run before any `WorkerPool` forks
    /// workers from this tokenizer — already-forked workers keep the old
    /// vocab, matcher, and cache seed (see [`WorkerPool`]).
    ///
    /// [`WorkerPool`]: crate::batch::WorkerPool
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
            // If `content` duplicates an already-present vocab byte
            // string, `vocab_inv` switches to the new ID (unconditional
            // overwrite). The short-cache overwrite that keeps a matching
            // pretoken resolving to `vocab_inv`'s answer happens in
            // `set_added_tokens` below, which re-derives every added-token
            // cache overwrite from the updated `vocab_inv` — the same
            // computation a fork's reseed + re-apply performs (see
            // [`Self::fork_sized`]), so parent and forked workers agree.
            Arc::make_mut(&mut self.vocab_inv).insert(vocab[idx].clone(), id);
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

    /// Shared piece walk of the added-token pipeline: split out added-token
    /// occurrences and hand each piece — the (possibly NFC-normalized)
    /// segment between occurrences, or the added token's ID — to `f` in
    /// input order. Scheme dispatch costs one enum match per 256-pretoken
    /// chunk fill (see [`PretokenizerType::pretokenize`] and
    /// `FastPretokenizerDispatch::fill_spans_keyed`), which delegates to
    /// the same out-of-line concrete fills a hardcoded pretokenizer uses.
    fn for_each_piece(&mut self, bytes: &[u8], mut f: impl FnMut(&mut Self, Piece<'_>)) {
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
            f(self, Piece::Segment(segment));
            match added {
                Some((end, id)) => {
                    f(self, Piece::Added(id));
                    pos = end;
                }
                None => break,
            }
        }
    }

    /// Encode raw text: split out added-token occurrences (emitted as their
    /// single token ID), pretokenize the segments between them with this
    /// tokenizer's pretokenization scheme, and BPE-encode each pretoken.
    /// This mirrors the full HuggingFace `tokenizers` encode pipeline.
    pub fn encode_with_added_tokens(&mut self, bytes: &[u8], mut f: impl FnMut(&[TokenId])) {
        let pt = self.pretokenizer_type;
        self.for_each_piece(bytes, |this, piece| match piece {
            Piece::Segment(segment) => this.memoized_encode(pt.pretokenize(segment), &mut f),
            Piece::Added(id) => f(&[id]),
        });
    }

    /// Flat variant of [`Self::encode_with_added_tokens`]: the identical
    /// token stream appended to `out` as raw u32 ids, routed through
    /// [`Self::memoized_encode_flat`] so segment tokens land directly in
    /// the caller's buffer (the batch engine's per-chunk id buffer).
    pub fn encode_with_added_tokens_flat(&mut self, bytes: &[u8], out: &mut Vec<u32>) {
        let pt = self.pretokenizer_type;
        self.for_each_piece(bytes, |this, piece| match piece {
            Piece::Segment(segment) => this.memoized_encode_flat(pt.pretokenize(segment), out),
            Piece::Added(id) => out.push(id.0),
        });
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
            // the array and no per-pretoken bounds clamp is needed — the
            // load is one fixed-offset ldr off the walking entry pointer.
            // Tail iterations prefetch stale or zero `meta`, and long
            // entries a length, not a hash — either way a masked,
            // in-bounds table line: harmless.
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
            // misses): straight to byte symbols and the merge loop
            // (`merge_short`, shared with the vocab seed), in a stack
            // buffer instead of the `Vec` scratch. The cache is pre-seeded
            // with the seed encoding of every short vocab entry (see
            // `seeded_pretoken_cache`), so a miss here is never a whole
            // vocab word — but nothing depends on that: `seed_symbols`
            // computes the correct encoding for any bytes under either
            // `ignore_merges` setting.
            let mut buf = [TokenId(0); SHORT_MERGE_MAX];
            let n = Self::seed_symbols(
                self.byte_remapping.as_ref(),
                self.pair_ranks.as_deref(),
                &self.merges,
                self.ignore_merges,
                &self.vocab_inv,
                bytes,
                &mut buf,
            );
            let symbols = &buf[..n];
            let (val, ext) = Self::pack_val(symbols, &mut self.token_arena);
            self.pretoken_cache.insert_at(slot, key, h, val, ext);
            out.extend_from_slice(token_ids_as_u32s(symbols));
        } else {
            // Long pretoken (> 15 bytes, rare): remap and run the merge
            // loop. Without `ignore_merges`, deliberately NO
            // whole-pretoken reverse-vocab (`vocab_inv`) shortcut here — a
            // vocab entry is not guaranteed to be derivable from its own
            // merges (qwen3_5 has ~50 entries > 15 bytes, multi-char CJK
            // phrases, that HF `tokenizers` without `ignore_merges`
            // encodes as their merge decomposition, never as the single
            // ID), so any such shortcut diverges from HF and from the
            // pre-cache baseline. The same rule holds for short keys via
            // the seeded merge results above. Do not reintroduce it. With
            // `ignore_merges` set, the shortcut IS HF's semantics, so it
            // applies — gated on the flag.
            let symbols = &mut self.symbol_scratch;
            symbols.clear();
            if self.ignore_merges
                && let Some(&id) = self.vocab_inv.get(bytes)
            {
                symbols.push(id);
            } else {
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(bytes.iter().map(|&b| br.mapping[b as usize])),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                match self.pair_ranks.as_deref() {
                    Some(table) => bpe_merge_symbols_by_rank(
                        &|a, b| table.rank(a, b),
                        symbols,
                        &mut self.merge_scratch,
                    ),
                    None => bpe_merge_symbols_with_scratch(
                        &self.merges,
                        symbols,
                        &mut self.merge_scratch,
                    ),
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

/// Helpers shared by the test modules below.
#[cfg(test)]
mod test_util {
    use super::*;

    pub(super) fn gpt2_path() -> std::path::PathBuf {
        crate::test_hub::gpt2_tokenizer_json()
    }

    /// Uncached reference encode of one pretoken: byte remap + plain merge
    /// loop over the merges HashMap (no pair-rank table, no cache, no
    /// short-merge kernels).
    pub(super) fn plain_encode_pretoken(tok: &Tokenizer, pretoken: &[u8], out: &mut Vec<u32>) {
        let mut symbols: Vec<TokenId> = match tok.byte_remapping.as_ref() {
            Some(br) => pretoken.iter().map(|&b| br.mapping[b as usize]).collect(),
            None => pretoken.iter().map(|&b| TokenId::from(b as u32)).collect(),
        };
        crate::bpe::bpe_merge_symbols(&tok.merges, &mut symbols);
        out.extend(symbols.iter().map(|t| t.0));
    }

    /// Pretoken lengths through the two-phase walker path
    /// (`fill_spans_keyed`), for comparison against the Iterator path.
    pub(super) fn two_phase_lens<'a>(mut p: impl PretokenSpans<'a>) -> Vec<usize> {
        let mut batch = SpanBatch::new();
        let mut lens = Vec::new();
        loop {
            let n = p.fill_spans_keyed(&mut batch, &|_| {});
            for i in 0..n {
                lens.push(batch.entries[i].span_len());
            }
            if n < PRETOKEN_CHUNK {
                return lens;
            }
        }
    }

    /// xorshift64: deterministic, dependency-free RNG for test inputs.
    pub(super) struct XorShift64(pub u64);

    impl XorShift64 {
        pub(super) fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_tokenizer::tiktoken::load_tiktoken;
    use std::io::Read;

    /// `add_special_token` whose content duplicates an existing vocab byte
    /// string must resolve to the added ID everywhere: `vocab_inv`, the
    /// parent's seeded cache (overwritten, not insert-if-absent), and
    /// forked workers (vocab reseed plus re-applied added-token
    /// overwrites). Regression test for the three-way disagreement where
    /// the parent kept the stale seed entry (old ID) while a fork's
    /// descending reseed picked the new ID.
    #[test]
    fn add_special_token_duplicate_content_agrees_across_forks() {
        let encode = |t: &mut Tokenizer, input: &[u8]| -> Vec<TokenId> {
            let mut out = Vec::new();
            t.memoized_encode(crate::pretokenize::pretokenize_as_iter(input), |tokens| {
                out.extend_from_slice(tokens)
            });
            out
        };

        // Case 1: added ID above the duplicate's ID.
        let mut merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher> =
            HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        merges.insert((TokenId(104), TokenId(105)), TokenId(256)); // 'h' 'i' -> "hi"
        let mut vocab: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
        vocab.push(b"hi".to_vec()); // id 256 = "hi"
        let mut tok = Tokenizer::new(merges, vocab, None);
        tok.add_special_token(b"hi".to_vec(), TokenId(1000));
        assert_eq!(tok.vocab_inv.get(b"hi".as_slice()), Some(&TokenId(1000)));
        let mut fork = tok.fork();
        assert_eq!(
            encode(&mut tok, b"hi"),
            vec![TokenId(1000)],
            "parent cache must resolve the duplicate to the added ID (vocab_inv's answer)"
        );
        assert_eq!(
            encode(&mut fork, b"hi"),
            vec![TokenId(1000)],
            "forked worker must agree with the parent"
        );

        // Case 2 (mirror): added ID fills an empty placeholder BELOW the
        // duplicate's ID; the fork's reseed alone would pick the higher
        // ID (the merge result), diverging from vocab_inv and the parent.
        let mut merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher> =
            HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        merges.insert((TokenId(104), TokenId(105)), TokenId(257)); // 'h' 'i' -> "hi"
        let mut vocab: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
        vocab.push(Vec::new()); // id 256: empty placeholder
        vocab.push(b"hi".to_vec()); // id 257 = "hi"
        let mut tok = Tokenizer::new(merges, vocab, None);
        tok.add_special_token(b"hi".to_vec(), TokenId(256));
        assert_eq!(tok.vocab_inv.get(b"hi".as_slice()), Some(&TokenId(256)));
        let mut fork = tok.fork();
        assert_eq!(encode(&mut tok, b"hi"), vec![TokenId(256)]);
        assert_eq!(encode(&mut fork, b"hi"), vec![TokenId(256)]);
    }

    /// GPT-2 must take the PairRankTable fast path, with the table agreeing
    /// with the merges map, and the vocab seed must serve every short vocab
    /// word as its own ID: every base GPT-2 entry is merge-reachable, so
    /// its merge decomposition IS the single own ID, and the one
    /// unreachable entry (<|endoftext|>, an added token) gets the
    /// `set_added_tokens` `[id]` overwrite.
    #[test]
    fn gpt2_pair_rank_table_and_vocab_seed() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        use crate::pretokenize::{pack_pretoken_key, pretoken_key_hash};
        let tokenizer = load_hf_bpe(super::test_util::gpt2_path()).expect("load GPT-2 tokenizer");

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
                .get_or_slot(key, pretoken_key_hash(key))
                .expect("short vocab entry must be seeded");
            // Every real vocab ID is < 2^24, so the seed is inline: 1 token,
            // the entry's own ID (GPT-2 has no duplicate byte strings and,
            // added-token overwrites included, no entry whose cached value
            // differs from its own ID).
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
        assert!(tokenizer.pretoken_cache.get_or_slot(key, h).is_ok());

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

    /// With no added tokens configured, `encode_with_added_tokens`'s piece
    /// walk reduces to one whole-input segment — its output must equal a
    /// direct `memoized_encode` of the same scheme's pretokens.
    #[test]
    fn encode_with_added_tokens_matches_memoized_encode_all_schemes() {
        let schemes = [
            PretokenizerType::GPT2,
            PretokenizerType::GPT4,
            PretokenizerType::Qwen2,
            PretokenizerType::Qwen35,
            PretokenizerType::Olmo3,
            PretokenizerType::DeepSeekV3,
            PretokenizerType::O200k,
            PretokenizerType::Nemotron,
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

/// Heavy correctness differentials for the optimized encode pipeline
/// (from the opt/verify-heavy campaign branch). The OWT-scale tests
/// (`#[ignore]`d; each doc comment states corpus size, runtime ballpark,
/// and the cargo command) check the CACHED paths (`memoized_encode` /
/// `encode_with_added_tokens_flat`: packed keys, open-addressing short
/// table, two-phase span walkers, branchless emit) against an UNCACHED
/// reference (per-pretoken plain `bpe_merge_symbols` over the merges
/// HashMap). Two fast tests probe `pack_pretoken_key` at every page
/// offset and pin the vocab-seed merge-decomposition rule. Boundary fuzz
/// and walker edge cases live in `walker_edge` below.
#[cfg(test)]
mod verify_heavy {
    use super::test_util::{XorShift64, gpt2_path, plain_encode_pretoken};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;
    use std::io::Read;

    fn load_owt(max_bytes: usize) -> Vec<u8> {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("open ~/data/owt_train.txt");
        let mut input = Vec::new();
        f.take(max_bytes as u64).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        input
    }

    /// Cut `input` at the first newline at or after `at` (whole input if none).
    fn cut_at_newline(input: &[u8], at: usize) -> &[u8] {
        let at = at.min(input.len());
        match memchr::memchr(b'\n', &input[at..]) {
            Some(off) => &input[..at + off + 1],
            None => input,
        }
    }

    /// Token-for-token comparison of the full cached public path
    /// (`encode_with_added_tokens_flat`) against an uncached mirror of its
    /// segment loop (added-token split + optional NFC + per-pretoken plain
    /// merge). Panics with byte offset, pretoken bytes, and both id streams
    /// on the first divergence.
    fn compare_cached_vs_reference(tok: &mut Tokenizer, input: &[u8], label: &str, verbose: bool) {
        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(input, &mut cached);

        let mut idx = 0usize;
        let mut nfc_buf = String::new();
        let mut scratch: Vec<u32> = Vec::new();
        let mut pos = 0usize;
        loop {
            let (seg_end, added) = match tok.find_added_token(input, pos) {
                Some((s, e, id)) => (s, Some((e, id))),
                None => (input.len(), None),
            };
            let segment = if tok.normalize_nfc {
                nfc_segment(&input[pos..seg_end], &mut nfc_buf)
            } else {
                &input[pos..seg_end]
            };
            let mut seg_off = 0usize;
            for pretoken in tok.pretokenizer_type.pretokenize(segment) {
                scratch.clear();
                plain_encode_pretoken(tok, pretoken.0, &mut scratch);
                let got = cached.get(idx..idx + scratch.len());
                if got != Some(&scratch[..]) {
                    let byte_off = pos + seg_off; // exact unless NFC changed lengths
                    let ctx_start = byte_off.saturating_sub(40).min(input.len());
                    let ctx_end = (byte_off + pretoken.0.len() + 40).min(input.len());
                    panic!(
                        "{label}: encode mismatch at byte offset ~{byte_off} (input len {}), token index {idx}\n  \
                         pretoken ({} bytes): {:?}\n  expected ids: {:?}\n  cached ids:   {:?}\n  context: {:?}",
                        input.len(),
                        pretoken.0.len(),
                        String::from_utf8_lossy(pretoken.0),
                        scratch,
                        &cached[idx.min(cached.len())..(idx + scratch.len() + 4).min(cached.len())],
                        String::from_utf8_lossy(&input[ctx_start..ctx_end]),
                    );
                }
                idx += scratch.len();
                seg_off += pretoken.0.len();
            }
            match added {
                Some((end, id)) => {
                    assert_eq!(
                        cached.get(idx).copied(),
                        Some(id.0),
                        "{label}: added-token id mismatch at bytes {pos}..{end}"
                    );
                    idx += 1;
                    pos = end;
                }
                None => break,
            }
        }
        assert_eq!(
            idx,
            cached.len(),
            "{label}: cached stream has {} extra trailing tokens",
            cached.len() - idx
        );
        if verbose {
            eprintln!(
                "{label}: all {idx} tokens match on {:.1} MB",
                input.len() as f64 / 1e6
            );
        }
    }

    /// Independent added-token differential: join ~1 MB corpus pieces with
    /// the tokenizer's first added token and check the public encode equals
    /// concat(plain-encode(piece), sep_id, ...) — the expected stream is
    /// built WITHOUT `find_added_token`, so the Aho-Corasick split itself
    /// is under test, not just mirrored.
    fn join_differential(tok: &mut Tokenizer, corpus: &[u8], label: &str) {
        let Some((sep, sep_id)) = tok.added_tokens.first().map(|(c, i)| (c.to_vec(), *i)) else {
            eprintln!("{label}: no added tokens registered; skipping join differential");
            return;
        };
        // OWT embeds document separators (e.g. <|endoftext|>) throughout;
        // mask every added-token occurrence so pieces are separator-free
        // and the expected stream can be built without find_added_token.
        let mut corpus: Vec<u8> = corpus.to_vec();
        for (content, _) in tok.added_tokens.clone() {
            let hits: Vec<usize> = memchr::memmem::find_iter(&corpus, &content[..]).collect();
            for pos in hits {
                corpus[pos] = b'~';
            }
        }
        let corpus = &corpus[..];
        let mut expected: Vec<u32> = Vec::new();
        let mut joined: Vec<u8> = Vec::new();
        let mut nfc_buf = String::new();
        let mut start = 0usize;
        let mut pieces = 0usize;
        while start < corpus.len() {
            let target = (start + (1 << 20)).min(corpus.len());
            let end = match memchr::memchr(b'\n', &corpus[target..]) {
                Some(off) => target + off + 1,
                None => corpus.len(),
            };
            let piece = &corpus[start..end];
            start = end;
            // Masking is single-byte, so no new occurrence can appear; but
            // keep the guard as a belt-and-braces skip.
            if tok
                .added_tokens
                .iter()
                .any(|(c, _)| memchr::memmem::find(piece, c).is_some())
            {
                continue;
            }
            joined.extend_from_slice(piece);
            joined.extend_from_slice(&sep);
            let seg = if tok.normalize_nfc {
                nfc_segment(piece, &mut nfc_buf)
            } else {
                piece
            };
            for pretoken in tok.pretokenizer_type.pretokenize(seg) {
                plain_encode_pretoken(tok, pretoken.0, &mut expected);
            }
            expected.push(sep_id.0);
            pieces += 1;
        }
        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&joined, &mut cached);
        if cached != expected {
            let i = expected
                .iter()
                .zip(&cached)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| expected.len().min(cached.len()));
            panic!(
                "{label}: join differential diverged at token index {i} \
                 (expected len {}, cached len {}):\n  expected[{i}..] = {:?}\n  cached[{i}..]   = {:?}",
                expected.len(),
                cached.len(),
                &expected[i..(i + 8).min(expected.len())],
                &cached[i..(i + 8).min(cached.len())],
            );
        }
        assert!(pieces > 0, "{label}: join differential ran on zero pieces (vacuous)");
        eprintln!(
            "{label}: join differential ok — {pieces} pieces, {} tokens, sep {:?} id {}",
            cached.len(),
            String::from_utf8_lossy(&sep),
            sep_id.0
        );
    }

    /// Token-for-token differential of the cached callback encode path
    /// (`memoized_encode`: packed keys, open-addressing table, inline
    /// values, prefetch pipeline) against the uncached reference
    /// (plain BPE merge per pretoken) on 50 MB of OWT. Runs in a few
    /// seconds in release mode.
    /// `cargo test --release verify_memoized_encode_matches_reference_owt_50m -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 50 MB of OWT; run explicitly in release mode"]
    fn verify_memoized_encode_matches_reference_owt_50m() {
        let mut tokenizer = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let all = load_owt(50_000_000);
        let input = &all[..];

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

    /// 1 GB of OWT through the public GPT-2 encode path
    /// (`encode_with_added_tokens_flat`) vs the uncached reference, plus a
    /// 100 MB added-token join differential. About half a minute in
    /// release mode (the uncached reference dominates).
    /// `cargo test --release verify_gpt2_public_encode_matches_reference_owt_1g -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 1 GB of OWT; run explicitly in release mode"]
    fn verify_gpt2_public_encode_matches_reference_owt_1g() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let input = load_owt(1_000_000_000);
        assert!(input.len() > 900_000_000, "corpus too small: {}", input.len());
        compare_cached_vs_reference(&mut tok, &input, "gpt2-raw-1g", true);
        let mut tok2 = load_hf_bpe(gpt2_path()).unwrap();
        join_differential(&mut tok2, cut_at_newline(&input, 100_000_000), "gpt2-join-100m");
    }

    /// ~200 MB of OWT through the public encode path of every non-GPT2
    /// tokenizer whose tokenizer.json loads (olmo3, qwen2, qwen3_5,
    /// deepseek_v3), vs the uncached reference; plus a 25 MB join
    /// differential each. About half a minute in release mode.
    /// `cargo test --release verify_multi_public_encode_matches_reference_owt_200m -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 200 MB of OWT per tokenizer; run explicitly in release mode"]
    fn verify_multi_public_encode_matches_reference_owt_200m() {
        let input = load_owt(200_000_000);
        assert!(input.len() > 190_000_000, "corpus too small: {}", input.len());
        let mut ran = 0usize;
        // qwen3_5 is the load-bearing tokenizer here: its vocab has ~200
        // merge-unreachable entries (CJK phrases, " Jap\u{f3}n", …) that
        // the raw-ID vocab seed used to return as single tokens, diverging
        // from HF (see verify_vocab_seeded_cache_matches_merge_decomposition);
        // the seed now stores merge decompositions. Kept last so the clean
        // tokenizers report first on a regression.
        for (name, repo_id) in [
            ("olmo3", "allenai/Olmo-3-1025-7B"),
            ("qwen2", "Qwen/Qwen2-1.5B-Instruct"),
            ("deepseek_v3", "deepseek-ai/DeepSeek-V3"),
            ("qwen3_5", "Qwen/Qwen3.5-9B"),
        ] {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo_id) else {
                eprintln!("{name}: {repo_id} tokenizer.json not in the HF cache; skipping");
                continue;
            };
            let mut tok = match load_hf_bpe(&path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{name}: failed to load ({e}); skipping");
                    continue;
                }
            };
            eprintln!(
                "{name}: scheme {:?}, nfc {}, {} added tokens, vocab {}",
                tok.pretokenizer_type,
                tok.normalize_nfc,
                tok.added_tokens.len(),
                tok.vocab_size()
            );
            compare_cached_vs_reference(&mut tok, &input, name, true);
            let mut tok2 = load_hf_bpe(&path).unwrap();
            join_differential(&mut tok2, cut_at_newline(&input, 25_000_000), name);
            ran += 1;
        }
        assert!(ran >= 3, "only {ran} tokenizers loaded — expected at least olmo3/qwen2/deepseek_v3");
    }

    /// Regression test for the vocab-seeded cache (commit d39bca2 originally
    /// seeded EVERY short vocab entry as pretoken -> [own id]): BPE encode
    /// semantics (HF `tokenizers` without `ignore_merges`, this repo's merge
    /// loop, and the pre-cache baseline 0e27c71) produce a whole-vocab-word
    /// token only when the merge rules can derive it. qwen3_5's vocab has
    /// ~200 merge-unreachable entries (mostly multi-char CJK phrases, plus
    /// e.g. " Jap\u{f3}n"); when one appears as a whole pretoken the pipeline
    /// must return the merge decomposition, exactly as if it had missed —
    /// the seed is precomputed misses (`merge_short`), never the raw ID.
    /// Ground truth verified against HF `tokenizers`: encode(" Jap\u{f3}n")
    /// on Qwen/Qwen3.5-9B's tokenizer.json gives [604, 385, 3064] ("ĠJ",
    /// "ap", "Ã³n"); the raw-ID seed returned [209344] (" Jap\u{f3}n").
    #[test]
    fn verify_vocab_seeded_cache_matches_merge_decomposition() {
        let Some(path) = crate::test_hub::hf_tokenizer_json("Qwen/Qwen3.5-9B") else {
            eprintln!("Skipping: Qwen/Qwen3.5-9B tokenizer.json not in the HF cache");
            return;
        };
        let mut tok = load_hf_bpe(&path).expect("load qwen3_5");
        let pretoken: &[u8] = " Jap\u{f3}n".as_bytes();

        let mut cached: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(pretoken, &mut cached);
        // HF `tokenizers` ground truth for this tokenizer.json.
        assert_eq!(
            cached,
            vec![604, 385, 3064],
            "seeded cache returned the merge-unreachable whole-vocab entry"
        );

        // Same rule on the long (> 15 byte) miss path, which used to take
        // a whole-pretoken vocab_inv shortcut: a merge-unreachable long
        // vocab entry must also encode as its decomposition. qwen3_5 id
        // 107517 is a 30-byte CJK phrase HF splits in 3.
        let long_entry = tok
            .vocab_entries()
            .find(|&(id, _)| id == 107517)
            .map(|(_, b)| b.to_vec())
            .expect("qwen3_5 vocab entry 107517");
        assert!(long_entry.len() > 15, "expected a long entry");
        let mut cached_long: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(&long_entry, &mut cached_long);
        assert_ne!(
            cached_long,
            vec![107517],
            "long miss path returned the merge-unreachable whole-vocab entry"
        );
    }

    /// `pack_pretoken_key`'s unaligned-16-byte fast path vs the naive lane
    /// copy at EVERY page offset (both branches of the page-boundary guard),
    /// all lengths 0..=15.
    #[test]
    fn verify_pack_pretoken_key_all_page_offsets() {
        use crate::pretokenize::pack_pretoken_key;
        let mut buf = vec![0u8; 12288];
        let mut rng = XorShift64(0x0123_4567_89AB_CDEF);
        for b in buf.iter_mut() {
            *b = rng.next_u64() as u8;
            if *b == 0 {
                *b = 1; // avoid zero lanes masking length-tag mistakes
            }
        }
        for start in 0..buf.len() - 16 {
            for n in 0..=15usize {
                let span = &buf[start..start + n];
                let key = pack_pretoken_key(span);
                let mut lanes = [0u8; 16];
                lanes[..n].copy_from_slice(span);
                let naive = if n == 0 {
                    0u128
                } else {
                    u128::from_le_bytes(lanes) | ((n as u128) << 120)
                };
                assert_eq!(
                    key,
                    Some(naive),
                    "pack_pretoken_key mismatch at buf offset {start} (page offset {}), len {n}",
                    (buf[start..].as_ptr() as usize) & 4095
                );
            }
        }
        // Length > 15 routes to the long map.
        assert_eq!(pack_pretoken_key(&buf[..16]), None);
    }
}

/// Alignment-invariance sweep: the walkers' output must not depend on the
/// span's heap address. (The full-suite flake of the edge-length test —
/// now `walker_edge::walker_edge_length_pretokens` — motivated this: same bytes,
/// different run -> different tokens can only come from address-dependent
/// framing in the SIMD batch walkers.)
#[cfg(test)]
mod verify_alignment {
    use super::test_util::{XorShift64, gpt2_path, two_phase_lens};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;

    #[test]
    fn verify_walker_alignment_invariance() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        // Inputs chosen to stress batch-edge machinery: long runs of one
        // class, class flips near multiples of 64, multi-byte chars
        // straddling batch edges, contractions, digit groups.
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        for n in [63usize, 64, 65, 127, 128, 129, 255, 256, 300, 4096] {
            for fill in [&b"a"[..], b"5", b" ", b"!", b"\n", "\u{e9}".as_bytes(), "\u{597d}".as_bytes()] {
                let mut v = Vec::new();
                while v.len() < n {
                    v.extend_from_slice(fill);
                }
                inputs.push(v);
            }
        }
        let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
        const PIECES: &[&str] = &[
            " the", " a", "word", "05", "  ", "\n", "'s", "n't", ",", " \u{e9}t\u{e9}",
            "\u{597d}\u{597d}", " 123", "...", "\t", " I'm", "\u{2014}", "e", " ",
        ];
        for _ in 0..200 {
            let target = 80 + (rng.next_u64() % 400) as usize;
            let mut v = Vec::new();
            while v.len() < target {
                v.extend_from_slice(
                    PIECES[(rng.next_u64() % PIECES.len() as u64) as usize].as_bytes(),
                );
            }
            inputs.push(v);
        }

        for (which, input) in inputs.iter().enumerate() {
            // Copy the same bytes at every offset 0..64 of a fresh buffer;
            // walker output must be identical for all of them.
            let mut ref_lens: Option<Vec<usize>> = None;
            let mut ref_ids: Option<Vec<u32>> = None;
            for off in 0..64usize {
                let mut buf = vec![0u8; off + input.len() + 64];
                buf[off..off + input.len()].copy_from_slice(input);
                let span = &buf[off..off + input.len()];
                let lens = two_phase_lens(FastR50kPretokenizer::new(span));
                let mut ids: Vec<u32> = Vec::new();
                tok.memoized_encode_flat(FastR50kPretokenizer::new(span), &mut ids);
                match (&ref_lens, &ref_ids) {
                    (None, _) => {
                        ref_lens = Some(lens);
                        ref_ids = Some(ids);
                    }
                    (Some(rl), Some(ri)) => {
                        assert!(
                            &lens == rl,
                            "input {which}: pretoken lens differ at offset {off}\n  base: {rl:?}\n  off{off}: {lens:?}\n  input: {:?}",
                            String::from_utf8_lossy(input)
                        );
                        assert!(
                            &ids == ri,
                            "input {which}: token ids differ at offset {off} on {:?}",
                            String::from_utf8_lossy(input)
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        eprintln!("alignment invariance: {} inputs x 64 offsets ok", inputs.len());
    }
}

/// Walker edge-condition tests: truncated multi-byte UTF-8 at the buffer
/// end (Bug A: `decode_cp` used to read up to 3 bytes past the slice and
/// return a pretoken end past `len`) and invalid-UTF-8 garbage codepoints
/// (Bug B: 0xF5..=0xFF leads decoded to cp > 0x10FFFF, and `class_of`'s
/// unchecked table load then read heap memory past the class table —
/// nondeterministic under concurrent allocation, causing the Iterator and
/// two-phase paths to split >65 KB invalid pretokens differently).
/// Correctness bar: every scheme partitions ARBITRARY bytes contiguously,
/// in bounds, identically on the Iterator (`next_span`) and two-phase
/// (`fill_spans_keyed`) paths, deterministically.
#[cfg(test)]
mod walker_edge {
    use super::test_util::{XorShift64, gpt2_path, plain_encode_pretoken, two_phase_lens};
    use super::*;
    use crate::load_tokenizer::hf::load_hf_bpe;

    /// Assert a scheme's pretokens are a contiguous, non-empty, in-bounds
    /// partition of `span` (required for encode correctness; also catches
    /// walkers running past the buffer on truncated UTF-8).
    fn check_partition<'a>(
        span: &'a [u8],
        it: impl Iterator<Item = Pretoken<'a>>,
        scheme: &str,
    ) {
        let mut off = 0usize;
        for p in it {
            assert!(
                std::ptr::eq(p.0.as_ptr(), span[off..].as_ptr()),
                "{scheme}: non-contiguous pretoken at byte {off} of {:?}",
                String::from_utf8_lossy(span)
            );
            assert!(!p.0.is_empty(), "{scheme}: empty pretoken at byte {off}");
            off += p.0.len();
            assert!(
                off <= span.len(),
                "{scheme}: pretoken overruns span end ({off} > {}) on {:?}",
                span.len(),
                String::from_utf8_lossy(span)
            );
        }
        assert_eq!(
            off,
            span.len(),
            "{scheme}: pretokens cover {off} of {} bytes of {:?}",
            span.len(),
            String::from_utf8_lossy(span)
        );
    }

    /// Partition check (Iterator path) plus cached encode (two-phase
    /// `fill_spans_keyed` path) vs the plain per-pretoken reference over
    /// the Iterator's pretokens — so the two walker paths are compared
    /// against each other on every span.
    fn check_scheme_encode<'a, P>(
        tok: &mut Tokenizer,
        span: &'a [u8],
        make: impl Fn(&'a [u8]) -> P,
        scheme: &str,
    ) where
        P: PretokenSpans<'a>,
        P: Iterator<Item = Pretoken<'a>>,
    {
        check_partition(span, make(span), scheme);
        let mut got: Vec<u32> = Vec::new();
        tok.memoized_encode_flat(make(span), &mut got);
        let mut expected: Vec<u32> = Vec::new();
        for p in make(span) {
            plain_encode_pretoken(tok, p.0, &mut expected);
        }
        assert!(
            got == expected,
            "{scheme}: cached encode mismatch on {:?} (len {}):\n  cached   {:?}\n  expected {:?}",
            String::from_utf8_lossy(span),
            span.len(),
            got,
            expected
        );
    }

    fn check_all_schemes(tok: &mut Tokenizer, span: &[u8]) {
        check_scheme_encode(tok, span, FastR50kPretokenizer::new, "r50k");
        check_scheme_encode(tok, span, FastCl100kPretokenizer::new, "cl100k");
        check_scheme_encode(tok, span, FastQwen2Pretokenizer::new, "qwen2");
        check_scheme_encode(tok, span, FastQwen35Pretokenizer::new, "qwen3_5");
        check_scheme_encode(tok, span, FastOlmo3Pretokenizer::new, "olmo3");
        check_scheme_encode(tok, span, FastDeepSeekV3Pretokenizer::new, "deepseek_v3");
    }

    /// Truncated multi-byte UTF-8 at the buffer end, every shape: for each
    /// lead-byte length (2/3/4) every truncation point (1..len-1 available
    /// continuation bytes missing), plus lone continuation bytes and
    /// invalid 0xF5..=0xFF leads, behind assorted prefixes that put the
    /// truncated char after a letter run / digit run / space / whitespace
    /// run / punctuation / another unicode char. Exactly-sized heap
    /// allocations so any walker overrun is an observable OOB.
    #[test]
    fn walker_truncated_utf8_tail() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        // Leads: 2-byte (0xC3), 3-byte (0xE2, and 0xE0 low), 4-byte (0xF0,
        // 0xF4 high), invalid leads (0xF5, 0xF8, 0xFF), continuation (0x80,
        // 0xBF), and 0xC0/0xC1 (invalid 2-byte leads).
        let leads: &[&[u8]] = &[
            b"\xc3",
            b"\xe2",
            b"\xe2\x80",
            b"\xe0",
            b"\xe0\xa0",
            b"\xf0",
            b"\xf0\x9f",
            b"\xf0\x9f\x99",
            b"\xf4",
            b"\xf4\x8f",
            b"\xf4\x8f\xbf",
            b"\xf5",
            b"\xf8\x88",
            b"\xff",
            b"\xff\xff",
            b"\xff\xff\xff",
            b"\x80",
            b"\xbf",
            b"\xc0",
            b"\xc1",
        ];
        let prefixes: &[&[u8]] = &[
            b"",
            b"a",
            b"hello",
            b"hello ",
            b"123",
            b" ",
            b"  \n",
            b"!?",
            "é".as_bytes(),
            "好".as_bytes(),
            b"'s",
            b"\xff\xff\xff\xff", // complete invalid run before the tail
        ];
        for &lead in leads {
            for &prefix in prefixes {
                let mut buf = Vec::with_capacity(prefix.len() + lead.len());
                buf.extend_from_slice(prefix);
                buf.extend_from_slice(lead);
                check_all_schemes(&mut tok, &buf);
                // Public path must not panic and must match the plain
                // reference over the Iterator's pretokens.
                let mut cached: Vec<u32> = Vec::new();
                tok.encode_with_added_tokens_flat(&buf, &mut cached);
                let mut expected: Vec<u32> = Vec::new();
                for p in FastR50kPretokenizer::new(&buf) {
                    plain_encode_pretoken(&tok, p.0, &mut expected);
                }
                assert!(
                    cached == expected,
                    "public path mismatch on {:?}: cached {:?} expected {:?}",
                    String::from_utf8_lossy(&buf),
                    cached,
                    expected
                );
            }
        }
    }

    /// Deterministic boundary fuzz: random spans (0-64 bytes; ASCII text,
    /// raw bytes incl. invalid UTF-8 and truncated tails, valid multi-byte
    /// UTF-8, whitespace runs) placed at the END of an exactly-sized
    /// allocation, through every scheme's walker (partition + cached-vs-
    /// plain encode) and the full public GPT-2 path. Fixed seed, no I/O
    /// beyond the tokenizer.
    #[test]
    fn walker_boundary_fuzz_memoized_vs_reference() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let mut rng = XorShift64(0x243F_6A88_85A3_08D3);
        const CHARS: &[&str] = &["é", "ü", "好", "日", "🙂", "ß", "—", "\u{0301}", "٣", "क"];
        let iters = if cfg!(debug_assertions) { 2_000 } else { 12_000 };
        for iter in 0..iters {
            let len = (rng.next_u64() % 65) as usize;
            let pad = (rng.next_u64() % 17) as usize;
            let mut buf: Vec<u8> = Vec::with_capacity(pad + len + 8);
            for _ in 0..pad {
                buf.push(rng.next_u64() as u8);
            }
            let span_start = buf.len();
            match rng.next_u64() % 4 {
                0 => {
                    // ASCII text: letters, digits, spaces, punct, contractions
                    const POOL: &[u8] = b" aetoAETO059'.,!-\n\t\"()s d";
                    while buf.len() - span_start < len {
                        buf.push(POOL[(rng.next_u64() % POOL.len() as u64) as usize]);
                    }
                }
                1 => {
                    // Raw bytes: full 0..=255, mostly invalid UTF-8,
                    // truncated tails included.
                    while buf.len() - span_start < len {
                        buf.push(rng.next_u64() as u8);
                    }
                }
                2 => {
                    // Valid UTF-8 mix: fill to >= len, then trim whole
                    // chars back to <= len so the span stays valid UTF-8.
                    while buf.len() - span_start < len {
                        if rng.next_u64().is_multiple_of(2) {
                            buf.push(b' ' + (rng.next_u64() % 94) as u8);
                        } else {
                            buf.extend_from_slice(
                                CHARS[(rng.next_u64() % CHARS.len() as u64) as usize].as_bytes(),
                            );
                        }
                    }
                    while buf.len() - span_start > len
                        || std::str::from_utf8(&buf[span_start..]).is_err()
                    {
                        buf.pop();
                    }
                }
                _ => {
                    // Whitespace-heavy
                    const POOL: &[u8] = b"   \n\n\t\r a5.";
                    while buf.len() - span_start < len {
                        buf.push(POOL[(rng.next_u64() % POOL.len() as u64) as usize]);
                    }
                }
            }
            buf.truncate(span_start + len.min(buf.len() - span_start));
            let (head, span) = buf.split_at(span_start);
            let _ = head;
            check_all_schemes(&mut tok, span);
            // Full public path (added-token scan + NFC gate + flat emit).
            let mut cached: Vec<u32> = Vec::new();
            let mut expected: Vec<u32> = Vec::new();
            // GPT-2's only added token is <|endoftext|>, absent from these
            // spans (13-byte pattern, span <= 64 random bytes — and the
            // reference below would not model it).
            for p in FastR50kPretokenizer::new(span) {
                plain_encode_pretoken(&tok, p.0, &mut expected);
            }
            tok.encode_with_added_tokens_flat(span, &mut cached);
            assert!(
                cached == expected,
                "public path mismatch at iter {iter} on {:?}: cached {:?} expected {:?}",
                String::from_utf8_lossy(span),
                cached,
                expected
            );
        }
        eprintln!("boundary fuzz: {iters} spans x 6 schemes ok");
    }

    /// Pretokens at the exact edge lengths of the key-packing and cache
    /// machinery (15/16 for the packed u128 key, 65535/65536 and beyond
    /// for long runs through the walkers), in letter/digit/space/punct/
    /// multi-byte/invalid fills, each in an exactly-sized allocation.
    #[test]
    fn walker_edge_length_pretokens() {
        let mut tok = load_hf_bpe(gpt2_path()).expect("load GPT-2 tokenizer");
        let lens: &[usize] = &[
            1, 2, 7, 8, 14, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 255, 256, 4095, 4096,
            4097, 65_535, 65_536, 65_537, 70_003,
        ];
        let fills: &[&[u8]] = &[
            b"a",
            b"5",
            b" ",
            b"!",
            b"\n",
            "\u{e9}".as_bytes(),   // é (2-byte letter)
            "\u{597d}".as_bytes(), // 好 (3-byte letter)
            b"\xff",               // invalid UTF-8
        ];
        for &n in lens {
            for fill in fills {
                // Repeat fill to >= n bytes, then cut to n only for 1-byte
                // fills (multi-byte fills keep whole chars).
                let reps = n / fill.len() + usize::from(n % fill.len() != 0);
                if reps == 0 {
                    continue;
                }
                let exact = if fill.len() == 1 { n } else { reps * fill.len() };
                let mut buf: Vec<u8> = Vec::with_capacity(exact);
                while buf.len() < exact {
                    buf.extend_from_slice(fill);
                }
                buf.truncate(exact);
                check_all_schemes(&mut tok, &buf);
                // Space-prefixed variant hits the space-fused starts.
                let mut buf2: Vec<u8> = Vec::with_capacity(exact + 1);
                buf2.push(b' ');
                buf2.extend_from_slice(&buf);
                check_all_schemes(&mut tok, &buf2);
                // 0xFF run with a letter tail: the exact shape of the
                // >65 KB nondeterminism (the last garbage 4-byte decode
                // straddles into the letters).
                if fill == b"\xff" && exact >= 4 {
                    let mut buf3 = buf.clone();
                    let e = buf3.len();
                    buf3[e - 3..].copy_from_slice(b"cba");
                    check_all_schemes(&mut tok, &buf3);
                }
            }
        }
        eprintln!("edge-length pretokens ok");
    }

    /// Bug B regression: the two spans that flaked (~1/25 full-suite runs)
    /// pre-fix — 65534 x 0xFF + "cba", bare and space-prefixed — walked
    /// repeatedly on both paths while background threads churn the heap.
    /// Pre-fix, `class_of` read past the class table for the garbage
    /// codepoints these 0xFF runs decode to, so the boundary near the run
    /// tail depended on whatever heap memory followed the table; the churn
    /// recreates the "concurrent test threads" condition deterministically
    /// enough that the old code fails this test within a few rounds.
    /// Post-fix every round must produce the identical partition on both
    /// paths.
    #[test]
    fn walker_ff_run_paths_agree_under_heap_churn() {
        // 16 spare bytes of capacity so the run is AddressSanitizer-clean:
        // `pack_pretoken_key`'s page-guarded 16-byte load may overread the
        // final short span within its page (by design — masked out, and
        // never crosses into an unmapped page), which ASAN's redzones
        // would otherwise flag on an exactly-sized allocation.
        // Exactly-sized allocations are covered by the other tests here.
        let mut a = Vec::with_capacity(65537 + 16);
        a.resize(65534, 0xFFu8);
        a.extend_from_slice(b"cba"); // len 65537
        let mut b = Vec::with_capacity(65538 + 16);
        b.push(b' ');
        b.extend_from_slice(&a); // len 65538
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let churners: Vec<_> = (0..4)
            .map(|t| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut keep: Vec<Vec<u8>> = Vec::new();
                    let mut i = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        // Vary size and content so the pages after any
                        // fresh allocation keep changing.
                        let sz = 1 << (12 + (i + t) % 8); // 4 KB .. 512 KB
                        keep.push(vec![(i as u8) ^ 0x5A; sz]);
                        if keep.len() > 8 {
                            keep.clear();
                        }
                        i += 1;
                    }
                })
            })
            .collect();
        let iter_lens = |span: &[u8]| -> Vec<usize> {
            FastR50kPretokenizer::new(span).map(|p| p.0.len()).collect()
        };
        let mut reference: Option<[Vec<usize>; 2]> = None;
        for round in 0..40 {
            let got = [
                {
                    let (i, t) = (iter_lens(&a), two_phase_lens(FastR50kPretokenizer::new(&a)));
                    assert_eq!(i, t, "round {round} span A: iterator vs two-phase");
                    i
                },
                {
                    let (i, t) = (iter_lens(&b), two_phase_lens(FastR50kPretokenizer::new(&b)));
                    assert_eq!(i, t, "round {round} span B: iterator vs two-phase");
                    i
                },
            ];
            match &reference {
                None => reference = Some(got),
                Some(r) => assert_eq!(
                    r,
                    &got,
                    "round {round}: partition changed between rounds (nondeterminism)"
                ),
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for c in churners {
            c.join().unwrap();
        }
    }
}
