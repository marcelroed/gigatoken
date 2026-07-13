use crate::bpe::{ByteRemapping, MergeScratch, bpe_merge_symbols_with_scratch, simple_bpe_merge};
use crate::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer, Pretoken,
    PretokenizerType,
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
    pub(crate) merges: HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId, rustc_hash::FxBuildHasher>,
    pub(crate) byte_remapping: Option<ByteRemapping>,
    /// Append-only arena of encoded token IDs. Cache entries store
    /// `(offset, len)` slices into this vector, avoiding per-entry
    /// `Arc` allocations and atomic refcount bumps on every cache hit.
    token_arena: Vec<TokenId>,
    /// Pretoken cache for the common case (≤ 15 bytes, ~99.9% of pretokens).
    /// The key packs the bytes into the low 15 bytes and the length into the
    /// top byte of a `u128`, so lookups are a single inlined 128-bit compare
    /// instead of a `memcmp` call, and hashing is two multiply-mixes instead of
    /// a byte loop. Keys are `Copy`, so cache misses no longer allocate.
    pretoken_cache: HashMap<u128, (u32, u32), rustc_hash::FxBuildHasher>,
    /// Direct-mapped front cache for packed pretokens. Natural-language
    /// pretokens are heavily Zipf-distributed, so most hits avoid probing the
    /// much larger SwissTable. Each entry keeps its key and arena range
    /// together so a hit usually requires one cache-line fetch.
    pretoken_front: Vec<PretokenFrontEntry>,
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

/// A 24-byte front-cache entry. Splitting the `u128` key into two words avoids
/// the 16-byte alignment padding that would otherwise make this 32 bytes.
#[derive(Clone, Copy, Default)]
struct PretokenFrontEntry {
    key_lo: u64,
    key_hi: u64,
    value: u64,
}

impl PretokenFrontEntry {
    #[inline(always)]
    fn new(key: u128, offset: u32, len: u32) -> Self {
        Self {
            key_lo: key as u64,
            key_hi: (key >> 64) as u64,
            value: offset as u64 | ((len as u64) << 32),
        }
    }

    #[inline(always)]
    fn get(self, key: u128) -> Option<(u32, u32)> {
        (self.key_lo == key as u64 && self.key_hi == (key >> 64) as u64)
            .then_some((self.value as u32, (self.value >> 32) as u32))
    }
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

/// Pack a pretoken of ≤ 15 bytes into a `u128`: bytes in the low 15 lanes,
/// length in the top byte. Returns `None` for longer pretokens, which use the
/// `Box<[u8]>` fallback map. Encoding the length means two pretokens of
/// different length can never collide.
///
/// The common path is a single unaligned 16-byte load followed by a mask, which
/// avoids both the variable-length `copy_from_slice` (a `memcpy` libc call) and
/// any per-byte branching. The load is only taken when it cannot cross a page
/// boundary, so it can never touch an unmapped page; the rare near-boundary case
/// falls back to a plain copy. Both paths produce the identical key.
/// Per-length `(mask, length-tag)` pairs for [`pack_pretoken_key`]: one 512-byte
/// L1-resident table lookup replaces a 128-bit variable shift + sub + tag shift
/// (u128 shifts lower to a multi-instruction sequence on aarch64).
const PACK_MASK_TAG: [(u128, u128); 16] = {
    let mut t = [(0u128, 0u128); 16];
    let mut n = 0;
    while n < 16 {
        let mask = if n == 0 { 0 } else { u128::MAX >> (8 * (16 - n)) };
        t[n] = (mask, (n as u128) << 120);
        n += 1;
    }
    t
};

/// Log2 of the direct-mapped packed-pretoken front cache size. 2^18 entries use
/// 6 MiB per tokenizer and retain the broad working set while the authoritative
/// HashMap keeps every key.
const PRETOKEN_FRONT_BITS: u32 = 18;

#[inline(always)]
fn pretoken_cache_hash(key: u128) -> u64 {
    let folded = (key as u64) ^ ((key >> 64) as u64);
    folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[inline(always)]
fn pretoken_front_index(hash: u64) -> usize {
    (hash >> (64 - PRETOKEN_FRONT_BITS)) as usize
}

#[inline(always)]
pub(crate) fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    if n == 0 {
        return Some(0);
    }
    let p = bytes.as_ptr();
    // Keep the low `n` bytes, zero the rest; lane 15 stays zero, ready for the
    // length tag.
    let (mask, tag) = PACK_MASK_TAG[n];
    let low = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: the offset within the (≥ 4096-byte) page is ≤ 4096 - 16, so a
        // 16-byte read stays inside the page holding `p`, which is mapped
        // because `p` points to at least one valid byte.
        let v = unsafe { (p as *const u128).read_unaligned() };
        v & mask
    } else {
        // Rare: `p` is within 16 bytes of a page boundary. Gather with a plain
        // copy (≤ 15 bytes) — correctness over speed on this cold path.
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        u128::from_le_bytes(lanes) & mask
    };
    Some(low | tag)
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
        Tokenizer {
            merges,
            vocab_inv,
            vocab,
            byte_remapping,
            token_arena: Vec::new(),
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            pretoken_front: vec![PretokenFrontEntry::default(); 1 << PRETOKEN_FRONT_BITS],
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
        }
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

        Ok(Tokenizer {
            merges,
            byte_remapping: ByteRemapping::from_byte_vocab(&vocab)?,
            vocab,
            vocab_inv,
            token_arena: Vec::new(),
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            pretoken_front: vec![PretokenFrontEntry::default(); 1 << PRETOKEN_FRONT_BITS],
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_matcher: None,
            normalize_nfc: false,
        })
    }

    /// Create a new tokenizer sharing the same model data but with an empty cache.
    /// Useful for per-thread encoding in parallel.
    pub fn fork(&self) -> Self {
        Tokenizer {
            merges: self.merges.clone(),
            vocab: self.vocab.clone(),
            vocab_inv: self.vocab_inv.clone(),
            byte_remapping: self.byte_remapping.clone(),
            token_arena: Vec::new(),
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            pretoken_front: vec![PretokenFrontEntry::default(); 1 << PRETOKEN_FRONT_BITS],
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
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
        if idx >= self.vocab.len() {
            self.vocab.resize(idx + 1, Arc::from(Vec::new().as_slice()));
        }
        if self.vocab[idx].is_empty() {
            self.vocab[idx] = content.clone().into();
            self.vocab_inv.insert(self.vocab[idx].clone(), id);
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

    /// For each pretoken in the input iterator, looks up the string in the
    /// cache, and if not found, encodes it and inserts it into the cache.
    /// Calls `f` with the encoded token slice for each pretoken.
    pub fn memoized_encode<'i>(
        &mut self,
        pretoken_iter: impl Iterator<Item = Pretoken<'i>>,
        mut f: impl FnMut(&[TokenId]),
    ) {
        for pretoken in pretoken_iter {
            let bytes = pretoken.as_ref();
            // Look up the cached encoding. Short pretokens (the overwhelming
            // majority) use the packed `u128` map; the rare long ones fall back
            // to the slice-keyed map. The key is computed once and reused on the
            // miss path's insert.
            let key = pack_pretoken_key(bytes);
            // Packed key zero represents an empty pretoken and is reserved as
            // the front cache's empty sentinel. Real pretokenizers never yield
            // empty slices, but keep the public iterator API correct for one.
            let front_key = key.filter(|&key| key != 0);
            let mut front_idx = 0;
            if let Some(key) = front_key {
                let hash = pretoken_cache_hash(key);
                front_idx = pretoken_front_index(hash);
                // SAFETY: pretoken_front_index returns PRETOKEN_FRONT_BITS bits
                // and the array has exactly 2^PRETOKEN_FRONT_BITS entries.
                if let Some((offset, len)) =
                    unsafe { *self.pretoken_front.get_unchecked(front_idx) }.get(key)
                {
                    let start = offset as usize;
                    // SAFETY: front entries are copied from authoritative cache
                    // entries, whose arena ranges remain valid forever.
                    f(unsafe { self.token_arena.get_unchecked(start..start + len as usize) });
                    continue;
                }
            }
            let cached = match key {
                Some(key) => self.pretoken_cache.get(&key).copied(),
                None => self.pretoken_cache_long.get(bytes).copied(),
            };
            if let Some((offset, len)) = cached {
                if let Some(key) = front_key {
                    self.pretoken_front[front_idx] = PretokenFrontEntry::new(key, offset, len);
                }
                let start = offset as usize;
                // SAFETY: every cached (offset, len) was recorded right after
                // appending those `len` tokens at `offset`, and `token_arena`
                // never shrinks, so the range is always in bounds.
                f(unsafe { self.token_arena.get_unchecked(start..start + len as usize) });
            } else {
                // Encode into reusable scratch and append straight to the
                // arena: no intermediate `Vec`, no `Cow` remap allocation.
                let symbols = &mut self.symbol_scratch;
                symbols.clear();
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(bytes.iter().map(|&b| br.mapping[b as usize])),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                bpe_merge_symbols_with_scratch(&self.merges, symbols, &mut self.merge_scratch);
                let offset = self.token_arena.len() as u32;
                let len = symbols.len() as u32;
                self.token_arena.extend_from_slice(symbols);
                match key {
                    Some(key) => {
                        self.pretoken_cache.insert(key, (offset, len));
                        if let Some(key) = front_key {
                            self.pretoken_front[front_idx] =
                                PretokenFrontEntry::new(key, offset, len);
                        }
                    }
                    None => {
                        self.pretoken_cache_long.insert(bytes.into(), (offset, len));
                    }
                }
                f(&self.token_arena[offset as usize..offset as usize + len as usize]);
            }
        }
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| self.vocab[token.0 as usize].as_ref())
            .copied()
    }

    /// Detailed cache stats for memory accounting (see examples/cache_memory.rs):
    /// (short_len, short_cap, long_len, long_cap, long_key_bytes, arena_len, arena_cap).
    /// The fixed-size 6 MiB front cache is not represented in the tuple.
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
            .field("byte_remapping", &self.byte_remapping.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_tokenizer::tiktoken::load_tiktoken;
    use std::io::Read;

    #[test]
    fn packed_front_cache_serves_repeated_pretokens() {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        let mut tokenizer = Tokenizer::new(merges, vocab, None);
        let bytes = b"hello";

        let mut first = Vec::new();
        tokenizer.memoized_encode([Pretoken(bytes)].into_iter(), |tokens| {
            first.extend(tokens.iter().map(|token| token.0));
        });
        let expected: Vec<u32> = bytes.iter().map(|&byte| byte as u32).collect();
        assert_eq!(first, expected);

        let key = pack_pretoken_key(bytes).unwrap();
        let front_idx = pretoken_front_index(pretoken_cache_hash(key));
        assert!(tokenizer.pretoken_front[front_idx].get(key).is_some());
        assert!(tokenizer.pretoken_cache.remove(&key).is_some());

        let mut repeated = Vec::new();
        tokenizer.memoized_encode([Pretoken(bytes)].into_iter(), |tokens| {
            repeated.extend(tokens.iter().map(|token| token.0));
        });
        assert_eq!(repeated, first);
        assert!(tokenizer.pretoken_cache.is_empty());

        // The zero key is the front-cache sentinel, so an empty pretoken must
        // still take the authoritative map path.
        tokenizer.memoized_encode([Pretoken(b"")].into_iter(), |tokens| {
            assert!(tokens.is_empty());
        });
        assert!(tokenizer.pretoken_cache.contains_key(&0));
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
