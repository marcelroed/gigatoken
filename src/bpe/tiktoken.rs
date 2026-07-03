use bumpalo::collections::CollectIn;
use bumpalo::collections::Vec as BumpVec;
use itertools::Itertools;

use crate::bpe::{ByteRemapping, MergeScratch, bpe_merge_symbols_with_scratch, simple_bpe_merge};
use crate::pretokenize::{Pretoken, PretokenizerType};
use crate::token::TokenId;
use eyre::Result;
use std::borrow::Cow;
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
    /// Sorted by content length descending so the first match at a position
    /// is the longest (leftmost-longest semantics).
    added_tokens: Vec<(Arc<[u8]>, TokenId)>,
    /// Distinct first bytes of `added_tokens`, for the candidate scan.
    added_first_bytes: Vec<u8>,
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
#[inline(always)]
fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    let p = bytes.as_ptr();
    // Keep the low `n` bytes, zero the rest. `n <= 15`, so the shift is < 128
    // and lane 15 stays zero, ready for the length tag.
    let mask = (1u128 << (n * 8)) - 1;
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
    Some(low | ((n as u128) << 120))
}

/// Tokenize a single pretoken by repeatedly applying BPE merges in order.
pub fn simple_bpe_merge_in_arena<'a, S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    pre_token: &[u8],
    merge_arena: &'a bumpalo::Bump,
) -> BumpVec<'a, TokenId> {
    let mut symbols: BumpVec<TokenId> = pre_token
        .iter()
        .map(|&b| TokenId::from(b as u32))
        .collect_in(merge_arena);

    loop {
        let candidate_merges = symbols
            .iter()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(*a, *b)).map(|v| (i, *v)));

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token);

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1);
        } else {
            break;
        }
    }
    symbols.shrink_to_fit();
    symbols
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
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_first_bytes: Vec::new(),
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
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: PretokenizerType::GPT2,
            added_tokens: Vec::new(),
            added_first_bytes: Vec::new(),
        })
    }

    /// Create a new tokenizer sharing the same model data but with an empty cache.
    /// Useful for per-thread encoding in parallel.
    pub fn fork(&self) -> Self {
        Tokenizer {
            merges: self.merges.clone(),
            vocab: self.vocab.clone(),
            vocab_inv: self.vocab_inv.clone(),
            byte_remapping: self.byte_remapping.as_ref().map(|br| ByteRemapping {
                mapping: br.mapping.clone(),
                unmap: br.unmap.clone(),
            }),
            token_arena: Vec::new(),
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            pretoken_cache_long: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            merge_scratch: MergeScratch::default(),
            symbol_scratch: Vec::new(),
            pretokenizer_type: self.pretokenizer_type,
            added_tokens: self.added_tokens.clone(),
            added_first_bytes: self.added_first_bytes.clone(),
        }
    }

    pub fn set_pretokenizer_type(&mut self, pretokenizer_type: PretokenizerType) {
        self.pretokenizer_type = pretokenizer_type;
    }

    pub fn pretokenizer_type(&self) -> PretokenizerType {
        self.pretokenizer_type
    }

    /// Set the added tokens matched atomically by
    /// [`Self::encode_with_added_tokens`]. Empty contents are ignored.
    pub fn set_added_tokens(&mut self, added_tokens: Vec<(Vec<u8>, TokenId)>) {
        let mut added_tokens: Vec<(Arc<[u8]>, TokenId)> = added_tokens
            .into_iter()
            .filter(|(content, _)| !content.is_empty())
            .map(|(content, id)| (content.into(), id))
            .collect();
        added_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut first_bytes: Vec<u8> = added_tokens.iter().map(|(c, _)| c[0]).collect();
        first_bytes.sort_unstable();
        first_bytes.dedup();
        self.added_tokens = added_tokens;
        self.added_first_bytes = first_bytes;
    }

    /// Find the leftmost added-token occurrence at or after `from`, taking
    /// the longest token when several match at the same position. Returns
    /// `(start, end, id)`.
    fn find_added_token(&self, bytes: &[u8], from: usize) -> Option<(usize, usize, TokenId)> {
        if self.added_tokens.is_empty() {
            return None;
        }
        let hay = &bytes[from..];
        let candidates: Box<dyn Iterator<Item = usize> + '_> =
            match self.added_first_bytes.as_slice() {
                &[a] => Box::new(memchr::memchr_iter(a, hay)),
                &[a, b] => Box::new(memchr::memchr2_iter(a, b, hay)),
                &[a, b, c] => Box::new(memchr::memchr3_iter(a, b, c, hay)),
                firsts => Box::new(
                    hay.iter()
                        .enumerate()
                        .filter(move |(_, b)| firsts.contains(b))
                        .map(|(i, _)| i),
                ),
            };
        for cand in candidates {
            for (content, id) in &self.added_tokens {
                if hay[cand..].starts_with(content) {
                    return Some((from + cand, from + cand + content.len(), *id));
                }
            }
        }
        None
    }

    /// Encode raw text: split out added-token occurrences (emitted as their
    /// single token ID), pretokenize the segments between them with this
    /// tokenizer's pretokenization scheme, and BPE-encode each pretoken.
    /// This mirrors the full HuggingFace `tokenizers` encode pipeline.
    pub fn encode_with_added_tokens(&mut self, bytes: &[u8], mut f: impl FnMut(&[TokenId])) {
        let pretokenizer_type = self.pretokenizer_type;
        let mut pos = 0;
        while pos < bytes.len() {
            match self.find_added_token(bytes, pos) {
                Some((start, end, id)) => {
                    self.memoized_encode(pretokenizer_type.pretokenize(&bytes[pos..start]), &mut f);
                    f(&[id]);
                    pos = end;
                }
                None => {
                    self.memoized_encode(pretokenizer_type.pretokenize(&bytes[pos..]), &mut f);
                    break;
                }
            }
        }
    }

    pub fn encode_pretoken(
        byte_remapping: Option<&ByteRemapping>,
        merges: &HashMap<(TokenId, TokenId), TokenId, rustc_hash::FxBuildHasher>,
        pretoken: Pretoken,
    ) -> Vec<TokenId> {
        let pretoken: Cow<[u8]> = if let Some(byte_remapping) = byte_remapping {
            pretoken
                .iter()
                .map(|&b| byte_remapping.mapping[b as usize])
                .collect::<Cow<[u8]>>()
        } else {
            Cow::Borrowed(pretoken.0)
        };
        simple_bpe_merge(merges, &pretoken)
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
            let cached = match key {
                Some(key) => self.pretoken_cache.get(&key).copied(),
                None => self.pretoken_cache_long.get(bytes).copied(),
            };
            if let Some((offset, len)) = cached {
                let start = offset as usize;
                f(&self.token_arena[start..start + len as usize]);
            } else {
                // Encode into reusable scratch and append straight to the
                // arena: no intermediate `Vec`, no `Cow` remap allocation.
                let symbols = &mut self.symbol_scratch;
                symbols.clear();
                match self.byte_remapping.as_ref() {
                    Some(br) => symbols.extend(
                        bytes
                            .iter()
                            .map(|&b| TokenId::from(br.mapping[b as usize] as u32)),
                    ),
                    None => symbols.extend(bytes.iter().map(|&b| TokenId::from(b as u32))),
                }
                bpe_merge_symbols_with_scratch(&self.merges, symbols, &mut self.merge_scratch);
                let offset = self.token_arena.len() as u32;
                let len = symbols.len() as u32;
                self.token_arena.extend_from_slice(symbols);
                match key {
                    Some(key) => {
                        self.pretoken_cache.insert(key, (offset, len));
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

    /// Get the number of pretokens currently in the cache.
    pub fn pretoken_cache_size(&self) -> usize {
        self.pretoken_cache.len() + self.pretoken_cache_long.len()
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
                let token_bytes = BASE64_STANDARD.decode(base64_token).unwrap();
                token_bytes
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
