pub mod sentencepiece;
pub mod tiktoken;

use crate::token::TokenId;
use eyre::{Result, anyhow};
use itertools::Itertools;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ByteRemapping — shared between tokenizer types
// ---------------------------------------------------------------------------

pub struct ByteRemapping {
    pub mapping: Vec<u8>, // Maps string byte to symbol byte
    pub unmap: Vec<u8>,   // Maps symbol byte to string byte
}

impl ByteRemapping {
    pub fn from_byte_vocab(vocab: &[impl AsRef<[u8]>]) -> Result<Option<Self>> {
        let byte_remapping = vocab[..256]
            .iter()
            .map(|b| {
                let b = b.as_ref();
                if b.len() != 1 {
                    anyhow!(
                        "Byte remapping failed because vocab entry for byte is not length 1: {:?}",
                        b
                    );
                }
                Ok(b[0])
            })
            .collect::<Result<Vec<u8>>>()?;

        // Only use the byte remapping if it's not the identity mapping
        let byte_remapping = byte_remapping
            .iter()
            .enumerate()
            .any(|(i, &b)| i != b as usize)
            .then_some(byte_remapping)
            .map(|mapping| {
                let mut unmap = vec![0_u8; 256];
                for (i, &b) in mapping.iter().enumerate() {
                    unmap[b as usize] = i as u8;
                }
                ByteRemapping {
                    unmap: mapping,
                    mapping: unmap,
                }
            });
        Ok(byte_remapping)
    }
    pub fn remap_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        bytes.iter().map(|&b| self.mapping[b as usize]).collect()
    }
    pub fn unmap_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        bytes.iter().map(|&b| self.unmap[b as usize]).collect()
    }
}

// ---------------------------------------------------------------------------
// Shared BPE merge functions
// ---------------------------------------------------------------------------

/// Apply BPE merges to an already-initialized symbol sequence.
/// Priority is determined by the merged token's ID (lower = first).
/// This is correct for tiktoken-style tokenizers where vocab ID equals merge rank.
pub fn bpe_merge_symbols<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
) {
    loop {
        let candidate_merges = symbols
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(a, b)).map(|&v| (i, v)));

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token);

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1);
        } else {
            break;
        }
    }
}

/// Apply BPE merges using explicit merge ranks for priority (lower rank = first).
/// The merge table maps `(token_a, token_b) → (merged_token, rank)`.
///
/// Uses a min-heap + doubly-linked list for O(n log n) performance instead of
/// the naive O(n × merges) scan.
/// This is only needed by SentencePiece-style tokenizers.
pub fn bpe_merge_symbols_ranked<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), (TokenId, u32), S>,
    symbols: &mut Vec<TokenId>,
) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = symbols.len();
    if n < 2 {
        return;
    }

    // Doubly-linked list via index arrays. NONE = no neighbor.
    const NONE: usize = usize::MAX;
    let mut next = vec![NONE; n];
    let mut prev = vec![NONE; n];
    let mut token = symbols.clone();

    for i in 0..n - 1 {
        next[i] = i + 1;
    }
    for i in 1..n {
        prev[i] = i - 1;
    }

    // Min-heap of (rank, position). Position refers to the left symbol of the pair.
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    // Seed with all initial pairs
    let mut i = 0;
    while i < n {
        let j = next[i];
        if j == NONE {
            break;
        }
        if let Some(&(_, rank)) = merges.get(&(token[i], token[j])) {
            heap.push(Reverse((rank, i)));
        }
        i = j;
    }

    while let Some(Reverse((rank, pos))) = heap.pop() {
        // Validate: pos must still be active and its right neighbor must exist
        let right = next[pos];
        if right == NONE {
            continue;
        }
        // Check the pair still matches (it may have been invalidated by an earlier merge)
        let pair = (token[pos], token[right]);
        match merges.get(&pair) {
            Some(&(merged_token, r)) if r == rank => {
                // Apply the merge: replace token[pos], remove right
                token[pos] = merged_token;
                let right_right = next[right];
                next[pos] = right_right;
                if right_right != NONE {
                    prev[right_right] = pos;
                }
                // Mark right as deleted
                next[right] = NONE;
                prev[right] = NONE;

                // Re-check pair with left neighbor
                let left = prev[pos];
                if left != NONE {
                    if let Some(&(_, rank)) = merges.get(&(token[left], token[pos])) {
                        heap.push(Reverse((rank, left)));
                    }
                }
                // Re-check pair with new right neighbor
                if next[pos] != NONE {
                    if let Some(&(_, rank)) = merges.get(&(token[pos], token[next[pos]])) {
                        heap.push(Reverse((rank, pos)));
                    }
                }
            }
            _ => continue, // Stale entry, skip
        }
    }

    // Collect surviving symbols via linked list traversal
    symbols.clear();
    let mut i = 0;
    // Find the head (first element with prev == NONE that's still in the list)
    // Since we never remove index 0's prev link, index 0 is always the head
    loop {
        symbols.push(token[i]);
        if next[i] == NONE {
            break;
        }
        i = next[i];
    }
}

/// Tokenize a single pretoken by mapping each byte to TokenId(byte_value)
/// then applying BPE merges (priority by merged token ID).
pub fn simple_bpe_merge<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    pre_token: &[u8],
) -> Vec<TokenId> {
    let mut symbols: Vec<TokenId> = pre_token.iter().map(|&b| TokenId::from(b as u32)).collect();
    bpe_merge_symbols(merges, &mut symbols);
    symbols
}

// Re-export the main types so existing `use crate::bpe::Tokenizer` still works.
pub use sentencepiece::SentencePieceBPE;
pub use tiktoken::Tokenizer;
