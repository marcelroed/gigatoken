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
    /// Maps each byte value to the token ID of its single-byte vocab entry.
    /// The IDs need not be < 256 (e.g. DeepSeek puts its byte tokens at
    /// 3..=258, after the special tokens).
    mapping: Vec<TokenId>,
}

impl ByteRemapping {
    /// Build the byte → token-ID table by scanning `vocab` for single-byte
    /// entries (lowest ID wins). Returns `None` when the mapping is the
    /// identity (token ID == byte value), and an error if some byte value
    /// has no single-byte token.
    pub fn from_byte_vocab(vocab: &[impl AsRef<[u8]>]) -> Result<Option<Self>> {
        const UNSET: u32 = u32::MAX;
        let mut mapping = vec![TokenId::from(UNSET); 256];
        for (id, entry) in vocab.iter().enumerate() {
            if let &[b] = entry.as_ref()
                && mapping[b as usize].0 == UNSET
            {
                mapping[b as usize] = TokenId::from(id as u32);
            }
        }
        if let Some(missing) = mapping.iter().position(|t| t.0 == UNSET) {
            return Err(anyhow!(
                "Byte remapping failed: no single-byte vocab entry for byte {missing:#04x}"
            ));
        }
        Ok(mapping
            .iter()
            .enumerate()
            .any(|(b, t)| t.0 != b as u32)
            .then_some(ByteRemapping { mapping }))
    }
}

// ---------------------------------------------------------------------------
// Shared BPE merge functions
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for [`bpe_merge_symbols_with_scratch`]. Holding one
/// of these across calls means the hot encode loop performs no per-pretoken
/// allocations for the linked list or the merge heap.
#[derive(Default)]
pub struct MergeScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    heap: Vec<std::cmp::Reverse<u64>>,
}

/// Pack a heap entry as `(merged_token << 32) | position`. `u64` ordering then
/// matches the old `(TokenId, usize)` tuple ordering (token ID first, position
/// as tie-break) while halving the element size the heap has to shuffle.
#[inline(always)]
fn pack_merge_entry(merged: TokenId, pos: u32) -> u64 {
    ((merged.0 as u64) << 32) | pos as u64
}

/// Apply BPE merges to an already-initialized symbol sequence.
/// Priority is determined by the merged token's ID (lower = first).
/// This is correct for tiktoken-style tokenizers where vocab ID equals merge rank.
///
/// Uses a min-heap + doubly-linked list for O(n log n) performance.
pub fn bpe_merge_symbols<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
) {
    bpe_merge_symbols_with_scratch(merges, symbols, &mut MergeScratch::default());
}

/// Like [`bpe_merge_symbols`], but reuses caller-provided scratch buffers so
/// repeated calls (one per cache-missing pretoken) do not allocate. Merges
/// `symbols` in place.
pub fn bpe_merge_symbols_with_scratch<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = symbols.len();
    if n < 2 {
        return;
    }

    // For short sequences (the overwhelming majority of pretokens), a linear
    // rank scan has far lower constants than heap + linked list.
    if n <= SMALL_MERGE_MAX {
        bpe_merge_symbols_small(merges, symbols);
        return;
    }

    // Doubly-linked list via u32 index arrays (pretokens are far shorter than
    // 2^32 symbols).
    const NONE: u32 = u32::MAX;
    let next = &mut scratch.next;
    let prev = &mut scratch.prev;
    next.clear();
    next.extend(1..n as u32);
    next.push(NONE);
    prev.clear();
    prev.push(NONE);
    prev.extend(0..n as u32 - 1);

    // Min-heap of packed (merged_token_id, position). Lower ID = higher
    // priority. Seed with all initial pairs, then heapify in O(n) instead of
    // pushing one at a time.
    let mut seeds = std::mem::take(&mut scratch.heap);
    seeds.clear();
    for i in 0..n - 1 {
        if let Some(&merged) = merges.get(&(symbols[i], symbols[i + 1])) {
            seeds.push(Reverse(pack_merge_entry(merged, i as u32)));
        }
    }
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::from(seeds);

    while let Some(Reverse(entry)) = heap.pop() {
        let pos = (entry & u32::MAX as u64) as usize;
        let expected_merged = TokenId::from((entry >> 32) as u32);
        // Validate: pos must still be active and its right neighbor must exist
        let right = next[pos];
        if right == NONE {
            continue;
        }
        let right = right as usize;
        // Check the pair still matches (it may have been invalidated by an earlier merge)
        let pair = (symbols[pos], symbols[right]);
        match merges.get(&pair) {
            Some(&merged) if merged == expected_merged => {
                // Apply the merge
                symbols[pos] = merged;
                let right_right = next[right];
                next[pos] = right_right;
                if right_right != NONE {
                    prev[right_right as usize] = pos as u32;
                }
                // `next[right] = NONE` invalidates any stale heap entries at
                // `right`; nothing reads `prev[right]` once it is unlinked.
                next[right] = NONE;

                // Re-check pair with left neighbor
                let left = prev[pos];
                if left != NONE
                    && let Some(&m) = merges.get(&(symbols[left as usize], symbols[pos])) {
                        heap.push(Reverse(pack_merge_entry(m, left)));
                    }
                // Re-check pair with new right neighbor
                if next[pos] != NONE
                    && let Some(&m) = merges.get(&(symbols[pos], symbols[next[pos] as usize])) {
                        heap.push(Reverse(pack_merge_entry(m, pos as u32)));
                    }
            }
            _ => continue,
        }
    }
    // The pop loop drained the heap; keep its capacity for the next call.
    scratch.heap = heap.into_vec();

    // Compact surviving symbols in place. Surviving indices are strictly
    // increasing and the k-th survivor has index >= k, so writes never
    // overtake reads.
    let mut write = 0;
    let mut i = 0;
    loop {
        symbols[write] = symbols[i];
        write += 1;
        if next[i] == NONE {
            break;
        }
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Sequences up to this length use the linear-scan merge instead of the heap.
const SMALL_MERGE_MAX: usize = 32;

/// BPE merge for short sequences (n <= SMALL_MERGE_MAX), in the style of
/// tiktoken's `byte_pair_merge`: keep a per-position rank (the merged token ID
/// of the pair starting there, or `u32::MAX`), find the minimum by linear scan,
/// merge, and recompute only the two affected neighbor ranks. The scan over a
/// few stack-resident `u32`s beats a `BinaryHeap`'s sift traffic at these
/// sizes, and merge priority (lowest merged ID, then lowest position) is
/// identical to the heap version's ordering.
fn bpe_merge_symbols_small<S: std::hash::BuildHasher>(
    merges: &HashMap<(TokenId, TokenId), TokenId, S>,
    symbols: &mut Vec<TokenId>,
) {
    let get_rank = |a: TokenId, b: TokenId| merges.get(&(a, b)).map_or(u32::MAX, |m| m.0);
    let n = symbols.len();
    debug_assert!((2..=SMALL_MERGE_MAX).contains(&n));
    // Stack-resident doubly-linked list, so a merge is O(1) pointer updates
    // instead of shifting the tail (which lowers to a libc memmove call).
    // Sentinels: next[last] == n, prev[0] == u8::MAX; both fail `< n` checks.
    let mut next = [0u8; SMALL_MERGE_MAX];
    let mut prev = [0u8; SMALL_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // ranks[i] = priority of merging the pair starting at active position i;
    // MAX when there is no merge or the position was merged away.
    let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = get_rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        symbols[i] = TokenId::from(best);
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            ranks[i] = get_rank(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            ranks[left] = get_rank(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place: list indices are strictly increasing, so
    // writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Naive O(n^2) BPE merge for very short sequences where heap overhead isn't worth it.
#[allow(dead_code)]
fn bpe_merge_symbols_naive<S: std::hash::BuildHasher>(
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
    let mut next: Vec<usize> = (1..n).chain(std::iter::once(NONE)).collect();
    let mut prev: Vec<usize> = std::iter::once(NONE).chain(0..n - 1).collect();
    let mut token = symbols.clone();

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
                if left != NONE
                    && let Some(&(_, rank)) = merges.get(&(token[left], token[pos])) {
                        heap.push(Reverse((rank, left)));
                    }
                // Re-check pair with new right neighbor
                if next[pos] != NONE
                    && let Some(&(_, rank)) = merges.get(&(token[pos], token[next[pos]])) {
                        heap.push(Reverse((rank, pos)));
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
