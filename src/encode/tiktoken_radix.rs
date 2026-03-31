use std::hash::{BuildHasher, Hash};

use rustc_hash::FxBuildHasher;
use voracious_radix_sort::{RadixSort, Radixable};

use crate::{bpe::Tokenizer, pretokenize::Pretoken, token::TokenId};

///! Complete radix-sort implementation for tiktoken style tokenizers.
///! Takes a stream of pretokens, fingerprints each of them and sorts by fingerprint, then applies a single merge per pretoken in fingerprint order.

#[derive(Copy, Clone, PartialEq, PartialOrd)]
struct IndexedFingerprint {
    fingerprint: u64,
    index: usize,
}

impl Radixable<u64> for IndexedFingerprint {
    type Key = u64;
    fn key(&self) -> Self::Key {
        self.fingerprint
    }
}

pub fn encode_radix_sort<'a>(
    pretokens: impl Iterator<Item = Pretoken<'a>>,
    tokenizer: &Tokenizer,
) -> Vec<TokenId> {
    let mut pretoken_vec = vec![];
    let hasher = FxBuildHasher::default();
    let mut sort_keys: Vec<IndexedFingerprint> = pretokens
        .enumerate()
        .map(|(i, pretoken)| {
            pretoken_vec.push(pretoken);
            IndexedFingerprint {
                fingerprint: hasher.hash_one(&pretoken),
                index: i,
            }
        })
        .collect();

    sort_keys.voracious_stable_sort();

    // Deduplicate the sorted keys, checking that the pretoken is identical to the representative of the group
    // let

    todo!()
}
