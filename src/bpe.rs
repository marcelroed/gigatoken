use bumpalo::collections::CollectIn;
use itertools::Itertools;

use crate::pretokenize::Pretoken;
use crate::token::TokenId;
use eyre::{Context, Result, anyhow};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::sync::Arc;
// pub fn encode_par(pretokenizeable: PretokenizeableSpec) {
//     match pretokenizeable {
//         PretokenizeableSpec::Bytes(b) => {
//             encode_par_bytes(b)
//         }
//         _ => {
//             unimplemented!("Currently only PretokenizeableSpec::Bytes is implemented");
//         }
//     }
// }

pub struct ByteRemapping {
    mapping: Vec<u8>, // Maps string byte to symbol byte
    unmap: Vec<u8>,   // Maps symbol byte to string byte
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

pub struct Tokenizer {
    merges: HashMap<(TokenId, TokenId), TokenId>, // Maps pairs of token ids to merged token id
    vocab: Vec<Arc<[u8]>>,                        // Maps token ids to byte sequences
    vocab_inv: HashMap<Arc<[u8]>, TokenId>,       // Maps byte sequences to token ids
    byte_remapping: Option<ByteRemapping>, // Remaps bytes, because some tokenizers do this for some reason
    pretoken_cache: HashMap<Arc<[u8]>, Arc<[TokenId]>, rustc_hash::FxBuildHasher>,
    // merge_arena: bumpalo::Bump, // Arena to use for BPE merging (avoid tons of alloc/dealloc)
}

use bumpalo::collections::Vec as BumpVec;

/// Tokenize a single pretoken by repeatedly applying BPE merges in order.
pub fn simple_bpe_merge<'a>(
    merges: &HashMap<(TokenId, TokenId), TokenId>,
    pre_token: &[u8],
) -> Vec<TokenId> {
    let mut symbols: Vec<TokenId> = pre_token.iter().map(|&b| TokenId::from(b as u32)).collect();

    loop {
        let candidate_merges = symbols
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .filter_map(|(i, (a, b))| merges.get(&(a, b)).map(|&v| (i, v)));

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1); // O(n) worst case
        } else {
            break;
        }
    }
    symbols
}

/// Tokenize a single pretoken by repeatedly applying BPE merges in order.
pub fn simple_bpe_merge_in_arena<'a>(
    merges: &HashMap<(TokenId, TokenId), TokenId>,
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

        let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

        if let Some((merge_index, merge_token)) = best_merge {
            symbols[merge_index] = merge_token;
            symbols.remove(merge_index + 1); // O(n) worst case
        } else {
            break;
        }
    }
    symbols.shrink_to_fit();
    symbols
}

impl Tokenizer {
    pub fn new(
        merges: HashMap<(TokenId, TokenId), TokenId>,
        vocab: Vec<Vec<u8>>,
        byte_remapping: Option<ByteRemapping>,
    ) -> Self {
        let vocab = vocab.into_iter().map(Into::into).collect::<Vec<Arc<_>>>();
        let vocab_inv = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect();
        Tokenizer {
            merges,
            vocab_inv,
            vocab,
            byte_remapping,
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            // merge_arena: bumpalo::Bump::new(),
        }
    }

    /// Given a list of tokens in rank order (by merge order), reconstructs the merges map and returns a Tokenizer.
    /// This process is necessary to load some tokenizers found in tiktoken.
    /// There are a few exceptions to this being correct, so make sure that this is only used on tokenizers that don't package merges.
    pub fn from_ranks(vocab: Vec<Vec<u8>>) -> Result<Self> {
        let mut merges = HashMap::new();
        let vocab = vocab
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Arc<[u8]>>>();
        let vocab_inv = vocab
            .iter()
            .cloned()
            .zip((0..).map(TokenId::from))
            .collect::<HashMap<_, TokenId>>();

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
            pretoken_cache: HashMap::with_hasher(rustc_hash::FxBuildHasher {}),
            // merge_arena: bumpalo::Bump::new(),
        })
    }

    pub fn encode_pretoken(
        byte_remapping: Option<&ByteRemapping>,
        merges: &HashMap<(TokenId, TokenId), TokenId>,
        pretoken: Pretoken,
    ) -> Vec<TokenId> {
        // TODO improve
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

    /// For each pretoken in the input iterator, looks up the string in the cache, and if not found, encodes it and inserts it into the cache.
    pub fn memoized_encode<'a, 'i>(
        &'a mut self,
        pretoken_iter: impl Iterator<Item = Pretoken<'i>>,
    ) -> impl Iterator<Item = Arc<[TokenId]>> {
        let pretoken_cache = &mut self.pretoken_cache;
        pretoken_iter.map(|pretoken: Pretoken| {
            let found_value = pretoken_cache.get(pretoken.as_ref());
            if let Some(v) = found_value {
                return v.clone();
            }
            let inserted_value: Arc<[TokenId]> =
                Self::encode_pretoken(self.byte_remapping.as_ref(), &self.merges, pretoken).into();
            pretoken_cache.insert(pretoken.as_ref().into(), inserted_value.clone());
            inserted_value
        })
    }

    pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8> {
        v.iter()
            .flat_map(|&token| {
                self.vocab[token.0 as usize].as_ref()
                // if let Some(byte_remapping) = &self.byte_remapping {
                //     return byte_remapping.unmap_bytes(&self.vocab[token as usize]);
                // }
                // return base_symbols;
            })
            .copied()
    }

    /// Get the number of pretokens currently in the cache
    pub fn pretoken_cache_size(&self) -> usize {
        self.pretoken_cache.len()
    }
}

// impl Tokenizer {
//     pub fn new(merges: HashMap<(u32, u32), u32>) -> Self {
//         Tokenizer { merges }
//     }

//     pub fn encode_pretoken(&self, pretoken: &[u8]) -> Vec<u32> {
//         let mut symbols: Vec<u32> = pretoken.iter().map(|&b| b as u32).collect();

//         loop {
//             let candidate_merges = symbols
//                 .iter()
//                 .tuple_windows()
//                 .enumerate()
//                 .filter_map(|(i, (a, b))| self.merges.get(&(*a, *b)).map(|v| (i, *v)));
//             let best_merge = candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

//             if let Some((merge_index, merge_token)) = best_merge {
//                 symbols[merge_index] = merge_token;
//                 symbols.remove(merge_index + 1); // O(n) worst case
//             } else {
//                 break;
//             }
//         }
//         symbols
//     }
// }

// pub fn encode_par_bytes(bytes: &[u8], tokenizer: ) {
//     let boundaries = find_boundaries(bytes);
//     let pretoken_mapping = DashMap::new();
//     boundaries
//         .iter()
//         .copied()
//         .tuple_windows()
//         .for_each(|(start, end)| {
//             let slice = &bytes[start..end];
//             pretokenize_as_iter(slice).for_each(|token| {
//                 pretoken_mapping.entry(token).or_insert_with(|| Rc::new(token));
//             })
//         });
// }

// pub fn encode(
//     re: &Regex,                        // Regex to pre-tokenize
//     vocab_inv_bytes: &[Option<u16>],   // Mapping from byte to token for initial vocab
//     merges: &HashMap<(u16, u16), u16>, // Tuple of tokens to merged token
//     text: &str,
// ) -> Vec<u16> {
//     let words: Vec<&str> = re.find_iter(text).map(|m| &text[m.0..m.1]).collect();
//     words
//         .into_iter()
//         .flat_map(|word| {
//             let word_bytes = word.as_bytes();
//             let mut symbols: Vec<u16> = word_bytes
//                 .iter()
//                 .map(|c| vocab_inv_bytes[*c as usize].unwrap())
//                 .collect();

//             loop {
//                 let candidate_merges = symbols
//                     .iter()
//                     .tuple_windows()
//                     .enumerate()
//                     .filter_map(|(i, (a, b))| merges.get(&(*a, *b)).map(|v| (i, *v)));
//                 let best_merge =
//                     candidate_merges.min_by_key(|(_index, merged_token)| *merged_token); // Earliest merge in list of merges

//                 if let Some((merge_index, merge_token)) = best_merge {
//                     symbols[merge_index] = merge_token;
//                     symbols.remove(merge_index + 1); // O(n) worst case
//                 } else {
//                     break;
//                 }
//             }
//             // println!("Merged {:?} into {:?}", word, word_bytes, symbols)

//             symbols
//         })
//         .collect()
// }

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
        std::fs::File::open("/Users/marcel/data/tokenizers/r50k_base.tiktoken")
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
        // for ((a, b), c) in merges.iter().take(20) {
        //     eprintln!("('{a}', '{b}') -> '{c}'");
        // }
    }

    #[test]
    fn basic_tokenization() {
        let text = "This is a test string. Please tokenize it!";
        let mut tokenizer = load_tiktoken("/Users/marcel/data/tokenizers/r50k_base.tiktoken")
            .expect("Failed to load tokenizer");
        let pretokenize_iter = crate::pretokenize::pretokenize_as_iter(text.as_bytes());
        let mut output = vec![];
        tokenizer
            .memoized_encode(pretokenize_iter)
            .for_each(|pretoken| output.extend_from_slice(&pretoken));
        assert!(tokenizer.byte_remapping.is_some());
        println!("Encoded: {:?}", output);
        let decoded = tokenizer.decode(&output).collect::<Vec<u8>>();
        println!("Decoded: {:?}", String::from_utf8_lossy(&decoded));
    }

    // #[test]
    // fn test_encode() {
    //     let re =
    //         Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    //             .unwrap();
    //     let vocab = HashMap::from([
    //         (0_u16, &b" "[..]),
    //         (1, b"a"),
    //         (2, b"c"),
    //         (3, b"e"),
    //         (4, b"h"),
    //         (5, b"t"),
    //         (6, b"th"),
    //         (7, b" c"),
    //         (8, b" a"),
    //         (9, b"the"),
    //         (10, b" at"),
    //     ]);
    //     let mut vocab_inv_bytes = vec![None; 256];
    //     vocab.iter().for_each(|(&k, &v)| {
    //         if v.len() == 1 {
    //             vocab_inv_bytes[v[0] as usize] = Some(k);
    //         }
    //     });

    //     let merges: HashMap<(u16, u16), u16> = vec![
    //         (&b"t"[..], &b"h"[..]),
    //         (b" ", b"c"),
    //         (b" ", b"a"),
    //         (b"th", b"e"),
    //         (b" a", b"t"),
    //     ]
    //     .into_iter()
    //     .map(|(e1, e2)| {
    //         let mut merged = Vec::from(e1);
    //         merged.append(&mut Vec::from(e2));
    //         let e1_token = vocab.iter().find(|&(_, &v)| v == e1).unwrap().0;
    //         let e2_token = vocab.iter().find(|&(_, &v)| v == e2).unwrap().0;
    //         let merged_token = vocab.iter().find(|&(_, &v)| v == merged).unwrap().0;
    //         ((*e1_token, *e2_token), *merged_token)
    //     })
    //     .collect();
    //     println!("{merges:?}");

    //     //let special_tokens = HashMap::new();

    //     let text = "the cat ate";
    //     let encoded = encode(&re, &vocab_inv_bytes, &merges, text);
    //     assert_eq!(encoded, vec![9, 7, 1, 5, 10, 3]);
    // }
}
