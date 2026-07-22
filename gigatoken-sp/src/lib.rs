//! gigatoken-sp — fast SentencePiece unigram (Viterbi) tokenizer.
//!
//! `gigatoken`'s README admits SentencePiece is "not nearly as optimized" as
//! its BPE path (it uses a BPE-style implementation). The canonical SentencePiece
//! model is the *unigram* language model: every substring has a score, and the
//! best tokenization is the Viterbi (max-sum) path over the string. This crate
//! provides that clean unigram implementation as a standalone, dependency-free
//! module.
//!
//! Convention: a word-initial space is the `▁` (U+2581) marker (SentencePiece).
//! Load a real model with `from_model_lines` (lines of `piece\tscore`).

pub mod sentencepiece;

pub use sentencepiece::SentencePieceTokenizer;
