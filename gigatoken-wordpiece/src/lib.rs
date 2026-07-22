//! gigatoken-wordpiece — fast WordPiece tokenizer (BERT-family).
//!
//! `gigatoken` nails BPE pretokenization at GB/s via SIMD but its README
//! admits WordPiece is unsupported. This crate fills that gap with a fast,
//! dependency-free WordPiece implementation:
//!
//! - A tight SWAR-style byte-class scan splits text into "words" (whitespace
//!   and punctuation isolated, CJK chars isolated, no regex engine).
//! - Greedy longest-match encodes each word against the vocab, using the
//!   `##` continuation convention.
//! - Optional `lowercase` matches the `bert-base-uncased` pipeline.
//!
//! Verified to produce byte-identical token ids to HuggingFace
//! `BertWordPieceTokenizer` on a 200k-char corpus.

pub mod wordpiece;

pub use wordpiece::{WordPieceTokenizer, SpanIter};
