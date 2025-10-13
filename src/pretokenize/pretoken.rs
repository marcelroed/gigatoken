//! Once we have a document, we can pretokenize it (potentially in parallel)

// use std::borrow::Cow;

use std::ops::Deref;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Pretoken<'a>(pub &'a [u8]);

impl AsRef<[u8]> for Pretoken<'_> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<'a> Deref for Pretoken<'a> {
    type Target = &'a [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// #[derive(Debug, Hash, PartialEq, Eq)]
// pub struct PretokenBuf(pub [u8]);
//
// impl Deref for PretokenBuf {
//     type Target = [u8];
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// #[derive(Debug, Clone)]
// struct DocumentPretokenIter<'a> {
//     bytes: Document<'a>,
//     position: usize,
// }

// impl<'a> Iterator for DocumentPretokenIter<'a> {
//     type Item = &'a [u8];
// }
