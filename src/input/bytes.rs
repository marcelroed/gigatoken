//! Strategies to load data from bytes, always without ownership.
//! This includes strategies to find parallel boundaries and production of a parallel iterator.

use std::borrow::Cow;

use crate::input::Document;
use crate::token::TokenId;
use regex::bytes::Regex;

/// Bytes are always just a single document
// pub(super) struct BytesDocIter<'a> {
//     slice: &'a [u8],
//     spec: BytesDocIterSpec,
// }

#[derive(Default, Debug, Clone)]
pub(crate) struct BytesDocIterSpec {
    special_tokens: Vec<(Vec<u8>, TokenId)>,
    regex: Option<Regex>,
}

// impl BytesDocIterSpec {
//     pub(self) fn new(special_tokens: Vec<(Vec<u8>, u32)>) -> Result<Self, String> {
//         if special_tokens.is_empty() {
//             return Ok(Self::default());
//         }
//         let regex_expression = special_tokens
//             .iter()
//             .map(|(v, _t)| unsafe { str::from_utf8_unchecked(v) })
//             .join("|");
//         let regex = Some(
//             RegexBuilder::new(&regex_expression)
//                 .build()
//                 .map_err(|e| format!("Failed to build regex {e}"))?,
//         );
//         Ok(BytesDocIterSpec {
//             special_tokens,
//             regex,
//         })
//     }
//     // pub(self) fn iterate_documents<'a>(
//     //     &'a self,
//     //     bytes: &'a [u8],
//     // ) -> Box<dyn Iterator<Item = (Document<'a>, Option<u32>)> + 'a> {
//     //     // If there are no tokens to split on, the whole file is a single document
//     //     let Some(regex) = &self.regex else {
//     //         return Box::new(std::iter::once((Document(bytes.into()), None)));
//     //     };
//     //     let regex = self.regex.as_ref().unwrap();
//     //     Box::new(regex.find_iter(bytes).scan(0, |last_start, m| {
//     //         let up_to_slice = &bytes[*last_start..m.start()];
//     //         let match_bytes = m.as_bytes();
//     //         let token = self
//     //             .special_tokens
//     //             .iter()
//     //             .find(|(bytes, _i)| bytes == match_bytes)
//     //             .unwrap();
//     //         *last_start = m.end();
//     //         Some((Document(up_to_slice.into()), Some(token.1)))
//     //     }))
//     // }
// }

struct BytesRepresentation<'a> {
    pub(self) bytes: Cow<'a, [u8]>,
}

impl<'a> IntoIterator for BytesRepresentation<'a> {
    type Item = Document<'a>;
    type IntoIter = std::iter::Once<Document<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(Document(self.bytes.into()))
    }
}

// pub(super) trait IntoBytesDocIter<'a> {
//     fn doc_iter(self) -> BytesDocIter<'a>;
// }

// impl<'a> IntoBytesDocIter<'a> for &'a [u8] {
//     fn doc_iter(self) -> BytesDocIter<'a> {
//         self.
//     }
// }

// impl<'a> Iterator for BytesDocIter<'a> {
//     type Item = Document<'a>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//
//     }
// }

// impl<'a> IntoParallelIterator for BytesRepresentation<'a> {
//     type Iter = ;

//     type Item = Cow<'a, [u8]>;

//     fn into_par_iter(self) -> Self::Iter {
//         self.bytes.
//     }
// }
