//! Contiguous byte resources and zero-copy document splitting.

use memchr::memmem;
use memmap2::Mmap;
use std::path::Path;

pub(crate) mod decompress;
pub mod file_source;
pub mod jsonl;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DocRef<'a>(pub &'a [u8]);

impl<'a> From<&'a [u8]> for DocRef<'a> {
    fn from(value: &'a [u8]) -> Self {
        DocRef(value)
    }
}

impl<'a> std::ops::Deref for DocRef<'a> {
    type Target = &'a [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Resource trait

/// A contiguous byte buffer that can be split into documents and parallel chunks.
pub trait Resource: Sync {
    fn as_bytes(&self) -> &[u8];

    /// Iterate documents by splitting on `separator`.
    /// If separator is empty, the entire buffer is one document.
    fn documents<'a>(&'a self, separator: &'a [u8]) -> DocumentIter<'a> {
        DocumentIter::new(self.as_bytes(), separator)
    }

    /// Split into `n` chunk iterators, each yielding documents.
    /// Chunk boundaries are aligned to separator positions so no document
    /// is split across chunks.
    fn par_document_chunks<'a>(&'a self, separator: &'a [u8], n: usize) -> Vec<DocumentIter<'a>> {
        par_document_chunks(self.as_bytes(), separator, n)
    }
}

// Covers slices, strings, Vec, and Mmap without parallel implementations that
// can drift; owned wrappers such as MmappedFile remain explicit below.
impl<T: AsRef<[u8]> + Sync + ?Sized> Resource for T {
    fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }
}

// MmappedFile

/// Owns a memory-mapped file and implements `Resource`.
pub struct MmappedFile {
    mmap: Mmap,
}

impl MmappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }
}

impl Resource for MmappedFile {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

// DocumentIter

/// Zero-copy iterator that splits a byte slice on a separator, yielding documents.
/// Empty documents (consecutive separators) are skipped.
pub struct DocumentIter<'a> {
    bytes: &'a [u8],
    separator: &'a [u8],
    /// Prebuilt searcher for `separator`, constructed once instead of per
    /// yielded document.
    finder: memmem::Finder<'a>,
    position: usize,
    end: usize,
}

impl<'a> DocumentIter<'a> {
    pub fn new(bytes: &'a [u8], separator: &'a [u8]) -> Self {
        Self::new_range(bytes, separator, 0, bytes.len())
    }

    fn new_range(bytes: &'a [u8], separator: &'a [u8], start: usize, end: usize) -> Self {
        Self {
            bytes,
            separator,
            finder: memmem::Finder::new(separator),
            position: start,
            end,
        }
    }
}

impl<'a> Iterator for DocumentIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Every terminal branch sets position to end before yielding, so
            // this offset is also the finished state; no separate flag is needed.
            if self.position >= self.end {
                return None;
            }

            let search_range = &self.bytes[self.position..self.end];

            if self.separator.is_empty() {
                self.position = self.end;
                return Some(search_range);
            }

            match self.finder.find(search_range) {
                Some(offset) => {
                    let doc = &self.bytes[self.position..self.position + offset];
                    self.position += offset + self.separator.len();
                    // Skip empty documents
                    if !doc.is_empty() {
                        return Some(doc);
                    }
                }
                None => {
                    self.position = self.end;
                    return Some(search_range);
                }
            }
        }
    }
}

// Parallel document chunking

/// Split `bytes` into `n` document iterators with chunk boundaries aligned
/// to separator positions. Each iterator covers a disjoint range.
fn par_document_chunks<'a>(
    bytes: &'a [u8],
    separator: &'a [u8],
    n: usize,
) -> Vec<DocumentIter<'a>> {
    if n <= 1 || bytes.is_empty() || separator.is_empty() {
        return vec![DocumentIter::new(bytes, separator)];
    }

    let chunk_size = bytes.len() / n;
    let finder = memmem::Finder::new(separator);

    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0usize);

    for i in 1..n {
        let target = i * chunk_size;
        // Scan forward from target to find the next separator
        match finder.find(&bytes[target..]) {
            Some(offset) => {
                let boundary = target + offset + separator.len();
                boundaries.push(boundary);
            }
            None => {
                // No separator found; remaining data goes in the last chunk
                break;
            }
        }
    }
    boundaries.push(bytes.len());
    boundaries.dedup();

    boundaries
        .windows(2)
        .map(|w| DocumentIter::new_range(bytes, separator, w[0], w[1]))
        .collect()
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_iter_basic() {
        let data = b"hello<|endoftext|>world<|endoftext|>foo";
        let sep = b"<|endoftext|>";
        let docs: Vec<&[u8]> = data.as_slice().documents(sep).collect();
        assert_eq!(docs, vec![b"hello".as_slice(), b"world", b"foo"]);
    }

    #[test]
    fn test_document_iter_no_separator() {
        let data = b"hello world";
        let docs: Vec<&[u8]> = data.as_slice().documents(b"<|endoftext|>").collect();
        assert_eq!(docs, vec![b"hello world".as_slice()]);
    }

    #[test]
    fn test_document_iter_empty_separator() {
        let data = b"hello world";
        let docs: Vec<&[u8]> = data.as_slice().documents(b"").collect();
        assert_eq!(docs, vec![b"hello world".as_slice()]);
    }

    #[test]
    fn test_document_iter_consecutive_separators() {
        let data = b"a<SEP><SEP>b";
        let docs: Vec<&[u8]> = data.as_slice().documents(b"<SEP>").collect();
        assert_eq!(docs, vec![b"a".as_slice(), b"b"]);
    }

    #[test]
    fn test_document_iter_separator_at_edges() {
        let data = b"<SEP>hello<SEP>";
        let docs: Vec<&[u8]> = data.as_slice().documents(b"<SEP>").collect();
        assert_eq!(docs, vec![b"hello".as_slice()]);
    }

    #[test]
    fn test_par_document_chunks_basic() {
        let sep = b"<|endoftext|>";
        let parts: Vec<&str> = (0..100)
            .map(|i| if i % 10 == 9 { "doc" } else { "word " })
            .collect();
        let data = parts.join(std::str::from_utf8(sep).unwrap());
        let bytes = data.as_bytes();

        let chunks = bytes.par_document_chunks(sep, 4);
        // All documents should be found across all chunks
        let all_docs: Vec<&[u8]> = chunks.into_iter().flatten().collect();
        let single_docs: Vec<&[u8]> = bytes.documents(sep).collect();
        assert_eq!(all_docs, single_docs);
    }

    #[test]
    fn test_par_chunks_single_thread() {
        let data = b"a<SEP>b<SEP>c";
        let chunks = data.as_slice().par_document_chunks(b"<SEP>", 1);
        assert_eq!(chunks.len(), 1);
        let docs: Vec<&[u8]> = chunks.into_iter().flatten().collect();
        assert_eq!(docs, vec![b"a".as_slice(), b"b", b"c"]);
    }
}
