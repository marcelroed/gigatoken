use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::input::decompress;
use crate::input::jsonl::{JsonLinesReader, JsonLinesSlice};
use crate::input::MmappedFile;
use crate::input::Resource;
use crate::pretokenize::{pretokenize_as_iter, pretokenize_par_bytes};

// ---------------------------------------------------------------------------
// File format detection: compression and content format are independent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) enum Compression {
    None,
    Gzip,
    Zstd,
}

#[derive(Debug, Clone, Copy)]
pub enum ContentFormat {
    PlainText,
    Jsonl,
}

/// Strip compression extension and detect compression type.
/// Returns (stem without compression ext, compression).
fn detect_compression(name: &str) -> (&str, Compression) {
    if let Some(stem) = name.strip_suffix(".zst").or_else(|| name.strip_suffix(".zstd")) {
        (stem, Compression::Zstd)
    } else if let Some(stem) = name.strip_suffix(".gz") {
        (stem, Compression::Gzip)
    } else {
        (name, Compression::None)
    }
}

/// Detect content format from the (uncompressed) filename stem.
fn detect_content_format(stem: &str) -> ContentFormat {
    if stem.ends_with(".jsonl") {
        ContentFormat::Jsonl
    } else {
        ContentFormat::PlainText
    }
}

fn detect_format(path: &Path) -> (ContentFormat, Compression) {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let (stem, compression) = detect_compression(name);
    let content = detect_content_format(stem);
    (content, compression)
}

// ---------------------------------------------------------------------------
// DocFormat: how a file's bytes split into documents
// ---------------------------------------------------------------------------

/// How to split a file's bytes into documents. Carried by the Python
/// `TextFileSource` / `JsonlFileSource` classes; compression is orthogonal
/// and always detected from the file extension.
#[derive(Debug, Clone)]
pub enum DocFormat {
    /// Plain text. With a separator, documents are the pieces between
    /// occurrences; without one, the whole file is a single document.
    Text { separator: Option<Vec<u8>> },
    /// JSON Lines: one document per line, text taken from `field`.
    Jsonl { field: String },
}

impl DocFormat {
    /// Separator to split text documents on. Empty means "whole input is one
    /// document", matching `DocumentIter`/`SeparatorReader` semantics.
    fn separator(&self) -> &[u8] {
        match self {
            DocFormat::Text { separator } => separator.as_deref().unwrap_or(b""),
            DocFormat::Jsonl { .. } => b"",
        }
    }
}

/// Default format for a bare path: JSONL (field "text") if the uncompressed
/// name ends in .jsonl, otherwise plain text with the whole file as one
/// document.
pub fn detect_default_format(path: &Path) -> DocFormat {
    match detect_format(path).0 {
        ContentFormat::Jsonl => DocFormat::Jsonl {
            field: "text".to_string(),
        },
        ContentFormat::PlainText => DocFormat::Text { separator: None },
    }
}

// ---------------------------------------------------------------------------
// Loading and chunking files for parallel encoding
// ---------------------------------------------------------------------------

/// A file's full contents: mmapped when stored uncompressed, otherwise
/// decompressed into memory (parallel chunking needs random access).
pub enum LoadedFile {
    Mmapped(MmappedFile),
    Owned(Vec<u8>),
}

impl LoadedFile {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            LoadedFile::Mmapped(m) => m.as_bytes(),
            LoadedFile::Owned(v) => v,
        }
    }
}

/// Open a file for encoding: mmap if uncompressed, else decompress fully.
pub fn load_file(path: &Path) -> Result<LoadedFile, std::io::Error> {
    use std::io::Read;
    let (_, compression) = detect_format(path);
    Ok(match compression {
        Compression::None => LoadedFile::Mmapped(MmappedFile::open(path)?),
        Compression::Gzip => {
            let mut buf = Vec::new();
            decompress::open_gzip(path)?.read_to_end(&mut buf)?;
            LoadedFile::Owned(buf)
        }
        Compression::Zstd => {
            let mut buf = Vec::new();
            decompress::open_zstd(path)?.read_to_end(&mut buf)?;
            LoadedFile::Owned(buf)
        }
    })
}

/// Cut `bytes` into ranges of roughly `target` bytes, each ending on a
/// document boundary so no document spans two chunks. A single range is
/// returned when the input is smaller than `target` or has no boundaries
/// (plain text without a separator is one document, which cannot be split
/// without changing tokenization).
pub fn chunk_ranges(
    bytes: &[u8],
    format: &DocFormat,
    target: usize,
) -> Vec<std::ops::Range<usize>> {
    let len = bytes.len();
    // `next_boundary(probe)` finds the first document boundary at or after
    // `probe` and returns (chunk_end, next_chunk_start).
    let cut = |next_boundary: &dyn Fn(usize) -> Option<(usize, usize)>| {
        let mut out = Vec::new();
        let mut start = 0;
        while start < len {
            let probe = start + target;
            match (probe < len).then(|| next_boundary(probe)).flatten() {
                Some((end, next_start)) => {
                    out.push(start..end);
                    start = next_start;
                }
                None => {
                    out.push(start..len);
                    break;
                }
            }
        }
        if out.is_empty() {
            out.push(0..0); // empty file: one empty chunk, so files stay 1:1
        }
        out
    };
    match format {
        DocFormat::Jsonl { .. } => cut(&|probe| {
            memchr::memchr(b'\n', &bytes[probe..]).map(|off| (probe + off + 1, probe + off + 1))
        }),
        DocFormat::Text { separator: Some(sep) } if !sep.is_empty() => {
            let finder = memchr::memmem::Finder::new(sep);
            cut(&|probe| {
                finder
                    .find(&bytes[probe..])
                    .map(|off| (probe + off, probe + off + sep.len()))
            })
        }
        // One chunk spanning the whole file (a single Range element, not 0..len values).
        DocFormat::Text { .. } => std::iter::once(0..len).collect(),
    }
}

// ---------------------------------------------------------------------------
// Per-file processing
// ---------------------------------------------------------------------------

fn pretokenize_plain_text_bytes(
    bytes: &[u8],
    separator: &[u8],
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let borrowed_counts = pretokenize_par_bytes(bytes, separator);
    borrowed_counts
        .into_iter()
        .map(|(k, v)| (k.as_ref().to_vec(), v))
        .collect()
}

/// Parallel JSONL pretokenization on a memory-mapped byte slice.
/// Splits at newline boundaries into N chunks, each chunk processes its
/// JSONL lines independently.
fn pretokenize_jsonl_par(
    bytes: &[u8],
    field: &str,
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let n_threads = rayon::current_num_threads();
    if bytes.is_empty() {
        return HashMap::default();
    }

    let boundaries = jsonl_chunk_boundaries(bytes, n_threads);

    boundaries
        .par_windows(2)
        .map(|w| {
            let chunk = &bytes[w[0]..w[1]];
            let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
            for doc in JsonLinesSlice::new(chunk, field) {
                for pretoken in pretokenize_as_iter(doc.as_ref()) {
                    *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
                }
            }
            counts
        })
        .reduce(HashMap::default, |mut acc, counts| {
            if acc.is_empty() {
                return counts;
            }
            for (k, v) in counts {
                *acc.entry(k).or_default() += v;
            }
            acc
        })
}

/// Pretokenize documents from a streaming reader.
/// For JSONL: each line is a document (field extracted from JSON).
/// For plain text: documents are split on `separator`.
/// Never buffers the entire decompressed file.
fn pretokenize_streaming(
    reader: impl std::io::BufRead,
    format: &DocFormat,
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
    let mut count_pretokens = |doc: &[u8]| {
        for pretoken in pretokenize_as_iter(doc) {
            *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
        }
    };

    match format {
        DocFormat::Jsonl { field } => {
            for doc in JsonLinesReader::new(reader, field) {
                count_pretokens(doc.as_ref());
            }
        }
        DocFormat::Text { .. } => {
            for doc in SeparatorReader::new(reader, format.separator()) {
                count_pretokens(&doc);
            }
        }
    }
    counts
}

fn pretokenize_file(
    path: &Path,
    format: &DocFormat,
    compression: Compression,
) -> Result<HashMap<Vec<u8>, usize, FxBuildHasher>, std::io::Error> {
    eprintln!("Processing {:?} ({:?}, {:?})", path, format, compression);

    // Uncompressed files: memory-map for parallel processing
    if matches!(compression, Compression::None) {
        let resource = MmappedFile::open(path)?;
        return Ok(match format {
            DocFormat::Text { .. } => {
                pretokenize_plain_text_bytes(resource.as_bytes(), format.separator())
            }
            DocFormat::Jsonl { field } => pretokenize_jsonl_par(resource.as_bytes(), field),
        });
    }

    // Compressed files: stream from reader (never fully in memory)
    Ok(match compression {
        Compression::None => unreachable!(),
        Compression::Gzip => pretokenize_streaming(decompress::open_gzip(path)?, format),
        Compression::Zstd => pretokenize_streaming(decompress::open_zstd(path)?, format),
    })
}

// ---------------------------------------------------------------------------
// SeparatorReader: streaming document splitter for plain text
// ---------------------------------------------------------------------------

/// Reads from a `BufRead` and yields documents split on a byte separator.
/// Each `next()` returns one document (the bytes between two separators).
/// Buffers only the current document, not the entire input.
struct SeparatorReader<R> {
    reader: R,
    separator: Vec<u8>,
    buf: Vec<u8>,
    finished: bool,
}

impl<R: std::io::BufRead> SeparatorReader<R> {
    fn new(reader: R, separator: &[u8]) -> Self {
        Self {
            reader,
            separator: separator.to_vec(),
            buf: Vec::with_capacity(4096),
            finished: false,
        }
    }
}

impl<R: std::io::BufRead> Iterator for SeparatorReader<R> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        if self.finished {
            return None;
        }

        // Empty separator: yield all remaining bytes as one document
        if self.separator.is_empty() {
            self.finished = true;
            let mut all = Vec::new();
            self.reader.read_to_end(&mut all).ok()?;
            return if all.is_empty() { None } else { Some(all) };
        }

        let finder = memchr::memmem::Finder::new(&self.separator);
        self.buf.clear();

        loop {
            let available = match self.reader.fill_buf() {
                Ok([]) => {
                    // EOF
                    self.finished = true;
                    return if self.buf.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(&mut self.buf))
                    };
                }
                Ok(buf) => buf,
                Err(_) => {
                    self.finished = true;
                    return None;
                }
            };

            // Search for separator in: [tail of buf where separator could span] + available
            // We need to handle the case where the separator spans the boundary between
            // self.buf and the new `available` data.
            let overlap = self.separator.len().saturating_sub(1).min(self.buf.len());
            let search_start = self.buf.len() - overlap;

            // Append available to buf, then search from search_start
            self.buf.extend_from_slice(available);
            let consumed = available.len();
            self.reader.consume(consumed);

            if let Some(pos) = finder.find(&self.buf[search_start..]) {
                let sep_start = search_start + pos;
                let doc = self.buf[..sep_start].to_vec();
                // Keep everything after the separator for the next call
                let remainder_start = sep_start + self.separator.len();
                let remainder = self.buf[remainder_start..].to_vec();
                self.buf = remainder;

                // Skip empty documents
                if doc.is_empty() {
                    continue;
                }
                return Some(doc);
            }
            // No separator found yet — continue reading
        }
    }
}

// ---------------------------------------------------------------------------
// FileSourceSpec — multi-file parallel pretokenization
// ---------------------------------------------------------------------------

pub struct FileSourceSpec {
    pub paths: Vec<PathBuf>,
    pub format: DocFormat,
}

/// Newline-aligned chunk boundaries for parallel JSONL processing.
fn jsonl_chunk_boundaries(bytes: &[u8], n_chunks: usize) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(n_chunks + 1);
    boundaries.push(0usize);
    let chunk_size = bytes.len() / n_chunks;
    for i in 1..n_chunks {
        let target = i * chunk_size;
        match memchr::memchr(b'\n', &bytes[target..]) {
            Some(offset) => boundaries.push(target + offset + 1),
            None => break,
        }
    }
    boundaries.push(bytes.len());
    boundaries.dedup();
    boundaries
}

impl FileSourceSpec {
    pub fn pretokenize(&self) -> Result<HashMap<Vec<u8>, usize, FxBuildHasher>, std::io::Error> {
        eprintln!(
            "FileSource: processing {} files across {} threads",
            self.paths.len(),
            rayon::current_num_threads()
        );

        self.paths
            .par_iter()
            .map(|path| {
                let (_, compression) = detect_format(path);
                pretokenize_file(path, &self.format, compression)
            })
            .try_reduce(HashMap::default, |mut acc, counts| {
                if acc.is_empty() {
                    return Ok(counts);
                }
                for (k, v) in counts {
                    *acc.entry(k).or_default() += v;
                }
                Ok(acc)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_compression() {
        assert!(matches!(detect_compression("data.jsonl.zst"), (_, Compression::Zstd)));
        assert!(matches!(detect_compression("data.jsonl.zstd"), (_, Compression::Zstd)));
        assert!(matches!(detect_compression("data.txt.gz"), (_, Compression::Gzip)));
        assert!(matches!(detect_compression("data.jsonl"), (_, Compression::None)));
        assert!(matches!(detect_compression("data.txt"), (_, Compression::None)));
    }

    #[test]
    fn test_detect_compression_strips_ext() {
        assert_eq!(detect_compression("data.jsonl.zst").0, "data.jsonl");
        assert_eq!(detect_compression("data.jsonl.zstd").0, "data.jsonl");
        assert_eq!(detect_compression("data.txt.gz").0, "data.txt");
        assert_eq!(detect_compression("data.jsonl.gz").0, "data.jsonl");
        assert_eq!(detect_compression("data.txt").0, "data.txt");
    }

    #[test]
    fn test_detect_format_combinations() {
        // jsonl + compression
        assert!(matches!(detect_format(Path::new("data.jsonl.zst")), (ContentFormat::Jsonl, Compression::Zstd)));
        assert!(matches!(detect_format(Path::new("data.jsonl.zstd")), (ContentFormat::Jsonl, Compression::Zstd)));
        assert!(matches!(detect_format(Path::new("data.jsonl.gz")), (ContentFormat::Jsonl, Compression::Gzip)));
        assert!(matches!(detect_format(Path::new("data.jsonl")), (ContentFormat::Jsonl, Compression::None)));

        // txt + compression
        assert!(matches!(detect_format(Path::new("data.txt.zst")), (ContentFormat::PlainText, Compression::Zstd)));
        assert!(matches!(detect_format(Path::new("data.txt.gz")), (ContentFormat::PlainText, Compression::Gzip)));
        assert!(matches!(detect_format(Path::new("data.txt")), (ContentFormat::PlainText, Compression::None)));

        // bare compression extension → plain text (unknown inner format)
        assert!(matches!(detect_format(Path::new("data.zst")), (ContentFormat::PlainText, Compression::Zstd)));
        assert!(matches!(detect_format(Path::new("data.gz")), (ContentFormat::PlainText, Compression::Gzip)));

        // unknown → plain text, no compression
        assert!(matches!(detect_format(Path::new("data.csv")), (ContentFormat::PlainText, Compression::None)));
    }
}
