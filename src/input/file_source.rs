use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::input::decompress;
use crate::input::jsonl::JsonLinesReader;
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
enum ContentFormat {
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

/// Pretokenize documents from a streaming reader.
/// For JSONL: each line is a document (field extracted from JSON).
/// For plain text: documents are split on `separator`.
/// Never buffers the entire decompressed file.
fn pretokenize_streaming(
    reader: impl std::io::BufRead,
    content: ContentFormat,
    field: &str,
    separator: &[u8],
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
    let mut count_pretokens = |doc: &[u8]| {
        for pretoken in pretokenize_as_iter(doc) {
            *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
        }
    };

    match content {
        ContentFormat::Jsonl => {
            for doc in JsonLinesReader::new(reader, field) {
                count_pretokens(doc.as_ref());
            }
        }
        ContentFormat::PlainText => {
            for doc in SeparatorReader::new(reader, separator) {
                count_pretokens(&doc);
            }
        }
    }
    counts
}

fn pretokenize_file(
    path: &Path,
    content: ContentFormat,
    compression: Compression,
    field: &str,
    separator: &[u8],
) -> Result<HashMap<Vec<u8>, usize, FxBuildHasher>, std::io::Error> {
    eprintln!("Processing {:?} ({:?}, {:?})", path, content, compression);

    if matches!(compression, Compression::None) && matches!(content, ContentFormat::PlainText) {
        // Uncompressed text: memory-map for zero-copy parallel document chunking
        let resource = MmappedFile::open(path)?;
        return Ok(pretokenize_plain_text_bytes(resource.as_bytes(), separator));
    }

    // Everything else: stream from reader (never fully in memory)
    Ok(match compression {
        Compression::None => pretokenize_streaming(decompress::open_plain(path)?, content, field, separator),
        Compression::Gzip => pretokenize_streaming(decompress::open_gzip(path)?, content, field, separator),
        Compression::Zstd => pretokenize_streaming(decompress::open_zstd(path)?, content, field, separator),
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
                Ok(buf) if buf.is_empty() => {
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

pub(crate) struct FileSourceSpec {
    pub paths: Vec<PathBuf>,
    pub field: String,
    pub separator: Vec<u8>,
}

impl FileSourceSpec {
    pub fn pretokenize(&self) -> Result<HashMap<Vec<u8>, usize, FxBuildHasher>, std::io::Error> {
        let files: Vec<_> = self
            .paths
            .iter()
            .map(|p| {
                let (content, compression) = detect_format(p);
                (p.clone(), content, compression)
            })
            .collect();

        eprintln!(
            "FileSource: processing {} files across {} threads",
            files.len(),
            rayon::current_num_threads()
        );

        files
            .par_iter()
            .map(|(path, content, compression)| {
                pretokenize_file(path, *content, *compression, &self.field, &self.separator)
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
