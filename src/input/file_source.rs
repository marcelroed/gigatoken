use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::input::decompress;
use crate::input::jsonl::JsonLinesIter;
use crate::input::MmappedFile;
use crate::input::Resource;
use crate::pretokenize::{pretokenize_as_iter, pretokenize_par_bytes, Pretokenize};

// ---------------------------------------------------------------------------
// File format detection: compression and content format are independent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Compression {
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

/// Load file bytes, decompressing if needed.
fn load_bytes(path: &Path, compression: Compression) -> std::io::Result<Vec<u8>> {
    match compression {
        Compression::None => std::fs::read(path),
        Compression::Gzip => decompress::decompress_gzip(path),
        Compression::Zstd => decompress::decompress_zstd(path),
    }
}

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

fn pretokenize_jsonl_from_bytes(
    bytes: &[u8],
    field: &str,
) -> HashMap<Vec<u8>, usize, FxBuildHasher> {
    let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
    for doc in JsonLinesIter::new(bytes, field) {
        for pretoken in pretokenize_as_iter(doc.as_ref()) {
            *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
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

    // Uncompressed plain text can be memory-mapped for zero-copy parallel processing
    if matches!(compression, Compression::None) && matches!(content, ContentFormat::PlainText) {
        let resource = MmappedFile::open(path)?;
        return Ok(pretokenize_plain_text_bytes(resource.as_bytes(), separator));
    }

    // Everything else: load (and decompress) into memory, then process
    let bytes = load_bytes(path, compression)?;
    Ok(match content {
        ContentFormat::PlainText => pretokenize_plain_text_bytes(&bytes, separator),
        ContentFormat::Jsonl => pretokenize_jsonl_from_bytes(&bytes, field),
    })
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
