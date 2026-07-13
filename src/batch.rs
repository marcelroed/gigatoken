//! Parallel chunked batch encoding: the engine behind encode_batch and
//! encode_files. Documents are grouped into coarse chunks (an oversized BPE
//! document is split at pretoken-safe boundaries), encoded by pooled workers
//! whose pretoken caches persist across calls, and reassembled into one flat
//! id buffer plus per-document row lengths.

use crate::Tokenizer;
use crate::bpe;
use crate::input::DocumentIter;
use crate::input::file_source::{DocFormat, chunk_ranges};
use std::ops::Range;
use std::sync::{Mutex, OnceLock, TryLockError};

/// Parallel chunks must hold at least this many bytes: a chunk this size
/// encodes for tens of milliseconds, so worker acquisition and rayon
/// scheduling/work-stealing overhead is noise. An input that does not fill
/// more than one chunk is encoded serially — for small inputs the thread
/// fan-out costs more than it saves.
const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Target bytes per parallel chunk: ~16 chunks per thread for work-stealing
/// load balancing, floored at MIN_CHUNK_BYTES so chunks stay coarse.
pub(crate) fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (16 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
}

/// Append one document's token ids to `ids` and its row length to `lens`.
pub(crate) fn encode_into(tokenizer: &mut Tokenizer, doc: &[u8], ids: &mut Vec<u32>, lens: &mut Vec<i64>) {
    let before = ids.len();
    tokenizer.encode_with_added_tokens(doc, |tokens| {
        ids.extend(tokens.iter().map(|&t| u32::from(t)))
    });
    lens.push((ids.len() - before) as i64);
}

/// SentencePiece analog of `encode_into`, using `encoder`'s pretoken cache.
pub(crate) fn sp_encode_into(
    encoder: &mut bpe::sentencepiece::Encoder<'_>,
    text: &str,
    ids: &mut Vec<u32>,
    lens: &mut Vec<i64>,
) {
    let before = ids.len();
    encoder.encode_raw_cb(text, &mut |tokens| {
        ids.extend(tokens.iter().map(|&t| u32::from(t)))
    });
    lens.push((ids.len() - before) as i64);
}

/// Iterate the documents in a byte region per `format`: JSONL lines,
/// separator-delimited text, or the whole region as one document.
pub(crate) fn for_each_doc(bytes: &[u8], format: &DocFormat, mut f: impl FnMut(&[u8])) {
    use crate::input::jsonl::JsonLinesSlice;
    match format {
        DocFormat::Jsonl { field } => {
            for doc in JsonLinesSlice::new(bytes, field) {
                f(doc.as_ref());
            }
        }
        DocFormat::Text {
            separator: Some(sep),
        } if !sep.is_empty() => {
            for doc in DocumentIter::new(bytes, sep) {
                f(doc);
            }
        }
        DocFormat::Text { .. } => f(bytes),
    }
}

/// Work unit for parallel encoding.
pub(crate) enum EncodeChunk<'a> {
    /// A run of whole documents, one output row each.
    Docs(Vec<&'a [u8]>),
    /// A byte region holding many documents, split during encoding
    /// (JSONL lines or separator-delimited text).
    Region {
        bytes: &'a [u8],
        format: &'a DocFormat,
    },
    /// A pretoken-safe fragment of one oversized document (see
    /// `pretokenize::safe_split_ranges`). Fragments of a document are
    /// consecutive chunks; `first` marks the document's first fragment.
    Fragment { bytes: &'a [u8], first: bool },
}

/// Token output of one chunk: a flat id buffer plus one length per document
/// row. `continues` means the first length extends the previous chunk's
/// last row (a non-first fragment of a split document).
pub(crate) struct ChunkTokens {
    pub(crate) ids: Vec<u32>,
    pub(crate) lens: Vec<i64>,
    pub(crate) continues: bool,
}

fn encode_chunk(tokenizer: &mut Tokenizer, chunk: &EncodeChunk) -> ChunkTokens {
    // Reserve the output once, from a bytes-per-token estimate on the low
    // side of natural language (~4.4 on OWT/GPT-2). Growing from empty
    // instead re-copies roughly the final size in doublings — per chunk,
    // on every chunk of a first pass.
    let byte_len = match chunk {
        EncodeChunk::Docs(docs) => docs.iter().map(|d| d.len()).sum::<usize>(),
        EncodeChunk::Region { bytes, .. } | EncodeChunk::Fragment { bytes, .. } => bytes.len(),
    };
    let mut ids = Vec::with_capacity(byte_len / 4 + 16);
    let mut lens = Vec::new();
    let mut continues = false;
    match chunk {
        EncodeChunk::Docs(docs) => {
            for doc in docs {
                encode_into(tokenizer, doc, &mut ids, &mut lens);
            }
        }
        EncodeChunk::Region { bytes, format } => {
            for_each_doc(bytes, format, |doc| {
                encode_into(tokenizer, doc, &mut ids, &mut lens)
            })
        }
        EncodeChunk::Fragment { bytes, first } => {
            encode_into(tokenizer, bytes, &mut ids, &mut lens);
            continues = !*first;
        }
    }
    ChunkTokens {
        ids,
        lens,
        continues,
    }
}

/// Group documents into parallel chunks of at least `target` cumulative
/// bytes. A document larger than the target is split into consecutive
/// Fragment chunks at pretoken-safe boundaries that no added-token
/// occurrence straddles, so even a single huge document is encoded across
/// all cores with token-identical output.
fn build_doc_chunks<'a>(
    docs: &[&'a [u8]],
    target: usize,
    added_tokens: &[&[u8]],
) -> Vec<EncodeChunk<'a>> {
    let mut chunks = Vec::new();
    let mut group: Vec<&[u8]> = Vec::new();
    let mut acc = 0usize;
    for &doc in docs {
        if doc.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
                acc = 0;
            }
            for (k, r) in crate::pretokenize::safe_split_ranges(doc, target, added_tokens)
                .into_iter()
                .enumerate()
            {
                chunks.push(EncodeChunk::Fragment {
                    bytes: &doc[r],
                    first: k == 0,
                });
            }
            continue;
        }
        group.push(doc);
        acc += doc.len();
        if acc >= target {
            chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(EncodeChunk::Docs(group));
    }
    chunks
}

/// Map items serially when there is at most one (small inputs skip the
/// thread fan-out), in parallel otherwise.
pub(crate) fn map_maybe_par<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use rayon::prelude::*;
    if items.len() <= 1 {
        items.iter().map(&f).collect()
    } else {
        items.par_iter().map(&f).collect()
    }
}

/// Encode all chunks with pooled workers — in parallel when there is more
/// than one chunk, serially otherwise.
pub(crate) fn encode_chunks_pooled(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
) -> Vec<ChunkTokens> {
    map_maybe_par(chunks, |c| {
        workers.with_worker(proto, |tok| encode_chunk(tok, c))
    })
}

/// Merge per-chunk outputs into one flat id buffer and per-document row
/// counts. The flat gather copies chunk buffers in parallel into a single
/// allocation.
pub(crate) fn assemble_ragged(chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<i64>) {
    use rayon::prelude::*;
    let mut counts: Vec<i64> = Vec::new();
    for chunk in &chunks {
        let mut lens = chunk.lens.iter().copied();
        if chunk.continues
            && let Some(l) = lens.next()
        {
            *counts
                .last_mut()
                .expect("continuation fragment before any document") += l;
        }
        counts.extend(lens);
    }
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    let mut flat = vec![0u32; total];
    // Ask for 2 MiB pages before first touch: the parallel copy below
    // faults in the whole (multi-GB) buffer, and huge pages cut the fault
    // count ~500x. No-op where THP is unavailable.
    #[cfg(target_os = "linux")]
    if total > 0 {
        // SAFETY: the range is exactly this allocation; MADV_HUGEPAGE only
        // hints page sizing.
        unsafe {
            libc::madvise(
                flat.as_mut_ptr() as *mut libc::c_void,
                total * std::mem::size_of::<u32>(),
                libc::MADV_HUGEPAGE,
            );
        }
    }
    let mut rest: &mut [u32] = &mut flat;
    let mut slices = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let (head, tail) = rest.split_at_mut(chunk.ids.len());
        slices.push(head);
        rest = tail;
    }
    slices
        .into_par_iter()
        .zip(chunks.par_iter())
        .for_each(|(dst, chunk)| dst.copy_from_slice(&chunk.ids));
    (flat, counts)
}

/// Persistent pool of forked tokenizer workers used by encode_batch and
/// encode_files. One slot per rayon thread, forked lazily on first use and
/// retained for the tokenizer's lifetime, so each worker's pretoken cache
/// stays warm when encoding is invoked repeatedly (e.g. in a loop).
pub struct WorkerPool {
    slots: OnceLock<Vec<Mutex<Option<Tokenizer>>>>,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerPool {
    pub fn new() -> Self {
        Self {
            slots: OnceLock::new(),
        }
    }

    /// Run `f` with exclusive access to a pooled worker. Rayon never runs
    /// more tasks concurrently than it has threads, and there is one slot
    /// per thread, so a free slot always exists; the yield loop only spins
    /// when non-rayon threads encode at the same time.
    fn with_worker<R>(&self, proto: &Tokenizer, f: impl FnOnce(&mut Tokenizer) -> R) -> R {
        let slots = self.slots.get_or_init(|| {
            (0..rayon::current_num_threads())
                .map(|_| Mutex::new(None))
                .collect()
        });
        loop {
            for slot in slots {
                match slot.try_lock() {
                    Ok(mut guard) => {
                        return f(guard.get_or_insert_with(|| proto.fork()));
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        // A worker panicked mid-encode; its cache may be
                        // inconsistent, so rebuild it from the prototype.
                        let mut guard = poisoned.into_inner();
                        *guard = None;
                        return f(guard.get_or_insert_with(|| proto.fork()));
                    }
                    Err(TryLockError::WouldBlock) => {}
                }
            }
            std::thread::yield_now();
        }
    }
}

/// Shared core of encode_batch / encode_files for pre-resolved document
/// slices: chunk (splitting oversized documents at pretoken-safe
/// boundaries), encode with pooled workers, and assemble the ragged result
/// (one flat id buffer plus per-document row lengths). Public so Rust
/// benches exercise the identical parallel path as the Python bindings.
pub fn encode_docs_ragged(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let added = proto.added_token_contents();
    let chunks = build_doc_chunks(docs, chunk_target_bytes(total), &added);
    let outs = encode_chunks_pooled(workers, proto, &chunks);
    assemble_ragged(outs)
}

/// SentencePiece analog of `encode_docs_ragged`: group whole documents into
/// parallel chunks and encode each with its own Encoder. SentencePiece
/// merges can span the whole document, so oversized documents are never
/// split.
pub(crate) fn sp_encode_docs_ragged(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&str],
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = texts.iter().map(|t| t.len()).sum();
    let target = chunk_target_bytes(total);
    let mut chunks: Vec<Vec<&str>> = Vec::new();
    let mut group: Vec<&str> = Vec::new();
    let mut acc = 0usize;
    for &text in texts {
        group.push(text);
        acc += text.len();
        if acc >= target {
            chunks.push(std::mem::take(&mut group));
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(group);
    }
    let outs = map_maybe_par(&chunks, |group| {
        let mut encoder = tokenizer.encoder();
        let mut ids: Vec<u32> = Vec::new();
        let mut lens: Vec<i64> = Vec::new();
        for text in group {
            sp_encode_into(&mut encoder, text, &mut ids, &mut lens);
        }
        ChunkTokens {
            ids,
            lens,
            continues: false,
        }
    });
    assemble_ragged(outs)
}

/// encode_files core for the BPE backend. With no separator each file is one
/// document (small files are grouped, huge ones split at pretoken-safe
/// boundaries); otherwise each file is cut into byte regions at document
/// boundaries and documents are extracted while encoding.
pub(crate) fn encode_files_docs(
    workers: &WorkerPool,
    proto: &Tokenizer,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    if matches!(format, DocFormat::Text { separator: None }) {
        return encode_docs_ragged(workers, proto, files);
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    let target = chunk_target_bytes(total);
    let chunks: Vec<EncodeChunk> = files
        .iter()
        .flat_map(|&bytes| {
            chunk_ranges(bytes, format, target)
                .into_iter()
                .map(move |r| EncodeChunk::Region {
                    bytes: &bytes[r],
                    format,
                })
        })
        .collect();
    let outs = encode_chunks_pooled(workers, proto, &chunks);
    assemble_ragged(outs)
}

/// encode_files core for the SentencePiece backend: cut files into byte
/// regions at document boundaries and encode each region's documents with a
/// per-chunk Encoder. Documents are assumed to be valid UTF-8.
pub(crate) fn sp_encode_files_docs(
    tokenizer: &bpe::SentencePieceBPE,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = files.iter().map(|f| f.len()).sum();
    let target = chunk_target_bytes(total);
    let chunks: Vec<(usize, Range<usize>)> = files
        .iter()
        .enumerate()
        .flat_map(|(i, &bytes)| {
            chunk_ranges(bytes, format, target)
                .into_iter()
                .map(move |r| (i, r))
        })
        .collect();
    let outs = map_maybe_par(&chunks, |(file, range)| {
        let bytes = &files[*file][range.clone()];
        let mut encoder = tokenizer.encoder();
        let mut ids: Vec<u32> = Vec::new();
        let mut lens: Vec<i64> = Vec::new();
        for_each_doc(bytes, format, |doc| {
            let text = unsafe { std::str::from_utf8_unchecked(doc) };
            sp_encode_into(&mut encoder, text, &mut ids, &mut lens);
        });
        ChunkTokens {
            ids,
            lens,
            continues: false,
        }
    });
    assemble_ragged(outs)
}
