//! Parallel chunked batch encoding: the engine behind encode_batch and
//! encode_files. Documents are grouped into coarse chunks (an oversized BPE
//! document is split at pretoken-safe boundaries), encoded by pooled workers
//! whose pretoken caches persist across calls, and reassembled into one flat
//! id buffer plus per-document row lengths.

use crate::Tokenizer;
use crate::bpe;
use crate::bpe::madvise_hugepage;
use crate::input::DocumentIter;
use crate::input::file_source::{DocFormat, chunk_ranges};
use std::ops::Range;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};

/// Parallel chunks must hold at least this many bytes: a chunk this size
/// encodes for tens of milliseconds, so worker acquisition and rayon
/// scheduling/work-stealing overhead is noise. An input that does not fill
/// more than one chunk is encoded serially — for small inputs the thread
/// fan-out costs more than it saves.
const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Results at least this large get their per-chunk buffers freed on a
/// background task after the gather returns (see `defer_drop`); smaller
/// ones drop inline.
const DEFERRED_DROP_MIN_BYTES: usize = 32 << 20;

/// Target bytes per parallel chunk: ~16 chunks per thread for work-stealing
/// load balancing, floored at MIN_CHUNK_BYTES so chunks stay coarse.
pub(crate) fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (16 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
}

/// Append one document's token ids to `ids` and its row length to `lens`.
pub(crate) fn encode_into(tokenizer: &mut Tokenizer, doc: &[u8], ids: &mut Vec<u32>, lens: &mut Vec<i64>) {
    let before = ids.len();
    tokenizer.encode_with_added_tokens_flat(doc, ids);
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

/// Whether LPT chunk sizing is enabled: killed by setting `GIGATOK_NO_LPT`
/// in the environment (to any value, empty included). Read once per encode
/// call — never in per-chunk or per-pretoken loops. Both shapes are token-
/// and order-identical; the switch changes chunk sizing only, and exists
/// so future measurement can flip it without a rebuild.
fn lpt_from_env() -> bool {
    std::env::var_os("GIGATOK_NO_LPT").is_none()
}

/// Group documents into parallel chunks of descending (LPT-scheduled)
/// sizes: ~2x-target chunks over the first ~80% of bytes, quarter-target
/// chunks over the last ~20%. Rayon hands out chunks in index order, so
/// whichever core draws the last chunk (on asymmetric parts, often an
/// E-core) strands the others behind a short tail instead of a full-size
/// one, while the big early chunks amortize per-chunk overhead. A document
/// larger than the target is split into consecutive Fragment chunks at
/// pretoken-safe boundaries that no added-token occurrence straddles, so
/// even a single huge document is encoded across all cores with
/// token-identical output.
///
/// With LPT disabled (GIGATOK_NO_LPT) this restores uniform sizing:
/// head_bytes = 0 makes every Docs group aim for `target`, and
/// frag_head = usize::MAX makes every fragment take the primary split
/// size `target` (the sub-split branch is never entered), which is
/// exactly the old safe_split_ranges(doc, target) loop. The oversize
/// threshold (`doc.len() > 2 * target`) is identical in both shapes.
fn build_doc_chunks<'a>(
    docs: &[&'a [u8]],
    total: usize,
    target: usize,
    added_tokens: &[&[u8]],
    lpt: bool,
) -> Vec<EncodeChunk<'a>> {
    let (head_bytes, group_big, frag_big, tail_target) = if lpt {
        (
            total - total / 5,
            2 * target,
            2 * target,
            (target / 4).max(MIN_CHUNK_BYTES),
        )
    } else {
        (0, target, target, target)
    };
    let mut chunks = Vec::new();
    let mut group: Vec<&[u8]> = Vec::new();
    // Bytes already assigned to chunks: positions below `head_bytes` take
    // the big target, the rest the small one.
    let mut emitted = 0usize;
    let mut acc = 0usize;
    for &doc in docs {
        if doc.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
                emitted += acc;
                acc = 0;
            }
            push_fragment_chunks(
                &mut chunks,
                doc,
                if lpt {
                    head_bytes.saturating_sub(emitted)
                } else {
                    usize::MAX
                },
                frag_big,
                tail_target,
                added_tokens,
            );
            emitted += doc.len();
            continue;
        }
        group.push(doc);
        acc += doc.len();
        let group_target = if emitted < head_bytes {
            group_big
        } else {
            tail_target
        };
        if acc >= group_target {
            chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
            emitted += acc;
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(EncodeChunk::Docs(group));
    }
    chunks
}

/// Split one oversized document into consecutive Fragment chunks with the
/// descending sizes of `build_doc_chunks`: `big`-sized fragments over the
/// first `head_len` bytes, `tail_target`-sized fragments after.
/// Sub-splitting a tail fragment preserves boundary safety: the pretoken
/// cut check is purely local (3 bytes around the cut), and an added-token
/// occurrence is orders of magnitude shorter than the >= MIN_CHUNK_BYTES
/// distance of any sub-cut from its fragment's (already safe) edges.
fn push_fragment_chunks<'a>(
    chunks: &mut Vec<EncodeChunk<'a>>,
    doc: &'a [u8],
    head_len: usize,
    big: usize,
    tail_target: usize,
    added_tokens: &[&[u8]],
) {
    let mut first = true;
    for r in crate::pretokenize::safe_split_ranges(doc, big, added_tokens) {
        if r.start < head_len || r.len() <= tail_target {
            chunks.push(EncodeChunk::Fragment {
                bytes: &doc[r],
                first: std::mem::take(&mut first),
            });
        } else {
            for sub in
                crate::pretokenize::safe_split_ranges(&doc[r.clone()], tail_target, added_tokens)
            {
                chunks.push(EncodeChunk::Fragment {
                    bytes: &doc[r.start + sub.start..r.start + sub.end],
                    first: std::mem::take(&mut first),
                });
            }
        }
    }
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

/// Concatenate per-chunk row lengths into per-document row counts, merging
/// `continues` fragments into the previous document's row.
fn row_counts(chunks: &[ChunkTokens]) -> Vec<i64> {
    let mut counts: Vec<i64> = Vec::new();
    for chunk in chunks {
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
    counts
}

/// Free spent chunk buffers off the caller's critical path. They total as
/// much memory as the gathered result, and tearing them down is munmap page
/// teardown under the address-space write lock: inline frees convoy the
/// gather copy's first-touch faults (read lock) behind each munmap (write
/// lock), and a serial free after the copy keeps the munmaps inside the
/// timed window. A detached background task pays the same teardown CPU
/// after the caller has returned, overlapped with whatever runs next, and
/// occupies at most one pool thread. Small results
/// (< `DEFERRED_DROP_MIN_BYTES` of tokens) just drop inline: their teardown
/// is microseconds and not worth holding the memory past return.
fn defer_drop(chunks: Vec<ChunkTokens>) {
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    if total * std::mem::size_of::<u32>() >= DEFERRED_DROP_MIN_BYTES {
        rayon::spawn(move || drop(chunks));
    }
}

/// The in-flight state of the overlapped gather: a flat id buffer reserved
/// at an upper bound BEFORE chunk sizes are known, plus a cursor over the
/// longest fully-encoded prefix of the chunk sequence.
///
/// Run as a separate phase after the encode, the final gather's first-touch
/// faults + memcpy are a serial tail, most of whose CPU is kernel
/// fault-path contention from every thread faulting the flat buffer at
/// once. Chunk completion is near-sequential (strict in-order handout,
/// descending LPT sizes), so a worker that finishes a chunk can commit the
/// ready prefix — offsets are exact, they are sums of *completed* chunk
/// sizes — while the tail still encodes. The copy work then hides inside
/// the encode phase at ~1 thread at a time (no fault convoy), leaving only
/// a small residual drain after the last chunk.
///
/// The upper bound: a token consumes at least one input byte, so
/// `total_bytes` tokens bounds the output; untouched reserved pages cost
/// address space only. Two escapes fall back to the collect-then-gather
/// path (`gather_flat`): the reservation itself failing (e.g. Linux
/// heuristic overcommit refusing a 4x-input VA block for a huge batch),
/// and the cursor overflowing the bound, which is impossible for plain
/// byte input and reachable only when NFC normalization expands bytes
/// (composition-exclusion pathologies) — `advance` stops committing and
/// `finish` returns None rather than write past the reservation.
struct Committer {
    /// Owns the reservation; the Vec struct itself is read or written only
    /// in `finish`. `UnsafeCell` so each `advance` derives the destination
    /// pointer fresh under the cursor lock: no pointer is captured across
    /// the struct's construction-time moves (a move retags the Vec's
    /// unique pointer under strict aliasing models).
    flat: UnsafeCell<Vec<u32>>,
    /// Reserved capacity in tokens; commits never write at or past it.
    cap: usize,
    cursor: Mutex<CommitCursor>,
}

struct CommitCursor {
    /// Index of the first uncommitted chunk.
    next: usize,
    /// Tokens committed so far == sum of ids.len() over chunks[..next].
    offset: usize,
    /// A chunk did not fit under `cap`; committing has stopped for good.
    overflowed: bool,
}

// SAFETY: the heap buffer behind `flat` is never reallocated while shared
// (nothing pushes to the Vec; it is resized only in `finish`, after all
// shared use has ended). During the shared phase the cell is used solely
// to derive the buffer pointer under the `cursor` lock, and every write
// through it lands in a disjoint, in-bounds range.
unsafe impl Send for Committer {}
unsafe impl Sync for Committer {}

impl Committer {
    /// Chunks committed per `advance` call. Bounds how long one worker is
    /// away from encoding (a backlog can pile up behind a long-held lock);
    /// anything left over is drained by later completions or `finish`.
    const MAX_DRAIN: usize = 8;

    /// Reserve `cap` tokens up front, or None (→ classic gather) if the
    /// allocator refuses.
    fn try_new(cap: usize) -> Option<Self> {
        let mut flat: Vec<u32> = Vec::new();
        if cap == 0 || flat.try_reserve_exact(cap).is_err() {
            return None;
        }
        // The commits fault this reservation in while the encode runs.
        madvise_hugepage(flat.as_mut_ptr() as *mut u8, cap * std::mem::size_of::<u32>());
        Some(Self {
            flat: UnsafeCell::new(flat),
            cap,
            cursor: Mutex::new(CommitCursor {
                next: 0,
                offset: 0,
                overflowed: false,
            }),
        })
    }

    /// Copy any freshly completed prefix chunks into the flat buffer.
    /// Non-blocking: if another worker is mid-commit, return to encoding —
    /// the current holder (or a later completion, or `finish`) picks the
    /// chunk up. A completion that lands between the holder's last check
    /// and its unlock is likewise deferred, never lost.
    fn advance(&self, outs: &[OnceLock<ChunkTokens>]) {
        let Ok(mut cur) = self.cursor.try_lock() else {
            return;
        };
        // SAFETY: the cursor lock is held; the Vec struct is not mutated
        // during the shared phase (see the `flat` field doc), so deriving
        // the buffer pointer only reads it.
        let base = unsafe { (*self.flat.get()).as_mut_ptr() };
        for _ in 0..Self::MAX_DRAIN {
            if cur.overflowed {
                return;
            }
            let Some(chunk) = outs.get(cur.next).and_then(OnceLock::get) else {
                return;
            };
            let len = chunk.ids.len();
            if self.cap - cur.offset < len {
                cur.overflowed = true;
                return;
            }
            // SAFETY: holding `cursor`, writing [offset, offset+len), which
            // is within the reservation (checked above) and disjoint from
            // every earlier commit (offset is monotone).
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.ids.as_ptr(), base.add(cur.offset), len);
            }
            cur.offset += len;
            cur.next += 1;
        }
    }

    /// After all chunks are encoded (and the scope joined): copy the
    /// uncommitted suffix in parallel, size the buffer to `total` and trim
    /// the reservation. None means the bound was overrun (see type docs) —
    /// caller falls back to the classic gather; the prefix copied so far is
    /// discarded (chunk buffers are still intact).
    fn finish(self, chunks: &[ChunkTokens], total: usize) -> Option<Vec<u32>> {
        use rayon::prelude::*;
        /// Raw destination pointer, shareable across the copy tasks. (The
        /// accessor keeps closure capture at the wrapper, not the field.)
        struct SyncPtr(*mut u32);
        // SAFETY: only used for the disjoint in-bounds writes below.
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}
        impl SyncPtr {
            /// SAFETY: `off` must be within the reservation.
            unsafe fn at(&self, off: usize) -> *mut u32 {
                unsafe { self.0.add(off) }
            }
        }

        let Committer { flat, cap, cursor } = self;
        let mut flat = flat.into_inner();
        let cur = cursor
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cur.overflowed || total > cap {
            return None;
        }
        let rest = &chunks[cur.next..];
        let mut offsets = Vec::with_capacity(rest.len());
        let mut offset = cur.offset;
        for chunk in rest {
            offsets.push(offset);
            offset += chunk.ids.len();
        }
        debug_assert_eq!(offset, total);
        // Derived AFTER the Vec moved out of the cell: a move retags the
        // Vec's unique pointer under strict aliasing models, so the suffix
        // writes and the `set_len` below go through a post-move pointer.
        let base = SyncPtr(flat.as_mut_ptr());
        // `with_max_len(1)` keeps the multi-MB copies stealable one by one.
        rest.par_iter()
            .zip(offsets)
            .with_max_len(1)
            .for_each(|(chunk, off)| {
                // SAFETY: exclusive access (workers are joined); suffix
                // ranges are disjoint from each other and from the
                // committed prefix, and end at total <= cap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk.ids.as_ptr(),
                        base.at(off),
                        chunk.ids.len(),
                    );
                }
            });
        // SAFETY: capacity >= cap >= total, and [0, total) was fully
        // initialized by the prefix commits plus the suffix copies above.
        unsafe {
            flat.set_len(total);
        }
        // Return the unused reservation. Large allocations trim in place on
        // the mainstream allocators (macOS libmalloc large entries, glibc
        // mmap'd chunks via mremap): pointer-stable, no copy, and the
        // untouched tail pages were never faulted so there is nothing to
        // tear down. An allocator that copies instead only costs one
        // memcpy; correctness is unaffected.
        flat.shrink_to_fit();
        Some(flat)
    }
}

/// Encode all chunks with pooled workers and gather them into one flat id
/// buffer plus per-document row counts — in parallel when there is more
/// than one chunk, serially otherwise. Each worker's caches are pre-sized
/// for its share of `total_bytes` (capacity hints only — see
/// `Tokenizer::fork_sized`; workers already forked on an earlier call keep
/// their warm caches).
///
/// Chunks are handed out in strict index order through an atomic counter
/// (one pulling task per rayon thread), not `par_iter`: recursive range
/// splitting lets a thread steal a subrange of early big chunks and still
/// be *starting* a 2×-target chunk after everyone else has reached the
/// small tail, stranding the rest of the pool behind that one straggler.
/// In-order handout makes the LPT descending-size order of
/// `build_doc_chunks` a guarantee, bounding the tail at roughly one small
/// chunk — and makes chunk completion near-sequential, which is what lets
/// the gather copy overlap the encode (see `Committer`).
pub(crate) fn encode_chunks_gathered(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
) -> (Vec<u32>, Vec<i64>) {
    // A token consumes >= 1 input byte, so total_bytes tokens is the
    // reservation bound (NFC expansion is caught by the overflow escape).
    encode_chunks_gathered_with_cap(workers, proto, chunks, total_bytes, total_bytes)
}

/// `encode_chunks_gathered` with the committer's reservation bound passed
/// explicitly, so tests can force the overflow fallback in-process.
fn encode_chunks_gathered_with_cap(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
    cap_tokens: usize,
) -> (Vec<u32>, Vec<i64>) {
    let share = total_bytes / rayon::current_num_threads().max(1);
    let encode = |c: &EncodeChunk| workers.with_worker(proto, share, |tok| encode_chunk(tok, c));
    if chunks.len() <= 1 {
        // Small inputs skip the thread fan-out — and a lone chunk's id
        // buffer IS the flat result, no gather copy at all.
        return match chunks.first() {
            Some(chunk) => {
                let out = encode(chunk);
                let counts = row_counts(std::slice::from_ref(&out));
                (out.ids, counts)
            }
            None => (Vec::new(), Vec::new()),
        };
    }
    let next = AtomicUsize::new(0);
    let outs: Vec<OnceLock<ChunkTokens>> = (0..chunks.len()).map(|_| OnceLock::new()).collect();
    let committer = Committer::try_new(cap_tokens);
    let tasks = rayon::current_num_threads().min(chunks.len());
    rayon::scope(|s| {
        for _ in 0..tasks {
            s.spawn(|_| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else {
                        // One last opportunistic drain on the way out: this
                        // worker's final chunk may have been skipped while
                        // another held the commit lock.
                        if let Some(c) = &committer {
                            c.advance(&outs);
                        }
                        break;
                    };
                    // Each index is claimed exactly once, so `set` cannot
                    // already be filled.
                    let _ = outs[i].set(encode(chunk));
                    if let Some(c) = &committer {
                        c.advance(&outs);
                    }
                }
            });
        }
    });
    let outs: Vec<ChunkTokens> = outs
        .into_iter()
        .map(|slot| slot.into_inner().expect("every claimed chunk was encoded"))
        .collect();
    let counts = row_counts(&outs);
    let total: usize = outs.iter().map(|c| c.ids.len()).sum();
    match committer.and_then(|c| c.finish(&outs, total)) {
        Some(flat) => {
            // The copies are done; the spent chunk buffers are dead weight.
            defer_drop(outs);
            (flat, counts)
        }
        None => (gather_flat(outs), counts),
    }
}

/// Merge per-chunk outputs into one flat id buffer and per-document row
/// counts. The flat gather copies chunk buffers in parallel into a single
/// allocation. (The SentencePiece paths gather this way; the BPE batch
/// path overlaps the copy with the encode — see `Committer` — and falls
/// back to this only when the up-front reservation is refused or overrun.)
pub(crate) fn assemble_ragged(chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<i64>) {
    let counts = row_counts(&chunks);
    (gather_flat(chunks), counts)
}

/// Copy all chunk id buffers into one freshly allocated flat buffer, in
/// parallel, freeing the spent chunk buffers off the critical path.
fn gather_flat(chunks: Vec<ChunkTokens>) -> Vec<u32> {
    use rayon::prelude::*;
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    let mut flat = vec![0u32; total];
    // The parallel copy below faults in the whole buffer.
    madvise_hugepage(flat.as_mut_ptr() as *mut u8, total * std::mem::size_of::<u32>());
    let mut rest: &mut [u32] = &mut flat;
    let mut slices = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let (head, tail) = rest.split_at_mut(chunk.ids.len());
        slices.push(head);
        rest = tail;
    }
    // `with_max_len(1)` keeps the multi-MB copies stealable one by one.
    slices
        .into_par_iter()
        .zip(chunks.par_iter())
        .with_max_len(1)
        .for_each(|(dst, chunk)| dst.copy_from_slice(&chunk.ids));
    defer_drop(chunks);
    flat
}

/// Persistent pool of forked tokenizer workers used by encode_batch and
/// encode_files. One slot per rayon thread, forked lazily on first use and
/// retained for the tokenizer's lifetime, so each worker's pretoken cache
/// stays warm when encoding is invoked repeatedly (e.g. in a loop).
///
/// Invariant: the prototype tokenizer must not be mutated between encodes
/// that share a pool. Workers are forked lazily (first use of each slot)
/// and never refreshed, so mutating the prototype (`add_special_token`,
/// `set_added_tokens`, `set_pretokenizer_type`, ...) after some slots have
/// forked leaves those workers on the OLD state while slots forked later
/// (or rebuilt after a worker panic) capture the new one — a chunk's
/// tokens would then depend on which slot the handout gave it. Finish all
/// model mutation before the pool's first encode (the loaders do), or use
/// a fresh pool after mutating. The Python bindings uphold this by
/// construction: no mutator is exposed after the pyclass is built.
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

    /// Run `f` with exclusive access to a pooled worker, forking one sized
    /// for `expected_bytes` of input if the slot is empty. Rayon never runs
    /// more tasks concurrently than it has threads, and there is one slot
    /// per thread, so a free slot always exists; the yield loop only spins
    /// when non-rayon threads encode at the same time.
    ///
    /// `proto` must be the same, unmutated prototype on every call for a
    /// given pool: forks are cached per slot and never compared against
    /// `proto` again, so a mutated (or different) prototype yields stale
    /// workers for already-filled slots (see the type-level invariant).
    fn with_worker<R>(
        &self,
        proto: &Tokenizer,
        expected_bytes: usize,
        f: impl FnOnce(&mut Tokenizer) -> R,
    ) -> R {
        let slots = self.slots.get_or_init(|| {
            (0..rayon::current_num_threads())
                .map(|_| Mutex::new(None))
                .collect()
        });
        loop {
            for slot in slots {
                match slot.try_lock() {
                    Ok(mut guard) => {
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        // A worker panicked mid-encode; its cache may be
                        // inconsistent, so rebuild it from the prototype.
                        let mut guard = poisoned.into_inner();
                        *guard = None;
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
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
///
/// Environment: setting `GIGATOK_NO_LPT` (to any value, empty included)
/// disables LPT chunk sizing in favor of uniform chunks — token- and
/// order-identical output, chunk shaping only; see `lpt_from_env`. The
/// variable is read once per call, never in per-chunk loops.
pub fn encode_docs_ragged(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
) -> (Vec<u32>, Vec<i64>) {
    encode_docs_ragged_with(workers, proto, docs, lpt_from_env())
}

/// `encode_docs_ragged` with the LPT switch passed explicitly instead of
/// read from the environment, so tests can cover both shapes in-process
/// without mutating process env.
pub(crate) fn encode_docs_ragged_with(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
    lpt: bool,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let added = proto.added_token_contents();
    let chunks = build_doc_chunks(docs, total, chunk_target_bytes(total), &added, lpt);
    encode_chunks_gathered(workers, proto, &chunks, total)
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
    encode_chunks_gathered(workers, proto, &chunks, total)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The parallel chunked path — LPT descending chunk sizes, pooled
    /// pre-sized workers, overlapped Committer gather (with
    /// collect-then-gather as the fallback) — must be
    /// token-identical, in the same order, to a serial per-document
    /// encode, with LPT both on and off (GIGATOK_NO_LPT), passed
    /// explicitly so no process env is mutated. A byte-level vocab makes
    /// any misordered or dropped chunk visible in the flat buffer, not
    /// just in the counts.
    #[test]
    fn parallel_ragged_matches_serial() {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|b| vec![b]).collect();
        let proto = Tokenizer::new(merges, vocab, None);

        // Deterministic pseudo-text with plenty of alnum-space-alpha cut
        // points for safe_split_ranges.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut text = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (state >> 33) as usize;
                    b"abcdefghijklmnopqrstuvwxyz0123456789    "[r % 40]
                })
                .collect()
        };
        // Mid-size docs that group into Docs chunks, one oversized doc
        // that splits into Fragment chunks (spanning the head/tail
        // boundary, so both fragment sizes appear), then small docs so
        // continuation rows land mid-output.
        let mut owned: Vec<Vec<u8>> = Vec::new();
        for _ in 0..30 {
            owned.push(text(300 << 10));
        }
        owned.push(text(12 << 20));
        for _ in 0..30 {
            owned.push(text(100 << 10));
        }
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }

        for lpt in [true, false] {
            // A fresh pool per shape so each run exercises the pre-sized
            // fork (slots fork lazily on first use).
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            assert_eq!(flat, ids_ref, "ids mismatch (lpt={lpt})");
        }
    }

    /// Parallel-vs-serial at scale on a REAL tokenizer: ~1 GB of OWT as a
    /// multi-doc group (small grouped docs, mid docs, oversized docs that
    /// fragment at pretoken-safe boundaries) with `<|endoftext|>` injected
    /// mid-doc and doc-final, LPT on and off. Token AND order identity
    /// against a serial per-document encode. A few seconds in release mode
    /// (both sides use the cached encode).
    /// `cargo test --release verify_parallel_ragged_matches_serial_owt_gpt2_1g -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 1 GB of OWT; run explicitly in release mode"]
    fn verify_parallel_ragged_matches_serial_owt_gpt2_1g() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        use std::io::Read;
        let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data/gpt2_tokenizer.json");
        let proto = load_hf_bpe(&tokenizer_path).expect("load GPT-2 tokenizer");
        let added = proto.added_token_contents();
        let sep: Vec<u8> = added.first().expect("GPT-2 has an added token").to_vec();

        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("open ~/data/owt_train.txt");
        let mut input = Vec::new();
        f.take(1_000_000_000).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        assert!(input.len() > 900_000_000, "corpus too small: {}", input.len());

        // Doc size pattern: small (group), mid, large; every 20th doc is
        // oversized (24 MB) so it splits into Fragment chunks.
        let sizes = [64 << 10, 300 << 10, 1 << 20, 100 << 10, 3 << 20];
        let mut owned: Vec<Vec<u8>> = Vec::new(); // docs with injected added tokens
        let mut ranges: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, inject)
        let mut pos = 0usize;
        let mut i = 0usize;
        while pos < input.len() {
            let want = if i % 20 == 19 { 24 << 20 } else { sizes[i % sizes.len()] };
            let end = (pos + want).min(input.len());
            // Inject the added token into every 7th doc (mid + tail).
            ranges.push((pos, end, i % 7 == 3));
            pos = end;
            i += 1;
        }
        for &(s, e, inject) in &ranges {
            if inject {
                let piece = &input[s..e];
                let mid = piece.len() / 2;
                let mut doc = Vec::with_capacity(piece.len() + 2 * sep.len());
                doc.extend_from_slice(&piece[..mid]);
                doc.extend_from_slice(&sep);
                doc.extend_from_slice(&piece[mid..]);
                doc.extend_from_slice(&sep);
                owned.push(doc);
            }
        }
        let mut docs: Vec<&[u8]> = Vec::with_capacity(ranges.len());
        let mut oi = 0usize;
        for &(s, e, inject) in &ranges {
            if inject {
                docs.push(&owned[oi]);
                oi += 1;
            } else {
                docs.push(&input[s..e]);
            }
        }
        eprintln!(
            "{} docs ({} with injected {:?}), {} bytes total",
            docs.len(),
            owned.len(),
            String::from_utf8_lossy(&sep),
            docs.iter().map(|d| d.len()).sum::<usize>()
        );

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }
        drop(serial);
        eprintln!("serial reference: {} tokens", ids_ref.len());

        for lpt in [true, false] {
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            if flat != ids_ref {
                let i = ids_ref
                    .iter()
                    .zip(&flat)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| ids_ref.len().min(flat.len()));
                panic!(
                    "ids mismatch (lpt={lpt}) at token {i}: serial[{i}..] = {:?}, parallel[{i}..] = {:?}",
                    &ids_ref[i..(i + 8).min(ids_ref.len())],
                    &flat[i..(i + 8).min(flat.len())],
                );
            }
            eprintln!("lpt={lpt}: {} tokens identical", flat.len());
        }
    }

    /// The overlapped gather's escape hatches must be output-identical to
    /// the committed path: cap 0 stands in for a refused up-front
    /// reservation (no committer at all), cap 1 overflows on the first
    /// commit, and a mid-range cap overflows mid-flight after a real
    /// prefix has been committed — the fallback must discard that prefix
    /// and re-gather from the (intact) chunk buffers.
    #[test]
    fn gather_fallbacks_match() {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|b| vec![b]).collect();
        let proto = Tokenizer::new(merges, vocab, None);

        let mut state = 0xD1B54A32D192ED03u64;
        let mut text = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (state >> 33) as usize;
                    b"abcdefghijklmnopqrstuvwxyz0123456789    "[r % 40]
                })
                .collect()
        };
        let owned: Vec<Vec<u8>> = (0..20).map(|_| text(1 << 20)).collect();
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let added = proto.added_token_contents();
        let chunks = build_doc_chunks(&docs, total, chunk_target_bytes(total), &added, true);
        assert!(chunks.len() > 1, "test must exercise the parallel path");

        let workers = WorkerPool::new();
        let (flat_ref, lens_ref) = encode_chunks_gathered(&workers, &proto, &chunks, total);
        // Byte-level vocab: one token per byte, so any cap below `total`
        // overflows; total / 3 overflows mid-flight with a committed
        // prefix behind it.
        for cap in [0, 1, total / 3] {
            let workers = WorkerPool::new();
            let (flat, lens) =
                encode_chunks_gathered_with_cap(&workers, &proto, &chunks, total, cap);
            assert_eq!(lens, lens_ref, "lens mismatch (cap={cap})");
            assert_eq!(flat, flat_ref, "ids mismatch (cap={cap})");
        }
    }
}
