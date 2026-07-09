#![feature(test)]
#![feature(portable_simd)]

pub(crate) mod bpe;
pub(crate) mod bpe_train;
pub(crate) mod encode;
pub(crate) mod input;
pub mod pretokenize;
pub(crate) mod simd;
pub(crate) mod token;
pub(crate) mod unicode_tables;
pub mod utils;
pub use crate::bpe::Tokenizer;
use crate::input::file_source::{
    DocFormat, FileSourceSpec, LoadedFile, chunk_ranges, detect_default_format, load_file,
};
use crate::input::{DocumentIter, MmappedFile, Resource};
pub mod load_tokenizer;
use itertools::Itertools;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, TryLockError};

// ---------------------------------------------------------------------------
// Helper: convert BPEResult to Python objects
// ---------------------------------------------------------------------------

/// Vocab dict plus ordered merge pairs, as Python objects.
type PyVocabAndMerges<'py> = (
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
);

fn bpe_result_to_python<'py>(
    py: Python<'py>,
    result: bpe_train::BPEResult,
) -> PyResult<PyVocabAndMerges<'py>> {
    let vocab_py = result
        .vocab
        .into_iter()
        .map(|(k, v)| (k, PyBytes::new(py, &v)))
        .sorted_by(|e1, e2| Ord::cmp(&e1.0, &e2.0))
        .into_py_dict(py);
    let merges_py: Vec<_> = result
        .merges
        .into_iter()
        .map(|(k, v)| (PyBytes::new(py, &k), PyBytes::new(py, &v)))
        .collect();
    Ok((vocab_py?, merges_py))
}

fn parse_tie_breaking(s: &str) -> PyResult<bpe_train::TieBreaking> {
    match s {
        "huggingface" => Ok(bpe_train::TieBreaking::HuggingFace),
        "raw_token_ids" => Ok(bpe_train::TieBreaking::RawTokenIds),
        "assembled_bytes" => Ok(bpe_train::TieBreaking::AssembledBytes),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "tie_breaking must be 'huggingface', 'raw_token_ids', or 'assembled_bytes', got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// FileSource Python classes
// ---------------------------------------------------------------------------

/// Base class for file sources. Not directly constructible from Python —
/// use `TextFileSource` or `JsonlFileSource`, which pin down the document
/// format and its parameters. Compression (.gz/.zst) is always detected
/// from the file extension, independent of the source type.
#[pyclass(subclass, from_py_object)]
#[derive(Clone)]
struct FileSource {
    paths: Vec<PathBuf>,
    format: DocFormat,
}

#[pymethods]
impl FileSource {
    fn __repr__(&self) -> String {
        let n = self.paths.len();
        match &self.format {
            DocFormat::Jsonl { field } => {
                format!("JsonlFileSource(paths=[{n} files], field={field:?})")
            }
            DocFormat::Text {
                separator: Some(sep),
            } => format!(
                "TextFileSource(paths=[{n} files], separator={:?})",
                String::from_utf8_lossy(sep)
            ),
            DocFormat::Text { separator: None } => {
                format!("TextFileSource(paths=[{n} files])")
            }
        }
    }
}

/// Plain-text files. With `separator`, documents are the pieces between
/// separator occurrences (the separator itself belongs to no document);
/// without one, each file is a single document.
#[pyclass(extends = FileSource)]
struct TextFileSource;

#[pymethods]
impl TextFileSource {
    #[new]
    #[pyo3(signature = (paths, separator = None))]
    fn new(paths: Vec<PathBuf>, separator: Option<Vec<u8>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(FileSource {
            paths,
            format: DocFormat::Text { separator },
        })
        .add_subclass(Self)
    }
}

/// JSON Lines files: one document per line, text taken from `field`.
#[pyclass(extends = FileSource)]
struct JsonlFileSource;

#[pymethods]
impl JsonlFileSource {
    #[new]
    #[pyo3(signature = (paths, field = "text"))]
    fn new(paths: Vec<PathBuf>, field: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(FileSource {
            paths,
            format: DocFormat::Jsonl {
                field: field.to_string(),
            },
        })
        .add_subclass(Self)
    }
}

// ---------------------------------------------------------------------------
// train_bpe Python function
// ---------------------------------------------------------------------------

#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (in_data, vocab_size, special_tokens, tie_breaking = "huggingface", separator = None))]
fn train_bpe<'py>(
    py: Python<'py>,
    in_data: Bound<'py, PyAny>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: &str,
    separator: Option<&[u8]>,
) -> PyResult<(
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
)> {
    assert!(
        vocab_size <= 2_usize.pow(32),
        "vocab_size must be less than 2^32"
    );
    let tie_breaking = parse_tie_breaking(tie_breaking)?;
    let separator = separator.unwrap_or(pretokenize::DEFAULT_SEPARATOR);

    // --- FileSource: multi-file parallel processing ---
    if let Ok(file_source) = in_data.extract::<FileSource>() {
        let spec = FileSourceSpec {
            paths: file_source.paths,
            format: file_source.format,
        };
        let counts = spec.pretokenize().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "FileSource processing failed: {}",
                e
            ))
        })?;
        let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
        return bpe_result_to_python(py, result);
    }

    // --- Single bytes or file path ---
    let mmap_resource;
    let bytes: &[u8] = if in_data.is_instance_of::<PyBytes>() {
        in_data.extract::<&[u8]>()?
    } else if let Ok(path) = in_data.extract::<PathBuf>() {
        if let Some(ext) = path.extension()
            && ext == "parquet"
        {
            #[cfg(not(feature = "parquet"))]
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "The 'parquet' feature is not enabled in this build, cannot read parquet files",
            ));
            #[cfg(feature = "parquet")]
            {
                let counts = pretokenize::pretokenize_par_parquet(&path);
                let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
                return bpe_result_to_python(py, result);
            }
        }
        mmap_resource = MmappedFile::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open file {:?}: {}",
                path, e
            ))
        })?;
        mmap_resource.as_bytes()
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "in_data must be bytes, a path, or a FileSource",
        ));
    };

    let counts = pretokenize::pretokenize_par_bytes(bytes, separator);
    let result = bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
    bpe_result_to_python(py, result)
}

// ---------------------------------------------------------------------------
// Other Python classes and functions
// ---------------------------------------------------------------------------

/// A document to encode: str (UTF-8 text) or bytes. Both variants borrow
/// the Python object's buffer without copying and are usable with the GIL
/// released. Paths are deliberately not accepted here — encoding from files
/// goes through `encode_files`, which mmaps and chunks them.
enum EncodeInput {
    Text(PyBackedStr),
    Bytes(PyBackedBytes),
}

impl EncodeInput {
    fn as_bytes(&self) -> &[u8] {
        match self {
            EncodeInput::Text(s) => s.as_bytes(),
            EncodeInput::Bytes(b) => b,
        }
    }
}

/// Shared front-end of encode_batch: extract the documents (a list of str, a
/// list of bytes, or an awkward Array of strings — whose flat buffer is used
/// directly, with no per-document Python objects), run `encode` on the
/// resolved byte slices with the GIL released, and hand the ragged result to
/// Python as an awkward.Array.
fn encode_batch_ragged<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    encode: impl Fn(&[&[u8]]) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    // Awkward input: encode straight from the flat content buffer.
    if let Some((content, in_counts)) = extract_awkward_docs(inputs)? {
        let content = content.readonly();
        let bytes: &[u8] = content.as_slice()?;
        let (flat, counts) = py.detach(|| -> PyResult<_> {
            let mut docs = Vec::with_capacity(in_counts.len());
            let mut pos = 0usize;
            for &n in &in_counts {
                docs.push(&bytes[pos..pos + n as usize]);
                pos += n as usize;
            }
            encode(&docs)
        })?;
        return ragged_to_python(py, flat, counts);
    }

    let inputs: Vec<Bound<'py, PyAny>> = inputs.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected a list of str, a list of bytes, or an awkward Array of strings",
        )
    })?;
    if inputs.is_empty() {
        return ragged_to_python(py, vec![], vec![]);
    }
    let mut docs = Vec::with_capacity(inputs.len());
    docs.push(extract_doc(&inputs[0])?);
    for obj in &inputs[1..] {
        let doc = extract_doc(obj)?;
        if std::mem::discriminant(&doc) != std::mem::discriminant(&docs[0]) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "all documents in a batch must be of the same type \
                 (a list of str or a list of bytes)",
            ));
        }
        docs.push(doc);
    }

    let (flat, counts) = py.detach(|| {
        let slices: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
        encode(&slices)
    })?;
    ragged_to_python(py, flat, counts)
}

/// Extract one document, pointing path-holders at encode_files.
fn extract_doc(obj: &Bound<'_, PyAny>) -> PyResult<EncodeInput> {
    if let Ok(s) = obj.extract::<PyBackedStr>() {
        return Ok(EncodeInput::Text(s));
    }
    if let Ok(b) = obj.extract::<PyBackedBytes>() {
        return Ok(EncodeInput::Bytes(b));
    }
    let hint = if obj.extract::<PathBuf>().is_ok() {
        "; to encode files, use encode_files"
    } else {
        ""
    };
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "expected str or bytes, got {}{hint}",
        obj.get_type()
    )))
}

/// Append one document's token ids to `ids` and its row length to `lens`.
fn encode_into(tokenizer: &mut Tokenizer, doc: &[u8], ids: &mut Vec<u32>, lens: &mut Vec<i64>) {
    let before = ids.len();
    tokenizer.encode_with_added_tokens(doc, |tokens| {
        for &e in tokens {
            ids.push(e.into())
        }
    });
    lens.push((ids.len() - before) as i64);
}

/// Work unit for parallel encoding.
enum EncodeChunk<'a> {
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
struct ChunkTokens {
    ids: Vec<u32>,
    lens: Vec<i64>,
    continues: bool,
}

fn encode_chunk(tokenizer: &mut Tokenizer, chunk: &EncodeChunk) -> ChunkTokens {
    use crate::input::jsonl::JsonLinesSlice;
    let mut ids = Vec::new();
    let mut lens = Vec::new();
    let mut continues = false;
    match chunk {
        EncodeChunk::Docs(docs) => {
            for doc in docs {
                encode_into(tokenizer, doc, &mut ids, &mut lens);
            }
        }
        EncodeChunk::Region { bytes, format } => match format {
            DocFormat::Jsonl { field } => {
                for doc in JsonLinesSlice::new(bytes, field) {
                    encode_into(tokenizer, doc.as_ref(), &mut ids, &mut lens);
                }
            }
            DocFormat::Text {
                separator: Some(sep),
            } if !sep.is_empty() => {
                for doc in DocumentIter::new(bytes, sep) {
                    encode_into(tokenizer, doc, &mut ids, &mut lens);
                }
            }
            DocFormat::Text { .. } => encode_into(tokenizer, bytes, &mut ids, &mut lens),
        },
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
            for (k, r) in pretokenize::safe_split_ranges(doc, target, added_tokens)
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

/// Encode all chunks with pooled workers — in parallel when there is more
/// than one chunk, serially otherwise (small inputs skip the thread fan-out).
fn encode_chunks_pooled(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
) -> Vec<ChunkTokens> {
    use rayon::prelude::*;
    if chunks.len() <= 1 {
        chunks
            .iter()
            .map(|c| workers.with_worker(proto, |tok| encode_chunk(tok, c)))
            .collect()
    } else {
        chunks
            .par_iter()
            .map(|c| workers.with_worker(proto, |tok| encode_chunk(tok, c)))
            .collect()
    }
}

/// Merge per-chunk outputs into one flat id buffer and per-document row
/// counts. The flat gather copies chunk buffers in parallel into a single
/// allocation.
fn assemble_ragged(chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<i64>) {
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

/// If `inputs` is an awkward Array of strings or bytestrings, pull out its
/// flat uint8 content and per-document counts directly — no per-document
/// Python objects are materialized. Returns None when `inputs` is not an
/// awkward Array (or awkward is not importable).
/// Flat uint8 content array plus per-document byte counts.
type FlatDocs<'py> = (Bound<'py, numpy::PyArray1<u8>>, Vec<i64>);

fn extract_awkward_docs<'py>(inputs: &Bound<'py, PyAny>) -> PyResult<Option<FlatDocs<'py>>> {
    let py = inputs.py();
    let Ok(ak) = py.import("awkward") else {
        return Ok(None);
    };
    if !inputs.is_instance(&ak.getattr("Array")?)? {
        return Ok(None);
    }
    let type_err = |_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "awkward input must be an array of strings or bytestrings",
        )
    };
    // Stripping the string/bytestring parameters turns the array into plain
    // lists of uint8, whose flattened content and row lengths are views of
    // the existing buffers.
    let raw = ak
        .call_method1("without_parameters", (inputs,))
        .map_err(type_err)?;
    let flat = ak.call_method1("flatten", (&raw,)).map_err(type_err)?;
    let content = ak
        .call_method1("to_numpy", (flat,))?
        .cast_into::<PyArray1<u8>>()
        .map_err(|e| type_err(e.into()))?;
    let counts = ak
        .call_method1("to_numpy", (ak.call_method1("num", (&raw,))?,))?
        .cast_into::<PyArray1<i64>>()
        .map_err(|e| type_err(e.into()))?;
    let counts = counts.readonly().as_slice()?.to_vec();
    Ok(Some((content, counts)))
}

/// Hand a ragged token batch to Python as an `awkward.Array`: one flat
/// contents array plus per-document counts — two allocations total instead
/// of one numpy array per document. Falls back to a list of zero-copy numpy
/// views when awkward is not importable.
fn ragged_to_python<'py>(
    py: Python<'py>,
    flat: Vec<u32>,
    counts: Vec<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let n_rows = counts.len();
    let content = flat.into_pyarray(py);
    let counts = counts.into_pyarray(py);
    match py.import("awkward") {
        Ok(ak) => ak.call_method1("unflatten", (content, counts)),
        Err(_) => {
            if n_rows == 0 {
                return Ok(pyo3::types::PyList::empty(py).into_any());
            }
            let np = py.import("numpy")?;
            let bounds = np.call_method1("cumsum", (&counts,))?;
            let split_at = bounds.get_item(pyo3::types::PySlice::new(py, 0, -1, 1))?;
            np.call_method1("split", (content, split_at))
        }
    }
}

/// Parallel chunks must hold at least this many bytes: a chunk this size
/// encodes for tens of milliseconds, so worker acquisition and rayon
/// scheduling/work-stealing overhead is noise. An input that does not fill
/// more than one chunk is encoded serially — for small inputs the thread
/// fan-out costs more than it saves.
const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Target bytes per parallel chunk: ~4 chunks per thread for work-stealing
/// load balancing, floored at MIN_CHUNK_BYTES so chunks stay coarse.
fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (16 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
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

/// Resolve an encode_files argument: a FileSource (TextFileSource /
/// JsonlFileSource), a single path, or a list of paths. Bare paths get a
/// default format from the first path's extension — all inputs in a batch
/// are assumed to be of the same type.
fn resolve_files_source(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<PathBuf>, DocFormat)> {
    if let Ok(fs) = obj.extract::<FileSource>() {
        return Ok((fs.paths, fs.format));
    }
    if let Ok(path) = obj.extract::<PathBuf>() {
        let format = detect_default_format(&path);
        return Ok((vec![path], format));
    }
    if let Ok(paths) = obj.extract::<Vec<PathBuf>>() {
        let format = paths
            .first()
            .map(|p| detect_default_format(p))
            .unwrap_or(DocFormat::Text { separator: None });
        return Ok((paths, format));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "expected a TextFileSource/JsonlFileSource, a path, or a list of paths, got {}",
        obj.get_type()
    )))
}

/// Load all files in parallel: mmap when stored uncompressed, decompress
/// .gz/.zst into memory otherwise (parallel chunking needs random access).
fn load_files(paths: &[PathBuf]) -> PyResult<Vec<LoadedFile>> {
    use rayon::prelude::*;
    paths
        .par_iter()
        .map(|p| {
            load_file(p).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}: {e}", p.display()))
            })
        })
        .collect()
}

#[pyclass]
struct BPETokenizer {
    tokenizer: Tokenizer,
    workers: WorkerPool,
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

impl BPETokenizer {
    /// See `encode_docs_ragged`. Call with the GIL released.
    fn encode_slices_ragged(&self, docs: &[&[u8]]) -> (Vec<u32>, Vec<i64>) {
        encode_docs_ragged(&self.workers, &self.tokenizer, docs)
    }
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn __new__() -> PyResult<Self> {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        Ok(Self {
            tokenizer: load_tokenizer::tiktoken::load_tiktoken(tiktoken_path)?,
            workers: WorkerPool::new(),
        })
    }
    #[staticmethod]
    fn from_tiktoken(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::tiktoken::load_tiktoken(&path)?,
            workers: WorkerPool::new(),
        })
    }
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_bpe(&path)?,
            workers: WorkerPool::new(),
        })
    }

    /// Encode a single document (str or bytes) with the main tokenizer,
    /// whose pretoken cache persists across calls.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let input = extract_doc(&input)?;
        let (mut ids, mut lens) = (Vec::new(), Vec::new());
        encode_into(&mut self.tokenizer, input.as_bytes(), &mut ids, &mut lens);
        Ok(ids.into_pyarray(py))
    }

    /// Encode a batch of documents in parallel with rayon, releasing the GIL.
    /// Takes a list of str or a list of bytes (all elements of the same
    /// type), or an awkward Array of strings/bytestrings — whose flat
    /// buffers are used directly, with no per-document Python objects. For
    /// files, use encode_files. Returns an awkward.Array with one row of
    /// token ids per document (a single flat buffer plus offsets, not one
    /// numpy array per document).
    ///
    /// Documents are grouped into chunks of at least MIN_CHUNK_BYTES (small
    /// batches are encoded serially), and a document larger than a chunk is
    /// split at pretoken-safe boundaries and reassembled with identical
    /// tokens — a single huge document still uses all cores. Chunks are
    /// encoded by pooled workers whose pretoken caches persist across calls.
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs| Ok(self.encode_slices_ragged(docs)))
    }

    /// Encode all documents from files in parallel, releasing the GIL.
    /// Returns an awkward.Array with one row of token ids per document.
    ///
    /// `source` is a TextFileSource / JsonlFileSource, a single path, or a
    /// list of paths (defaults per extension: .jsonl → JSONL with field
    /// "text", anything else → plain text with each file as one document).
    /// Everything happens in Rust: files are mmapped (or decompressed into
    /// memory for .gz/.zst) and cut into chunks at document boundaries; a
    /// file that is one huge document is split at pretoken-safe boundaries
    /// and reassembled with identical tokens, so it still uses all cores.
    /// Chunks are encoded by pooled workers whose pretoken caches persist
    /// across calls.
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (paths, format) = resolve_files_source(&source)?;
        let (flat, counts) = py.detach(|| -> PyResult<_> {
            let files = load_files(&paths)?;
            // One document per file: group small files, split huge ones at
            // pretoken-safe boundaries.
            if matches!(&format, DocFormat::Text { separator: None }) {
                let docs: Vec<&[u8]> = files.iter().map(|f| f.as_bytes()).collect();
                return Ok(self.encode_slices_ragged(&docs));
            }
            // Many documents per file: cut byte regions at document
            // boundaries, documents are extracted while encoding.
            let total: usize = files.iter().map(|f| f.as_bytes().len()).sum();
            let target = chunk_target_bytes(total);
            let chunks: Vec<EncodeChunk> = files
                .iter()
                .flat_map(|f| {
                    let bytes = f.as_bytes();
                    chunk_ranges(bytes, &format, target)
                        .into_iter()
                        .map(|r| EncodeChunk::Region {
                            bytes: &bytes[r],
                            format: &format,
                        })
                })
                .collect();
            let outs = encode_chunks_pooled(&self.workers, &self.tokenizer, &chunks);
            Ok(assemble_ragged(outs))
        })?;
        ragged_to_python(py, flat, counts)
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let token_ids: Vec<crate::token::TokenId> = if let Ok(arr) = tokens.cast::<PyArray1<u32>>()
        {
            let arr = arr.readonly();
            arr.as_slice()?.iter().map(|&t| t.into()).collect()
        } else {
            tokens
                .extract::<Vec<u32>>()?
                .into_iter()
                .map(Into::into)
                .collect()
        };
        Ok(self.tokenizer.decode(&token_ids).collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

#[pyclass]
struct SentencePieceTokenizer {
    tokenizer: bpe::SentencePieceBPE,
}

impl SentencePieceTokenizer {
    /// Shared core of encode_batch for pre-resolved document slices: group
    /// whole documents into parallel chunks and encode each with its own
    /// Encoder. SentencePiece merges can span the whole document, so
    /// oversized documents are never split. Call with the GIL released.
    fn encode_slices_ragged(&self, docs: &[&[u8]]) -> PyResult<(Vec<u32>, Vec<i64>)> {
        use rayon::prelude::*;
        let texts: Vec<&str> = docs
            .iter()
            .map(|d| {
                std::str::from_utf8(d).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "invalid UTF-8 in document: {e}"
                    ))
                })
            })
            .collect::<PyResult<_>>()?;
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let target = chunk_target_bytes(total);
        let mut chunks: Vec<Vec<&str>> = Vec::new();
        let mut group: Vec<&str> = Vec::new();
        let mut acc = 0usize;
        for &text in &texts {
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
        let encode_group = |group: &Vec<&str>| -> ChunkTokens {
            let mut encoder = self.tokenizer.encoder();
            let mut ids: Vec<u32> = Vec::new();
            let mut lens: Vec<i64> = Vec::new();
            for text in group {
                let before = ids.len();
                ids.extend(
                    encoder
                        .encode_raw(text)
                        .into_iter()
                        .map(|t| -> u32 { t.into() }),
                );
                lens.push((ids.len() - before) as i64);
            }
            ChunkTokens {
                ids,
                lens,
                continues: false,
            }
        };
        let outs: Vec<ChunkTokens> = if chunks.len() <= 1 {
            chunks.iter().map(encode_group).collect()
        } else {
            chunks.par_iter().map(encode_group).collect()
        };
        Ok(assemble_ragged(outs))
    }
}

#[pymethods]
impl SentencePieceTokenizer {
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_sentencepiece(&path)?,
        })
    }

    /// Encode a batch of documents in parallel, releasing the GIL. Accepts
    /// the same inputs and returns the same awkward.Array shape as
    /// BPETokenizer.encode_batch. Documents must be valid UTF-8.
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs| self.encode_slices_ragged(docs))
    }

    fn encode<'py>(&self, py: Python<'py>, input: &str) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(self
            .tokenizer
            .encoder()
            .encode_raw(input)
            .into_iter()
            .map(|t| t.into())
            .collect::<Vec<u32>>()
            .into_pyarray(py))
    }

    /// Encode all documents from files in parallel. Accepts the same
    /// sources and applies the same chunking policy as
    /// BPETokenizer.encode_files, and likewise returns an awkward.Array
    /// with one row of token ids per document.
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use input::jsonl::JsonLinesSlice;
        use rayon::prelude::*;

        let (paths, format) = resolve_files_source(&source)?;
        let (flat, counts) = py.detach(|| -> PyResult<_> {
            let files = load_files(&paths)?;
            let total: usize = files.iter().map(|f| f.as_bytes().len()).sum();
            let target = chunk_target_bytes(total);
            let chunks: Vec<(usize, Range<usize>)> = files
                .iter()
                .enumerate()
                .flat_map(|(i, f)| {
                    chunk_ranges(f.as_bytes(), &format, target)
                        .into_iter()
                        .map(move |r| (i, r))
                })
                .collect();
            let encode_region = |(file, range): &(usize, Range<usize>)| -> ChunkTokens {
                let bytes = &files[*file].as_bytes()[range.clone()];
                let mut encoder = self.tokenizer.encoder();
                let mut ids: Vec<u32> = Vec::new();
                let mut lens: Vec<i64> = Vec::new();
                let mut encode = |doc: &[u8]| {
                    let text = unsafe { std::str::from_utf8_unchecked(doc) };
                    let before = ids.len();
                    ids.extend(
                        encoder
                            .encode_raw(text)
                            .into_iter()
                            .map(|t| -> u32 { t.into() }),
                    );
                    lens.push((ids.len() - before) as i64);
                };
                match &format {
                    DocFormat::Jsonl { field } => {
                        for doc in JsonLinesSlice::new(bytes, field) {
                            encode(doc.as_ref());
                        }
                    }
                    DocFormat::Text {
                        separator: Some(sep),
                    } if !sep.is_empty() => {
                        for doc in DocumentIter::new(bytes, sep) {
                            encode(doc);
                        }
                    }
                    DocFormat::Text { .. } => encode(bytes),
                }
                ChunkTokens {
                    ids,
                    lens,
                    continues: false,
                }
            };
            let outs: Vec<ChunkTokens> = if chunks.len() <= 1 {
                chunks.iter().map(encode_region).collect()
            } else {
                chunks.par_iter().map(encode_region).collect()
            };
            Ok(assemble_ragged(outs))
        })?;
        ragged_to_python(py, flat, counts)
    }

    fn encode_no_normalize<'py>(
        &self,
        py: Python<'py>,
        input: &str,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(self
            .tokenizer
            .encoder()
            .encode_normalized(input)
            .into_iter()
            .map(|t| t.into())
            .collect::<Vec<u32>>()
            .into_pyarray(py))
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let token_ids: Vec<crate::token::TokenId> = if let Ok(arr) = tokens.cast::<PyArray1<u32>>()
        {
            let arr = arr.readonly();
            arr.as_slice()?.iter().map(|&t| t.into()).collect()
        } else {
            tokens
                .extract::<Vec<u32>>()?
                .into_iter()
                .map(Into::into)
                .collect()
        };
        Ok(self.tokenizer.decode(&token_ids))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

/// Load a tokenizer from in-memory HuggingFace `tokenizer.json` contents
/// (str or bytes). Returns a SentencePieceTokenizer when the model uses
/// byte_fallback, a BPETokenizer otherwise — the same split as the two
/// classes' from_hf constructors.
#[pyfunction]
fn load_hf_json(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let backed_str;
    let backed_bytes;
    let bytes: &[u8] = if let Ok(s) = data.extract::<PyBackedStr>() {
        backed_str = s;
        backed_str.as_bytes()
    } else if let Ok(b) = data.extract::<PyBackedBytes>() {
        backed_bytes = b;
        &backed_bytes
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "expected tokenizer.json contents as str or bytes, got {}",
            data.get_type()
        )));
    };
    match load_tokenizer::hf::load_hf_slice(bytes)? {
        load_tokenizer::hf::HfTokenizer::Bpe(tokenizer) => Ok(Py::new(
            py,
            BPETokenizer {
                tokenizer,
                workers: WorkerPool::new(),
            },
        )?
        .into_any()),
        load_tokenizer::hf::HfTokenizer::SentencePiece(tokenizer) => {
            Ok(Py::new(py, SentencePieceTokenizer { tokenizer })?.into_any())
        }
    }
}

#[pyclass]
struct PretokenizerIter {
    /// Byte offset into `bytes`; the pretokenizer is stateless beyond this, so
    /// each `__next__` resumes a fresh `FastR50kPretokenizer` at this position.
    pos: usize,
    bytes: Py<PyBytes>,
}

#[pymethods]
impl PretokenizerIter {
    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, PretokenizerIter> {
        slf
    }

    fn __next__<'py>(&'py mut self, py: Python<'py>) -> Option<&'py [u8]> {
        let bytes: &'py [u8] = self.bytes.as_bytes(py);
        let mut iter = pretokenize::FastR50kPretokenizer::with_pos(bytes, self.pos);
        let result = iter.next();
        self.pos = iter.pos();
        Some(result?.0)
    }
}

#[pyfunction]
fn pretokenizer<'py>(text: Bound<'py, PyBytes>) -> PyResult<PretokenizerIter> {
    Ok(PretokenizerIter {
        pos: 0,
        bytes: text.into(),
    })
}

#[pyfunction]
#[pyo3(signature = (text, separator = None))]
fn pretokenized_counts<'py>(
    text: Bound<'py, PyBytes>,
    separator: Option<&[u8]>,
) -> PyResult<Vec<(Bound<'py, PyBytes>, usize)>> {
    let separator = separator.unwrap_or(pretokenize::DEFAULT_SEPARATOR);
    let tokens_counts = pretokenize::pretokenize_par_bytes(text.as_bytes(), separator);
    let tokens_counts = tokens_counts
        .into_iter()
        .map(|(k, v)| (PyBytes::new(text.py(), k.as_ref()), v))
        .collect::<Vec<_>>();
    Ok(tokens_counts)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn gigatok_rs<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<FileSource>()?;
    m.add_class::<TextFileSource>()?;
    m.add_class::<JsonlFileSource>()?;
    m.add_class::<PretokenizerIter>()?;
    m.add_class::<BPETokenizer>()?;
    m.add_class::<SentencePieceTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    m.add_function(wrap_pyfunction!(load_hf_json, m)?)?;
    Ok(())
}
