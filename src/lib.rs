#![feature(portable_simd)]

pub(crate) mod batch;
pub(crate) mod bindings;
pub(crate) mod bpe;
pub(crate) mod bpe_train;
pub(crate) mod input;
pub mod pretokenize;
pub(crate) mod token;
pub use crate::batch::{WorkerPool, encode_docs_ragged};
pub use crate::bpe::Tokenizer;
pub use crate::bpe::sentencepiece::EncodeState;
pub mod load_tokenizer;

use crate::batch::{
    encode_docs_ragged_serial, encode_files_docs, encode_files_docs_serial, encode_into,
    sp_encode_docs_ragged, sp_encode_docs_ragged_serial, sp_encode_files_docs,
    sp_encode_files_docs_serial,
};
use crate::bindings::bridge::{
    EncodeInput, encode_batch_ragged, extract_doc, extract_token_ids, merges_to_pylist,
    utf8_doc, vocab_to_pydict,
};
use crate::bindings::padding;
use crate::bindings::pretokenize::{PretokenizerIter, pretokenized_counts, pretokenizer};
use crate::bindings::sources::{
    FileSource, JsonlFileSource, TextFileSource, encode_files_ragged,
};
use crate::bindings::train::train_bpe;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

#[pyclass]
struct BPETokenizer {
    tokenizer: Tokenizer,
    workers: WorkerPool,
}

impl BPETokenizer {
    /// See `batch::encode_docs_ragged` / `batch::encode_docs_ragged_serial`.
    /// Call with the GIL released.
    fn encode_slices_ragged(&self, docs: &[&[u8]], parallel: bool) -> (Vec<u32>, Vec<i64>) {
        if parallel {
            encode_docs_ragged(&self.workers, &self.tokenizer, docs)
        } else {
            encode_docs_ragged_serial(&self.workers, &self.tokenizer, docs)
        }
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
    ///
    /// `parallel=False` encodes everything on the calling thread instead,
    /// with identical output, never touching the process-global thread pool
    /// — for calls inside multiprocessing worker processes (the
    /// gigatoken.Tokenizer wrapper detects those and passes it
    /// automatically).
    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs| Ok(self.encode_slices_ragged(docs, parallel)))
    }

    /// encode_batch assembled into one padded/truncated (rows x width)
    /// uint32 matrix plus each row's real length, serving the compatibility
    /// APIs — see src/bindings/padding.rs for the semantics (`options` is a
    /// PadTruncate) and gigatoken.Tokenizer.encode_batch_padded for the
    /// friendly keyword signature.
    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn encode_batch_padded<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: padding::PadTruncate,
        parallel: bool,
    ) -> PyResult<padding::PaddedMatrix<'py>> {
        padding::encode_batch_matrix(py, &inputs, options, parallel, |docs| {
            Ok(self.encode_slices_ragged(docs, parallel))
        })
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
    ///
    /// `parallel=False` loads and encodes everything on the calling thread
    /// instead, with identical output, never touching the process-global
    /// thread pool — for calls inside multiprocessing worker processes (the
    /// gigatoken.Tokenizer wrapper detects those and passes it
    /// automatically).
    #[pyo3(signature = (source, *, parallel = true))]
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_files_ragged(py, &source, parallel, |files, format| {
            if parallel {
                encode_files_docs(&self.workers, &self.tokenizer, files, format)
            } else {
                encode_files_docs_serial(&self.workers, &self.tokenizer, files, format)
            }
        })
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// The vocabulary as a freshly built dict mapping token ID to token
    /// bytes, in ID order, including added tokens.
    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        vocab_to_pydict(py, self.tokenizer.vocab_entries())
    }

    /// The merge rules as a freshly built list of `(left, right)` byte
    /// pairs in merge-priority order.
    #[getter]
    fn merges<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        merges_to_pylist(py, self.tokenizer.merge_entries())
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        Ok(self.tokenizer.decode(&extract_token_ids(&tokens)?).collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

#[pyclass]
struct SentencePieceTokenizer {
    tokenizer: bpe::SentencePieceBPE,
    /// Pretoken cache + scratch for single-document `encode`, persisting
    /// across calls (parallel paths use per-worker states instead).
    state: bpe::sentencepiece::EncodeState,
}

impl SentencePieceTokenizer {
    /// See `batch::sp_encode_docs_ragged` (`_serial` when `parallel` is
    /// false); documents must be valid UTF-8. Call with the GIL released.
    fn encode_slices_ragged(&self, docs: &[&[u8]], parallel: bool) -> PyResult<(Vec<u32>, Vec<i64>)> {
        let texts: Vec<&str> = docs.iter().map(|d| utf8_doc(d)).collect::<PyResult<_>>()?;
        Ok(if parallel {
            sp_encode_docs_ragged(&self.tokenizer, &texts)
        } else {
            sp_encode_docs_ragged_serial(&self.tokenizer, &texts)
        })
    }
}

#[pymethods]
impl SentencePieceTokenizer {
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_sentencepiece(&path)?,
            state: bpe::sentencepiece::EncodeState::new(),
        })
    }

    /// Encode a batch of documents in parallel, releasing the GIL. Accepts
    /// the same inputs, returns the same awkward.Array shape, and honors the
    /// same `parallel` keyword as BPETokenizer.encode_batch. Documents must
    /// be valid UTF-8.
    #[pyo3(signature = (inputs, *, parallel = true))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_batch_ragged(py, &inputs, |docs| self.encode_slices_ragged(docs, parallel))
    }

    /// See BPETokenizer.encode_batch_padded: the same padded-matrix batch
    /// encode, for the SentencePiece backend.
    #[pyo3(signature = (inputs, options, *, parallel = true))]
    fn encode_batch_padded<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyAny>,
        options: padding::PadTruncate,
        parallel: bool,
    ) -> PyResult<padding::PaddedMatrix<'py>> {
        padding::encode_batch_matrix(py, &inputs, options, parallel, |docs| {
            self.encode_slices_ragged(docs, parallel)
        })
    }

    /// Encode a single document (str or UTF-8 bytes), with a pretoken cache
    /// that persists across calls.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        input: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let input = extract_doc(&input)?;
        let text: &str = match &input {
            EncodeInput::Text(s) => s,
            EncodeInput::Bytes(b) => utf8_doc(b)?,
        };
        let mut ids: Vec<u32> = Vec::new();
        self.tokenizer
            .encode_raw_cb(&mut self.state, text, &mut |tokens| {
                ids.extend(tokens.iter().map(|&t| u32::from(t)))
            });
        Ok(ids.into_pyarray(py))
    }

    /// Encode all documents from files in parallel. Accepts the same
    /// sources, applies the same chunking policy, and honors the same
    /// `parallel` keyword as BPETokenizer.encode_files, and likewise
    /// returns an awkward.Array with one row of token ids per document.
    #[pyo3(signature = (source, *, parallel = true))]
    fn encode_files<'py>(
        &self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        encode_files_ragged(py, &source, parallel, |files, format| {
            if parallel {
                sp_encode_files_docs(&self.tokenizer, files, format)
            } else {
                sp_encode_files_docs_serial(&self.tokenizer, files, format)
            }
        })
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// The vocabulary as a freshly built dict mapping token ID to token
    /// bytes, in ID order, including added tokens.
    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        vocab_to_pydict(py, self.tokenizer.vocab_entries())
    }

    /// The merge rules as a freshly built list of `(left, right)` byte
    /// pairs in merge-priority order.
    #[getter]
    fn merges<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        merges_to_pylist(py, self.tokenizer.merge_entries())
    }

    fn encode_no_normalize<'py>(
        &mut self,
        py: Python<'py>,
        input: &str,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let mut ids: Vec<u32> = Vec::new();
        self.tokenizer
            .encode_normalized_cb(&mut self.state, input, &mut |tokens| {
                ids.extend(tokens.iter().map(|&t| u32::from(t)))
            });
        Ok(ids.into_pyarray(py))
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        Ok(self.tokenizer.decode(&extract_token_ids(&tokens)?))
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
        load_tokenizer::hf::HfTokenizer::SentencePiece(tokenizer) => Ok(Py::new(
            py,
            SentencePieceTokenizer {
                tokenizer,
                state: bpe::sentencepiece::EncodeState::new(),
            },
        )?
        .into_any()),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn gigatoken_rs<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<FileSource>()?;
    m.add_class::<TextFileSource>()?;
    m.add_class::<JsonlFileSource>()?;
    m.add_class::<PretokenizerIter>()?;
    m.add_class::<padding::PadTruncate>()?;
    m.add_class::<BPETokenizer>()?;
    m.add_class::<SentencePieceTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    m.add_function(wrap_pyfunction!(load_hf_json, m)?)?;
    Ok(())
}
