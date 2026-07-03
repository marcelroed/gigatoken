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
use crate::bpe::Tokenizer;
use crate::input::{MmappedFile, Resource};
pub mod load_tokenizer;
use itertools::Itertools;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Helper: convert BPEResult to Python objects
// ---------------------------------------------------------------------------

fn bpe_result_to_python<'py>(
    py: Python<'py>,
    result: bpe_train::BPEResult,
) -> PyResult<(
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
)> {
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
// FileSource Python class
// ---------------------------------------------------------------------------

#[pyclass(from_py_object)]
#[derive(Clone)]
struct FileSource {
    paths: Vec<PathBuf>,
    field: String,
    separator: Vec<u8>,
}

#[pymethods]
impl FileSource {
    #[new]
    #[pyo3(signature = (paths, field = "text", separator = None))]
    fn new(paths: Vec<PathBuf>, field: &str, separator: Option<&[u8]>) -> Self {
        Self {
            paths,
            field: field.to_string(),
            separator: separator.unwrap_or(pretokenize::DEFAULT_SEPARATOR).to_vec(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FileSource(paths=[{} files], field={:?})",
            self.paths.len(),
            self.field
        )
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
        let spec = input::file_source::FileSourceSpec {
            paths: file_source.paths,
            field: file_source.field,
            separator: file_source.separator,
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

#[pyclass]
struct BPETokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn __new__() -> PyResult<Self> {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let tiktoken_path = data_dir.join("tokenizers/r50k_base.tiktoken");
        Ok(Self {
            tokenizer: load_tokenizer::tiktoken::load_tiktoken(tiktoken_path)?,
        })
    }
    #[staticmethod]
    fn from_tiktoken(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::tiktoken::load_tiktoken(&path)?,
        })
    }
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_bpe(&path)?,
        })
    }
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        input: &[u8],
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let mut v = vec![];
        self.tokenizer.encode_with_added_tokens(input, |tokens| {
            for &e in tokens {
                v.push(e.into())
            }
        });
        Ok(v.into_pyarray(py))
    }

    /// Encode a batch of documents in parallel with rayon, releasing the GIL.
    /// Uses the tokenizer's own pretokenization scheme and added tokens.
    /// Documents are processed in coarse chunks so each fork (a full clone of
    /// the merge/vocab maps, plus a cold pretoken cache) is amortized over
    /// many documents instead of one per work-stealing split.
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
        use rayon::prelude::*;
        let n_chunks = 4 * rayon::current_num_threads();
        let chunk_size = inputs.len().div_ceil(n_chunks).max(1);
        let results: Vec<Vec<Vec<u32>>> = py.detach(|| {
            inputs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut tokenizer = self.tokenizer.fork();
                    chunk
                        .iter()
                        .map(|doc| {
                            let mut v = vec![];
                            tokenizer.encode_with_added_tokens(doc, |tokens| {
                                for &e in tokens {
                                    v.push(e.into())
                                }
                            });
                            v
                        })
                        .collect()
                })
                .collect()
        });
        Ok(results
            .into_iter()
            .flatten()
            .map(|v| v.into_pyarray(py))
            .collect())
    }

    /// Encode all documents from a FileSource in parallel.
    /// Everything happens in Rust: mmap, JSONL parse, pretokenize, BPE merge.
    fn encode_file<'py>(
        &self,
        py: Python<'py>,
        file_source: FileSource,
    ) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
        use input::jsonl::JsonLinesSlice;
        use rayon::prelude::*;

        let spec = input::file_source::FileSourceSpec {
            paths: file_source.paths,
            field: file_source.field,
            separator: file_source.separator,
        };
        let files = spec
            .mmap_files()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let mut all_results: Vec<Bound<'py, PyArray1<u32>>> = Vec::new();
        for (mmap, boundaries, _content) in &files {
            let bytes = mmap.as_bytes();
            let chunk_results: Vec<Vec<Vec<u32>>> = boundaries
                .par_windows(2)
                .map(|w| {
                    let chunk = &bytes[w[0]..w[1]];
                    let mut tokenizer = self.tokenizer.fork();
                    JsonLinesSlice::new(chunk, spec.field())
                        .map(|doc| {
                            let mut v = vec![];
                            tokenizer.encode_with_added_tokens(doc.as_ref(), |tokens| {
                                for &e in tokens {
                                    v.push(e.into())
                                }
                            });
                            v
                        })
                        .collect()
                })
                .collect();
            for chunk in chunk_results {
                all_results.extend(chunk.into_iter().map(|v| v.into_pyarray(py)));
            }
        }
        Ok(all_results)
    }

    fn decode(&self, tokens: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let token_ids: Vec<crate::token::TokenId> =
            if let Ok(arr) = tokens.cast::<PyArray1<u32>>() {
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

#[pymethods]
impl SentencePieceTokenizer {
    #[staticmethod]
    fn from_hf(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_sentencepiece(&path)?,
        })
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

    /// Encode all documents from a FileSource in parallel.
    fn encode_file<'py>(
        &self,
        py: Python<'py>,
        file_source: FileSource,
    ) -> PyResult<Vec<Bound<'py, PyArray1<u32>>>> {
        use input::jsonl::JsonLinesSlice;
        use rayon::prelude::*;

        let spec = input::file_source::FileSourceSpec {
            paths: file_source.paths,
            field: file_source.field,
            separator: file_source.separator,
        };
        let files = spec
            .mmap_files()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let mut all_results: Vec<Bound<'py, PyArray1<u32>>> = Vec::new();
        for (mmap, boundaries, _content) in &files {
            let bytes = mmap.as_bytes();
            let chunk_results: Vec<Vec<Vec<u32>>> = boundaries
                .par_windows(2)
                .map(|w| {
                    let chunk = &bytes[w[0]..w[1]];
                    let mut encoder = self.tokenizer.encoder();
                    JsonLinesSlice::new(chunk, spec.field())
                        .map(|doc| {
                            let text = unsafe { std::str::from_utf8_unchecked(doc.as_ref()) };
                            encoder
                                .encode_raw(text)
                                .into_iter()
                                .map(|t| t.into())
                                .collect()
                        })
                        .collect()
                })
                .collect();
            for chunk in chunk_results {
                all_results.extend(chunk.into_iter().map(|v| v.into_pyarray(py)));
            }
        }
        Ok(all_results)
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
        let token_ids: Vec<crate::token::TokenId> =
            if let Ok(arr) = tokens.cast::<PyArray1<u32>>() {
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
fn jeton_rs<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<FileSource>()?;
    m.add_class::<PretokenizerIter>()?;
    m.add_class::<BPETokenizer>()?;
    m.add_class::<SentencePieceTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    Ok(())
}
