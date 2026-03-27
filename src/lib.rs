#![feature(test)]
#![feature(portable_simd)]

pub(crate) mod bpe;
pub(crate) mod bpe_train;
pub(crate) mod input;
pub mod pretokenize;
pub(crate) mod simd;
pub(crate) mod token;
pub(crate) mod unicode_tables;
pub mod utils;
use crate::bpe::Tokenizer;
use crate::input::{MmappedFile, Resource};
use crate::pretokenize::pretokenize_as_iter;
pub(crate) mod load_tokenizer;
use itertools::Itertools;
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
            separator: separator
                .unwrap_or(pretokenize::DEFAULT_SEPARATOR)
                .to_vec(),
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
        let counts = spec
            .pretokenize()
            .map_err(|e| {
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
                let result =
                    bpe_train::train_bpe(counts, vocab_size, special_tokens, tie_breaking);
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
    fn from_tiktoken(path: &str) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::tiktoken::load_tiktoken(path)?,
        })
    }
    fn encode(&mut self, input: &[u8]) -> PyResult<Vec<u32>> {
        let iter = self.tokenizer.memoized_encode(pretokenize_as_iter(input));
        let mut v = vec![];
        for arc in iter {
            for &e in arc.into_iter() {
                v.push(e.into())
            }
        }
        Ok(v)
    }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

#[pyclass]
struct LlamaTokenizer {
    tokenizer: bpe::SentencePieceBPE,
}

#[pymethods]
impl LlamaTokenizer {
    #[staticmethod]
    fn from_hf(path: &str) -> PyResult<Self> {
        Ok(Self {
            tokenizer: load_tokenizer::hf::load_hf_tokenizer(path)?,
        })
    }

    fn encode(&self, input: &str) -> PyResult<Vec<u32>> {
        let normalized = bpe::SentencePieceBPE::normalize(input);
        Ok(self
            .tokenizer
            .encode(&normalized)
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    fn encode_no_normalize(&self, input: &str) -> PyResult<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(input)
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    fn decode(&self, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
        let token_ids: Vec<crate::token::TokenId> = tokens.into_iter().map(Into::into).collect();
        Ok(self.tokenizer.decode(&token_ids))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.tokenizer))
    }
}

#[pyclass]
struct PretokenizerIter {
    pretokenizer_iter: pretokenize::PretokenizerIter<'static>,
    bytes: Py<PyBytes>,
}

#[pymethods]
impl PretokenizerIter {
    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, PretokenizerIter> {
        slf
    }

    fn __next__<'py>(&'py mut self, py: Python<'py>) -> Option<&'py [u8]> {
        let bytes: &'py [u8] = self.bytes.as_bytes(py);
        let result: Option<&'py [u8]> = self.pretokenizer_iter.py_next(bytes);
        result
    }
}

#[pyfunction]
fn pretokenizer<'py>(text: Bound<'py, PyBytes>) -> PyResult<PretokenizerIter> {
    let tokens_iter = pretokenize::pretokenize_as_iter((&[]).as_slice().into());
    Ok(PretokenizerIter {
        pretokenizer_iter: tokens_iter,
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
    m.add_class::<LlamaTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    Ok(())
}
