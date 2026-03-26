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
use crate::pretokenize::pretokenize_as_iter;
pub(crate) mod load_tokenizer;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::path::{Path, PathBuf};

#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (in_data, vocab_size, special_tokens, tie_breaking = "huggingface"))]
fn train_bpe<'py>(
    py: Python<'py>,
    in_data: Bound<'py, PyAny>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    tie_breaking: &str,
) -> PyResult<(
    Bound<'py, PyDict>,
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
)> {
    println!("Started function");
    assert!(
        vocab_size <= 2_usize.pow(32),
        "vocab_size must be less than 2^32"
    );

    let tie_breaking = match tie_breaking {
        "huggingface" => bpe_train::TieBreaking::HuggingFace,
        "raw_token_ids" => bpe_train::TieBreaking::RawTokenIds,
        "assembled_bytes" => bpe_train::TieBreaking::AssembledBytes,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "tie_breaking must be 'huggingface', 'raw_token_ids', or 'assembled_bytes', got {other:?}"
            )));
        }
    };

    // Check which input we got for
    let mut bytes_memmapped = None;
    let pretokenizeable = if in_data.is_instance_of::<PyBytes>() {
        bpe_train::PretokenizeableSpec::Bytes(in_data.extract::<&[u8]>()?)
    } else if let Ok(path) = in_data.extract::<PathBuf>() {
        println!("Input is a path");
        if let Some(ext) = path.extension()
            && ext == "parquet"
        {
            eprintln!("Path is a parquet file");
            #[cfg(not(feature = "parquet"))]
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "The 'parquet' feature is not enabled in this build, cannot read parquet files",
            ));
            #[cfg(feature = "parquet")]
            bpe_train::PretokenizeableSpec::Parquet(path)
        } else {
            // Memmap the file and treat it as a slice of bytes
            let file = std::fs::File::open(&path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to open file {:?}: {}",
                    path, e
                ))
            })?;
            bytes_memmapped = Some(unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to mmap file {:?}: {}",
                    path, e
                ))
            })?);
            bpe_train::PretokenizeableSpec::Bytes(&bytes_memmapped.unwrap())
        }
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "in_data must be bytes or a path",
        ));
    };

    // Train BPE
    let bpe_train::BPEResult { vocab, merges } =
        bpe_train::train_bpe(pretokenizeable, vocab_size, special_tokens, tie_breaking);

    // Convert vocab to Python
    let vocab_py = vocab
        .into_iter()
        .map(|(k, v)| (k, PyBytes::new(py, &v)))
        .sorted_by(|e1, e2| Ord::cmp(&e1.0, &e2.0))
        .into_py_dict(py);

    // Convert merges to Python
    let merges_py: Vec<_> = merges
        .into_iter()
        .map(|(k, v)| (PyBytes::new(py, &k), PyBytes::new(py, &v)))
        .collect();

    Ok((vocab_py?, merges_py))
}

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

    /// Mumbo jumbo to shuffle around lifetimes so this fits into PyO3's restrictions
    fn __next__<'py>(&'py mut self, py: Python<'py>) -> Option<&'py [u8]> {
        let bytes: &'py [u8] = self.bytes.as_bytes(py);
        let result: Option<&'py [u8]> = self.pretokenizer_iter.py_next(bytes);
        result
    }
}

#[pyfunction]
fn pretokenizer<'py>(text: Bound<'py, PyBytes>) -> PyResult<PretokenizerIter> {
    // let text = text.as_bytes();
    let tokens_iter = pretokenize::pretokenize_as_iter((&[]).as_slice().into());
    Ok(PretokenizerIter {
        pretokenizer_iter: tokens_iter,
        bytes: text.into(),
    })
}

#[pyfunction]
fn pretokenized_counts<'py>(
    text: Bound<'py, PyBytes>,
) -> PyResult<Vec<(Bound<'py, PyBytes>, usize)>> {
    let tokens_counts =
        pretokenize::pretokenize_par(bpe_train::PretokenizeableSpec::Bytes(text.as_bytes()));
    // Convert keys to PyBytes

    let tokens_counts = tokens_counts
        .into_iter()
        .map(|(k, v)| (PyBytes::new(text.py(), k.as_ref()), v))
        .collect::<Vec<_>>();

    Ok(tokens_counts)
}

/// A Python module implemented in Rust.
#[pymodule]
fn jeton_rs<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_class::<RustTokenizer>()?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<PretokenizerIter>()?;
    m.add_class::<BPETokenizer>()?;
    m.add_class::<LlamaTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenized_counts, m)?)?;
    Ok(())
}
