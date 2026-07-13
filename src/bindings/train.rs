//! Python binding for BPE training: the train_bpe function and the
//! conversion of its result into Python vocab/merges objects.

use super::sources::FileSource;
use crate::bpe_train;
use crate::input::file_source::FileSourceSpec;
use crate::input::{MmappedFile, Resource};
use crate::pretokenize;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::path::PathBuf;

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

#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (in_data, vocab_size, special_tokens, tie_breaking = "huggingface", separator = None))]
pub(crate) fn train_bpe<'py>(
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
