//! Python<->Rust bridging shared by the encode/decode bindings: document
//! and token-id extraction from Python objects, vocab/merges conversion the
//! other way, and the shared front-ends that resolve an encode_batch input
//! and hand the ragged result back to Python.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::path::PathBuf;

/// A document to encode: str (UTF-8 text) or bytes. Both variants borrow
/// the Python object's buffer without copying and are usable with the GIL
/// released. Paths are deliberately not accepted here — encoding from files
/// goes through `encode_files`, which mmaps and chunks them.
pub(crate) enum EncodeInput {
    Text(PyBackedStr),
    Bytes(PyBackedBytes),
}

impl EncodeInput {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            EncodeInput::Text(s) => s.as_bytes(),
            EncodeInput::Bytes(b) => b,
        }
    }
}

/// Extract one document, pointing path-holders at encode_files.
pub(crate) fn extract_doc(obj: &Bound<'_, PyAny>) -> PyResult<EncodeInput> {
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

/// View one document's bytes as UTF-8 text for the SentencePiece path.
pub(crate) fn utf8_doc(doc: &[u8]) -> PyResult<&str> {
    std::str::from_utf8(doc).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid UTF-8 in document: {e}"))
    })
}

/// Extract decode() input: a numpy uint32 array or any sequence of ints.
pub(crate) fn extract_token_ids(tokens: &Bound<'_, PyAny>) -> PyResult<Vec<crate::token::TokenId>> {
    if let Ok(arr) = tokens.cast::<PyArray1<u32>>() {
        let arr = arr.readonly();
        Ok(arr.as_slice()?.iter().map(|&t| t.into()).collect())
    } else {
        Ok(tokens
            .extract::<Vec<u32>>()?
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

/// Build a `vocab` getter's dict from `(id, bytes)` entries.
pub(crate) fn vocab_to_pydict<'py, 'a>(
    py: Python<'py>,
    entries: impl Iterator<Item = (u32, &'a [u8])>,
) -> PyResult<Bound<'py, PyDict>> {
    entries
        .map(|(id, bytes)| (id, PyBytes::new(py, bytes)))
        .into_py_dict(py)
}

/// Build a `merges` getter's list from `(left, right)` byte pairs.
pub(crate) fn merges_to_pylist<'py>(
    py: Python<'py>,
    entries: Vec<(&[u8], &[u8])>,
) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    entries
        .into_iter()
        .map(|(a, b)| (PyBytes::new(py, a), PyBytes::new(py, b)))
        .collect()
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
pub(crate) fn ragged_to_python<'py>(
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

/// Shared front-end of encode_batch: extract the documents (a list of str, a
/// list of bytes, or an awkward Array of strings — whose flat buffer is used
/// directly, with no per-document Python objects) and run `encode` on the
/// resolved byte slices with the GIL released, returning the ragged result
/// as one flat id buffer plus per-document row lengths.
pub(crate) fn encode_batch_flat<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    encode: impl Fn(&[&[u8]]) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<(Vec<u32>, Vec<i64>)> {
    // Awkward input: encode straight from the flat content buffer.
    if let Some((content, in_counts)) = extract_awkward_docs(inputs)? {
        let content = content.readonly();
        let bytes: &[u8] = content.as_slice()?;
        return py.detach(|| -> PyResult<_> {
            let mut docs = Vec::with_capacity(in_counts.len());
            let mut pos = 0usize;
            for &n in &in_counts {
                docs.push(&bytes[pos..pos + n as usize]);
                pos += n as usize;
            }
            encode(&docs)
        });
    }

    let inputs: Vec<Bound<'py, PyAny>> = inputs.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected a list of str, a list of bytes, or an awkward Array of strings",
        )
    })?;
    if inputs.is_empty() {
        return Ok((vec![], vec![]));
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

    py.detach(|| {
        let slices: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
        encode(&slices)
    })
}

/// encode_batch: `encode_batch_flat`, handed to Python as an awkward.Array.
pub(crate) fn encode_batch_ragged<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    encode: impl Fn(&[&[u8]]) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    let (flat, counts) = encode_batch_flat(py, inputs, encode)?;
    ragged_to_python(py, flat, counts)
}
