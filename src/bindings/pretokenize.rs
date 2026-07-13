//! Python bindings exposing the pretokenizer directly (an iterator over one
//! document's pretokens and a parallel pretoken-count function) — used for
//! inspection and tests, not by the encode path.

use crate::pretokenize;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
pub(crate) struct PretokenizerIter {
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
pub(crate) fn pretokenizer<'py>(text: Bound<'py, PyBytes>) -> PyResult<PretokenizerIter> {
    Ok(PretokenizerIter {
        pos: 0,
        bytes: text.into(),
    })
}

#[pyfunction]
#[pyo3(signature = (text, separator = None))]
pub(crate) fn pretokenized_counts<'py>(
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
