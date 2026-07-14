//! Padded/truncated batch assembly, serving the drop-in compatibility APIs
//! (gigatoken.HFCompat's padding/truncation support). Kept out of lib.rs so
//! the main encode path stays easy to read: the `encode_batch_padded`
//! bindings there are one-call forwards into this module, and the options
//! travel as the Rust-defined `PadTruncate` class, so per call they cost a
//! typed downcast rather than dict-key lookups. The friendly keyword
//! signature lives on `gigatoken.Tokenizer.encode_batch_padded`.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// How `encode_batch_padded` assembles rows into a rectangular matrix.
/// A frozen pyclass: fields are validated once at construction, and bad
/// names or types fail there with a TypeError instead of at encode time.
#[pyclass(frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct PadTruncate {
    pub pad_id: u32,
    /// Truncation limit and/or fixed pad width, depending on the two flags.
    pub max_length: Option<usize>,
    /// Pad every row to exactly `max_length` instead of the longest row.
    pub pad_to_max_length: bool,
    /// Keep at most `max_length` ids per row, counting prefix and suffix.
    pub truncate: bool,
    /// Put the padding before the tokens instead of after them.
    pub pad_left: bool,
    /// Drop tokens from the start of a row instead of the end.
    pub truncate_left: bool,
    /// Special-token ids written before / after every row's tokens; they
    /// count toward the truncation budget, like HF post-processors.
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
}

#[pymethods]
impl PadTruncate {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (pad_id, max_length=None, pad_to_max_length=false, truncate=false, pad_left=false, truncate_left=false, prefix=Vec::new(), suffix=Vec::new()))]
    fn new(
        pad_id: u32,
        max_length: Option<usize>,
        pad_to_max_length: bool,
        truncate: bool,
        pad_left: bool,
        truncate_left: bool,
        prefix: Vec<u32>,
        suffix: Vec<u32>,
    ) -> Self {
        Self {
            pad_id,
            max_length,
            pad_to_max_length,
            truncate,
            pad_left,
            truncate_left,
            prefix,
            suffix,
        }
    }

    fn __repr__(&self) -> String {
        let py_bool = |b: bool| if b { "True" } else { "False" };
        let max_length = self
            .max_length
            .map_or("None".to_string(), |m| m.to_string());
        format!(
            "PadTruncate(pad_id={}, max_length={max_length}, pad_to_max_length={}, truncate={}, \
             pad_left={}, truncate_left={}, prefix={:?}, suffix={:?})",
            self.pad_id,
            py_bool(self.pad_to_max_length),
            py_bool(self.truncate),
            py_bool(self.pad_left),
            py_bool(self.truncate_left),
            self.prefix,
            self.suffix,
        )
    }
}

/// The (rows x width) id matrix plus each row's real (unpadded) length —
/// everything an attention mask needs.
pub type PaddedMatrix<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<i64>>);

/// Shared back-end of the `encode_batch_padded` bindings: encode like
/// encode_batch, then pad/truncate into one (rows x width) uint32 matrix.
pub fn encode_batch_matrix<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    opts: PadTruncate,
    parallel: bool,
    encode: impl Fn(&[&[u8]]) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<PaddedMatrix<'py>> {
    let (flat, counts) = super::bridge::encode_batch_flat(py, inputs, encode)?;
    let (data, lengths, width) = py.detach(|| pad_truncate_matrix(&flat, &counts, &opts, parallel))?;
    let rows = lengths.len();
    let matrix = data.into_pyarray(py).reshape([rows, width])?;
    Ok((matrix, lengths.into_pyarray(py)))
}

/// Assemble a ragged token batch into one row-major (rows x width) matrix in
/// a single parallel copy pass. Each row is `prefix` + the (optionally
/// truncated) tokens + `suffix`, padded with `pad_id` to the longest row (or
/// to exactly `max_length` when `pad_to_max_length`). Returns the matrix
/// data, each row's real (unpadded) length, and the width.
fn pad_truncate_matrix(
    flat: &[u32],
    counts: &[i64],
    opts: &PadTruncate,
    parallel: bool,
) -> PyResult<(Vec<u32>, Vec<i64>, usize)> {
    use rayon::prelude::*;
    let value_err = |msg: String| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg);
    let extra = opts.prefix.len() + opts.suffix.len();
    let cap = if opts.truncate {
        let max = opts
            .max_length
            .ok_or_else(|| value_err("truncate requires max_length".to_string()))?;
        max.checked_sub(extra).ok_or_else(|| {
            value_err(format!(
                "max_length={max} leaves no room for the {extra} special tokens added per sequence"
            ))
        })?
    } else {
        usize::MAX
    };
    let mut offsets = Vec::with_capacity(counts.len());
    let mut lengths = Vec::with_capacity(counts.len());
    let mut longest = 0usize;
    let mut pos = 0usize;
    for &count in counts {
        let kept = (count as usize).min(cap);
        offsets.push(pos);
        pos += count as usize;
        lengths.push((kept + extra) as i64);
        longest = longest.max(kept + extra);
    }
    let width = if opts.pad_to_max_length {
        let max = opts
            .max_length
            .ok_or_else(|| value_err("pad_to_max_length requires max_length".to_string()))?;
        if longest > max {
            return Err(value_err(format!(
                "a sequence is {longest} ids long but padding to max_length={max} was requested \
                 without truncation; enable truncation or raise max_length"
            )));
        }
        max
    } else {
        longest
    };
    let mut out = vec![opts.pad_id; counts.len() * width];
    if width > 0 {
        let fill = |(i, row): (usize, &mut [u32])| {
            let count = counts[i] as usize;
            let kept = count.min(cap);
            let tokens = &flat[offsets[i]..offsets[i] + count];
            let tokens = if opts.truncate_left {
                &tokens[count - kept..]
            } else {
                &tokens[..kept]
            };
            let len = kept + extra;
            let start = if opts.pad_left { width - len } else { 0 };
            row[start..start + opts.prefix.len()].copy_from_slice(&opts.prefix);
            row[start + opts.prefix.len()..start + opts.prefix.len() + kept].copy_from_slice(tokens);
            row[start + len - opts.suffix.len()..start + len].copy_from_slice(&opts.suffix);
        };
        if parallel {
            out.par_chunks_mut(width).enumerate().for_each(fill);
        } else {
            out.chunks_mut(width).enumerate().for_each(fill);
        }
    }
    Ok((out, lengths, width))
}
