//! The FileSource Python classes and the helpers that turn an encode_files
//! or train_bpe argument into loaded, format-tagged file contents.

use crate::input::Resource;
use crate::input::file_source::{DocFormat, LoadedFile, detect_default_format, load_file};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Base class for file sources. Not directly constructible from Python —
/// use `TextFileSource` or `JsonlFileSource`, which pin down the document
/// format and its parameters. Compression (.gz/.zst) is always detected
/// from the file extension, independent of the source type.
#[pyclass(subclass, from_py_object)]
#[derive(Clone)]
pub(crate) struct FileSource {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) format: DocFormat,
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
pub(crate) struct TextFileSource;

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
pub(crate) struct JsonlFileSource;

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

/// Resolve an encode_files argument: a FileSource (TextFileSource /
/// JsonlFileSource), a single path, or a list of paths. Bare paths get a
/// default format from the first path's extension — all inputs in a batch
/// are assumed to be of the same type.
pub(crate) fn resolve_files_source(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<PathBuf>, DocFormat)> {
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

/// Shared scaffold of the encode_files pymethods: resolve the source
/// argument, load the files with the GIL released, hand their contents and
/// document format to `encode` (still detached), and return the ragged
/// result as an awkward Array. The per-backend encoding lives in
/// `batch::encode_files_docs` / `batch::sp_encode_files_docs`.
pub(crate) fn encode_files_ragged<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    parallel: bool,
    encode: impl FnOnce(&[&[u8]], &DocFormat) -> (Vec<u32>, Vec<i64>) + Send,
) -> PyResult<Bound<'py, PyAny>> {
    let (paths, format) = resolve_files_source(source)?;
    let (flat, counts) = py.detach(|| -> PyResult<_> {
        let files = load_files(&paths, parallel)?;
        let bytes: Vec<&[u8]> = files.iter().map(|f| f.as_bytes()).collect();
        Ok(encode(&bytes, &format))
    })?;
    super::bridge::ragged_to_python(py, flat, counts)
}

/// Load all files: mmap when stored uncompressed, decompress .gz/.zst into
/// memory otherwise (parallel chunking needs random access). In parallel
/// with rayon, or serially on the calling thread when `parallel` is false
/// (the sequential encode paths must never touch the rayon pool).
pub(crate) fn load_files(paths: &[PathBuf], parallel: bool) -> PyResult<Vec<LoadedFile>> {
    use rayon::prelude::*;
    let load = |p: &PathBuf| {
        load_file(p).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}: {e}", p.display()))
        })
    };
    if parallel {
        paths.par_iter().map(load).collect()
    } else {
        paths.iter().map(load).collect()
    }
}
