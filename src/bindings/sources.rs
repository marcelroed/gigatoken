//! The FileSource Python classes and the helpers that turn an encode_files
//! or train_bpe argument into loaded, format-tagged file contents.

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

/// Load all files in parallel: mmap when stored uncompressed, decompress
/// .gz/.zst into memory otherwise (parallel chunking needs random access).
pub(crate) fn load_files(paths: &[PathBuf]) -> PyResult<Vec<LoadedFile>> {
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
