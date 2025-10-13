// Different ways to construct (parallel) document iterators from file or Python input
use memmap2::Mmap;
use rayon::prelude::*;
use std::ops::Deref;
use std::{borrow::Cow, error::Error, path::Path};

mod bytes;
mod jsonl;
mod py;

#[derive(Debug, Clone)]
pub(crate) struct Document<'a>(Cow<'a, [u8]>);

impl<'a> From<&'a [u8]> for Document<'a> {
    fn from(value: &'a [u8]) -> Self {
        Document(Cow::Borrowed(value))
    }
}
impl<'a> From<Vec<u8>> for Document<'a> {
    fn from(value: Vec<u8>) -> Self {
        Document(value.into())
    }
}

impl<'a> Deref for Document<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0.as_ref()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DocRef<'a>(pub &'a [u8]);

impl<'a> From<&'a [u8]> for DocRef<'a> {
    fn from(value: &'a [u8]) -> Self {
        DocRef(value)
    }
}

impl<'a> From<&'a Document<'a>> for DocRef<'a> {
    fn from(value: &'a Document<'a>) -> Self {
        DocRef(value.as_ref())
    }
}

impl<'a> Deref for DocRef<'a> {
    type Target = &'a [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(crate) enum DocReader {
    FromFiles,
    PythonBytes,
}

pub(crate) enum InputData {
    MmapFile(Mmap),
    PythonBytes,
}

pub(crate) fn read_file(path: impl AsRef<Path>) -> Result<Mmap, String> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => todo!(),
        Some("parquet") => todo!(),
        Some("txt") | None => {
            // TODO: Make this a one-time warning
            eprintln!("Path {path:?} is being treated as a UTF-8 blob.")
        }
        Some(x) => {
            eprintln!(
                "Path {path:?} has extension {x} which was not recognized. Falling back to reading it as a blob of UTF-8."
            )
        }
    }
    let file = std::fs::File::open(path).map_err(|e| format!("{e}"))?;
    unsafe { Mmap::map(&file) }.map_err(|e| format!("{e}"))
}

/// Memmap each file
pub fn iterate_files(
    path_iterator: impl ParallelIterator<Item = impl AsRef<Path>>,
) -> impl ParallelIterator<Item = Result<impl AsRef<[u8]>, String>> {
    path_iterator
        .map(|path| {
            let file = std::fs::File::open(path.as_ref())?;
            unsafe { Mmap::map(&file) }
        })
        .map(|r| r.map_err(|e| format!("Failed to open file: {e}")))
}

// /// Create a parallel iterator over num_chunks number of inputs using a closure to find boundaries
// pub(crate) trait DocumentParallelCoarse<'a> {
//     fn par_iter_coarse(
//         &self,
//         first_boundary_fn: impl FnOnce(&[u8], usize) -> usize,
//     ) -> impl ParallelIterator<Item = impl Iterator<Item = Document<'a>>>;
// }
