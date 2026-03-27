use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::file_source::Compression;

/// Open a file with optional decompression, returning a buffered reader.
pub(crate) fn open_reader(
    path: &Path,
    compression: Compression,
) -> io::Result<Box<dyn BufRead + Send>> {
    let file = File::open(path)?;
    match compression {
        Compression::None => Ok(Box::new(BufReader::new(file))),
        Compression::Gzip => Ok(Box::new(BufReader::new(
            flate2::read::GzDecoder::new(file),
        ))),
        Compression::Zstd => Ok(Box::new(BufReader::new(zstd::Decoder::new(file)?))),
    }
}
