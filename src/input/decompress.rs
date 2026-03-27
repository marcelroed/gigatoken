use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

pub(crate) fn open_plain(path: &Path) -> io::Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path)?))
}

pub(crate) fn open_gzip(path: &Path) -> io::Result<BufReader<flate2::read::GzDecoder<File>>> {
    Ok(BufReader::new(flate2::read::GzDecoder::new(File::open(path)?)))
}

pub(crate) fn open_zstd(path: &Path) -> io::Result<BufReader<zstd::Decoder<'static, BufReader<File>>>> {
    Ok(BufReader::new(zstd::Decoder::new(File::open(path)?)?))
}
