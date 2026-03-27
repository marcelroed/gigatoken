use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(crate) fn decompress_gzip(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    Ok(buf)
}

pub(crate) fn decompress_zstd(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut decoder = zstd::Decoder::new(file)?;
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    Ok(buf)
}
