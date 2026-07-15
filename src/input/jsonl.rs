use std::io::BufRead;

use sonic_rs::JsonValueTrait;

/// Zero-copy JSONL iterator over a byte slice (e.g. from mmap).
/// Used for parallel chunking of uncompressed JSONL files.
pub struct JsonLinesSlice<'a> {
    slice: &'a [u8],
    position: usize,
    field: &'a str,
}

impl<'a> JsonLinesSlice<'a> {
    pub fn new(slice: &'a [u8], field: &'a str) -> Self {
        Self {
            slice,
            position: 0,
            field,
        }
    }
}

impl<'a> Iterator for JsonLinesSlice<'a> {
    // JSON extraction does not expose a borrow that can outlive its parsed value.
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Skip newlines
            while self.position < self.slice.len() && self.slice[self.position] == b'\n' {
                self.position += 1;
            }
            if self.position >= self.slice.len() {
                return None;
            }

            let line_end = memchr::memchr(b'\n', &self.slice[self.position..])
                .map(|i| self.position + i)
                .unwrap_or(self.slice.len());
            let line = &self.slice[self.position..line_end];
            self.position = line_end + 1;

            if line.is_empty() {
                continue;
            }

            let value = sonic_rs::get_from_slice(line, &[self.field]).ok()?;
            let text = value.as_str()?;
            return Some(text.as_bytes().to_vec());
        }
    }
}

/// Streaming JSONL iterator over a `BufRead` source.
/// Reads one line at a time — never buffers the entire file.
pub(crate) struct JsonLinesReader<R> {
    reader: R,
    field: String,
    line_buf: Vec<u8>,
}

impl<R: BufRead> JsonLinesReader<R> {
    pub(crate) fn new(reader: R, field: &str) -> Self {
        Self {
            reader,
            field: field.to_string(),
            line_buf: Vec::with_capacity(4096),
        }
    }
}

impl<R: BufRead> Iterator for JsonLinesReader<R> {
    // Own each result because `line_buf` is reused on the next iteration.
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_buf.clear();
            let bytes_read = self.reader.read_until(b'\n', &mut self.line_buf).ok()?;
            if bytes_read == 0 {
                return None; // EOF
            }

            let line = self.line_buf.as_slice();
            // Skip empty lines
            if line.iter().all(|&b| b == b'\n' || b == b'\r') {
                continue;
            }

            let value = sonic_rs::get_from_slice(line, &[self.field.as_str()]).ok()?;
            let text = value.as_str()?;
            return Some(text.as_bytes().to_vec());
        }
    }
}
