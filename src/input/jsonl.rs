use std::io::BufRead;

use crate::input::Document;
use sonic_rs::JsonValueTrait;

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
    type Item = Document<'static>;

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
            return Some(Document::from(text.as_bytes().to_vec()));
        }
    }
}
