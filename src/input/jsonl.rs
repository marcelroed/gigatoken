use crate::input::Document;
use sonic_rs::JsonValueTrait;

pub(crate) struct JsonLinesIter<'a> {
    slice: &'a [u8],
    position: usize,
    text_fieldname: &'a str,
}

impl<'a> JsonLinesIter<'a> {
    pub(crate) fn new(slice: &'a [u8], text_fieldname: &'a str) -> Self {
        Self {
            slice,
            position: 0,
            text_fieldname,
        }
    }
}

/// Iterate documents in a .jsonl file
impl<'a> Iterator for JsonLinesIter<'a> {
    type Item = Document<'static>; // Will always be owned because the Json needs to be parsed
    fn next(&mut self) -> Option<Self::Item> {
        // Skip any trailing newlines between records
        while self.position < self.slice.len() && self.slice[self.position] == b'\n' {
            self.position += 1;
        }
        if self.position >= self.slice.len() {
            return None;
        }

        // Find the end of this line
        let line_end = memchr::memchr(b'\n', &self.slice[self.position..])
            .map(|i| self.position + i)
            .unwrap_or(self.slice.len());
        let line = &self.slice[self.position..line_end];
        self.position = line_end + 1;

        let value = sonic_rs::get_from_slice(line, &[self.text_fieldname]).ok()?;
        let text = value.as_str()?;
        Some(Document::from(text.as_bytes().to_vec()))
    }
}
