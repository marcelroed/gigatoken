use crate::input::DocRef;
use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::unicode;

#[derive(Clone, Debug)]
pub enum PretokenizerState {
    Start,   // Not matched anything yet
    Nonchar, // Matched some non-alphanumeric and non-whitespace characters, continue until something matching
    Apostrophe,
    AsciiSpace,
    Whitespace(u8),
    Letter,
    Number,
    Save,   // Save the current token and start a new one
    Finish, // Ran out of tokens
}

pub struct UTF8Iterator<'a> {
    bytes: DocRef<'a>,
    pos: usize,
}

enum StartResult {
    Apostrophe,
    Letter,
    Number,
    AsciiSpace,
    Whitespace(u8),
    Nonchar,
}

enum WhitespaceResult {
    AsciiSpace,
    Whitespace(u8),
    Neither,
}

enum ApostropheResult {
    Matched,
    NotMatched,
}

pub(crate) struct OutOfBytesError {}

impl<'a> UTF8Iterator<'a> {
    pub(crate) fn new(doc: DocRef<'a>) -> Self {
        Self { bytes: doc, pos: 0 }
    }

    pub fn replace_bytes<'b>(&self, bytes: &'b [u8]) -> UTF8Iterator<'b> {
        UTF8Iterator {
            bytes: bytes.into(),
            pos: self.pos,
        }
    }

    /// Returns the next codepoint as a char (u32) and its length in bytes.
    /// We need the length to rewind if it needs to be reprocessed.
    fn next_codepoint_and_length(&mut self) -> Option<(char, usize)> {
        let cp = unsafe { str::from_utf8_unchecked(&self.bytes[self.pos..]) }
            .chars()
            .next()?;
        let len = cp.len_utf8();
        self.pos += len;
        Some((cp, len))
    }

    // #[inline(never)]
    fn start_check(&mut self) -> Result<StartResult, OutOfBytesError> {
        if self.pos >= self.bytes.0.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        if byte.is_ascii() {
            self.pos += 1;
            Ok(match byte {
                b'A'..=b'Z' | b'a'..=b'z' => StartResult::Letter,
                b' ' => StartResult::AsciiSpace,
                9..=13 => StartResult::Whitespace(1),
                b'0'..=b'9' => StartResult::Number,
                b'\'' => StartResult::Apostrophe,
                _ => StartResult::Nonchar,
            })
        } else {
            let (next_codepoint, len) =
                self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
            let gc = unicode::get_general_category(next_codepoint);
            Ok(if unicode::is_gc_letter(gc) {
                StartResult::Letter
            } else if unicode::is_gc_number(gc) {
                StartResult::Number
            } else if unicode::is_whitespace(next_codepoint) {
                StartResult::Whitespace(len as u8)
            } else {
                StartResult::Nonchar
            })
        }
    }

    // #[inline(never)]
    fn whitespace_check(&mut self) -> Result<WhitespaceResult, OutOfBytesError> {
        if self.pos >= self.bytes.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        if byte.is_ascii() {
            Ok(match byte {
                b' ' => {
                    self.pos += 1;
                    WhitespaceResult::AsciiSpace
                }
                9..=13 => {
                    self.pos += 1;
                    WhitespaceResult::Whitespace(1)
                }
                _ => WhitespaceResult::Neither,
            })
        } else {
            let (next_codepoint, len) =
                self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
            Ok(if unicode::is_whitespace(next_codepoint) {
                WhitespaceResult::Whitespace(len as u8)
            } else {
                self.pos -= len;
                WhitespaceResult::Neither
            })
        }
    }

    // #[inline(never)]
    fn letter_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'A'..=b'Z' | b'a'..=b'z' => {
                        self.pos += 1;
                    }
                    _ => {
                        return Ok(());
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                if !unicode::is_letter(next_codepoint) {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }

    // #[inline(never)]
    fn number_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'0'..=b'9' => {
                        self.pos += 1;
                    }
                    _ => {
                        return Ok(());
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                if !unicode::is_number(next_codepoint) {
                    self.pos -= len; // Rewind
                    return Ok(());
                }
            }
        }
    }

    // #[inline(never)]
    fn other_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b' ' | 9..=13 => {
                        // Matches anything (not apostrophe though)
                        return Ok(());
                    }
                    _ => {
                        self.pos += 1;
                    }
                }
            } else {
                let (next_codepoint, len) =
                    self.next_codepoint_and_length().ok_or(OutOfBytesError {})?;
                let gc = unicode::get_general_category(next_codepoint);
                if unicode::is_gc_letter(gc)
                    || unicode::is_gc_number(gc)
                    || unicode::is_whitespace(next_codepoint)
                {
                    self.pos -= len;
                    return Ok(()); // We matched a letter or number, so we stop here
                }
            }
        }
    }

    // #[inline(never)]
    fn apostrophe_check(&mut self) -> Result<ApostropheResult, OutOfBytesError> {
        if self.pos >= self.bytes.len() {
            return Err(OutOfBytesError {});
        }
        let byte = self.bytes[self.pos];
        match byte {
            b's' | b'd' | b'm' | b't' => {
                self.pos += 1;
                Ok(ApostropheResult::Matched)
            }
            b'l' | b'v' | b'r' => {
                if self.pos + 1 >= self.bytes.len() {
                    // Not enough bytes for a two-letter contraction ('ll, 've, 're)
                    return Ok(ApostropheResult::NotMatched);
                }
                let next_byte = self.bytes[self.pos + 1];
                match (byte, next_byte) {
                    (b'l', b'l') | (b'v', b'e') | (b'r', b'e') => {
                        self.pos += 2;
                        Ok(ApostropheResult::Matched)
                    }
                    _ => Ok(ApostropheResult::NotMatched),
                }
            }
            _ => Ok(ApostropheResult::NotMatched),
        }
    }
}

pub struct PretokenizerIter<'a> {
    iter: UTF8Iterator<'a>,
    starting: usize,
    state: PretokenizerState,
}

impl<'a> PretokenizerIter<'a> {
    pub fn new(input: &'a [u8]) -> PretokenizerIter<'a> {
        PretokenizerIter {
            iter: UTF8Iterator::new(input.into()),
            starting: 0,
            state: PretokenizerState::Start,
        }
    }
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = Pretoken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (state_after, new_pretoken) = loop {
            self.state = match self.state {
                PretokenizerState::Start => match self.iter.start_check() {
                    Ok(StartResult::Apostrophe) => {
                        if self.starting == self.iter.pos - 1 {
                            PretokenizerState::Apostrophe
                        } else {
                            // Only treat as apostrophe if we don't have a preceding space
                            PretokenizerState::Nonchar
                        }
                    }
                    Ok(StartResult::Letter) => PretokenizerState::Letter,
                    Ok(StartResult::Number) => PretokenizerState::Number,
                    Ok(StartResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(StartResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(StartResult::Nonchar) => PretokenizerState::Nonchar,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Save => {
                    let saved_tokens = &self.iter.bytes[self.starting..self.iter.pos];
                    self.starting = self.iter.pos;
                    break (PretokenizerState::Start, saved_tokens);
                }
                PretokenizerState::Apostrophe => match self.iter.apostrophe_check() {
                    Ok(ApostropheResult::Matched) => PretokenizerState::Save,
                    Ok(ApostropheResult::NotMatched) => PretokenizerState::Nonchar,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Nonchar => match self.iter.other_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Letter => match self.iter.letter_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Number => match self.iter.number_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Whitespace(prev_wslen) => match self.iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token =
                            &self.iter.bytes[self.starting..self.iter.pos - (prev_wslen as usize)];
                        self.starting = self.iter.pos - (prev_wslen as usize);
                        if saved_token.is_empty() {
                            PretokenizerState::Save
                        } else {
                            break (PretokenizerState::Save, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::AsciiSpace => match self.iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => PretokenizerState::Whitespace(wslen),
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token = &self.iter.bytes[self.starting..self.iter.pos - 1];
                        if saved_token.is_empty() {
                            self.starting = self.iter.pos - 1;
                            PretokenizerState::Start
                        } else {
                            self.starting = self.iter.pos - 1;
                            break (PretokenizerState::Start, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => {
                        let saved_token = &self.iter.bytes[self.starting..self.iter.pos];
                        self.starting = self.iter.pos;
                        break (PretokenizerState::Finish, saved_token);
                    }
                },
                PretokenizerState::Finish => {
                    let saved_token = &self.iter.bytes[self.starting..self.iter.pos];
                    self.starting = self.iter.pos;
                    break (PretokenizerState::Finish, saved_token);
                }
            }
        };
        self.state = state_after;
        if new_pretoken.is_empty() {
            return None;
        }
        Some(Pretoken(new_pretoken))
    }
}

impl<'a> PretokenizerIter<'a> {
    pub fn replace_bytes<'b>(&self, bytes: &'b [u8]) -> PretokenizerIter<'b> {
        let iter = self.iter.replace_bytes(bytes);
        PretokenizerIter {
            iter,
            starting: self.starting,
            state: self.state.clone(),
        }
    }
}

impl PretokenizerIter<'static> {
    // If we contain a 'static, assume it's a dummy.
    // This is needed only for PyO3 bindings.
    pub fn py_next<'a>(&mut self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        let mut py_self = self.replace_bytes(bytes);
        let result = py_self.next();
        *self = py_self.replace_bytes(&[]);
        Some(result?.0)
    }
}

pub fn pretokenize_as_iter(bytes: &[u8]) -> PretokenizerIter<'_> {
    PretokenizerIter {
        iter: UTF8Iterator::new(bytes.into()),
        starting: 0,
        state: PretokenizerState::Start,
    }
}
