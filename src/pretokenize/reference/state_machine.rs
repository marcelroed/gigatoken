//! Hand-rolled state-machine pretokenizer. Kept as a reference implementation
//! and benchmark baseline; the production pretokenizer is
//! `fast::r50k::FastR50kPretokenizer` (see `pretokenize_as_iter`).
use crate::input::DocRef;
use crate::pretokenize::pretoken::Pretoken;
use crate::pretokenize::unicode;

// ---------------------------------------------------------------------------
// State-machine implementation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum PretokenizerState {
    Start,
    Nonchar,
    Apostrophe,
    AsciiSpace,
    Whitespace(u8),
    Letter,
    Number,
    Save,
    Finish,
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
    fn next_codepoint_and_length(&mut self) -> Option<(char, usize)> {
        let cp = unsafe { str::from_utf8_unchecked(&self.bytes[self.pos..]) }
            .chars()
            .next()?;
        let len = cp.len_utf8();
        self.pos += len;
        Some((cp, len))
    }

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
                    self.pos -= len;
                    return Ok(());
                }
            }
        }
    }

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
                    self.pos -= len;
                    return Ok(());
                }
            }
        }
    }

    fn other_check(&mut self) -> Result<(), OutOfBytesError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Err(OutOfBytesError {});
            }
            let byte = self.bytes[self.pos];
            if byte.is_ascii() {
                match byte {
                    b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b' ' | 9..=13 => {
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
                    return Ok(());
                }
            }
        }
    }

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

// ---------------------------------------------------------------------------
// PretokenizerIter — state-machine pretokenizer
// ---------------------------------------------------------------------------

pub struct PretokenizerIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    state: PretokenizerState,
}

impl<'a> PretokenizerIter<'a> {
    pub fn new(input: &'a [u8]) -> PretokenizerIter<'a> {
        PretokenizerIter {
            bytes: input,
            pos: 0,
            state: PretokenizerState::Start,
        }
    }
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = Pretoken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_state_machine()
    }
}

impl<'a> PretokenizerIter<'a> {
    #[inline]
    fn next_state_machine(&mut self) -> Option<Pretoken<'a>> {
        let mut iter = UTF8Iterator {
            bytes: self.bytes.into(),
            pos: self.pos,
        };
        let starting = self.pos;
        let mut cur_starting = starting;

        let (state_after, new_pretoken) = loop {
            self.state = match self.state {
                PretokenizerState::Start => match iter.start_check() {
                    Ok(StartResult::Apostrophe) => {
                        if cur_starting == iter.pos - 1 {
                            PretokenizerState::Apostrophe
                        } else {
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
                    let saved_tokens = &self.bytes[cur_starting..iter.pos];
                    cur_starting = iter.pos;
                    break (PretokenizerState::Start, saved_tokens);
                }
                PretokenizerState::Apostrophe => match iter.apostrophe_check() {
                    Ok(ApostropheResult::Matched) => PretokenizerState::Save,
                    Ok(ApostropheResult::NotMatched) => PretokenizerState::Nonchar,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Nonchar => match iter.other_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Letter => match iter.letter_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Number => match iter.number_check() {
                    Ok(_) => PretokenizerState::Save,
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::Whitespace(prev_wslen) => match iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => {
                        PretokenizerState::Whitespace(wslen)
                    }
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token =
                            &self.bytes[cur_starting..iter.pos - (prev_wslen as usize)];
                        cur_starting = iter.pos - (prev_wslen as usize);
                        if saved_token.is_empty() {
                            PretokenizerState::Save
                        } else {
                            // The next token starts fresh at the reserved last
                            // whitespace char; resuming in `Save` would emit an
                            // empty span and end the stream.
                            break (PretokenizerState::Start, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => PretokenizerState::Finish,
                },
                PretokenizerState::AsciiSpace => match iter.whitespace_check() {
                    Ok(WhitespaceResult::AsciiSpace) => PretokenizerState::AsciiSpace,
                    Ok(WhitespaceResult::Whitespace(wslen)) => {
                        PretokenizerState::Whitespace(wslen)
                    }
                    Ok(WhitespaceResult::Neither) => {
                        let saved_token = &self.bytes[cur_starting..iter.pos - 1];
                        if saved_token.is_empty() {
                            cur_starting = iter.pos - 1;
                            PretokenizerState::Start
                        } else {
                            cur_starting = iter.pos - 1;
                            break (PretokenizerState::Start, saved_token);
                        }
                    }
                    Err(OutOfBytesError {}) => {
                        let saved_token = &self.bytes[cur_starting..iter.pos];
                        cur_starting = iter.pos;
                        break (PretokenizerState::Finish, saved_token);
                    }
                },
                PretokenizerState::Finish => {
                    let saved_token = &self.bytes[cur_starting..iter.pos];
                    cur_starting = iter.pos;
                    break (PretokenizerState::Finish, saved_token);
                }
            }
        };
        self.state = state_after;
        self.pos = cur_starting;
        if new_pretoken.is_empty() {
            return None;
        }
        Some(Pretoken(new_pretoken))
    }
}

