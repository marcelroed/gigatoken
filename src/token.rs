use std::fmt::Formatter;

/// Supports at most a vocab size of 2^32
#[derive(Copy, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
#[repr(transparent)]
pub struct TokenId(pub u32);

impl From<u32> for TokenId {
    fn from(value: u32) -> Self {
        TokenId(value)
    }
}

impl From<usize> for TokenId {
    fn from(value: usize) -> Self {
        TokenId(value as u32)
    }
}

impl From<i32> for TokenId {
    fn from(value: i32) -> Self {
        TokenId(value as u32)
    }
}

impl From<TokenId> for u32 {
    fn from(val: TokenId) -> Self {
        val.0
    }
}

impl From<TokenId> for usize {
    fn from(val: TokenId) -> Self {
        val.0 as usize
    }
}

impl std::fmt::Debug for TokenId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("<{}>", self.0))
    }
}
