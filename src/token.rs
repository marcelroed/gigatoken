/// Supports at most a vocab size of 2^32
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Ord, PartialOrd)]
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

impl Into<u32> for TokenId {
    fn into(self) -> u32 {
        self.0
    }
}
