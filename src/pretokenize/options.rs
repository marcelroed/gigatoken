use crate::pretokenize::Pretoken;
use crate::pretokenize::fast::{FastCl100kPretokenizer, FastQwen2Pretokenizer, FastR50kPretokenizer};

/// Which pretokenization scheme (regex) a tokenizer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokenizerType {
    GPT2, // Also used by llama, also known as r50k
    GPT4, // cl100k
    Qwen2,      // Slightly adapted from GPT4, also used by Qwen3
    DeepSeekV3, // o200k, also used by GPT-4o
}

impl PretokenizerType {
    /// Fast pretokenizer for this scheme.
    ///
    /// The returned enum dispatches once per token; for hot loops over a
    /// known scheme, use the concrete iterator types directly.
    #[inline]
    pub fn pretokenize<'a>(&self, bytes: &'a [u8]) -> FastPretokenizerDispatch<'a> {
        match self {
            PretokenizerType::GPT2 => {
                FastPretokenizerDispatch::R50k(FastR50kPretokenizer::new(bytes))
            }
            PretokenizerType::GPT4 => {
                FastPretokenizerDispatch::Cl100k(FastCl100kPretokenizer::new(bytes))
            }
            PretokenizerType::Qwen2 => {
                FastPretokenizerDispatch::Qwen2(FastQwen2Pretokenizer::new(bytes))
            }
            PretokenizerType::DeepSeekV3 => {
                unimplemented!("no fast pretokenizer for {self:?} yet")
            }
        }
    }
}

/// Runtime-selected fast pretokenizer; add a variant here when implementing
/// a new scheme under `fast`.
pub enum FastPretokenizerDispatch<'a> {
    R50k(FastR50kPretokenizer<'a>),
    Cl100k(FastCl100kPretokenizer<'a>),
    Qwen2(FastQwen2Pretokenizer<'a>),
}

impl<'a> Iterator for FastPretokenizerDispatch<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        match self {
            FastPretokenizerDispatch::R50k(it) => it.next(),
            FastPretokenizerDispatch::Cl100k(it) => it.next(),
            FastPretokenizerDispatch::Qwen2(it) => it.next(),
        }
    }
}
