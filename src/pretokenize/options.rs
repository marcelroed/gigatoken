use crate::pretokenize::Pretoken;
use crate::pretokenize::fast::{
    FastCl100kPretokenizer, FastOlmo3Pretokenizer, FastQwen2Pretokenizer, FastR50kPretokenizer,
};

/// Which pretokenization scheme (regex) a tokenizer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokenizerType {
    GPT2, // Also used by llama, also known as r50k
    GPT4, // cl100k
    Qwen2,      // Slightly adapted from GPT4, also used by Qwen3
    Olmo3,      // dolma2: Qwen2 scheme with cl100k's \p{N}{1,3}; used by Olmo 2/3
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
            PretokenizerType::Olmo3 => {
                FastPretokenizerDispatch::Olmo3(FastOlmo3Pretokenizer::new(bytes))
            }
            PretokenizerType::DeepSeekV3 => {
                unimplemented!("no fast pretokenizer for {self:?} yet")
            }
        }
    }

    /// Identify the scheme from the `Split` regex found in a HuggingFace
    /// `tokenizer.json` pre_tokenizer. Returns `None` for unknown patterns.
    pub fn from_split_regex(pattern: &str) -> Option<Self> {
        match pattern {
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::GPT2)
            }
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s+"
            | r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::GPT4)
            }
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::Qwen2)
            }
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::Olmo3)
            }
            _ => None,
        }
    }
}

/// Runtime-selected fast pretokenizer; add a variant here when implementing
/// a new scheme under `fast`.
pub enum FastPretokenizerDispatch<'a> {
    R50k(FastR50kPretokenizer<'a>),
    Cl100k(FastCl100kPretokenizer<'a>),
    Qwen2(FastQwen2Pretokenizer<'a>),
    Olmo3(FastOlmo3Pretokenizer<'a>),
}

impl<'a> Iterator for FastPretokenizerDispatch<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        match self {
            FastPretokenizerDispatch::R50k(it) => it.next(),
            FastPretokenizerDispatch::Cl100k(it) => it.next(),
            FastPretokenizerDispatch::Qwen2(it) => it.next(),
            FastPretokenizerDispatch::Olmo3(it) => it.next(),
        }
    }
}
