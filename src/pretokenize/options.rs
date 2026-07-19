use crate::pretokenize::Pretoken;
use crate::pretokenize::fast::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastKimiPretokenizer,
    FastNemotronPretokenizer, FastO200kPretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer,
};

/// Which pretokenization scheme (regex) a tokenizer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokenizerType {
    GPT2, // Also used by llama, also known as r50k
    GPT4, // cl100k
    Qwen2,      // Slightly adapted from GPT4, also used by Qwen3
    Qwen35,     // Qwen2 with `[\p{L}\p{M}]+` letter runs, marks excluded from punct runs
    Olmo3,      // dolma2: Qwen2 scheme with cl100k's \p{N}{1,3}; used by Olmo 2/3
    DeepSeekV3, // Sequence of three Splits (digits, CJK, main); used by DeepSeek V3/V3.1/V4
    O200k,      // o200k_base: case-structured letter runs; GPT-4o, gpt-oss
    Nemotron,   // o200k without contractions, single-digit \p{N}; nvidia Nemotron-3
    Kimi,       // o200k with [\p{Han}]+ runs and no `/` tail absorption; moonshotai Kimi-K2 line
}

/// The three Split regexes of the DeepSeek V3/V4 pre_tokenizer Sequence, as
/// they appear in tokenizer.json (the third contains literal CR/LF chars,
/// not `\r`/`\n` escapes).
const DEEPSEEK_V3_SPLIT_REGEXES: [&str; 3] = [
    r"\p{N}{1,3}",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

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
            PretokenizerType::Qwen35 => {
                FastPretokenizerDispatch::Qwen35(FastQwen35Pretokenizer::new(bytes))
            }
            PretokenizerType::Olmo3 => {
                FastPretokenizerDispatch::Olmo3(FastOlmo3Pretokenizer::new(bytes))
            }
            PretokenizerType::DeepSeekV3 => {
                FastPretokenizerDispatch::DeepSeekV3(FastDeepSeekV3Pretokenizer::new(bytes))
            }
            PretokenizerType::O200k => {
                FastPretokenizerDispatch::O200k(FastO200kPretokenizer::new(bytes))
            }
            PretokenizerType::Nemotron => {
                FastPretokenizerDispatch::Nemotron(FastNemotronPretokenizer::new(bytes))
            }
            PretokenizerType::Kimi => {
                FastPretokenizerDispatch::Kimi(FastKimiPretokenizer::new(bytes))
            }
        }
    }

    /// The scheme named by a lowercase identifier, as used by loaders whose
    /// source format carries no split regex (e.g. tiktoken-rank repos, whose
    /// regex lives in remote code) and the Python bindings. Accepts each
    /// variant's canonical name plus the common tiktoken aliases.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "gpt2" | "r50k" => PretokenizerType::GPT2,
            "gpt4" | "cl100k" => PretokenizerType::GPT4,
            "qwen2" => PretokenizerType::Qwen2,
            "qwen35" => PretokenizerType::Qwen35,
            "olmo3" => PretokenizerType::Olmo3,
            "deepseek_v3" => PretokenizerType::DeepSeekV3,
            "o200k" => PretokenizerType::O200k,
            "nemotron" => PretokenizerType::Nemotron,
            "kimi" => PretokenizerType::Kimi,
            _ => return None,
        })
    }

    /// Identify the scheme from the ordered list of `Split` regexes found in
    /// a HuggingFace `tokenizer.json` pre_tokenizer. Returns `None` for
    /// unknown patterns.
    pub fn from_split_regexes(patterns: &[&str]) -> Option<Self> {
        match patterns {
            [p] => Self::from_split_regex(p),
            _ if patterns == DEEPSEEK_V3_SPLIT_REGEXES => Some(PretokenizerType::DeepSeekV3),
            _ => None,
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
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::Qwen35)
            }
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::Olmo3)
            }
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::O200k)
            }
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
                Some(PretokenizerType::Nemotron)
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
    Qwen35(FastQwen35Pretokenizer<'a>),
    Olmo3(FastOlmo3Pretokenizer<'a>),
    DeepSeekV3(FastDeepSeekV3Pretokenizer<'a>),
    O200k(FastO200kPretokenizer<'a>),
    Nemotron(FastNemotronPretokenizer<'a>),
    Kimi(FastKimiPretokenizer<'a>),
}

impl<'a> Iterator for FastPretokenizerDispatch<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        match self {
            FastPretokenizerDispatch::R50k(it) => it.next(),
            FastPretokenizerDispatch::Cl100k(it) => it.next(),
            FastPretokenizerDispatch::Qwen2(it) => it.next(),
            FastPretokenizerDispatch::Qwen35(it) => it.next(),
            FastPretokenizerDispatch::Olmo3(it) => it.next(),
            FastPretokenizerDispatch::DeepSeekV3(it) => it.next(),
            FastPretokenizerDispatch::O200k(it) => it.next(),
            FastPretokenizerDispatch::Nemotron(it) => it.next(),
            FastPretokenizerDispatch::Kimi(it) => it.next(),
        }
    }
}

// SAFETY: pure delegation to the concrete pretokenizers' (contract-upholding)
// fills; no entries are written here.
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastPretokenizerDispatch<'a> {
    /// One dispatch per chunk instead of one per pretoken, delegating to
    /// the concrete pretokenizers' fused chunk fills.
    #[inline]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        use crate::pretokenize::PretokenSpans;
        match self {
            FastPretokenizerDispatch::R50k(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Cl100k(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Qwen2(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Qwen35(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Olmo3(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::DeepSeekV3(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::O200k(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Nemotron(it) => it.fill_spans_keyed(batch, prefetch),
            FastPretokenizerDispatch::Kimi(it) => it.fill_spans_keyed(batch, prefetch),
        }
    }
}

