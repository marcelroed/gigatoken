use crate::bpe::Tokenizer;
use crate::pretokenize::PretokenizerType;
use eyre::{Context, Result, ensure};
use std::path::Path;

/// The base64-per-line mergeable ranks of a .tiktoken/tiktoken.model file,
/// in rank order (merges are reconstructed from this list).
fn load_tiktoken_ranks(file_path: impl AsRef<Path>) -> Result<Vec<Vec<u8>>> {
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::prelude::*;
    use std::io::Read;
    let mut buf = String::new();
    std::fs::File::open(&file_path)
        .with_context(|| format!("Failed to read {}", file_path.as_ref().display()))?
        .read_to_string(&mut buf)?;
    buf.lines()
        .enumerate()
        .map(|(i, line)| {
            let (base64_token, id_str) = line
                .split_once(' ')
                .ok_or_else(|| eyre::eyre!("line {i} has no rank field"))?;
            let id = id_str.trim().parse::<u32>()?;
            ensure!(id == i as u32, "rank {id} at line {i}: ranks must be dense");
            Ok(BASE64_STANDARD.decode(base64_token)?)
        })
        .collect()
}

pub fn load_tiktoken(file_path: impl AsRef<Path>) -> Result<Tokenizer> {
    let rank_vocab = load_tiktoken_ranks(file_path)?;
    let n_ranks = rank_vocab.len() as u32;
    let mut tokenizer = Tokenizer::from_ranks(rank_vocab)?;
    // Tiktoken vocab files carry no special tokens; GPT-2-family vocabs
    // (gpt2/r50k) place <|endoftext|> at the id right after the mergeable
    // ranks. Register it so tiktoken- and tokenizer.json-loaded tokenizers
    // encode and decode identically.
    tokenizer.add_special_token(b"<|endoftext|>".to_vec(), n_ranks.into());
    Ok(tokenizer)
}

/// The fields of a HuggingFace tokenizer_config.json a tiktoken-rank repo
/// needs: the special tokens (there is no tokenizer.json to carry them).
#[derive(serde::Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: std::collections::BTreeMap<String, AddedTokenJson>,
}

#[derive(serde::Deserialize)]
struct AddedTokenJson {
    content: String,
}

/// Load a tokenizer from a tiktoken rank file plus a HuggingFace
/// `tokenizer_config.json` whose `added_tokens_decoder` carries the special
/// tokens — the layout of repos that ship no tokenizer.json (e.g. the
/// moonshotai Kimi/Moonlight line's `tiktoken.model`). The pretokenizer
/// scheme is passed by the caller: such repos define their split regex only
/// in remote code, so no shipped file can name it.
pub fn load_tiktoken_model(
    model_path: impl AsRef<Path>,
    config_path: impl AsRef<Path>,
    pretokenizer: PretokenizerType,
) -> Result<Tokenizer> {
    let rank_vocab = load_tiktoken_ranks(model_path)?;
    let n_ranks = rank_vocab.len() as u32;
    let mut tokenizer = Tokenizer::from_ranks(rank_vocab)?;
    tokenizer.set_pretokenizer_type(pretokenizer);
    let config_path = config_path.as_ref();
    let config: TokenizerConfigJson = sonic_rs::from_slice(
        &std::fs::read(config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?,
    )
    .map_err(|e| eyre::eyre!("Failed to parse {}: {e}", config_path.display()))?;
    for (id, token) in &config.added_tokens_decoder {
        let id = id.parse::<u32>().with_context(|| {
            format!("added_tokens_decoder id {id:?} in {}", config_path.display())
        })?;
        ensure!(
            id >= n_ranks,
            "added token {:?} (id {id}) overlaps the {n_ranks} mergeable ranks",
            token.content
        );
        tokenizer.add_special_token(token.content.clone().into_bytes(), id.into());
    }
    Ok(tokenizer)
}
