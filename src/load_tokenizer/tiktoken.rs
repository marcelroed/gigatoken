use crate::bpe::Tokenizer;
use eyre::{Context, Result};
use std::path::Path;

pub fn load_tiktoken(file_path: impl AsRef<Path>) -> Result<Tokenizer> {
    use crate::bpe::Tokenizer;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::prelude::*;
    use std::io::Read;
    let mut buf = String::new();
    std::fs::File::open(&file_path)
        .with_context(|| format!("Failed to read {:?}", file_path.as_ref()))?
        .read_to_string(&mut buf)?;

    let vocab: Vec<Vec<u8>> = buf
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let (base64_token, id_str) = line.split_once(' ').unwrap();
            let id = id_str.trim().parse::<u32>().unwrap();
            assert_eq!(id, i as u32);
            let token_bytes: Vec<u8> = BASE64_STANDARD.decode(base64_token).unwrap();
            token_bytes
        })
        .collect();

    // Reorder based on

    Tokenizer::from_ranks(vocab)
}
