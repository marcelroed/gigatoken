use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::pretokenize::FastR50kPretokenizer;
use std::path::PathBuf;
use std::time::Instant;

mod common;
fn main() {
    common::allow_thp();
    // ENCODE_TOKENIZER overrides the tokenizer.json (e.g.
    // data/qwen3_tokenizer.json to bench the qwen2-scheme encode path);
    // encoding then runs through the scheme dispatch instead of the
    // hardcoded r50k pretokenizer.
    let tokenizer_override = std::env::var("ENCODE_TOKENIZER").ok().map(PathBuf::from);
    let tokenizer_path = tokenizer_override.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/gpt2_tokenizer.json")
    });
    eprintln!("Loading tokenizer from {tokenizer_path:?}...");
    let mut tokenizer = load_hf_bpe(&tokenizer_path).expect("Could not load tokenizer");

    let input = common::load_owt_input(None);
    let size_gb = input.len() as f64 / 1e9;
    // Encode the whole buffer in one pass (matches real usage; the pretokenizer
    // handles newlines itself, so pre-splitting into lines is unnecessary).
    let buf: &[u8] = &input;

    // ENCODE_PASSES=N re-encodes the same buffer N times; passes after the
    // first run with a fully warm pretoken cache, isolating the hit path.
    let passes: usize = std::env::var("ENCODE_PASSES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    eprintln!("Encoding (single-threaded)...");
    // Flat output buffer, reserved once from the batch engine's
    // bytes-per-token estimate (see batch::encode_chunk) and reused across
    // passes; the token count is its length.
    let mut out: Vec<u32> = Vec::with_capacity(input.len() / 4 + 16);
    for pass in 0..passes {
        out.clear();
        let start = Instant::now();
        if tokenizer_override.is_some() {
            let pretokens = tokenizer.pretokenizer_type().pretokenize(buf);
            tokenizer.memoized_encode_flat(pretokens, &mut out);
        } else {
            tokenizer.memoized_encode_flat(FastR50kPretokenizer::new(buf), &mut out);
        }
        let total_tokens = out.len();
        let elapsed = start.elapsed().as_secs_f64();
        let throughput_gb = size_gb / elapsed;

        eprintln!(
            "pass {pass}: {total_tokens} tokens in {elapsed:.2}s — {throughput_gb:.2} GB/s ({:.0} MB/s)",
            throughput_gb * 1000.0
        );
    }
    let (sl, sc, ll, lc, lkb, al, ac) = tokenizer.cache_mem_stats();
    eprintln!(
        "cache: short {sl} entries (cap {sc}), long {ll} (cap {lc}, {lkb} key bytes), arena {al} tokens (cap {ac})"
    );
}
