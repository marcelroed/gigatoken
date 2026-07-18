//! Reference pretokenizers: benchmark baselines and test oracles only.
//!
//! Nothing in here runs in the library's encode path — the production
//! pretokenizers are the mask scanners in [`super::fast`] (see
//! `pretokenize_as_iter` / `PretokenizerType`). These implementations are
//! kept so `benches/pretokenize.rs` can race the production scanners
//! against the designs they replaced, and as independent oracles in
//! differential tests:
//!
//! - [`state_machine`]: the byte-class DFA, the original correctness
//!   reference (also exercised from other modules' tests).
//! - [`combinator`]: winnow parser-combinator implementation.
//! - [`simd`]: first-generation portable-SIMD prototype.
//! - [`avx512`]: hand-rolled AVX-512 prototype. Compile-time gated on
//!   AVX-512BW/VL (build with `-C target-cpu=native` or equivalent to
//!   include it) — as a baseline it is deliberately not runtime-dispatched,
//!   unlike the production scanners, which detect their tier at runtime on
//!   any build.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
pub mod avx512;
pub mod combinator;
pub mod simd;
pub mod state_machine;
