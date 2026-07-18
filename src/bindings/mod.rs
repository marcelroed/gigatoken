//! pyo3 glue for everything except the two tokenizer classes, which stay in
//! lib.rs so the main path of the API remains front and center. One
//! submodule per feature: Python<->Rust bridging shared by the bindings
//! (bridge), the FileSource/BytesSource classes (sources), BPE training (train), padded
//! compat-API encoding (padding), the compat layers' special-token scanner
//! (matcher), and the pretokenizer helpers (pretokenize).

pub(crate) mod bridge;
pub(crate) mod hub;
pub(crate) mod matcher;
pub(crate) mod padding;
pub(crate) mod pretokenize;
pub(crate) mod sources;
pub(crate) mod train;
