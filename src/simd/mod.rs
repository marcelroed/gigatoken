#[cfg(target_feature = "avx512vbmi")]
pub(crate) mod avx512;
#[cfg(target_feature = "avx512vbmi")]
pub(crate) use avx512::ascii;
#[cfg(target_feature = "avx512vbmi")]
pub(crate) use avx512::unicode;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod m_series;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use m_series::ascii;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use m_series::unicode;
