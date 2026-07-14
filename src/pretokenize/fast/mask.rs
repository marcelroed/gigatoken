//! Shared infrastructure for mask-scanner pretokenizers.
//!
//! A mask-scanner pretokenizer processes 64-byte batches: SIMD classifies
//! every byte, bitmask algebra derives "a token starts here" bits, and
//! `next()` pops one bit per token — no per-token dispatch branches, which
//! is what makes it ~2x the serial scalar scanners (see
//! pretokenizer_optimization_log.md step 15).
//!
//! A scheme plugs in two functions ([`MaskScheme`]):
//! - `advance`: the scalar ground truth (also the no-SIMD iterator),
//! - `batch_masks`: `(usable, bad)` bitmasks for a 64-byte batch. `usable`
//!   bits are trustworthy token starts; `bad` marks zones (non-ASCII the
//!   scheme doesn't classify in-mask, batch-edge ambiguities) that
//!   [`MaskState`] re-derives through `advance`, never emitting a token
//!   across an unresolved zone.
//!
//! Layering, bottom to top:
//! 1. Platform SIMD primitives (`movemask64`, `ascii_masks` on NEON;
//!    `ascii_masks_avx512` / `ascii_masks_avx2` on x86-64) — the only
//!    per-platform code.
//! 2. Bit-domain helpers shared across schemes — platform-independent
//!    u64 algebra and per-char table classification
//!    (`classify_uni_chars`, `char_through`, `nn_at_full`,
//!    `digit_run_splits3`), parameterized by each scheme's codepoint
//!    classifier.
//! 3. Per-scheme `batch_masks` boundary algebra (in the scheme's module).
//! 4. [`MaskState`] — the scheme-agnostic batch walker: segments, bad-zone
//!    gaps, scalar tail, one-batch-ahead precompute; scalar overruns stay
//!    on the 64-byte grid so the precompute survives them.
//! 5. [`MaskState::fill_spans_two_phase`] — the chunked pull the encode
//!    loop uses: the same masks and trust rules as `next_span`, but
//!    harvested a chunk at a time into a flat boundary buffer and emitted
//!    in a branch-free counted loop.

use crate::pretokenize::unicode::{self, CharClass};

// -----------------------------------------------------------------------
// Platform SIMD primitives: aarch64 NEON (compile-time, always present)
// and x86_64 AVX-512 or AVX2 (runtime-detected; scalar fallback
// otherwise).
// -----------------------------------------------------------------------

/// Does this x86_64 CPU have the full AVX-512 tier (Zen 4/5, Ice
/// Lake+)? Schemes dispatch their batch classifier on this: the AVX-512
/// front-end when true, the AVX2 one otherwise.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx512_scanner_available() -> bool {
    // std's feature cache makes this an atomic load + bit test after the
    // first call.
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// Does this x86_64 CPU have the AVX2 tier (Haswell+, all Zen)? The bit
/// features (BMI1/2, LZCNT, POPCNT) arrived with or before AVX2 on every
/// AVX2 CPU, but are detected explicitly since the boundary algebra's
/// codegen relies on them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_scanner_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// Is the SIMD mask scanner usable on this machine? aarch64 always has
/// NEON; x86_64 requires AVX-512 (Zen 4/5, Ice Lake+) or AVX2 (Haswell+,
/// Zen 1-3), detected at runtime. When this returns false, [`MaskState`]
/// runs every token through the scheme's scalar `advance`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    avx512_scanner_available() || avx2_scanner_available()
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    cfg!(target_arch = "aarch64")
}

// The x86-64 batch classifiers are annotated
// `#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]`
// (AVX-512 tier) or `#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]`
// (AVX2 tier). Besides the wide byte ops, the scalar-visible bit features
// (BMI1/2, LZCNT, POPCNT) are enabled so the boundary algebra inlined
// into those functions compiles to tzcnt/lzcnt/blsr instead of
// baseline-x86 bsf sequences. The sets must stay in sync with
// [`avx512_scanner_available`] / [`avx2_scanner_available`].

/// simdjson-style movemask: 4 mask vectors (64 lanes of 0x00/0xFF) -> u64,
/// bit i = lane i.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn movemask64(
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    v3: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let mut a0 = vandq_u8(v0, w);
        let a1 = vandq_u8(v1, w);
        let mut a2 = vandq_u8(v2, w);
        let a3 = vandq_u8(v3, w);
        // The 4-`addp` reduction tree (simdjson's arm64 movemask), pinned
        // as asm. Written with `vpaddq_u8`, LLVM rewrites every pairwise
        // add into a uzp1/uzp2/orr triple — adjacent weighted lanes have
        // disjoint bits, so add == or, and the canonical or-form never
        // re-forms addp — inflating each call from 9 to 17 vector ops
        // (4-7 calls per 64-byte batch across the schemes). The weighted
        // `and`s stay outside so the scheduler still interleaves
        // neighboring calls. `addp(x, x)` lane 0..7 equals the old
        // `addp(x, zero)` lanes 0..7; only lane u64 0 is read.
        core::arch::asm!(
            "addp {a0:v}.16b, {a0:v}.16b, {a1:v}.16b",
            "addp {a2:v}.16b, {a2:v}.16b, {a3:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a2:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a0:v}.16b",
            a0 = inout(vreg) a0,
            a1 = in(vreg) a1,
            a2 = inout(vreg) a2,
            a3 = in(vreg) a3,
            options(pure, nomem, nostack, preserves_flags),
        );
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
    }
}

/// One u64 mask (bit i = byte scan+i) per byte predicate, for 64 bytes.
/// The working currency of scheme boundary algebra: everything after this
/// is platform-independent u64 bit math.
#[derive(Clone, Copy, Default)]
pub(crate) struct AsciiMasks {
    /// ASCII letters.
    pub l: u64,
    /// ASCII digits.
    pub d: u64,
    /// Space (0x20) only.
    pub s: u64,
    /// Non-newline ASCII whitespace: \t, \x0b, \x0c.
    pub wt: u64,
    /// Newlines: \r, \n.
    pub n: u64,
    /// Non-ASCII bytes (>= 0x80).
    pub hi: u64,
    /// ASCII apostrophes.
    pub ap: u64,
}

/// Classify `bytes[scan..scan+64]` (requires `scan + 64 <= bytes.len()`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ascii_masks(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::aarch64::*;
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let mut l = [vdupq_n_u8(0); 4];
        let mut d = [vdupq_n_u8(0); 4];
        let mut s = [vdupq_n_u8(0); 4];
        let mut wt = [vdupq_n_u8(0); 4];
        let mut n = [vdupq_n_u8(0); 4];
        let mut hi = [vdupq_n_u8(0); 4];
        let mut ap = [vdupq_n_u8(0); 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            l[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            d[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            s[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            n[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\r')),
                vceqq_u8(v, vdupq_n_u8(b'\n')),
            );
            // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
            wt[i] = vbicq_u8(
                vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)),
                n[i],
            );
            hi[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            ap[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }
        AsciiMasks {
            l: movemask64(l[0], l[1], l[2], l[3]),
            d: movemask64(d[0], d[1], d[2], d[3]),
            s: movemask64(s[0], s[1], s[2], s[3]),
            wt: movemask64(wt[0], wt[1], wt[2], wt[3]),
            n: movemask64(n[0], n[1], n[2], n[3]),
            hi: movemask64(hi[0], hi[1], hi[2], hi[3]),
            ap: movemask64(ap[0], ap[1], ap[2], ap[3]),
        }
    }
}

/// Classify `bytes[scan..scan+64]` with AVX-512 (requires
/// `scan + 64 <= bytes.len()`). One 64-byte load and one k-register
/// compare per predicate: a `__mmask64` IS the u64 the bit algebra wants,
/// so there is no movemask ladder and no lazy any-tests — every field
/// (including `hi` and `ap`) is computed unconditionally.
///
/// Runtime-gated: callers reach this only after
/// [`simd_scanner_available`] reported AVX-512 support (enforced by
/// [`MaskState`], which otherwise never leaves the scalar path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
pub(crate) fn ascii_masks_avx512(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(scan) as *const _);
        let lowered = _mm512_or_si512(v, _mm512_set1_epi8(0x20));
        let l = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(lowered, _mm512_set1_epi8(b'a' as i8)),
            _mm512_set1_epi8(25),
        );
        let d = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'0' as i8)),
            _mm512_set1_epi8(9),
        );
        let s = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b' ' as i8));
        let n = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\r' as i8))
            | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8));
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let wt = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(9)),
            _mm512_set1_epi8(4),
        ) & !n;
        let hi = _mm512_movepi8_mask(v) as u64;
        let ap = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\'' as i8));
        AsciiMasks { l, d, s, wt, n, hi, ap }
    }
}

/// Classify `bytes[scan..scan+64]` with AVX2 (requires
/// `scan + 64 <= bytes.len()`). Two 32-byte loads; each predicate is one
/// vector compare per half plus a `vpmovmskb` ladder into the u64 the bit
/// algebra wants — more mask-extraction traffic than the AVX-512 version
/// (whose k-register compares ARE the u64s), but the output currency is
/// identical, so everything downstream is shared. AVX2 has no unsigned
/// byte compare; `x <= lim` is `min_epu8(x, lim) == x`.
///
/// Runtime-gated: callers reach this only after
/// [`avx2_scanner_available`] reported AVX2 support (enforced by the
/// schemes' dispatch, behind [`MaskState`]'s `simd_scanner_available`
/// gate).
///
/// `#[inline(never)]` is load-bearing: inlined, LLVM's vector combiner
/// sees the compare vectors behind the returned u64s and pulls the
/// caller's scalar boundary algebra back into the byte-vector domain,
/// expanding every mask<->vector crossing into vpinsrb/vpextrb ladders
/// (~240 byte ops per batch, measured 3.5x slower end to end on Zen 2).
/// The AVX-512 tier has no such domain to return to (k-register compares
/// ARE the u64s), so it stays inline.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub(crate) fn ascii_masks_avx2(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        // Closures inherit the enclosing fn's target features.
        let le = |v: __m256i, lim: __m256i| -> __m256i {
            _mm256_cmpeq_epi8(_mm256_min_epu8(v, lim), v)
        };
        let mm = |m0: __m256i, m1: __m256i| -> u64 {
            (_mm256_movemask_epi8(m0) as u32 as u64)
                | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };

        let p = bytes.as_ptr().add(scan);
        let v0 = _mm256_loadu_si256(p as *const _);
        let v1 = _mm256_loadu_si256(p.add(32) as *const _);

        let x20 = _mm256_set1_epi8(0x20);
        let ca = _mm256_set1_epi8(b'a' as i8);
        let c25 = _mm256_set1_epi8(25);
        let l = mm(
            le(_mm256_sub_epi8(_mm256_or_si256(v0, x20), ca), c25),
            le(_mm256_sub_epi8(_mm256_or_si256(v1, x20), ca), c25),
        );
        let c0 = _mm256_set1_epi8(b'0' as i8);
        let c9 = _mm256_set1_epi8(9);
        let d = mm(
            le(_mm256_sub_epi8(v0, c0), c9),
            le(_mm256_sub_epi8(v1, c0), c9),
        );
        let sp = _mm256_set1_epi8(b' ' as i8);
        let s = mm(_mm256_cmpeq_epi8(v0, sp), _mm256_cmpeq_epi8(v1, sp));
        let cr = _mm256_set1_epi8(b'\r' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        let n = mm(
            _mm256_or_si256(_mm256_cmpeq_epi8(v0, cr), _mm256_cmpeq_epi8(v0, lf)),
            _mm256_or_si256(_mm256_cmpeq_epi8(v1, cr), _mm256_cmpeq_epi8(v1, lf)),
        );
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let c4 = _mm256_set1_epi8(4);
        let wt = mm(
            le(_mm256_sub_epi8(v0, c9), c4),
            le(_mm256_sub_epi8(v1, c9), c4),
        ) & !n;
        let hi = mm(v0, v1); // vpmovmskb takes the sign bit directly
        let apc = _mm256_set1_epi8(b'\'' as i8);
        let ap = mm(_mm256_cmpeq_epi8(v0, apc), _mm256_cmpeq_epi8(v1, apc));
        AsciiMasks { l, d, s, wt, n, hi, ap }
    }
}

// -----------------------------------------------------------------------
// Bit-domain helpers (platform-independent)
// -----------------------------------------------------------------------

/// Is the char starting at `idx` NOT whitespace (`\S` for a `(?!\S)`
/// lookahead)? Full answer via the packed table; the caller guarantees
/// the whole char is in bounds (a `scan + 70 <= len` batch guard covers
/// every call site's worst case).
#[inline(always)]
pub(crate) fn nn_at_full(bytes: &[u8], idx: usize) -> bool {
    use super::{decode_cp_inbounds, is_ascii_ws};
    let b = bytes[idx];
    if b < 0x80 {
        return !is_ascii_ws(b);
    }
    // SAFETY: caller guarantees the whole char in bounds (scan + 70
    // <= len batch guard), so idx + 4 <= len for any lead here.
    let (cp, _) = unsafe { decode_cp_inbounds(bytes, idx) };
    unicode::class_of(cp) != CharClass::Whitespace
}

/// The char containing byte `pos - 1` (`pos > 0`, valid UTF-8): its
/// class, lead index, and end (exclusive). `end > pos` iff the char
/// straddles across `pos`. ASCII classifies with the byte predicates;
/// multi-byte chars walk back to their lead (at most 3 bytes) and use
/// the packed table — this is what lets a batch after a unicode char
/// compute true boundary carries instead of deferring to a bad zone.
/// `class`: the scheme's codepoint classifier (`unicode::class_of`, or a
/// mark-folding view like `unicode::class_of_marks_join`).
#[inline(always)]
pub(crate) fn char_through(
    bytes: &[u8],
    pos: usize,
    class: impl Fn(u32) -> CharClass,
) -> (CharClass, usize, usize) {
    use super::{decode_cp_inbounds, is_ascii_ws, is_digit, is_letter};
    let b = bytes[pos - 1];
    if b < 0x80 {
        let cls = if is_letter(b) {
            CharClass::Letter
        } else if is_digit(b) {
            CharClass::Number
        } else if is_ascii_ws(b) {
            CharClass::Whitespace
        } else {
            CharClass::Other
        };
        return (cls, pos - 1, pos);
    }
    let mut j = pos - 1;
    while j > 0 && bytes[j] & 0xC0 == 0x80 {
        j -= 1;
    }
    // SAFETY: caller guarantees `pos` is inside a batch with the
    // scan + 70 <= len lookahead guard and j < pos, so j + 4 <= len.
    let (cp, l) = unsafe { decode_cp_inbounds(bytes, j) };
    (class(cp), j, j + l)
}

/// Per-byte class masks for a batch's unicode chars, classified with the
/// packed table (`unicode::class_of`) — the same lookup the scalar paths
/// do. Every byte of a classified char carries the char's class, so
/// byte-adjacency == char-adjacency and the schemes' u64 boundary
/// algebra applies unchanged.
#[derive(Clone, Copy, Default)]
pub(crate) struct UniClasses {
    /// Letter / number / other / whitespace bytes.
    pub l: u64,
    pub n: u64,
    pub o: u64,
    pub ws: u64,
    /// Whitespace lead bits by char length, for the char-length-aware
    /// `(?!\S)` shift tests. Deferred ws chars (see `resid`) are not
    /// included.
    pub w2: u64,
    pub w3: u64,
    /// Lead bits of all classified chars by length, for schemes that
    /// shift a test by the previous char's length (the cl100k family's
    /// two-chars-back rule).
    pub lead2: u64,
    pub lead3: u64,
    pub lead4: u64,
    /// Continuation bytes of classified chars.
    pub cont: u64,
    /// Bytes only the scalar path can decide: whitespace chars straddling
    /// the batch end (their run-split bookkeeping crosses the boundary),
    /// number chars when `NUMBERS` is false, and stray continuation
    /// bytes. Class masks stay truthful for these bytes so neighbors'
    /// algebra is exact; callers turn `resid` into bad zones (±1 smear).
    pub resid: u64,
}

/// Classify every unicode char whose lead bit is in `m` (typically
/// `hi & !claimed-straddle-in-bytes`) for `bytes[scan..scan+64]`.
/// Requires `scan + 70 <= len` (lookahead for a 4-byte char led at bit
/// 63). A char spilling off the batch end is classified via that
/// lookahead; only its in-batch bytes get class bits, and the next
/// batch's `char_through` walk-back covers the remainder. `NUMBERS`:
/// false for schemes whose digit grouping is char-counted (`\p{N}{1,3}`
/// byte masks can't express multi-byte chars), true otherwise.
/// `LEADS`: whether to fill the per-length lead masks (only schemes with
/// a shift-by-prev-char-length rule need them).
///
/// The loop stays branchy on purpose: a branchless csel-selected
/// decode/classify body measured 0.986x (predicted branches beat data
/// chains, log step 13/17).
#[inline(always)]
pub(crate) fn classify_uni_chars<const NUMBERS: bool, const LEADS: bool>(
    bytes: &[u8],
    scan: usize,
    mut m: u64,
    class: impl Fn(u32) -> CharClass,
) -> UniClasses {
    use super::decode_cp_inbounds;
    let mut u = UniClasses::default();
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        let b = bytes[scan + i];
        if b < 0xC2 {
            u.resid |= 1 << i; // stray continuation byte (invalid UTF-8)
            continue;
        }
        let l = if b < 0xE0 {
            2
        } else if b < 0xF0 {
            3
        } else {
            4
        };
        let chm = ((1u64 << l) - 1) << i; // in-batch bytes (excess drops)
        let lead = 1u64 << i;
        // SAFETY: scan + 70 <= len (this fn's contract), i <= 63, so
        // scan + i + 4 <= len even for a 4-byte lead at bit 63.
        let (cp, _) = unsafe { decode_cp_inbounds(bytes, scan + i) };
        match class(cp) {
            CharClass::Letter => u.l |= chm,
            CharClass::Number => {
                u.n |= chm;
                if !NUMBERS {
                    u.resid |= chm;
                }
            }
            CharClass::Other => u.o |= chm,
            CharClass::Whitespace => {
                u.ws |= chm;
                if i + l > 64 || l == 4 {
                    // Straddling-out ws (and defensively: no 4-byte cp
                    // is ws in Unicode) stays a bad zone; its true class
                    // marks keep neighbors' `(?!\S)` tests exact.
                    u.resid |= chm;
                } else if l == 2 {
                    u.w2 |= lead;
                } else {
                    u.w3 |= lead;
                }
            }
        }
        if LEADS {
            match l {
                2 => u.lead2 |= lead,
                3 => u.lead3 |= lead,
                _ => u.lead4 |= lead,
            }
        }
        u.cont |= chm & !lead;
        m &= !chm;
    }
    u
}

/// Token-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run
/// splits into 3-char tokens, so boundaries sit at run start + 3k. (For a
/// plain `\p{N}` scheme every digit is a start — no helper needed.)
#[inline(always)]
pub(crate) fn digit_run_splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1); // run starts
    // A start at p re-arms at p+3 while the run continues: hop condition
    // c = "p..p+3 all digits". Log-doubling covers 64-bit runs in 5 steps.
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

// -----------------------------------------------------------------------
// The batch walker
// -----------------------------------------------------------------------

/// The two per-scheme hooks of a mask-scanner pretokenizer.
pub(crate) trait MaskScheme {
    /// Scalar ground truth: end of the token starting at `pos`
    /// (`pos < bytes.len()`, `pos` on a token boundary).
    fn advance(bytes: &[u8], pos: usize) -> usize;

    /// `(usable, bad)` for `bytes[scan..scan+64]` (`scan+64 <= len`):
    /// `usable` bit k = trustworthy token start at scan+k; `bad` bit k =
    /// byte scan+k needs the scalar path. `usable & bad` must be 0.
    ///
    /// On x86_64 the implementations dispatch into AVX-512 or AVX2 code
    /// and must only be called when [`simd_scanner_available`] is true —
    /// [`MaskState`] guarantees this by never leaving the scalar path
    /// otherwise.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64);
}

/// Scheme-agnostic mask-scanner state: pops trusted boundary bits, walks
/// bad zones through the scheme's scalar `advance`, runs the buffer tail
/// scalar, and precomputes one batch ahead so the SIMD chain retires under
/// the previous batch's pops. Without SIMD support (non-aarch64/x86_64
/// targets, or an x86_64 CPU without AVX-512 or AVX2) `scalar_until`
/// starts at `usize::MAX`, so every token takes the scalar path.
pub(crate) struct MaskState {
    /// Start of the pending (not yet emitted) token.
    pub pos: usize,
    /// Base of the next batch to scan.
    scan: usize,
    /// Base the `rem`/`batch_*` bits refer to.
    mask_base: usize,
    /// Boundary bits of the current segment (trusted, pop-ready).
    rem: u64,
    /// Full usable mask of the current batch (later segments).
    batch_usable: u64,
    /// Bad zones of the current batch not yet passed.
    batch_bad: u64,
    /// Emit tokens via the scalar advance while `pos < scalar_until`.
    scalar_until: usize,
    /// Eagerly computed masks for the batch at `pre_base` (usize::MAX =
    /// none).
    pre_base: usize,
    pre_usable: u64,
    pre_bad: u64,
}

impl MaskState {
    #[inline]
    pub(crate) fn new(pos: usize) -> Self {
        let scalar_until = if simd_scanner_available() { pos } else { usize::MAX };
        Self {
            pos,
            scan: pos,
            mask_base: pos,
            rem: 0,
            batch_usable: 0,
            batch_bad: 0,
            scalar_until,
            pre_base: usize::MAX,
            pre_usable: 0,
            pre_bad: 0,
        }
    }

    /// Load the segment of `batch_usable` bits in [from_bit, next bad run)
    /// into `rem` and aim `scalar_until` past that bad run at the next
    /// trusted boundary (or the batch end).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    fn load_segment(&mut self, from_bit: u32) {
        let live = u64::MAX << from_bit;
        let seg_bad = self.batch_bad & live;
        if seg_bad == 0 {
            self.rem = self.batch_usable & live;
            self.batch_bad = 0;
        } else {
            let nb = seg_bad.trailing_zeros();
            self.rem = self.batch_usable & live & ((1u64 << nb) - 1);
            let rest = self.batch_usable & (u64::MAX << nb);
            self.scalar_until = if rest != 0 {
                self.mask_base + rest.trailing_zeros() as usize
            } else {
                self.mask_base + 64
            };
        }
        // A bit at the pending token's own start is not an end. Branchless:
        // whether the pending token starts exactly at this segment's first
        // bit is a ~20% coin flip on natural text.
        let at_start = self.pos == self.mask_base + from_bit as usize;
        self.rem &= !(u64::from(at_start) << from_bit);
    }

    /// The next token's byte range, or None at end of input.
    #[inline(always)]
    pub(crate) fn next_span<S: MaskScheme>(&mut self, bytes: &[u8]) -> Option<(usize, usize)> {
        let len = bytes.len();
        loop {
            if self.rem != 0 {
                let tz = self.rem.trailing_zeros() as usize;
                let end = self.mask_base + tz;
                self.rem &= self.rem - 1;
                let start = self.pos;
                self.pos = end;
                return Some((start, end));
            }
            if self.pos < self.scalar_until {
                if self.pos >= len {
                    return None;
                }
                let start = self.pos;
                let end = S::advance(bytes, start);
                self.pos = end;
                return Some((start, end));
            }
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            {
                // Continue with the current batch's next trusted segment
                // after a scalar gap (each batch is computed exactly once).
                if self.batch_bad != 0 && self.pos < self.mask_base + 64 {
                    self.load_segment((self.pos - self.mask_base) as u32);
                    continue;
                }
                self.batch_bad = 0;
                // Resume after a scalar overrun WITHOUT leaving the
                // 64-byte grid: the precomputed next batch (and the
                // prefetch chain behind it) stays valid, where rebasing
                // to the token boundary invalidated it on every bad-zone
                // overrun — a large part of a deferral's ~800-cycle
                // cost. Grid bits below `pos` may be stale run-internal
                // bits (a ws or digit run the scalar walked through can
                // cross the grid base); they are masked by the
                // `from_bit` passed to load_segment below, and every
                // path that puts `pos` inside such a run goes through a
                // deferral first, so those bits are never trusted.
                while self.scan + 64 <= self.pos {
                    self.scan += 64;
                }
                if self.scan + 64 > len {
                    // Tail: scalar to the end of the buffer.
                    self.scalar_until = usize::MAX;
                    continue;
                }
                let (usable, bad) = if self.pre_base == self.scan {
                    (self.pre_usable, self.pre_bad)
                } else {
                    S::batch_masks(bytes, self.scan)
                };
                self.mask_base = self.scan;
                self.scan += 64;
                self.batch_usable = usable;
                self.batch_bad = bad;
                // Kick off the next batch now; its SIMD chain overlaps this
                // batch's pops instead of stalling the next refill. Also
                // done for dirty batches: a scalar overrun past the batch
                // end just leaves the precompute unused (`pre_base` misses),
                // while gaps that resolve inside the batch — the common
                // case — keep the pipeline primed. Dirty batches used to
                // skip this, and paying the whole SIMD chain latency at the
                // next refill was a large part of their ~270-cycle cost.
                if self.scan + 64 <= len {
                    let (u2, b2) = S::batch_masks(bytes, self.scan);
                    self.pre_base = self.scan;
                    self.pre_usable = u2;
                    self.pre_bad = b2;
                } else {
                    self.pre_base = usize::MAX;
                }
                // An overrun may have left `pos` inside this grid batch;
                // start from its bit so stale bits below never pop. The
                // no-overrun case keeps the constant argument (and its
                // folded codegen) — schemes with few bad zones take that
                // branch essentially always.
                if self.pos > self.mask_base {
                    self.load_segment((self.pos - self.mask_base) as u32);
                } else {
                    self.load_segment(0);
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                // Unreachable: scalar_until is usize::MAX on this arch.
                self.scalar_until = usize::MAX;
            }
        }
    }
}

// -----------------------------------------------------------------------
// Two-phase chunked span fill
// -----------------------------------------------------------------------

/// Set-bit positions of a byte, packed in 8 u16 lanes (unused lanes 0,
/// never read). 4 KB, L1-resident alongside the unicode class table.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static BIT_POS: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut b = 1usize;
    while b < 256 {
        let mut j = 0;
        let mut w = 0;
        while j < 8 {
            if b >> j & 1 == 1 {
                t[b][w] = j as u16;
                w += 1;
            }
            j += 1;
        }
        b += 1;
    }
    t
};

/// Append the set-bit positions of `m`, offset by `rel` (wrapping), to
/// `out[0..popcount]` with no data-dependent branch: 8 fixed iterations,
/// one unconditional 8-lane store each at that octet's exclusive-prefix
/// popcount, so 64 bits cost the same straight-line code regardless of
/// population. Scribbles up to `out[popcount + 7]`; callers reserve the
/// slack. Returns the popcount.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn flatten_bits(m: u64, rel: u16, out: *mut u16) -> usize {
    // Per-octet popcounts (SWAR); one multiply turns them into inclusive
    // prefix sums, and a byte shift makes them exclusive write offsets —
    // the 8 stores below are mutually independent.
    let mut x = m;
    x -= (x >> 1) & 0x5555_5555_5555_5555;
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    let incl = x.wrapping_mul(0x0101_0101_0101_0101);
    let excl = incl << 8;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        for j in 0..8 {
            let b = (m >> (8 * j)) as u8 as usize;
            let w = (excl >> (8 * j)) as u8 as usize;
            let v = vld1q_u16(BIT_POS[b].as_ptr());
            let v = vaddq_u16(v, vdupq_n_u16(rel.wrapping_add(8 * j as u16)));
            vst1q_u16(out.add(w), v);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        for j in 0..8 {
            let b = (m >> (8 * j)) as u8 as usize;
            let w = (excl >> (8 * j)) as u8 as usize;
            let e = &BIT_POS[b];
            let base = rel.wrapping_add(8 * j as u16);
            // Fixed 8-lane copy: autovectorizes to one 16-byte store.
            for t in 0..8 {
                out.add(w + t).write(e[t].wrapping_add(base));
            }
        }
    }
    (incl >> 56) as usize
}

/// `pack_mask_halves(m)` for each clamped length `m` in 1..=15 (entry 0
/// unused), as one 16-byte row so the phase-B emission loop loads both
/// halves with a single `ldp`. That loop is issue-width-bound (~34
/// instructions/span before, at 4 stores + 2 loads it is nowhere near
/// the load/store port limits), so trading the 7-op per-half shift/select
/// chain for 1 always-L1-hot load (256 B, 4 lines) is a straight
/// instruction-count cut. The ALU form stays in `pack_mask_halves` for
/// the latency-chained per-span paths, where a dependent load on the
/// `n -> key -> store` chain measured 2.43% of process.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
static PACK_MASK_TABLE: [[u64; 2]; 16] = {
    let mut t = [[0u64; 2]; 16];
    let mut n = 1;
    while n <= 15 {
        let s = (n * 8) as u32;
        t[n][0] = if n < 8 { u64::MAX >> (64 - s) } else { u64::MAX };
        t[n][1] = if n > 8 { u64::MAX >> (128 - s) } else { 0 };
        n += 1;
    }
    t
};

/// Boundary scratch of one fill: PRETOKEN_CHUNK live entries, one batch of
/// overshoot from the last harvested batch (64 in-batch boundaries plus one
/// scalar-overrun end), and [`flatten_bits`]' 8-lane scribble slack, with
/// margin.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const BOUND_BUF: usize = crate::pretokenize::PRETOKEN_CHUNK + 144;

/// Boundary offsets are u16-relative to the fill base; a batch is only
/// harvested while every position it can contribute (base + 63, or the
/// tail's `len`, both < base + 64) still fits.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const REL_LIMIT: isize = u16::MAX as isize - 127;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl MaskState {
    /// Two-phase `fill_spans_keyed` body: phase A harvests one chunk's
    /// boundary positions into a flat buffer (branchless [`flatten_bits`]
    /// per clean batch; the scheme's scalar `advance` through bad zones,
    /// with `next_span`'s exact segment/overrun/tail trust rules), then
    /// phase B turns consecutive boundary pairs into batch entries
    /// in a counted loop with no data-dependent branch. The per-span
    /// refill ladder, pop-exit and pack mispredicts of the fused
    /// `next_span` loop (the dominant share of encode's 25% discarded
    /// issue bandwidth) collapse into one predictable branch per 64-byte
    /// batch.
    ///
    /// Boundary sets are identical to `next_span`'s by construction: the
    /// same `batch_masks` bits, the same scalar re-derivation for bad
    /// zones. Leftover boundaries past the chunk are discarded and `scan`
    /// rewound to the grid batch containing `pos` — masks are pure
    /// functions of the bytes, so the ~1 recomputed batch per fill buys
    /// carry-free fills, and a later `next_span` (which only ever advances
    /// `scan`) cannot skip the discarded bits. All other fields are reset
    /// so iterator and chunked pulls compose in any order.
    ///
    /// Callers must ensure [`simd_scanner_available`] (the scheme's
    /// `batch_masks` is unsafe to call otherwise on x86_64).
    #[inline(always)]
    pub(crate) fn fill_spans_two_phase<'a, S: MaskScheme>(
        &mut self,
        bytes: &'a [u8],
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        use crate::pretokenize::{
            PRETOKEN_CHUNK, pack_pretoken_key, pretoken_key_hash,
        };
        debug_assert!(simd_scanner_available());
        let len = bytes.len();
        let mut pending = self.pos;
        let mut scan = self.scan;
        // Rewind onto the grid batch containing `pending` after iterator
        // pops (next_span keeps consumed batches' bits in `rem`, which
        // this path recomputes). The forward direction is normalized at
        // each refill below.
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        let mut n = 0usize;
        // Opaque table base: LLVM rematerializes the static's address as
        // an adrp+add pair inside the per-span emission loop (constant
        // addresses are "free to recompute" to the register allocator);
        // pinning it here keeps the loop at one indexed ldp per span.
        let pack_masks: *const [u64; 2] = std::hint::black_box(PACK_MASK_TABLE.as_ptr());

        'refill: while n < PRETOKEN_CHUNK && pending < len {
            // Skip grid batches wholly behind `pending` (a direct-emitted
            // long span or a dropped overrun end can leave `scan` far
            // back); keeps `resume - base <= 63` for every batch below.
            if pending >= scan + 64 {
                scan += 64 * ((pending - scan) / 64);
            }
            let fill_base = pending;
            let needed = PRETOKEN_CHUNK - n;
            let mut buf = [std::mem::MaybeUninit::<u16>::uninit(); BOUND_BUF];
            let bufp = buf.as_mut_ptr() as *mut u16;
            let mut nb = 0usize;
            // Boundary bits at or below `resume` are settled: the pending
            // token's own start, or stale run-internal bits behind a
            // scalar overrun (see next_span's grid-keeping comment).
            let mut resume = pending;
            let mut exhausted = false;
            // A scalar end past the u16 window: dropped and re-derived
            // next fill, unless it is the fill's first boundary (emitted
            // directly below).
            let mut overflow_end: Option<usize> = None;

            // Phase A: harvest boundary positions.
            'harvest: while nb < needed {
                if scan.wrapping_sub(fill_base) as isize > REL_LIMIT {
                    break; // re-base: offsets would leave the u16 window
                }
                if scan + 64 > len {
                    // Scalar tail to end of input.
                    let mut p = if nb > 0 {
                        fill_base + unsafe { *bufp.add(nb - 1) } as usize
                    } else {
                        fill_base
                    };
                    while p < len && nb < needed {
                        p = S::advance(bytes, p);
                        // p <= len < scan + 64, within the u16 window per
                        // the REL_LIMIT check above.
                        unsafe { bufp.add(nb).write((p - fill_base) as u16) };
                        nb += 1;
                    }
                    exhausted = p >= len;
                    break;
                }
                let base = scan;
                let (usable, bad) = S::batch_masks(bytes, base);
                // At a resume point r (the pending token's start): usable
                // bits at or below r are dead (the pending start itself,
                // or stale run-internal bits behind a scalar overrun), but
                // a bad bit AT r must stay live — load_segment's
                // `live = MAX << from_bit` plus the at_start clear. A zone
                // starting exactly at r has to route r through the scalar
                // path, or the stale post-zone usable bit would be
                // trusted. Only the fill's first batch and post-overrun
                // batches have such bits.
                let (mut ulive, mut blive) = if resume >= base {
                    debug_assert!(resume - base < 64);
                    let k = resume - base;
                    ((u64::MAX << k) << 1, u64::MAX << k)
                } else {
                    (u64::MAX, u64::MAX)
                };
                let rel = base.wrapping_sub(fill_base) as u16;
                if bad & blive == 0 {
                    debug_assert!(nb + 72 <= BOUND_BUF);
                    nb += unsafe { flatten_bits(usable & ulive, rel, bufp.add(nb)) };
                    scan = base + 64;
                    continue;
                }
                // Dirty batch: per segment, trusted prefix bits then the
                // scheme's scalar advance through the zone up to the next
                // trusted boundary — load_segment's rules, emitting into
                // the buffer.
                loop {
                    let seg_bad = bad & blive;
                    if seg_bad == 0 {
                        debug_assert!(nb + 72 <= BOUND_BUF);
                        nb += unsafe { flatten_bits(usable & ulive, rel, bufp.add(nb)) };
                        scan = base + 64;
                        break;
                    }
                    let fb = seg_bad.trailing_zeros();
                    let prefix = usable & ulive & !(u64::MAX << fb);
                    debug_assert!(nb + 72 <= BOUND_BUF);
                    nb += unsafe { flatten_bits(prefix, rel, bufp.add(nb)) };
                    let mut p = if nb > 0 {
                        fill_base + unsafe { *bufp.add(nb - 1) } as usize
                    } else {
                        fill_base
                    };
                    let rest = usable & (u64::MAX << fb);
                    let until = if rest != 0 {
                        base + rest.trailing_zeros() as usize
                    } else {
                        base + 64
                    };
                    // until <= base + 64 <= len, so `advance` stays in
                    // bounds; it may overrun `until` and the batch end.
                    while p < until {
                        p = S::advance(bytes, p);
                        let relp = p - fill_base;
                        if relp > u16::MAX as usize {
                            overflow_end = Some(p);
                            break 'harvest;
                        }
                        debug_assert!(nb < BOUND_BUF);
                        unsafe { bufp.add(nb).write(relp as u16) };
                        nb += 1;
                    }
                    if p >= base + 64 {
                        // Overrun past the batch: stay on the grid and
                        // resume in the batch containing p, bits at or
                        // below p masked (they can be stale run-internal
                        // bits, exactly as in next_span).
                        scan = base + 64 * ((p - base) / 64);
                        resume = p;
                        break;
                    }
                    // Resume inside the batch at p: same at-start/bad-bit
                    // split as the batch-entry masks above.
                    blive = u64::MAX << (p - base);
                    ulive = blive << 1;
                }
            }

            if nb == 0 {
                // No boundary inside the u16 window: a > 65 KB pretoken.
                // Emit it alone through the careful pack.
                debug_assert!(!exhausted);
                let end = overflow_end.unwrap_or_else(|| S::advance(bytes, fill_base));
                let span = &bytes[fill_base..end];
                let (key, h) = match pack_pretoken_key(span) {
                    Some(key) => (key, pretoken_key_hash(key)),
                    None => (0, 0),
                };
                prefetch(h);
                let meta = if key != 0 { h } else { span.len() as u64 };
                batch.entries[n] = crate::pretokenize::BatchEntry {
                    key,
                    ptr: span.as_ptr(),
                    meta,
                };
                n += 1;
                pending = end;
                continue 'refill;
            }

            // Phase B: flat emission with no data-dependent branch. One
            // hoisted check proves every 16-byte key load in-bounds of the
            // input slice; only a fill reaching within 16 bytes of EOF
            // routes through the careful per-span pack.
            let m = nb.min(needed);
            let last_end = unsafe { *bufp.add(m - 1) } as usize;
            let entries = &mut batch.entries[n..n + m];
            let base_ptr = unsafe { bytes.as_ptr().add(fill_base) };
            // `prev`/`end` in usize: the u16 boundary domain forced two
            // `& 0xffff` masks and a duplicated 15-compare per span (the
            // compiler cannot see end >= prev in u16 subtraction).
            let mut prev = 0usize;
            if fill_base + last_end + 16 <= len {
                for (i, e) in entries.iter_mut().enumerate() {
                    let end = unsafe { *bufp.add(i) } as usize;
                    let tok_len = end - prev;
                    let p = unsafe { base_ptr.add(prev) };
                    prev = end;
                    // SAFETY: p + 16 <= base_ptr + last_end + 16 <= end of
                    // the input slice (hoisted check above).
                    let raw = unsafe { (p as *const u128).read_unaligned() };
                    // Branchless pack_pretoken_key: one ldp from
                    // PACK_MASK_TABLE instead of the 7-op per-half ALU
                    // chain (see the table's docs). tok_len >= 1
                    // (boundaries are strictly increasing), so the clamped
                    // length is in the table's 1..=15 domain. Long spans
                    // take key 0 (pretoken_key_hash(0) == 0) through the
                    // `keep` AND-mask — an if/select here gets if-converted
                    // into a real branch (LLVM hoists it to skip the two
                    // loads), reintroducing the pattern-free n > 15 branch
                    // this loop exists to avoid.
                    let m = tok_len.min(15);
                    // SAFETY: m <= 15, in the 16-entry table.
                    let [mask_lo, mask_hi] = unsafe { *pack_masks.add(m) };
                    let keep = ((tok_len <= 15) as u64).wrapping_neg();
                    let klo = (raw as u64) & mask_lo & keep;
                    let khi = (((raw >> 64) as u64 & mask_hi) | ((m as u64) << 56)) & keep;
                    let key = (klo as u128) | ((khi as u128) << 64);
                    let hv = pretoken_key_hash(key);
                    prefetch(hv);
                    // meta = hash for short spans, length for long ones
                    // (see `BatchEntry::meta`), in the same AND-mask style
                    // as the key routing — a select gets if-converted.
                    let meta = (hv & keep) | (tok_len as u64 & !keep);
                    e.key = key;
                    e.ptr = p;
                    e.meta = meta;
                }
            } else {
                for (i, e) in entries.iter_mut().enumerate() {
                    let end = unsafe { *bufp.add(i) } as usize;
                    let tok_len = end - prev;
                    let p = unsafe { base_ptr.add(prev) };
                    prev = end;
                    // SAFETY: as above for the span bounds.
                    let span = unsafe { std::slice::from_raw_parts(p, tok_len) };
                    let (key, hv) = match pack_pretoken_key(span) {
                        Some(key) => (key, pretoken_key_hash(key)),
                        None => (0, 0),
                    };
                    prefetch(hv);
                    let meta = if key != 0 { hv } else { tok_len as u64 };
                    e.key = key;
                    e.ptr = p;
                    e.meta = meta;
                }
            }
            n += m;
            pending = fill_base + prev;
            if exhausted {
                debug_assert_eq!(pending, len);
                break;
            }
        }

        // Rewind past discarded leftover boundaries and leave the state as
        // a fresh resume at `pending` for either pull style.
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        self.pos = pending;
        self.scan = scan;
        self.mask_base = scan;
        self.rem = 0;
        self.batch_usable = 0;
        self.batch_bad = 0;
        self.scalar_until = pending;
        self.pre_base = usize::MAX;
        n
    }
}
