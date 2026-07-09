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
//!    `ascii_masks_avx512` on x86-64) — the only per-platform code.
//! 2. Bit-domain helpers shared across schemes — platform-independent
//!    u64 algebra and per-char table classification
//!    (`classify_uni_chars`, `char_through`, `nn_at_full`,
//!    `digit_run_splits3`), parameterized by each scheme's codepoint
//!    classifier.
//! 3. Per-scheme `batch_masks` boundary algebra (in the scheme's module).
//! 4. [`MaskState`] — the scheme-agnostic batch walker: segments, bad-zone
//!    gaps, scalar tail, one-batch-ahead precompute; scalar overruns stay
//!    on the 64-byte grid so the precompute survives them.

use crate::pretokenize::unicode::{self, CharClass};

// -----------------------------------------------------------------------
// Platform SIMD primitives: aarch64 NEON (compile-time, always present)
// and x86_64 AVX-512 (runtime-detected; scalar fallback otherwise).
// -----------------------------------------------------------------------

/// Is the SIMD mask scanner usable on this machine? aarch64 always has
/// NEON; x86_64 requires AVX-512 (Zen 4/5, Ice Lake+), detected at
/// runtime — CPUs are assumed to have either the full Zen 5 feature set
/// or none of it, so there is no AVX2-only tier. When this returns
/// false, [`MaskState`] runs every token through the scheme's scalar
/// `advance`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
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

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    cfg!(target_arch = "aarch64")
}

// The AVX-512 batch classifiers are all annotated
// `#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]`.
// Besides the 512-bit byte ops (F/BW), the scalar-visible bit features
// (BMI1/2, LZCNT, POPCNT) are enabled so the boundary algebra inlined
// into those functions compiles to tzcnt/lzcnt/blsr instead of
// baseline-x86 bsf sequences. The set must stay in sync with
// [`simd_scanner_available`].

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
        let s0 = vpaddq_u8(vandq_u8(v0, w), vandq_u8(v1, w));
        let s1 = vpaddq_u8(vandq_u8(v2, w), vandq_u8(v3, w));
        let s = vpaddq_u8(vpaddq_u8(s0, s1), vdupq_n_u8(0));
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(s))
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

// -----------------------------------------------------------------------
// Bit-domain helpers (platform-independent)
// -----------------------------------------------------------------------

/// Is the char starting at `idx` NOT whitespace (`\S` for a `(?!\S)`
/// lookahead)? Full answer via the packed table; the caller guarantees
/// the whole char is in bounds (a `scan + 70 <= len` batch guard covers
/// every call site's worst case).
#[inline(always)]
pub(crate) fn nn_at_full(bytes: &[u8], idx: usize) -> bool {
    use super::{decode_cp, is_ascii_ws};
    let b = bytes[idx];
    if b < 0x80 {
        return !is_ascii_ws(b);
    }
    let (cp, _) = unsafe { decode_cp(bytes, idx) };
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
    use super::{decode_cp, is_ascii_ws, is_digit, is_letter};
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
    let (cp, l) = unsafe { decode_cp(bytes, j) };
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
    use super::decode_cp;
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
        let (cp, _) = unsafe { decode_cp(bytes, scan + i) };
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
    /// On x86_64 the implementations dispatch into AVX-512 code and must
    /// only be called when [`simd_scanner_available`] is true —
    /// [`MaskState`] guarantees this by never leaving the scalar path
    /// otherwise.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64);
}

/// Scheme-agnostic mask-scanner state: pops trusted boundary bits, walks
/// bad zones through the scheme's scalar `advance`, runs the buffer tail
/// scalar, and precomputes one batch ahead so the SIMD chain retires under
/// the previous batch's pops. Without SIMD support (non-aarch64/x86_64
/// targets, or an x86_64 CPU without AVX-512) `scalar_until` starts at
/// `usize::MAX`, so every token takes the scalar path.
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
