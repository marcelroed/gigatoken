//! Fast pretokenizer for the GPT-2 (r50k_base) regex:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! On aarch64 (NEON) and x86_64 with AVX-512 or AVX2 (runtime-detected)
//! the iterator runs a simdjson-style mask scanner: 64-byte
//! batches are classified with SIMD into per-byte u64 class masks, the
//! token-boundary bits are derived with shifted-mask algebra in scalar
//! registers (log step 17; the original vector-register algebra of step
//! 15 measured the same and was retired for the simpler form), and the
//! walker pops one bit per token — no per-token dispatch branches, which
//! sidesteps the ~8 cy/token branch-miss floor of the scalar scanner
//! (log step 13). Apostrophes get a contraction bit-fixup; batches with
//! any non-ASCII (~21% of OWT) take `extended_masks`, which classifies
//! every unicode char with the packed table so it joins the same
//! algebra. Chars straddling a batch edge are resolved with lookahead
//! and a prev-char walk-back, so bad zones (scalar re-derivation) remain
//! only for edge-straddling whitespace, contractions at the batch edge,
//! and invalid UTF-8 — ~0.4% of batches. Measured 2,460-2,600 MB/s on
//! 1 GB OWT (pretokenize_profile, min-of-N interleaved; 2,132 at step
//! 15, 983 for the scalar scanner).
//!
//! The scalar path (`advance_pos`, SWAR letter runs + arithmetic
//! predicates) remains the reference implementation, the no-SIMD
//! fallback, and the executor for bad zones and buffer tails.
//!
//! `advance_pos` is a pure free function (`(bytes, pos) -> end`) rather than
//! a `&mut self` method: keeping the cursor in a register instead of writing
//! `self.pos` at every scan step shortens the per-token dependency chain and
//! is worth ~30% throughput. Non-ASCII characters are classified with one
//! packed-table load (`unicode::class_of`) on a hand-rolled UTF-8 decode.
//!
//! A windowed multi-cursor variant (finding guaranteed boundaries and running
//! 2-4 independent `advance_pos` chains interleaved, queueing token ends) was
//! benchmarked at 0.80-0.95x of this streaming version: the queue traffic and
//! interleaved branch history cost more than the extra ILP recovers.

use super::mask::{self, MaskScheme, MaskState};
use super::{
    decode_cp, is_ascii_ws, is_digit, is_letter, scan_digits_from, scan_letters_from,
    scan_other_from,
};
use crate::pretokenize::unicode::{self, CharClass};
use crate::pretokenize::Pretoken;

// -----------------------------------------------------------------------
// FastR50kPretokenizer
// -----------------------------------------------------------------------

/// Boundary and bad-zone bitmasks for `bytes[scan..scan+64]` (requires
/// `scan + 64 <= bytes.len()`). Bit `k` of `usable` = a trustworthy token
/// start at `scan + k`; `bad` marks bytes whose boundaries must be
/// re-derived by `advance_pos`, and no token may be emitted across an
/// unresolved bad zone.
///
/// NEON classifies the ASCII classes (letter, digit, space, whitespace:
/// 4 movemasks; apostrophe and non-ASCII behind horizontal any-tests)
/// and the boundary bits come from u64 shifted-mask algebra in scalar
/// registers, as in `cl100k_family::batch_masks`. Batches with any
/// non-ASCII byte (~21% on OWT, mostly curly quotes) take
/// [`extended_masks`]. Inlining that path here measured 0.98x, and
/// `#[inline(never)]` on this whole function 0.94x — the split keeps the
/// walker's register allocation clean (step 15's lesson) while the hot
/// ASCII algebra stays inline.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
    use std::arch::aarch64::*;
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut lv = [zero; 4];
        let mut dv = [zero; 4];
        let mut sv = [zero; 4];
        let mut wsv = [zero; 4];
        let mut hiv = [zero; 4];
        let mut apv = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            lv[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            dv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            sv[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            wsv[i] = vorrq_u8(
                sv[i],
                vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)),
            );
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
        }

        let lb = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let db = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        // Apostrophes only matter for the contraction fixup below.
        let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
        let ap64 = if vmaxvq_u8(ap_any) != 0 {
            mask::movemask64(apv[0], apv[1], apv[2], apv[3])
        } else {
            0
        };

        // Any non-ASCII byte routes to the extended classifier, which
        // reuses the ASCII masks computed above.
        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0 {
            let hi64 = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return extended_masks(bytes, scan, lb, db, s64, wsa, hi64, ap64);
        }

        ascii_batch_algebra(bytes, scan, lb, db, s64, wsa, ap64)
    }
}

/// x86 counterpart of the NEON `batch_masks`, monomorphized on the SIMD
/// tier: the classification is [`mask::ascii_masks_avx512`] (one 64-byte
/// load and one k-register compare per class) or
/// [`mask::ascii_masks_avx2`] (two 32-byte loads, compare + vpmovmskb per
/// class); the boundary algebra and the extended (non-ASCII) path are the
/// same shared scalar code. `#[inline(always)]` (with no `target_feature`
/// of its own) so the body fuses into whichever feature region calls it —
/// the tier-monomorphized fill wrappers
/// (`MaskState::fill_spans_two_phase_{avx512,avx2}_crc`), where LLVM's
/// cost model declined to inline the previous `#[target_feature]` form
/// and left a call per 64-byte batch, or the runtime-dispatched
/// `MaskScheme::batch_masks` that `next_span` uses.
///
/// # Safety
///
/// The selected tier must have been runtime-detected
/// ([`mask::avx512_scanner_available`] /
/// [`mask::avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification
        // (up to a 4-byte char starting at scan + 66); scalar batch.
        return (0, u64::MAX);
    }
    let am = if AVX512 {
        // SAFETY: the caller detected the AVX-512 tier (fn contract).
        unsafe { mask::ascii_masks_avx512(bytes, scan) }
    } else {
        // SAFETY: the caller detected the AVX2 tier (fn contract).
        unsafe { mask::ascii_masks_avx2(bytes, scan) }
    };
    let wsa = am.s | am.wt | am.n;
    if am.hi != 0 {
        // SAFETY: both detected tiers include the BMI1/BMI2/LZCNT/POPCNT
        // bit features `extended_masks` re-declares (fn contract).
        return unsafe { extended_masks(bytes, scan, am.l, am.d, am.s, wsa, am.hi, am.ap) };
    }
    ascii_batch_algebra(bytes, scan, am.l, am.d, am.s, wsa, am.ap)
}

/// Pure-ASCII boundary algebra shared by the NEON and AVX-512 batch
/// classifiers (the batch has no non-ASCII byte; `wsa` = all ASCII
/// whitespace). Everything here is platform-independent u64 bit math.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_batch_algebra(
    bytes: &[u8],
    scan: usize,
    lb: u64,
    db: u64,
    s64: u64,
    wsa: u64,
    ap64: u64,
) -> (u64, u64) {
    let ob = !(lb | db | wsa); // hi == 0 on this path

    // Bit-0 carries from the char before the batch. This batch is
    // pure ASCII, so a multi-byte prev char always ends exactly at
    // the boundary and the walk-back gives true carries — no bad
    // zone.
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else {
        carries_at(bytes, scan)
    };

    let cont_same =
        (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsa & !cont_same & !after_sp;

    // Ws-run split (`\s+(?!\S)`); bit 63 needs the real lookahead
    // char. The ASCII case is branchless — "is byte 63 ws" is a
    // ~20% coin flip on natural text, so testing it costs a
    // mispredict every few batches. Only a non-ASCII lookahead
    // byte (rare) branches, for the table-backed ws check.
    let mut split_ok = wsa & (!wsa >> 1); // bit 63: shifted-in 0
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 {
        split_ok |= (u64::from(!is_ascii_ws(nb64)) << 63) & wsa;
    } else if wsa >> 63 != 0
        // SAFETY: this classifier's scan + 70 <= len batch guard puts the
        // decode at scan + 64 in bounds (needs scan + 68 <= len).
        && unsafe { mask::nn_at_full(bytes, scan + 64) }
    {
        split_ok |= 1 << 63;
    }
    let pwsb = (wsa << 1) | pws;
    let wsboundary = wsa & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = 0u64;

    // Contraction fixup (see extended_masks for the rules).
    if ap64 != 0 {
        let mut cand = ap64 & boundary;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            let k = match bytes[scan + i + 1] {
                b's' | b'd' | b'm' | b't' => 2,
                b'l' if bytes[scan + i + 2] == b'l' => 3,
                b'v' if bytes[scan + i + 2] == b'e' => 3,
                b'r' if bytes[scan + i + 2] == b'e' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << (i + 1));
                boundary |= 1u64 << (i + k);
            }
        }
    }
    (boundary & !bad, bad)
}

/// Slow(er) path for batches containing non-ASCII: every unicode char in
/// (or straddling into/out of) the batch is classified with the packed
/// table via [`mask::classify_uni_chars`] — the same lookup the scalar
/// path would do — and joins the per-byte effective class masks, so
/// byte-adjacency == char-adjacency and the u64 boundary algebra applies
/// unchanged. Takes the ASCII class masks the caller already computed
/// (`ws64` = all ASCII whitespace).
///
/// Bad zones remain only for whitespace chars straddling a batch edge
/// (their `\s+(?!\S)` bookkeeping crosses the boundary), stray
/// continuation bytes (invalid UTF-8), and contractions at the batch
/// edge. An earlier version pattern-matched common leads with ~95 vector
/// ops (`mask::unicode_leads`) before falling back to per-char bad
/// zones; the direct table loop (typical hi batch: 1-3 unicode chars,
/// table hot in cache) plus edge-char resolution was worth ~13% end to
/// end. `#[inline(never)]`: inlining this into the walker wrecks the
/// clean path's register allocation (step 15).
///
/// On x86_64 the `target_feature` re-declaration keeps the bit-scan
/// loops on tzcnt/lzcnt/blsr in a baseline (non-native) build — this
/// function is out-of-line, so without it the ~21% of OWT batches
/// landing here would compile against baseline x86-64 even though every
/// caller is a SIMD batch classifier. Only the bit features are enabled
/// (not the callers' vector features): both the AVX-512 and AVX2 tiers
/// call this, so it must never emit instructions beyond the AVX2 tier's
/// set. Measured neutral on the OWT mask-compute diagnostic
/// (ASCII-dominated); kept for codegen parity on non-ASCII-heavy corpora
/// where this path dominates.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn extended_masks(
    bytes: &[u8],
    scan: usize,
    l64: u64,
    d64: u64,
    s64: u64,
    ws64: u64,
    hi64: u64,
    ap64: u64,
) -> (u64, u64) {
    let wsa = ws64;

    // Class-table LazyLock resolved once; the per-char classify below is
    // then a bare slice index.
    let ct = unicode::ClassTable::get();
    let class = move |cp| ct.class_of(cp);

    // Bit-0 carries via the prev-char walk-back; a char straddling into
    // this batch claims its continuation bytes with its class. Without
    // this, every batch following a unicode char became a bad zone at
    // bit 0, and a bad zone costs ~800 cycles in walker re-entries and
    // cold scalar gaps.
    let mut claim = mask::UniClasses::default();
    let (pl, pd, ps, pws, po) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else if bytes[scan - 1] < 0x80 {
        carries_at(bytes, scan)
    } else {
        // SAFETY: scan > 0 on this branch, and the classifier's
        // scan + 70 <= len batch guard covers pos + 3 <= len.
        let (cls, _lead, end) = unsafe { mask::char_through(bytes, scan, class) };
        let chm = if end > scan { (1u64 << (end - scan)) - 1 } else { 0 };
        claim.cont = chm;
        match cls {
            CharClass::Letter => {
                claim.l = chm;
                (1, 0, 0, 0, 0)
            }
            CharClass::Number => {
                claim.n = chm;
                (0, 1, 0, 0, 0)
            }
            CharClass::Other => {
                claim.o = chm;
                (0, 0, 0, 0, 1)
            }
            CharClass::Whitespace => {
                // A ws char straddling in defers to the scalar path (its
                // run-split bookkeeping needs the pre-batch extent) but
                // still marks its true class for neighbors' algebra.
                claim.ws = chm;
                claim.resid = chm;
                (0, 0, u64::from(bytes[scan - 1] == b' '), 1, 0)
            }
        }
    };

    // SAFETY: this classifier's scan + 70 <= len batch guard is exactly
    // `classify_uni_chars`' contract.
    let uni = unsafe {
        mask::classify_uni_chars::<true, false>(bytes, scan, hi64 & !claim.cont, class)
    };

    // Effective per-byte classes: every byte of a classified char carries
    // the char's class, so the same algebra as the pure-ASCII path
    // applies.
    let lb = l64 | claim.l | uni.l;
    let db = d64 | claim.n | uni.n;
    let wsb = wsa | claim.ws | uni.ws;
    let ob = !(l64 | d64 | wsa | hi64) | claim.o | uni.o;
    let contm = claim.cont | uni.cont;
    let resid = claim.resid | uni.resid;

    let cont_same =
        (lb & ((lb << 1) | pl)) | (db & ((db << 1) | pd)) | (ob & ((ob << 1) | po));
    let after_sp = (s64 << 1) | ps;
    let nb = !wsb & !cont_same & !after_sp & !contm;

    // Ws-run split: char-length-aware "followed by non-ws" test. All ws
    // chars whose lookahead crosses the batch edge look at byte 64: an
    // ASCII ws at 63, a 2-byte ws led at 62, a 3-byte ws led at 61
    // (later leads straddle out and are already bad zones). The ASCII
    // case is branchless as in the fast path; multi-byte edge leads are
    // rare enough to branch.
    let nn = !wsb;
    let mut split_ok = (wsa & (nn >> 1)) | (uni.w2 & (nn >> 2)) | (uni.w3 & (nn >> 3));
    let ws_leads = wsa | uni.w2 | uni.w3;
    let edge_mb = (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    if nb64 < 0x80 && edge_mb == 0 {
        split_ok = (split_ok & !(1 << 63)) | ((u64::from(!is_ascii_ws(nb64)) << 63) & wsa);
    } else {
        let edge = edge_mb | ((1 << 63) & wsa);
        if edge != 0 {
            // SAFETY: this classifier's scan + 70 <= len batch guard puts
            // the decode at scan + 64 in bounds (needs scan + 68 <= len).
            if unsafe { mask::nn_at_full(bytes, scan + 64) } {
                split_ok |= edge;
            } else {
                split_ok &= !edge;
            }
        }
    }
    let pwsb = (wsb << 1) | pws;
    let wsboundary = ws_leads & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    let mut bad = resid | resid << 1 | resid >> 1;

    // Contraction fixup: an apostrophe at a token start absorbs an
    // s/d/m/t/ll/ve/re suffix. One that could reach past bit 63 defers
    // to the scalar path (the next batch cannot see the moved boundary).
    let mut cand = ap64 & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let k = match bytes[scan + i + 1] {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + i + 2] == b'l' => 3,
            b'v' if bytes[scan + i + 2] == b'e' => 3,
            b'r' if bytes[scan + i + 2] == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}

/// `(pl, pd, ps, pws, po)` boundary carries for the char ending at
/// `scan - 1` (`scan > 0`), multi-byte aware via [`mask::char_through`].
/// `ps` (the ` ?` absorb) is ASCII 0x20 only. The ASCII case (almost
/// every call) is branchless — a class if-chain here is a per-batch
/// mispredict on natural text. Callers are the batch classifiers, whose
/// `scan + 70 <= len` guard covers the walk-back decode.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn carries_at(bytes: &[u8], scan: usize) -> (u64, u64, u64, u64, u64) {
    let b = bytes[scan - 1];
    if b < 0x80 {
        let (l, d, w) = (is_letter(b), is_digit(b), is_ascii_ws(b));
        let bit = |c: bool| u64::from(c);
        return (bit(l), bit(d), bit(b == b' '), bit(w), bit(!l && !d && !w));
    }
    // SAFETY: scan > 0 (this fn's caller contract), and the calling batch
    // classifier's scan + 70 <= len guard covers pos + 3 <= len.
    match unsafe { mask::char_through(bytes, scan, unicode::class_of) }.0 {
        CharClass::Letter => (1, 0, 0, 0, 0),
        CharClass::Number => (0, 1, 0, 0, 0),
        CharClass::Whitespace => (0, 0, 0, 1, 0),
        CharClass::Other => (0, 0, 0, 0, 1),
    }
}
pub(crate) struct R50kScheme;

impl MaskScheme for R50kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance_pos(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { batch_masks_x86::<AVX512>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs on the mask scanner above via the shared
/// [`MaskState`] batch walker; elsewhere every token takes `advance_pos`.
pub struct FastR50kPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastR50kPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    /// Used by the Python bindings, which re-borrow the underlying buffer on
    /// every `__next__` call.
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, state: MaskState::new(pos) }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }
}

impl<'a> Iterator for FastR50kPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<R50kScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastR50kPretokenizer, R50kScheme);

/// Advance past one token starting at `start`; returns the token's end.
/// `start` must be < `bytes.len()` and a valid token start.
/// Uses direct comparison chains instead of LUT + jump table to avoid
/// GOT indirection and improve branch prediction on common patterns.
///
/// Byte loads here look redundant but are effectively free: their addresses
/// depend only on `start`, so they issue in parallel under speculation. A
/// variant that did one u64 load and extracted bytes/scanned letters
/// in-register measured 0.84x — the shifts serialize after the load, while
/// independent L1 loads don't.
#[inline(always)]
fn advance_pos(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let b0 = unsafe { *bytes.get_unchecked(start) };

    // Bare ASCII letter start (~5% of OWT tokens; most words carry a space)
    if is_letter(b0) {
        return scan_letters_from(bytes, start + 1);
    }

    // Hot path: space before content (~78% of tokens, ~75% space+letters)
    if b0 == b' ' {
        if start + 1 < len {
            let b1 = unsafe { *bytes.get_unchecked(start + 1) };
            if is_letter(b1) {
                return scan_letters_from(bytes, start + 2);
            }
            if is_digit(b1) {
                return scan_digits_from(bytes, start + 2);
            }
            if b1 >= 0x80 {
                let (cp, l) = unsafe { decode_cp(bytes, start + 1) };
                let p = start + 1 + l;
                return match unicode::class_of(cp) {
                    CharClass::Letter => scan_letters_from(bytes, p),
                    CharClass::Number => scan_digits_from(bytes, p),
                    CharClass::Whitespace => advance_ws(bytes, p, start),
                    CharClass::Other => scan_other_from(bytes, p),
                };
            }
            if is_ascii_ws(b1) {
                return advance_ws(bytes, start + 1, start);
            }
            return scan_other_from(bytes, start + 2);
        }
        return start + 1;
    }

    // Non-ASCII
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, start) };
        let p = start + l;
        return match unicode::class_of(cp) {
            CharClass::Letter => scan_letters_from(bytes, p),
            CharClass::Number => scan_digits_from(bytes, p),
            CharClass::Whitespace => advance_ws(bytes, p, start),
            CharClass::Other => scan_other_from(bytes, p),
        };
    }

    // Digit
    if is_digit(b0) {
        return scan_digits_from(bytes, start + 1);
    }

    // Apostrophe / contraction
    if b0 == b'\'' {
        match bytes.get(start + 1) {
            Some(b's' | b'd' | b'm' | b't') => return start + 2,
            Some(b'l') if bytes.get(start + 2) == Some(&b'l') => return start + 3,
            Some(b'v') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            Some(b'r') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            _ => return scan_other_from(bytes, start + 1),
        }
    }

    // Whitespace (tab, newline, etc.)
    if b0.wrapping_sub(9) < 5 {
        return advance_ws(bytes, start + 1, start);
    }

    // Other (punctuation, symbols)
    scan_other_from(bytes, start + 1)
}

/// Advance through whitespace. `scan_pos` is where to continue scanning,
/// `token_start` is where the token began (for the split-off-last-char logic).
#[inline(always)]
fn advance_ws(bytes: &[u8], scan_pos: usize, token_start: usize) -> usize {
    let len = bytes.len();
    let mut p = scan_pos;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if is_ascii_ws(b) {
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if unicode::class_of(cp) == CharClass::Whitespace {
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if p < len {
        let ws_bytes = p - token_start;
        if ws_bytes >= 2 {
            let mut last = p - 1;
            while last > token_start && unsafe { *bytes.get_unchecked(last) } & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > token_start {
                return last;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_matches_state_machine_owt() {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");
        let max = 5_000_000.min(all_bytes.len());
        let mut end = max;
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        let input = &all_bytes[..end];

        let mut sm = crate::pretokenize::PretokenizerIter::new(input);
        let mut fast = FastR50kPretokenizer::new(input);
        let mut idx = 0usize;

        loop {
            match (sm.next(), fast.next()) {
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a.0, b.0,
                        "Mismatch at token {idx}: sm={:?} fast={:?}",
                        String::from_utf8_lossy(a.0),
                        String::from_utf8_lossy(b.0),
                    );
                }
                (None, None) => break,
                (Some(a), None) => panic!("SM extra at {idx}: {:?}", String::from_utf8_lossy(a.0)),
                (None, Some(b)) => {
                    panic!("Fast extra at {idx}: {:?}", String::from_utf8_lossy(b.0))
                }
            }
            idx += 1;
        }
        eprintln!("All {idx} tokens match.");
    }

    fn load_owt(max_bytes: usize) -> Vec<u8> {
        let data_dir = std::env::home_dir().unwrap().join("data");
        let all_bytes = std::fs::read(data_dir.join("owt_train.txt"))
            .expect("Could not read ~/data/owt_train.txt");
        let mut end = max_bytes.min(all_bytes.len());
        while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
            end -= 1;
        }
        all_bytes[..end].to_vec()
    }

    /// Reference iterator: plain `advance_pos` loop (the pre-mask-scanner
    /// implementation, itself validated against the state machine by
    /// `fast_matches_state_machine_owt`). The differential tests check the
    /// shipped mask-scanner iterator against it token for token.
    struct ScalarIter<'a> {
        bytes: &'a [u8],
        pos: usize,
    }
    impl<'a> ScalarIter<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, pos: 0 }
        }
    }
    impl<'a> Iterator for ScalarIter<'a> {
        type Item = Pretoken<'a>;
        #[inline]
        fn next(&mut self) -> Option<Pretoken<'a>> {
            let start = self.pos;
            if start >= self.bytes.len() {
                return None;
            }
            let end = advance_pos(self.bytes, start);
            self.pos = end;
            Some(Pretoken(&self.bytes[start..end]))
        }
    }

    /// Test-local wrapper over the shared batch walker, kept because
    /// `cargo test` builds skip fat LTO: shipped symbols are not inlined
    /// into test loops and measure ~1.4x slow, so the interleaved perf
    /// harness needs a locally instantiated iterator. Must produce exactly
    /// the shipped tokenization.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    struct MaskIter<'a> {
        bytes: &'a [u8],
        state: MaskState,
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    impl<'a> MaskIter<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, state: MaskState::new(0) }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    impl<'a> Iterator for MaskIter<'a> {
        type Item = Pretoken<'a>;
        #[inline]
        fn next(&mut self) -> Option<Pretoken<'a>> {
            let (start, end) = self.state.next_span::<R50kScheme>(self.bytes)?;
            Some(Pretoken(&self.bytes[start..end]))
        }
    }

    /// The batch classifier must actually engage on any machine with the
    /// assumed feature sets (aarch64 NEON; x86_64 AVX-512 or AVX2):
    /// on plain ASCII text it must report real token starts and no bad
    /// zones. Guards against a broken runtime detection or classifier
    /// silently passing the differential tests via the scalar fallback.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn batch_classifier_engages_on_ascii() {
        if !mask::simd_scanner_available() {
            eprintln!("no SIMD scanner on this CPU; skipping");
            return;
        }
        // Longer than scan + 70: the classifier needs byte-64+ lookahead.
        let text = b"The quick brown fox jumps over the lazy dog while 42 geese watch on quietly";
        let (usable, bad) = R50kScheme::batch_masks(text, 0);
        assert_eq!(bad, 0, "plain ASCII must produce no bad zones");
        let mut starts = vec![];
        let mut p = 0;
        while p < text.len() {
            starts.push(p);
            p = advance_pos(text, p);
        }
        for i in 0..64usize {
            if usable >> i & 1 == 1 {
                assert!(starts.contains(&i), "usable bit {i} is not a token start");
            }
        }
        assert!(usable.count_ones() >= 10, "classifier found too few boundaries");
    }

    /// Token-for-token differential check of `MaskIter` vs the shipped
    /// iterator on crafted edge cases.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn mask_iter_matches_shipped_edge_cases() {
        let cases: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b" ".to_vec(),
            b"a".to_vec(),
            b"hello world".to_vec(),
            b"  double  spaces  ".to_vec(),
            b"a\n\nb".to_vec(),
            b"a \n b".to_vec(),
            b"tabs\tand\nnewlines\r\n end".to_vec(),
            b"don't can't we'll they've you're I'm he's 'tis 'twas".to_vec(),
            b"DON'T CAN'T 'S 'LL".to_vec(),
            b"x'y z' 'a '' ' ".to_vec(),
            b"3.14 100,000 2nd a1b2".to_vec(),
            b"!!! ?! #hashtag @user (paren) [brack]".to_vec(),
            "café résumé naïve".as_bytes().to_vec(),
            "日本語のテキスト and English".as_bytes().to_vec(),
            "space\u{00A0}nbsp \u{00A0} runs".as_bytes().to_vec(),
            "emoji 🎉🎊 mix".as_bytes().to_vec(),
            "µ§±².5 ×÷".as_bytes().to_vec(),
            b"ws at end   ".to_vec(),
            b"   ws at start".to_vec(),
            // Exactly chunk-sized and chunk-straddling patterns.
            b"abcdefghijklmnop".to_vec(),
            b"abcdefghijklmno ".to_vec(),
            b"abcdefghijklmn 'll xyz".to_vec(),
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec(),
            b"a b c d e f g h i j k l m n o p q r s t u v w x".to_vec(),
            [b"word ".repeat(10), b"\xE2\x80\x82ws".to_vec()].concat(),
        ];
        for (ci, case) in cases.iter().enumerate() {
            let shipped: Vec<&[u8]> = ScalarIter::new(case).map(|t| t.0).collect();
            let masked: Vec<&[u8]> = FastR50kPretokenizer::new(case).map(|t| t.0).collect();
            assert_eq!(
                shipped,
                masked,
                "case {ci} diverged: {:?}",
                String::from_utf8_lossy(case)
            );
        }
    }

    /// Differential fuzz: random mixes of letters, digits, ws, punctuation,
    /// apostrophes, and multi-byte UTF-8 at every length 0..~200.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn mask_iter_matches_shipped_fuzz() {
        let pieces: &[&str] = &[
            "a", "B", "z", "9", "0", " ", "  ", "\n", "\t", "\r\n", "'", "'s", "'ll", "'re",
            "!", ".", ",", "(", "é", "ß", "日", "🎉", "\u{00A0}", "\u{2003}", "word", "12",
            "’", "’s", "“", "”", "–", "—", "…", "\u{2009}", "\u{200B}", "\u{2028}",
            "\u{202F}", "×", "÷", "«", "µ", "café", "éé", "naïve", "Α", "а", "ſ", "'ſ",
            "\u{661}\u{662}", "\u{FF11}", "क", "\u{940}", "\u{1D54F}", "€", "™", "\u{301}",
        ];
        let mut state = 0x243F6A8885A308D3u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..4000 {
            let target = (round % 200) + 1;
            let mut buf = Vec::new();
            while buf.len() < target {
                buf.extend_from_slice(pieces[(rng() % pieces.len() as u64) as usize].as_bytes());
            }
            let shipped: Vec<&[u8]> = ScalarIter::new(&buf).map(|t| t.0).collect();
            let masked: Vec<&[u8]> = FastR50kPretokenizer::new(&buf).map(|t| t.0).collect();
            assert_eq!(
                shipped,
                masked,
                "round {round} diverged on {:?}",
                String::from_utf8_lossy(&buf)
            );
        }
    }

    /// Differential check on the FULL OWT file (~11.9 GB), token for token.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore]
    fn mask_iter_matches_shipped_owt_full() {
        let input = load_owt(usize::MAX);
        eprintln!("loaded {} bytes", input.len());
        let mut shipped = ScalarIter::new(&input);
        let mut masked = FastR50kPretokenizer::new(&input);
        let mut idx = 0usize;
        loop {
            match (shipped.next(), masked.next()) {
                (Some(a), Some(b)) => {
                    if a.0 != b.0 {
                        panic!(
                            "token {idx} diverged: scalar={:?} masked={:?}",
                            String::from_utf8_lossy(a.0),
                            String::from_utf8_lossy(b.0)
                        );
                    }
                }
                (None, None) => break,
                (a, b) => panic!("length mismatch at {idx}: {:?} vs {:?}", a.is_some(), b.is_some()),
            }
            idx += 1;
        }
        eprintln!("all {idx} tokens match on full OWT");
    }

    /// Differential check on real OWT (100 MB), token for token.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore]
    fn mask_iter_matches_shipped_owt() {
        let input = load_owt(100_000_000);
        let mut shipped = ScalarIter::new(&input);
        let mut masked = FastR50kPretokenizer::new(&input);
        let mut idx = 0usize;
        loop {
            match (shipped.next(), masked.next()) {
                (Some(a), Some(b)) => assert_eq!(
                    a.0,
                    b.0,
                    "token {idx} diverged: shipped={:?} masked={:?}",
                    String::from_utf8_lossy(a.0),
                    String::from_utf8_lossy(b.0)
                ),
                (None, None) => break,
                (a, b) => panic!("length mismatch at {idx}: {:?} vs {:?}", a.is_some(), b.is_some()),
            }
            idx += 1;
        }
        eprintln!("all {idx} tokens match");
    }

    /// Iterator-level interleaved A/B: local copy of the shipped scalar
    /// iterator vs the mask scanner. Both variants are LOCAL copies: test
    /// builds skip fat LTO, so shipped symbols are not inlined here and
    /// measure ~1.4x slow (see project memory / step 14 of the log).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore]
    fn ab_r50k_mask_iter_interleaved() {
        fn drive_scalar(input: &[u8]) -> (usize, u64) {
            let mut n = 0usize;
            let mut acc = 0u64;
            for t in ScalarIter::new(input) {
                std::hint::black_box(t.0);
                acc = acc.wrapping_add(t.0.len() as u64);
                n += 1;
            }
            (n, acc)
        }
        fn drive_mask(input: &[u8]) -> (usize, u64) {
            let mut n = 0usize;
            let mut acc = 0u64;
            for t in MaskIter::new(input) {
                std::hint::black_box(t.0);
                acc = acc.wrapping_add(t.0.len() as u64);
                n += 1;
            }
            (n, acc)
        }

        let input = load_owt(100_000_000);
        let mb = input.len() as f64 / 1e6;
        let a = drive_scalar(&input);
        let b = drive_mask(&input);
        assert_eq!(a, b, "variants disagree");

        let (mut best_a, mut best_b) = (f64::INFINITY, f64::INFINITY);
        for round in 0..7 {
            let t = std::time::Instant::now();
            std::hint::black_box(drive_scalar(&input));
            let da = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            std::hint::black_box(drive_mask(&input));
            let db = t.elapsed().as_secs_f64();
            best_a = best_a.min(da);
            best_b = best_b.min(db);
            eprintln!(
                "round {round}: scalar {:.0} MB/s | mask {:.0} MB/s",
                mb / da,
                mb / db
            );
        }
        eprintln!(
            "best: scalar {:.0} MB/s | mask {:.0} MB/s | mask/scalar {:.3}x",
            mb / best_a,
            mb / best_b,
            best_a / best_b
        );
    }

    /// Diagnostics for the mask scanner: fallback rate on real OWT, and
    /// mask-vs-scalar throughput on a sanitized buffer (apostrophes and
    /// non-ASCII replaced) that never triggers the fallback.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    #[ignore]
    fn mask_iter_diagnostics() {
        let input = load_owt(100_000_000);

        // Bad-zone census: dirty batches, and bytes covered by the first
        // gap (prefix end to resume point) as a scalar-work proxy.
        let (mut batches, mut dirty_batches, mut gap_bytes) = (0usize, 0usize, 0u64);
        let mut scan = 0usize;
        while scan + 64 <= input.len() {
            let (usable, bad) = R50kScheme::batch_masks(&input, scan);
            batches += 1;
            if bad != 0 {
                dirty_batches += 1;
                let first_bad = bad.trailing_zeros();
                let rest = usable & (u64::MAX << first_bad);
                let resume = if rest != 0 { rest.trailing_zeros() } else { 64 };
                gap_bytes += u64::from(resume - first_bad);
            }
            scan += 64;
        }
        eprintln!(
            "batches: {batches}, dirty: {dirty_batches} ({:.2}%), first-gap bytes {:.2}%",
            100.0 * dirty_batches as f64 / batches as f64,
            100.0 * gap_bytes as f64 / input.len() as f64
        );

        // Sanitized buffer: no apostrophes, no high bytes.
        let clean: Vec<u8> = input
            .iter()
            .map(|&b| if b >= 0x80 || b == b'\'' { b'x' } else { b })
            .collect();
        let mb = clean.len() as f64 / 1e6;

        fn drive<'a, I: Iterator<Item = Pretoken<'a>>>(it: I) -> (usize, u64) {
            let mut n = 0usize;
            let mut acc = 0u64;
            for t in it {
                std::hint::black_box(t.0);
                acc = acc.wrapping_add(t.0.len() as u64);
                n += 1;
            }
            (n, acc)
        }
        assert_eq!(
            drive(ScalarIter::new(&clean)),
            drive(MaskIter::new(&clean))
        );
        let (mut best_a, mut best_b) = (f64::INFINITY, f64::INFINITY);
        for round in 0..5 {
            let t = std::time::Instant::now();
            std::hint::black_box(drive(ScalarIter::new(&clean)));
            let da = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            std::hint::black_box(drive(MaskIter::new(&clean)));
            let db = t.elapsed().as_secs_f64();
            best_a = best_a.min(da);
            best_b = best_b.min(db);
            eprintln!(
                "clean round {round}: scalar {:.0} MB/s | mask {:.0} MB/s",
                mb / da,
                mb / db
            );
        }
        eprintln!(
            "clean best: scalar {:.0} | mask {:.0} MB/s | {:.3}x",
            mb / best_a,
            mb / best_b,
            best_a / best_b
        );
    }

    #[test]
    #[ignore]
    fn r50k_token_stats_owt() {
        let input = load_owt(10_000_000);
        let mut counts = [0usize; 8]; // letter, sp+letter, sp+digit, sp+other, digit, ws, apos, other/nonascii
        let mut letter_tokens = 0usize;
        let mut in_window = 0usize;
        let mut total = 0usize;
        let mut pos = 0usize;
        while pos < input.len() {
            let end = advance_pos(&input, pos);
            let b0 = input[pos];
            let idx = if is_letter(b0) {
                0
            } else if b0 == b' ' {
                match input.get(pos + 1) {
                    Some(&b) if is_letter(b) => 1,
                    Some(&b) if is_digit(b) => 2,
                    _ => 3,
                }
            } else if is_digit(b0) {
                4
            } else if is_ascii_ws(b0) {
                5
            } else if b0 == b'\'' {
                6
            } else {
                7
            };
            counts[idx] += 1;
            if idx == 0 || idx == 1 {
                letter_tokens += 1;
                if end - pos <= 8 {
                    in_window += 1;
                }
            }
            total += 1;
            pos = end;
        }
        let pct = |n: usize| 100.0 * n as f64 / total as f64;
        eprintln!("total tokens: {total}");
        for (name, n) in [
            "letter", "sp+letter", "sp+digit", "sp+other", "digit", "ws", "apos", "other/hi",
        ]
        .iter()
        .zip(counts)
        {
            eprintln!("{name:>10}: {n:>9} ({:.1}%)", pct(n));
        }
        eprintln!(
            "letter tokens resolved in 8-byte window: {:.1}%",
            100.0 * in_window as f64 / letter_tokens as f64
        );
        eprintln!(
            "avg token len: {:.2} bytes",
            input.len() as f64 / total as f64
        );
    }

    /// Best-of-5 throughput of `advance_pos` over `bytes`, in MB/s.
    fn measure(bytes: &[u8], label: &str) -> f64 {
        let mb = bytes.len() as f64 / 1e6;
        let (n, _) = drive(bytes, advance_pos);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            std::hint::black_box(drive(bytes, advance_pos));
            best = best.min(t.elapsed().as_secs_f64());
        }
        let mbs = mb / best;
        // cycles/token estimate assumes ~4.5 GHz P-core
        let cpt = 4.5e9 * best / n as f64;
        eprintln!(
            "{label:>22}: {mbs:>5.0} MB/s | {:.2} B/token | ~{cpt:.1} cy/token",
            bytes.len() as f64 / n as f64
        );
        mbs
    }

    #[test]
    #[ignore]
    fn r50k_prediction_floor() {
        // Pure single-path floor: one token shape, perfectly predictable.
        let hello: Vec<u8> = b" hello".repeat(16_000_000 / 6);
        measure(&hello, "' hello' x N");

        // Same multiset of real OWT tokens, predictable vs shuffled order.
        // Space-prefixed tokens are adjacency-safe: concatenating them in any
        // order re-tokenizes to exactly the same tokens.
        let input = load_owt(100_000_000);
        let bag: Vec<&[u8]> = FastR50kPretokenizer::new(&input)
            .map(|t| t.0)
            .filter(|t| t[0] == b' ' && t.len() > 1)
            .collect();
        eprintln!("bag: {} space-prefixed tokens", bag.len());

        let mut sorted = bag.clone();
        sorted.sort_unstable();
        let predictable: Vec<u8> = sorted.concat();
        drop(sorted);

        let mut shuffled = bag;
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };
        for i in (1..shuffled.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let unpredictable: Vec<u8> = shuffled.concat();
        let n_bag = shuffled.len();
        drop(shuffled);

        assert_eq!(drive(&predictable, advance_pos).0, n_bag);
        assert_eq!(drive(&unpredictable, advance_pos).0, n_bag);

        measure(&predictable, "sorted (predictable)");
        measure(&unpredictable, "shuffled (same bag)");
        measure(&input, "natural OWT");
    }

    fn drive(bytes: &[u8], f: impl Fn(&[u8], usize) -> usize) -> (usize, u64) {
        let mut pos = 0usize;
        let mut n = 0usize;
        let mut acc = 0u64;
        while pos < bytes.len() {
            let end = f(bytes, pos);
            acc = acc.wrapping_add(end as u64);
            n += 1;
            pos = end;
        }
        (n, acc)
    }

    #[test]
    #[ignore]
    fn aa_r50k_advance_interleaved() {
        // A/A control: same implementation through two separately
        // monomorphized drivers, to gauge code-layout noise in this harness.
        let input = load_owt(100_000_000);
        let mb = input.len() as f64 / 1e6;
        let copy_a = advance_pos;
        let copy_b = |b: &[u8], s: usize| advance_pos(b, s);
        std::hint::black_box(drive(&input, copy_a));
        std::hint::black_box(drive(&input, copy_b));
        let (mut best_a, mut best_b) = (f64::INFINITY, f64::INFINITY);
        for round in 0..7 {
            let t = std::time::Instant::now();
            std::hint::black_box(drive(&input, copy_a));
            let da = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            std::hint::black_box(drive(&input, copy_b));
            let db = t.elapsed().as_secs_f64();
            best_a = best_a.min(da);
            best_b = best_b.min(db);
            eprintln!("round {round}: A {:.0} MB/s | A' {:.0} MB/s", mb / da, mb / db);
        }
        eprintln!(
            "best: A {:.0} MB/s | A' {:.0} MB/s | ratio {:.3}x",
            mb / best_a,
            mb / best_b,
            best_a / best_b
        );
    }

    /// Interleaved same-binary A/B harness (min-of-7): swap an experimental
    /// `advance_pos` candidate in below and run with `--ignored --nocapture`.
    /// Run `aa_r50k_advance_interleaved` first as the layout-noise control.
    /// (Isolated A/B of the packed class table vs the old chained ICU
    /// predicates measured 0.90x for the ICU version, i.e. the table alone
    /// is worth ~1.11x.)
    #[test]
    #[ignore]
    fn ab_r50k_advance_interleaved() {
        let input = load_owt(100_000_000);
        let mb = input.len() as f64 / 1e6;
        eprintln!("input: {mb:.1} MB");

        // Experimental candidate under test; placeholder = shipped impl.
        let experimental = |b: &[u8], s: usize| advance_pos(b, s);

        // Warmup + equivalence check
        let (n_a, acc_a) = drive(&input, advance_pos);
        let (n_b, acc_b) = drive(&input, experimental);
        assert_eq!((n_a, acc_a), (n_b, acc_b), "variants disagree");

        let mut best_a = f64::INFINITY;
        let mut best_b = f64::INFINITY;
        for round in 0..7 {
            let t = std::time::Instant::now();
            let r = drive(&input, advance_pos);
            let da = t.elapsed().as_secs_f64();
            std::hint::black_box(r);

            let t = std::time::Instant::now();
            let r = drive(&input, experimental);
            let db = t.elapsed().as_secs_f64();
            std::hint::black_box(r);

            best_a = best_a.min(da);
            best_b = best_b.min(db);
            eprintln!(
                "round {round}: shipped {:.0} MB/s | experimental {:.0} MB/s",
                mb / da,
                mb / db
            );
        }
        eprintln!(
            "best: shipped {:.0} MB/s | experimental {:.0} MB/s | ratio {:.3}x",
            mb / best_a,
            mb / best_b,
            best_a / best_b
        );
    }

}
