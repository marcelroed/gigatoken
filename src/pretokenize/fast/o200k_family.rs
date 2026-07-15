//! Shared implementation for the o200k regex family: o200k_base (GPT-4o,
//! gpt-oss) and the Nemotron-3 variant. Their patterns share the shape
//!
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+SUF?|
//!  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*SUF?|
//!  \p{N}{1,3} or \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! where `SUF = (?i:'s|'t|'re|'ve|'m|'ll|'d)` exists only in o200k
//! (`CONTRACTIONS`), and the digit group is `\p{N}{1,3}` for o200k vs
//! `\p{N}` for Nemotron (`DIGITS3`).
//!
//! Differences from the cl100k family:
//! - Letter runs are case-structured. Under leftmost-greedy backtracking
//!   the two letter alternatives reduce to a phase automaton (see
//!   [`CaseState`]): ULC phase until the first strict-lower (Ll), then
//!   LLC phase, where a strict-upper (Lu/Lt) ends the token; a run ending
//!   while still in ULC phase backtracks to its last caseless/mark char
//!   ("camelCase" -> `camel|Case`, "HTTPResponse" one token,
//!   "AxxB" -> `Axx|B` for caseless x). For pure-ASCII text there are no
//!   caseless letters, so the rule IS pairwise: a boundary sits before
//!   `[A-Z]` exactly when the previous char is `[a-z]` — what the ASCII
//!   mask algebra uses; caseless-before-upper needs the phase and
//!   lookahead, so the extended path defers those (rare) chars.
//! - Contractions are attached suffixes of the letter alternatives, not a
//!   standalone alternative: "don't" is ONE token, and the char after a
//!   consumed suffix always starts a new token ("can'ts" -> `can't|s`).
//!   A contraction applies only when the apostrophe directly follows a
//!   letter-run char; elsewhere `'` is ordinary punctuation (which may
//!   still prefix a letter run: "3'ts" -> `3|'ts`).
//! - Punctuation runs absorb a `[\r\n/]*` tail. Since `/` is itself in
//!   the punct class, the absorbed tail always begins with a newline:
//!   ".\n//" is one token.
//! - Marks (`\p{M}`) are dual-class: they join letter runs (they sit in
//!   both letter brackets) AND continue `[^\s\p{L}\p{N}]+` punct runs.
//!   Their effective class is run-contextual, so the mask scanner routes
//!   mark chars (rare) through the scalar path as bad zones.
//!
//! The boundary algebra below mirrors `cl100k_family` — see that module
//! and pretokenizer_optimization_log.md step 16 for the base rules.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::mask::{self, AsciiMasks};
use super::{decode_cp, is_ascii_ws, is_digit, is_letter, scan_numbers_max3};
use crate::pretokenize::unicode::{O200kCharClass, o200k_class_of};

#[inline(always)]
fn is_upper_ascii(b: u8) -> bool {
    b.wrapping_sub(b'A') < 26
}

/// Is this byte in the `[\r\n/]` tail class?
#[inline(always)]
fn is_tail_byte(b: u8) -> bool {
    matches!(b, b'\r' | b'\n' | b'/')
}

// -----------------------------------------------------------------------
// Scalar ground truth
// -----------------------------------------------------------------------

/// Scan state of a case-structured letter run, mirroring the two-bracket
/// alternatives under leftmost-greedy backtracking. `U`: still inside
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*` (no strict-lower seen yet);
/// `last_cl_end` is the end offset of the last caseless/mark char in the
/// phase (0 = none) — the backtrack split point if the run ends on a
/// strict-upper tail. `L`: inside `[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`, where a
/// strict-upper char ends the token unconditionally.
#[derive(Clone, Copy)]
enum CaseState {
    U { last_cl_end: usize },
    L,
}

/// Initial [`CaseState`] for a run starting on an ASCII letter (ASCII has
/// no caseless letters).
#[inline(always)]
fn ascii_letter_state(b: u8) -> CaseState {
    if is_upper_ascii(b) {
        CaseState::U { last_cl_end: 0 }
    } else {
        CaseState::L
    }
}

/// If the char at `pos` is a letter-run member (`\p{L}` or `\p{M}`),
/// return (offset past it, initial scan state).
#[inline(always)]
fn letter_run_first(bytes: &[u8], pos: usize) -> Option<(usize, CaseState)> {
    let &b = bytes.get(pos)?;
    if is_letter(b) {
        return Some((pos + 1, ascii_letter_state(b)));
    }
    if b >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        match o200k_class_of(cp) {
            O200kCharClass::Upper => return Some((pos + l, CaseState::U { last_cl_end: 0 })),
            O200kCharClass::Lower => return Some((pos + l, CaseState::L)),
            O200kCharClass::Caseless | O200kCharClass::Mark => {
                return Some((pos + l, CaseState::U { last_cl_end: pos + l }));
            }
            _ => {}
        }
    }
    None
}

/// Letter-run continuation with the o200k casing rules: scan run members
/// (letters and marks) from `pos` with the phase automaton described on
/// [`CaseState`]. In phase U a strict-upper char continues the token and
/// a strict-lower switches to phase L; in phase L a strict-upper ends
/// the token. A run that ends while still in phase U backtracks to the
/// last caseless/mark char, splitting off the trailing strict-upper run
/// (which the next `advance` then consumes whole): "AxxB" -> `Axx|B`
/// for caseless x, "HTTPResponse" and "Z\u{5BF}\u{416}dz" one token.
#[inline(always)]
fn scan_case_run(bytes: &[u8], mut pos: usize, mut st: CaseState) -> usize {
    let len = bytes.len();
    loop {
        while pos < len {
            let b = unsafe { *bytes.get_unchecked(pos) };
            if is_upper_ascii(b) {
                if matches!(st, CaseState::L) {
                    return pos;
                }
                pos += 1;
            } else if is_letter(b) {
                st = CaseState::L;
                pos += 1;
            } else {
                break;
            }
        }
        if pos < len && unsafe { *bytes.get_unchecked(pos) } >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, pos) };
            match o200k_class_of(cp) {
                O200kCharClass::Upper => {
                    if matches!(st, CaseState::L) {
                        return pos;
                    }
                    pos += l;
                }
                O200kCharClass::Lower => {
                    st = CaseState::L;
                    pos += l;
                }
                O200kCharClass::Caseless | O200kCharClass::Mark => {
                    pos += l;
                    if let CaseState::U { ref mut last_cl_end } = st {
                        *last_cl_end = pos;
                    }
                }
                _ => break,
            }
            continue;
        }
        break;
    }
    // End of the letter run. A phase-U run ending on a strict-upper tail
    // backtracks to the last caseless/mark char (`[LLC]+` must consume at
    // least one char, and only a caseless/mark can be given back).
    match st {
        CaseState::U { last_cl_end } if last_cl_end != 0 => last_cl_end,
        _ => pos,
    }
}

/// Attach a `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` suffix to a letter token
/// ending at `end`, when the scheme has contractions.
#[inline(always)]
fn try_suffix<const CONTRACTIONS: bool>(bytes: &[u8], end: usize) -> usize {
    if !CONTRACTIONS || bytes.get(end) != Some(&b'\'') {
        return end;
    }
    match bytes.get(end + 1).map(u8::to_ascii_lowercase) {
        Some(b's' | b'd' | b'm' | b't') => end + 2,
        Some(b'l') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'l') => end + 3,
        Some(b'v') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'e') => end + 3,
        Some(b'r') if bytes.get(end + 2).map(u8::to_ascii_lowercase) == Some(b'e') => end + 3,
        // U+017F LATIN SMALL LETTER LONG S case-folds to 's' under `(?i)`
        Some(0xC5) if bytes.get(end + 2) == Some(&0xBF) => end + 3,
        _ => end,
    }
}

/// `[^\s\p{L}\p{N}]+` from `pos` (punctuation, symbols, marks, controls —
/// everything except letters, numbers, and whitespace).
#[inline(always)]
fn scan_punct_from(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut p = pos;
    loop {
        while p < len {
            let b = unsafe { *bytes.get_unchecked(p) };
            if b >= 0x80 {
                break;
            }
            if is_letter(b) || is_digit(b) || is_ascii_ws(b) {
                return p;
            }
            p += 1;
        }
        if p < len {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if matches!(
                o200k_class_of(cp),
                O200kCharClass::Other | O200kCharClass::Mark
            ) {
                p += l;
                continue;
            }
        }
        return p;
    }
}

/// `[\r\n/]*`: the tail absorbed after a punctuation run.
#[inline(always)]
fn scan_tail(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && is_tail_byte(unsafe { *bytes.get_unchecked(pos) }) {
        pos += 1;
    }
    pos
}

/// Whitespace-led token starting at `start`: `\s*[\r\n]+` | `\s+(?!\S)` |
/// `\s+`, in that priority. Precondition: the letter-prefix and
/// space+punct alternatives were ruled out.
#[inline(always)]
fn ws_token_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl_end = 0usize; // 0 = run contains no \r\n
    let mut last_char_start = start;
    while p < len {
        let b = unsafe { *bytes.get_unchecked(p) };
        if b == b'\r' || b == b'\n' {
            last_char_start = p;
            p += 1;
            last_nl_end = p;
        } else if is_ascii_ws(b) {
            last_char_start = p;
            p += 1;
        } else if b >= 0x80 {
            let (cp, l) = unsafe { decode_cp(bytes, p) };
            if o200k_class_of(cp) == O200kCharClass::Whitespace {
                last_char_start = p;
                p += l;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if last_nl_end != 0 {
        return last_nl_end; // `\s*[\r\n]+`: through the last newline
    }
    if p >= len {
        return p; // `\s+(?!\S)`: lookahead succeeds at EOS
    }
    if last_char_start > start {
        return last_char_start; // `\s+(?!\S)`: all but the last ws char
    }
    p // `\s+`: single whitespace char before content
}

/// `\p{N}{1,3}` or `\p{N}` starting at a digit char ending at `first_end`.
#[inline(always)]
fn digit_token_end<const DIGITS3: bool>(bytes: &[u8], first_end: usize) -> usize {
    if DIGITS3 {
        scan_numbers_max3(bytes, first_end, 1)
    } else {
        first_end
    }
}

/// Advance past one token starting at `pos`. Returns the new position.
/// `pos` must be < `bytes.len()`.
#[inline(always)]
pub(crate) fn advance_pos<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    pos: usize,
) -> usize {
    let b0 = unsafe { *bytes.get_unchecked(pos) };

    // Hot path 1: ASCII letter run (empty prefix)
    if is_letter(b0) {
        let e = scan_case_run(bytes, pos + 1, ascii_letter_state(b0));
        return try_suffix::<CONTRACTIONS>(bytes, e);
    }

    // Hot path 2: space prefix
    if b0 == b' ' {
        let Some(&b1) = bytes.get(pos + 1) else {
            return pos + 1; // trailing lone space (`\s+(?!\S)` at EOS)
        };
        if is_letter(b1) {
            let e = scan_case_run(bytes, pos + 2, ascii_letter_state(b1));
            return try_suffix::<CONTRACTIONS>(bytes, e);
        }
        if b1 < 0x80 {
            if is_digit(b1) {
                return pos + 1; // numbers never absorb the space
            }
            if is_ascii_ws(b1) {
                return ws_token_end(bytes, pos);
            }
            // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
            let p = scan_punct_from(bytes, pos + 2);
            return scan_tail(bytes, p);
        }
        let (cp, l) = unsafe { decode_cp(bytes, pos + 1) };
        let p1 = pos + 1 + l;
        return match o200k_class_of(cp) {
            O200kCharClass::Upper => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run(bytes, p1, CaseState::U { last_cl_end: 0 }),
            ),
            O200kCharClass::Lower => {
                try_suffix::<CONTRACTIONS>(bytes, scan_case_run(bytes, p1, CaseState::L))
            }
            O200kCharClass::Caseless | O200kCharClass::Mark => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run(bytes, p1, CaseState::U { last_cl_end: p1 }),
            ),
            O200kCharClass::Whitespace => ws_token_end(bytes, pos),
            O200kCharClass::Number => pos + 1,
            O200kCharClass::Other => scan_tail(bytes, scan_punct_from(bytes, p1)),
        };
    }

    // Non-ASCII first char
    if b0 >= 0x80 {
        let (cp, l) = unsafe { decode_cp(bytes, pos) };
        let p0 = pos + l;
        return match o200k_class_of(cp) {
            O200kCharClass::Upper => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run(bytes, p0, CaseState::U { last_cl_end: 0 }),
            ),
            O200kCharClass::Lower => {
                try_suffix::<CONTRACTIONS>(bytes, scan_case_run(bytes, p0, CaseState::L))
            }
            O200kCharClass::Caseless | O200kCharClass::Mark => try_suffix::<CONTRACTIONS>(
                bytes,
                scan_case_run(bytes, p0, CaseState::U { last_cl_end: p0 }),
            ),
            O200kCharClass::Number => digit_token_end::<DIGITS3>(bytes, p0),
            // Any non-letter/number char except \r\n may prefix a run
            class => {
                if let Some((e, st)) = letter_run_first(bytes, p0) {
                    return try_suffix::<CONTRACTIONS>(bytes, scan_case_run(bytes, e, st));
                }
                if class == O200kCharClass::Whitespace {
                    ws_token_end(bytes, pos)
                } else {
                    scan_tail(bytes, scan_punct_from(bytes, p0))
                }
            }
        };
    }

    // ASCII digit
    if is_digit(b0) {
        return digit_token_end::<DIGITS3>(bytes, pos + 1);
    }

    // \r and \n are excluded from the letter-run prefix
    if b0 == b'\r' || b0 == b'\n' {
        return ws_token_end(bytes, pos);
    }

    // Other ASCII whitespace (\t, \x0b, \x0c) may prefix a letter run
    if is_ascii_ws(b0) {
        if let Some((e, st)) = letter_run_first(bytes, pos + 1) {
            return try_suffix::<CONTRACTIONS>(bytes, scan_case_run(bytes, e, st));
        }
        return ws_token_end(bytes, pos);
    }

    // ASCII punctuation/symbol/control (including `'`: o200k has no
    // standalone contraction alternative, so a leading apostrophe is
    // ordinary punctuation / a letter-run prefix: "'sound" is one token)
    if let Some((e, st)) = letter_run_first(bytes, pos + 1) {
        return try_suffix::<CONTRACTIONS>(bytes, scan_case_run(bytes, e, st));
    }
    scan_tail(bytes, scan_punct_from(bytes, pos + 1))
}

// -----------------------------------------------------------------------
// Mask-scanner boundary algebra
// -----------------------------------------------------------------------

/// Smear `seed` upward (toward higher bits) through contiguous set bits of
/// `within`, in log steps.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn smear_up(seed: u64, within: u64) -> u64 {
    let mut a = seed;
    let mut m = within;
    let mut sh = 1u32;
    while sh < 64 {
        a |= (a << sh) & m;
        m &= m << sh;
        sh <<= 1;
    }
    a
}

/// Per-byte class masks for a batch's unicode chars under the o200k
/// classifier — the o200k analogue of [`mask::UniClasses`], with the
/// case-split letter masks the scheme needs. Every byte of a classified
/// char carries the char's class, so byte-adjacency == char-adjacency.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy, Default)]
struct OUni {
    /// All letter-run bytes (upper + lower + caseless; marks excluded —
    /// they are contextual and deferred via `mk`).
    l: u64,
    /// Strict-upper (Lu/Lt) bytes.
    u: u64,
    /// Caseless-letter (Lm/Lo) bytes.
    cl: u64,
    /// Number / punct-run / whitespace bytes.
    n: u64,
    o: u64,
    ws: u64,
    /// Whitespace lead bits by char length (for `(?!\S)` shift tests).
    w2: u64,
    w3: u64,
    /// Lead bits of all classified chars by length (two-chars-back rule).
    lead2: u64,
    lead3: u64,
    lead4: u64,
    /// Continuation bytes of classified chars.
    cont: u64,
    /// Bytes only the scalar path can decide (±1 bad smear): number chars
    /// (char-counted grouping), ws straddling the batch end, stray
    /// continuation bytes.
    resid: u64,
    /// Mark bytes (±4 bad smear). A mark's run-contextual class can
    /// affect boundaries up to two CHARS after it, which multi-byte
    /// followers can push past the 4-byte smear; those stragglers are
    /// wrongly-cleared bits (extending the scalar walk) or wrongly-set
    /// bits interior to a token starting inside the zone, both killed by
    /// MaskState's resume masking after the scalar overrun — the same
    /// invariant the resid zones rely on.
    mk: u64,
}

/// Boundary carries from the chars before the batch (o200k variant of the
/// cl100k family's `Carries`, plus the case and absorbed-tail bits).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy, Default)]
struct OCarries {
    /// P1 is a letter / strict-upper / caseless / space (0x20) /
    /// non-newline non-space ws / punct / any ws / digit.
    pl: u64,
    pu: u64,
    pcl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    /// P2 is punct-or-space, for a char lead at bit 0.
    c2_os: u64,
    /// Same test positioned at the first lead AFTER a straddling-in P1.
    b2b_in: u64,
    /// P1 is an absorbed `[\r\n/]*` tail byte whose token may continue
    /// into this batch (seeds the tail smear at bit 0).
    p_abs: bool,
    /// The tail walkback could not resolve (pathological run): the batch's
    /// leading tail-class run must be a bad zone.
    force_bad_lead: bool,
}

/// Was the tail-class byte at `scan - 1` absorbed by a punct run's
/// `[\r\n/]*` tail (as opposed to being a fresh punct-run `/` or a
/// ws-run newline)? Walks the tail-class run back (bounded) and
/// classifies the byte before it. `None`: unresolved (over-long run, or a
/// preceding mark whose own class is run-contextual).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn prev_tail_absorbed(bytes: &[u8], scan: usize) -> Option<bool> {
    debug_assert!(scan >= 1 && is_tail_byte(bytes[scan - 1]));
    let mut r = scan - 1;
    let mut steps = 0;
    while r > 0 && is_tail_byte(bytes[r - 1]) {
        r -= 1;
        steps += 1;
        if steps > 8 {
            return None;
        }
    }
    // T-run = bytes[r..scan]. The `[\r\n/]*` tail is greedy, so once
    // absorption triggers — at the first newline that directly follows a
    // punct-run char (an in-run slash, or the pre-run char for a
    // run-leading newline) — everything to the run's end is absorbed.
    // Before the trigger, newlines are ws-run members and slashes are
    // ordinary punct-run bytes.
    let run = &bytes[r..scan];
    let mut trigger = usize::MAX;
    let mut seen_slash = false;
    for (j, &b) in run.iter().enumerate() {
        if b == b'/' {
            seen_slash = true;
            continue;
        }
        if seen_slash {
            trigger = j;
            break;
        }
        if j == 0 {
            // Run-leading newline: the pre-run char decides.
            if r == 0 {
                continue;
            }
            let pb = bytes[r - 1];
            let pred_punct = if pb < 0x80 {
                if !is_letter(pb) && !is_digit(pb) && !is_ascii_ws(pb) {
                    Some(true)
                } else {
                    Some(false)
                }
            } else {
                let mut k = r - 1;
                while k > 0 && bytes[k] & 0xC0 == 0x80 {
                    k -= 1;
                }
                let (cp, _) = unsafe { decode_cp(bytes, k) };
                match o200k_class_of(cp) {
                    O200kCharClass::Other => Some(true),
                    // A mark continues whatever run precedes it.
                    O200kCharClass::Mark => None,
                    _ => Some(false),
                }
            };
            match pred_punct {
                Some(true) => {
                    trigger = 0;
                    break;
                }
                Some(false) => {}
                None => return None,
            }
        }
    }
    Some(scan - 1 - r >= trigger)
}

/// Two-back "punct or space" test (`c2_os`) for the ASCII byte at `idx`.
/// A slash may be an absorbed tail byte — a token end, neither punct-run
/// member nor space — so it resolves through the walkback. `None`:
/// unresolved (callers set `force_bad_lead`).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn c2_os_ascii(bytes: &[u8], idx: usize) -> Option<u64> {
    let b2 = bytes[idx];
    if b2 == b'/' {
        return prev_tail_absorbed(bytes, idx + 1).map(|abs| u64::from(!abs));
    }
    Some(u64::from(
        b2 == b' ' || (!is_letter(b2) && !is_digit(b2) && !is_ascii_ws(b2)),
    ))
}

/// Pure-ASCII carries. Requires `scan > 0`, `bytes[scan-1] < 0x80` (and
/// `bytes[scan-2] < 0x80` when present), and `bytes[scan-1]` NOT a
/// tail-class byte (those route through [`prev_tail_absorbed`] first).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_carries(bytes: &[u8], scan: usize) -> OCarries {
    let b = bytes[scan - 1];
    debug_assert!(!is_tail_byte(b));
    let bit = |c: bool| u64::from(c);
    let (l, d, w) = (is_letter(b), is_digit(b), is_ascii_ws(b));
    let (c2_os, c2_unresolved) = if scan >= 2 {
        match c2_os_ascii(bytes, scan - 2) {
            Some(v) => (v, false),
            None => (0, true),
        }
    } else {
        (0, false)
    };
    OCarries {
        force_bad_lead: c2_unresolved,
        pl: bit(l),
        pu: bit(is_upper_ascii(b)),
        pcl: 0,
        ps: bit(b == b' '),
        pwt: bit(w && b != b' '), // \r\n excluded by the debug_assert
        po: bit(!l && !d && !w),
        pws: bit(w),
        pd: bit(d),
        c2_os,
        b2b_in: 0,
        p_abs: false,
    }
}

/// Carries when the byte before the batch is tail-class (`[\r\n/]`):
/// resolves absorbed-tail vs fresh-run via [`prev_tail_absorbed`]. An
/// absorbed tail ended the previous token, so every "P1 is X" carry is
/// zero and only the tail-continuation seed survives.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(never)]
fn tail_carries(bytes: &[u8], scan: usize) -> OCarries {
    match prev_tail_absorbed(bytes, scan) {
        Some(true) => OCarries { p_abs: true, ..OCarries::default() },
        Some(false) => {
            let b = bytes[scan - 1];
            let bit = |c: bool| u64::from(c);
            // A fresh '/' is an ordinary punct byte; \r\n are newlines
            // (ws class, po = 0). c2_os as in ascii_carries when P2 is
            // ASCII; a non-ASCII P2 next to a tail byte is rare — defer.
            if scan >= 2 && bytes[scan - 2] >= 0x80 {
                return OCarries {
                    force_bad_lead: true,
                    ..OCarries::default()
                };
            }
            let c2_os = if scan >= 2 {
                let b2 = bytes[scan - 2];
                bit(b2 == b' ' || (!is_letter(b2) && !is_digit(b2) && !is_ascii_ws(b2)))
            } else {
                0
            };
            OCarries {
                po: bit(b == b'/'),
                pws: bit(b != b'/'),
                c2_os,
                ..OCarries::default()
            }
        }
        None => OCarries {
            force_bad_lead: true,
            ..OCarries::default()
        },
    }
}

/// Classify every unicode char whose lead bit is in `m` for
/// `bytes[scan..scan+64]` with the o200k classifier — the o200k analogue
/// of [`mask::classify_uni_chars`] (NUMBERS = false, LEADS = true), with
/// case-split letter masks and marks deferred via `mk`.
///
/// # Safety
///
/// `scan + 70 <= bytes.len()` (the batch classifiers' lookahead guard).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn classify_uni_o200k(bytes: &[u8], scan: usize, mut m: u64) -> OUni {
    use super::decode_cp_inbounds;
    let mut u = OUni::default();
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
        match o200k_class_of(cp) {
            O200kCharClass::Upper => {
                u.l |= chm;
                u.u |= chm;
            }
            O200kCharClass::Lower => u.l |= chm,
            O200kCharClass::Caseless => {
                u.l |= chm;
                u.cl |= chm;
            }
            O200kCharClass::Mark => {
                // Contextual (letter-run joiner AND punct-run member):
                // punct-class for the neighbors' mask algebra, deferred
                // (±4) for everything the context could change.
                u.o |= chm;
                u.mk |= chm;
            }
            O200kCharClass::Number => {
                u.n |= chm;
                u.resid |= chm; // char-counted grouping: scalar
            }
            O200kCharClass::Other => u.o |= chm,
            O200kCharClass::Whitespace => {
                u.ws |= chm;
                if i + l > 64 || l == 4 {
                    u.resid |= chm; // straddling-out ws stays a bad zone
                } else if l == 2 {
                    u.w2 |= lead;
                } else {
                    u.w3 |= lead;
                }
            }
        }
        match l {
            2 => u.lead2 |= lead,
            3 => u.lead3 |= lead,
            _ => u.lead4 |= lead,
        }
        u.cont |= chm & !lead;
        m &= !chm;
    }
    u
}

/// ASCII class masks the o200k algebra needs on top of
/// [`mask::AsciiMasks`]: strict-uppercase letters and slashes.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy, Default)]
struct OAsciiExtra {
    up: u64,
    sl: u64,
}

/// Slow(er) path for batches with non-ASCII in or just before them — the
/// o200k analogue of `cl100k_family::family_extended_masks`: carries walk
/// back through multi-byte chars with the o200k classifier, unicode chars
/// join the effective class masks, then the shared algebra applies.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
fn o200k_extended_masks<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
    am: AsciiMasks,
    ax: OAsciiExtra,
) -> (u64, u64) {
    use super::decode_cp_inbounds;

    /// The char containing byte `pos - 1`: its o200k class, lead index,
    /// and end (exclusive) — [`mask::char_through`] with the o200k
    /// classifier. Same safety contract (`pos > 0`, `pos + 3 <= len` when
    /// `bytes[pos-1]` is non-ASCII; the batch guard covers callers).
    #[inline(always)]
    unsafe fn char_through_o200k(bytes: &[u8], pos: usize) -> (O200kCharClass, usize, usize) {
        let b = bytes[pos - 1];
        if b < 0x80 {
            let cls = if is_upper_ascii(b) {
                O200kCharClass::Upper
            } else if is_letter(b) {
                O200kCharClass::Lower
            } else if is_digit(b) {
                O200kCharClass::Number
            } else if is_ascii_ws(b) {
                O200kCharClass::Whitespace
            } else {
                O200kCharClass::Other
            };
            return (cls, pos - 1, pos);
        }
        let mut j = pos - 1;
        while j > 0 && bytes[j] & 0xC0 == 0x80 {
            j -= 1;
        }
        // SAFETY: j < pos and pos + 3 <= len (contract), so j + 4 <= len.
        let (cp, l) = unsafe { decode_cp_inbounds(bytes, j) };
        (o200k_class_of(cp), j, j + l)
    }

    let mut cl = OUni::default();
    let cr = if scan == 0 {
        OCarries::default()
    } else if bytes[scan - 1] < 0x80 && is_tail_byte(bytes[scan - 1]) {
        tail_carries(bytes, scan)
    } else if bytes[scan - 1] < 0x80 && (scan < 2 || bytes[scan - 2] < 0x80) {
        ascii_carries(bytes, scan)
    } else {
        // A multi-byte char within two bytes of the batch start.
        // SAFETY: scan > 0, and the batch guard covers pos + 3 <= len.
        let (c1, j1, e1) = unsafe { char_through_o200k(bytes, scan) };
        let chm = if e1 > scan { (1u64 << (e1 - scan)) - 1 } else { 0 };
        cl.cont = chm;
        let (c2v, c2_defer) = if j1 == 0 {
            (0, false)
        } else if bytes[j1 - 1] == b'/' {
            // An absorbed tail slash is a token end, not a punct-run char.
            match prev_tail_absorbed(bytes, j1) {
                Some(abs) => (u64::from(!abs), false),
                None => (0, true),
            }
        } else {
            // SAFETY: j1 > 0, j1 < scan keeps the decode in the guard.
            let c2c = unsafe { char_through_o200k(bytes, j1) }.0;
            (
                u64::from(
                    bytes[j1 - 1] == b' '
                        || matches!(c2c, O200kCharClass::Other | O200kCharClass::Mark),
                ),
                // A mark P2 makes the two-back test run-contextual.
                c2c == O200kCharClass::Mark,
            )
        };
        let mut c = OCarries::default();
        if e1 > scan {
            c.b2b_in = c2v << (e1 - scan);
        } else {
            c.c2_os = c2v;
        }
        if c2_defer {
            c.force_bad_lead = true;
        }
        c.pd = u64::from(c1 == O200kCharClass::Number);
        match c1 {
            O200kCharClass::Upper => {
                cl.l |= chm;
                cl.u |= chm;
                c.pl = 1;
                c.pu = 1;
            }
            O200kCharClass::Lower => {
                cl.l |= chm;
                c.pl = 1;
            }
            O200kCharClass::Caseless => {
                cl.l |= chm;
                cl.cl |= chm;
                c.pl = 1;
                c.pcl = 1;
            }
            O200kCharClass::Mark => {
                // Contextual: defer the batch front to the scalar path.
                cl.o |= chm;
                cl.mk |= chm | 1; // bit 0 seeds the ±4 smear even when
                // the mark sits entirely before the batch
                c.po = 1;
            }
            O200kCharClass::Number => {
                cl.n |= chm;
                // A digit char straddling INTO the batch: the leading
                // ASCII digit run's `\p{N}{1,3}` phase started before the
                // batch, and the `pd` seed below can't see it (bit 0 is a
                // continuation byte, not an ASCII digit). Defer via resid
                // so the bad<<1 seed catches the run.
                cl.resid |= chm;
            }
            O200kCharClass::Other => {
                cl.o |= chm;
                c.po = 1;
            }
            O200kCharClass::Whitespace => {
                cl.ws |= chm;
                if e1 > scan {
                    cl.resid |= chm;
                }
                let pb = bytes[scan - 1];
                c.ps = u64::from(pb == b' ');
                let nl = pb == b'\r' || pb == b'\n';
                c.pwt = u64::from(pb < 0x80 && pb != b' ' && !nl || pb >= 0x80);
                c.pws = 1;
            }
        }
        c
    };

    let mut uni = if am.hi != 0 {
        // SAFETY: the batch guard is exactly `classify_uni_o200k`'s
        // contract.
        unsafe { classify_uni_o200k(bytes, scan, am.hi & !cl.cont) }
    } else {
        OUni::default()
    };
    uni.l |= cl.l;
    uni.u |= cl.u;
    uni.cl |= cl.cl;
    uni.n |= cl.n;
    uni.o |= cl.o;
    uni.ws |= cl.ws;
    uni.cont |= cl.cont;
    uni.resid |= cl.resid;
    uni.mk |= cl.mk;

    o200k_algebra::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax, cr, uni)
}

/// The o200k family's shared u64 boundary algebra over per-byte class
/// masks — `cl100k_family::family_algebra` with the o200k rules: casing
/// boundaries inside letter runs, suffix contractions, `[\r\n/]*` punct
/// tails. `uni` is all-zero on the pure-ASCII path.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn o200k_algebra<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
    am: AsciiMasks,
    ax: OAsciiExtra,
    cr: OCarries,
    uni: OUni,
) -> (u64, u64) {
    let OCarries {
        pl,
        pu,
        pcl,
        ps,
        pwt,
        po,
        pws,
        pd,
        c2_os,
        b2b_in,
        p_abs,
        force_bad_lead,
    } = cr;
    let contm = uni.cont;

    // Effective per-byte classes.
    let lb = am.l | uni.l;
    let ub = ax.up | uni.u;
    let clb = uni.cl;
    let sb = am.s;
    let wtb = am.wt | uni.ws;
    let ob = !(am.l | am.d | am.s | am.wt | am.n | am.hi) | uni.o;
    let ws_all = sb | wtb | am.n;

    // --- Absorbed `[\r\n/]*` tails -----------------------------------------
    // Seed: a newline right after a punct byte (a slash after a punct run
    // is already a run member, so tails always begin with a newline), or
    // a tail continuing from before the batch. Smear through [\r\n/].
    let tcls = am.n | ax.sl;
    let abs_seed = (am.n & ((ob << 1) | po)) | (u64::from(p_abs) & tcls);
    let abs_t = if abs_seed == 0 { 0 } else { smear_up(abs_seed, tcls) };
    // Absorbed bytes are no longer punct-run members for any boundary rule.
    let ob_eff = ob & !abs_t;

    // --- Letters -------------------------------------------------------------
    let len1 = !(contm | uni.lead2 | uni.lead3 | uni.lead4);
    let c_test = ((ob_eff | sb) << 1) | po | ps; // bit 0: byte scan-1 in O|S
    let b2back = ((c_test & len1) << 1)
        | ((c_test & uni.lead2) << 2)
        | ((c_test & uni.lead3) << 3)
        | ((c_test & uni.lead4) << 4)
        | c2_os
        | b2b_in;
    let p_l = (lb << 1) | pl;
    let p_u = (ub << 1) | pu;
    let p_cl = (clb << 1) | pcl;
    let p_s = (sb << 1) | ps;
    let p_wt = (wtb << 1) | pwt;
    let p_o = (ob_eff << 1) | po;
    let absorb = p_o & !b2back;
    // Casing boundary: a strict-upper char after a strict-lower one. (For
    // ASCII text this is the whole rule; see the module docs.)
    let p_sl = p_l & !p_u & !p_cl;
    let b_su = ub & !contm & p_sl;
    let b_letters =
        (lb & !contm & !p_l & !p_s & !p_wt & !absorb) | b_su;

    // --- Digits: `\p{N}{1,3}` or `\p{N}` -------------------------------------
    let b_digits = if DIGITS3 && am.d & (am.d >> 1) != 0 {
        mask::digit_run_splits3(am.d)
    } else {
        am.d
    };

    // --- Punct: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` ----------------------------------
    let b_punct = ob_eff & !contm & !p_o & !p_s;

    // --- Bad zones ------------------------------------------------------------
    let resid = uni.resid;
    let mut bad = resid | resid << 1 | resid >> 1;
    // Marks are run-contextual: they, and anything whose boundary rules
    // can see them (up to two chars back — 4 bytes of lookahead for the
    // following leads), go to the scalar path.
    let mk = uni.mk;
    if mk != 0 {
        bad |= mk | mk << 1 | mk << 2 | mk << 3 | mk << 4 | mk >> 1;
    }
    // A strict-upper char after a caseless letter: phase- and
    // lookahead-dependent (see the module docs) — scalar.
    bad |= ub & !contm & ((clb << 1) | pcl);
    if force_bad_lead {
        // Unresolved carries: the leading tail-class run (plus the byte
        // after it) can't be trusted.
        bad |= smear_up(tcls & 1, tcls) << 1 | 0b11;
    }

    // --- Whitespace -----------------------------------------------------------
    let ws_eff = ws_all & !abs_t;

    // Byte-64 lookahead: is the char at the next batch's first byte
    // non-ws? (See cl100k_family for the full reasoning.)
    let nb64 = bytes[scan + 64]; // in bounds: scan + 70 <= len
    let nn64 = if nb64 < 0x80 {
        !is_ascii_ws(nb64)
    } else {
        // SAFETY: the batch guard puts the decode at scan + 64 in bounds.
        bad >> 63 == 0 && unsafe { mask::nn_at_full(bytes, scan + 64) }
    };
    let nn64m = u64::from(nn64).wrapping_neg();

    // An absorbed tail touching the batch end continues iff byte 64 is
    // tail-class; the next batch's `tail_carries` walkback re-derives the
    // context either way, so nothing defers here. A ws run touching the
    // batch end still defers when byte 64 is ws (its last newline may lie
    // beyond this batch).
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX); // whole batch one ws run
        }
        let h = 63 - nonws.leading_zeros(); // highest non-ws bit (< 63)
        bad |= u64::MAX << (h + 1);
    }

    // A digit run whose `\p{N}{1,3}` phase did not start inside this batch
    // (continuation from before it, or after a bad zone) defers.
    if DIGITS3 {
        let seed = (am.d & (bad << 1)) | (am.d & pd);
        if seed != 0 {
            bad |= smear_up(seed, am.d);
        }
    }

    // Base rule (NL-free runs; NL runs are overridden below).
    let ws_leads1 = (am.s | am.wt | am.n) & ws_eff;
    let ws_leads = (ws_leads1 | uni.w2 | uni.w3) & !abs_t;
    let p_ws = (ws_eff << 1) | pws;
    let edge_last = (ws_leads1 & (1 << 63)) | (uni.w2 & (1 << 62)) | (uni.w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws >> 1))
        | (uni.w2 & (nonws >> 2))
        | (uni.w3 & (nonws >> 3))
        | (edge_last & nn64m);
    let mut b_ws = ws_leads & (!p_ws | split_ok);

    // Override every run containing a (non-absorbed) newline: one token
    // through the run's last newline, then tail rules.
    let mut runs_n = am.n & ws_eff & !bad;
    while runs_n != 0 {
        let f = runs_n.trailing_zeros();
        let below_gap = nonws & ((1u64 << f) - 1);
        let a = if below_gap == 0 { 0 } else { 64 - below_gap.leading_zeros() };
        let e = (nonws & (u64::MAX << f)).trailing_zeros();
        let run_mask = (u64::MAX << a) & !u64::MAX.unbounded_shl(e);
        b_ws &= !run_mask;
        b_ws |= 1u64 << a;
        let q = 63 - (am.n & run_mask).leading_zeros(); // last NL in run
        if (q + 1) < e {
            b_ws |= 1u64 << (q + 1);
            let tail = run_mask & (u64::MAX << (q + 1));
            let tail_leads = ws_leads & tail;
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        runs_n &= !run_mask;
    }

    let mut boundary = b_letters | b_digits | b_punct | b_ws;

    // --- Contractions: suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` ----------------
    // An apostrophe right after a letter-run char merges the suffix into
    // that token and forces a boundary right after it. ('ſ is non-ASCII:
    // an apostrophe before any non-ASCII char defers.)
    if CONTRACTIONS {
        let mut cand = am.ap & boundary & p_l & !bad;
        let mut last_forced = usize::MAX;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i <= 2 {
                // The preceding letter could itself end an earlier
                // contraction that started before the batch — scalar.
                bad |= 0b111u64 << i;
                continue;
            }
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            if i == last_forced {
                // "x'll'd": the letter before this apostrophe is a
                // consumed suffix's last char; a new (prefix) match
                // starts here instead.
                continue;
            }
            // The letter before this apostrophe may itself be a consumed
            // suffix's last char resolved where last_forced can't see it
            // (a scalar-walked zone like 'ſ, or a fixup before the
            // batch): locally ambiguous, defer.
            let p = scan + i;
            let prev_suffix_possible = (bytes[p - 2] == b'\''
                && matches!(bytes[p - 1] | 0x20, b's' | b'd' | b'm' | b't'))
                || (p >= 3
                    && bytes[p - 3] == b'\''
                    && (matches!(
                        (bytes[p - 2] | 0x20, bytes[p - 1] | 0x20),
                        (b'l', b'l') | (b'v', b'e') | (b'r', b'e')
                    ) || (bytes[p - 2] == 0xC5 && bytes[p - 1] == 0xBF)));
            if prev_suffix_possible {
                bad |= 0b111u64 << i;
                continue;
            }
            let b1 = bytes[scan + i + 1];
            if b1 >= 0x80 {
                bad |= 0b111u64 << i;
                continue;
            }
            let k = match b1 | 0x20 {
                b's' | b'd' | b'm' | b't' => 2,
                b'l' if bytes[scan + i + 2] | 0x20 == b'l' => 3,
                b'v' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
                b'r' if bytes[scan + i + 2] | 0x20 == b'e' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << i);
                boundary &= !(((1u64 << (k - 1)) - 1) << (i + 1));
                boundary |= 1u64 << (i + k);
                last_forced = i + k;
            }
        }
    }

    (boundary & !bad, bad)
}

// -----------------------------------------------------------------------
// Batch classifiers (per-arch front-ends)
// -----------------------------------------------------------------------

/// Carries for a batch known to have only ASCII in and just before it:
/// tail-class prev bytes route through the walkback, everything else
/// through the branchless ASCII carries.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn ascii_batch_carries(bytes: &[u8], scan: usize) -> OCarries {
    if scan == 0 {
        OCarries::default()
    } else if is_tail_byte(bytes[scan - 1]) {
        tail_carries(bytes, scan)
    } else {
        ascii_carries(bytes, scan)
    }
}

/// `(usable, bad)` for `bytes[scan..scan+64]` under the o200k-family
/// rules — same contract as the cl100k family's `batch_masks`.
///
/// NEON front-end: classifies the ASCII classes (letter, upper, digit,
/// space, whitespace, newline) with movemasks; apostrophe and slash sit
/// behind horizontal any-tests. Batches with non-ASCII in or just before
/// them take [`o200k_extended_masks`].
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn batch_masks<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    use std::arch::aarch64::*;
    let len = bytes.len();
    if scan + 70 > len {
        // Not enough lookahead for the batch-edge char classification.
        return (0, u64::MAX);
    }
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut lv = [zero; 4];
        let mut uv = [zero; 4];
        let mut dv = [zero; 4];
        let mut sv = [zero; 4];
        let mut wsv = [zero; 4];
        let mut nv = [zero; 4];
        let mut hiv = [zero; 4];
        let mut apv = [zero; 4];
        let mut slv = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            lv[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            uv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'A')), vdupq_n_u8(25));
            dv[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            sv[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            wsv[i] = vorrq_u8(
                sv[i],
                vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)),
            );
            nv[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\r')),
                vceqq_u8(v, vdupq_n_u8(b'\n')),
            );
            hiv[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            apv[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
            slv[i] = vceqq_u8(v, vdupq_n_u8(b'/'));
        }
        let l64 = mask::movemask64(lv[0], lv[1], lv[2], lv[3]);
        let u64_ = mask::movemask64(uv[0], uv[1], uv[2], uv[3]);
        let d64 = mask::movemask64(dv[0], dv[1], dv[2], dv[3]);
        let s64 = mask::movemask64(sv[0], sv[1], sv[2], sv[3]);
        let wsa = mask::movemask64(wsv[0], wsv[1], wsv[2], wsv[3]);
        let n64 = mask::movemask64(nv[0], nv[1], nv[2], nv[3]);

        // Apostrophes only matter for the contraction fixup, slashes only
        // for the tail smear: movemask each lazily.
        let ap64 = if CONTRACTIONS {
            let ap_any = vorrq_u8(vorrq_u8(apv[0], apv[1]), vorrq_u8(apv[2], apv[3]));
            if vmaxvq_u8(ap_any) != 0 {
                mask::movemask64(apv[0], apv[1], apv[2], apv[3])
            } else {
                0
            }
        } else {
            0
        };
        let sl_any = vorrq_u8(vorrq_u8(slv[0], slv[1]), vorrq_u8(slv[2], slv[3]));
        let sl64 = if vmaxvq_u8(sl_any) != 0 {
            mask::movemask64(slv[0], slv[1], slv[2], slv[3])
        } else {
            0
        };

        let am = AsciiMasks {
            l: l64,
            d: d64,
            s: s64,
            wt: wsa & !s64 & !n64,
            n: n64,
            hi: 0,
            ap: ap64,
        };
        let ax = OAsciiExtra { up: u64_, sl: sl64 };

        let hi_any = vorrq_u8(vorrq_u8(hiv[0], hiv[1]), vorrq_u8(hiv[2], hiv[3]));
        if vmaxvq_u8(hi_any) != 0
            || (scan >= 1 && bytes[scan - 1] >= 0x80)
            || (scan >= 2 && bytes[scan - 2] >= 0x80)
        {
            let mut am = am;
            am.hi = mask::movemask64(hiv[0], hiv[1], hiv[2], hiv[3]);
            return o200k_extended_masks::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax);
        }

        let cr = ascii_batch_carries(bytes, scan);
        o200k_algebra::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax, cr, OUni::default())
    }
}

/// x86-64 front-end: same contract as the NEON `batch_masks` above,
/// dispatching on the runtime-detected SIMD tier (AVX-512 or AVX2).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn batch_masks<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    debug_assert!(mask::simd_scanner_available());
    if mask::avx512_scanner_available() {
        // SAFETY: runtime AVX-512 detection right above.
        unsafe { batch_masks_avx512::<CONTRACTIONS, DIGITS3>(bytes, scan) }
    } else {
        // SAFETY: MaskState enables the mask-scanner path only after
        // runtime detection; without AVX-512 that detection was AVX2's.
        unsafe { batch_masks_avx2::<CONTRACTIONS, DIGITS3>(bytes, scan) }
    }
}

/// The strict-uppercase and slash masks for `bytes[scan..scan+64]`,
/// AVX-512 tier.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn ascii_extra_avx512(bytes: &[u8], scan: usize) -> OAsciiExtra {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(scan) as *const _);
        let up = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'A' as i8)),
            _mm512_set1_epi8(25),
        );
        let sl = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'/' as i8));
        OAsciiExtra { up, sl }
    }
}

/// The strict-uppercase and slash masks for `bytes[scan..scan+64]`,
/// AVX2 tier. `#[inline(never)]` for the same vector-domain reason as
/// [`mask::ascii_masks_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
fn ascii_extra_avx2(bytes: &[u8], scan: usize) -> OAsciiExtra {
    use std::arch::x86_64::*;
    unsafe {
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
        let ca = _mm256_set1_epi8(b'A' as i8);
        let c25 = _mm256_set1_epi8(25);
        let up = mm(
            le(_mm256_sub_epi8(v0, ca), c25),
            le(_mm256_sub_epi8(v1, ca), c25),
        );
        let slc = _mm256_set1_epi8(b'/' as i8);
        let sl = mm(_mm256_cmpeq_epi8(v0, slc), _mm256_cmpeq_epi8(v1, slc));
        OAsciiExtra { up, sl }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn batch_masks_avx512<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    let len = bytes.len();
    if scan + 70 > len {
        return (0, u64::MAX);
    }
    let am = mask::ascii_masks_avx512(bytes, scan);
    let ax = ascii_extra_avx512(bytes, scan);
    if am.hi != 0
        || (scan >= 1 && bytes[scan - 1] >= 0x80)
        || (scan >= 2 && bytes[scan - 2] >= 0x80)
    {
        return o200k_extended_masks::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax);
    }
    let cr = ascii_batch_carries(bytes, scan);
    o200k_algebra::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax, cr, OUni::default())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn batch_masks_avx2<const CONTRACTIONS: bool, const DIGITS3: bool>(
    bytes: &[u8],
    scan: usize,
) -> (u64, u64) {
    let len = bytes.len();
    if scan + 70 > len {
        return (0, u64::MAX);
    }
    let am = mask::ascii_masks_avx2(bytes, scan);
    let ax = ascii_extra_avx2(bytes, scan);
    if am.hi != 0
        || (scan >= 1 && bytes[scan - 1] >= 0x80)
        || (scan >= 2 && bytes[scan - 2] >= 0x80)
    {
        return o200k_extended_masks::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax);
    }
    let cr = ascii_batch_carries(bytes, scan);
    o200k_algebra::<CONTRACTIONS, DIGITS3>(bytes, scan, am, ax, cr, OUni::default())
}

#[cfg(test)]
mod tests {
    use crate::pretokenize::fast::mask::{MaskScheme, MaskState};
    use crate::pretokenize::fast::nemotron::NemotronScheme;
    use crate::pretokenize::fast::o200k::O200kScheme;

    fn scalar_tokens<S: MaskScheme>(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut pos = 0;
        let mut out = vec![];
        while pos < bytes.len() {
            let e = S::advance(bytes, pos);
            out.push(bytes[pos..e].to_vec());
            pos = e;
        }
        out
    }

    fn mask_tokens<S: MaskScheme>(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut st = MaskState::new(0);
        let mut out = vec![];
        while let Some((s, e)) = st.next_span::<S>(bytes) {
            out.push(bytes[s..e].to_vec());
        }
        out
    }

    /// Scalar-vs-mask token streams for both family schemes.
    #[track_caller]
    fn check_all(buf: &[u8]) {
        for (name, a, b) in [
            ("o200k", scalar_tokens::<O200kScheme>(buf), mask_tokens::<O200kScheme>(buf)),
            ("nemotron", scalar_tokens::<NemotronScheme>(buf), mask_tokens::<NemotronScheme>(buf)),
        ] {
            if a != b {
                let i = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                panic!(
                    "{name} diverged at token {i} on {:?}\n  scalar: {:?}\n  mask:   {:?}",
                    String::from_utf8_lossy(buf),
                    a.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
                    b.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
                );
            }
        }
    }

    /// Alignment-preserving shrink: replace chars with 'a' and trim the
    /// tail while the buffer still diverges, then report the minimum.
    fn shrink_and_report(bytes: &[u8]) -> String {
        let fails = |b: &[u8]| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_all(b))).is_err()
        };
        let mut buf = bytes.to_vec();
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i < buf.len() {
                let mut l = 1;
                if buf[i] >= 0xC2 {
                    l = if buf[i] < 0xE0 { 2 } else if buf[i] < 0xF0 { 3 } else { 4 };
                }
                let saved: Vec<u8> = buf[i..i + l].to_vec();
                if saved != vec![b'a'; l] {
                    for j in 0..l {
                        buf[i + j] = b'a';
                    }
                    if fails(&buf) {
                        changed = true;
                    } else {
                        buf[i..i + l].copy_from_slice(&saved);
                    }
                }
                i += l;
            }
            while buf.len() > 1 && buf[buf.len() - 1] == b'a' && fails(&buf[..buf.len() - 1]) {
                buf.pop();
                changed = true;
            }
        }
        let mut report = format!("(len {}) {:?}", buf.len(), String::from_utf8_lossy(&buf));
        for (name, a, b) in [
            ("o200k", scalar_tokens::<O200kScheme>(&buf), mask_tokens::<O200kScheme>(&buf)),
            (
                "nemotron",
                scalar_tokens::<NemotronScheme>(&buf),
                mask_tokens::<NemotronScheme>(&buf),
            ),
        ] {
            if a != b {
                let i = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                report.push_str(&format!(
                    "\n  {name} token {i}: scalar {:?} mask {:?}",
                    a.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
                    b.get(i).map(|t| String::from_utf8_lossy(t).into_owned()),
                ));
            }
        }
        report
    }

    /// Crafted cases, padded so they cross the batch (not scalar-tail) path.
    #[test]
    fn o200k_family_mask_matches_scalar_padded_cases() {
        let pad = "The quick brown fox jumps over the lazy dog again and again. ";
        let cases = [
            "January 24, 2015 and 12345678 numbers 1 22 333 4444",
            "don't DON'T they'Ll 'sound 'lx x'y '' ' \u{2019}s x'll'd don't's",
            "camelCase HTTPResponse PascalCase aB ABc parseHTMLDocument iOS",
            "!hello !Hello !!hello ?!x a-b ... !!!\n\nnext",
            "paths /usr/bin/env a/b a//b http://x.com/y .\n//x !\n/\n/x",
            "//\n/ x/\n x\n/ \n/ don't\n don'\n",
            "tabs\tand\nnewlines\r\n mixed  \n  runs \n\n\n deep",
            "hi!\n\ndef hi !!\n\nabc \"quoted\" (paren)",
            "caf\u{e9} r\u{e9}sum\u{e9} \u{201c}word\u{201d} \u{2014}dash\u{2013} \u{00a0}nbsp",
            "CAF\u{c9} \u{2003}em \u{2009}thin\u{2028}ls x\u{e9}\u{e9}y \u{416}dz Z\u{5bf}\u{416}dz",
            "price: $5.99! 100,000.00 3.14159 2nd 3rd 4th",
            "a\u{2028}b a\u{2028}\n \n\n\t x \n\n ",
            "mixed 1\u{662}3x \u{661}\u{662}\u{663} arabic",
            "\u{65e5}\u{672c}\u{8a9e}ABC abc\u{65e5}\u{672c}Def \u{416}\u{416}dz",
            "e\u{301}f !!\u{301}x '\u{301}s x\u{301}'s A\u{301}B",
            // A mark's contextual class reaching a boundary past the ±4
            // bad smear (4-byte punct char in between): relies on the
            // walker's scalar-overrun masking.
            "a\u{301}\u{1f389}b !\u{301}\u{1f389}x a\u{301}\u{1f389}'s a\u{301}\n\n\n\n/x",
        ];
        for case in cases {
            for lead in [0usize, 1, 37, 61, 62, 63, 64, 65] {
                let mut buf = pad.as_bytes().repeat(4)[..pad.len() * 2 + lead].to_vec();
                buf.extend_from_slice(case.as_bytes());
                buf.extend_from_slice(pad.as_bytes());
                buf.extend_from_slice(case.as_bytes());
                check_all(&buf);
            }
        }
    }

    /// Differential fuzz across both family schemes.
    #[test]
    fn o200k_family_mask_matches_scalar_fuzz() {
        let pieces: &[&str] = &[
            "a", "B", "z", "9", "0", " ", "  ", "\n", "\t", "\r\n", "\r", "'", "'s", "'LL",
            "'t", "!", ".", ",", "(", "/", "//", "/\n", "\n/", "é", "É", "ß", "日", "🎉",
            "\u{00A0}", "\u{2003}", "word", "Word", "WORD", "camelCase", "12", "1234", "’",
            "“", "”", "–", "—", "…", "\u{2009}", "\u{200B}", "\u{2028}", "\u{202F}", "×",
            "÷", "«", "µ", "café", "éé", "naïve", "Α", "α", "а", "А", "Ж", "ж", "ǅ", "\n\n",
            "!x", "!X", "\tx", " x", " X", "?!", "\u{301}", "ſ", "'ſ", "'\u{301}",
            "\u{661}\u{662}", "\u{FF11}", "क", "\u{940}", "\u{1D54F}", "€", "™",
            "…\u{2028}", "AxB", "ABc", "aB", "\u{416}dz", "x'", "n't",
        ];
        let mut state = 0x243F6A8885A308D3u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..4000 {
            let target = 80 + (round % 400);
            let mut buf = Vec::new();
            while buf.len() < target {
                buf.extend_from_slice(pieces[(rng() % pieces.len() as u64) as usize].as_bytes());
            }
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_all(&buf)))
                .is_ok();
            if !ok {
                panic!(
                    "round {round} diverged; minimal case: {}",
                    shrink_and_report(&buf)
                );
            }
        }
    }
}

#[cfg(test)]
mod owt_tests {
    use crate::pretokenize::fast::mask::{MaskScheme, MaskState};
    use crate::pretokenize::fast::nemotron::NemotronScheme;
    use crate::pretokenize::fast::o200k::O200kScheme;

    fn check_streaming<S: MaskScheme>(bytes: &[u8], scheme: &str) {
        let mut st = MaskState::new(0);
        let mut pos = 0usize;
        let mut idx = 0usize;
        while pos < bytes.len() {
            let scalar_end = S::advance(bytes, pos);
            match st.next_span::<S>(bytes) {
                Some((s, e)) => assert!(
                    s == pos && e == scalar_end,
                    "{scheme} diverged at token {idx} (byte {pos}): scalar {pos}..{scalar_end} \
                     mask {s}..{e}: {:?} vs {:?}",
                    String::from_utf8_lossy(&bytes[pos..scalar_end]),
                    String::from_utf8_lossy(&bytes[s..e]),
                ),
                None => panic!("{scheme} ended early at token {idx} (byte {pos})"),
            }
            pos = scalar_end;
            idx += 1;
        }
        assert!(st.next_span::<S>(bytes).is_none(), "{scheme} produced extra tokens");
        eprintln!("{scheme}: all {idx} tokens match");
    }

    fn check_streaming_all(bytes: &[u8]) {
        check_streaming::<O200kScheme>(bytes, "o200k");
        check_streaming::<NemotronScheme>(bytes, "nemotron");
    }

    /// 100 MB of OWT, mask vs scalar.
    #[test]
    #[ignore]
    fn o200k_family_mask_matches_scalar_owt() {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        use std::io::Read;
        let f = std::fs::File::open(&path).unwrap();
        let mut input = Vec::new();
        f.take(100_000_000).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        check_streaming_all(&input);
    }

    /// Full-OWT (~12 GB) mask-vs-scalar differential for both schemes.
    #[test]
    #[ignore = "reads the full ~12 GB OWT file"]
    fn o200k_family_mask_matches_scalar_owt_full() {
        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let input = std::fs::read(&path).expect("Could not read ~/data/owt_train.txt");
        eprintln!("loaded {} bytes", input.len());
        check_streaming_all(&input);
    }
}

