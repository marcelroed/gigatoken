use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

use crate::bpe::Tokenizer;
use crate::pretokenize::pretoken_fast::FastPretokenizer;
use crate::token::TokenId;

// ---------------------------------------------------------------------------
// High-throughput streaming encoder
//
// 32-byte cache-aligned probe slots (2 per cache line). Each slot stores:
// - 8-byte fingerprint for fast rejection during probing
// - 8-byte pretoken prefix for inline byte verification (covers <=8 byte pretokens
//   without pointer chase; longer pretokens verified via entries array)
// - 4-byte token value or tok_store offset
// - 4-byte entry index (for >8 byte verification)
// - 2-byte pretoken length, 2-byte token count
//
// Other optimizations:
// - wyhash-style hash (1 multiply for short keys)
// - Raw pointer output writes (no Vec overhead per token)
// - Single-token inline storage in slot (no tok_store read for 1-token results)
// ---------------------------------------------------------------------------

#[inline(always)]
fn fast_hash(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    if len <= 8 {
        let (a, b) = if len >= 4 {
            let lo = unsafe { (ptr as *const u32).read_unaligned() } as u64;
            let hi = unsafe { ((ptr.add(len - 4)) as *const u32).read_unaligned() } as u64;
            (lo, hi)
        } else if len >= 2 {
            let lo = unsafe { (ptr as *const u16).read_unaligned() } as u64;
            let hi = unsafe { ((ptr.add(len - 2)) as *const u16).read_unaligned() } as u64;
            (lo, hi)
        } else {
            (bytes[0] as u64, 0)
        };
        wymix(a ^ 0xa0761d6478bd642f, b ^ 0xe7037ed1a0b428db) ^ (len as u64).wrapping_mul(0x9e3779b97f4a7c15)
    } else {
        let mut h: u64 = len as u64 ^ 0xa0761d6478bd642f;
        let mut i = 0;
        while i + 8 <= len {
            let w = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
            h = wymix(h ^ w, 0xe7037ed1a0b428db);
            i += 8;
        }
        if i < len {
            let tail = unsafe { (ptr.add(len - 8.min(len)) as *const u64).read_unaligned() };
            h = wymix(h ^ tail, 0x8a5cd789635d2dff);
        }
        h
    }
}

#[inline(always)]
fn wymix(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ (r >> 64) as u64
}

/// Read the pretoken as a packed u64 (zero-padded if < 8 bytes).
/// Uses overlapping reads to minimise branches.
#[inline(always)]
fn pack_prefix8(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    if len >= 8 {
        unsafe { (ptr as *const u64).read_unaligned() }
    } else if len >= 4 {
        let lo = unsafe { (ptr as *const u32).read_unaligned() } as u64;
        let hi = unsafe { ((ptr.add(len - 4)) as *const u32).read_unaligned() } as u64;
        lo | (hi << 32)
    } else if len >= 2 {
        let lo = unsafe { (ptr as *const u16).read_unaligned() } as u64;
        let hi = bytes[len - 1] as u64;
        lo | (hi << 16)
    } else if len == 1 {
        bytes[0] as u64
    } else {
        0
    }
}

/// 32-byte cache-aligned probe slot.
#[derive(Copy, Clone)]
#[repr(C, align(32))]
struct Slot {
    fp: u64,         // 8: hash | 1; 0 = empty
    prefix8: u64,    // 8: first 8 bytes of pretoken, packed
    tok_or_idx: u32, // 4: n_tok==1 → token value; n_tok>1 → tok_store index
    entry_idx: u32,  // 4: index into entries (for >8 byte verification)
    pt_len: u16,     // 2
    n_tok: u16,      // 2
    _pad: u32,       // 4
}

impl Slot {
    const EMPTY: Self = Slot {
        fp: 0, prefix8: 0, tok_or_idx: 0, entry_idx: 0,
        pt_len: 0, n_tok: 0, _pad: 0,
    };
}

/// Pointer to canonical pretoken bytes (only needed for pretokens > 8 bytes).
#[derive(Copy, Clone)]
struct EntryPtr(*const u8);
unsafe impl Send for EntryPtr {}
unsafe impl Sync for EntryPtr {}

/// Encode all lines, returning (flat token buffer, line-boundary offsets).
pub fn encode_lines(lines: &[&[u8]], tokenizer: &Tokenizer) -> (Vec<TokenId>, Vec<usize>) {
    let merges = &tokenizer.merges;
    let remap = tokenizer.byte_remapping.as_ref().map(|br| br.mapping.as_slice());
    let b2t: [TokenId; 256] = {
        let mut t = [TokenId(0); 256];
        for i in 0..256 {
            t[i] = TokenId(match remap { Some(r) => r[i] as u32, None => i as u32 });
        }
        t
    };

    let total_bytes: usize = lines.iter().map(|l| l.len()).sum();
    let est_unique = (total_bytes / 300).max(4096);
    let cap = (est_unique * 2).next_power_of_two().max(4096);
    let mut mask = cap - 1;

    let mut slots: Vec<Slot> = vec![Slot::EMPTY; cap];
    let mut entries: Vec<EntryPtr> = Vec::with_capacity(est_unique);
    let mut tok_store: Vec<TokenId> = Vec::with_capacity(est_unique * 2);
    let mut scratch: Vec<TokenId> = Vec::with_capacity(128);
    let mut n_entries = 0usize;

    let init_cap = (total_bytes * 3 / 10).max(1024);
    let mut output: Vec<TokenId> = Vec::with_capacity(init_cap);
    let mut out_base = output.as_mut_ptr();
    let mut out_len = 0usize;
    let mut out_cap = output.capacity();

    let mut boundaries: Vec<usize> = Vec::with_capacity(lines.len() + 1);
    boundaries.push(0);

    for &line in lines {
        if out_len + line.len() > out_cap {
            unsafe { output.set_len(out_len); }
            output.reserve(line.len() + out_cap / 4);
            out_base = output.as_mut_ptr();
            out_cap = output.capacity();
        }

        let mut pt = FastPretokenizer::new(line);
        while let Some(pretoken) = pt.next() {
            let bytes = pretoken.0;

            if bytes.len() == 1 {
                unsafe { *out_base.add(out_len) = b2t[bytes[0] as usize]; }
                out_len += 1;
                continue;
            }

            let fp = fast_hash(bytes) | 1;
            let blen = bytes.len() as u16;
            let prefix = pack_prefix8(bytes);
            let mut si = fp as usize & mask;

            loop {
                let slot = unsafe { slots.get_unchecked(si) };
                if slot.fp == 0 {
                    // ---- Cache miss: BPE encode ----
                    scratch.clear();
                    for &b in bytes { scratch.push(b2t[b as usize]); }
                    bpe_merge(&mut scratch, merges);
                    let nt = scratch.len();

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            scratch.as_ptr(), out_base.add(out_len), nt,
                        );
                    }
                    out_len += nt;

                    let ei = entries.len() as u32;
                    entries.push(EntryPtr(bytes.as_ptr()));

                    let new_slot = if nt == 1 {
                        Slot { fp, prefix8: prefix, tok_or_idx: scratch[0].0, entry_idx: ei, pt_len: blen, n_tok: 1, _pad: 0 }
                    } else {
                        let ts = tok_store.len() as u32;
                        tok_store.extend_from_slice(&scratch);
                        Slot { fp, prefix8: prefix, tok_or_idx: ts, entry_idx: ei, pt_len: blen, n_tok: nt as u16, _pad: 0 }
                    };
                    slots[si] = new_slot;
                    n_entries += 1;

                    if n_entries * 2 > slots.len() {
                        grow(&mut slots, &mut mask);
                    }
                    break;
                }
                if slot.fp == fp && slot.pt_len == blen && slot.prefix8 == prefix {
                    // For <= 8 bytes, prefix8 is the complete content, so this is verified.
                    // For > 8 bytes, compare the tail via the entry pointer.
                    let verified = blen <= 8 || {
                        let p = unsafe { entries.get_unchecked(slot.entry_idx as usize).0 };
                        let tail = unsafe { std::slice::from_raw_parts(p.add(8), blen as usize - 8) };
                        tail == &bytes[8..]
                    };
                    if verified {
                        let n = slot.n_tok as usize;
                        if n == 1 {
                            unsafe { *out_base.add(out_len) = TokenId(slot.tok_or_idx); }
                            out_len += 1;
                        } else {
                            let s = slot.tok_or_idx as usize;
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    tok_store.as_ptr().add(s), out_base.add(out_len), n,
                                );
                            }
                            out_len += n;
                        }
                        break;
                    }
                }
                si = (si + 1) & mask;
            }
        }
        boundaries.push(out_len);
    }
    unsafe { output.set_len(out_len); }

    (output, boundaries)
}

fn grow(slots: &mut Vec<Slot>, mask: &mut usize) {
    let new_cap = slots.len() * 2;
    let new_mask = new_cap - 1;
    let mut new_slots = vec![Slot::EMPTY; new_cap];
    for &s in slots.iter() {
        if s.fp == 0 { continue; }
        let mut i = s.fp as usize & new_mask;
        while new_slots[i].fp != 0 { i = (i + 1) & new_mask; }
        new_slots[i] = s;
    }
    *slots = new_slots;
    *mask = new_mask;
}

#[inline]
fn bpe_merge(symbols: &mut Vec<TokenId>, merges: &HashMap<(TokenId, TokenId), TokenId, FxBuildHasher>) {
    let mut len = symbols.len();
    if len < 2 { return; }
    loop {
        let mut best_rank = u32::MAX;
        let mut best_pos = 0;
        for i in 0..len - 1 {
            if let Some(&m) = merges.get(&(symbols[i], symbols[i + 1])) {
                if m.0 < best_rank { best_rank = m.0; best_pos = i; }
            }
        }
        if best_rank == u32::MAX { break; }
        symbols[best_pos] = TokenId(best_rank);
        symbols.remove(best_pos + 1);
        len -= 1;
    }
}
