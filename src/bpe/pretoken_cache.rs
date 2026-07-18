//! Open-addressing cache for short (≤ 15 byte) pretoken encodings. Three
//! properties of the encode loop drive the design (measured on 1 GB OWT,
//! Zen 2):
//!
//! - The table holds ~1.3M unique pretokens (~99.4% hit rate), far beyond
//!   L2/L3, so a lookup in the Zipf tail is a random DRAM access. hashbrown
//!   spends two cache lines per probe (control bytes + entry); this table's
//!   32-byte entries are self-contained and bucketed into line-aligned
//!   pairs, so a probe touches exactly one line, and the two prefetch
//!   flavors let the encode loop stage that line ([`Self::prefetch_l2`] a
//!   chunk ahead, [`Self::prefetch`] a few probes ahead) instead of
//!   stalling on it.
//! - 228M output tokens / 208M pretokens: ~90% of pretokens encode to ONE
//!   token and ~98% to at most two. The value is a packed `u64` plus an
//!   extension word (see `tiktoken::pack_val_inline`) holding up to four
//!   tokens inline — one dependent load, and no second random access into
//!   the token arena.
//! - At ~64 MB the table also blows the dTLB through 4 KiB pages, so the
//!   backing memory is 2 MiB-aligned and `MADV_HUGEPAGE`d — with THP
//!   available the whole table sits in a few dozen dTLB entries. (Note:
//!   processes launched under `PR_SET_THP_DISABLE` — some sandboxes and
//!   session managers do this — silently get 4 KiB pages anyway.)
//!
//! Linear probing over aligned pairs: a bucket is slots `idx` and `idx + 1`
//! with `idx` even, so both share one 64 B line, and [`Self::probe_pair`]
//! resolves the overwhelmingly common displacement-0/1 hit branch-free from
//! that single line. Inserts fill the first empty slot of the walk; growth
//! doubles at 3/4 load. Key 0 marks empty slots: a real key always has its
//! nonzero length in the top byte (`pack_pretoken_key` tags length; empty
//! pretokens pack to key 0, which the encode loop routes to the long map,
//! never here).

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ptr::NonNull;

/// One slot: the packed pretoken key plus its packed encoding — `val`
/// (count, spill flag, tokens 1-2) and `ext` (tokens 3-4, see
/// `tiktoken::pack_val_inline`). Exactly 32 bytes: two slots per cache
/// line, never straddling one.
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    val: u64,
    ext: u64,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);

const EMPTY_KEY: u128 = 0;

/// The table's slot array: a manually managed, zeroed (== all-empty,
/// since `EMPTY_KEY` is 0), 2 MiB-aligned allocation marked
/// `MADV_HUGEPAGE`. A plain `Box<[Entry]>` can neither over-align nor
/// keep dealloc's layout in sync with an over-aligned alloc.
struct Slots {
    ptr: NonNull<Entry>,
    cap: usize,
}

impl Slots {
    const HUGE_PAGE: usize = 2 * 1024 * 1024;

    fn new_zeroed(cap: usize) -> Self {
        let layout = Self::layout(cap);
        // SAFETY: layout has nonzero size (cap >= 1).
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw as *mut Entry) else {
            handle_alloc_error(layout)
        };
        // Hint huge pages BEFORE first touch. `alloc_zeroed` on a 2 MiB-
        // aligned layout is aligned_alloc + an explicit memset that faults
        // the whole fresh mapping in as 4 KiB pages, after which the hint
        // is a no-op for this run (khugepaged collapses far too slowly to
        // matter): the table then walks the dTLB on every probe, and Zen
        // drops software prefetches that miss the dTLB — measured +15%
        // cold / +7% warm encode from this ordering alone (see
        // profiling/zen5_st_profile.md §3). Madvised first, the zeroing
        // write below faults it in as 2 MiB pages.
        super::madvise_hugepage(raw, layout.size());
        // SAFETY: raw is a live allocation of exactly layout.size() bytes.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        Self { ptr, cap }
    }

    fn layout(cap: usize) -> Layout {
        let size = cap * std::mem::size_of::<Entry>();
        // Huge-page alignment only once the table outgrows one huge page;
        // small tables (fresh tokenizers encoding little text) stay modest.
        // Floor of 64 so an even-indexed pair always shares one cache line.
        let align = Self::HUGE_PAGE.min(size.next_power_of_two()).max(64);
        Layout::from_size_align(size, align).expect("table layout overflow")
    }

    #[inline(always)]
    unsafe fn get(&self, idx: usize) -> &Entry {
        debug_assert!(idx < self.cap);
        // SAFETY: caller guarantees idx < cap.
        unsafe { &*self.ptr.as_ptr().add(idx) }
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, idx: usize) -> &mut Entry {
        debug_assert!(idx < self.cap);
        // SAFETY: caller guarantees idx < cap.
        unsafe { &mut *self.ptr.as_ptr().add(idx) }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        // SAFETY: allocated in `new_zeroed` with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.cap)) };
    }
}

// SAFETY: Slots owns its allocation exclusively, like Box<[Entry]>.
unsafe impl Send for Slots {}
unsafe impl Sync for Slots {}

/// Request the line holding `p` into L1 (`L1 = true`) or L2 only — the
/// shared ladder behind [`ShortPretokenCache::prefetch_l2`] and
/// [`ProbeView::prefetch`]. No-op on arches without a prefetch hint.
#[inline(always)]
fn prefetch_line<const L1: bool>(p: *const Entry) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
        if L1 {
            _mm_prefetch(p as *const i8, _MM_HINT_T0);
        } else {
            _mm_prefetch(p as *const i8, _MM_HINT_T1);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        if L1 {
            core::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        } else {
            core::arch::asm!(
                "prfm pldl2keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

pub(crate) struct ShortPretokenCache {
    slots: Slots,
    /// `cap - 1` (capacity is a power of two).
    mask: usize,
    len: usize,
}

impl ShortPretokenCache {
    fn with_pow2_capacity(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two() && cap >= 2);
        Self { slots: Slots::new_zeroed(cap), mask: cap - 1, len: 0 }
    }

    /// A table sized to hold at least `n` entries without growing (same
    /// 3/4-load threshold as [`Self::insert`], with a 2^16-slot
    /// floor), starting from at least `min_slots` slots. The
    /// vocab-seeding path inserts ~50k entries up front; sizing for them
    /// avoids rehashing mid-seed. `min_slots` lets a parallel worker start
    /// at the final capacity expected for its share of the input (see
    /// `Tokenizer::fork_sized`), skipping the doubling-rehash churn of a
    /// cold run — a starting point only, the table still grows past it at
    /// 3/4 load. Either way the table is constructed exactly once, at the
    /// max of the two requirements.
    pub(crate) fn with_at_least(n: usize, min_slots: usize) -> Self {
        let mut cap = min_slots.max(1 << 16).next_power_of_two();
        while (n + 1) * 4 > cap * 3 {
            cap *= 2;
        }
        Self::with_pow2_capacity(cap)
    }

    /// Address of `h`'s home pair; both slots share the addressed line.
    #[inline(always)]
    fn pair_ptr(&self, h: u64) -> *const Entry {
        // SAFETY: the masked even index is <= mask - 1 < cap.
        unsafe { self.slots.ptr.as_ptr().add((h as usize) & self.mask & !1) }
    }

    /// Request the probe's cache line into L2 only. The encode loop calls
    /// this a full chunk (hundreds of cycles) before the probe — enough to
    /// cover DRAM — without evicting the span walker's L1 working set the
    /// way a chunk's worth of L1 prefetches would.
    #[inline(always)]
    pub(crate) fn prefetch_l2(&self, h: u64) {
        prefetch_line::<false>(self.pair_ptr(h));
    }

    /// A raw, `Copy` snapshot of the probe parameters for the emit loop's
    /// hot path. Holding one across a chunk keeps the table base and mask
    /// in registers instead of being reloaded from `self` every iteration
    /// (the slow path's `&mut self` calls otherwise force the reload).
    /// Invalidated by [`Self::insert`] (which may grow the table): callers
    /// must take a fresh view after any insert.
    pub(crate) fn probe_view(&self) -> ProbeView {
        ProbeView { base: self.slots.ptr.as_ptr(), pair_mask: self.mask & !1 }
    }

    /// Look up `key`, walking pairs from its home bucket. Inserts fill the
    /// first empty slot of the walk, so any pair holding an empty slot
    /// terminates it. A miss (`Err`) also reports where the key belongs —
    /// the first empty slot of the walk, which is exactly what
    /// [`Self::first_empty`] would find (every pair before the
    /// terminating one was full), discovered by loads the lookup performs
    /// anyway. [`Self::insert_at`] then skips re-walking the chain. The
    /// slot stays valid until the next insert or grow: lookups never
    /// mutate, and the encode miss path computes the entry's value
    /// without touching the table.
    pub(crate) fn get_or_slot(&self, key: u128, h: u64) -> Result<(u64, u64), usize> {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            let e0 = unsafe { self.slots.get(idx) };
            let e1 = unsafe { self.slots.get(idx + 1) };
            if e0.key == key {
                return Ok((e0.val, e0.ext));
            }
            if e1.key == key {
                return Ok((e1.val, e1.ext));
            }
            if e0.key == EMPTY_KEY {
                return Err(idx);
            }
            if e1.key == EMPTY_KEY {
                return Err(idx + 1);
            }
            idx = (idx + 2) & self.mask;
        }
    }

    /// First empty slot of `h`'s pair walk (load < 1 guarantees one).
    fn first_empty(&self, h: u64) -> usize {
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            unsafe {
                if self.slots.get(idx).key == EMPTY_KEY {
                    return idx;
                }
                if self.slots.get(idx + 1).key == EMPTY_KEY {
                    return idx + 1;
                }
            }
            idx = (idx + 2) & self.mask;
        }
    }

    /// Insert a key known to be absent (the encode loop only inserts after
    /// a [`Self::get_or_slot`] miss).
    pub(crate) fn insert(&mut self, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
        }
        let idx = self.first_empty(h);
        // SAFETY: first_empty returns an in-bounds index.
        unsafe { *self.slots.get_mut(idx) = Entry { key, val, ext } };
        self.len += 1;
    }

    /// [`Self::insert`] with the destination already known from a
    /// [`Self::get_or_slot`] miss on the same `key`/`h` (with no insert or
    /// grow in between), skipping the `first_empty` chain walk. A growth
    /// pass invalidates `slot`, so that branch recomputes it.
    pub(crate) fn insert_at(&mut self, slot: usize, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut slot = slot;
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
            slot = self.first_empty(h);
        }
        debug_assert_eq!(slot, self.first_empty(h));
        // SAFETY: get_or_slot and first_empty return in-bounds indices.
        unsafe { *self.slots.get_mut(slot) = Entry { key, val, ext } };
        self.len += 1;
    }

    /// Insert `key`, overwriting its value if the key is already present
    /// (the plain [`Self::insert`] assumes absence). Cold loader-phase
    /// entry point for the vocab-seed sync (`set_added_tokens`' overwrite
    /// and restore loops and `fork_sized`'s added-token re-apply), where
    /// an added token's content can duplicate an already-seeded vocab
    /// byte string and must take over its entry.
    pub(crate) fn replace(&mut self, key: u128, h: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            let (k0, k1) = unsafe { (self.slots.get(idx).key, self.slots.get(idx + 1).key) };
            if k0 == key {
                // SAFETY: idx is in bounds (masked above).
                unsafe { *self.slots.get_mut(idx) = Entry { key, val, ext } };
                return;
            }
            if k1 == key {
                // SAFETY: idx + 1 <= mask (masked, even idx).
                unsafe { *self.slots.get_mut(idx + 1) = Entry { key, val, ext } };
                return;
            }
            if k0 == EMPTY_KEY || k1 == EMPTY_KEY {
                // Absent: a fresh insert (with its own growth check).
                self.insert(key, h, val, ext);
                return;
            }
            idx = (idx + 2) & self.mask;
        }
    }

    #[cold]
    fn grow(&mut self) {
        self.grow_to(self.slots.cap * 2);
    }

    /// Rebuild the table at `new_cap` slots, reinserting every live entry.
    #[cold]
    fn grow_to(&mut self, new_cap: usize) {
        debug_assert!(new_cap.is_power_of_two() && new_cap > self.slots.cap);
        let old = std::mem::replace(&mut self.slots, Slots::new_zeroed(new_cap));
        self.mask = new_cap - 1;
        for i in 0..old.cap {
            // SAFETY: i < old.cap.
            let e = *unsafe { old.get(i) };
            if e.key == EMPTY_KEY {
                continue;
            }
            // Must be the same hash the inserts' `h` came from.
            let idx = self.first_empty(crate::pretokenize::pretoken_key_hash(e.key));
            // SAFETY: first_empty returns an in-bounds index.
            unsafe { *self.slots.get_mut(idx) = e };
        }
    }

    /// Grow straight to a table that holds `n` entries without another
    /// resize (the same 3/4-load threshold as [`Self::insert`]), skipping
    /// the doubling ladder's repeated rebuilds: a cold multi-GB run
    /// otherwise re-zeroes and re-walks every live entry ~7 times as the
    /// table outgrows each power of two — random scatter writes into a
    /// fresh DRAM-cold allocation each time. No-op when the current
    /// capacity already holds `n`; an underestimate still grows correctly
    /// through [`Self::insert`]. See `Tokenizer::reserve_caches_for_input`.
    pub(crate) fn reserve_entries(&mut self, n: usize) {
        if (n + 1) * 4 <= self.slots.cap * 3 {
            return;
        }
        let mut cap = self.slots.cap;
        while (n + 1) * 4 > cap * 3 {
            cap *= 2;
        }
        self.grow_to(cap);
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.cap
    }
}

/// See [`ShortPretokenCache::probe_view`]. The pointer is borrowed from
/// the cache's live allocation; a view taken before an insert may dangle
/// after it (inserts can grow), so views are chunk-scoped.
#[derive(Clone, Copy)]
pub(crate) struct ProbeView {
    base: *const Entry,
    /// `slot mask & !1`, pre-folded: the emit loop computes a pair address
    /// from this on every probe AND every prefetch, and the fold keeps one
    /// ALU op (and one live temp in a loop that already spills) off the
    /// probe-address critical path.
    pair_mask: usize,
}

impl ProbeView {
    /// Address of `h`'s home pair; both slots share the addressed line.
    #[inline(always)]
    fn pair_ptr(&self, h: u64) -> *const Entry {
        // SAFETY: the masked even index is <= pair_mask <= cap - 2.
        unsafe { self.base.add((h as usize) & self.pair_mask) }
    }

    /// Request the probe's cache line into L1, a few probes ahead of
    /// [`Self::probe_pair`] (covers the L2 hit latency; the line was staged
    /// into L2 by [`ShortPretokenCache::prefetch_l2`] a chunk earlier).
    #[inline(always)]
    pub(crate) fn prefetch(&self, h: u64) {
        prefetch_line::<true>(self.pair_ptr(h));
    }

    /// Branchless probe of `key`'s home pair: both compares fold into one
    /// `found` flag and two selects, touching exactly one cache line. On
    /// `!found` the returned value lanes are another entry's (the emit
    /// loop's predicate discards them); keys displaced past their pair and
    /// genuine misses both come back `!found` — the slow path disambiguates
    /// via [`ShortPretokenCache::get_or_slot`]. Callers must not pass `key == 0`
    /// expecting a miss: empty slots compare equal to it (the emit
    /// predicate rejects the false hit through its own `val != 0` term —
    /// an empty slot's `val` reads 0).
    ///
    /// The selects run over unconditionally loaded `val`/`ext` of BOTH
    /// slots so they are register-value selects. Every pure-Rust spelling
    /// (`if`, mask arithmetic) gets canonicalized by LLVM into an address
    /// select — csel of a slot pointer feeding a second, dependent load —
    /// putting an extra L1 latency on the probe's critical path (the next
    /// thing waiting on `val` is the emit store and the cursor advance,
    /// the loop's only carried dependency), so on aarch64 the compare and
    /// the select are one asm `cmp`/`ccmp`/`csel`/`csel`/`cset` block: the
    /// `csel`s ride the key compare's flags directly, saving the
    /// test-and-branch pair a separate bool input would cost. Loading all
    /// four words up front costs two more loads per probe from the same
    /// already-touched line, all issued in parallel, none dependent on the
    /// compares.
    #[inline(always)]
    pub(crate) fn probe_pair(&self, key: u128, h: u64) -> (u64, u64, bool) {
        let p = self.pair_ptr(h);
        // SAFETY: pair_ptr's index is masked and even, so idx + 1 <= mask;
        // the base is live for the view's chunk (see type docs).
        let (e0, e1) = unsafe { (&*p, &*p.add(1)) };
        // The second slot's key compare: folded into the compare/select
        // asm block on aarch64, a plain bool everywhere else.
        #[cfg(not(target_arch = "aarch64"))]
        let m1 = e1.key == key;
        #[cfg(target_arch = "aarch64")]
        let (val, ext, found) = {
            let (mut val, mut ext) = (e0.val, e0.ext);
            let (m0, found): (u64, u64);
            // SAFETY: register-only compares and conditional selects; no
            // memory access, no stack use (NZCV is clobbered, which the
            // default options already declare).
            unsafe {
                core::arch::asm!(
                    "cmp {klo}, {e0klo}",
                    "ccmp {khi}, {e0khi}, #0, eq",
                    "csel {val}, {val}, {v1}, eq",
                    "csel {ext}, {ext}, {x1}, eq",
                    "cset {m0}, eq",
                    "cmp {klo}, {e1klo}",
                    "ccmp {khi}, {e1khi}, #0, eq",
                    "csinc {found}, {m0}, xzr, ne",
                    klo = in(reg) key as u64,
                    khi = in(reg) (key >> 64) as u64,
                    e0klo = in(reg) e0.key as u64,
                    e0khi = in(reg) (e0.key >> 64) as u64,
                    e1klo = in(reg) e1.key as u64,
                    e1khi = in(reg) (e1.key >> 64) as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    v1 = in(reg) e1.val,
                    x1 = in(reg) e1.ext,
                    m0 = out(reg) m0,
                    found = out(reg) found,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext, found != 0)
        };
        #[cfg(target_arch = "x86_64")]
        let (val, ext, found) = {
            let m0 = e0.key == key;
            // LLVM canonicalizes every pure-Rust spelling into an
            // address-cmov feeding a dependent load — the extra L1 latency
            // this function exists to avoid; the asm pins register-value
            // `cmovne`s over the four unconditionally loaded words instead.
            // cmov is baseline x86-64 (no feature gate); evidence in
            // profiling/x86_port_plan.md §1.1.
            let (mut val, mut ext) = (e1.val, e1.ext);
            // SAFETY: register-only test + conditional moves; no memory
            // access, no stack use.
            unsafe {
                core::arch::asm!(
                    "test {m}, {m}",
                    "cmovne {val}, {v0}",
                    "cmovne {ext}, {x0}",
                    m = in(reg) m0 as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    v0 = in(reg) e0.val,
                    x0 = in(reg) e0.ext,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext, m0 | m1)
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let (val, ext, found) = {
            let m0 = e0.key == key;
            let sel = (m0 as u64).wrapping_neg();
            (
                (e0.val & sel) | (e1.val & !sel),
                (e0.ext & sel) | (e1.ext & !sel),
                m0 | m1,
            )
        };
        (val, ext, found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::pretoken_key_hash;

    /// The encode miss path inserts at the slot its failed lookup
    /// reported. Drive that exact flow across several growth passes and
    /// verify every entry stays retrievable — i.e. `get_or_slot`'s miss
    /// slot always agrees with where `insert` would have placed the key.
    #[test]
    fn get_or_slot_insert_at_roundtrip() {
        let mut cache = ShortPretokenCache::with_pow2_capacity(64);
        // Multiplier chosen to scatter keys; count forces multiple grows
        // from the 64-slot start (grow threshold is 3/4 load).
        let keys: Vec<u128> = (1u128..=500).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
        for (i, &key) in keys.iter().enumerate() {
            let h = pretoken_key_hash(key);
            match cache.get_or_slot(key, h) {
                Ok(_) => panic!("key {i} present before insert"),
                Err(slot) => cache.insert_at(slot, key, h, i as u64, !(i as u64)),
            }
        }
        assert_eq!(cache.len(), keys.len());
        for (i, &key) in keys.iter().enumerate() {
            let h = pretoken_key_hash(key);
            assert_eq!(cache.get_or_slot(key, h), Ok((i as u64, !(i as u64))), "key {i}");
        }
    }
}
