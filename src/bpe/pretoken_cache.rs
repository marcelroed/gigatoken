//! Open-addressing cache for short (≤ 15 byte) pretoken encodings, replacing
//! the `HashMap<u128, (u32, u32)>` it grew out of. Three properties of the
//! encode loop drive the design (measured on 1 GB OWT, Zen 2):
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

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
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
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw as *mut Entry) else {
            handle_alloc_error(layout)
        };
        #[cfg(target_os = "linux")]
        // SAFETY: the range is exactly this fresh allocation, 2 MiB-aligned;
        // MADV_HUGEPAGE only hints page sizing and cannot invalidate it.
        unsafe {
            libc::madvise(raw as *mut libc::c_void, layout.size(), libc::MADV_HUGEPAGE);
        }
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

pub(crate) struct ShortPretokenCache {
    slots: Slots,
    /// `cap - 1` (capacity is a power of two).
    mask: usize,
    len: usize,
}

impl ShortPretokenCache {
    pub(crate) fn new() -> Self {
        Self::with_pow2_capacity(1 << 16)
    }

    fn with_pow2_capacity(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two() && cap >= 2);
        Self { slots: Slots::new_zeroed(cap), mask: cap - 1, len: 0 }
    }

    /// A table sized to hold at least `n` entries without growing (same
    /// 3/4-load threshold as [`Self::insert`], same 2^16 floor as
    /// [`Self::new`]), starting from at least `min_slots` slots. The
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

    /// Request the probe's cache line into L1, a few probes ahead of
    /// [`Self::probe_pair`] (covers the L2 hit latency; the line was staged
    /// into L2 by [`Self::prefetch_l2`] a chunk earlier).
    #[inline(always)]
    pub(crate) fn prefetch(&self, h: u64) {
        let p = self.pair_ptr(h);
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; any address is allowed.
        unsafe {
            core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: prefetch has no memory effects; any address is allowed.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = p;
    }

    /// Request the probe's cache line into L2 only. The encode loop calls
    /// this a full chunk (hundreds of cycles) before the probe — enough to
    /// cover DRAM — without evicting the span walker's L1 working set the
    /// way a chunk's worth of L1 prefetches would.
    #[inline(always)]
    pub(crate) fn prefetch_l2(&self, h: u64) {
        let p = self.pair_ptr(h);
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; any address is allowed.
        unsafe {
            core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T1);
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: prefetch has no memory effects; any address is allowed.
        unsafe {
            core::arch::asm!(
                "prfm pldl2keep, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = p;
    }

    /// Branchless probe of `key`'s home pair: both compares fold into one
    /// `found` flag and two selects, touching exactly one cache line. On
    /// `!found` the returned value lanes are another entry's (the emit
    /// loop's predicate discards them); keys displaced past their pair and
    /// genuine misses both come back `!found` — the slow path disambiguates
    /// via [`Self::get`]. Callers must not pass `key == 0` expecting a
    /// miss: empty slots compare equal to it (the emit predicate carries
    /// its own `key != 0` term).
    #[inline(always)]
    pub(crate) fn probe_pair(&self, key: u128, h: u64) -> (u64, u64, bool) {
        let idx = (h as usize) & self.mask & !1;
        // SAFETY: idx is masked and even, so idx + 1 <= mask.
        let e0 = unsafe { self.slots.get(idx) };
        let e1 = unsafe { self.slots.get(idx + 1) };
        let m0 = e0.key == key;
        let m1 = e1.key == key;
        let val = if m0 { e0.val } else { e1.val };
        let ext = if m0 { e0.ext } else { e1.ext };
        (val, ext, m0 | m1)
    }

    /// Look up `key`, walking pairs from its home bucket. Inserts fill the
    /// first empty slot of the walk, so any pair holding an empty slot
    /// terminates it.
    pub(crate) fn get(&self, key: u128, h: u64) -> Option<(u64, u64)> {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut idx = (h as usize) & self.mask & !1;
        loop {
            // SAFETY: idx is masked and even, so idx + 1 <= mask.
            let e0 = unsafe { self.slots.get(idx) };
            let e1 = unsafe { self.slots.get(idx + 1) };
            if e0.key == key {
                return Some((e0.val, e0.ext));
            }
            if e1.key == key {
                return Some((e1.val, e1.ext));
            }
            if e0.key == EMPTY_KEY || e1.key == EMPTY_KEY {
                return None;
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
    /// a [`Self::get`] miss).
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

    #[cold]
    fn grow(&mut self) {
        let new_cap = self.slots.cap * 2;
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

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.cap
    }
}
