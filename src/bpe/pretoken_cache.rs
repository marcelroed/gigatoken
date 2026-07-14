//! Open-addressing cache for short (≤ 15 byte) pretoken encodings, replacing
//! the `HashMap<u128, (u32, u32)>` it grew out of. Three properties of the
//! encode loop drive the design (measured on 1 GB OWT, Zen 2):
//!
//! - The table holds ~1.3M unique pretokens (~99.4% hit rate), far beyond
//!   L2/L3, so a lookup in the Zipf tail is a random DRAM access. hashbrown
//!   spends two cache lines per probe (control bytes + entry); this table's
//!   32-byte entries are self-contained, so a probe touches exactly one
//!   line, and [`Self::prefetch`] lets the encode loop request that line a
//!   chunk ahead of the probe, hiding the miss latency behind other work
//!   instead of stalling on it.
//! - 228M output tokens / 208M pretokens: ~90% of pretokens encode to ONE
//!   token and ~98% to at most two. The value is a single packed `u64`
//!   (see `tiktoken::pack_fast_val_inline`) holding those tokens inline —
//!   one dependent load, and no second random access into the token arena.
//! - At ~64 MB the table also blows the dTLB through 4 KiB pages, so the
//!   backing memory is 2 MiB-aligned and `MADV_HUGEPAGE`d — with THP
//!   available the whole table sits in a few dozen dTLB entries. (Note:
//!   processes launched under `PR_SET_THP_DISABLE` — some sandboxes and
//!   session managers do this — silently get 4 KiB pages anyway.)
//!
//! Linear probing; growth doubles at 3/4 load. Key 0 marks empty slots: a
//! real key always has its nonzero length in the top byte
//! (`pack_pretoken_key` tags length; empty pretokens pack to key 0, which
//! the encode loop routes to the long map, never here).

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ptr::NonNull;

/// One slot: the packed pretoken key plus its packed encoding.
/// 24 data bytes pad to 32 with `u128` alignment: two slots per cache
/// line, never straddling one.
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    val: u64,
}

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
        let align = Self::HUGE_PAGE
            .min(size.next_power_of_two())
            .max(std::mem::align_of::<Entry>());
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
        debug_assert!(cap.is_power_of_two());
        Self { slots: Slots::new_zeroed(cap), mask: cap - 1, len: 0 }
    }

    /// A table sized to hold at least `n` entries without growing (same
    /// 3/4-load threshold as [`Self::insert`], same 2^16 floor as
    /// [`Self::new`]). The vocab-seeding path inserts ~50k entries up
    /// front; pre-sizing avoids rehashing them mid-seed.
    pub(crate) fn with_at_least(n: usize) -> Self {
        let mut cap = 1usize << 16;
        while (n + 1) * 4 > cap * 3 {
            cap *= 2;
        }
        Self::with_pow2_capacity(cap)
    }

    /// Request the probe's cache line ahead of [`Self::get`]. The encode
    /// loop calls this a chunk of pretokens early; by probe time the line
    /// is (usually) in L1 regardless of where it started.
    #[inline(always)]
    pub(crate) fn prefetch(&self, h: u64) {
        let idx = (h as usize) & self.mask;
        // SAFETY: idx <= mask < cap.
        let p = unsafe { self.slots.ptr.as_ptr().add(idx) };
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

    /// Look up `key`, returning its packed value.
    #[inline(always)]
    pub(crate) fn get(&self, key: u128, h: u64) -> Option<u64> {
        let mut idx = (h as usize) & self.mask;
        loop {
            // SAFETY: idx is masked to the table.
            let e = unsafe { self.slots.get(idx) };
            if e.key == key {
                return Some(e.val);
            }
            if e.key == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Insert a key known to be absent (the encode loop only inserts after
    /// a [`Self::get`] miss).
    pub(crate) fn insert(&mut self, key: u128, h: u64, val: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
        }
        let mut idx = (h as usize) & self.mask;
        // SAFETY: idx is masked; load < 1 guarantees an empty slot exists.
        unsafe {
            while self.slots.get(idx).key != EMPTY_KEY {
                idx = (idx + 1) & self.mask;
            }
            *self.slots.get_mut(idx) = Entry { key, val };
        }
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
            let mut idx =
                (crate::pretokenize::pretoken_key_hash(e.key) as usize) & self.mask;
            // SAFETY: idx is masked; the doubled table has empty slots.
            unsafe {
                while self.slots.get(idx).key != EMPTY_KEY {
                    idx = (idx + 1) & self.mask;
                }
                *self.slots.get_mut(idx) = e;
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.cap
    }
}
