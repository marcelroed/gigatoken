# Pretokenizer Optimization Log

Optimizing the GPT-2 pretokenizer regex:
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Target: 1 GiB/s single-threaded throughput on 100 MB of OpenWebText.

Platform: Apple Silicon (ARM), `cargo bench` with `lto = "fat"`.

## Baseline

| Implementation | Throughput |
|----------------|-----------|
| `fancy-regex` | ~47 MiB/s |
| State machine (hand-rolled) | ~380 MiB/s |
| Winnow combinators + NEON SIMD | ~462 MiB/s |

The winnow+NEON implementation was the existing best. It uses NEON intrinsics (`vld1q_u8`, `vcgtq_u8`, etc.) to scan 16 bytes at a time inside letter/digit/other runs, with scalar fallback for non-ASCII and run transitions.

---

## Step 1: LUT dispatch + SWAR letter scanning

**File:** `src/pretokenize/pretoken_fast.rs` (new)

**What changed:**
- Replaced the winnow parser combinator framework (`alt()`, `trace()`, `backtrack()`, `ModalResult`) with a direct `Iterator` implementation — zero framework overhead per token.
- Replaced NEON SIMD intrinsics with SWAR (SIMD Within A Register): loads 8 bytes as a `u64`, applies branchless arithmetic to check all 8 bytes for the letter property simultaneously.
- Added a 256-byte LUT (`CLASS[256]`) for O(1) first-byte classification — dispatches directly to the right scan function without cascading `if/else` or `alt()` backtracking.
- Used arithmetic byte predicates instead of LUT lookups inside scan loops: `is_letter(b) = (b | 0x20).wrapping_sub(b'a') < 26`, `is_digit(b) = b.wrapping_sub(b'0') < 10`.

**SWAR letter check (the key technique):**
```rust
let word: u64 = read_unaligned(ptr); // load 8 bytes
let lowered = word | 0x2020_2020_2020_2020; // case-fold
let ge_a = (lowered | 0x8080..).wrapping_sub(0x6161..); // >= 'a'
let le_z = 0xFAFA..wrapping_sub(lowered);               // <= 'z'
let mask = ge_a & le_z & 0x8080..;                       // bit 7 set per letter
// Find first non-letter:
(!mask & HI).to_le().trailing_zeros() / 8
```

**Why it matters:** The SWAR technique processes 8 ASCII letters per iteration with ~6 arithmetic ops — no architecture-specific intrinsics. The LUT dispatch eliminates the alt/backtrack overhead that winnow pays for every token start (trying contraction, then letter_run, then number_run, etc. until one succeeds). Arithmetic predicates avoid data-dependent LUT loads inside hot scan loops.

**Result:** 544 → **830 MiB/s** (first version, without `get_unchecked`)

---

## Step 2: Unsafe `get_unchecked` in scan loops

**What changed:**
- Replaced bounds-checked `bytes[self.pos]` with `unsafe { *bytes.get_unchecked(self.pos) }` in all hot scan loops.
- Used `unsafe { *bytes.get_unchecked(start) }` for first-byte dispatch.

**Why it matters:** Bounds checks in tight loops generate conditional branches that the CPU must predict. Since we always check `self.pos < len` before indexing, the bounds check is redundant — removing it eliminates ~1 branch per byte in letter/digit/other scans.

**Result:** 830 → **840 MiB/s** (~1% improvement, within noise for most runs but consistently measurable)

---

## Step 3: Arithmetic space dispatch (no second LUT lookup)

**What changed:**
- When the first byte is `' '` (space), the second byte determines the token type. Previously this did a second `CLASS[b1]` LUT lookup. Replaced with direct arithmetic checks: `is_letter(b1)`, `is_digit(b1)`, `is_ascii_ws(b1)`, `b1 >= 0x80`.

**Why it matters:** Avoids a data-dependent load (LUT indexed by `b1`) in the most common token start pattern (`" word"`). The arithmetic checks can execute in parallel in the ALU without waiting for the LUT load to complete.

**Result:** No measurable change (~840 MiB/s), but eliminates a potential latency bottleneck on architectures with higher L1 latency.

---

## Step 4: Hot/cold split with `#[cold]` + `#[inline(never)]` (REVERTED)

**What changed:**
- Moved unicode handling (non-ASCII letter/digit/other continuation) into separate `#[cold] #[inline(never)]` functions, keeping only ASCII logic in the hot path.

**Why it matters (in theory):** Reduces the instruction footprint of the hot path, improving icache utilization. The cold functions are rarely called (English text is >99% ASCII).

**Result:** **Regressed to 580 MiB/s.** The `#[inline(never)]` barrier prevented LLVM from optimizing the combined ASCII+unicode loop. The function call overhead (~5 cycles) per unicode encounter was worse than the icache benefit. **Reverted.**

---

## Step 5: `advance()` / `count()` separation

**What changed:**
- Extracted a shared `advance(&mut self)` method that advances `self.pos` past one token without constructing a `Pretoken` slice.
- `next()` calls `advance()` then returns the slice.
- `count()` calls `advance()` in a tight loop, avoiding `Option<Pretoken>` construction.

**Why it matters:** The `count()` hot loop becomes `while pos < len { advance(); n += 1; }` — no `Option` wrapping/unwrapping, no slice construction. This shaves a few nanoseconds per token from the benchmark.

**Result:** ~840 → **848 MiB/s** (small but consistent improvement)

---

## Step 6: Two-pass approach — classification buffer + SWAR transition counting (EXPERIMENTAL)

**What changed:**
- Pass 1: Classify every byte via LUT into a per-byte class buffer (`cb[CHUNK+1]`).
- Pass 2: SWAR XOR adjacent classes to detect transitions; `count_ones()` for the count. Merges WHITESPACE→SPACE and APOSTROPHE→OTHER to suppress false transitions.
- Pass 3: Sequential fixups for whitespace splits (2+ ws followed by non-ws), contractions, and non-ASCII.

**Challenges encountered:**
- SPACE and WHITESPACE are different class values but represent the same merged class for transition purposes — required merging before XOR.
- APOSTROPHE and OTHER needed merging too (non-contraction `'` is scanned as "other").
- Contractions like `c'mon` where the letter after the contraction continues required careful handling — only subtract the APOS→LETTER transition when the byte AFTER the contraction is a different class.
- Non-ASCII bytes needed real unicode classification in Pass 1 to avoid spurious transitions.
- Multi-byte UTF-8 characters spanning chunk boundaries needed chunk-end alignment.

**Why it didn't work:**
- The classification buffer doubles memory traffic (write all classes, then read them back).
- Three passes over the data (classify + SWAR count + fixups) read more total bytes than one-pass.
- The SWAR transition detection saves branch mispredictions but the memory bandwidth cost dominates.

**Result:** **306-354 MiB/s** — 2.5x slower than one-pass. The approach is algorithmically correct (verified on 5 MB) but not competitive for performance.

---

## Step 7: PGO — Profile-Guided Optimization (NO EFFECT)

**What changed:**
- Built with `-Cprofile-generate`, ran the benchmark to collect branch profiles, then rebuilt with `-Cprofile-use`.

**Why it didn't help:**
- The SWAR inner loop is already branchless — no branch probabilities to optimize.
- The word-boundary misprediction is fundamentally unpredictable — the branch outcome depends on input data (is the next byte a letter or not?), not on static code patterns.
- The LUT dispatch compiles to a jump table — PGO can't improve indirect branch prediction.

**Result:** 842 MiB/s (within noise of the 847 MiB/s baseline). PGO actually *hurt* the state machine implementation by 13%.

---

## Step 8: Dual-cursor ILP exploitation

**What changed:**
- Refactored `advance()` from a method into a free function `advance_pos(bytes, pos) -> new_pos` with standalone scan helpers (`scan_letters_from`, `scan_digits_from`, etc.).
- Added `find_split()`: searches for a safe split point near the midpoint — a `\n` followed by a non-whitespace ASCII byte, which guarantees a token boundary.
- Implemented `count_dual_cursor()`: splits the input at the safe boundary, then runs two independent cursors in an interleaved loop:

```rust
while p1 < split && p2 < len {
    p1 = advance_pos(bytes, p1);
    p2 = advance_pos(bytes, p2);
    count += 2;
}
```

**Why it matters:**

The fundamental bottleneck at ~840 MiB/s was **latency, not throughput**. Each token has a serial dependency chain:

```
find_end(token N) → pos_N → load byte at pos_N → classify → scan → find_end(token N+1) → ...
```

This chain is ~25-27 cycles on modern CPUs. The OoO engine has spare execution units sitting idle while waiting for each chain to resolve.

The dual-cursor technique creates two completely independent chains — different positions, different memory addresses, different registers. The OoO engine interleaves their micro-ops across execution ports: while cursor 1 is stalled waiting for a SWAR comparison result, cursor 2's loads and ALU ops execute on the otherwise-idle ports. Even though the two `advance_pos` calls look sequential in source, the CPU sees them as independent instruction streams and overlaps them.

**Result:** 840 → **1,049 MiB/s (1.05 GiB/s)** — 25% speedup, crossing the 1 GiB/s target.

---

## Summary

| Step | Technique | Throughput | Delta |
|------|-----------|-----------|-------|
| Baseline | Winnow + NEON | 462 MiB/s | — |
| 1 | LUT dispatch + SWAR | 830 MiB/s | +1.80x |
| 2 | `get_unchecked` | 840 MiB/s | +1.01x |
| 3 | Arithmetic space dispatch | 840 MiB/s | ~1.00x |
| 4 | Hot/cold split | 580 MiB/s | **reverted** |
| 5 | advance/count separation | 848 MiB/s | +1.01x |
| 6 | Two-pass SWAR transitions | 354 MiB/s | **not used** |
| 7 | PGO | 842 MiB/s | ~1.00x |
| **8** | **Dual-cursor ILP** | **1,049 MiB/s** | **+1.25x** |

**Total speedup over winnow+NEON: 2.27x**
**Total speedup over regex: 22.3x**

Key lessons:
- SWAR is the single biggest win — portable, no intrinsics, processes 8 bytes/iteration.
- Framework overhead (winnow's alt/backtrack) matters more than SIMD width for this workload.
- Multi-pass approaches lose to single-pass due to memory bandwidth, even when branch-free.
- PGO doesn't help when the bottleneck is data-dependent branches.
- ILP exploitation via dual cursors provides free speedup by filling pipeline bubbles.
