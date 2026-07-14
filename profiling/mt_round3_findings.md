# MT encode profile analysis — samply_mt_round3 (10 GB cold, GPT-2, 16 threads, M4 Max)

Trace: `samply_mt_round3.json.gz` (4 kHz requested; effective per-thread rate
~0.4–2.6 ms because 17 threads overload the sampler — all CPU accounting below
uses `threadCPUDelta` (kernel-reported µs), which is authoritative regardless
of sample rate). Run measured 1.462 s encode = 6842 MB/s.
Analysis: `analyze_mt.py` (per-thread adaptation of opt/profiling/analyze.py),
outputs in `mt3_analysis/`.

## Thread structure

17 threads: T00 = main (bench thread), T01–T16 = rayon workers, spawned
lazily at t=1844.5 ms (first `rayon::current_num_threads()` call inside
`encode_docs_ragged`). Before that the main thread loads the tokenizer
(78–115 ms) and reads the 10 GB corpus (118–1844 ms). The encode window is
t = 1844.4 → 3306.4 ms (1462 ms — matches the bench's 1.462 s exactly).

**The main thread does not participate in the encode**: `par_iter` /
`rayon::scope` from a non-pool thread blocks on a latch. T00 shows 0 ms of
encode CPU; its only in-window work is the *serial* chunk-buffer free at the
end (below).

## The 1462 ms window, bucket by bucket

| phase | wall ms | % of window | CPU ms | notes |
|---|---|---|---|---|
| pool spawn + split scan + chunk build | 2.3 | 0.2% | ~2 | `safe_split_ranges` probes ~300 cut points; never appears in samples |
| worker fork + seed | 27 | 1.8% | 290 + 47 | 16 workers × 128 MB pre-sized table, explicitly memset (Rust `alloc_zeroed` with 2 MiB align = `aligned_alloc`+`write_bytes`, not calloc); bandwidth-bound at ~74 GB/s aggregate |
| encode steady state | ~1051 | 71.9% | ~15,300 | ~14.6 of 16 threads busy per 25 ms bin |
| encode straggler ramp-down | 2925→3029 = 104 | 7.1% | ~700 busy / **~1024 thread-ms idle** | see below |
| barrier→assemble head (counts loop + 9.1 GB calloc) | 0.5 | 0.03% | ~0 | `vec![0u32; total]` is lazy zero-fill mmap; free |
| gather par memcpy | 3029→3192 = 163 | 11.1% | 2000 | 9.1 GB copy; ~12.3 threads busy. Pure memcpy would be ~300 ms CPU; the other ~1.7 s is ~555k zero-fill page faults (16 KB pages) on first touch of `flat`, i.e. kernel-side page zeroing + fault entry |
| serial free of per-chunk id buffers | 3192→3306 = **114** | **7.8%** | 111 | `chunks: Vec<ChunkTokens>` (9.1 GB in ~307 Vecs) dropped **on the main thread** at the end of `assemble_ragged` — 100%-busy munmap, single-threaded, inside the timed window |

MT-specific overhead beyond steady-state encode ≈ 27 + ~64 (idle-equivalent
of the straggler tail) + 163 + 114 ≈ **368 ms = 25% of the window**.
Removing all of it would put the same encode work at ~9.1 GB/s.

Per-worker encode CPU ranges 916–1057 ms (T16 low / T05 high) — consistent
with E-core/P-core mixing under macOS scheduling, not with idle time (idle
condvar CPU is < 5 ms per worker until the tail).

## The straggler tail is an LPT ordering violation, not E-core noise

Per-worker last-encode-sample times spread from 2924.8 (T13) to 3028.8
(T05/T08). Tail chunks are ~9.8 MB (≈ 10 ms P-core / ≈ 28 ms E-core), so a
true in-order LPT handout would bound the spread at ~1 tail chunk. Observed:
104 ms. The straggler threads (tid …53, …54) were **continuously busy
encoding from ~2936 to 3029 with no gaps** — the signature of a 78 MB "big"
head chunk (≈ 87 ms on a P-core) *starting* at ~2940, long after other
threads had moved to the tail.

Cause: `build_doc_chunks` orders chunks big→small assuming rayon hands them
out in index order, but `par_iter().map().collect()` distributes work by
recursive range splitting — a thread can steal a subrange of big chunks and
still be *starting* one late in the run. The LPT ordering is only advisory
under range splitting.

Cost: Σ per-worker idle between own-last-encode and gather-start ≈ 1024
thread-ms ≈ **64 ms of window** at 16 threads.

## What the data justifies (implemented)

1. **Fuse per-chunk buffer frees into the parallel gather copy**
   (`assemble_ragged` consumes `chunks` by value in the copy loop, one chunk
   per rayon task). The 114 ms serial munmap phase disappears from the
   critical path; frees are distributed across workers and overlapped with
   the copy (the munmap kernel work partially serializes on the vm-map
   lock, but the user-side memcpy of other threads proceeds meanwhile).
   Upper bound saving ≈ 114 ms (7.8%).

2. **Strict in-order chunk handout** (`encode_chunks_pooled` replaces
   `par_iter` with an atomic-counter pull loop over the chunk array, one
   rayon task per thread). This makes the LPT descending-size order a
   guarantee instead of a hint: no thread can start a 2×-target head chunk
   after the tail has begun, bounding the straggler spread at roughly one
   tail chunk (~10–28 ms) instead of one big chunk (~87+ ms). Expected
   saving ≈ 40–60 ms (3–4%).

Combined expectation: window 1462 → ~1310 ms (≈ 7.6 GB/s) on the same
machine/input, straggler- and munmap-bound phases mostly gone.

## Evaluated and rejected (for the record)

- **Prefault/madvise the flat buffer before/during encode**: `total` (and
  hence the allocation) is only known when the last chunk finishes; macOS
  has no THP/MADV_HUGEPAGE, superpages are unavailable on Apple Silicon,
  and the zero-fill faults *are* the first touch by the copy, which is
  already parallel. Over-allocating by an upper bound (1 token/byte → 40 GB
  VA) to allow early copy would hand callers a 4.4×-capacity Vec or gamble
  on in-place `realloc` shrink of a 9 GB block — not worth it without the
  ability to benchmark the shrink path.
- **Overlap gather with encode via finished workers copying complete prefix
  chunks**: blocked on the same fact — the destination buffer cannot exist
  until the last chunk's size is known. The straggler fix shrinks exactly
  the window where this would have helped.
- **Cheaper fork+seed (27 ms wall)**: the 128 MB/worker table memset could
  become lazy zero-fill (mmap) — but seeding ~50k entries scatters over
  ~99.8% of the table's 16 KB pages, so the same 2 GB of kernel page
  zeroing would happen anyway, plus ~130k fault entries. COW-remapping a
  prototype seeded table (mach_vm_remap) spreads the cost but adds
  platform-specific complexity for ≤ 27 ms; deferred.
- **Doc split scan**: ~0 ms measured; nothing to shrink.
- **Per-chunk output Vec first-touch**: faults are spread inside the encode
  phase and inherent to holding 9.1 GB of results; recycling buffers would
  require the gather to run concurrently (see above).
- **Finer tail chunks (LPT knob)**: without strict handout order the knob
  doesn't bind; revisit after (2) if a residual tail shows up in a fresh
  trace.
