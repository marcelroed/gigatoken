# Cold-encode optimization campaign — gigatoken BPE encode, rounds 1–5 + verification

Five rounds of profile-driven optimization of the cold encode path
(single-threaded `encode_st` and 16-thread `encode_docs_ragged`), followed by
a heavy verification round and a fix round, run as a multi-agent campaign:
analysis agents propose from traces, implementation agents build each
technique on its own branch, and a central harness measures everything
sequentially. Workload throughout: one cold 10 GB OWT pass per process, GPT-2
tokenizer unless noted, Apple M4 Max (12P+4E), corpus in page cache. Every
variant passed the GPT-2 token-identity gates in §2 before being measured or
merged; the closing verification round (§7) extended those gates to
full-corpus, multi-tokenizer differentials, caught one real regression the
GPT-2 gates could not see (qwen3_5 vocab seeding) plus two pre-existing
walker bugs, and all three were root-caused and fixed before the final
numbers below were taken.

## 1. Executive summary

Definitive final three-way showdown (same session, interleaved sequential
A/B, fresh process per sample, all token counts bit-identical):

| variant | MT mean (best) | ST mean | ST output mode |
|---|---|---|---|
| `encode-opt-main` (this campaign, final) | **8792 MB/s** (8912) | **1039 MB/s** | **materializing** all ~2.28 G tokens (~9.1 GB) into a flat buffer |
| PR #4 head | 5538 MB/s | 953 MB/s | counting only |
| original `encode-perf` baseline | 2826 MB/s | 605 MB/s | counting only |

- MT cold encode: **+58.8% over PR #4, 3.11x the `encode-perf` baseline.**
- ST: the mainline *materializes* every token faster than the predecessors
  merely *count* them (1039 vs 953 MB/s). Like-for-like materializing
  comparisons measured earlier put PR #4 at ~601–767 MB/s.
- Generalization beyond GPT-2 (MT, mainline vs PR #4, measured earlier the
  same day — **pre-mt5**, so the final margins are larger): olmo3 7269 vs
  4921 MB/s (+48%); qwen2 6045 vs 4226 MB/s (+43%).
- 11 techniques merged across 5 rounds; 6 measured and rejected or held
  (kept on their branches with the numbers that killed them). The round-5
  null (`opt/walker5`) came with a dependency-structure proof that the
  walker is at its microarchitectural floor (§5), closing that line.
- A five-lens adversarial review of the full diff (§6) fixed one API
  soundness regression, one BE-portability gap, and one added-token edge
  case (`1ffff3c`), with hot-path codegen verified bit-identical. The heavy
  verification round (§7) then found what diff review could not: a
  data-dependent token-identity regression visible only on one tokenizer's
  vocab, plus two invalid-UTF-8 walker bugs (one pre-dating the campaign).
  Both fix branches merged with all gates green and hot-loop asm verified
  unchanged.

## 2. Methodology

The campaign's rules, all of which earned their keep:

- **One technique, one branch, one worktree.** Every candidate lives on
  `opt/<name>` and was developed in an isolated worktree. Merges into
  `encode-opt-main` happen only after measurement; rejected branches are
  kept, not deleted — the negative results are part of the record.
- **Strictly sequential, interleaved A/B.** This bench varies ±8%
  run-to-run, so nothing runs in parallel, ever, and A/B comparisons
  interleave variant and control samples within one session (min-of-N /
  means over interleaved samples). One cold pass per process — no warm
  reuse inside a process.
- **Token-identity gates before any measurement.** Exact token counts at
  100 MB (`encode_st` 22,834,020; `encode_doc` 22,723,342) plus a 50 MB
  token-by-token differential against the reference encoder. A variant that
  fails identity is a bug, not a candidate. These gates are GPT-2-based;
  §7's verification round extended them to full-corpus and multi-tokenizer
  differentials and showed that extension was necessary — one round-1
  technique was token-correct for GPT-2 but not for qwen3_5.
- **Central measurement only.** Implementation agents were forbidden from
  benchmarking; all numbers come from one sequential harness on an
  otherwise-idle machine. This removes both machine contention and
  motivated measurement.
- **Profile → propose → build → measure → merge/reject** per round, with the
  next round's profile taken on the new mainline.

## 3. Profiling infrastructure

Documented in detail in `profiling/report.md`; the short version:

- **Parity bench profile**: `[profile.bench]` inherits `release` (fat LTO,
  identical codegen) + `debug = true`, packed dSYM. Verified 1.6% delta vs
  plain release on identical full passes — inside the noise band.
- **samply 0.13.1 at 4 kHz**, `--save-only` (unsymbolicated) with offline
  symbolication: `atos` against the dSYM with `-i` expands **inline frames
  with exact file:line at each sampled address** — the thing that makes
  attribution trustworthy under fat LTO + `#[inline(always)]`. System
  dylibs resolved from the presymbolicate sidecar. `analyze.py` produces
  phase/bucket splits, top-functions, hot lines, and flamegraph-format
  collapsed stacks; `analyze_mt.py` adds per-thread accounting keyed on
  `threadCPUDelta` (authoritative when 17 threads overload the sampler).
- **xctrace CPU Counters (PMU)**, "CPU Bottlenecks" mode, exported via
  XPath with id/ref decompression and duration weighting (`pmu_summary.py`)
  — unweighted export over-counts rare rows.
- Sanity checks each round: profile-total vs wall (99.5%), bucket
  reproducibility across independent traces (within ~1 pt), tops matching
  code expectations.

Traces and per-round analyses live under `profiling/traces/*_analysis/` and
`profiling/mt*_analysis/`; MT round findings in
`profiling/mt_round3_findings.md`, `profiling/mt4_analysis/mt_round4_findings.md`,
and `profiling/mt5_analysis/mt_round5_findings.md`.

## 4. Round by round

All per-round percentages are interleaved A/B wins **against that round's
control** (the then-current `encode-opt-main`), MT unless marked ST.

### Round 1 — from the baseline ST profile

Baseline profile (`profiling/report.md`): 10.5 s encode, walker 49.4%,
miss-path merge 15.4%, driver 14.6%, cache probe 14.2%, key pack 5.6%;
PMU verdict: **25.1% of P-core issue bandwidth discarded to bad
speculation** (§5). Five proposals (`scratchpad/proposals/*.md`), five
branches:

| branch | verdict |
|---|---|
| `opt/probe-emit` — branchless probe/emit, 4-token inline entries, flat-buffer output, staged prefetch | **merged, +6.2% MT / +27.6% ST-materializing** |
| `opt/walker-twophase` — two-phase branchless span walker | **merged, +3.4% MT** |
| `opt/keypack` — arithmetic pack masks, buffered keyed fill, CRC32 key hash | **merged, +1.7%** |
| `opt/miss-path` — PairRankTable, vocab-seeded short cache, stack/NEON short merge | **merged, +1.3%** |
| `opt/mt-alloc` — pre-sized worker caches + streaming gather + LPT + Arc-shared tables | **−5.9%, rejected as a unit** |

The mt-alloc regression was bisected with env-var kill switches
(`1aa6210`): the streaming-gather committer thread was the culprit; the
other three components were salvaged as `opt/mt-salvage` (pre-size from
batch share, Arc-share merges/vocab/vocab_inv across forks, LPT descending
chunk order) and merged. probe-emit and miss-path were semantically
unified at merge (`160bdd0`).

Postscript: the miss-path branch's vocab seeding (and a companion
whole-pretoken `vocab_inv` shortcut on the long miss path) carried the one
token-identity bug of the campaign — correct for GPT-2/olmo3/qwen2/
deepseek_v3, wrong for qwen3_5's 201 merge-unreachable vocab entries. Found
by §7's differentials, fixed in `8723407`.

### Round 2

- `opt/emit2` — register-resident cursors, parallel pair loads,
  value-select probe: **merged, +5.2% MT / +7.4% ST**.
- `opt/phaseb` — pointer-indexed emission loops, careful-pack suffix
  split, popcount-idiom dodge in `flatten_bits`: **−1.5%, rejected.**
- `opt/rank2` — packed rank slots, narrow NEON scan, fingerprint long map:
  **±0 standalone, held out.** Partially re-taken in round 4 when the
  round-3 profile showed `PairRankTable::rank` had grown in relative share.

### Round 3 — first MT trace

`mt_round3_findings.md` decomposed the 1462 ms MT window: 71.9% steady-state
encode, then three orchestration losses — 163 ms gather copy, **114 ms
serial free** of ~9.1 GB of chunk buffers on the main thread, and a
**104 ms straggler tail** caused by rayon's range splitting turning the LPT
big→small chunk order into a hint (a 78 MB head chunk observed *starting*
after other threads reached the tail).

- `opt/mt3` — fused parallel copy+drop gather, atomic-counter strict
  in-order LPT handout: **merged, +6.2% MT.** Round-4 trace confirmed both
  predictions: straggler spread 104 → 23 ms, gather+free 277 → 214 ms.
- `opt/walker3` — pinned `movemask64` addp tree as asm, table-based
  branchless phase-B key pack: **merged, +2.1% MT / +6.2% ST.**
- `opt/emit3` — two variants (L1 staging buffer + memcpy flush; direct
  stores + `prfm pstl1keep`): **−4..−7% both, rejected.** The existing
  emit loop's stores were already not the bottleneck; adding a staging hop
  or extra prefetch traffic only cost.

### Round 4 — the gather convoy

`mt_round4_findings.md` on the round-3 mainline (1336 ms window): the fused
gather ran at only **7.2/16 threads busy** — a classic vm-map
reader/writer convoy. First-touch zero-fill faults on the 9.1 GB flat
buffer take the vm-map read lock; the interleaved chunk-buffer munmaps
take the **write** lock (~0.7 ms × 307), each stalling every concurrently
faulting thread. 205 ms of actual teardown CPU was costing ~1.4 s of
blocked thread-time.

- `opt/mt4` — defer chunk-buffer teardown to a detached background task
  after `assemble_ragged` returns; Heaps-law worker table sizing
  (OWT-calibrated distinct(n) ≈ 3.45·n^0.62 ⇒ 2^21 slots instead of 2^22
  for a 10 GB/16 share, halving 2 GB of fork memset): **merged, +3.4% MT.**
- `opt/driver4` — AoS `SpanBatch` (one 32 B entry per pretoken instead of
  parallel arrays), unclamped prefetch-slack: **merged, +0.7% MT /
  +1.9% ST.**
- `opt/miss4` — packed rank slots, narrow NEON scan, memcpy emits, fused
  miss→insert slot (the surviving parts of rank2 plus new work):
  **merged, +0.9% ST.**
- `opt/walker4` — dual-nibble TBL classify, cmtst asm pin, ctpop-idiom
  dodge: **±0, rejected despite cutting 113 → 88 instructions per 64 B
  batch.** The walker is not retiring-limited; round 5's dependency
  analysis (below) explains exactly why.

### Round 5 — the last MT lever, and the walker's floor

The round-5 MT trace (`mt5_analysis/mt_round5_findings.md`, 1226 ms window)
first verified round 4 on prediction — deferred drop removed every in-window
munmap (gather 214 → 170 ms, 7.2 → 12.2/16 threads busy), Heaps sizing
halved the fork memset (ramp 32 → 19 ms) — and then exposed the next layer:
the gather's memcpy **CPU nearly doubled** (1123 → 2079 ms) while its wall
shrank. Round 4's threads had been *sleeping* on the vm-map write lock;
with the munmaps gone, 16 threads fault-concurrently on the 9.1 GB flat
buffer and the cost converts into in-kernel fault-path contention — more
than half the gather's CPU. Ramp and tail were measured at their floors;
the gather was the only remaining orchestration target.

- `opt/mt5` — fold the gather into the encode phase: reserve the flat
  buffer up front at the strict upper bound (one token consumes ≥ 1 input
  byte, so `total_bytes` tokens; untouched pages cost VA only), and let a
  worker that finishes a chunk try-lock a commit cursor and copy the ready
  prefix at exact offsets while the tail still encodes (bounded drain, no
  committer thread, contended finishers return to encoding; classic
  collect-then-gather fallback on reservation failure or NFC-expansion
  overflow). Single-chunk inputs return the chunk's id buffer directly —
  no gather copy at all. **Merged, +4.4% MT (8680 vs 8311 MB/s mean,
  interleaved).** The copy CPU hides inside the encode phase at ~1
  committer at a time, which also dilutes the fault contention.
- `opt/walker5` — flatten-popcount register pin + input-stream prefetch:
  **±0, rejected.** But the branch's dependency-structure analysis (asm of
  the shipped r50k fill monomorphization + sampled profile) is the
  campaign's definitive walker verdict — see §5. **The walker is at its
  floor; the 128B-walker idea is retired permanently.**

With mt5 merged, MT non-steady overhead is down to a ~19 ms fork ramp
(task-0 latency, already fully overlapped per worker), a one-tail-chunk
straggler (~5 ms window-equivalent, the in-order-LPT bound), and a small
post-join suffix drain — MT orchestration is essentially done.

## 5. The microarchitectural story

**ST: the encoder was branch-misprediction-bound, not memory-bound.**
Baseline PMU decomposition (10.4 s encode window, 4.0 GHz, 41.8 G cycles):
54.9% retiring, **25.1% discarded (mispredicted-path)**, 11.4% backend
(dependent loads), 7.4% frontend. The discarded quarter concentrated in
per-pretoken data-dependent branch ladders: the walker's `rem != 0`
bit-walk and segment-refill exits, `pack_pretoken_key`'s length/page
branches with a dependent `PACK_MASK[n]` table load, and the probe/emit
hit/spill/miss triad. The merged fixes attack exactly these:

- **Walker**: two-phase structure — phase A extracts all span boundaries
  of a 64 B block branchlessly into a buffer, phase B consumes them with a
  table-based branchless key pack (walker-twophase, walker3).
- **Key pack / hash**: arithmetic (shift-based) masks replace the
  dependent table load; CRC32 hardware hash (keypack).
- **Probe/emit**: 4-token inline cache entries emitted unconditionally
  into a flat output buffer with branchless length advance; paired probe
  compares; staged prefetch (probe-emit, emit2); the rare spill path
  deferred off the hot loop.
- **Miss path**: hashbrown pair-rank probes replaced by a flat
  `PairRankTable`; vocab-seeded short cache so every short vocab word hits
  on first touch; stack-array + NEON merge for short pretokens (miss-path,
  miss4). This is disproportionately an MT win: 16 cold per-worker caches
  each pay the miss path until warm.

**The walker's floor, proven.** After de-branching, the walker settled at
~40% of ST encode (from 49.4% at baseline) and three attempts to shrink it
further — phaseb (R2), walker4 (R4), walker5 (R5) — all measured ≤ 0.
Walker5's dependency analysis explains all three in one stroke:

- **Phase B (emission) is issue-bound**: exactly 25 instructions/span with
  zero fat, running at IPC ~6.5–7 of the 8-wide issue width — within ~15%
  of the floor for its mandatory instruction stream. The hot-line
  concentration at `e.ptr = p` is retire pileup, not a stall to fix.
- **Phase A (harvest) is chain-latency-bound**: ~190 dynamic
  instructions/batch at ~56–65 cycles/batch (IPC ~3), because the per-batch
  critical chain (ldp → classify → weighted-and → 4× addp → fmov v→x →
  scalar algebra → SWAR mul → BIT_POS load → store) is an irreducible ~50
  cycles, with weak (~1.3x) cross-batch overlap. Under predicted branches
  the dynamic op stream of a restructured or instruction-trimmed loop is
  identical, so restructuring (phaseb) and instruction cuts (walker4)
  *could not* move it — and a hypothetical 128 B walker only removes
  ~30–40 of ~380 instructions per 128 B, i.e. more instruction cuts.
  That lever is retired.

The other negative results trace the same boundary: emit3 (store staging)
optimized a dimension the core had slack in. On this core, only removing
mispredicts or shortening dependent-load chains moved the ST number.

**MT: orchestration, not encode, was the recoverable gap — and it is now
closed.** The steady-state encode scales; the campaign's MT wins came from
the phase structure around it, diagnosed one trace at a time: serial-prefix
allocation → pre-size + Arc-share (R1 salvage); scheduler-order LPT
violation → atomic in-order handout (R3); serial 9 GB teardown → fused
parallel copy+drop (R3) → which exposed the vm-map read/write convoy →
deferred teardown off the timed window (R4) → which converted blocked
threads into visible fault-path contention in the gather copy → overlap the
copy with the encode via prefix commit (R5). MT-specific overhead beyond
steady-state encode went from ~25% of the round-3 window to a fork ramp and
one tail chunk, both measured at their structural floors.

## 6. Adversarial review round

After round 4, five parallel review agents audited the full diff
(`49ff4cb..ff363be`, ~2.3 k insertions) through five lenses: unsafe/memory,
token identity, concurrency, edge cases + cross-round invariant
interactions, portability/API (reports in `scratchpad/review/*.md`).

**Zero critical correctness bugs in shipped paths found by diff review.**
Confirmed findings, all fixed in `1ffff3c`:

1. **SpanBatch/PretokenSpans soundness (MAJOR, API boundary)** — safe
   external code could implement the safe `PretokenSpans` trait and drive
   `SpanBatch::span`'s `from_raw_parts` with arbitrary `(ptr, len)`. Fixed:
   `unsafe trait` with a documented fill contract, `BatchEntry` fields and
   `SpanBatch::entries` now `pub(crate)`, per-impl SAFETY notes.
2. **Big-endian compile guard** — key packing and emit-loop token-lane
   stores assume LE; BE targets previously got silent wrong tokens, now a
   `compile_error!`.
3. **`add_special_token` duplicate-content seed divergence** — an added
   token whose content duplicates an existing short vocab entry left the
   parent's seeded cache on the old ID while forks reseeded to the new one.
   Fixed with insert-not-insert-if-absent seed sync + fork-side re-apply;
   regression test added (verified to fail pre-fix).
4. **Proto-mutation-after-fork invariant** documented on `WorkerPool` and
   the three mutators (latent, Rust-API-only).
5. Doc corrections (stale consumer list, `GIGATOK_NO_LPT`, aarch64-linux
   `+crc` note).

Post-fix verification: full release test suite clean (59+58 tests), 100 MB
identity counts exact, and the hot-path functions' generated asm
**bit-identical** to pre-fix — the soundness fixes cost zero cycles.

The instructive coda: the one genuine token-identity bug of the campaign
(§7, finding a) was *not* findable by reading the diff — it required the
right tokenizer's vocab (qwen3_5) at data scale. Diff review and data-scale
differentials are complements, not substitutes.

## 7. Heavy verification round, and the fixes it forced

Branch `opt/verify-heavy` (`e93835c`, `fef613a`) built a differential
suite far past the per-round gates:

- **Full-corpus (11.9 GB OWT) differentials, all green on real text**:
  mask-iterator vs shipped walker, family scalar-vs-mask, olmo3-vs-regex.
- **Public-path differentials**: 1 GB OWT through
  `encode_with_added_tokens_flat` vs an uncached plain-merge mirror
  (GPT-2), 200 MB each for olmo3/qwen2/qwen3_5/deepseek_v3, added-token
  join differentials built without `find_added_token`, 1 GB
  parallel-vs-serial ragged with fragmenting oversized docs and injected
  `<|endoftext|>` (LPT on and off).
- **Adversarial-shape tests**: end-of-allocation boundary fuzz (all six
  schemes), exact edge-length pretokens (15/16, 64/65, 65535/65536 ×
  letter/digit/space/punct/multibyte/invalid fills), packed-key vs naive
  lanes at every page offset.

Three real findings:

1. **CRITICAL (campaign regression): qwen3_5 wrong tokens from vocab
   seeding.** Round 1's seeded cache stored every short vocab entry as
   `pretoken → [own id]`, and the long miss path kept a whole-pretoken
   `vocab_inv` shortcut — both assume every vocab entry is derivable from
   its own merges. False for qwen3_5: 201 merge-unreachable entries
   (multi-char CJK phrases, `" Japón"`, …). `encode(" Japón")` returned
   `[209344]` instead of HF's `[604, 385, 3064]`. The baseline had no such
   shortcut; gpt2/olmo3/qwen2/deepseek_v3 hid the bug because their only
   unreachable entries are added-token contents, split out before
   pretokenization.
2. **Pre-existing (baseline `0e27c71`): truncated-UTF-8 tail overrun.**
   `decode_cp` trusted the lead byte, read up to 3 bytes past the buffer on
   a truncated multi-byte sequence, and could emit a pretoken end past
   `len` — a slice panic on the Iterator path, a silently out-of-bounds
   span on the SpanBatch path. All six schemes affected; reachable from the
   public `&[u8]` API.
3. **Nondeterministic splits of >65 KB invalid-UTF-8 pretokens** (~1/25
   full-suite runs, only with concurrent threads in the process; never in
   isolation): Iterator and two-phase paths split 0xFF-run pretokens
   differently at the u16-window edge.

### The fix round

- `opt/fix-seeding` (`8723407`) — the seed is now literally "precomputed
  misses": `seeded_pretoken_cache` runs the short merge over each entry's
  bytes via `merge_short`, factored out of `encode_pretoken_miss` so seed
  and miss can never disagree; merge-reachable entries still seed as their
  single own ID, so nothing changes for them. The long-path `vocab_inv`
  shortcut is removed with a comment forbidding its reintroduction, and
  `set_added_tokens` now owns the added-token cache sync, making seed-level
  cache state a **pure function of (vocab, added_tokens)** — exactly what
  fork reconstruction assumes. The qwen3_5 repro is un-ignored and passes,
  the 200 MB qwen3_5 public differential passes (and the other four still
  do), and the hot probe/emit loop's disassembly is instruction-identical.
- `opt/fix-walker-edge` (`8a012b3`) — finding 3 root-caused **with an
  AddressSanitizer proof**: `decode_cp` decodes invalid leads 0xF5–0xFF
  through the 4-byte branch, assembling "codepoints" up to 0x1FFFFF —
  past Unicode — and `class_of`'s `get_unchecked` then read up to ~246 KB
  past the 272 KiB class table: heap garbage whose contents depend on other
  threads' allocations (ASAN reports a heap-use-after-free read in
  `r50k::extended_masks` under heap churn). The Iterator and two-phase
  paths classify at different times, hence the scheduling-dependent split.
  Fix: the 4-byte branch clamps the assembled codepoint to ≤ 0x10FFFF, and
  `pos + 4 > len` routes to a `#[cold]` per-byte-bounded tail decode
  (truncated sequences consume exactly the remaining bytes and classify as
  Other), fixing finding 2 in the same stroke. Valid UTF-8 decodes
  **bit-identically**; hot-loop asm spot-checked identical; a 250-round
  concurrent FF-run stress (fails within a few rounds pre-fix) runs clean.

Both branches merged with all gates green (full release suite, 100 MB
identity counts, 50 MB differential, benches building). Only after this
did the final §1 numbers get measured.

## 8. What's left

- **The walker is done.** Floor proven (§5): phase B at issue width,
  phase A within ~2x of its SIMD-throughput bound with the gap owned by an
  irreducible ~50-cycle classify chain. No further walker rounds; the
  128B-walker idea from the pre-campaign notes is retired.
- **MT orchestration is essentially done** per the round-5 trace: ramp and
  tail at structural floors, gather overlapped into encode. The remaining
  MT gap is **per-worker steady state**: encode CPU is ~14.7 s MT vs ~11 s
  ST for the same input — E-core cycles plus 16 cold caches re-deriving the
  Zipf head 16× (Σ per-worker distinct ≈ 16 M vs 5.5 M ST). Levers: the
  miss path on colder workers, and probe footprint (a shared warm-head
  structure is the open idea, with the usual coherence-traffic risk).
- **x86-64 / EPYC porting.** The aarch64 asm pins (`movemask64` addp tree,
  walker3/walker5 pins) and NEON paths all have portable fallbacks, so x86
  builds run — but an EPYC target currently gets round-1-era codegen on
  those paths; the AVX2 equivalents are unwritten. The MT gather numbers
  are macOS-specific (16 KB pages, no THP): Linux THP will change the
  fault/convoy arithmetic and the prefix-commit win should be re-measured
  there.
- **CRC feature flag on aarch64-linux.** The CRC32 key hash needs
  `-C target-feature=+crc` on aarch64-linux (documented); worth wiring as a
  build default or runtime dispatch before non-macOS deployment.

## 9. Branch inventory

| branch | disposition | measured delta (vs round control) |
|---|---|---|
| `opt/probe-emit` | merged R1 | +6.2% MT / +27.6% ST-materializing |
| `opt/walker-twophase` | merged R1 | +3.4% MT |
| `opt/keypack` | merged R1 | +1.7% |
| `opt/miss-path` | merged R1 (`160bdd0`) | +1.3%; vocab-seed identity bug found in verification, fixed in `8723407` |
| `opt/mt-alloc` | rejected R1 (kept) | −5.9% as a unit; bisected to streaming-gather committer thread |
| `opt/mt-salvage` | merged R1 | salvage of mt-alloc: pre-size + Arc-share + LPT |
| `opt/emit2` | merged R2 | +5.2% MT / +7.4% ST |
| `opt/phaseb` | rejected R2 (kept) | −1.5%; explained by the walker5 chain analysis |
| `opt/rank2` | held R2 (kept) | ±0 standalone; parts re-taken in miss4 |
| `opt/mt3` | merged R3 | +6.2% MT |
| `opt/walker3` | merged R3 | +2.1% MT / +6.2% ST |
| `opt/emit3` | rejected R3 (kept) | −4..−7% (both variants) |
| `opt/mt4` | merged R4 | +3.4% MT |
| `opt/driver4` | merged R4 | +0.7% MT / +1.9% ST |
| `opt/miss4` | merged R4 | +0.9% ST |
| `opt/walker4` | rejected R4 (kept) | ±0 despite 113→88 inst/batch; explained by the walker5 chain analysis |
| `opt/mt5` | merged R5 (`4772682`) | +4.4% MT (8680 vs 8311 mean) |
| `opt/walker5` | rejected R5 (kept) | ±0; its dependency analysis proves the walker floor |
| `opt/verify-heavy` | merged (verification) | no perf delta; 3 findings (1 campaign, 2 pre-existing/mixed) |
| `opt/fix-seeding` | merged (`8723407`) | identity fix; hot probe/emit asm instruction-identical |
| `opt/fix-walker-edge` | merged (`8a012b3`) | correctness fix; valid-UTF-8 path bit-identical |
| review fixes | `1ffff3c` on mainline | hot-path asm bit-identical |
| `opt/split-table` | rejected post-R5 (kept, `288fa8c`) | −22% MT / −32% ST; see §10 |

## 10. Postscript: split-table experiment — rejected

One more cache-footprint idea was built and measured after round 5:
`opt/split-table` (`288fa8c`) splits the short-pretoken cache into a
16-byte-entry "tiny" table for ≤ 7-byte pretokens — which carry 81.7% of
probe volume — alongside the existing wide table, halving the table bytes
touched by the common case.

The implementation was correct: all differentials passed, including the
200 MB qwen3_5 and 1 GB GPT-2 public-path suites. The performance was
decisively not. Interleaved A/B against mainline measured:

- **MT 6639 vs 8521 MB/s mean (−22%)**
- **ST 698 vs 1024 MB/s (−32%)**

Two costs outweigh the footprint halving on M4: the dual-probe ALU
overhead (~10–14 extra ops per pretoken to route and probe two tables),
and a **+0.71 pp slow-path rate** from tiny-table values that need 3+
tokens spilling out of the 16-byte entry. This is consistent with §5's
story — on this core the probe path was never footprint-bound, so paying
issue-bandwidth to shrink it loses.

Open question the rejection leaves behind: on the 255-thread EPYC target,
aggregate table bytes across workers matter far more than on a
16-thread M4, so the footprint argument may still hold there. The M4
numbers make it a low-priority retry; any retry should start from
`opt/split-table` rather than reimplementing.

## 11. Final fresh-eyes review

After the mt5/fix merges, a fresh review agent audited the three newest
change sets — the mt5 `Committer`, fix-seeding, and fix-walker-edge — plus
the two hand-resolved merges, against the `4772682` blobs
(`scratchpad/review/final_fixes.md`). Verdict: **no critical or major
correctness bugs**; the full hunt list (Committer gaplessness, panic
safety, memory ordering, overflow fallback; seed-equals-miss purity and
fork exactness; walker guard arithmetic and truncated-tail semantics
across all six schemes) verified sound.

Minor findings fixed on mainline in `dd28809`:

- **Committer post-move pointer re-derivation** — the reservation's base
  pointer was captured before the Vec moved three times, a
  strict-aliasing (Stacked-Borrows move-retag) hazard; fine under today's
  semantics but now closed outright.
- **`SyncPtr::at` marked `unsafe fn`** — it performed pointer `add` under
  a caller contract while presenting a safe signature.

Remaining minors were accepted and documented rather than fixed:
`set_added_tokens` can strand superseded arena slices across repeated
mutation (bounded, loader-phase, at most a few hundred KB on realistic
tokenizers), and `gather_fallbacks_match`'s "mid-flight overflow after a
committed prefix" case is timing-probable rather than guaranteed (the
identity assertion holds on both escape paths regardless).

## 12. Test-suite consolidation

Five rounds of agent branches accreted overlapping test modules into
`tiktoken.rs`/`batch.rs`; `a13c125` consolidated them (−456 lines) with a
full branch-diff audit confirming nothing from `opt/verify-heavy` was
lost:

- Duplicated fuzz/edge tests dropped in favor of the walker_edge ports,
  which are strictly stronger (no truncated-UTF-8 tail sanitization —
  that bug is fixed — plus the 0xFF+letter-tail shape).
- Stale `#[ignore]` known-bug repros removed: both bugs are fixed and
  covered by passing, non-ignored regression tests; the assert-free
  diagnostic scratch module used to hunt the 0xFF-run flake deleted.
- Heavy OWT differentials renamed into size tiers (`..._50m`, `..._200m`,
  `..._1g`) with their run commands documented, so the cheap/expensive
  split is explicit.

## 13. Campaign ledger

**30 Fable subagents** end to end: 1 profiling setup, 5 brainstorm/
proposal, 11 implementation, 2 merge/bisect, 6 review, 1 heavy
verification, 2 fix, 1 test consolidation, 1 report — plus this addendum
agent and an x86-port-prep agent running concurrently on `opt/x86-port`
(in flight at the time of writing; not counted above).
