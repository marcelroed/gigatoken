# x86-64 (EPYC / Zen 2) port of the encode-opt campaign

Branch `opt/x86-port` off `encode-opt-main` @ dd28809. Prepared on the M4 Max
box: everything here is either (a) implemented and compile/codegen-verified
for x86-64, or (b) a prioritized measurement plan for the EPYC session. **No
timing was run on this machine** — Rosetta numbers would be garbage and the
production target is a 255-thread Zen 2 EPYC anyway.

Codegen evidence below comes from two sources, both with
`-C target-cpu=znver2` (and cross-checked with `x86-64-v3` where noted):

- standalone shape files replicating the exact code shapes
  (`rustc -O --emit asm --target x86_64-apple-darwin`);
- the real fat-LTO bench binary
  (`RUSTFLAGS='-C target-cpu=znver2 -C link-args=-Wl,-undefined,dynamic_lookup'
  cargo build --release --bench encode_st --target x86_64-apple-darwin`,
  then `objdump -d`). The extra link arg only defers libpython symbols so the
  cdylib/bench link on a python-less x86 prefix; it does not affect codegen.

aarch64 stays bit-identical: every change is `cfg(target_arch = "x86_64")`-
gated (the one shared cfg edit, the multiply-hash fallback's `not(...)`,
excludes only x86+sse4.2 combos, so every aarch64 configuration selects
exactly the arm it did before). Verified functionally after rebuild:
`ENCODE_MB=100` encode_st = 22834020 tokens, encode_doc (gpt2) = 22723342
tokens, `cargo test --release` all green.

---

## 1. Implemented on this branch

### 1.1 `ProbeView::probe_pair`: cmov-of-VALUES asm pin (src/bpe/pretoken_cache.rs)

The x86 fallback was the mask-arithmetic select. LLVM re-canonicalizes it —
on znver2 **and** x86-64-v3 alike — into the same dependent-address shape the
round-2 agent fought on aarch64:

```
sete   %r11b                    ; m0
leaq   16(%rdi,%rax), %r9       ; &e0.val
leaq   48(%rdi,%rax), %rax      ; &e1.val
cmoveq %r9, %rax                ; ADDRESS select
movq   (%rax), %r13             ; dependent load of val
...
notb %r9b; andl $1; shll $5     ; recompute select as 0/32 offset
movq   24(%r9,%r8), %r9         ; ext: computed-index dependent load
```

i.e. an extra L1 load latency on the probe's critical path (the next thing
waiting on `val` is the emit store and the cursor advance, the loop's only
carried dependency). The `if m0 { v0 } else { v1 }` spelling compiles to the
same computed-index loads (`setne; shll $5; movq 16(%r9,%r8)`).

Implemented the x86 analog of the aarch64 csel pin: `test` + two `cmovne`
over the four unconditionally loaded value words (cmov is baseline x86-64,
no feature gate). In the real fat-LTO emit loop
(`probe_emit_chunk` inlined into the bench main) the pin survives:

```
vmovdqa (%r15), %xmm0             ; key (u128)
vpxor  (%rsi,%r12), %xmm0, %xmm1  ; e0.key ^ key
vpxor  0x20(%rsi,%r12), %xmm0, %xmm0
movq   0x30(%rsi,%r12), %r10      ; e1.val   } all four value words
movq   0x38(%rsi,%r12), %rbx      ; e1.ext   } loaded up front,
movq   0x10(%rsi,%r12), %rdi      ; e0.val   } independent of the
movq   0x18(%rsi,%r12), %r12      ; e0.ext   } compares
vptest %xmm1, %xmm1 ; sete %r11b  ; m0
vptest %xmm0, %xmm0 ; sete %r13b  ; m1
testq  %r11, %r11
cmovneq %rdi, %r10                ; val: register-value select
cmovneq %r12, %rbx                ; ext: register-value select
...
movq %r12, (%rdi,%rax,4)          ; two 8-byte stores
movq %rbx, 0x8(%rdi,%rax,4)
andl $0x7f, %r10d; addq %r10, %rax ; cursor advance off the cmov'd val
```

Two loads more per probe from the same already-touched line, all issued in
parallel — the identical trade the aarch64 pin makes. (Bonus visible above:
LLVM turned the u128 key compares into `vpxor`+`vptest`, one 16-byte load per
slot key.) The `fast` predicate is split into three short-circuit branches
(key==0 / !found / spill) rather than one combined test; all three predict
not-taken ~99% of the time. Left alone — listed under measurements (§3.4).

### 1.2 `pretoken_key_hash`: SSE4.2 CRC32C arm, runtime-selected (src/pretokenize/mod.rs)

x86 previously always took the multiply fold. Added
`_mm_crc32_u64(_mm_crc32_u64(0, lo), hi)`, mirroring the aarch64 `__crc32d`
arm:

- same key0→hash0 property (`crc32(0,0) == 0` — the fill loops' long-pretoken
  route stores hash 0);
- linear over GF(2), low 32 bits see every key bit — fine for any table under
  2^32 slots;
- Zen 2: `crc32q` is 1 µop, 3-cycle latency, 1/cycle throughput → 6-cycle
  chain vs the ~7-cycle 5-op multiply fold, and 3 µops vs 5 in a loop that is
  partly issue-bound.

**Selection is per-process at runtime, not per-build.** `sse4.2` is NOT in
baseline x86-64, and no wheel/CI config sets `-C target-cpu` /
`target-feature` (the distributed wheels must stay baseline), so a
compile-time `cfg(target_feature = "sse4.2")` arm would ship in no artifact.
Instead one process-immutable bit, `crc_hash_selected()` =
`is_x86_feature_detected!("sse4.2")` (std-cached CPUID; const-folds to `true`
when `sse4.2` is statically enabled, e.g. `-C target-cpu=znver2`), picks the
arm:

- `pretoken_key_hash` — the entry every cold/slow site uses (`grow`'s rehash,
  vocab seeding, `set_added_tokens`, the encode slow paths, tests) — branches
  on the bit per call (a cached atomic load + test, nothing on the hot path);
- the three hot fill loops (`fill_spans_two_phase`,
  `fill_spans_keyed_with{,_buf}`) dispatch ONCE per fill (≤ 256 spans) on the
  same bit into an `#[target_feature(enable = "sse4.2")]` monomorphization
  whose per-span hash is the CRC arm with zero per-span checks (the `false`
  monomorphization embeds the fold; see `fill_span_hash`'s reachability
  contract).

Single-function discipline is therefore structural, not per-build: hash(key)
= `crc_hash_selected() ? crc32c : fold` everywhere, and the bit is a pure
function of the CPU, so fills, `grow`, and the seeding paths agree in every
process — baseline wheels included. (Deliberately NOT tied to the fill
loops' AVX2 scanner gate: AVX2 does not architecturally imply SSE4.2, and
the deepseek/scalar fill routes bypass that gate entirely.)

Codegen, baseline (no `-C target-cpu`) fat-LTO bench binary: `crc32q`
present in every sse4.2 fill monomorphization and in
`pretoken_key_hash_crc32c` for the cold sites; the fold multiply remains
only in the (runtime-unreachable-on-SSE4.2-CPUs) non-CRC arms. With
`-C target-cpu=znver2` the guards const-fold: `xorl; crc32q; crc32q` inlined
at all six `fill_spans_keyed_mask` monomorphizations (twice per scheme:
phase-B fast loop + careful near-EOF loop), plus `grow`,
`seeded_pretoken_cache`, and `set_added_tokens`, and no *pretoken-hash*
multiply fold (the ror-25 shape) anywhere in that build — the ~11 residual
`0x9E3779B97F4A7C15` multiply sites in the znver2 binary are
`PairRankTable`'s separate, arch-independent hash (src/bpe/mod.rs), not the
pretoken hash.

The aarch64 arms are untouched and still compile-time: the aarch64-linux
`crc` note (generic aarch64-linux needs `-C target-feature=+crc`) still
applies there.

Compile-verified: `cargo check --release --target x86_64-apple-darwin` plain
(baseline → multiply arm) and with `RUSTFLAGS='-C target-cpu=znver2'` (crc
arm), plus the full fat-LTO build above. `x86_64-unknown-linux-gnu` check
stops at zstd-sys needing a Linux C sysroot on this Mac — the Rust cfg
surface is covered by the darwin-x86_64 checks (the only linux-gated code,
the three `madvise` blocks, is arch-independent and predates the campaign);
run one `cargo check` on the EPYC box for completeness.

---

## 2. Audited: correct on x86 as-is (no code change)

### 2.1 movemask64 `addp` pin — does not apply on x86

The round-3/4 `addp` reduction pin is inside the aarch64-only `movemask64`.
Both x86 tiers never form 0x00/0xFF vectors to reduce: AVX2 uses a
`vpmovmskb` ladder per predicate, AVX-512's k-register compares ARE the u64.
Checked `ascii_masks_avx2` in the znver2 binary: `#[inline(never)]` held
(standalone symbol, called `callq` from every scheme), all splat constants
are constant-pool `vmovdqa (%rip)` loads hoisted once — no splat
rematerialization — `andnq` (BMI1) for `wt & !n`, `vzeroupper` on exit.
The documented reason for `inline(never)` (LLVM pulling the caller's scalar
bit algebra back into the byte-vector domain, measured 3.5x on Zen 2) is a
Zen 2 measurement from the portability review; keep it.

### 2.2 Two-phase fill: dispatches on x86, phase B is branch-free on znver2

`fill_spans_keyed_mask` → `simd_scanner_available()` → true on any
AVX2 machine (all Zen), so `fill_spans_two_phase` IS the x86 path — Zen 2
takes the AVX2 tier (`avx512_scanner_available()` false). Codegen of the
phase-B emission loop in the real binary (r50k monomorphization, identical
shape in the other five):

- `min(tok_len, 15)`: `cmovbq` (value cmov);
- `PACK_MASK_TABLE` row: two loads off the shifted index (x86 has no `ldp`;
  both hit the same 16-byte row / cache line);
- the `keep` long-span guard: **folded to `cmovaeq` of zero on both key
  halves and meta** — the AND-mask spelling did its job; no re-introduced
  data-dependent branch (the aarch64 pathology the comment warns about does
  not reappear here);
- hash: inlined `crc32q` pair; then `andl %edx, %r11d` (mask) +
  `prefetcht1 (%rsi,%rdi)` — the L2 prefetch mapped correctly.

The `black_box(PACK_MASK_TABLE.as_ptr())` pin (aarch64: stops adrp+add
remat) is harmless on x86: the pointer sits in a register/stack slot,
loaded outside the loop.

`flatten_bits` (BIT_POS LUT store loop): the portable `for t in 0..8` copy
autovectorizes exactly as the code comment claims — per octet one
`vpaddw (mem), %xmm0` (LUT load folded into the add) + `vmovdqu` 16-byte
store, fully unrolled, SWAR prefix sums intact. LLVM additionally recognizes
`incl >> 56` and emits a hardware `popcntq` for the return value — a
beneficial fold, not a round-trip. Minor: each octet re-splats
`rel + 8j` (`vmovd` + `vpbroadcastw`, 8x per call); a hoisted splat +
constant-vector increments would save ~16 µops/call — plan-only (§3.5).

The `cmtst` pin and the flatten_bits popcount pin live in walker5, which is
unmerged — out of scope, per instructions.

### 2.3 Prefetch hint mappings

Both flavors already have x86 arms and they are the right ones:
- `ShortPretokenCache::prefetch_l2` = `_MM_HINT_T1` ↔ aarch64 `pldl2keep`
  (fill-phase, a chunk ahead) — emitted as `prefetcht1` in the fill loops;
- `ProbeView::prefetch` = `_MM_HINT_T0` ↔ aarch64 `pldl1keep` (probe-stage,
  D=16 ahead) — emitted as `prefetcht0` in the emit loop (warmup + steady
  state).

No write prefetch survived the campaign anywhere in src/ (`grep pstl /
prefetch_w / _MM_HINT_ET`: zero hits), so nothing to map. Semantic note for
Zen 2: `prefetcht1` fills L2 (and L3 shadow), `prefetcht0` fills L1+L2 —
same intent as the ARM hints; but Zen drops software prefetches that miss
the dTLB, which makes §3.2 (THP verification) a prerequisite for the
prefetch chain working at all on the big table.

### 2.4 MADV_HUGEPAGE paths — intact post-campaign

Three sites, all `cfg(target_os = "linux")`, all present at dd28809:
1. `Slots::new_zeroed` (pretoken_cache.rs) — the 2 MiB-aligned short-table
   allocation;
2. `Committer::try_new` (batch.rs) — the overlapped-gather reservation;
3. `gather_flat` (batch.rs) — the fallback flat buffer.

Bench-harness interaction: both `encode_st` and `encode_doc` mains already
clear `PR_SET_THP_DISABLE` via `prctl` on Linux (session managers set it and
silently veto MADV_HUGEPAGE). For the EPYC session additionally verify:
- `/sys/kernel/mm/transparent_hugepage/enabled` is `madvise` or `always`;
- after warmup, `grep -B3 AnonHugePages /proc/<pid>/smaps | grep -A3 <table
  region>` shows the table actually backed by 2 MiB pages;
- if the Python bindings path is benched, the prctl is NOT run there — an
  encode driven from Python inherits the launcher's THP policy. Check how
  the production harness launches.

---

## 3. EPYC measurement plan (prioritized)

Protocol for every A/B below (established campaign practice): sequential
runs only, never parallel; variants interleaved A,B,A,B,…, min-of-N (N ≥ 5
for encode_st at ±8% run-to-run variance; whole-file encodes by path); token
counts asserted equal to the aarch64 reference values every run. Build with
`RUSTFLAGS='-C target-cpu=znver2'`; on Zen 2 `native` ≈ `znver2`, but pin it
explicitly so the wheel/CI story is reproducible. Rebuild between variants
(`cargo bench --no-run`), verify with `objdump` that the shape under test is
present before timing.

### 3.1 Baseline + the two implemented items (first hour)

1. `ENCODE_MB=100` (then 1000) `encode_st`, gpt2: baseline tokens/s on the
   branch as-is. Sanity: 22834020 tokens at 100 MB.
2. probe_pair A/B: this branch vs the same build with the x86 asm arm
   commented back to mask-arith. Expected value: removes one L1 latency
   (Zen 2 ~4-5 cycles) plus a 2-3 µop address computation from the
   per-pretoken critical chain. The aarch64 analog was worth low-single-
   digit % of warm encode; Zen 2's shallower OoO window and slower L1
   (vs M4) argue for at least as much. Estimate: +1.5-4% encode_st warm.
3. CRC hash A/B (`-C target-cpu=znver2` vs the same minus
   `-C target-feature=-sse4.2`... simplest: patch the cfg off). Shortens the
   fill-loop hash chain by ~1-2 cycles/span and 2 µops. Estimate: +0.5-2%.
   Also confirms no hash-quality regression on the real table (probe walk
   lengths: `cache: short N entries (cap C)` line should show the same
   occupancy; watch probe_emit_slow share in a profile).

### 3.2 THP / dTLB (prerequisite for the prefetch story)

The whole prefetch architecture (L2 stage a chunk ahead, L1 promote D=16
ahead) presumes the table's lines can be prefetched; Zen 2 drops SW
prefetches on dTLB miss. Verify AnonHugePages backing (see §2.4). If the
production launcher disables THP, that single config issue can be worth
more than every micro-optimization in this document combined (the doc-
comment history: ~500x fault count and dTLB coverage of the 64 MB+ table).

### 3.3 Prefetch distance re-calibration

`D=16` (probe-stage) and the chunk-size L2 stage distance were tuned on M4
Max (L2 hit ~26 cycles there; Zen 2 L2 ~12 cycles but DRAM ~110 ns and only
2 outstanding prefetches per load port... different balance entirely).
A/B D ∈ {8, 16, 24, 32} (compile-time const, cheap sweep) and, if the L2
stage looks weak in a profile (probe stalls despite prefetches), consider
prefetching with T0 at the fill stage on x86 (Zen's L2→L1 promotion is
cheaper than ARM's; T0-everything is a known-good shape on some Zen loads).
Estimate: 0-3%, possibly negative — measure, don't guess.

### 3.4 Emit-loop predicate shape

The `fast` predicate compiles to three predicted-not-taken branches (§1.1).
Alternatives to A/B if a profile shows front-end pressure: fold
`found & !spill & (key != 0)` into one register test + single branch
(LLVM currently refuses; would need a `black_box` or asm-flag trick).
Only chase this if `topdown`-style PMU counters show branch/fetch bound in
the emit loop. Estimate: 0-1.5%.

### 3.5 flatten_bits splat hoist (micro)

Hoist the `rel` splat out of the 8-octet unroll (one `vpbroadcastw` + 8
`vpaddw` with constant vectors, or accumulate `+8` increments). Saves ~16
µops per 64-byte batch in phase A. Only worth testing if phase A shows up
in the profile at all (on M4 it is dwarfed by phase B and the scalar
zones). Estimate: 0-1%.

### 3.6 AVX-512 note

Zen 2 has no AVX-512: the AVX2 tier is the production path; `ascii_masks_avx2`'s
`inline(never)` and the vpmovmskb ladder are the objects of interest in the
pretokenize profile. If a Zen 4/5 or Ice Lake box ever becomes the target,
the AVX-512 tier is already written and dispatched — re-run the same suite
there before trusting it (it has never been timed on metal either).

---

## 4. 255-thread parameter audit (all calibrated at 16 threads on M4)

Everything below is arch-neutral code whose CONSTANTS encode 16-thread
assumptions. None of it should change before being measured on the EPYC box
with a production-shaped input (multi-GB, many docs).

| Parameter | Value / rule | 16-thread meaning | At 255 threads | Risk / action |
|---|---|---|---|---|
| `chunk_target_bytes` | `total/(16·threads)`, floor 1 MB (`MIN_CHUNK_BYTES`) | ~16 chunks/thread | 4080 chunks; 10 GB input → 2.45 MB chunks (floor not hit until total < 4.1 GB) | Below ~4 GB input every chunk is 1 MB and chunk count collapses toward `total/1MB` < 4080 → fewer than 16/thread; work-stealing granularity fine, but per-chunk overhead × 4080 and the `outs: Vec<OnceLock>` scan in `Committer::advance` grow. Measure encode_doc at 1, 10, 50 GB; consider raising the divisor only with data. |
| LPT shape | 2×target head over first 80% of bytes, target/4 tail chunks | tail rides E-cores | 255 homogeneous cores; tail chunks 0.6 MB (floor-clamped to 1 MB) | The E-core rationale is gone; `GIGATOK_NO_LPT` exists precisely for this A/B. Run both. |
| `Committer::MAX_DRAIN` | 8 chunks per `advance` | bounded lock hold | 255 workers completing ~4080 chunks race one Mutex; try_lock skips are the safety valve, but the holder copies up to 8×~1-2.5 MB with 254 threads generating completions | Watch commit-lock hold times and the residual drain in `finish`. Levers: MAX_DRAIN down (shorter holds) or chunked `copy_nonoverlapping` outside the lock (bigger change). Measure first: the try_lock design may just work. |
| `DEFERRED_DROP_MIN_BYTES` | 32 MB | background drop off critical path | unchanged semantics; the deferred drop occupies 1 of 255 threads — cheaper than ever | No action. |
| Heaps table sizing (`fork_sized`) | `3.45·share^0.62 · 4/3 · 1.4`, clamp 2^16..2^22 slots (2 MB..128 MB @ 32 B/slot) | 10 GB/16 → ~2^21 slots, 64 MB/worker | share = total/255. 10 GB → ~39 MB/worker → ~176k distinct → 2^19 slots = **16 MB/worker → ~4.2 GB aggregate**. Clamp max (128 MB × 255 = **32.6 GB**): `fork_sized` clamps to 2^16..2^22 slots **before** `.next_power_of_two()`, so any raw estimate > 2^21 rounds UP to the full 2^22-slot / 128 MB table — reached at share ≈ **0.78 GB** ⇒ total ≈ **200 GB** at 255 workers (at 16 workers the 128 MB/worker regime already starts at ~13 GB total; the 10 GB/16 example sits just under the boundary, estimate 1.83M < 2^21). Unreachable in one batch at the planned 10-50 GB inputs, but a *long-lived pool re-used across many calls* grows each worker toward its corpus-lifetime distinct count with no shrink path | Flag: aggregate table memory = workers × table. 4-8 GB steady-state is fine on a 512 GB-1 TB EPYC but must be a conscious decision; check RSS after a long run. Also 255 × zeroed-table memset + ~50k-insert vocab seed at pool warmup (the "fork+seed ramp" was ~32 ms at 16 workers) — measure first-call latency; consider lazy/staggered forks only if it hurts. |
| `token_arena` / long-map hints | `share/256` capped 16M entries; `share/8192` capped 1M | ~64 MB cap | share smaller → hints smaller; caps unreachable | No action. |
| Rayon pool size | `current_num_threads` everywhere | 16 | 255 (SMT2 on 128c?) | Decide threads = physical cores vs SMT for encode (memory-bound: SMT often negative). A/B `RAYON_NUM_THREADS=128` vs 255. Also NUMA: 2-socket or 4-quadrant NPS config changes DRAM locality of the shared flat buffer and each worker's table. A/B `numactl --interleave=all` vs default first-touch. |
| `PRETOKEN_CHUNK` + `SPAN_BATCH_SLACK` | chunk of spans between fill and probe phases | sized so L2-staged lines survive until probe on M4 (L2 per-core) | Zen 2 L2 is 512 KB/core private, L3 16 MB per 4-core CCX, victim | If the chunk's staged lines (PRETOKEN_CHUNK × 64 B) plus the walker set overflow 512 KB the T1 stage thrashes; likely fine (chunk ≪ 512 KB) but confirm PRETOKEN_CHUNK × 64 ≤ ~256 KB and watch L2 miss rate in the probe loop. |

Also on the list for the MT session: verify `probe_emit_slow`'s share and
the miss-path `get_or_slot` walk lengths on OWT-shaped data with the CRC
hash (different bit mixing → different clustering; full-key compare makes it
a perf question only), and re-run the `GIGATOK_NO_LPT` A/B noted above.

---

## 5. Top 5 for the EPYC session, in order

1. **THP sanity** (§3.2): confirm MADV_HUGEPAGE takes effect under the
   production launcher (prctl, sysfs mode, smaps). Gates everything else.
2. **encode_st baseline + probe_pair pin A/B + CRC hash A/B** (§3.1):
   the two implemented wins; confirm counts (22834020 @ 100 MB gpt2) and
   collect the first Zen 2 profile (xctrace → perf/AMDuProf equivalent;
   weight by cycles).
3. **Thread topology sweep** (§4): RAYON_NUM_THREADS ∈ {64, 128, 255} ×
   numactl {default, interleave} on encode_doc 10 GB — the biggest single
   unknown on this box, and it sets the stage for every MT number after it.
4. **LPT / chunking A/B at 255 threads** (§4): GIGATOK_NO_LPT on/off,
   1/10/50 GB inputs; watch Committer lock contention and finish() residual.
5. **Prefetch distance sweep** (§3.3): D ∈ {8,16,24,32} + T0-vs-T1 fill
   stage, after THP is confirmed (meaningless before).

---

## 6. Zen 5 session (Ryzen 7 9800X3D, 16 threads, AVX-512): first on-metal x86 numbers

First x86 metal timing of the campaign (2026-07-14; the plan above targeted
Zen 2 EPYC — that session is still pending). Baseline runtime-dispatch build
(no `-C target-cpu`, exactly what wheels ship): the AVX-512 tier, the CRC
hash, and the probe_pair pin all engage via runtime detection. THP sysfs
mode `madvise` (bench prctl clears PR_SET_THP_DISABLE), corpora from
`~/data/owt_train.txt`.

Correctness first, all green on this box: `cargo test --release` (67 + 65),
`verify_memoized_encode_matches_reference_owt_50m`,
`verify_gpt2_public_encode_matches_reference_owt_1g` (incl. the 100 MB join
differential), `verify_multi_public_encode_matches_reference_owt_200m`,
`verify_parallel_ragged_matches_serial_owt_gpt2_1g` (both LPT modes),
`family_mask_matches_scalar_owt`, `mask_iter_matches_shipped_owt`. Token
counts match the aarch64 references every run (22834020 @ 100 MB gpt2,
228107519 @ 1 GB).

A/B protocol as in §3: interleaved variants, min-of-5, `encode_st` gpt2,
`ENCODE_MB=300 ENCODE_PASSES=3` (pass 0 cold, 1-2 warm), sequential runs.

| Variant pair | Cold (pass 0) | Warm (pass 1-2) | Verdict |
|---|---|---|---|
| probe_pair asm pin vs mask-arith (§3.1.2) | 583-591 vs 570-575 MB/s (**+2.5%**) | 1013-1031 vs 969-985 (**+4.6%**) | Pin confirmed on metal; keep. |
| CRC32C hash vs multiply fold (§3.1.3) | 587-592 vs 570-577 (**+2.2%**) | 1021-1029 vs 959-969 (**+6.3%**) | CRC arm confirmed; keep. |
| AVX-512 tier vs forced-AVX2 tier (§3.6) | 586-591 vs 576-585 (**+1%**) | 1018-1029 vs 1008-1025 (**+0.5-1%**) | AVX-512 dispatch is the right default on Zen 5 (k-register masks save the vpmovmskb ladder); margin is small because the boundary algebra, not classification, dominates. |
| Short-merge min-rank scan, AVX-512/AVX2 port vs scalar (NEW) | 471-480 vs 482-485 MB/s @ 100 MB; 661-670 vs 670-676 @ 1 GB (**-1%**) | n/a (miss path only) | **Negative — not dispatched.** The x86 horizontal reduce is a 4-dependent-op chain + vector→GPR `vmovd` on the merge loop's serial chain, and the `target_feature` boundary forces a real call per short merge (NEON inlines, gate-free); Zen 5 predicts the scalar scan's `rank < best` branches well at typical n ≈ 4-6, so M4's mispredict-driven win does not transfer. Both arms kept in `src/bpe/mod.rs` as tested reference (`short_merges_match_vec_merge_loop` covers them), dispatch stays scalar on x86_64. |

Not re-run here (Zen 2 EPYC items, still open): thread-topology and NUMA
sweeps, 255-thread LPT/chunking, prefetch-distance recalibration, smaps
AnonHugePages verification under the production launcher.

## 7. Zen 5 optimization round (profile-guided; branch encode-opt-x86-perf)

Driven by the perf deep-profile in `profiling/zen5_st_profile.md` (its §5 is
the ranked opportunity list this round worked through). Protocol as §3;
full-dataset numbers are 3 sequential runs with the page cache warm.

Applied, in order, each A/B-verified (interleaved min-of-5, 1 GB gpt2):

1. **THP ordering + alignment fixes — cold +15.6% / warm +8.0%.** Two
   distinct bugs: (a) `Slots::new_zeroed` zeroed via `alloc_zeroed` (=
   `aligned_alloc` + memset) *before* `madvise_hugepage`, so the table
   faulted in as 4 KiB pages and the hint was dead for the run — now
   madvise-then-zero; (b) `madvise_hugepage` passed Vec/malloc pointers
   (offset 16 into the mmap by the allocator header) straight to madvise →
   EINVAL, silently — every Vec-backed call (Committer reservation,
   gather_flat) has been a no-op since the sites were added; now aligns
   the start inward. Bench-side: input/output buffers madvised
   (`common::madvise_hugepage_capacity`), per-chunk MT `ids` madvised, and
   `benches/encode.rs` was missing `allow_thp()` entirely (it ran with
   THP vetoed under session managers that set PR_SET_THP_DISABLE).
   Verified: 1.93 GB AnonHugePages across table+input+output mid-run;
   dTLB walks collapse as in the profile's §3 A/B.
2. **`ProbeView` pre-folded pair mask — cold +1.8% / warm +2.2%.** The
   emit loop reloaded a spilled loop-invariant mask on the probe-address
   critical path every iteration (x86 register pressure the ARM build
   never saw, profile §4.2). Storing `mask & !1` removes one ALU op and
   one live temp; perf annotate confirms the stack round-trip is gone.
3. **`PairRankTable` dense grid widened to 2^11 (16 MiB, Arc-shared) —
   cold +2.8%, warm neutral.** Round-2+ merge lookups between the ~1.8k
   earliest (most frequent) merged IDs — hits AND no-merge misses — now
   answer in one L3-resident load instead of a sparse probe walk
   (profile §4.3/§5.4b).

Tried, measured, and NOT kept (all interleaved, same protocol):

- Prefetch D=32 (needs SPAN_BATCH_SLACK=32): neutral vs D=16 post-THP.
- AVX-512 short-merge scan re-test post-THP/dense: still −2.5% cold; the
  §6 verdict stands even with the memory stalls removed.
- Non-temporal `vmovntdq` gather copy in the Committer: neutral-to-−1%
  (the commit copy runs under the cursor Mutex; NT's longer store
  latency extends lock holds, and at 6 GB/s the box is not RFO-bound).

Full-dataset (11.9 GB OWT, page cache warm), before → after this round:

| bench | before | after |
|---|---|---|
| encode_st full pass 0 (gpt2) | 759–768 MB/s | **946–949 MB/s (+24%)** |
| encode 16T full (gpt2) | 4.99–5.15 GB/s | **6.08–6.10 GB/s (+18%)** |
| ST 1 GB cold / warm | 686 / 1013 MB/s | 835 / 1126 MB/s (+22% / +11%) |

Token counts bit-identical throughout (2717102153 ST / 2704046552 MT full;
228107519 @ 1 GB). Full suite + heavy differentials green on the final tree.

Still open (est. ≤ 1–3% each, from the profile's §5): `vpcompressb`
flatten_bits and the AVX-512 masked key load in phase B (both need an
AVX-512 monomorphization of the fill loop), rank-slot prefetch in the
merge scan, PGO for cold I-cache. MT topology: 16T > 8T (+14%) and LPT >
no-LPT (+3%) — both defaults already right on this box.

Rebase note: after rebasing onto the simplification commits (ff7c821 +
8d704a3, "A/B-verified neutral" on the ARM box), the same interleaved
protocol on this box measured the simplified tree **+5% cold** (610-625 vs
585-594 MB/s) and warm-neutral (~1014-1028 both) against the pre-rebase
binary — i.e. the simplification is a small cold-path win on Zen 5, not
just neutral. Full suite + the heavy differentials (1 GB gpt2 + join,
200 MB qwen3.5 multi, 1 GB parallel ragged both LPT modes) re-verified
green post-rebase, counts unchanged. Protocol reminder this session
re-taught: back-to-back NON-interleaved sessions on this box can differ
by ~4% from post-compile thermal state — a standalone triple-run of the
rebased binary first read as a 4% regression that interleaving inverted.
Never compare across sessions; interleave or it didn't happen.

## 8. Zen 5 round 2: the §7 leftovers + the qwen2-family classifier

Fresh profile on main @ b092ad7 (worktree `encode-profiling`) reproduced
the §7 residual shape exactly (gpt2 1 GB: 833 cold / 1122 warm; fill
44.7% / emit 38.7% / merge_short 16.8% cold), plus a first profile of the
qwen2-scheme path (Qwen3-8B tokenizer, same corpus: 720 cold / 965 warm —
the gap is classification: `batch_masks_avx512::<class_of>` 17.6% warm +
`family_extended_masks` 8.1%, vs r50k's 13.1% total). All changes below
interleaved min-of-4/5 at 1 GB, token counts bit-identical throughout.

Applied:

1. **`vpcompressb` flatten_bits (§5.5) — gpt2 +3-6% cold / +4-8% warm;
   carries to every mask scheme.** New `fill_spans_two_phase_crc_avx512`
   shim (`+avx512vbmi2`, gated on `avx512_fill_available()`, dispatched
   once per fill) monomorphizes the fill with `const X86_AVX512`; phase A's
   8-octet BIT_POS LUT walk becomes compress-iota + widen + `vpaddw rel` +
   two unconditional 64 B stores (BOUND_BUF slack grown 144 → 208 for the
   128-lane scribble; worst-case cursor analysis in the comment).
2. **Rank-slot prefetch in the short-merge refresh (§5.4a) — cold +0.8%,
   warm neutral.** After the min-rank scan picks `best`, both refresh
   pairs' first rank addresses (dense cell or first sparse slot) are known
   before the list surgery; `PairRankTable::prefetch_rank` issues both
   `prefetcht0`s there, overlapping the two serial loads with the surgery.
3. **Family classifier: LazyLock hoist + 2-byte fast lane — qwen2 warm
   +1.3%, cold +0.5%; r50k neutral.** Per-char `class_of` paid a LazyLock
   state check + Box-pointer load before the table index; schemes now
   resolve a `ClassTable` handle once per batch and pass a capturing
   closure. `classify_uni_chars` gained a lead `< 0xE0` lane (inline
   2-byte decode, constant masks) — western corpora's dominant non-ASCII
   case. Family fuzz differentials (all four schemes) + 100 MB OWT
   streaming differentials green.

Tried, measured, and REVERTED:

- **AVX-512 masked key load in phase B (§5.6): −36% warm / −30% cold.**
  The scalar pack's 16-byte load address depends only on `prev`, so it
  issues before `tok_len` resolves; `vmovdqu8 {k}{z}` made the load wait
  on the boundary→len→kmask chain and added a `vpextrq` GPR crossing
  before the CRC — ~90% of loop samples piled on the masked load and its
  dependents. Overlapped work became serialized. Do not re-try (comment
  at the phase-B loop records this).

Full-dataset (11.9 GB OWT, page cache warm, interleaved ×2), before →
after this round:

| bench | before | after |
|---|---|---|
| encode_st full pass 0 (gpt2) | 936-942 MB/s | **1005-1009 MB/s (+7.2%)** |
| encode_st full pass 0 (Qwen3-8B) | 805-810 MB/s | **862-863 MB/s (+7.0%)** |
| gpt2 1 GB cold / warm | 827 / 1120 | 879 / 1210 (+6% / +8%) |
| Qwen3-8B 1 GB cold / warm | 711 / 960 | 759 / 1043 (+6.7% / +8.6%) |

Counts: 2717102153 (gpt2 full), 2624517214 (Qwen3-8B full), 228107519 /
220398751 @ 1 GB — identical on every build. Still open: §5.7 PGO for
cold I-cache; qwen2 carry-forwarding across sequential batches (assessed,
skipped — the fill walker's rewinds make the invalidation story unsafe
and the sequential-case saving is ~10 L1-hot ops per batch).

Rebase note: this round was rebased onto a4bcfcb's x86-tier fill
monomorphization; the `const X86_AVX512` bool + `_crc_avx512` shim above
became a fourth tier value `X86_TIER_AVX512_VBMI2` (wrapper
`fill_spans_two_phase_avx512_vbmi2_crc` = AVX-512 tier features +
`avx512vbmi2`, still gated once per fill on `avx512_fill_available()`;
plain-AVX-512 CPUs without VBMI2 keep the `X86_TIER_AVX512` wrapper and
scalar flatten), and the classifier hoist moved into the schemes'
collapsed `batch_masks_x86` bodies (a silent rebase hazard: git
auto-applied the hoist only to the aarch64 `batch_masks`, leaving the
hot x86 bodies on the bare classifier — caught in review of the merged
tree). Numbers above predate the rebase; post-rebase interleaved re-A/B
vs the new main (which a4bcfcb itself sped up to 851 cold / ~1162 warm
gpt2, 747 / ~1020 qwen at 1 GB): gpt2 895 / ~1241 (+5.2% / +6.5%),
qwen 780 / ~1074 (+4.3% / +5.4%); full-file gpt2 ~1002-1027 vs ~974,
qwen 889 vs ~847 MB/s. Counts unchanged on every build.
