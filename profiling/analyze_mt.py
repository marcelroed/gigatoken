#!/usr/bin/env python3
"""Per-thread analysis of a samply profile of the encode_doc MT benchmark.

Adapted from opt/profiling/analyze.py (encode_st, single thread): this
variant walks EVERY thread, attributes threadCPUDelta (µs, kernel-reported
user+system CPU between consecutive samples) to a stack-derived category,
and reconstructs a per-thread busy/phase timeline so the serial head, the
straggler tail and the gather phase of the 1.46 s encode window can be
quantified in ms.

Usage:
  python3 analyze_mt.py TRACE.json.gz --bin path/to/encode_doc-HASH [-o OUTDIR]
"""

import argparse
import bisect
import gzip
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

SENTINEL = 0xFFFFFFFF0


def atos_resolve(lib_path, dsym_path, rel_addrs):
    if not rel_addrs:
        return {}
    obj = dsym_path if dsym_path and os.path.exists(dsym_path) else lib_path
    out = {}
    addrs = sorted(rel_addrs)
    sent_hex = hex(SENTINEL)
    CHUNK = 500
    for i in range(0, len(addrs), CHUNK):
        chunk = addrs[i : i + CHUNK]
        query = []
        for a in chunk:
            query += [hex(a), sent_hex]
        cmd = ["atos", "-o", obj, "-arch", "arm64", "-offset", "-i", "-fullPath"] + query
        res = subprocess.run(cmd, capture_output=True, text=True)
        groups, cur = [], []
        for line in res.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line == sent_hex:
                groups.append(cur)
                cur = []
            else:
                cur.append(line)
        if len(groups) != len(chunk):
            raise RuntimeError(f"atos group mismatch: {len(groups)} vs {len(chunk)}")
        for a, g in zip(chunk, groups):
            frames = []
            for line in g:
                m = re.match(r"^(.*?) \(in .*?\)(?: \((.*?):(\d+)\))?$", line)
                if m:
                    sym, f, ln = m.group(1), m.group(2), m.group(3)
                    frames.append((sym, f, int(ln) if ln else None))
                else:
                    frames.append((line, None, None))
            out[a] = frames if frames else [(f"0x{a:x}", None, None)]
    return out


def load_sidecar_symbols(trace_path):
    sidecar = re.sub(r"\.json(\.gz)?$", ".json.syms.json", trace_path)
    if not os.path.exists(sidecar):
        return {}
    d = json.load(open(sidecar))
    strs = d["string_table"]
    tables = {}
    for lib in d["data"]:
        entries = [
            (e["rva"], e.get("size", 0), strs[e["symbol"]]) for e in lib["symbol_table"]
        ]
        entries.sort()
        tables[lib["debug_name"]] = (
            [e[0] for e in entries],
            entries,
        )
    return tables


def sidecar_lookup(table, addr):
    rvas, entries = table
    i = bisect.bisect_right(rvas, addr) - 1
    if i >= 0:
        rva, size, name = entries[i]
        if addr < rva + max(size, 1) or size == 0:
            return name
    return None


def demangle_all(names):
    uniq = sorted(set(names))
    try:
        res = subprocess.run(
            ["rustfilt"], input="\n".join(uniq), capture_output=True, text=True
        )
        dem = res.stdout.split("\n")
        return dict(zip(uniq, dem))
    except FileNotFoundError:
        return {n: n for n in uniq}


def strip_generics(name):
    if name.startswith("<"):
        depth, i = 0, 0
        for i, ch in enumerate(name):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    break
        inner = name[1:i]
        inner = inner.split(" as ")[0]
        inner = strip_generics(inner) if "<" in inner else inner
        inner = inner.split("::")[-1] if "::" in inner else inner
        name = inner + name[i + 1 :]
    out = []
    depth = 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    s = "".join(out)
    while "::::" in s:
        s = s.replace("::::", "::")
    if s.endswith("::"):
        s = s[:-2]
    return s


# ---------------------------------------------------------------------------
# Categorization for the MT structure. First match over the whole stack
# (leaf-first) wins for LEAF rules; stack-wide markers checked afterwards.

LEAF_RULES = [
    # (substring of leaf symbol, category)
    ("__psynch_cvwait", "idle: condvar wait"),
    ("_pthread_cond_wait", "idle: condvar wait"),
    ("thread_park", "idle: condvar wait"),
    ("swtch_pri", "idle: yield"),
    ("sched_yield", "idle: yield"),
    ("thread_switch", "idle: yield"),
    ("kevent", "idle: kevent"),
    ("__workq", "idle: workq"),
    ("__read", "kernel: read"),
    ("__pread", "kernel: read"),
    ("madvise", "kernel: madvise"),
    ("__munmap", "kernel: munmap"),
    ("mach_vm_deallocate", "kernel: munmap"),
    ("__mmap", "kernel: mmap"),
    ("mach_vm_allocate", "kernel: mmap"),
    ("mach_vm_map", "kernel: mmap"),
    ("vm_copy", "kernel: vm"),
    ("bzero", "memset/bzero"),
    ("memset", "memset/bzero"),
]

STACK_RULES = [
    # (substring anywhere in stack, category) -- ordered, first hit wins
    ("seeded_pretoken_cache", "worker fork+seed"),
    ("fork_sized", "worker fork+seed"),
    ("assemble_ragged", "gather/assemble"),
    ("safe_split_ranges", "head: split scan"),
    ("build_doc_chunks", "head: split scan"),
    ("encode_chunk", "encode"),
    ("memoized_encode", "encode"),
    ("encode_with_added_tokens", "encode"),
    ("with_worker", "encode (worker acquire)"),
    ("load_owt_input", "bench: corpus read"),
    ("load_hf_bpe", "bench: tokenizer load"),
    ("drop_in_place", "drop/free"),
    ("_free", "drop/free"),
    ("rayon", "rayon overhead"),
    ("crossbeam", "rayon overhead"),
]


def categorize(pairs_leaf_first):
    if not pairs_leaf_first:
        return "<no stack>"
    leaf_sym = pairs_leaf_first[0][0]
    for pat, cat in LEAF_RULES:
        if pat in leaf_sym:
            # Refine kernel/memset/idle leaves with the project context so
            # e.g. memcpy inside the gather is separable from encode.
            for sym, _f in pairs_leaf_first:
                if "assemble_ragged" in sym:
                    return cat + " [gather]"
                if "seeded_pretoken_cache" in sym or "fork_sized" in sym:
                    return cat + " [fork+seed]"
                if "encode_chunk" in sym or "memoized_encode" in sym:
                    return cat + " [encode]"
                if "load_owt_input" in sym:
                    return cat + " [corpus read]"
            return cat
    if "memcpy" in leaf_sym or "platform_memmove" in leaf_sym:
        for sym, _f in pairs_leaf_first:
            if "assemble_ragged" in sym:
                return "gather memcpy"
        # falls through: memcpy inside encode attributes to encode below
    for sym, _f in pairs_leaf_first:
        for pat, cat in STACK_RULES:
            if pat in sym:
                return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bin", required=True)
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--bin-ms", type=float, default=25.0, help="timeline bin width")
    args = ap.parse_args()

    outdir = args.outdir or os.path.splitext(args.trace)[0].replace(".json", "") + "_mt_analysis"
    os.makedirs(outdir, exist_ok=True)

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        prof = json.load(f)
    libs = prof["libs"]
    bin_abs = os.path.abspath(args.bin)
    dsym = f"{bin_abs}.dSYM/Contents/Resources/DWARF/{os.path.basename(bin_abs)}"
    sidecar = load_sidecar_symbols(args.trace)

    threads = [t for t in prof["threads"] if t["samples"]["length"] > 0]

    # ---- pass 1: collect unique (lib, addr) across ALL threads ----------
    addrs_by_lib = defaultdict(set)
    thread_frame_meta = []  # per thread: (frame_lib, frame_addr)
    for t in threads:
        ft, fut, rt = t["frameTable"], t["funcTable"], t["resourceTable"]
        frame_lib = []
        for fi in range(ft["length"]):
            func = ft["func"][fi]
            res = fut["resource"][func]
            frame_lib.append(rt["lib"][res] if res is not None and res >= 0 else None)
        thread_frame_meta.append((frame_lib, ft["address"]))
        for fi in range(ft["length"]):
            li, a = frame_lib[fi], ft["address"][fi]
            if li is not None and a is not None:
                addrs_by_lib[li].add(a)

    resolved = {}
    for li, addrs in addrs_by_lib.items():
        libname = libs[li]["name"]
        if os.path.basename(libs[li]["path"]) == os.path.basename(bin_abs):
            try:
                r = atos_resolve(bin_abs, dsym, addrs)
            except Exception as e:
                print(f"warn: atos failed: {e}", file=sys.stderr)
                r = {a: [(f"{libname}+{hex(a)}", None, None)] for a in addrs}
        else:
            table = sidecar.get(libname)
            r = {}
            for a in addrs:
                name = sidecar_lookup(table, a) if table else None
                r[a] = [(name or f"{libname}+{hex(a)}", None, None)]
        for a, frames in r.items():
            resolved[(li, a)] = frames

    all_syms = [s for frames in resolved.values() for s, _, _ in frames]
    dem = demangle_all(all_syms)
    for k, frames in resolved.items():
        resolved[k] = [(strip_generics(dem.get(s, s) or s), f, ln) for s, f, ln in frames]

    # ---- pass 2: per-thread walk ----------------------------------------
    BIN = args.bin_ms
    report = open(os.path.join(outdir, "mt_report.txt"), "w")
    grand_cat_cpu = Counter()  # category -> cpu ms (all threads)
    timeline = defaultdict(Counter)  # bin index -> {category: cpu ms}
    thread_rows = []
    folded = Counter()

    for tidx, t in enumerate(threads):
        strs = t["stringArray"]
        ft, fut = t["frameTable"], t["funcTable"]
        st = t["stackTable"]
        frame_lib, frame_addr = thread_frame_meta[tidx]
        s = t["samples"]
        n = s["length"]
        td = s["timeDeltas"]
        cpud = s["threadCPUDelta"]

        stack_frames_cache = {}

        def stack_frames(si):
            if si in stack_frames_cache:
                return stack_frames_cache[si]
            chain = []
            cur = si
            while cur is not None:
                chain.append(st["frame"][cur])
                cur = st["prefix"][cur]
            stack_frames_cache[si] = chain
            return chain

        def frame_inline_stack(fi):
            li, a = frame_lib[fi], frame_addr[fi]
            if li is None or a is None:
                func = ft["func"][fi]
                return [(strs[fut["name"][func]], None, None)]
            return resolved.get((li, a), [("?", None, None)])

        pairs_cache = {}

        def stack_pairs(si):
            if si in pairs_cache:
                return pairs_cache[si]
            pairs = []
            for fi in stack_frames(si):
                for sym, f, ln in frame_inline_stack(fi):
                    pairs.append((sym, f))
            pairs_cache[si] = pairs
            return pairs

        cat_cpu = Counter()  # category -> cpu ms
        cat_first_last = {}  # category -> [first_ms, last_ms]
        tnow = 0.0
        total_cpu = 0.0
        name = t["name"] if len(t["name"]) < 24 else t["name"][:24]
        is_main = t["isMainThread"]
        for i in range(n):
            tnow += td[i]
            cpu_ms = (cpud[i] or 0) / 1000.0
            total_cpu += cpu_ms
            si = s["stack"][i]
            pairs = stack_pairs(si) if si is not None else []
            cat = categorize(pairs)
            cat_cpu[cat] += cpu_ms
            grand_cat_cpu[cat] += cpu_ms
            if cpu_ms > 0.05:
                fl = cat_first_last.setdefault(cat, [tnow, tnow])
                fl[1] = tnow
            # attribute cpu to the bin of the sample time (cpu accrued since
            # the previous sample; fine at 25 ms bins)
            timeline[int(tnow // BIN)][(("T%02d" % tidx), cat)] += cpu_ms
            if pairs:
                folded[
                    ("main;" if is_main else "worker;")
                    + ";".join(nm for nm, _ in reversed(pairs))
                ] += cpu_ms

        thread_rows.append((tidx, t["tid"], name, is_main, total_cpu, cat_cpu, cat_first_last))

    # ---- outputs ---------------------------------------------------------
    report.write(f"trace: {args.trace}\n")
    report.write("CPU attributed from threadCPUDelta (kernel user+system µs).\n\n")

    report.write("== Grand totals: CPU ms by category (all threads) ==\n")
    tot = sum(grand_cat_cpu.values())
    for cat, ms in grand_cat_cpu.most_common():
        report.write(f"{ms:9.1f} ms  {100*ms/tot:5.1f}%  {cat}\n")
    report.write(f"{tot:9.1f} ms  total CPU\n\n")

    report.write("== Per-thread breakdown ==\n")
    for tidx, tid, name, is_main, total_cpu, cat_cpu, cfl in thread_rows:
        role = "MAIN" if is_main else "work"
        report.write(f"\n-- T{tidx:02d} {role} tid={tid} cpu={total_cpu:.1f} ms --\n")
        for cat, ms in cat_cpu.most_common():
            fl = cfl.get(cat)
            span = f"  [{fl[0]:8.1f} .. {fl[1]:8.1f}]" if fl else ""
            report.write(f"  {ms:8.1f} ms  {cat}{span}\n")

    # per-thread phase edges of interest
    report.write("\n== Phase edges per thread (ms, first..last sample with cpu) ==\n")
    for key in ["encode", "gather memcpy", "gather/assemble", "worker fork+seed"]:
        report.write(f"\n[{key}]\n")
        for tidx, tid, name, is_main, total_cpu, cat_cpu, cfl in thread_rows:
            hits = {c: fl for c, fl in cfl.items() if c == key or c.endswith(f"[{key}]")}
            if key in cfl:
                fl = cfl[key]
                report.write(
                    f"  T{tidx:02d}{'*' if is_main else ' '} {fl[0]:8.1f} .. {fl[1]:8.1f}  ({cat_cpu[key]:7.1f} ms cpu)\n"
                )

    # timeline: aggregate across threads per category
    report.write(f"\n== Timeline ({BIN:.0f} ms bins): CPU ms by category ==\n")
    all_bins = sorted(timeline.keys())
    # collapse categories for readability
    def short_cat(cat):
        if cat.startswith("idle"):
            return "idle"
        if cat.startswith("kernel"):
            return cat
        return cat

    cats_order = [c for c, _ in grand_cat_cpu.most_common(12)]
    report.write("bin_start_ms  " + "  ".join(f"{c[:18]:>18}" for c in cats_order) + "\n")
    for b in all_bins:
        row = Counter()
        for (tname, cat), ms in timeline[b].items():
            row[cat] += ms
        report.write(
            f"{b*BIN:12.0f}  "
            + "  ".join(f"{row.get(c, 0):18.1f}" for c in cats_order)
            + "\n"
        )

    report.close()

    with open(os.path.join(outdir, "collapsed_mt.folded"), "w") as f:
        for stack, w in folded.most_common():
            f.write(f"{stack} {int(w*1000)}\n")

    print(f"outputs in {outdir}/: mt_report.txt collapsed_mt.folded")


if __name__ == "__main__":
    main()
