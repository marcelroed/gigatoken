#!/usr/bin/env python3
"""Analyze a samply profile of the encode_st bench with inline-frame resolution.

Pipeline:
  1. Load the Firefox-profiler JSON (samply --save-only output, unsymbolicated).
  2. Resolve every unique (library, relative address) with `atos -offset -i`
     against the binary/dSYM: full inline stacks + source file:line.
  3. Demangle Rust v0 symbols with rustfilt, strip generic noise.
  4. Cut samples by measured phase: the bench's <trace>.phases.json sidecar
     (benches/common/mod.rs Phases, written when profile.sh sets PHASE_FILE)
     holds epoch-ns phase boundaries; encode pass 0 reports as "encode cold",
     later passes as "encode warm". Falls back to stack markers when the
     sidecar is absent.
  5. Emit, per encode phase:
     - category buckets (walker / cache / BPE merge / ...)
     - project-function rollup: self time with std/intrinsic inline leaves
       folded into their nearest project caller (keyed by symbol, so LTO
       file splits merge), plus inclusive time, plus an inclusive-sorted
       structure view
     - top inline leaves (the finest attribution, as before)
     - source-line annotation of the hottest project functions, with std
       leaves annotating their project call site (hot_lines.txt)
     - collapsed stacks (.folded), phase-prefixed, for flamegraphs
   A report header carries provenance (git rev/dirty, tokenizer, input size,
   per-pass throughput) and the analyzer refuses silently-stale symbolication:
   the trace's recorded binary UUID must match the binary on disk.

Usage:
  python3 analyze.py TRACE.json.gz --bin path/to/encode_st-HASH [-o OUTDIR]
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Symbolication


SENTINEL = 0xFFFFFFFF0  # unmapped offset; atos echoes it back verbatim


def atos_resolve(lib_path, dsym_path, rel_addrs):
    """Resolve relative addresses to inline stacks via atos.

    Returns {addr: [(symbol, file, line), ...]} innermost-first. A sentinel
    (unmapped) address is interleaved between queries; atos echoes it back as
    a bare hex line, giving unambiguous per-address group boundaries.
    """
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
        cmd = [
            "atos", "-o", obj, "-arch", "arm64", "-offset", "-i", "-fullPath",
        ] + query
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
            raise RuntimeError(
                f"atos group mismatch for {lib_path}: {len(groups)} groups, "
                f"{len(chunk)} addrs"
            )
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
    """Load samply's --unstable-presymbolicate sidecar (.syms.json).

    Returns {debug_name: sorted [(rva, size, symbol_name)]} for range lookup.
    """
    sidecar = re.sub(r"\.json(\.gz)?$", ".json.syms.json", trace_path)
    if not os.path.exists(sidecar):
        return {}
    d = json.load(open(sidecar))
    strs = d["string_table"]
    tables = {}
    for lib in d["data"]:
        entries = [
            (e["rva"], e.get("size", 0), strs[e["symbol"]])
            for e in lib["symbol_table"]
        ]
        entries.sort()
        tables[lib["debug_name"]] = entries
    return tables


def sidecar_lookup(table, addr):
    """Find the covering symbol range for addr in a sorted (rva,size,name) list."""
    import bisect

    i = bisect.bisect_right([e[0] for e in table], addr) - 1
    if i >= 0:
        rva, size, name = table[i]
        if addr < rva + max(size, 1) or size == 0:
            return name
    return None


def demangle_all(names):
    """Batch-demangle through rustfilt; returns dict mangled->demangled."""
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
    """Shorten demangled Rust v0 names.

    `<A as B>::f::<T>` -> `A::f`; generic argument lists are dropped, but a
    leading qualified-self `<Type ...>` is replaced by the type's last path
    segments so trait-impl methods keep their type name.
    """
    # Replace a leading <Self as Trait> / <Self> with Self's short name.
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
        # shorten the self type itself (drop its generics, keep last segment)
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


# --------------------------------------------------------------------------
# Bucketing: map an inline stack (innermost-first) to a category.

CATEGORY_RULES = [
    # (predicate on (symbol, file), category) -- first match on innermost
    # frame wins; some rules look at the whole stack.
    ("pack_pretoken_key", None, "cache: key pack"),
    ("ShortPretokenCache", "pretoken_cache", "cache: probe/insert"),
    ("LongPretokenCache", "pretoken_cache", "cache: probe/insert (long)"),
    (None, "pretoken_cache.rs", "cache: probe/insert"),
    ("byte_pair_merge", None, "bpe: merge (miss fallback)"),
    ("encode_pretoken", None, "bpe: merge (miss fallback)"),
    (None, "pretokenize/fast", "pretokenizer walker"),
    (None, "pretokenize/mod.rs", "pretokenizer walker"),
    (None, "pretokenize/unicode", "pretokenizer walker"),
    ("memoized_encode", "tiktoken.rs", "encode driver (spans/output)"),
    (None, "tiktoken.rs", "encode driver (spans/output)"),
    ("memcpy", None, "libsystem memcpy/memset"),
    ("memset", None, "libsystem memcpy/memset"),
    ("_platform_", None, "libsystem memcpy/memset"),
    ("malloc", None, "malloc/free"),
    ("free", None, "malloc/free"),
    ("nanov2", None, "malloc/free"),
    ("read", None, "kernel: read/syscalls"),
    ("madvise", None, "kernel: read/syscalls"),
    ("mach_", None, "kernel: read/syscalls"),
]


def categorize(pairs_leaf_first):
    """pairs_leaf_first: flattened [(sym, file)] over the whole stack, leaf
    first, with inline frames expanded. Walk outward; the first frame that
    matches any rule decides the category, so inlined std helpers attribute
    to their nearest categorized caller."""
    for sym, f in pairs_leaf_first:
        for pat_sym, pat_file, cat in CATEGORY_RULES:
            if pat_sym and pat_sym not in sym:
                continue
            if pat_file and (not f or pat_file not in f):
                continue
            return cat
        if "load_owt_input" in sym:
            return "corpus read phase"
        if "load_hf_bpe" in sym or "load_tokenizer" in sym:
            return "tokenizer load phase"
    return "other"


# Fallback phase attribution via stack markers, used only when the trace has
# no .phases.json sidecar (recorded by benches/common/mod.rs Phases when
# profile.sh sets PHASE_FILE). The sidecar is authoritative: it cuts samples
# by measured wall-clock boundaries and separates cold from warm passes,
# which stack markers cannot.
PHASE_MARKERS = {
    "memoized_encode": "encode cold",
    "load_owt_input": "read corpus",
    "load_hf_bpe": "tokenizer load",
}


def load_phases_sidecar(trace_path):
    """[(phase name, start epoch ns, end epoch ns)], meta dict — or None."""
    sidecar = re.sub(r"\.(json\.gz|json|trace)$", ".phases.json", trace_path)
    if sidecar == trace_path or not os.path.exists(sidecar):
        return None
    d = json.load(open(sidecar))
    phases = [
        (p["name"], p["start_epoch_ns"], p["end_epoch_ns"]) for p in d["phases"]
    ]
    return phases, d.get("meta", {})


def phase_group(name):
    """Collapse per-pass phases: pass 0 is the cold encode, all later passes
    are the warm (cache-hit) encode; setup phases keep their own names."""
    m = re.match(r"encode pass (\d+)", name)
    if m:
        return "encode cold" if m.group(1) == "0" else "encode warm"
    return name


def git_provenance(src_root):
    """'<shortrev>[ (dirty)]' of the profiled worktree, or None."""
    try:
        rev = subprocess.run(
            ["git", "-C", src_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not rev:
            return None
        dirty = subprocess.run(
            ["git", "-C", src_root, "status", "--porcelain", "-uno"],
            capture_output=True, text=True,
        ).stdout.strip()
        return rev + (" (dirty)" if dirty else "")
    except OSError:
        return None


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bin", required=True, help="path to profiled encode_st binary")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--hot-regions", type=int, default=3)
    args = ap.parse_args()

    outdir = args.outdir or os.path.splitext(args.trace)[0].replace(
        ".json", ""
    ) + "_analysis"
    os.makedirs(outdir, exist_ok=True)

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        prof = json.load(f)
    interval_ms = prof["meta"]["interval"]
    threads = [t for t in prof["threads"] if t["samples"]["length"] > 0]
    t = max(threads, key=lambda th: th["samples"]["length"])

    strs = t["stringArray"]
    ft, fut, rt = t["frameTable"], t["funcTable"], t["resourceTable"]
    st = t["stackTable"]
    libs = prof["libs"]

    bin_abs = os.path.abspath(args.bin)
    dsym = f"{bin_abs}.dSYM/Contents/Resources/DWARF/{os.path.basename(bin_abs)}"

    # ---- unique addresses per lib -------------------------------------
    frame_lib = []  # frame index -> lib index (or None)
    frame_addr = ft["address"]
    for fi in range(ft["length"]):
        func = ft["func"][fi]
        res = fut["resource"][func]
        frame_lib.append(rt["lib"][res] if res is not None and res >= 0 else None)

    addrs_by_lib = defaultdict(set)
    for fi in range(ft["length"]):
        li, a = frame_lib[fi], frame_addr[fi]
        if li is not None and a is not None:
            addrs_by_lib[li].add(a)

    def lib_obj_path(li):
        lib = libs[li]
        p = lib.get("debugPath") or lib["path"]
        if not os.path.isabs(p):
            # relative to the recording cwd; try relative to the binary dir
            cand = os.path.normpath(
                os.path.join(os.path.dirname(bin_abs), "..", "..", "..", p)
            )
            p = cand if os.path.exists(cand) else os.path.abspath(p)
        return p

    # Staleness guard: cargo reuses the same target hash across source
    # edits, so target/release/deps/<bin>-<hash> gets overwritten in place
    # and an old trace would silently symbolicate against the wrong binary.
    # samply records each lib's breakpadId (its Mach-O UUID + age); compare
    # it against the current binary before trusting any symbol.
    stale_binary = None
    for lib in libs:
        if os.path.basename(lib["path"]) == os.path.basename(bin_abs):
            # breakpadId = 32-hex Mach-O UUID + an age suffix (usually "0")
            trace_bid = (lib.get("breakpadId") or "").lower()
            try:
                out = subprocess.run(
                    ["dwarfdump", "--uuid", bin_abs], capture_output=True, text=True
                ).stdout
                m = re.search(r"UUID: ([0-9A-Fa-f-]+)", out)
                cur = m.group(1).replace("-", "").lower() if m else ""
            except OSError:
                cur = ""
            if trace_bid and cur and not trace_bid.startswith(cur):
                stale_binary = (
                    f"BINARY MISMATCH: trace was recorded against UUID "
                    f"{trace_bid} but {bin_abs} now has UUID {cur} — the "
                    f"binary was rebuilt since recording; every symbol, line "
                    f"and category below is unreliable. Re-record the trace."
                )
                print(f"warn: {stale_binary}", file=sys.stderr)
            break

    sidecar = load_sidecar_symbols(args.trace)
    resolved = {}  # (li, addr) -> [(sym, file, line)]
    for li, addrs in addrs_by_lib.items():
        libname = libs[li]["name"]
        if os.path.basename(libs[li]["path"]) == os.path.basename(bin_abs):
            # Main binary: exact per-address inline stacks from DWARF.
            try:
                r = atos_resolve(bin_abs, dsym, addrs)
            except Exception as e:
                print(f"warn: atos failed for {bin_abs}: {e}", file=sys.stderr)
                r = {a: [(f"{libname}+{hex(a)}", None, None)] for a in addrs}
        else:
            # System dylibs (dyld shared cache): symbol names from the
            # samply presymbolicate sidecar; no inline/line info needed.
            table = sidecar.get(libname, [])
            r = {}
            for a in addrs:
                name = sidecar_lookup(table, a) if table else None
                r[a] = [(name or f"{libname}+{hex(a)}", None, None)]
        for a, frames in r.items():
            resolved[(li, a)] = frames

    # ---- demangle ------------------------------------------------------
    all_syms = [s for frames in resolved.values() for s, _, _ in frames]
    dem = demangle_all(all_syms)
    for k, frames in resolved.items():
        resolved[k] = [
            (strip_generics(dem.get(s, s) or s), f, ln) for s, f, ln in frames
        ]

    def frame_inline_stack(fi):
        li, a = frame_lib[fi], frame_addr[fi]
        if li is None or a is None:
            func = ft["func"][fi]
            return [(strs[fut["name"][func]], None, None)]
        return resolved.get((li, a), [("?", None, None)])

    # project root = worktree containing the profiled binary
    # (bin lives at <root>/target/release/deps/<bin>)
    src_root = os.path.normpath(os.path.join(os.path.dirname(bin_abs), "../../.."))

    def is_project(sym, f):
        return "gigatoken" in sym or (f is not None and f.startswith(src_root))

    # ---- phase boundaries ------------------------------------------------
    # The bench's .phases.json sidecar records epoch-ns phase boundaries;
    # samply's meta.startTime is epoch ms and cumsum(timeDeltas) is ms since
    # then, so each sample maps to a measured phase (a few ms of clock skew
    # against seconds-long phases). Per-pass phases collapse into
    # "encode cold" (pass 0) and "encode warm" (later passes).
    sidecar_ph = load_phases_sidecar(args.trace)
    bench_meta = {}
    phase_bounds = []  # (start_ms_rel, end_ms_rel, group) in trace time
    phase_wall_s = Counter()  # group -> wall seconds (from the sidecar)
    if sidecar_ph:
        phases, bench_meta = sidecar_ph
        t0_ms = prof["meta"]["startTime"]
        for name, s_ns, e_ns in phases:
            g = phase_group(name)
            phase_bounds.append((s_ns / 1e6 - t0_ms, e_ns / 1e6 - t0_ms, g))
            phase_wall_s[g] += (e_ns - s_ns) / 1e9

    def phase_of(t_ms, pairs_leaf_first):
        if phase_bounds:
            for s, e, g in phase_bounds:
                if s <= t_ms < e:
                    return g
            return "<outside phases>"
        for marker, ph in PHASE_MARKERS.items():
            if any(marker in nm for nm, _ in pairs_leaf_first):
                return ph
        return "other"

    # ---- walk samples ----------------------------------------------------
    samples = t["samples"]
    n = samples["length"]
    weights = samples["weight"] or [1.0] * n
    cpu_deltas = samples.get("threadCPUDelta")
    time_deltas = samples["timeDeltas"]

    # expand each stack index once
    stack_frames_cache = {}

    def stack_frames(si):
        """stack index -> list of frame indices, leaf first."""
        if si in stack_frames_cache:
            return stack_frames_cache[si]
        chain = []
        cur = si
        while cur is not None:
            chain.append(st["frame"][cur])
            cur = st["prefix"][cur]
        stack_frames_cache[si] = chain
        return chain

    group_w = Counter()  # phase group -> sample weight
    group_cpu = Counter()  # phase group -> cpu ms
    cat_w = defaultdict(Counter)  # group -> category -> weight
    # Project rollup: self time by the nearest project frame (std/intrinsic
    # inline leaves fold into their project caller), keyed by symbol alone so
    # LTO file splits (mod.rs vs lib.rs) merge; plus inclusive time (samples
    # anywhere under the frame).
    roll_self = defaultdict(Counter)  # group -> project sym -> weight
    roll_incl = defaultdict(Counter)
    leaf_w = defaultdict(Counter)  # group -> (sym, file, ctx) -> weight
    line_w = defaultdict(Counter)  # group -> (file, line) -> weight
    line_owner = {}
    folded = Counter()
    total_w = 0.0
    tnow = 0.0

    for idx in range(n):
        si = samples["stack"][idx]
        w = weights[idx]
        tnow += time_deltas[idx]
        total_w += w
        if si is None:
            group_w["<no stack>"] += w
            continue
        chain = stack_frames(si)  # leaf first
        # flattened (sym, file, line) list for the full stack, leaf first,
        # with inline frames expanded
        flat = []
        for fi in chain:
            flat.extend(frame_inline_stack(fi))
        pairs_leaf_first = [(sym, f) for sym, f, _ in flat]
        g = phase_of(tnow, pairs_leaf_first)
        group_w[g] += w
        if cpu_deltas and cpu_deltas[idx] is not None:
            group_cpu[g] += cpu_deltas[idx] / 1000.0  # us -> ms
        sym0, f0, _ln0 = flat[0]
        # nearest project frame: rollup identity + leaf-detail context
        proj = None
        for sym_c, f_c, _ in flat:
            if is_project(sym_c, f_c):
                proj = sym_c
                break
        roll_self[g][proj or sym0] += w
        for sym_u in {sym_c for sym_c, f_c, _ in flat if is_project(sym_c, f_c)}:
            roll_incl[g][sym_u] += w
        ctx = "" if (proj is None or proj == sym0) else proj
        leaf_w[g][(sym0, f0, ctx)] += w
        # line attribution at the nearest project frame that has file:line,
        # so std inline leaves annotate their project call site
        for sym_c, f_c, ln_c in flat:
            if f_c and ln_c and f_c.startswith(src_root):
                line_w[g][(f_c, ln_c)] += w
                line_owner[(f_c, ln_c)] = sym_c
                break
        cat_w[g][categorize(pairs_leaf_first)] += w
        folded[g + ";" + ";".join(nm for nm, _ in reversed(pairs_leaf_first))] += w

    total_ms = total_w * interval_ms
    encode_groups = [
        g for g in ("encode cold", "encode warm", "encode")
        if g in group_w
    ] or [max(group_w, key=group_w.get)]

    # ---- outputs ---------------------------------------------------------
    def sec(w):
        return w * interval_ms / 1000.0

    git_rev = git_provenance(src_root)
    report_path = os.path.join(outdir, "top_functions.txt")
    with open(report_path, "w") as f:
        if stale_binary:
            f.write(f"!!! {stale_binary}\n\n")
        f.write(f"trace: {args.trace}\nbinary: {args.bin}\n")
        if git_rev:
            f.write(f"git: {git_rev}\n")
        for k, v in bench_meta.items():
            f.write(f"{k}: {v}\n")
        f.write(
            f"samples: {int(total_w)}  interval: {interval_ms} ms  "
            f"total: {total_ms/1000:.2f} s\n"
        )

        src = "measured sidecar timestamps" if phase_bounds else "stack markers"
        f.write(f"\n== Phases ({src}) ==\n")
        f.write("  wall(s)  samples  cpu(s)  phase\n")
        for g, w in group_w.most_common():
            wall = f"{phase_wall_s[g]:8.2f}" if g in phase_wall_s else f"{sec(w):8.2f}"
            cpu = group_cpu.get(g)
            f.write(
                f"{wall}  {100*w/total_w:6.2f}%  "
                f"{(cpu or 0)/1000:6.2f}  {g}\n"
            )

        for g in encode_groups:
            gw = group_w[g]
            if not gw:
                continue

            def gpct(w):
                return 100.0 * w / gw

            f.write(f"\n---- {g}  ({sec(gw):.2f} s profiled) ----\n")
            f.write("\n== Category buckets (self time) ==\n")
            for cat, w in cat_w[g].most_common():
                f.write(f"{gpct(w):6.2f}%  {sec(w):7.2f}s  {cat}\n")
            f.write(
                "\n== Top project functions "
                "(self = leaf incl. std inlined into it; incl = anywhere in stack) ==\n"
            )
            f.write("  self%    self(s)   incl%  function\n")
            incl = roll_incl[g]
            for sym, w in roll_self[g].most_common(args.top):
                f.write(
                    f"{gpct(w):7.2f}%  {sec(w):7.2f}s  {gpct(incl.get(sym, w)):6.1f}%  {sym}\n"
                )
            f.write(
                "\n== Top project functions by inclusive time "
                "(structure: where the phase's time sits) ==\n"
            )
            for sym, w in incl.most_common(10):
                f.write(f"{gpct(w):6.1f}%  {sym}\n")
            f.write("\n== Top inline leaves (finest attribution) ==\n")
            for (sym, file, ctx), w in leaf_w[g].most_common(15):
                floc = f"  [{os.path.basename(file) if file else '?'}]"
                ctx_s = f"  <- {ctx}" if ctx else ""
                f.write(f"{gpct(w):6.2f}%  {sec(w):7.2f}s  {sym}{floc}{ctx_s}\n")

    with open(os.path.join(outdir, "collapsed.folded"), "w") as f:
        for stack, w in folded.most_common():
            f.write(f"{stack} {int(w)}\n")

    # source-line annotation of the hottest project functions per encode
    # group; line weights sit on the nearest project frame, so std inline
    # leaves (ptr::write etc.) annotate their project call site.
    with open(os.path.join(outdir, "hot_lines.txt"), "w") as f:
        for g in encode_groups:
            gw = group_w[g]
            if not gw:
                continue
            f.write(f"\n######## {g} ########\n")
            hot_syms = []
            for sym, _w in roll_self[g].most_common(100):
                owned = [k for k in line_w[g] if line_owner.get(k) == sym]
                if owned:
                    hot_syms.append((sym, owned))
                if len(hot_syms) >= args.hot_regions:
                    break
            for sym, owned in hot_syms:
                by_file = defaultdict(list)
                for file, ln in owned:
                    by_file[file].append(ln)
                f.write(f"\n===== {sym} =====\n")
                rows = sorted(
                    ((line_w[g][(fl, ln)], fl, ln) for fl, lns in by_file.items() for ln in lns),
                    reverse=True,
                )[:15]
                srcs = {}
                for w, fl, ln in rows:
                    if fl not in srcs:
                        try:
                            srcs[fl] = open(fl).read().split("\n")
                        except OSError:
                            srcs[fl] = None
                    src_lines = srcs[fl]
                    text = (
                        src_lines[ln - 1].strip()
                        if src_lines and ln - 1 < len(src_lines)
                        else ""
                    )
                    f.write(
                        f"{100*w/gw:6.2f}%  {fl.split('/')[-1]}:{ln:<5} {text}\n"
                    )

    print(f"total: {total_ms/1000:.2f}s profiled, {int(total_w)} samples")
    print(f"outputs in {outdir}/: top_functions.txt collapsed.folded hot_lines.txt")
    for g, w in group_w.most_common():
        print(f"  {100*w/total_w:6.2f}%  {g}")


if __name__ == "__main__":
    main()
