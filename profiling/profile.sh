#!/usr/bin/env bash
# Record a CPU profile of a single-threaded cold encode_st run.
#
# Usage: ./profile.sh [ENCODE_MB] [samply|counters|both] [label]
#   ENCODE_MB  MB of ~/data/owt_train.txt to encode (default 10000 = full cold pass)
#   mode       samply   -> Firefox-profiler trace w/ presymbolicate sidecar (default)
#              counters -> xctrace 'CPU Counters' (CPU Bottlenecks guided mode, PMU)
#              both     -> one after the other (never concurrently)
#   label      trace filename suffix (default "<MB>mb")
#
# The bench builds with [profile.bench] = release codegen (fat LTO) + full
# debug info (see Cargo.toml), so profiles measure the real binary and
# inline frames resolve. Never run two profiling/bench processes at once.
#
# Env: PROFILE_OUT overrides the trace output directory.
#      ENCODE_TOKENIZER selects the tokenizer: a local tokenizer.json path or
#      a HuggingFace repo id (e.g. Qwen/Qwen2-1.5B-Instruct; fetched into the
#      standard HF cache on first use, served from it afterwards).
set -euo pipefail

MB="${1:-10000}"
MODE="${2:-samply}"
LABEL="${3:-${MB}mb}"

# Repo root: this script lives in <root>/profiling/ (or set PROFILE_ROOT).
ROOT="${PROFILE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${PROFILE_OUT:-$ROOT/profiling/traces}"
mkdir -p "$OUT"
cd "$ROOT"

echo "== building bench (release codegen + debuginfo) =="
BUILD_LOG=$(cargo bench --no-run --bench encode_st 2>&1) || {
    echo "$BUILD_LOG"; exit 1; }
BIN=$(echo "$BUILD_LOG" | sed -n 's|.*(\(target/release/deps/encode_st-[0-9a-f]*\)).*|\1|p' | head -1)
[ -n "$BIN" ] || { echo "could not locate bench binary"; echo "$BUILD_LOG"; exit 1; }
echo "binary: $ROOT/$BIN"

# A repo-id ENCODE_TOKENIZER downloads on first use; do that (plus a 1 MB
# encode) outside the recording so the trace only measures encoding.
if [[ -n "${ENCODE_TOKENIZER:-}" ]]; then
    echo "== prefetching tokenizer ($ENCODE_TOKENIZER) =="
    ENCODE_MB=1 ENCODE_TOKENIZER="$ENCODE_TOKENIZER" "./$BIN" >/dev/null
fi

if [[ "$MODE" == "samply" || "$MODE" == "both" ]]; then
    TRACE="$OUT/samply_$LABEL.json.gz"
    echo "== samply record (4 kHz, main thread) -> $TRACE =="
    # samply must launch the binary directly (it cannot inject into
    # signed system binaries like `env`), so ENCODE_MB is set here.
    ENCODE_MB="$MB" samply record --save-only -r 4000 --main-thread-only \
        --unstable-presymbolicate -o "$TRACE" "./$BIN"
    # remember which binary produced the trace, for analyze.sh
    echo "$ROOT/$BIN" > "${TRACE%.json.gz}.bin"
    echo "trace: $TRACE"
fi

if [[ "$MODE" == "counters" || "$MODE" == "both" ]]; then
    TRACE="$OUT/counters_$LABEL.trace"
    echo "== xctrace CPU Counters (CPU Bottlenecks mode) -> $TRACE =="
    rm -rf "$TRACE"
    xcrun xctrace record --template 'CPU Counters' --output "$TRACE" \
        --env ENCODE_MB="$MB" ${ENCODE_TOKENIZER:+--env ENCODE_TOKENIZER="$ENCODE_TOKENIZER"} \
        --launch -- "./$BIN"
    echo "trace: $TRACE"
fi
