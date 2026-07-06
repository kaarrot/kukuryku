#!/usr/bin/env bash
# Micro-bench for the tract conv/im2col run-time work (see docs/tract-support-plan.md).
#
# Runs kokoro-tract on a FIXED single sentence (fixed phoneme count N, so the
# symbolic plan compiles one plan and every run does identical work). Reports:
#   - pure `infer` seconds (synthesize() = stage1.run + regulator + stage2.run),
#     best-of-N to cut WSL/scheduler noise — this is the number Tier-1 should move;
#   - the stage-2 per-op profile (KOKORO_TRACT_PROFILE), so we see the Im2col share.
#
# Usage:  tools/bench_conv.sh [label]     # label tags the saved result file
# Output: tools/bench_out/<label>.txt     (compare before/after with `diff`)
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-run}"
RUNS="${BENCH_RUNS:-4}"
OUT_DIR="tools/bench_out"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$LABEL.txt"

# Fixed, single-sentence input (one chunk => one fixed N, no streaming variance).
SENTENCE="The old lighthouse keeper climbed the winding staircase every single evening at dusk, carrying his heavy brass lantern up the narrow stone steps to make certain the great lamp would burn steadily and brightly through the long and stormy night."

echo "== building (release) =="
cargo build --release --features tract --bin kokoro-tract 2>&1 | tail -1

run_once() { # prints the "[kokoro] done:" line
  cargo run --release --features tract --bin kokoro-tract -- "$SENTENCE" 2>&1 \
    | grep -E "\[kokoro\] (done|\[1/1\])"
}

echo "== timing: best of $RUNS (pure infer secs) ==" | tee "$OUT"
BEST=""
for i in $(seq 1 "$RUNS"); do
  LINE="$(run_once | grep 'done:')"
  SECS="$(sed -n 's/.*infer \([0-9.]*\)s.*/\1/p' <<<"$LINE")"
  RTF="$(sed -n 's/.*RTF \([0-9.]*\).*/\1/p' <<<"$LINE")"
  printf '  run %d: infer %ss  RTF %s\n' "$i" "$SECS" "$RTF" | tee -a "$OUT"
  if [ -z "$BEST" ] || awk "BEGIN{exit !($SECS < $BEST)}"; then BEST="$SECS"; BEST_RTF="$RTF"; fi
done
printf 'BEST: infer %ss  RTF %s\n' "$BEST" "$BEST_RTF" | tee -a "$OUT"

echo "== stage-2 op profile (1 run, instrumented) ==" | tee -a "$OUT"
KOKORO_TRACT_PROFILE=1 cargo run --release --features tract --bin kokoro-tract -- "$SENTENCE" 2>&1 \
  | sed -n '/stage2 profile/,/\[1\/1\]/p' | tee -a "$OUT"

echo
echo "saved -> $OUT"
