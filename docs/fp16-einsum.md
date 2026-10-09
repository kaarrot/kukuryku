# FP16 EinSum: the lever + what remains

Companion to `thermal-perf-audit.md`. The audit identified "fast f16↔f32
conversion" as the top lead and estimated ~12% savings. In practice the
first unaddressed lever was elsewhere: tract's `declutter` pass renames
MatMul-like ops to `EinSum` *before* the FP16 translator looks at them, so
the old filter — matching `MatMul` / `Gemm` / `Conv` substrings — accepted
only a tiny slice of the model.

## What changed

One string added to the FP16 accept list in `src/tract_backend.rs`:

```rust
const ACCEPT_OPS: &[&str] = &["MatMul", "Gemm", "Conv", "EinSum"];
```

Plus a small refactor in `Stage::run` so each input is cast to the
compiled plan's expected per-input dtype (int64 `input_ids` stays int64,
floats are cast per-input), replacing the previous blanket "if fp16 then
cast everything to f16". Without this, enabling FP16 on stage 1 would
have corrupted the embedding lookup.

## Why the previous FP16 commit missed most of the model

Before this change, `apply_fp16` ran on both stages but only a handful
of nodes matched the filter:

| Stage | Nodes matching old filter | Total matmul-like ops (post-declutter) |
|---|---|---|
| stage 1 (encoder) | 3 | ~181 (178 `EinSum` + 3) |
| stage 2 (vocoder) | 74 | ~248 (174 `EinSum` + 74 `Conv`) |

Stage 1 was effectively running in pure f32 even though "FP16 is on"
was printed at startup. Stage 2 was running f16 on its 74 Conv ops only
— enough to generate the 15.6% Cast overhead the audit saw, but leaving
the 174 EinSum ops (most of the vocoder's attention/dense matmuls) in
f32. The signature in the profile that gave this away was **stage 1's
`Cast = 0%`**: a stage with real FP16 translation always shows Cast
nodes bracketing the translated subgraphs. Stage 1 had none, which is
only consistent with "nothing was translated at all."

## Measured impact

Reference sentence from the audit (two sentences, 86 + 155 tokens,
14.80s of audio). Governor on, 3 gold threads on cpu4-6:

| Metric | Before | After |
|---|---|---|
| Done RTF (infer) | 1.320 | **1.110** |
| Wall RTF | 1.090 | **0.916** |
| First audio latency | 7.68s | 6.50s |
| Sentence 1 RTF | 1.301 | 1.091 |
| Sentence 2 RTF | 1.332 | 1.121 |
| Inter-sentence gap | 2.95s | 1.58s |
| Stage 1 nodes translated | 3 | 181 |
| Stage 2 nodes translated | 74 | 247 |

Wall RTF below 1.0 on a single long sentence has not been achieved on
this handset before — paragraphs were already there via lookahead
overlap, long single sentences never were.

The gap collapse (2.95 → 1.58s) is the signal that stage 2 got
meaningfully faster on its own: stage-1 overlap was already saturated,
so a smaller stage 2 is the only thing that moves this number.

## How to verify

- `KOKORO_TRACT_PROFILE=1 target/release/ryk -v "<sentence>"` prints
  per-op time share. With this change, stage 1 should show Cast nodes
  and reduced OptMatMul time; stage 2's OptMatMul and Cast shares both
  shift.
- At build time, each stage logs `fp16 <stem>: N of M nodes translated`.
  Non-trivial N on both stages confirms the filter is matching.

## Outstanding leads (ranked, after this change)

Estimates are speculative unless noted. The audit's #1 ("fast Cast
kernel") is still on the table but is no longer the lowest-hanging fruit
by a wide margin now that EinSum is translated.

### 1. Re-profile and recompute the top cost centres

**Zero code, must do first.** The 59.5% OptMatMul / 15.6% Cast / 6.4%
Snake shares in the audit were measured with only 74 of 248 stage-2
matmul-like ops in f16. With all 247 translated the Cast count very
likely drops (adjacent EinSum chains run in f16 end-to-end with no
Cast between them) and the remaining cost centres reorder. Everything
below should be re-ranked after one `KOKORO_TRACT_PROFILE=1` run on the
new binary.

### 2. SIMD + threaded f16↔f32 Cast kernel in tract (audit #1)

Whatever Cast remains after re-profiling. tract's current Cast is
one-element-at-a-time single-threaded; the `half` crate ships
SIMD slice conversions that are bit-identical. **Invasive** — requires
a patch to `tract-linalg` or a crate-level override. Estimated ceiling
depends on #1's result.

### 3. Lift the Samsung clock cap (audit #2)

Gold cores held at 2.016 GHz vs. 2.42 GHz hardware max; ~10–17% headroom.
With root: raise `scaling_max_freq`. Without root: Game Launcher /
"Enhanced processing" has been reported to override the cap on some
firmware; **never measured on this handset**. Trivial to test:
`cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq` under each
mode.

### 4. Make 4-gold the Android default (`KOKORO_TRACT_CPUSET=4-7`)

Measured on this S10: `KOKORO_TRACT_CPUSET=4-7 KOKORO_TRACT_THREADS=4`
drives RTF ~1.0 vs. ~1.11 with the current 3-gold default. Pre-FP16
coverage this override only bought ~5% ("the prime only buys ~5% — so
this is a thermal win, not a realtime one", commit `4f9b955`); with the
EinSum fix making GEMM throughput matter more, extra-core returns
roughly doubled.

Not landed yet because the default-flip reverses a deliberate
thermal-protective policy and we have not re-validated two things:

- **Sustained thermal state** with 4 golds + full FP16 coverage. Short
  runs are safe (cores stay 44–65°C) but a 10-minute paragraph session
  has not been measured.
- **`ryk --serve` daemon UI impact.** Prime core is Android EAS's
  preferred foreground-UI target; bursty inference on it may stutter
  the active app.

To land as default: flip `auto_android_cpuset` to include the prime
(top-two capacity groups when the top is a singleton), update the two
tests that assert `vec![4,5,6]`, and move the README's "4-gold
override" section into the normal default description.

`KOKORO_TRACT_FP16=1` is also sometimes passed alongside this but is a
no-op on aarch64 with `asimdhp` — FP16 is already auto-enabled there.
`KOKORO_TRACT_THREADS=4` similarly follows from `CPUSET=4-7`
(`resolve_threads` derives count from pin-set size). The only real
knob here is the cpuset.

### 5. Sentence splitting at clause boundaries

Long single sentences stay above 1.0 wall RTF when the lookahead has
nothing to overlap with. Splitting the sentence at commas/semicolons
produces two chunks and the stage-1/stage-2 overlap kicks in. Needs a
listening check for prosody seams at the split point.

### 6. Alignment matrix sparse representation

`src/tract_backend.rs:1084` builds a dense `[N, total_frames]` f32
matrix that is almost all zeros (exactly one `1.0` per column). The
downstream op treats it as a dense matmul input. For a 155-token
sentence at ~465 frames that is ~288 KB of near-zero multiplies per
inference. Replace with a repeat/gather. Only worth doing if the
re-profile shows the alignment matmul is a measurable fraction of
stage 2's `OptMatMul` time.

### 7. Snake in f16

6.4% of stage 2 pre-EinSum, likely similar after. Currently DENY-listed
because f16 Snake was observed to silence output in earlier experiments.
A numerically-safer formulation — e.g. compute the exponent in f32 and
the pointwise multiply in f16 — might be worth revisiting if the Cast
elimination from EinSum coverage doesn't already shrink the Snake
share by removing adjacent Casts.

### 8. tract version bump

Upstream tract ships NEON GEMM/int8 improvements regularly. One
`Cargo.toml` edit + rebuild tests the delta. Risk: regression.

### 9. Stage-boundary zero copy (tested, dead end for now)

Changing `f32_tensor` to preserve dtype and skipping the matching cast
at stage-2 input was tried and measured flat RTF (1.320 → 1.321). The
boundary casts were not where the time went. Keep in mind if a future
profile shows these casts growing.

### 10. Scalar-op FP16 neighborhood translation (tested, dead end)

A fixpoint pass that promoted `OptMulByScalar` etc. to f16 when it
reduced boundary Casts was tried and gave zero measurable RTF
improvement for ~75 lines of code. Reverted.

### Low-value / dismissed

- PGO: 1–4%, works in Termux.
- Fat LTO: 0–2% over thin LTO.
- int8: slower, tract has no dotprod int8 kernels.
- BOLT: not available in Termux.
- mimalloc: already enabled (~8% from the original integration).

## Immediate next step

Run `KOKORO_TRACT_PROFILE=1` on the new binary and update #1 above
with the new per-op breakdown. Everything below #1 should be re-ranked
against that data before more code is written.
