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

### 2. SIMD f16↔f32 Cast kernel in tract (audit #1) — done

Landed in `third_party/tract/tract-data/src/tensor.rs` (`cast_to_dt`):
F16→F32 and F32→F16 go through `half`'s slice conversion
(`convert_to_f32_slice` / `convert_from_f32_slice`, 4-wide NEON) instead
of `natural_cast`, which ran a runtime fp16 check plus a scalar convert
per element. Measured on the A76: 2.51 → 0.52 ns/element. Stage 2 makes
353 Casts per run on `[1,128,~45k]`-class tensors, all on the main
thread, so this was 16–17% of stage-2 time.

- Stage-2 Cast per sentence: 1.17 s → 0.41–0.45 s (share 16.9% → 6.6–7.2%).
- Output WAV bit-identical (same md5): exact widen, round-to-nearest-even
  narrow, same as the scalar path.
- Still single-threaded. What is left (6.6%) is mostly memory traffic.

### 3. Lift the Samsung clock cap (audit #2)

Gold cores held at 2.016 GHz vs. 2.42 GHz hardware max; ~10–17% headroom.
With root: raise `scaling_max_freq`. Without root: Game Launcher /
"Enhanced processing" has been reported to override the cap on some
firmware; **never measured on this handset**. Trivial to test:
`cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq` under each
mode.

### 4. Make 4-gold the Android default (`KOKORO_TRACT_CPUSET=4-7`) — rejected

**Measured against battery and heat, not landing.** On the 3-sentence
paragraph, `CPUSET=4-7` used 61.8 s of process CPU vs 57–58 s for the
3-gold pool, peaked at 85 °C, and tripped the (then 78 °C) thermal hold
at sentence 2, which put it back on the 2-thread pool. The prime core
runs at the same 2016 MHz cap as the golds, so a fourth thread adds
contention and core power for ~4% speed. See `battery-and-heat.md`.

Original note: `KOKORO_TRACT_CPUSET=4-7 KOKORO_TRACT_THREADS=4`
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

### 11. ConvTranspose (`Deconv`) in f16 (tested, not landed)

tract declutters ConvTranspose to `Deconv` (lowercase c), so the
`Conv` accept entry misses the vocoder upsamplers (`generator/ups.*`)
and the F0/N `pool` ops. `DeconvSum` already has an aarch64 fp16 path,
and adding `"Deconv"` translates 6 more nodes (247 → 253). Measured
with stage 1 held identical so the outputs align sample for sample:

| Stage 2 | Process CPU (2 runs) | Spectral conv. vs f32 stage 2 | LSD vs f32 |
|---|---|---|---|
| FP16 today | 56.6 / 57.0 s | 0.0931 | 4.64 dB |
| FP16 + Deconv | 56.3 / 56.7 s | 0.0958 | 4.97 dB |

0.5% less CPU, inside run-to-run noise, for an output that differs from
today's FP16 by 0.064 spectral convergence (3.8 dB LSD) and sits further
from f32. Left in f32; `fp16_filter_is_gemm_only` pins that.

Waveform SNR is useless for this kind of check: today's FP16 stage 2
scores 1.8 dB SNR / 0.67 correlation against f32 stage 2 with identical
durations, because the vocoder regenerates phase. And across runs whose
stage 1 differs at all, durations shift by a frame here and there, so
compare stage-1 tensors (`KOKORO_TRACT_DUMP`) or spectra, not samples.

### 12. `Pow(x, 3)` → `x * Square(x)` (landed)

ALBERT's `gelu_new` computes `x³` on every layer; tract's
`declutter_pow` only rewrote exponents 2 and 0.5, so it ran generic
`powf`: 12 calls, ~7.8% of stage 1 (0.27 s for 157 tokens, on the
little cores and on the first-audio path). `declutter_pow` now rewrites
exponent 3 to `x * Square(x)` when that does not change the output
shape (`third_party/tract/tract-core/src/ops/math/mod.rs`).

Checked on stage-1 dumps: in pure f32 the durations are identical and
the 640-ch features agree to 111.7 dB SNR (ulp level). Under FP16 the
change is no bigger than FP16's own noise: vs the f32 run, 2 of 88
durations move by one frame (3 before) and the 640-ch features stay at
34.1 dB (34.0 before).

### 13. Cheaper `sin²` in `Snake` / `SinSq` (landed)

`Snake` (48 per run, ~7% of stage 2) is compute-bound, not bandwidth-bound:
about 3.3 GB/s on `[1,128,29281]` while the `Sub`/`Mul`/`Add` passes beside
it run at 11–15 GB/s. It evaluated `ssin_f32` (cephes, octant select and two
polynomials) and squared it. `sin_sq_f32` uses that sin² has period π:
reduce to w in [-π/2, π/2], then a degree-8 polynomial in w², evaluated with
Estrin's scheme because Horner's FMA chain was latency-bound once vectorized
(3.77 → 3.14 ns/element for the same coefficients). The rounding uses the
1.5·2²³ trick so it vectorizes without libm `roundf` on x86 builds.
`Sin` itself still uses `ssin_f32`.

- Single thread, A76, `[1,128,29281]`: 5.2 → 3.1 ns/element.
- In the stage-2 profile (alternating A/B, one 107-token sentence): `Snake`
  0.292 s → 0.187–0.190 s (7.0% → ~4.6% of stage 2). Wall RTF moved within run
  noise (±0.03), so treat it as ~2.4% of stage 2, not a visible RTF step.
- Error against f64: 2.8e-7 absolute for |r| < 12000 (`ssin_f32` was 1.1e-7;
  f16 noise downstream is ~1e-3). Past ~1e5 an f32 phase has no resolution
  left, and `ssin_f32` is garbage there too, so there is no clamp: one cost
  1.1 of 4.2 ns/element and bought nothing the old kernel had.
- Output: durations unchanged (same sample count), 22% of samples
  bit-identical, spectral convergence 0.012 and LSD 1.65 dB against the old
  FP16 output (the rejected `Deconv` change in #11 moved SC by 0.064).
  Perturbing the sine at 1e-7 flips f16 roundings in the next conv, which the
  vocoder then spreads, so waveform SNR (35 dB) is the wrong yardstick; listen
  to it before trusting it.

### 14. Fused AdaIN: instance norm + affine in one op (landed)

Stage 2 has 65 `Reduce<Sum>` nodes; 62 of them are the mean of an AdaIN instance norm
that lowers to ten tract nodes and six full passes over the activation, all
memory-bound at 11-15 GB/s:
`Reduce<Sum>`, `Mul(1/F)`, `Sub`, `Reduce<SumOfSquares>`, `Mul(1/F)`, `Add(eps)`,
`Rsqrt`, `Mul`, `Mul(1+gamma)`, `Add(beta)`. Together that was ~0.39 s of a
~4.1 s stage-2 profile. The `AdaIN` op (`tract-core/src/ops/math/mod.rs`, hooked
in `Add`'s declutter after Snake) does it per `(n, c)` row: read for the mean, read
for the variance, write once. A row is at most 117 KB, so the second read and the
write hit L2.

- Matches `Add(Mul(g, Mul(Sub(x, Mul(Sum(x), d)), Rsqrt(Mul(SumSq, d) + eps))), beta)`
  in either operand order, with single consumers throughout, `xc` read only by the
  variance and the multiply, f32, a leading 1 and a known C, and nothing removed
  being a model output. 62 of the 65 match; the other three (e.g. `F0.0/norm1`, whose
  centred value has a second reader) have a different shape and stay unfused. The `1/F` factors are `Recip` nodes under the symbolic plan, so
  they are inputs to the op and multiplied where the graph multiplied them, not
  assumed to be 1/F.
- **Numerics: bit-identical to the unfused graph.** The first version summed in f32
  and the audio moved by 0.0416 spectral convergence / 2.8 dB LSD (Snake alone: 0.012):
  tract's `SumOfSquares` accumulates in f64 and `Sum` is the tract-linalg `sum_f32`
  kernel, so every row's mean and variance came out a few ulps different, and f16
  spread that. Now the op calls the same `sum_f32` and the same f64 lane scheme (the
  centred value formed on the fly), and applies `((x - mean) * rs) * g + beta` in the
  graph's order. The WAV md5 equals the Snake-commit build's on four inputs (31,800 to
  674,400 samples, 1 to 77 words). The f64 variance costs nothing measurable: `AdaIN`
  is 0.121 s with it and 0.124 s with f32.
- Stage 2, three alternating rounds, one 107-token sentence, cool phone: 4.09 s ->
  3.82 s (-6.6%). `AdaIN` takes 0.12 s in place of ~0.39 s of separate ops. Infer RTF
  1.041 -> 1.006, wall RTF 1.078 -> 1.044. Stage-2 node count 1892 -> 1334.
- Standalone kernel (A76, one thread, `[1,128,29281]`): 9.3 ms for the six passes vs
  5.6 ms fused.
- `AdaIN` is in the FP16 deny list; it stays f32, so a Cast still sits on each side.

Not done, and where the rest is: f16 input (reading the conv output directly drops the
Cast in, about half of the 7% Cast share) and fusing the Snake / LeakyRelu that consumes
the result and writing f16 (drops the Cast out and a pass). Those change the data path,
not just the order of the same operations, so they need their own parity story.

### Low-value / dismissed

- PGO: 1–4%, works in Termux.
- Fat LTO: 0–2% over thin LTO.
- int8: slower, tract has no dotprod int8 kernels.
- BOLT: not available in Termux.
- mimalloc: already enabled (~8% from the original integration).

## Where stage 2 goes now (2026-10-16)

`KOKORO_TRACT_PROFILE=1`, 155-token sentence, after #2 and #12. Stage 2:
OptMatMul 65.2%, Cast 7.2%, Snake 7.2%, OptMulByScalar 3.2%, Resize 2.6%,
OptSubByScalar 2.3%, Scan 2.1%. The f16 GEMMs are two thirds of the
vocoder and nothing else is above ~7%. Stage 1: OptMatMul 68.7%, Scan
(LSTM) 16.4%, Softmax 4.3%, Cast 2.9%; `Pow` (0.27 s, 7.8%) is now
`Square` (0.01 s) plus 12 cheap multiplies.
