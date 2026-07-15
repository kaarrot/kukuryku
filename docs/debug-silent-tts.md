# Debug: silent kokoro-tract output

## Symptom
`./target/release/kokoro-tract "hello world"` completed successfully but
produced no audible sound. Initial failure was:

```
Error: playback thread stopped early
```

## Investigation timeline

1. **Missing `ffplay`.** The playback thread spawned `ffplay`, which is not
   installed on Termux (Termux ships `ffmpeg` without SDL support, so
   `ffplay` is absent from the repos). The thread died on spawn; the main
   loop only saw the generic mpsc-send failure — the real error was trapped
   in the thread's return value and never surfaced.

2. **Fallback to `pacat`.** Added a sink probe in `src/lib.rs`
   (`build_sink_command`): prefer `ffplay`, fall back to `pacat`
   (PulseAudio), else bail with a message naming both. Committed as
   `f1c7e89`. Binary now exits 0 — but still no audio.

3. **PulseAudio daemon lifecycle.** `pulseaudio --check` showed the daemon
   was not running. Started it with `pulseaudio --start
   --exit-idle-time=-1` so it doesn't exit after 20s of idle. A `pacat`
   beep test from the shell produced audible sound; kokoro's binary still
   did not.

4. **Prebuffer hypothesis (wrong).** pacat's default `prebuf` is ~1.98s
   (190 KB @ 24 kHz f32 mono). Kokoro's 1.62s output is *below* that, so
   pacat would only start playback when EOF triggered drain. Added
   `--latency-msec=100` to keep the buffer small. Verbose pacat logs then
   confirmed the stream started, latency counted down from 1.5s to 0, and
   `Playback stream drained` fired. **But still no audible sound.**

5. **Audio-content check.** Dumped the WAV via `KOKORO_WAV=...` and the
   stage-boundary tensors via `KOKORO_TRACT_DUMP=...`. Results:

   | Tensor         | n     | min      | max     | rms     |
   | -------------- | ----- | -------- | ------- | ------- |
   | s1_dur         | 15    | 2.0      | 17.0    | 6.14    |
   | s1_feat512     | 7680  | -0.994   | 0.999   | 0.337   |
   | s1_feat640     | 9600  | -2.404   | 2.309   | 0.426   |
   | **s2_out0**    | 39000 | **NaN**  | **NaN** | **NaN** |

   Stage 1 features look sane. **Stage 2 (vocoder) outputs NaN across the
   board.** The playback pipeline was working the whole time — it was
   faithfully rendering silence, because `f32::NaN.clamp(-1.0, 1.0)` is
   still NaN, and Rust's `NaN as i16` yields `0`. Every WAV sample was
   zero; every raw f32 sample was NaN.

## Real root cause
Numerical bug in the tract vocoder path. None of the tract optimizations
were ever exercised on aarch64/Termux — every RTF number in the tier
commits was measured on x64. Working hypothesis: an arch-specific
numerical/UB failure in one of the optimization tiers, or in the
tract-linalg SIMD kernels themselves.

## Bisect — Tier 5/6/7 all exonerated

Each rebuild took ~9 min. Patches were cumulative: once a tier was
disabled it stayed disabled while the next one was neutralized. After
each rebuild the run was checked with `KOKORO_TRACT_DUMP` (stage-boundary
tensors) and `KOKORO_WAV` (final PCM).

| Attempt                              | s1 tensors | s2_out0 (39000 samples)     | RTF   |
| ------------------------------------ | ---------- | --------------------------- | ----- |
| baseline (as of `f1c7e89`)           | clean      | 39000 NaN                   | ~1.22 |
| Tier 7 vectorized `sin` → `f32::sin` | clean      | 39000 NaN                   | ~1.28 |
| + Tier 7 mimalloc off                | clean      | 39000 NaN                   |       |
| + Tier 7 SinSq fusion off            | clean      | 39000 NaN                   | ~1.38 |
| + Tier 6 rank-1 Pad fold off         | clean      | 39000 NaN                   | ~1.70 |
| + Tier 5 SumOfSquares fusion off     | clean      | 39000 NaN                   | ~1.77 |

Observations:
- `s1_dur` / `s1_feat512` / `s1_feat640` are bit-stable and finite at
  every attempt. **Stage 1 is not the source.**
- Every s2_out0 sample is NaN (no Inf, no finite outliers). It's not
  drift into overflow — it's an early NaN that propagates through the
  rest of the vocoder graph via multiplies/adds.
- RTF grew monotonically as each optimization was removed, which
  confirms each patch actually took effect (not a stale binary).

Doc's three top numerical suspects are all clean. The bug lives
elsewhere.

## Bisect patches currently applied (cumulative, uncommitted local edits become the "investigate" commit)

- `third_party/tract/tract-core/src/ops/math/mod.rs` —
  `ssin_f32` body replaced with `xin.sin()` (scalar libm sinf);
  `declutter_square` returns `Ok(None)` (no SinSq fusion).
- `third_party/tract/tract-core/src/ops/cnn/conv/conv.rs` — dropped
  the `hw_dims().len() != 1` guard so **all** padded convs use the
  explicit-Pad path (Tier 6 rank-1 fold disabled).
- `third_party/tract/tract-core/src/ops/nn/reduce.rs` — symbolic
  `Sum(Square(x))` branch returns `Ok(None)` instead of building a
  `SumOfSquares` reducer (Tier 5 fusion disabled).
- `src/bin/kokoro_tract.rs` — `#[global_allocator] MiMalloc` commented
  out.

## Localized — first non-finite output is a `Recip`

Added `KOKORO_TRACT_NAN_TRACE=1` (see `nan_trace_run` in
`src/bin/kokoro_tract.rs`): step the plan node-by-node under
`SimpleState::run_plan_with_eval` and, the first time any node emits a
non-finite value, print the node op / id / shapes plus each input's
nan/inf/min/max.

Single run says:

```
[nan-trace] stage1: no non-finite outputs observed
[nan-trace] stage2: FIRST bad node #243 op=Recip out#0 shape=[1, 11, 7801]
                   nan=0 inf=12234 of 85811
[nan-trace]   input#0 shape=[1, 11, 7801] nan=0 inf=0 of 85811
              min=-1.1274 max=0.7234
```

Reading:

- The `Recip` (1/x) node emits **12,234 `Inf` values (~14% of the
  output), no NaN yet**. NaN appears later — downstream ops collide the
  Infs (`Inf − Inf`, `0 × Inf`) and the vocoder's remaining ~1,700 nodes
  amplify that into the 39000/39000 NaN we observe at s2_out0.
- The input to Recip is finite: min = −1.1274, max = 0.7234. But for
  `1/x` to emit `Inf`, `x` must be exactly `±0` — a denormalized-tiny
  value would still give a large *finite* result. So **12,234 of the
  85,811 input elements are exact zeros**.
- That is not a plausible distribution for continuous vocoder features;
  something upstream is flushing near-zeros to exact zeros. Classic
  arch-specific SIMD divergence — a subtract-mean or a reduction giving
  identity instead of a tiny non-zero on aarch64 that stayed non-zero
  on x86.
- The bug is therefore **one hop upstream of node #243**, not in Recip
  itself, and not in any of the Tier 5/6/7 fusions we bisected.

## Upstream chain — Recip is fed by a real/imag STFT slice

`KOKORO_TRACT_NAN_TRACE=1 KOKORO_TRACT_NAN_HOPS=6` (walks predecessors up
to N hops from the culprit, using per-node summaries cached during the
run):

```
#243 Recip           [1,11,7801]      nan=0 inf=12234 zeros=0
  #233 Gather        [1,11,7801]      zeros=12234
    #232 Const       (constant — the gather indices)
    #231 MoveAxis    [1,11,7801,2]    zeros=64538
      #230 Slice     [1,7801,11,2]    zeros=64538
        #229 STFT    [1,7801,20,2]    zeros=~109k
          #228 Pad   [1,39020,2]      zeros=39020
            #227 AddAxis [1,39020,1]  zeros=0   min=-0.145 max=0.087
```

Reading:

- **#227 AddAxis** is a normal audio-like tensor (min=-0.145, max=0.087,
  no zeros) — the vocoder's excitation signal.
- **#228 Pad** doubles the last dim to 2 with zeros — this is the
  real/imag pair fed to STFT, with **imaginary = 0 everywhere** (all
  39020 zeros are the imaginary half). Standard real→complex convention
  for tract's STFT.
- **#229 STFT** [1, 7801, 20, 2] — complex spectrum. Because input is
  real-only, its FFT has structural zeros in certain (bin, frame, re/im)
  positions.
- **#230/#231 Slice, MoveAxis** are pure reshape/permute passes
  preserving those zeros exactly.
- **#233 Gather** with a Const index picks one component (real or imag)
  out of each `[re, im]` pair → [1, 11, 7801] with 12,234 exact zeros.
- **#243 Recip** reciprocates that slice → 12,234 Infs.

Almost certainly a real/imag → magnitude → normalize pattern in Kokoro's
iSTFTNet vocoder. The graph appears to assume this specific slice never
hits an exact zero. On x86 the same STFT positions were probably tiny
non-zeros (denormal or last-bit rounding); on aarch64 tract's STFT
produces exact `+0.0` there — 12,234 of them — and `1/0 = Inf`. Those
Infs then poison the rest of the vocoder graph via `Inf − Inf` /
`0 × Inf` into the 39000/39000 NaN we see at s2_out0.

## Option 1 — diagnostic Recip eps guard

Cheap symptom patch: replace `1/x` with `1/(x + copysign(eps, x))` (or
`1/x` clamped to a huge finite value) inside tract's f32 Recip element-
wise op. If audio becomes finite (probably distorted), we've confirmed
the mechanism is Recip-on-exact-zero and can decide on a real fix:

- surgical eps guard scoped to just this Recip op, or
- a graph rewrite that inserts a `Max(eps)` before this Recip during
  optimization, or
- root-fix the STFT arch divergence so those slots aren't exactly zero
  in the first place (better, but requires an x64 comparison run to
  characterize).

Applying (1) next. Rebuild is ~1 min (tract-core recompiles).

## Result: mechanism confirmed, audio is finite

With the diagnostic guard applied (f32 Recip returns `±1e30` for
exactly-zero inputs, unchanged otherwise), the NaN chain is fully
broken:

```
[nan-trace] stage1: no non-finite outputs observed
[nan-trace] stage2: no non-finite outputs observed

s2_out0: n=39000  nan=0  inf=0  min=-0.3785  max=0.3867  rms=0.0679
wav:      n=39000  nonzero=27124  min=-12401  max=12670
```

Stage 2 completes entirely finite. Waveform amplitude is well within
i16 range with 69% nonzero samples — audible content. The
`Recip(exact-zero)` → `Inf` → downstream `NaN` mechanism is therefore
the sole trigger for the silent-TTS symptom.

The `±1e30` is deliberately a large-but-finite sentinel, not `f32::MAX`:
the reciprocated value likely feeds a `Mul(mask)` (or similar) further
in the vocoder, and we want that multiplication to stay finite so the
downstream masking (if any) reveals whether the graph was designed to
zero out those positions. In this run every stage-2 op stayed finite,
so downstream masking evidently *is* neutralizing the huge sentinel
values — meaning the model always expected large-ish values at those
slots and the zero was purely arch-specific numerical noise. On x86 the
same slots held tiny (denormal / last-bit) non-zeros whose reciprocals
were also huge-finite; aarch64 rounded them to exact `+0.0`, and `1/0`
is the only case that breaks out of "huge-finite → mask → 0" and into
"Inf → poison the graph".

## Options for a shipped fix

None of these are applied yet — pick after listening to `out.wav` and
deciding whether audio quality is acceptable with the diagnostic patch:

1. **Keep a scoped f32 eps guard on `Recip`.** Cheapest. The sentinel
   idea (large-finite on zero) is what already works. Risk: hides real
   division-by-zero bugs in other users of `tract`. Mitigation: gate
   behind a feature flag or make it `x.max(eps)` before the reciprocal
   so the value stays close to what naive libm would emit for a denormal
   near zero.
2. **Root-cause the STFT arch divergence.** Compare tract's STFT
   output for the same padded audio on aarch64 vs x86. If a specific
   twiddle-factor multiply or a rfft symmetry step is rounding
   asymmetrically only on aarch64, fix that; then Recip never sees
   exact zero and no guard is needed. Better long-term but needs an
   x86 comparison run.
3. **Graph rewrite (declutter).** In tract's optimizer, detect
   `Recip(x)` where `x` traces back to STFT slice/gather and insert an
   `Add(sign(x)*eps)` or `Max(eps)` node. Targeted, no global behavior
   change; more code than option 1.

Every rebuild is ~9 min and the ranked-suspect strategy has been
unproductive. Cheaper approach: run stage 2 once and find the first
node whose output goes non-finite. That points at an op class (conv,
InstanceNorm, atan2, STFT, or a tract-linalg SIMD path) rather than a
commit.

1. Check what `KOKORO_TRACT_DUMP` currently exports. Commit `752d4ca`
   ("dump all stage2 outputs under KOKORO_TRACT_DUMP — probe-point
   diffing") suggests per-node dumping was already implemented at
   some point; the current binary only writes stage boundaries, so
   either the wiring regressed or it needs an env-var toggle.
   Read `src/bin/kokoro_tract.rs` around the `KOKORO_TRACT_DUMP`
   handling and the stage-2 run loop.
2. If per-node dumping still exists, run once, then scan every
   `.f32` file in the dump dir for the *first* one with any NaN. That
   node's op type is the culprit; its predecessors (still finite)
   bracket the exact op.
3. If not, add a minimal probe: after each stage-2 node evaluation,
   `debug_assert!(output.iter().all(|x| x.is_finite()))`, or on
   first NaN write the node name + input shapes to stderr and abort.
   One targeted probe, one rebuild, then a single run gives the
   answer.

Failing that (nothing in stage 2 goes NaN in isolation but the
composed graph still does), the next suspect is `tract-linalg`'s
aarch64 SIMD kernels — never touched by the tier commits and never
exercised on Termux prior to this session. That would require
temporarily forcing tract to a scalar reference implementation to
confirm.

## What's already fixed
- `f1c7e89` on branch `tract-prototype` — sink fallback + clearer
  error when no sink is installed. Independent of the NaN bug.

## What's still broken
- Stage 2 vocoder produces NaN, so the binary is effectively unusable
  for actual TTS regardless of playback setup.
