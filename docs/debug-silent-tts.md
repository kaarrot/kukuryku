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

## Next step — localize the failing op, don't keep bisecting tiers

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
