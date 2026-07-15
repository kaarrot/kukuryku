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
Numerical bug in the tract vocoder path. Likely suspects, in order of
suspicion (recent optimization commits on this branch):

- `81aad21` **Tier 5** — single-pass SumOfSquares for symbolic
  InstanceNorm variance. Loss-of-cancellation can drive variance
  slightly negative → `sqrt` → NaN. Very common failure mode for
  Welford-style rewrites of variance.
- `be7916f` **Tier 7** — mimalloc + SinSq + vectorized `sin`. If the
  vectorized sin overreads past a tail or the SinSq path returns
  garbage for inputs near k·π, downstream Snake activations propagate
  NaN through the vocoder.
- `21552d3` **Tier 6** — folding lazy-im2col zero-Pad into the gather.
  An off-by-one on padding indices could read uninitialized (indef)
  memory in a conv kernel.
- `9a2c1d3` **Tier 4** — fused AdaIN/Snake scale multiplication.
- `e33cdb5` / `1059607` — Tier 3/2 lazy im2col + SIMD fusion.

## What's already fixed
- `f1c7e89` on branch `tract-prototype` — sink fallback + clearer error
  when no sink is installed. Independent of the NaN bug.

## What's still broken
- Stage 2 vocoder produces NaN, so the binary is effectively unusable
  for actual TTS regardless of playback setup.
