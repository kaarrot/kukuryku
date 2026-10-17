# Battery and heat on the S10e: what actually moves energy

Companion to `thermal-perf-audit.md` and `fp16-einsum.md`. Those chased
RTF. This one asks what costs battery and heats the phone during a long
read, and which of the existing knobs change that. Measured 2026-10-16 on
the Galaxy S10e (SD855, kernel 4.14), against `11f792c`.

## The rule

**Energy per audio second = work × energy per unit of work.** Only those two
terms matter, and from Termux we can only change the first.

- While stage 2 runs, the gold cluster sits at **2016 MHz** whatever the
  thread count. That is Samsung's policy cap (`scaling_max_freq` 2016000 vs
  `cpuinfo_max_freq` 2419200 on cpu4; cpu7 is capped the same) and an app
  cannot lower or raise it. The scheduler is WALT + schedutil with schedtune
  (`/top-app`), no uclamp.
- So **fewer threads do not save battery.** Two threads do the same CPU work
  as three, at the same clock and voltage, over a longer time. Measured on
  the same paragraph: 58.00 s vs 57.77 s of process CPU (governor off vs on,
  which ran most sentences on 2 threads), while those sentences slowed from
  RTF ~1.15 to ~1.35.
- **Over minutes, fewer threads do not cool the phone either.** Skin and
  battery temperature follow average power. At realtime playback, average
  power is joules per audio second, which the thread count does not change.
  Fewer threads lower the instantaneous core temperature by a few degrees
  (82 °C vs 85 °C peak) and nothing else, unless synthesis falls behind
  realtime and leaves gaps.
- The gold zones (`cpu-1-*-usr`) reach **80–85 °C within 25–45 s** of normal
  3-gold load, in every configuration. The kernel's passive trips are at
  110 °C (`cpu-1-*-step`) and 125 °C (`*-usr`).

The only ways left are less work, which is this change, or a lower clock,
which needs root or the OS (see the end).

## How it was measured

The phone was charging throughout, so battery current is not a load
number. The proxies:

- process CPU seconds (bash `time`, user + sys, includes `pacat`),
- CPU split by cluster (per-thread `/proc` deltas, attributed to the CPU each
  thread was last seen on),
- `cpu4` `time_in_state` (seconds the gold cluster spent at 2016 MHz),
- the hottest `cpu-1-*-usr` zone, sampled with shell builtins only.

Each run started with the golds at ≤ 52 °C and PulseAudio killed, so every
run starts it. Run-to-run noise is about ±2–4%. A helix process was
spinning one thread at ~25% of the prime core the whole time.

`tools/power_bench.sh` had two bugs that made its earlier numbers unusable.
It read `current_now` as µA, but Samsung reports mA there, so watts and J/audio-s
came out 1000× too small. That unit error, plus 3-decimal rounding, produced
the "0.002 → 0.001 J/audio_s" in `5cbecbc`. The same commit's "82 °C → 75 °C"
peak is a floor, not a reading: the peak column included `lmh-dcvs`, a static
75 °C sentinel on this firmware. Its sampler also forked ~200 `cat`
processes per sample, so the "1 Hz" loop took ~2 s per sample under load
and heated the phone itself. Both are fixed: it prefers Samsung's
`batt_current_ua_now`, samples with builtins, integrates over real timestamps,
reports `cpu_s` and a net-of-idle column, and `RYK_BENCH_KEEP=dir` keeps the raw
samples. **Run it unplugged** to turn the CPU savings below into joules.

## Results

3-sentence paragraph (27 s of audio), `KOKORO_GOVERNOR=off` to isolate
compute, two runs each:

| | Process CPU | Gold CPU | Little CPU | cpu4 s at 2016 MHz | Infer RTF | Wall RTF | Peak |
|---|---|---|---|---|---|---|---|
| `11f792c` | 59.6 / 60.3 s | 47.6 / 48.1 s | 9.8 / 9.9 s | 23.9 / 24.2 | 1.122 / 1.131 | 0.899 / 0.901 | 85 °C |
| this change | 56.6 / 57.0 s | 45.6 / 45.9 s | 8.8 / 8.9 s | 21.4 / 22.5 | 1.002 / 1.024 | 0.803 / 0.824 | 85 °C |

About −5% process CPU, −9% gold-cluster time at full clock (synthesis
finishes sooner, so the cluster drops to idle clocks sooner),
−10% little-core CPU (the `Pow` rewrite in stage 1), and infer RTF 1.13 → 1.01.

8-sentence text (68 s of audio), default governor:

| | Infer RTF | Wall RTF | Process CPU | Gold / little | Peak / mean |
|---|---|---|---|---|---|
| `11f792c` | 1.308 | 0.997 | 143.7 s | 115.4 / 25.3 s | 83 / 72 °C |
| this change | 1.041 | 0.856 | 136.3 s | 110.6 / 22.5 s | 85 / 71 °C |

The old default dropped to the 2-thread pool after sentence 1 (10.3 s
buffered ≥ 6 s) and ran sentences 2–8 at RTF 1.30–1.40. The new default
stays on 3 threads (0.98–1.08) and uses 5% less CPU for the same text.

Unplugged, `tools/power_bench.sh` on the same paragraph, default settings,
`11f792c` and this change alternating, cooled to ≤ 52 °C before each run, four
pairs, no suspends (checked with `CLOCK_BOOTTIME − CLOCK_MONOTONIC`):

| | Process CPU | Mean power | J / audio s | Net J / audio s | Wall RTF | Peak |
|---|---|---|---|---|---|---|
| `11f792c` | 59.1–61.7 s (60.3) | 2.28–3.29 W | 4.00–5.39 (mean 4.62) | 2.08–3.85 (2.93) | 1.03–1.23 | 70–83 °C |
| this change | 56.4–56.7 s (56.6) | 2.68–3.48 W | 4.30–5.60 (mean 4.68) | 2.25–3.16 (2.75) | 0.79–0.82 | 76–85 °C |

**The battery cannot tell the two apart.** The 6% less CPU is real and very
repeatable, but it is a small slice of whole-phone power, and the idle floor
alone (screen, the Claude Code session) moved between 0.8 and 1.5 W from one
run to the next. The new build also finishes with the cores 4–8 °C hotter,
because it keeps 3 threads busy instead of dropping to 2. Leakage rises with
temperature, which may eat part of the saving. What the change clearly buys
is speed at the same energy: wall RTF ~0.81 vs ~1.05–1.23, and no gaps. The
old pacing pool left a 3.2 s silence in one of these four runs.

The biggest battery item found was not in ryk. An orphaned helix (its
terminal long closed, parent init) had been spinning one thread on the prime
core for six hours. Killing it took idle power from 1.4–1.8 W to ~1.1 W and
the same ryk run from 3.2–3.8 W to 2.3–3.5 W, an order of magnitude more than
anything in this change. An earlier set of runs with it still alive is
discarded. One of those runs also caught a ~12 s suspend during sentence 1,
before the first chunk, in an interim build that did not open `pacat` until
then. The sink is opened at start again (item 6 below).

## What changed

1. **f16↔f32 Cast through `half`'s NEON slice conversion** (tract-data).
   It was 16–17% of stage 2, single-threaded on the main thread. The output
   is bit-identical. Details in `fp16-einsum.md` #2.
2. **`Pow(x, 3)` → `x * Square(x)`** (tract-core `declutter_pow`). `gelu_new`
   in every ALBERT layer, ~8% of stage 1. In f32 the durations are identical;
   under FP16 it changes the output by no more than FP16's own noise. See
   `fp16-einsum.md` #12.
3. **Governor: no thread drop for buffered audio.** `pace` and `thermal` keep
   the 20 s lookahead cap (`GOV_MAX_AHEAD`), which bounds memory and wasted
   work. Only a thermal hold uses the 2-thread pool now.
4. **Thermal hold recalibrated: on at 90 °C, off below 84 °C**
   (`KOKORO_THERM_HOT_C`). At 78/72 it tripped in normal operation and then
   never released: the 2-thread pool itself runs at ~82 °C, above the old
   release point. Measured case: it tripped at sentence 6 of 8 and held RTF at
   ~1.35 for the rest of the read. It is now a brake for an abnormally hot
   phone (hot room, charging, in a pocket).
5. **`buffered_secs` follows playback.** `written` advanced only after a whole
   chunk's `write_all`, so the sentence being played counted as unplayed, up to
   ~20 s too high. That could hit `GOV_MAX_AHEAD` early and serialize stage 1.
   The player now writes in 100 ms slices.
6. **Audio path:**
   - PulseAudio is started only from the playback thread, which is pinned to
     the littles. Before, `StreamPlayer::create` ran `pulseaudio --start` on the
     gold-pinned main thread, so the daemon inherited cpus 4–6 (`taskset` showed
     `70`; now `f`).
   - One-shot runs start it while the model compiles. It used to block the main
     thread for ~0.7 s before sentence 1.
   - The sink is opened as soon as the run (or a `--serve` job) starts, as
     before. An interim version opened `pacat` only at the first chunk to save
     the idle stream's ~1.5% of a little core. On an unplugged run the phone
     then suspended for ~12 s during sentence 1, because the open audio stream
     is what keeps Android awake with the screen off. The idle cost is noise
     next to synthesis, so the stream stays open.
   - `--load=module-sles-sink` is added only if `default.pa` does not already
     load an OpenSL/AAudio sink. Termux's stock one does, so every auto-start
     made a duplicate `OpenSL_ES_sink.2`.
   - The `--serve` sink grace is 30 s instead of 10 min. A connected `pacat`
     kept the PulseAudio/OpenSL sink RUNNING for 10 min after every `--send`
     (~1.5% of a core plus the Android audio path). Each job warms PulseAudio
     while its first sentence synthesizes, so the shorter grace costs no first
     audio latency.
   - The `--serve` accept loop sleeps in `poll(2)` on the littles. It used to
     poll every 250 ms on a gold core for the daemon's whole life (40 wakeups per
     10 s idle, now 0).
7. **`tools/power_bench.sh`**, as above.

## What did not help

- **2-thread pool for pacing**: same CPU, slower (above).
- **4 golds (`KOKORO_TRACT_CPUSET=4-7`)**: 61.8 s of process CPU vs 57–58 s,
  85 °C peak, hold tripped at sentence 2. The prime core is capped at the
  same 2016 MHz, so it adds power for ~4% speed. Not the default, and the
  README now says why.
- **ConvTranspose in f16**: −0.5% CPU, within noise, for a measurable shift
  in the vocoder output (`fp16-einsum.md` #11).

## What is left

- Stage 2's f16 GEMMs are two thirds of its time, at a clock nothing here can
  change. Stage 2 alone runs at ~0.7× realtime on a paragraph. That headroom
  is what a lower clock would spend, and a lower clock is the only remaining
  lever with a V² payoff. It needs root (`scaling_max_freq`) or the OS. If One
  UI's power saving offers a CPU-speed limit, try it with `power_bench.sh`
  unplugged and watch `gap_s`.
- Measuring a 6% CPU change in joules needs a quieter phone than this one:
  screen off with a wakelock, or many more runs.
- The `--serve` idle path (a sink held 10 min after every `--send` vs 30 s) is
  probably a bigger battery item than anything in synthesis, because an open
  audio stream can keep the phone from suspending. It is not measured here.
