# Battery follow-up validation at 8501a62

The four follow-up validation steps are complete. Speech, real audio integration,
the Android helper and live controls passed. The user confirmed that the wired
headset and notification controls work. Reducing the sink grace from 30 s to 5 s
closed the audio processes about 25 s sooner; this run did not establish battery
energy savings.

Test device: Samsung SM-G970U, Snapdragon 855, Android 12 / SDK 31, Termux.
Implementation: `8501a62` (`daemon: reduce idle work and fix playback position tracking`).
Speech and daemon validation used `target/release/ryk`, SHA-256
`13287858ddbfc3178f0ac79b103e15edaeed732d2d911bdb387faf42fd0cd64f`.
The installed `~/bin/ryk` remains unchanged, SHA-256
`d1349b027a0cecedbb3c63b8f6b88ad8fc2b7a2f053e411e7bc01e595604e92d`.

Local logs, drivers, JSON records and the smoke-test WAV are under
`tools/bench_out/battery-live-8501a62/`. That directory is git-ignored; the tables
below preserve the main results in the repository.

## Speech and first audio

The README's two-sentence lighthouse smoke test used the default thermal
governor, three gold cores for stage 2, little cores for stage 1 and real `pacat`.

| Metric | Result |
| --- | ---: |
| Generated audio | 14.80 s |
| Inference / infer RTF | 11.54 s / 0.780 |
| Synthesis wall time / wall RTF | 9.95 s / 0.672 |
| Reported synthesis gap | 0.00 s |
| First-audio log timestamp from synthesis start | 4.71 s |
| Complete cold command, including load/compile and playback | 24.34 s |

The 4.71 s timestamp excludes model compilation and is not cold command-to-audible
latency. A later warm replacement request logged first audio at 2.40 s from its
synthesis start. Logs and WAV: `rtf-smoke.log`, `rtf-smoke.json`, `rtf-smoke.wav`.

These are post-change smoke-test measurements. No matched old/new RTF comparison
was run, so they do not establish a speed improvement or regression. Synthesis
core counts, thread settings and thermal policy were unchanged by this batch.

## Real audio and Android integration

- The release library suite previously passed with 156 tests and one ignored
  helper-build test, followed by 20 consecutive passing repetitions.
- The ignored real `ecj`/`dx` helper-build test was run separately: one passed,
  7.79 s. The resulting JAR was used for the remaining checks.
- `--headset-check` passed all five Android API steps.
- `python3 tools/rykkeys/chain-test.py --notify` passed all 51 checks in 61 s.
  This dispatched real Android media keys through the real helper, covering
  single/double/triple clicks, custom mappings, one verb per gesture, disconnect
  exit and notification actions, against the chain test's fake daemon.
- `tests/real_sink.rs` passed against real `pacat`. Pause rolled back buffered
  playback, and resume, Back and Cancel stopped/restarted the sink as expected.
  `RYK_EARCONS=0` isolated the main sink because this test counts `pacat`
  processes; a cancel earcon intentionally opens another stream. Earcons were
  enabled in the subsequent live daemon checks.

To avoid rebuilding the same large release library under another target alias,
the existing real-sink test source was compiled with `rustc --test` against
`target/release/deps/libkukuryku-a5ee676d1e74fcbe.rlib` and the matching release
`anyhow` dependency. The production library SHA-256 was
`dfab8ac0ab90d9e2dbf294f48d5f9b37c63ed5a8303c8950f9c57f1428f7b9e0`.
`real-sink-build.json` records the exact command and source hash;
`real-sink-test.log` records the passing run.

## Live daemon and physical controls

The scripted live run used real speech, the Android helper and default earcons.
Its rewind window was shortened to 20 s and daemon idle timeout to 2 s so expiry
could be checked promptly; the normal 600 s rewind default is unchanged.

- All 15 observed JVM threads had the requested CPU mask `0-3`.
- Pause survived four seconds without control traffic, beyond the two-second
  idle timeout; Resume, repeated Back, Rewind and Cancel succeeded.
- Cancel during stage-2 work stopped synthesis about 214 ms after the request.
  The old queued job was skipped and a fresh request produced speech.
- The sink closed about 4.74 s after observed idle. Back at 9.28 s after idle
  opened a new sink and replayed retained speech.
- After replay, history expiry closed the helper watch and the daemon exited
  successfully about 22.29 s after observed idle.
- The initial six-sentence uninterrupted passage generated 20.25 s of audio,
  infer RTF 0.809, wall RTF 0.644, and reported zero synthesis gaps.

The user then tried the separate interactive run with a wired headset and
notification buttons and confirmed: **"all controls work"**. Earlier missed
playback was clarified as the user being away, rather than an audio fault.
The confirmation establishes control functionality; it does not provide a
separate detailed rating of restart clicks or cancel-tone loudness.
Evidence: `live-controls.json`, `live-daemon.log`, `interactive-daemon.log` and
`human-feedback.json`.

## Paired energy measurement including the audio tail

This isolates the sink-grace change using the **same new binary** for both arms:
`RYK_SINK_IDLE_MS=30000` (A) and `5000` (B). It does not measure the combined
before/after effect of all changes in the commit.

Protocol:

- Four pairs were declared before collection, in order `AB`, `BA`, `AB`, `BA`.
- Each run used a separate warmed daemon, default thermal governor, the real
  headset helper, notifications off and actual wired-headset audio. Model load,
  compilation, warm-up and warm audio-server shutdown were outside measurement.
- Voice `af_heart`, speed 1.0, text: "This is a short battery test. The audio stream
  should close soon." Every run generated 4.62 s of audio with zero reported gaps.
- A ten-second idle baseline preceded a fixed 70 s window from the request,
  including inference, playback, sink grace, PulseAudio idle exit and remaining
  idle time. The daemon stayed alive throughout the window.
- The user confirmed unplugged, screen off and untouched, with no prior Termux
  wake lock. Discharging status was checked throughout. Screen state could not
  be independently observed. Music volume stayed at 12/15; all captured stream
  volumes were identical before and after all eight runs.
- A temporary Termux wake lock prevented sampler suspension. The observer ran
  on CPU 0 and sampled `batt_current_ua_now` and `voltage_now` at 20 Hz. Joules
  were integrated by the trapezoid rule with the verified discharge sign.
  Process presence was sampled approximately once per second. No sample gap
  exceeded 0.092 s; both audio processes exited inside every measurement window.
- Gold temperature was required to be at most 46 C before measurement. The
  recorded baseline-start temperatures ranged from 26.8 to 41.3 C; temperatures
  were bounded, rather than equal between paired runs.

### Results

Whole-phone energy over each approximately 70 s window:

| Pair | Run order | 30 s grace (J) | 5 s grace (J) | 5 s minus 30 s (J) |
| --- | --- | ---: | ---: | ---: |
| 1 | AB | 56.14 | 47.37 | -8.76 |
| 2 | BA | 58.10 | 91.45 | +33.35 |
| 3 | AB | 72.27 | 45.30 | -26.97 |
| 4 | BA | 66.48 | 44.34 | -22.14 |
| Mean | | 63.24 | 57.11 | -6.13 |

The paired mean difference was **-6.13 J**, with a two-sided 95% Student-t
interval of **[-49.76, +37.50] J** (four pairs, three degrees of freedom).
Energy savings are therefore inconclusive in this experiment.

| Observed process exit from request | 30 s grace mean | 5 s grace mean | Paired difference |
| --- | ---: | ---: | ---: |
| `pacat` | 37.06 s | 12.00 s | -25.06 s |
| PulseAudio server | 56.77 s | 31.99 s | -24.77 s |

The corresponding paired 95% intervals were [-25.81, -24.31] s for `pacat`
and [-25.39, -24.15] s for PulseAudio. Exit timestamps have approximately
one-second observation resolution. Earlier audio shutdown is established.

Idle baselines varied from 0.51 to 1.20 W. Subtracting each run's baseline gave
a paired difference of +3.68 J, 95% interval [-53.09, +60.45] J, also inconclusive.
Some baseline-subtracted totals were negative because baseline power exceeded
later average power; these are not negative application energy. Four pairs,
background activity, unequal temperatures and fuel-gauge readings limit
precision. The wake lock and sampler also prevent this experiment from proving
unattended suspend/Doze savings.

Data: `energy-tail-controlled/manifest.json`, `summary.json`, and each run's
`result.json`, `samples.csv`, `process-observations.json`, volume snapshots and
`daemon.log`. The local driver is `idle-tail-energy.py`; it reuses the battery
sampler and integration functions in `tools/energy_warm_ab.py`.

An initial attempt under `energy-tail/` was rejected on its volume-consistency
check and excluded in full. The complete eight-run comparison above used the
revised protocol and stable volume; no completed controlled run was excluded.

## Outcome and remaining scope

The selected fixes passed their software and live control checks. The shorter
sink grace removes roughly 25 s of audio-process lifetime while preserving
replay. Battery savings for this change, and for the full commit, remain
unproven. A larger controlled energy study would be needed to establish them.

The temporary wake lock was released and no private test daemon, helper,
`pacat` or PulseAudio process remained. Installation was not part of validation;
the new executable remains at `target/release/ryk`.
The polling and speculative inference/frequency changes deferred in
[`battery-followups-b366946.md`](battery-followups-b366946.md) remain deferred.
