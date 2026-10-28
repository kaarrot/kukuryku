# AdaIN fusion validation

Validation on 2026-10-28, on `main`, using the existing baseline and fused
binaries. The installed `~/bin/ryk` from `playback-control` is not part of the
experiment.

## Correctness review

No correctness defect was found in the new AdaIN/activation/Cast fusion.

- Normalization retains the existing FP32 mean kernel, eight FP64 variance
  accumulator lanes, and the original FP32 affine association. Snake calls the
  same scalar expression as the standalone operation; LeakyRelu calls the same
  kernel. Side inputs remain FP32.
- FP16 input is widened with the same `half` slice conversion used by Cast;
  output is narrowed after the activation. Intermediate FP32 storage changes,
  rather than the locations of arithmetic rounding boundaries.
- The scratch buffer preserves the row's address modulo 16. For widened FP16
  input this is `(row_index * row_length) % 4` FP32 items; for FP32 input it is
  taken from the input row pointer. This preserves the mean kernel's grouping
  of prefix, aligned middle and tail. Scratch storage is local to each worker
  thread; row writes are disjoint.
- `sole_reader` excludes model outputs and requires exactly one consumer edge.
  It guards activation absorption, output-Cast absorption and bypassing the
  input Cast. Thus another reader or a model output retains the intermediate
  value it needs. The graph rewrite keeps all five side inputs. The fused op's
  identity comparison includes its activation constants and input/output types.
- Type/rank checks, per-channel/batch affine inputs and all four FP16/FP32
  input/output combinations were reviewed. Existing tests include model-output
  preservation, all four storage combinations, multiple row lengths and all
  four address phases. End-to-end runs exercise the parallel executor too.

The cached release test executable
`third_party/tract/tract-core/target/release/deps/tract_core-96fcf1b24ed8cf3d`
was rerun with `ops::math::tests:: --test-threads=1`: **14 passed, 0 failed**.
This includes all the fusion, address-phase and Snake tests. It was not rebuilt.

The current runtime source was compared with the saved build source
`scratchpad/math-mod.good.rs`: they match exactly before the test module.
Only two additional tests differ. The comparison and source hash are saved in
`tools/bench_out/fusion-validation-20261028/source-provenance.json`.

These checks establish the behavior of the reviewed paths and tested inputs;
they are not an exhaustive proof over arbitrary floating-point inputs or every
platform.

## Raw FP32 parity

`tools/verify_fusion_parity.py` runs one sentence per process, because the
existing stage-boundary dump replaces its files for each sentence. It uses a
discard sink for parity only, with profiling disabled. Both binaries use the
same local model assets and default FP16 stage-2 translation.

| Voice | Speed | Sentence | Audio duration |
|---|---:|---|---:|
| af_heart | 0.8 | Short | 1.825 s |
| af_heart | 1.0 | Medium | 6.700 s |
| af_heart | 1.2 | Long | 12.775 s |
| am_michael | 0.8 | Short | 2.050 s |
| am_michael | 1.0 | Medium | 7.525 s |
| am_michael | 1.2 | Long | 13.875 s |

**All six cases have byte-identical raw FP32 output** for all four dumps:
`s1_feat640`, `s1_feat512`, `s1_dur`, and `s2_out0`. All values are finite.
Output sample counts and duration predictions match exactly. The corresponding
six WAV pairs also match. This checks the raw waveform before PCM quantization,
in addition to the earlier corpus's WAV parity.

The earlier fusion experiment's 14 WAVs were independently hashed too; all
match the original corpus's SHA-256
`7784e1c7eff9e4bd82f9c6287681b11e806b070e680a6e3b4589ce95c7eaced7`.

Raw dumps, logs, text and hashes:
`tools/bench_out/fusion-validation-20261028/parity/summary.json` and its per-case
directories.

## Warm playback protocol

`tools/bench_warm_playback.py` runs A/B/B/A with a separate private daemon and
socket for each run. The models compile once; an 18-second warmup exercises the
five sentence shapes before cooling. Each measured request repeats the same
five-sentence paragraph ten times: **180 seconds of af_heart audio at speed 1**.

- Real `pacat`/PulseAudio playback; no discard sink, WAV writing or profiler.
- Normal `thermal` governor and FP16 stage 2 for both binaries.
- Start below 40 C in the live gold-cluster temperature sensors, after a
  15-second idle power sample. A cooldown timeout aborts rather than starting
  hot. Actual starting temperatures are retained with each result.
- The measurement starts before request submission and ends after the real
  audio child has drained and exited. The sink grace is reduced equally to
  200 ms to delimit completion, rather than including the production 30-second
  post-playback grace. Model compilation and warmup are outside this interval.
- Battery must report `Discharging` throughout. Current comes from Samsung's
  explicit microamp file `batt_current_ua_now`; voltage is in microvolts.
  Whole-phone energy is trapezoidal integration of their product over actual
  monotonic timestamps, sampled approximately once per second.
- CPU time is the daemon process's threads plus live/waited-for children,
  including espeak and pacat. It excludes detached PulseAudio; battery energy
  includes the whole phone and audio system.
- Music volume is recorded before and after each run. Screen state is not
  accessible through this session; the user was asked to keep it and phone
  activity constant. Android background activity is not independently controlled.
- Gap values are the application's synthesis-arrival estimates, not a
  hardware audio-underrun counter.

The supervising Codex process was found to consume approximately one CPU core
even while waiting. In the final protocol all of its threads were temporarily
pinned to efficiency CPU 0. Its measured average load was 92.8% of one core for
A and 93.0% for B. Inference retained its normal affinity. The supervisor's
original thread affinities were restored and independently checked afterward.
This is still a comparison with a loaded efficiency core, rather than a
measurement of an otherwise idle phone.

An initial baseline trial was excluded because the first harness used this
kernel's unavailable `/proc/PID/task/TID/children` files and could not verify
playback drain. It stopped at synthesis completion. Its data are retained under
`excluded-missing-child-observation/` and are not part of the final A/B/B/A set.
The corrected harness reads PPIDs from `/proc/PID/stat`; live-child discovery
and actual pacat completion were verified with a short playback probe before
restarting the experiment.

A subsequent pass was interrupted during cooldown before its closing baseline:
the supervisor was busy on the fast cores and starting temperatures differed by
11 C. Its two runs of the same fused binary differed by 33% in energy. These
diagnostic records are retained under `excluded-supervisor-load/`. Neither
excluded pass contributes to the final results below.

## Warm playback results

All four final runs completed and are retained; none was discarded for its
energy result. All had 180 seconds of audio, zero reported synthesis gaps,
unchanged music volume (9/15), and `Discharging` battery status throughout.
Observed pacat processes drained and exited before each measurement ended.
Gold temperatures peaked at 80.0–85.0 C, below the 90 C hold threshold.

| Order | Binary | Start gold | CPU seconds / audio second | Joules / audio second | Idle watts |
|---|---|---:|---:|---:|---:|
| 1 | A baseline | 37.1 C | 1.77250 | 2.89356 | 0.715 |
| 2 | B fused | 38.3 C | 1.76839 | 3.70850 | 1.468 |
| 3 | B fused | 38.3 C | 1.75794 | 2.80570 | 0.783 |
| 4 | A baseline | 39.8 C | 1.76939 | 3.50149 | 1.716 |

| Metric | A mean | B mean | Change |
|---|---:|---:|---:|
| CPU seconds / audio second | 1.77094 | 1.76317 | -0.44% |
| Process CPU seconds per request | 318.77 s | 317.37 s | -1.40 s |
| Reported inference time | 169.255 s | 165.950 s | -1.95% |
| Inference RTF | 0.9405 | 0.9220 | -0.0185 |
| Whole-phone joules / audio second | 3.19752 | 3.25710 | +1.86% |
| Idle-adjusted joules / audio second | 1.95250 | 2.10715 | +7.92% |
| Request through playback drain | 184.199 s | 183.928 s | -0.271 s |
| Reported synthesis gaps | 0 s | 0 s | No difference |

CPU time falls in both pairs: by 0.74 s and 2.06 s per 180 seconds of audio.
This is a small reduction in observed CPU work, consistent with the earlier
silent-synthesis experiment's approximately 0.5% CPU reduction. Inference-time
differences are +0.20 s and -6.81 s, so the mean timing gain is uneven in this
small warm sample. The earlier four-pair silent run had a 2.9% mean infer-time
gain. Playback elapsed time is constrained by the same 180 seconds of audio.

**No battery improvement is demonstrated.** The raw energy differences between
pairs have opposite signs: **+0.81494 and -0.69580 J/audio-second**. The two
baseline readings differ by 21%; the two fused readings differ by 32%. Idle
power also changes greatly. Subtracting a separate 15-second idle observation
does not control background changes during the following three-minute run.
The raw and idle-adjusted means both favor A in this sample, but the variation
does not establish a repeatable energy regression or saving from fusion.
Screen state and Android background activity remain unverified.

Energy and endpoint CPU totals were independently recomputed from all four CSV
files and checked against the harness results. Music volume, discharging status,
audio durations, gaps, actual pacat observation and installed-binary hash were
also checked. Raw data, per-run logs, results, affinity restoration and the
recomputed summary are under:

`tools/bench_out/fusion-validation-20261028/warm-playback/`.

## Recommendation

The fusion is ready for a commit as a modest performance and intermediate-storage
optimization: the reviewed arithmetic and guards are sound on the tested paths,
raw FP32 parity passes, and both CPU-time pairs improve slightly. Its battery
benefit remains unproven; the commit description should not claim an energy
reduction.

For the battery objective, prioritize reducing matmul work or improving its
kernels: the earlier profile attributes roughly three quarters of stage-2 time
to matmul. A future energy comparison should run with the supervising coding
session closed, screen state fixed, and Android background activity quiet. More
three-minute repeats under the current noisy conditions are unlikely to resolve
a sub-percent effect efficiently. Parallel-Cast threshold tuning remains deferred.

The installed `~/bin/ryk` is unchanged (SHA-256
`1bcec71b55e8e61f8937a62590c102455ca60938aaf12c24b4ee3352a18fc8cf`).

## Binary provenance and reproduction

- A: `tools/bench_out/fusion-main-7d34861/ryk-main-profile`, SHA-256
  `28fb8b8787c2400f868527ac4f64c3de0a1e3941570e129cccf711d394e2fe65`.
- B: saved `scratchpad/ryk-fused` (also matches `target/release/ryk`), SHA-256
  `c7d585311e79d663ec37fbb3f1949f5080de779504d54cc5bc949d4db257b7dd`.
- The parallel-Cast experiment is absent from both binaries; `cast.rs` matches
  `HEAD`.

The scratch root is
`/data/data/com.termux/files/usr/tmp/claude-10325/-data-data-com-termux-files-home-PRJ-kukuryku/9a05e291-6c8c-4a4d-b708-e29f12835edf/scratchpad/`.

```sh
python3 tools/verify_fusion_parity.py --baseline BASELINE --fused FUSED --out NEW_PARITY_DIR
python3 -u tools/bench_warm_playback.py --baseline BASELINE --fused FUSED --out NEW_PLAYBACK_DIR
```

For the final run here, the playback command additionally used
`--cool-c 40 --supervisor-pid 24758`. The PID option is only for a deliberately
selected supervisor owned by the current user; it changes that process's thread
affinities during the experiment and restores them on normal completion or
Python exception unwind. It is unnecessary when running without the coding
session. The actual restoration is recorded in `supervisor-affinity.json`.

Both output directories must be new. The playback command requires an unplugged
phone and real pacat; its default workload is about twelve minutes of audio plus
warmup and cooldown time. Neither command installs a binary or changes branches.
