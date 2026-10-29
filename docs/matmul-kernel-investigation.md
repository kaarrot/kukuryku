# Matmul kernel investigation

Investigation on `main`, starting from the AdaIN fusion commit `c250b05`.
The installed `~/bin/ryk` remains on `playback-control` and is not replaced.

The useful change is in stage 1: selecting the registered A55 16x8 kernel
reduces whole-inference CPU time by 5.7% in four warm alternating pairs on this
phone. All ten raw speech comparisons match byte-for-byte. The vocoder's current
16x8 generic tile beats every other tested registered tile. A same-executable
control confirms 5.3% less total inference CPU time. The A55 stage-1 choice is
now the default on ARM64 Android, with a legacy opt-out. Extremely narrow
matrices can regress by about 0.4 ms per FFN multiply; battery-energy savings
have not been measured.

## Findings from the existing profiles

Recomputed from all three saved fused `fz-prof/B-*.prof` rounds. These are
per-stage medians, not CPU or battery-energy measurements.

| Stage | Sentence | Op time | OptMatMul time | Share | Lazy-convolution matmul time |
|---|---|---:|---:|---:|---:|
| 1 | Short | 0.3396 s | 0.2536 s | 74.7% | 0.0119 s |
| 1 | Medium | 2.2519 s | 1.5597 s | 69.3% | 0.0585 s |
| 1 | Long | 5.0694 s | 3.3634 s | 66.4% | 0.1228 s |
| 2 | Short | 0.9735 s | 0.7404 s | 76.1% | 0.6333 s |
| 2 | Medium | 3.9019 s | 2.9896 s | 76.6% | 2.6968 s |
| 2 | Long | 8.5749 s | 6.4240 s | 74.9% | 5.8359 s |

On the long sentence, lazy convolutions alone account for about 68% of stage-2
time. Their panel gathers execute inside `OptMatMul`; the separate `LazyIm2col`
node mostly prepares offsets. A packed-only GEMM benchmark would miss that work.

The biggest single shape is M=128, K=1408, N=70201: 128 input/output channels
and an 11-tap filter. Convolution attributes were read from the local ONNX models
using field numbers from the repository's `onnx.proto3`, rather than guessing
filter widths or dilation from output shapes.

An optimized-plan inventory built with the same precision translation confirms:

- The 74 FP16 vocoder convolution nodes use `arm64fp16_mmm_f16_16x8_gen`.
- Stage-1 symbolic dense layers generally offer `64x1_gen` for a vector and
  `64x3_gen` for a matrix. The actual sentence inputs select the latter.
- Six known stage-1 FP16 vector operations use `128x1_gen`.

The two selection paths differ. `Conv::choose_impl` calls the default FP16
picker with a symbolic N, which selects 16x8. `einsum::kernel_selection` groups
implementations by compatible left packing and favors a family supporting both
vector and matrix cases; the 64x1/64x3 pair wins that heuristic. It does not
benchmark this choice on the little cores used for stage 1.

## Probe and method

The probes link the existing release rlibs; they do not rebuild or overwrite
`ryk`. A matching thin-LTO link was required. The inventory/convolution probe
took 728.9 seconds to build; the smaller dense-only probe took 52.1 seconds.

`tools/bench_matmul.rs` includes the repository's exact `lazy_im2col.rs` source
and uses the normal MMM executor. Its stride-1 NCHW gather offsets follow
`build_lazy_params`/`PatchSpec`: channels are outermost, then kernel taps, with
the same zero padding, dilation and checked boundary ranges. It includes bias
and, where specified, a residual add. It uses the production [channels, frames]
output strides. Weights, output storage, offsets and scratch are prepared and
warmed before timing; panel gathering and fused epilogues remain in the timed
multiply. Output allocation and deferred-offset construction are excluded.

`tools/bench_dense_matmul.rs` uses the production dense [tokens, channels]
output strides and bias, with constant weights prepacked. Its timed region
includes packing the [tokens, K] activation and the multiply. Both probes use
deterministic finite FP16 values and compare every output bit across variants.
They use the system allocator rather than the application's mimalloc.

Six variants are tested: generic 16x8, 32x4, 32x6 and 64x3, plus A55-specific
16x8 and 32x4. Each representative case has six rounds, rotating all six variants
through every execution position. This is a microbenchmark, not a measurement
of whole-pipeline or battery savings.

- Convolutions: three gold workers pinned to CPUs 4–6.
- Dense layers: single-threaded, pinned to little CPU 1.
- Supervising Codex process temporarily isolated on CPU 0. See the affinity
  restoration correction below.
- Case starts capped at 55 C; dense cases check both little and gold sensors.
- CPU time is process CPU time, including worker threads for convolution.
- All 13 convolution cases (including two edge checks) and 12 dense cases
  (including one edge check) had finite, byte-identical FP16 outputs for all six
  kernels. Seven additional very-narrow dense cases also matched exactly.

## Vocoder: keep the existing tile

Median process CPU time relative to the current 16x8 generic tile:

| Case | 32x4 generic | 32x6 generic | 64x3 generic | 16x8 A55 |
|---|---:|---:|---:|---:|
| 128 channels, 11 taps, N=70201 | +7.5% | +8.1% | +48.3% | +47.1% |
| Same, with residual add | +8.5% | +6.8% | +48.8% | +45.4% |
| 128 channels, 7 taps, N=70201 | +6.1% | +8.4% | +46.0% | +45.9% |
| 256 channels, 11 taps, N=11700 | +27.9% | +7.9% | +73.9% | +49.2% |
| 128 channels, 11 taps, N=32161 | +5.3% | +5.5% | +49.3% | +46.7% |
| 128 channels, 11 taps, N=7081 | +8.1% | +8.3% | +44.7% | +45.2% |

The current tile wins all 11 representative convolution cases. Additional
dilations (3 and 5), 3-tap filters, and a 256-channel residual case show the same
direction. Switching the vocoder wholesale to another registered tile is not
supported by these measurements.

## Stage 1: a promising selection change

Median CPU time for activation packing plus multiply, compared with the current
64x3 generic kernel. Sequence lengths include BOS/EOS tokens.

| Dense layer (M, K) | N=13: 16x8 A55 | N=116: 16x8 A55 | N=244: 16x8 A55 |
|---|---:|---:|---:|
| Projection (768, 768) | -36.9% | -34.3% | -40.5% |
| FFN output (768, 2048) | -36.6% | -35.7% | -42.7% |
| FFN input (2048, 768) | -35.5% | -33.5% | -36.2% |

These three groups account for about 81% of stage-1 MatMul wall time in the long
profile. On N=244, predictor shapes (256,512) and (256,128) also improve by 36.6%
and 33.3%. The generic 16x8 tile helps too, but the A55-specific version is faster
on the little core. Its poor gold-core results above show why a global CPU-kind
override would be inappropriate.

The measured local CPU reduction is substantial. It should not be reported as a
34–43% reduction in overall inference or battery use: stage 2 still dominates,
stages overlap, and these probes omit some application work.

The seven very-narrow cases expose a limitation. N=3 means one recognized
phoneme symbol plus the two boundary tokens. The CLI reports the unpadded token
count; even `a` used N=5 in the parity corpus.

| N=3 dense layer | 64x3 generic CPU time | 16x8 A55 CPU time | Difference |
|---|---:|---:|---:|
| FFN output (768,2048) | 1.981 ms | 2.404 ms | +0.424 ms (+21.4%) |
| FFN input (2048,768) | 1.581 ms | 1.969 ms | +0.388 ms (+24.6%) |
| Projection (768,768) | 0.907 ms | 0.897 ms | -0.010 ms (-1.1%) |
| Predictor (256,512) | 0.121 ms | 0.125 ms | +0.003 ms (+2.6%) |

The first FFN shape improves by 11.2% at N=4, 17.7% at N=6 and 30.7% at N=8.
At twelve repetitions of each FFN layer, the N=3 regressions imply about 10 ms
of extra CPU work in those layers combined, before gains elsewhere. This is an
estimate from local kernel timings, not a measured whole-request penalty.
The consistent typical-input benefit outweighs that small absolute edge cost
for this phone's reading workload, so the default uses A55 with an opt-out.

## Whole-pipeline experiment

`KOKORO_TRACT_S1_DENSE_KERNEL=16x8-a55` sets a model property before stage-1
optimization. The EinSum strategy honors it for FP16 matrix operations using a
compatible registered implementation. Known N=1 operations retain the vector
strategy, and other precisions keep their usual selection. Stage 2 receives no
property, so its concurrent compilation and vocoder kernels are unaffected.
The measured experimental binary required the explicit value above. No
process-global environment mutation or CPU-kind override is used.

Per-node JSON profiles additionally include `Op::info()`, exposing the selected
kernel and dimensions for validation. This metadata is outside per-op timers;
unprofiled runs remain the basis for performance acceptance.

The release build passed with `cargo build --release --locked --offline --bin
ryk` in 31 minutes 50 seconds. The experimental executable is saved separately
as `ryk-s1-kernel`, with the override disabled by default in that experimental
build. It is preserved separately from the subsequent default-enabled build.
The installed playback executable was not replaced.

A profiled scope check found 172 stage-1 matrix operations using the requested
16x8 A55 tile. Six known matrix-vector operations still use 128x1 generic, and
three convolutions still use 16x8 generic. Stage 2 retains its existing kernel
families, including all 74 FP16 vocoder convolutions on 16x8 generic; no A55 tile
appears in stage 2.

Raw parity passed for ten cases: `af_heart` and `am_michael`, each with short,
medium and long text at speeds 0.8, 1.0 and 1.2 respectively, plus `a` and `Hi.`
at speed 1.0. Both stage-1 feature tensors, duration predictions and stage-2
waveforms are finite and byte-identical in every case. All ten WAV pairs also
match. This verifies the tested inputs, rather than all possible shapes.

## Default and opt-out

On ARM64 Android, an absent, empty or `default` value of
`KOKORO_TRACT_S1_DENSE_KERNEL` now chooses `16x8-a55`. This runs only when FP16
is enabled and only for `stage1.onnx`; stage-2 kernels and known N=1 operations
keep their previous selection. Other platforms retain their previous default.

To restore the complete previous stage-1 kernel strategy, use:

```sh
KOKORO_TRACT_S1_DENSE_KERNEL=legacy target/release/ryk 'Hello there.'
```

`=0` is an equivalent opt-out. `=64x3-gen` explicitly forces a matrix tile and
is not equivalent to restoring tract's vector/matrix strategy; use `legacy`
for a genuine old-configuration comparison. The explicit experiment values
`16x8-a55`, `16x8-gen`, `32x4-a55`, `32x4-gen`, `32x6-gen` and `64x3-gen` remain
available. Select the environment when starting a daemon; it compiles its plans
once and retains that choice.

The default targets Android's normal little-core stage-1 placement. The
measurements do not establish a win on every ARM core or when users move stage 1
to the gold cores. The opt-out preserves that configuration choice.

The default-enabled locked offline release build passed in 13 minutes 43
seconds. It is saved as `ryk-default`, SHA-256
`4325720bbbdb3f61141cb279c515796cc74d13479845d436312d60c281f9e3ab`.
The same executable was then tested with no kernel environment override versus
`legacy` on all ten speech cases. All forty raw FP32 tensors are finite and
byte-identical between modes and also match the earlier fused-main reference;
WAVs and audio durations match as well.

JSON profiles confirm 172 matrices use A55 under the new default. Both `legacy`
and `0` reproduce the original full kernel plan exactly. All stage-2 kernel
choices and the nine known stage-1 vector/convolution nodes are preserved.
Two optimized stage-1 node names differ between the default and legacy plans;
scope verification compares their kernel counts and the preserved nodes rather
than requiring every optimized name to remain identical. Build/source hashes
and these checks are archived in `default-build.json`,
`default-validation.json` and `default-raw-parity.json`.

## Warm inference CPU comparison

`tools/bench_warm_inference.py` keeps both daemons warm, compiles once and warms
each with the full corpus before measuring. Each request produces 22.8 seconds
of `af_heart` audio at speed 1.0, with short, medium and long sentences. Stage 1
uses CPU 1, stage 2 uses three workers on CPUs 4–6, and the governor is disabled.
The observer and supervising Codex process use CPU 0. Requests start below 50 C
on the gold sensors, and playback is discarded.

CPU endpoints use process user/system time, including worker threads and waited
children, from just before submitting a request until its sink closes. Model
compilation and warmup are excluded. The phone was charging, so these runs
measure CPU work and latency, not joules or battery drain.

Four alternating pairs ran in AB / BA / AB / BA order. A is the saved fused-main
executable, B is the new build with `KOKORO_TRACT_S1_DENSE_KERNEL=16x8-a55`.

| Metric, mean per request | A | B | Difference |
|---|---:|---:|---:|
| CPU seconds | 39.155 | 36.918 | -2.238 (-5.7%) |
| CPU seconds / audio second | 1.717 | 1.619 | -5.7% |
| Summed inference time | 20.818 s | 18.720 s | -2.098 s (-10.1%) |
| Inference RTF | 0.913 | 0.821 | -0.092 |
| Synthesis wall time | 16.550 s | 14.553 s | -1.998 s (-12.1%) |

CPU savings were 2.41, 2.13, 2.27 and 2.14 seconds in the four pairs (5.5–6.1%).
All four pairs also improved summed inference time. The first A run ended at
67.4 C, versus 52.8–55.9 C for the other runs, despite similar starting
temperatures. Excluding that entire first pair still gives 5.6% lower mean CPU
time. This small sample does not remove all frequency or thermal effects.

Summed inference time includes the two overlapping stages; it is not elapsed
request time. The corpus increases sentence length, and the sink discards audio,
so reported gap estimates must not be treated as observed playback gaps.

An additional ABBA control using the same new executable for A and B, with only
the B environment override, isolates kernel selection from rebuild effects.
Mean CPU time falls from 39.315 to 37.225 seconds (-5.3%), and summed inference
time from 20.955 to 18.500 seconds (-11.7%). Both pairs improve: CPU savings are
2.37 seconds (6.0%) and 1.81 seconds (4.6%). Thus the CPU improvement repeats
when executable code and layout are identical. Its results are recorded
separately in `warm-inference-same-binary/`, rather than pooled with the first
comparison.

## Affinity restoration correction

On this phone, `sched_getaffinity` can omit temporarily parked CPUs even while
`/proc/PID/status` retains the full requested `Cpus_allowed_list`. Saving and
restoring only the effective mask had narrowed the supervisor's requested mask
between earlier probe runs. The probes themselves explicitly pinned their
tested cores, and the supervisor was isolated on CPU 0 during those timings.

The warm benchmark helper now saves the requested mask from proc, restores it
for existing and newly created threads, and verifies restoration. The
supervisor's original full 0–7 requested mask was restored before the warm CPU
comparisons. A separate child-process check passed restoration across parked
CPUs. The warm runs record both the masks and successful restoration in
`supervisor-affinity.json`.

## Remaining optimization work

Keep the vocoder kernel selection as it is. AdaIN/activation/precision fusion
is committed, and existing FP16 tile selection has now been measured. Two
implementation items from the original review remain: measure and avoid
thread-pool dispatch for tiny operations, then consider a specialized contiguous
one-dimensional Resize path. Neither has a demonstrated CPU or energy gain yet.
Parallel Cast tuning remains deferred because its latency improvement did not
reduce total CPU work.

The next battery test should compare joules per audio second during warm,
unplugged real playback in A/B/B/A order with matched screen state, volume,
text and temperature. Less CPU time is encouraging, but does not establish the
battery benefit.

## Artifacts and reproduction

The commit includes the per-round timings, profile summaries, scope and parity
records under [profiling/matmul-c250b05](profiling/matmul-c250b05/README.md).
The archive documents provenance and normalization; it contains no binaries,
model weights, WAVs or full tensor dumps.

The complete local profiles, inventory, convolution attributes, source
snapshots, build logs, benchmark CSVs and summaries are in:

`tools/bench_out/matmul-c250b05/`.

The saved fused baseline is `ryk-before-kernel`, SHA-256
`c7d585311e79d663ec37fbb3f1949f5080de779504d54cc5bc949d4db257b7dd`.
It matches the AdaIN validation's fused binary. The probe build records its
compiler command, libraries and source provenance in each `build.json`.

The experimental executable SHA-256 is
`ce98e9f771f90ef230c2900bdb6033bdfcb8da06c0803a2d192b4b5845fb88b1`.
The application build log and source hashes, kernel scope profiles, raw parity
results and warm manifests are stored alongside the probe results. These hashes
identify the measured opt-in executable; the default-enabled build is recorded
separately.

```sh
python3 tools/build_matmul_probe.py --out tools/bench_out/matmul-probe
KOKORO_TRACT_S1_DENSE_KERNEL=legacy \
  tools/bench_out/matmul-probe/matmul-probe inventory ASSETS inventory.json
tools/bench_out/matmul-probe/matmul-probe bench 128 128 11 1 70201 3 6 0

python3 tools/build_matmul_probe.py --dense-only --out tools/bench_out/dense-probe
tools/bench_out/dense-probe/matmul-probe 768 2048 244 1 6

python3 tools/verify_fusion_parity.py \
  --baseline tools/bench_out/matmul-c250b05/ryk-before-kernel \
  --fused tools/bench_out/matmul-c250b05/ryk-s1-kernel \
  --fused-env KOKORO_TRACT_S1_DENSE_KERNEL=16x8-a55 \
  --extra-text a --extra-text 'Hi.' \
  --out tools/bench_out/matmul-parity-repeat

python3 tools/bench_warm_inference.py \
  --baseline tools/bench_out/matmul-c250b05/ryk-before-kernel \
  --variant tools/bench_out/matmul-c250b05/ryk-s1-kernel \
  --variant-env KOKORO_TRACT_S1_DENSE_KERNEL=16x8-a55 \
  --out tools/bench_out/matmul-warm-repeat \
  --supervisor-pid OWNED_SUPERVISOR_PID
```

For the original same-executable control, pass the saved opt-in `ryk-s1-kernel`
for both `--baseline` and `--variant`, keep the B override, and select `--rounds
2` with a fresh output directory. With a default-enabled build, pass that build
for both binaries, set `--baseline-env KOKORO_TRACT_S1_DENSE_KERNEL=legacy`, and
omit the B override. `--supervisor-pid` is optional; supply an owned process
only when it must be kept off the measured cores.

The convolution command arguments are M, input channels, kernel width, dilation,
output width, workers, rounds and residual flag. The dense arguments are M, K,
N, core and rounds. These probes pin themselves to the specified phone cores;
the outer scripts used for the reported runs also cooled the phone and isolated
the supervisor. The installed playback binary is never invoked or modified.
