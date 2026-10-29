# Tiny-operation dispatch investigation

Investigation on `main` after `5c79700`. The installed playback-control binary
is preserved; experiments use separate saved executables.

The measured result is 3.28% less inference CPU time in six same-executable
alternating pairs, with every pair improving. A separate saved-main upgrade
comparison confirms 3.12% less CPU time. Latency is roughly flat to slightly
better across these small samples. All ten raw speech comparisons are exact.
This supports the conservative matmul caller fast path, while row-op dispatch
remains unchanged. Battery-energy savings have not been measured.

## Baseline evidence

The existing source enters `pool.install` for every multithreaded matmul,
including a vector with only one outer tile. The matrix row-outer path contains
two nested installs. The parallel ranges split outer tiles only: rows for packed
matrices/vectors, columns when the lazy input prefers column-outer traversal.

Saved fused profiles contain 142 stage-2 FP16 vector nodes, mostly style
projections: 60 have M=256/K=128, 48 M=128/K=128, 14 M=512/K=128, and the rest
include M=1024, 1090, 514 and symbolic frame-dependent sizes. There is also a
large FP32 vector used by the source generator. All 143 together account for
about 17–32 ms of instrumented wall time per sentence. This is a modest part of
total inference. It excludes matmuls inside Scan: the two long-sentence LSTM
Scans together take about 0.19 seconds, but the top-level profile does not expose
their internal dispatch shapes or counts.

## Microbenchmark method

`tools/bench_tiny_dispatch.rs` compares `Executor::SingleThread` to the existing
three-worker pool. Caller and workers are pinned to gold CPUs 4–6; the observer
and supervising Codex process use CPU 0. Six rounds alternate arm order. Cases
start at or below 50 C on the gold sensors.

Matmuls use the exact registered kernels, prepacked constant weights and
activations, fused bias and the production [tokens, channels] output layout.
Scratch setup inside `run_with_scratch_space` is timed. Packing, output
allocation and initial warmup are excluded. This isolates dispatch plus compute,
rather than measuring the complete layer. Inputs are deterministic synthetic
finite values. Single-call serial/pool outputs are compared bit-for-bit.

For rows, the builder extracts the exact private `sin_sq_f32`, `snake_f32`,
`adain_row` and address-aligned scratch functions from the current math source.
It benchmarks FP32 and FP16 normalization/activation rows through the existing
`par_chunks_mut`, including the FP16 slice conversions. It does not compile a
separate approximation of the row arithmetic.

Each measured arm runs a batch, after 100 ms for prior workers to park. Process
CPU time includes all worker threads and a 30 ms post-batch tail to count worker
search/spin activity. Wall time excludes that tail. Medians below are per-call
batch CPU time, in microseconds. These are local repeated-cache workloads with
the system allocator, not battery or whole-inference percentages.

## Matmul results

| Case | Pool CPU (us) | Serial CPU (us) | Serial CPU change | Serial wall change |
|---|---:|---:|---:|---:|
| f16vec-128 | 62.67 | 3.46 | -94.5% | -89.4% |
| f16vec-256 | 75.33 | 4.97 | -93.4% | -84.9% |
| f16vec-512 | 107.57 | 15.54 | -85.6% | -72.3% |
| f16vec-1024 | 103.00 | 31.47 | -69.4% | -41.6% |
| f16vec-1090 | 169.80 | 36.10 | -78.7% | -60.2% |
| f16vec-dynamic | 255.48 | 164.60 | -35.6% | +25.4% |
| f32vec-256-256 | 131.94 | 25.28 | -80.8% | -70.5% |
| f32vec-1024-256 | 244.55 | 107.60 | -56.0% | -22.7% |
| f32vec-single-tile | 51.91 | 1.88 | -96.4% | -93.8% |
| f16mat-one-row-tile | 97.57 | 13.82 | -85.8% | -76.1% |
| f16mat-256-128-13 | 146.47 | 53.18 | -63.7% | -31.3% |
| f16mat-control | 1207.28 | 1631.47 | +35.1% | +139.8% |

The `f16vec` cases have K=128/N=1; the dynamic control has M=2340/K=256/N=1.
The FP32 vector cases have N=1. The one-row-tile matrix has M=64/K=128/N=13;
the large matrix control is M=512/K=512/N=59. The matrix kernel's MR=64 means
that one-row-tile case has one parallel range item, regardless of its N.
The vector probe kernels are `128x1_gen` for FP16 and `32x1_gen` for FP32;
the matrix probe uses `64x3_gen`. These sample registered implementations;
full-pipeline acceptance measures the kernels actually selected by the model.

## Row results: retain parallel execution

There is no consistent element-count threshold supported by these samples.
For example, serial FP16 LeakyRelu normalization at C=512/F=59 saves 17.8% CPU
but is 57.8% slower. At C=1024/F=59 it saves 10.8% CPU but is 76.9% slower.
At C=512/F=13 it is 98.2% slower with no measured CPU saving. A small synthetic
C=128/F=13 normalization case does improve, but the actual model's C=128 Snake
rows are much longer. On a representative C=256/F=1180 Snake pass, serial uses
69.6% more CPU and takes over three times as long.

A single-row control eliminates dispatch overhead as expected, but it is not
representative of the model's normal 128–1090-channel normalization nodes.
Keep AdaIN and Snake dispatch unchanged in this experiment.

## Candidate matmul policy

The candidate uses the existing sequential tile loop when the actual parallel
outer range has at most one tile, or when the sum of fused matmul work is at
most 262,144 multiply-accumulates. The estimate is M*N*sum(K), with saturating
arithmetic; epilogue-only operations do not qualify through a zero K estimate.
The same matmul kernel, packing, scratch preparation and per-tile arithmetic
remain in use. Calls with multiple outer tiles above the cutoff keep the current
dispatch path. A call can have substantial work but only one outer tile; keeping
it on the caller does not remove any available parallelism in the current loop.

The candidate default is limited to ARM64 Android. Set
`TRACT_CPU_MMM_MIN_WORK=0` to restore the previous pool dispatch, or a positive
integer to experiment with the work cutoff. A positive cutoff also enables the
single-outer-tile shortcut. The value is read once per process. Stage 1 already
uses a single-thread executor and does not consult the policy.

`TRACT_CPU_MMM_TRACE=1` prints each distinct multithreaded dispatch geometry and
its chosen route once. Trace output is diagnostic; performance acceptance uses
unprofiled runs with trace disabled.

## Build, route verification and correctness

The locked offline release build passed in 25 minutes 45 seconds. The saved
candidate is `ryk-tiny-dispatch`, SHA-256
`fcbcb746ff17142e4675e0a3784aef397498623bbd257ff0f2fcc42a6940a90a`.
The source hashes are recorded alongside the build. No arithmetic or kernel
selection source was changed; the modification selects one of the existing
execution loops after scratch preparation.

A `Hello there.` trace observes the same 41 distinct multithreaded matmul
geometries with the policy disabled and enabled. The old route sends all to the
pool; the candidate sends 13 geometries to the caller. They include the actual
recurrent M=256/N=1/K=256 workload, the style-vector projections, and some
single-outer-tile matrices. These are unique geometries, not call counts or time
shares. All routing decisions agree with the cutoff and outer-range rule.

Ten speech cases compare the same executable with `TRACT_CPU_MMM_MIN_WORK=0`
against its default. They cover `af_heart` and `am_michael`, short/medium/long
text at speeds 0.8/1.0/1.2 respectively, plus `a` and `Hi.` at speed 1.0.
All forty raw FP32 tensors (two stage-1 feature tensors, duration predictions
and stage-2 waveform) are finite and byte-identical between modes and match the
saved main reference. WAVs and audio durations also match. This validates the
tested inputs, rather than every possible shape.

## Warm full-pipeline A/B

Six alternating pairs use the same new executable on both sides. A sets
`TRACT_CPU_MMM_MIN_WORK=0`; B uses the 262,144 default. Each daemon compiles once
and warms the full corpus before measurement. Every request generates 22.8
seconds of af_heart audio at speed 1.0. Stage 1 uses CPU 1, stage 2 uses three
workers on CPUs 4–6, governor is off, observer/supervisor use CPU 0, and each
request starts at or below 50 C on the gold sensors.

CPU endpoints include the whole daemon and waited children, from request
submission through discard-sink completion. The optional per-thread observations
partition persistent daemon threads by requested affinity and record CPU and
context-switch deltas. All twelve requests have zero unmatched persistent
threads. Child CPU is included in the process total, not the thread groups.

| Mean per 22.8 s request | A: previous dispatch | B: caller fast path | Change |
|---|---:|---:|---:|
| Total CPU seconds | 37.088 | 35.873 | -1.215 (-3.28%) |
| CPU seconds / audio second | 1.627 | 1.573 | -3.28% |
| Gold-thread CPU seconds | 31.223 | 29.967 | -1.257 (-4.02%) |
| Little-thread CPU seconds | 5.402 | 5.442 | +0.040 (+0.74%) |
| Summed inference time | 18.700 s | 18.495 s | -0.205 s (-1.10%) |
| Synthesis wall time | 14.637 s | 14.477 s | -0.160 s (-1.09%) |
| Gold voluntary context switches | 19,585 | 2,575 | -86.9% |
| Gold involuntary context switches | 26,729 | 4,909 | -81.6% |

CPU savings in the six pairs are 1.27, 1.08, 1.01, 1.57, 1.20 and 1.16 seconds
(2.7–4.3%). Gold CPU falls in every pair, by 0.96–1.42 seconds; stage-1 CPU does
not show a corresponding reduction. Summed inference time improves in every
pair. Synthesis wall time improves in four pairs; the other two regress by only
0.06 and 0.03 seconds. The latency gain is modest and uneven, while the CPU
direction is consistent.

The context-switch observations support reduced worker coordination as a
mechanism, but do not attribute every saved CPU cycle to a particular syscall
or operation. Summed inference time overlaps stages; it is not elapsed request
time. Reported gaps with a discard sink are synthesis-arrival estimates, not
audible playback gaps. The phone was charging: these CPU measurements do not
establish joules per audio second or a battery saving.

A separate two-pair comparison against the saved main executable checks the
actual upgrade across builds; its records are kept separately from the same-
executable policy comparison.

| Upgrade control, mean | Saved main | Candidate | Change |
|---|---:|---:|---:|
| Total CPU seconds | 37.025 | 35.870 | -3.12% |
| Summed inference time | 18.365 s | 18.340 s | -0.14% |
| Synthesis wall time | 14.200 s | 14.370 s | +1.20% |

Both upgrade pairs save CPU (1.28 and 1.03 seconds); synthesis wall time is
0.06 and 0.28 seconds slower. This smaller control confirms the CPU reduction
but does not establish a repeatable latency gain. Combined with the primary
comparison, describe the result as less CPU work with roughly preserved latency.

## Decision

Use the conservative matmul fast path as the ARM64 Android default, with
`TRACT_CPU_MMM_MIN_WORK=0` as its opt-out. Keep the three-
worker pool for the substantial convolution work and keep row-op dispatch
unchanged. A lower threshold or new row threshold needs a separate measured
reason; the present data do not justify broader serialization.

Benchmark work was performed on `main`; the installed `~/bin/ryk` remains on
playback-control. Direct energy validation still needs warm unplugged comparisons
with audio playback and controlled screen/background activity. The next original
implementation item is the specialized one-dimensional Resize path.

## Reproduction and artifacts

The compact profiling archive is under
[`profiling/tiny-dispatch-5c79700`](profiling/tiny-dispatch-5c79700/README.md).
Local complete artifacts are under `tools/bench_out/dispatch-5c79700/`, including the
saved main executable, probe build/source hashes, per-round CSVs, temperatures
and initial parity outcomes. The baseline executable SHA-256 is
`4325720bbbdb3f61141cb279c515796cc74d13479845d436312d60c281f9e3ab`.

```sh
python3 tools/build_matmul_probe.py --dispatch-only --out tools/bench_out/dispatch-probe
python3 tools/bench_tiny_dispatch.py \
  --probe tools/bench_out/dispatch-probe/matmul-probe \
  --out tools/bench_out/dispatch-micro-repeat \
  --supervisor-pid OWNED_SUPERVISOR_PID

python3 tools/bench_warm_inference.py \
  --baseline tools/bench_out/dispatch-5c79700/ryk-tiny-dispatch \
  --variant tools/bench_out/dispatch-5c79700/ryk-tiny-dispatch \
  --baseline-env TRACT_CPU_MMM_MIN_WORK=0 --rounds 6 --thread-stats \
  --out tools/bench_out/dispatch-warm-repeat \
  --supervisor-pid OWNED_SUPERVISOR_PID
```

The supervisor option is optional and should refer only to an owned process
that needs isolation. The helper restores its requested affinities afterward.
The probes are specific to this phone's ARM64 kernels and CPU numbering.
