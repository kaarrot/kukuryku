# Contiguous last-axis Resize investigation

Investigation on `main` after `8f4cf95`. The installed playback-control binary
is preserved. The exact saved main executable is `ryk-baseline`, SHA-256
`fcbcb746ff17142e4675e0a3784aef397498623bbd257ff0f2fcc42a6940a90a`.

The final bounded-workspace implementation reduces whole-inference CPU time by
1.04% in six alternating same-executable pairs, with roughly unchanged elapsed
generation time. All eleven numeric tests and forty raw speech-tensor
comparisons pass exactly. A separate four-pair saved-main comparison confirms
lower CPU time. These are CPU measurements on a charging phone, not battery
energy measurements. The [measurement archive](profiling/resize-8f4cf95/README.md)
keeps final results and the earlier prototype separate.

## Current-main profile and scope

Three passes use the standard short/medium/long corpus, FP16 enabled, stage 1
on CPU 1, three stage-2 workers on CPUs 4–6, governor off and a discard sink.
The observer and supervising Codex process use CPU 0 during measurements. Each
pass includes six Resize nodes per sentence, all FP32 and all changing only the
contiguous last axis. Median instrumented wall times are:

| Sentence | Base frame count | Combined Resize time | Stage-2 share |
|---|---:|---:|---:|
| short | 59 | 0.0247 s | 2.65% |
| medium | 268 | 0.1113 s | 2.77% |
| long | 585 | 0.2430 s | 2.90% |

The model's initializer scales and Resize attributes were read from the local
ONNX model using field numbers in the repository's `onnx.proto3`; no inferred
float scale is substituted for a stored initializer value.

| Workload | Channels | Last-axis resize | Mode |
|---|---:|---|---|
| F0 and noise encoder upsample (two nodes) | 512 | F to 2F | asymmetric / nearest / floor |
| Decoder upsample | 1090 | F to 2F | asymmetric / nearest / floor |
| Pitch upsample | 1 | 2F to 600F | asymmetric / nearest / floor |
| Harmonic source downsample | 9 | 600F to 2F | half-pixel / linear |
| Harmonic source upsample | 9 | 2F to 600F | half-pixel / linear |

Other coordinate modes, nearest rounding modes and layouts keep the generic
path. This avoids broadening the numerical change beyond the observed workload.

## Implementation

The old evaluator copies the complete input into an ndarray, then uses dynamic
multidimensional indexing for every output element. Source coordinates, clamp
indices and interpolation fractions are recomputed for every channel row.

The candidate borrows the input tensor when exactly one eligible last axis
changes. This eliminates the full-input copy, including the large harmonic
source input that is subsequently downsampled. For eligible last-axis steps
after earlier generic axis changes, it can also reuse an already owned array.

- Nearest/asymmetric/floor computes source indices once, coalesces identical
  consecutive indices into runs, then fills those runs for each row. No integer
  repeat assumption replaces the original FP32 coordinate calculation.
- Half-pixel linear precomputes left indices and fractions for at most 1024
  output positions at a time and reuses each block across rows. The position
  workspace is bounded at 16 KiB on this 64-bit target, independent of output
  length. A single row calculates positions directly without a table. The
  expression remains `left * (1 - fraction) + right * fraction` with the original
  FP32 association. This bound applies to the linear position table, not to
  output allocation or nearest-mode run storage.
- Input is not modified. Output remains contiguous and has the original shape.
- Unsupported modes, axes, nonstandard layouts, empty input axes and invalid
  scales decline the fast path. The previous generic loop is retained.

`TRACT_ONNX_RESIZE_FAST=0` restores the previous generic evaluation, including
its input copy. The setting is read once per process. Default enables the fast
path for eligible FP32 arrays; it uses no hardware-specific arithmetic.
Per-node JSON profiles also expose coordinate, interpolation and nearest modes
and the fast-path setting, outside the op timer.

## Numeric checks

The numeric paths live in a small module independent of ONNX parsing, so the
exact functions can be compiled into a focused test/probe against the cached
release data library. The first prototype passed ten optimized numeric tests.
The bounded-workspace version passed eleven, adding exact and partial block
boundaries. They cover:

- Short rows and fractional scales; actual model shapes at F=59, 268 and 585.
- Exact and partial blocks at output lengths 1023, 1024, 1025, 2048 and 2049.
- Batch/rank variations and sizes-derived scale ratios.
- A borrowed input view, matching the application's direct-input path.
- Empty outputs and empty outer dimensions.
- Generic earlier-axis resizing followed by a fast last-axis step.
- Unsupported modes, non-last axes and noncontiguous layouts retaining fallback.
- Exact nearest copying of signed zero, subnormals, infinities and NaN payloads.
- Linear behavior with infinities and zero-weight multiplies (NaN classification).

The legacy half-pixel implementation extrapolates at the first negative source
coordinate. For input `[2,10]`, scale 2 yields `[0,4,8,10]`; that behavior is
explicitly preserved. The fast path also retains both weighted multiplies when
the left and right indices coincide, rather than replacing them with a copy.
Finite values are compared bit-for-bit. No reassociation or reduced precision
is introduced.

## Final bounded-workspace measurements

The release build passed in 14 minutes 9 seconds. Its saved executable is
`ryk-resize-bounded`, SHA-256
`7afe5677789c2d2a3bfb4d9c5d7c2a9e2804bcf8f64169ee36b46741cb31226f`.
The recorded source hashes match the final runtime code.

Ten speech cases use two voices at speeds 0.8, 1.0 and 1.2, plus `a` and `Hi.`.
Generic and fast modes produce forty finite, byte-identical raw FP32 feature,
duration and waveform tensors. Their hashes also match the saved-main reference;
WAVs and durations match. The eleven numeric tests above include both complete
and partial 1024-position blocks.

Two instrumented A/B rounds contain 72 Resize calls. Median combined Resize
wall time for each sentence is:

| Base frames | Generic median | Fast median | Change |
|---|---:|---:|---:|
| 59 | 0.025308 s | 0.003314 s | -86.9% |
| 268 | 0.113638 s | 0.012224 s | -89.2% |
| 585 | 0.246326 s | 0.033588 s | -86.4% |

The fifteen isolated geometries, each with six alternating rounds, use 86–99%
less local CPU time. The probe includes output allocation/drop and the old
evaluator's input copy; it excludes the final Tensor conversion and model
execution. It uses the system allocator rather than the application's mimalloc.
These large operation-level reductions are not whole-inference percentages.

### Primary warm comparison

Both arms use the same final executable. A sets `TRACT_ONNX_RESIZE_FAST=0`;
B uses the default. Six pairs alternate A/B then B/A order. Both daemons compile
once and warm the complete corpus before measurement. Each request generates
22.8 seconds of audio with `af_heart` at speed 1, stage 1 on CPU 1 and three
stage-2 workers on CPUs 4–6. The application governor is off; the OS frequency
policy remains in place. A discard sink excludes real playback. The nominal
gold-cluster cooldown is 45 C; individual start/end temperatures are archived.
The observer and owned supervisor use CPU 0 during measurements.

| Mean per 22.8 s request | A: generic | B: fast | Change |
|---|---:|---:|---:|
| Total process/child CPU | 35.755 s | 35.382 s | -1.04% |
| CPU seconds per audio second | 1.5682 | 1.5518 | -1.04% |
| Persistent gold-thread CPU | 29.860 s | 29.438 s | -1.41% |
| Persistent little-thread CPU | 5.433 s | 5.478 s | +0.83% |
| Summed inference durations | 18.123 s | 18.018 s | -0.58% |
| Elapsed synthesis time | 14.100 s | 14.067 s | -0.24% |

Paired total CPU differences (B minus A) are -0.27, -0.26, -0.66, -0.41, -0.30
and -0.34 seconds: all six improve. Elapsed synthesis differences have mixed
signs, from -0.22 to +0.11 seconds, supporting roughly unchanged latency.
Summed inference durations include stages that overlap; total process CPU sums
work on all cores. Neither is interchangeable with elapsed synthesis time.

The CPU endpoints include the daemon and waited children; per-thread groups
cover persistent daemon threads only. All runs have zero unmatched persistent
threads and zero observed major faults. Telemetry samples every 100 ms and
records RSS, minor faults and reported CPU 1/4 frequencies. Its sampled mean
peak RSS is 501.8 MB for A and 511.4 MB for B (+1.9%); minor faults fall by 5.0%.
Do not claim a whole-process memory reduction. Reported frequency samples
include idle intervals and are not used to infer cycles or normalize CPU time.

### Saved-main upgrade control

Four more alternating pairs compare the saved `8f4cf95` executable with the
final candidate under the same warm protocol. These are separate measurements,
not pooled with the same-executable comparison.

| Mean per 22.8 s request | Saved main | Final candidate | Change |
|---|---:|---:|---:|
| Total process/child CPU | 35.783 s | 35.235 s | -1.53% |
| CPU seconds per audio second | 1.5694 | 1.5454 | -1.53% |
| Persistent gold-thread CPU | 29.920 s | 29.343 s | -1.93% |
| Summed inference durations | 18.470 s | 17.958 s | -2.77% |
| Elapsed synthesis time | 14.443 s | 13.955 s | -3.38% |

All four paired CPU differences improve: -0.61, -0.37, -0.52 and -0.69 seconds.
This confirms a modest CPU benefit against the previous executable. The larger
elapsed-time change is not reproduced by the primary comparison, so the
conservative result is about 1% less inference CPU with latency preserved.
Sampled mean peak RSS falls by 1.7% in this control, while minor faults rise by
3.9%; those directions differ from the primary campaign. All major-fault deltas
and unmatched persistent-thread counts remain zero.

## Earlier prototype (retained separately)

The first release build passed in 15 minutes 26 seconds; its saved executable is
`ryk-resize`, SHA-256
`06b70ccbd21d9d50d76387afd5a8dbe147106758907fe7a68e2a6a76ad1ac7a8`.
All ten speech cases (two voices, three speeds, plus `a` and `Hi.`) have finite,
byte-identical raw FP32 features, durations and waveforms between generic and
fast modes and against the saved main reference. WAVs and durations match too.

Two profiled A/B rounds show combined Resize op time dropping by 87–90%:

| Base frames | Generic median | Fast median | Change |
|---|---:|---:|---:|
| 59 | 0.02981 s | 0.00388 s | -87.0% |
| 268 | 0.13843 s | 0.01675 s | -87.9% |
| 585 | 0.31495 s | 0.03160 s | -90.0% |

Fifteen isolated geometries, each with six alternating rounds, show 76–99% less
local CPU time. The probe includes output allocation and the generic path's
input copy, but excludes model execution and final Tensor conversion. It uses
the system allocator, while the application uses mimalloc.

The first six-pair warm campaign is too variable to establish the much smaller
whole-inference CPU effect: paired CPU differences range from -3.80 to +1.27
seconds, and a separate saved-main two-pair comparison has opposite signs.
All results are retained; the initial mean is not treated as proof of a saving.

A six-pair confirmation with a nominal 45 C cooldown and 100 ms telemetry is
more stable. Mean process CPU falls from 35.765 to 35.543 seconds (-0.62%), with
all six pairs improving by 0.08–0.36 seconds. Gold-thread CPU falls from 29.922
to 29.692 seconds (-0.77%). Mean synthesis wall time falls from 14.330 to
14.087 seconds (-1.70%). No major faults are observed; reported CPU frequencies
vary within requests and are not used to normalize these CPU totals.

That confirmation also shows an undesirable memory cost: the mean sampled peak
RSS is about 510 MB for generic and 555 MB for fast, with more minor faults in
the fast runs. RSS includes allocator-retained memory, and the full cause of
that difference is not established. The first linear implementation built a
position table spanning the output: 5,616,000 bytes for the long upsample.
The final candidate replaces that table with a reusable 16,384-byte block.
Its output values remain independent, and the new block tests are exact.

The first version's data and source identity are preserved separately and are
not pooled with the final bounded version.

## Decision and remaining battery validation

Keep the eligible contiguous Resize path enabled by default, with
`TRACT_ONNX_RESIZE_FAST=0` available to restore the old evaluator. The final
implementation preserves measured output exactly, bounds the linear position
workspace and consistently lowers CPU usage on the measured corpus. Other
Resize modes and layouts retain the generic implementation.

Benchmark work stays on `main`; `~/bin/ryk` is unchanged. These charging-phone
discard-sink trials do not establish joules saved, battery life or playback-gap
behavior. Screen state and background activity were not independently verified.
Direct energy validation remains a warm, unplugged real-playback comparison in
A/B/B/A order with matched screen state, volume, text and starting temperature,
reporting joules and CPU seconds per audio second plus measured playback gaps.

## Reproduction and artifacts

The [compact archive](profiling/resize-8f4cf95/README.md) includes per-round
measurements, source and executable hashes, raw-output hashes, profile records
and CPU telemetry. All complete local results are under
`tools/bench_out/resize-8f4cf95/`: baseline identity,
three current-main profile rounds, model attributes/scales, test/probe builds,
source hashes and the application build log. Binaries, weights and tensor dumps
are not intended for version control.

```sh
python3 tools/build_resize_probe.py --out tools/bench_out/resize-probe
tools/bench_out/resize-probe/resize-tests --test-threads=1
python3 tools/bench_resize.py \
  --probe tools/bench_out/resize-probe/resize-probe \
  --profiles tools/bench_out/resize-8f4cf95/baseline-profile/summary.json \
  --attributes tools/bench_out/resize-8f4cf95/resize-attributes.json \
  --out tools/bench_out/resize-micro-repeat \
  --supervisor-pid OWNED_SUPERVISOR_PID

python3 tools/verify_fusion_parity.py \
  --baseline tools/bench_out/resize-8f4cf95/ryk-resize-bounded \
  --fused tools/bench_out/resize-8f4cf95/ryk-resize-bounded \
  --baseline-env TRACT_ONNX_RESIZE_FAST=0 \
  --baseline-env KOKORO_TRACT_S1_CPUSET=1 \
  --fused-env KOKORO_TRACT_S1_CPUSET=1 \
  --extra-text a --extra-text 'Hi.' \
  --out tools/bench_out/resize-parity-repeat

python3 tools/bench_warm_inference.py \
  --baseline tools/bench_out/resize-8f4cf95/ryk-resize-bounded \
  --variant tools/bench_out/resize-8f4cf95/ryk-resize-bounded \
  --baseline-env TRACT_ONNX_RESIZE_FAST=0 \
  --rounds 6 --cool-c 45 --thread-stats --cpu-telemetry \
  --out tools/bench_out/resize-warm-repeat \
  --supervisor-pid OWNED_SUPERVISOR_PID
```

The supervisor option is optional and only for an owned process needing
isolation. Requested CPU affinities are restored afterward. The benchmark
caller uses gold CPUs 4–6; CPU numbering is phone-specific.
