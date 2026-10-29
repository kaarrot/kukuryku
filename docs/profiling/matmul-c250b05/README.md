# Matmul measurements, 2026-10-28

Evidence for the ARM64 Android stage-1 A55 16x8 default. Start with the
[investigation](../../matmul-kernel-investigation.md) for interpretation and
limits. The baseline is fused `main` at `c250b05`, using the saved executable
identified in `warm-ab.json`. The phone is a Snapdragon Samsung S10e, with stage
1 on little CPU 1 and three stage-2 workers on gold CPUs 4–6. The supervising
Codex process was isolated on CPU 0. The installed playback binary was not used.

## Files

- `profiles.json`: three rounds, two stages and three sentence lengths; summed
  per-op wall times and convolution groups from the saved fused profiles.
- `stage1-matrices.json`: shape/kernel groups, call counts and wall times for
  stage-1 matrices in those rounds. Group keys are M, K expression, N, kernel
  name and dtype.
- `convolution-attributes.json`: local ONNX convolution attributes used to
  select representative filter widths, strides and dilations.
- `convolution-layouts.json`: optimized-plan axes, shapes, K expressions and
  kernels for the 77 convolution nodes referenced in the profiles.
- `convolution.csv`: 408 individual kernel measurements, including native lazy
  gathers and fused bias/residual work, with six registered kernels.
- `dense.csv`: 402 individual dense measurements, including activation packing,
  for short/medium/long shapes and an edge check.
- `dense-narrow.csv`: 252 additional dense measurements at N=3, 4, 6 and 8.
- `*-cases.json`: exact probe arguments, temperature observations and parity
  outcomes for each of those CSV case labels.
- `warm-ab.json`: all eight warm requests in AB/BA/AB/BA order, including CPU
  time, audio duration, summed inference time, synthesis wall time and start/end
  gold temperature. A is the fused baseline; B explicitly selects A55 stage 1.
- `warm-same-binary-abba.json`: four additional warm requests with the same
  executable on both sides; only B has the kernel environment override.
- `raw-parity.json`: hashes, equality, finite checks and element counts for four
  raw FP32 tensors across ten speech cases; WAV equality and durations included.
- `kernel-scope.json`: optimized kernel-family counts after the A55 override.
- `*-build.json`, `experimental-source-hashes.json` and
  `experimental-binary.json`: build commands, source/library identity and
  executable hashes for the probes and opt-in application experiment.
- `default-build.json`: successful default-enabled release build, source hashes
  and executable hash. This build follows the measured opt-in experiment.
- `default-validation.json`: checks for the default and both opt-outs. Legacy
  and `0` match the saved original kernel plan; stage 2 and known stage-1 vector
  and convolution nodes are preserved. Two optimized names differ in the
  default plan, so the scope check uses counts and preserved-node identities.
- `default-raw-parity.json`: ten cases comparing the default against legacy in
  the same new executable. All forty raw FP32 tensors also match the earlier
  saved reference. These are correctness checks, not another CPU benchmark.
- `provenance.json`: original local files and their SHA-256 hashes, including
  all eighteen source profile JSONs. CSV values are unchanged; a `case` column
  is added. JSON paths are normalized to the repository, `ASSETS`, or the
  original `fz-prof/` directory rather than an absolute scratch path. Nine
  stage-1 convolution group labels are corrected using the recorded matrix
  axes, since their output layout is [batch, tokens, channels]. The original
  labels and corrected labels are recorded; times and node names are unchanged.

## Reading the numbers

CSV times are seconds. The `total_cpu_s` dense column includes activation
packing and multiply; convolution `cpu_s` includes worker CPU time. The two
edge checks for convolution and one dense edge check use one round. Other cases
rotate all six kernels through each position over six rounds. Use medians per
case/kernel, then compare to `64x3_gen` for dense and `16x8_gen` for convolution.
`bit_equal=true` requires exact FP16 output bits and finite values. Probe input
values are deterministic synthetic values, not recorded speech activations.

Warm tests discard audio and exclude compilation and warmup from the measured
request CPU endpoints. The phone was charging: these data do not measure joules
or prove a battery saving. Stages overlap, so summed inference time is not wall
time. Reported gap estimates are not observed audible gaps. The first baseline
warm request ended hotter than the other requests; the report includes that
limitation and the result with its entire pair excluded.

For each stage/length, take the median over its three profile rounds to reproduce
the investigation table. Profile times are instrumented wall times, not CPU or
energy shares. The `conv_groups` keys in `profiles.json` are M (output channels),
input channels (K/filter width), filter width, dilation, N (output width),
residual flag and dtype. All grouped convolutions have group=1. The matrix axes
come from `convolution-layouts.json`, rather than assuming a channel position in
the output shape. These are grouped workloads, not a complete convolution
geometry.

The complete local artifacts remain in `tools/bench_out/matmul-c250b05/`.
Binaries, models, WAVs and tensor payloads are excluded from this archive.
