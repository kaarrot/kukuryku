# Tiny-operation dispatch measurements

See the [investigation](../../tiny-dispatch-investigation.md) for interpretation.
The baseline is `main` at `5c79700` on the Snapdragon S10e. Experiments preserve
the installed `~/bin/ryk` from playback-control.

- `micro.csv`: 288 serial/pool measurement arms across 24 workloads. Six rounds
  alternate execution order. The `case` column identifies the probe arguments
  in `micro-summary.json`; all other columns retain the raw values.
- `micro-summary.json`: exact arguments, start/end gold temperatures and median
  per-call CPU/wall times. Caller and three pool workers use CPUs 4–6; the
  observer and Codex supervisor use CPU 0.
- `probe-build.json`: compiler command, source hashes and cached library paths
  for the statically linked diagnostic, normalized to repository-relative paths.
  Row arithmetic is extracted unchanged from the math source by the builder.
- `provenance.json`: original CSV paths/hashes and the baseline linalg source
  identity. The initial probe links the pre-policy libraries. The harness now
  explicitly sets `TRACT_CPU_MMM_MIN_WORK=0` so rebuilding it against a library
  with the candidate policy still measures the original pool arm.
- `baseline.json`, `application-build.json`: saved-main executable identity,
  successful release command and candidate source hashes.
- `dispatch-geometries.json`: unique runtime geometries for legacy and candidate
  routes. These are distinct shapes, not frequencies. The candidate serializes
  13 of 41 geometries under the recorded cutoff/outer-range rule.
- `validation.json`, `raw-parity.json`: ten speech cases, forty finite raw FP32
  tensors with exact hashes matching the main reference, and matching durations
  and WAVs. The candidate executable hash is recorded here.
- `warm-same-binary.json`: all twelve warm requests in six alternating pairs.
  A disables the dispatch policy; B uses the default. `*-analysis.json` retains
  per-pair differences, means and relative changes.
- `warm-main-upgrade.json`: four additional warm requests against the saved
  main executable, with analysis kept separate from the same-executable run.

Warm records include persistent-thread CPU/context-switch deltas grouped by
requested affinity, as well as total process/child CPU. All runs have zero
unmatched persistent threads. Little-thread CPU is not interchangeable with
gold-thread CPU for energy accounting, and neither is a joule measurement.
JSON paths use the repository or `ASSETS` in place of local absolute paths;
source hashes and normalization are recorded in the provenance file.

`a,b,c` are M,K,N for matmuls, or channels,frames,unused for rows. Times are
seconds per batch; divide by `repeats` for per-call figures. `work_wall_s`
excludes the post-batch tail. `cpu_including_tail_s` is process CPU including all
workers and a 30 ms tail, after 100 ms for the previous arm's workers to park.
The initial serial/pool result is checked for finite values and exact bits.
The repeated in-place row workloads exercise the output of the previous call;
they are isolated operation probes, not repeated speech synthesis.

Matmul weights and activations are already packed; packing and allocation are
excluded. Scratch preparation and fused bias are included. The system allocator
is used. Large microbenchmark percentages must not be reported as whole-pipeline
or battery improvements. No energy measurement is represented in these files.
