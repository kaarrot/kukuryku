# Contiguous Resize measurements

See the [investigation](../../resize-investigation.md) for interpretation.
The baseline is `main` at `8f4cf95` on the Snapdragon S10e. The installed
playback-control executable is unchanged. Final results use the bounded linear
position table; files prefixed `v1-` describe the earlier full-output table and
must not be pooled with the final implementation.

- `baseline.json`, `baseline-profile.json`, `model-attributes.json`: saved-main
  executable identity, three profile passes and model initializer scales/modes.
- `bounded-application-build.json`, `probe-build.json`, `numeric-tests.json`:
  successful release/probe commands, cached-library identities, runtime source
  hashes and eleven passing optimized numeric tests.
- `bounded-validation.json`, `bounded-raw-parity.json`: final executable hash,
  ten speech cases and forty finite raw FP32 tensors matching byte-for-byte.
  Validation also checks their hashes against the saved-main reference.
- `bounded-profile-ab.json`: 72 Resize calls in two A/B rounds, including node
  names, mode/flag information, tensor geometry and instrumented wall times.
- `bounded-micro.json`, `bounded-micro.csv`: fifteen distinct geometries and
  180 measurement arms from six alternating rounds. CSV adds a `case` column
  while preserving the original fields. Times are seconds per batch; divide
  by `repeats` for per-call figures. Generic includes the old input copy;
  allocation/drop are timed, final Tensor conversion and model execution are
  excluded. The probe uses the system allocator, not the app's mimalloc.
- `warm-bounded.json`, `warm-bounded-analysis.json`: twelve requests in six
  alternating same-executable pairs. A disables the fast evaluator with
  `TRACT_ONNX_RESIZE_FAST=0`; B uses the default. Total CPU falls 1.04%; elapsed
  generation time is roughly unchanged. Every paired CPU difference improves.
- `warm-upgrade.json`, `warm-upgrade-analysis.json`: eight additional requests
  comparing the saved main executable with the final candidate. All four paired
  CPU differences improve; mean CPU falls 1.53%. Keep this control separate.
- `v1-*.json`, `v1-micro.csv`, `numeric-tests-v1.json`, `probe-build-v1.json`:
  the initial prototype and its noisy first warm campaign, saved-main control
  and cooler confirmation. The prototype's sampled RSS cost motivated bounding
  the position table. `v1-to-bounded.patch` can be reversed against the final
  numeric module to reconstruct its source; original source hashes are recorded.
- `cpu-telemetry.csv`: 4693 raw samples across the v1 confirmation, final primary
  and final upgrade campaigns. `campaign` and `run` identify original CSVs.
  Sampling is nominally every 100 ms, with a final sample after CPU endpoints.
- `provenance.json`: original JSON/CSV paths, SHA-256 identities, row counts and
  normalization rules. JSON paths use the repository or `ASSETS` in place of
  local absolute paths. Binaries, weights, WAVs and raw tensor payloads are
  excluded; complete originals remain under `tools/bench_out/resize-8f4cf95/`.

Warm tests compile and warm each daemon once, exclude that work, use a discard
sink, stage 1 on CPU 1 and three stage-2 workers on CPUs 4–6. Observer/supervisor
use CPU 0, with supervisor requested affinity restored afterward. Per-run
temperatures, total process/child CPU, persistent-thread CPU/context switches,
sampled peak RSS and faults are recorded. Final runs have no unmatched persistent
threads or major-fault deltas. Per-thread totals exclude children and are not
identical to the process/child total.

CPU time sums work across cores; elapsed synthesis time measures overlapping
stages. Neither CPU seconds nor reported frequency samples are energy readings.
RSS is sampled, includes retained allocator memory and is not a workspace-size
measurement. Trials used a charging phone without real playback; no joule or
battery-life improvement is established here.
