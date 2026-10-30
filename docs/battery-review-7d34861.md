# Battery optimization review — main at 7d34861

Reviewed 2026-10-28. Objective: reduce energy per second of generated speech while
preserving roughly realtime playback and current audio quality.

There is further potential, primarily in eliminating intermediate tensor passes
and improving the dominant matrix kernels. The easy configuration changes have
mostly already landed. No new battery-saving percentage is established by this
review. Application and inference source were not changed.

## Evidence and measurement limits

The current source includes FP16 GEMMs and bias/residual propagation, fused AdaIN,
the cheaper Snake sine-square implementation, lazy im2col, SIMD precision
conversion, mimalloc, stage-1 lookahead, three middle cores for stage 2, and
reduced idle audio/polling overhead. Older reviews against 4ece158 propose several
things that have since landed; they are not all remaining opportunities.

One short profile was run with the existing `target/release/ryk`, using:

```sh
KOKORO_VERBOSE=1 KOKORO_TRACT_PROFILE=1 KOKORO_TRACT_PROFILE_NODES=25 \
  target/release/ryk 'She stood at the lighthouse and watched the boats come in.'
```

The actual invocation also selected the existing local model directory explicitly
and used a 60-second timeout. It completed successfully, generated 3.38 seconds
of audio, and reported infer RTF 0.972, wall RTF 1.100, gap 0.00 seconds, total
process wall time 12.10 seconds. A single sentence cannot benefit from overlap
between consecutive sentences. The wall RTF includes 0.42 seconds of preparation;
total process time additionally includes loading/compilation and playback drain.

**This is an indicative profile, not a verified HEAD benchmark.** The existing
binary's dependency file references `src/control.rs`, `src/headset.rs`, and
`tools/rykkeys/RykKeys.java`, which are absent from this checkout. It was not
rebuilt. Its log confirms the recent inference optimizations (486 stage-2 nodes
translated, 62 AdaIN calls, 331 Cast calls), but does not prove source identity.
Binary SHA-256:
`92059b82a87ffc1ce86ebf44e8d7dfaf7acf586ea9e62a38c60452bd4079c654`.
The [complete profile](battery-review-7d34861-existing-binary.log) is retained.

| Operator | Stage 2 wall time | Share of stage 2 |
|---|---:|---:|
| OptMatMul | 1.548 s | 73.2% |
| Cast | 0.139 s | 6.6% |
| Snake | 0.101 s | 4.8% |
| AdaIN | 0.060 s | 2.8% |
| Resize | 0.056 s | 2.7% |
| Scan | 0.041 s | 1.9% |

Stage 1: OptMatMul 71.3%, Scan 16.8%, packing 2.6%, Cast 2.1%.
These are instrumented wall-time shares, not CPU-time or energy shares. There was
no new joule measurement, comparative test, or sustained thermal test. Device
reads showed discharging at 18% and cpu4 `scaling_max_freq=2016000`; the review
used one short run rather than a battery benchmark campaign.

## Ranked implementation opportunities

### 1. Fuse normalization, activation and precision conversion

**Best first implementation candidate: concrete redundant memory traffic.**

`AdaIn::eval` requires FP32, as does `Snake::eval_in_place` in
[`tract-core/src/ops/math/mod.rs`](../third_party/tract/tract-core/src/ops/math/mod.rs).
Both are deliberately excluded from FP16 translation in
[`fp16_translate_node`](../src/tract_backend.rs). Thus, applicable paths still
materialize widened FP32 activations and later narrow them for the next FP16
convolution. AdaIN and its following activation also make separate passes.

Implement a fused path only for matching, single-consumer chains. First combine
AdaIN and Snake/LeakyReLU while preserving FP32 arithmetic. Then support FP16
input/output around that combined op: widen a row into reusable scratch, retain
the existing reduction order and FP64 variance accumulation, apply the activation,
and narrow at the same point as the original graph. Preserve FP32 output wherever
another consumer requires it. This can remove full-tensor temporary writes and
reads without changing the model's mathematical operations.

The 6.6% Cast share gives scale, not a promised saving: only part is removable,
and widening/narrowing instructions still have to execute. Additional savings
would come from fewer activation passes, allocations, and pool dispatches. The
earlier experiment that broadly converted scalar ops to FP16 is not this design;
it added boundaries and damaged output. Avoid introducing extra FP16 roundings.

### 2. Benchmark the existing FP16 matrix kernels on representative shapes

**Largest remaining compute target, but benefit is not established.**

[`tract-linalg/src/arm64.rs`](../third_party/tract/tract-linalg/src/arm64.rs)
uses a simple FP16 picker: generic 16x8 versus 32x4 based on N padding. It does not
use M/K or a measured cost model for this phone. The implementation registers
64x3 and 32x6 alternatives in
[`arm64fp16.rs`](../third_party/tract/tract-linalg/src/arm64/arm64fp16.rs), which
that default FP16 picker does not select. Android also misses the Linux-only CPU
part auto-detection branch and normally selects `Generic`.

Benchmark the full convolution path, including packing, edges and fused epilogues,
on the largest actual shapes. The indicative profile is dominated by convolutions
producing `[1,128,16201]`, with earlier `[1,256,2700]` shapes also prominent. Keep
kernel selection compatible with symbolic shapes and prepacked weights.

Do not assume the ops are memory-bound: timing shares alone do not establish that.
Nor does an older FP32 experiment rule out differences between these FP16 kernels.
As a scale illustration, cutting the measured stage-2 matmul time by 10% would
remove about 0.155 seconds, or 4.7% of the reported 3.28-second inference time;
this is arithmetic, not an observed improvement or energy forecast.

CPU-kind selection is process-global. Forcing A55 kernels globally also changes
the gold-core vocoder; a proper stage/core-specific choice needs care.

### 3. Avoid dispatching tiny operations to the entire thread pool

**Plausible energy improvement; first measure dispatch and waiting overhead.**

[`frame/mmm/mod.rs`](../third_party/tract/tract-linalg/src/frame/mmm/mod.rs)
enters `pool.install` for every multithreaded GEMM, including matrix-vector and
small-tile workloads, without an estimated-work threshold. The per-channel Snake
and AdaIN paths likewise enter the parallel row helper directly. The generic
`par_elementwise` path already has a threshold, so it should not be counted as a
missing optimization everywhere.

Use a serial path below a measured M/N/K-work or element-count threshold, and
coarser batches above it. Keep the three-thread pool for substantial convolutions.
The locked Rayon 1.13.0 source yields through 33 search rounds before sleeping;
it does eventually park. There is no evidence of permanently spinning idle pools.
Avoid changing Rayon's global sleep policy before measuring the dispatch cost.

### 4. Specialize the contiguous one-dimensional Resize path

**Smaller, well-scoped work reduction.**

[`tract-onnx/src/ops/resize.rs`](../third_party/tract/tract-onnx/src/ops/resize.rs)
uses dynamic multidimensional indexing and computes coordinates separately for
each output element. For last-axis resizing, compute source indices/weights once
per output position and reuse them across contiguous channel rows. A nearest
mode can avoid loading both neighbours; an eligible exact repeat path can avoid
interpolation entirely. Preserve the model's coordinate transformation, rounding,
boundary handling and FP32 arithmetic exactly, especially in the harmonic source.

Resize was 2.7% of stage 2 in the indicative profile; removing its entire cost
would only remove about 1.7% of total inference time, and a real optimization
will remove less. This is a useful bounded task, not the main battery solution.

## Lower-priority findings

- `LazyIm2Col::eval` rebuilds deferred geometry/offset arrays each execution in
  [`lazy_im2col.rs`](../third_party/tract/tract-core/src/ops/cnn/conv/lazy_im2col.rs).
  A bounded cache keyed by full geometry, dtype and packing could help repeated
  shapes or equivalent convolutions. A per-node last-shape cache will often miss
  when sentence lengths vary. Also, `Stage::run` calls `SimplePlan::run`, which
  constructs fresh execution state each time: an OpState cache alone would not
  persist between sentences. Measure this setup separately from GEMM time first.
- [`prepare_from_phonemes`](../src/lib.rs) reparses the constant vocabulary and
  rereads the whole voice file for each sentence. Cache the vocabulary and the
  voice data with a bounded policy. This is small next to inference, and a file
  read does not imply a physical flash read because of the OS page cache.
- The length regulator constructs a dense one-hot alignment matrix. Replacing
  the two alignment GEMMs with exact frame-index gathers removes unnecessary
  multiply/add work, but first measure those particular nodes. This short profile
  does not establish them as large costs. The current top-node report omits node
  names/IDs from its output; add them before using it to attribute these GEMMs.
- Scan remains FP32 and takes 16.8% of stage 1 in the indicative profile. Simply
  adding Scan to the FP16 allowlist is not a safe small change: the translator
  recurses with an unrestricted translator, and recurrence/duration accuracy
  matters. Consider mixed-precision inner GEMMs only after the vocoder work.

## Configuration and session-level savings

The prior [battery measurements](battery-and-heat.md) found no clear energy win
from reducing three gold threads to two, while latency worsened at the same
2.016 GHz cap. This is device/workload evidence, not a universal rule about thread
count. A smaller lookahead buffer chiefly saves abandoned work when a read stops;
it does not reduce the required model work for a complete listen.

An OS-supported lower CPU frequency/voltage point is worth an unplugged A/B test
if smooth playback has enough measured headroom. Lower frequency is distinct
from lowering thread count. [Linux CPUFreq documentation](https://docs.kernel.org/admin-guide/pm/cpufreq.html)
describes the frequency/voltage and performance/power tradeoff. Whether One UI
actually lowers this phone's existing cap must be checked on the device; this
review did not change it or establish a usable lower cap. Lower watts alone are
insufficient if the job takes proportionally longer.

For repeated short requests, `ryk --send` already avoids repeated model loading
and compilation. If using it, test `RYK_SINK_IDLE_MS=5000` when starting the daemon
instead of the 30-second default. This targets the idle audio tail, not active
inference. Existing daemons retain their startup setting. Preserve the held/open
sink during synthesis because the earlier screen-off experiment suspended there.
The remaining 200 ms hold poll can become a channel notification on release.
Do not count PulseAudio's process exit grace as an equal duration of active audio
hardware: its sink may suspend earlier, and this needs separate measurement.

## Validation before accepting a battery claim

1. Build a baseline from this exact commit; the existing executable is not a
   verified baseline. Compare one change at a time with identical compiler,
   assets, text, voice, speed and governor settings.
2. For fusion/Resize/packing changes, check finite output, exact rounded durations,
   and waveform parity where arithmetic is intended to remain identical. Include
   short/long inputs, multiple voices and listening checks for numerical changes.
3. Run alternating unplugged A/B trials with controlled screen/background load
   and matched starting temperature. Keep raw samples. Report joules/audio-second,
   CPU-seconds/audio-second, latency and gaps, rather than peak temperature alone.
4. Measure warm sustained reading separately from cold startup and post-playback
   idle. `tools/power_bench.sh` currently includes cold compilation and playback in
   its energy window but excludes compilation from wall RTF. That is a useful
   cold-session measure, not an isolated steady-inference measure.
5. Use several-minute listening runs to check the final result under sustained
   heat. Retain a playback margin; a short-run RTF of exactly 1.0 leaves little
   protection against scheduler or thermal variation.

Recommended order: establish the exact-build baseline, prototype AdaIN/activation/
Cast fusion, then benchmark FP16 kernel choices and small-operation thresholds.
Expect measured incremental improvements; there is no evidence here for a
dramatic battery-life multiplier with this same CPU model and quality.
