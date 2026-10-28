# AdaIN / activation / Cast fusion investigation

2026-10-28, `main` at `7d34861`, with profiling-only changes. No fusion has been
implemented yet. This supersedes the uncertain binary provenance in the earlier
battery review: `~/bin/ryk` intentionally uses `playback-control`, while these
experiments use a fresh build from `main`.

## Finding

**The graph supports this optimization.** Every one of the 62 stage-2 AdaIN nodes
has exactly one consumer, an activation: 42 Snake and 20 LeakyRelu. Their
intermediates are not model outputs. The counts are identical across three
sentence lengths.

41 of those chains have eligible FP16 boundaries on both sides:

```text
FP16 tensor -> Cast(F32) -> AdaIN -> Snake/LeakyRelu -> Cast(F16)
```

The input Cast, AdaIN and activation each have a single consumer and are not
model outputs. The 41 cases comprise 30 Snake and 11 LeakyRelu chains. Another 18
chains start with FP32 data but have the removable output Cast; three have no
direct output Cast. Together, the eligible boundaries cover 100 of the current
331 stage-2 Cast nodes (82 in the 41 complete chains).

This is an opportunity to remove full-tensor intermediates and separate passes,
while retaining FP32 normalization/activation arithmetic. It is not a proposal
to perform normalization or Snake's sine polynomial in FP16.

## Fresh-build measurements

Built with `cargo build --release --locked --offline --bin ryk`, using Rust 1.99.0
and the repository's release settings (thin LTO, one codegen unit). Build passed
in 13m 36s, with the existing vendored tract-hir macro warning. The profiling
patch adds node names/IDs and optional JSON graph/tensor metadata; it changes no
inference operations. The binary was copied to:

`tools/bench_out/fusion-main-7d34861/ryk-main-profile`

SHA-256: `28fb8b8787c2400f868527ac4f64c3de0a1e3941570e129cccf711d394e2fe65`.
The installed `~/bin/ryk` was checked by SHA-256 before and after and is unchanged.

The corpus contains three sentences, increasing in length: “Hello there”, a
lighthouse/harbour sentence, and the long lighthouse-keeper benchmark sentence.
Voice `af_heart`, speed 1, FP16 default, three gold workers, stage 1 on the little
cores. `KOKORO_GOVERNOR=off` held the pool configuration constant for this short
investigation. The pre-run gold-core snapshot peaked at 51.3 C.

| Phoneme tokens | Audio | Stage-2 eval | All AdaIN + adjacent activation/boundary Casts | Of that, Casts | 41 complete chains, Casts only |
|---|---:|---:|---:|---:|---:|
| 11 | 1.48 s | 0.936 s | 0.099 s (10.6%) | 0.034 s (3.6%) | 0.031 s (3.3%) |
| 114 | 6.70 s | 4.021 s | 0.437 s (10.9%) | 0.158 s (3.9%) | 0.144 s (3.6%) |
| 242 | 14.62 s | 8.808 s | 1.035 s (11.7%) | 0.425 s (4.8%) | 0.369 s (4.2%) |

Percentages are of stage-2 eval time. These are **costs of existing regions, not
achievable savings**: normalization, activation and precision conversion still
have arithmetic to execute inside a fused implementation. Matrix multiplication
remains 72.5–74.5% of stage 2. A modest improvement is worth testing; these data
do not justify promising a 10% inference or battery improvement.

There is substantial tensor traffic to target. On the long sentence:

- Summed FP32 AdaIN activation sizes: **993.50 MiB** across 62 nodes, not simultaneous
  live memory. Removing one separate read/write pass over all these activations
  would eliminate approximately **1.94 GiB of logical tensor traffic**. This is
  not a DRAM-counter measurement; cache reuse affects external-memory traffic.
- The 41 complete FP16 chains account for **703.84 MiB** of that sum.
- The largest activation is `[1,128,70201]`, **34.28 MiB** in FP32. Its row is
  **274.22 KiB**. A reusable row buffer can avoid a full-sized FP32 intermediate,
  but do not assume every row fits a particular core's private cache. This is not
  a measured process-RSS reduction.

For example, the long profile contains this complete chain:

```text
#1426 Cast(F32)
  -> #1435 /decoder/decoder/generator/resblocks.5/adain2.0/Add_3 (AdaIN)
  -> #1436 Snake
  -> #1437 Cast(F16)
```

## Numerical constraints discovered during the investigation

The current [AdaIN implementation](../third_party/tract/tract-core/src/ops/math/mod.rs)
preserves the original reduction order. A fused implementation should retain:

1. The same `sum_f32` kernel and reduction grouping for the mean.
2. FP32 centering/squaring, eight FP64 accumulator lanes, and the current
   remainder and lane-reduction order for variance.
3. The existing affine association `((x - mean) * rs) * g + beta` and the existing
   Snake polynomial / LeakyRelu behavior, including boundary values.
4. Exact FP16 widening, and narrowing at the original output boundary, without
   inserting an extra FP16 rounding between normalization and activation.
5. The existing side-input precision: style scale, bias, reciprocal lengths and
   epsilon remain FP32.

**A row scratch buffer introduces a subtle extra requirement.**
[`reduce_slice_with_alignment`](../third_party/tract/tract-linalg/src/frame/element_wise_helper.rs)
splits the row into an alignment prefix, a vectorized middle, and a tail. The
AArch64 sum kernel uses 16-byte alignment and groups 16 elements at a time. Moving
each row to an always-aligned scratch address can change its FP32 summation order,
even when every input value is identical. Many measured row lengths end in 1,
so successive original FP32 rows have different offsets modulo 16 bytes.

When replacing a widened tensor with row scratch, preserve the virtual FP32 row's
alignment offset (or explicitly reproduce the original prefix/middle/tail
grouping). Calling the same sum kernel on an arbitrarily aligned new row is not
sufficient to guarantee parity. Scratch must also be reused per worker rather
than allocated once per channel.

## Proposed implementation sequence

1. **Fuse AdaIN and its activation in FP32 first.** Cover all 62 single-consumer
   pairs; execute the activation in the normalization's final apply loop. Keep
   the original input storage and reduction order. This removes the intervening
   activation read/write pass and one pool dispatch per pair, while keeping the
   existing Cast boundaries as a reference.
2. **Absorb the boundaries for the 41 complete chains.** Use FP16 input/output
   storage, FP32 internal values and the existing FP64 variance reduction. Use
   aligned, reusable row scratch as necessary. Extend to the other 18 output-only
   boundaries after validating the first path.
3. Match actual op types, tensor dtypes/shapes, single consumers and model-output
   constraints. Preserve FP32 users of a branched intermediate. Avoid changing
   the general FP16 allowlist; these mixed-storage operators need deliberate
   handling after precision translation.
4. Check raw FP32 output tensors and durations against the unfused baseline over
   several voices, lengths and speeds. Test all row-alignment offsets, tails,
   branching and model-output cases. Then measure uninstrumented CPU time and
   unplugged joules/audio-second in alternating A/B runs.

The first implementation can be evaluated independently; there is no need to
accept numerical changes merely to combine two passes.

## Validation and limits

- Profiled and unprofiled executions both completed successfully on all three
  lengths. The resulting **16-bit WAV files are byte-identical**, SHA-256
  `7784e1c7eff9e4bd82f9c6287681b11e806b070e680a6e3b4589ce95c7eaced7`.
  This validates the diagnostic path for this corpus; it is not a test of a
  future fused operator or proof of raw-FP32 equality.
- Unprofiled per-sentence infer RTF: **0.917 / 0.948 / 0.929**. Combined infer RTF
  0.934; wall RTF 0.723 from stage overlap. The deliberately increasing sentence
  lengths caused 6.84 s of reported gaps (7.52 s when profiled), so this corpus is
  a shape probe, not a demonstration of uninterrupted realtime playback.
- JSON timings cover op eval only. Metadata collection/serialization is outside
  the timers but can still perturb allocator/cache/scheduling behavior. Use an
  uninstrumented run for performance acceptance.
- One profiled pass, three lengths, one voice; no fused-kernel A/B test yet.
  The phone was charging during this investigation, so no battery-energy claim
  is made. No production inference logic was changed.
- `git diff --check` passed. All six JSON files parsed; per-node eval times sum
  to the recorded totals and FP16/FP32 tensor byte counts match their shapes.

## Reproduce and inspect

The new diagnostic switch is `KOKORO_TRACT_PROFILE_JSON=<directory>`. It enables
profiling and writes one JSON file per stage invocation, including node IDs,
names, ops, input/output dtypes, concrete shapes, bytes, edges and model-output
flags. Top-N text output now includes node IDs and names too.

```sh
KOKORO_VERBOSE=1 KOKORO_GOVERNOR=off \
KOKORO_TRACT_PROFILE_JSON=tools/bench_out/fusion-main-7d34861/profiles \
KOKORO_TRACT_PROFILE_NODES=12 \
KOKORO_WAV=tools/bench_out/fusion-main-7d34861/profiled.wav \
tools/bench_out/fusion-main-7d34861/ryk-main-profile \
  < tools/bench_out/fusion-main-7d34861/corpus.txt

python3 tools/analyze_fusion_profile.py \
  tools/bench_out/fusion-main-7d34861/profiles/stage2-*.json \
  --json tools/bench_out/fusion-main-7d34861/summary.json
```

Local raw artifacts are in the git-ignored
[`tools/bench_out/fusion-main-7d34861/`](../tools/bench_out/fusion-main-7d34861/):
build log, compiler/commit/binary hashes, profiling patch, corpus, six JSON
profiles, analysis, both run logs and both WAV files. The invocation also set
`KOKORO_TRACT_DIR` to the existing local model directory and a 120-second timeout.
