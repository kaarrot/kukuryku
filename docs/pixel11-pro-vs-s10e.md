# Pixel 11 Pro versus Galaxy S10e

Measured on 2026-11-18 at `e03d0e3`. The Pixel 11 Pro comfortably meets the
RTF < 1.0 target on the lighthouse test: mean infer RTF **0.306**, versus
**0.780** in the Galaxy S10e record, approximately **2.55× faster**. Mean
synthesis wall RTF is **0.263**, versus **0.672**. All three generated WAVs
match the migrated S10e reference sample for sample, and the user confirmed
that playback sounds normal.

A separate **144-second continuous playback** run measured infer RTF **0.534**
with zero reported gaps. Individual sentences ranged from 0.306 to 0.975 RTF,
so the sustained result includes more performance variation than the short test.

## Device and executable

| | Galaxy S10e | Pixel 11 Pro |
| --- | --- | --- |
| Reported model | Samsung SM-G970U | Pixel 11 Pro (`grizzly`) |
| SoC | Snapdragon 855 | Tensor G6 |
| Android | 12 / SDK 31 | 17 / SDK 37 |
| Stage-1 / playback CPU set by default | `0-3` | `0-1` |
| Stage-2 CPU set by default | `4-6`, three workers | `2-5`, four workers |
| Excluded singleton prime CPU | `7` | `6` |

Pixel identity, CPU capacities and topology come from this phone's `getprop`
and sysfs, rather than inferred specifications. Its three capacity groups are
627 (`0-1`), 800 (`2-5`) and 1024 (`6`). The existing automatic selection
correctly maps stage 1 to the lowest-capacity group and stage 2 to the four
middle cores. The Android stage-1 FP16 dense-kernel default remains the
existing A55 tile; no Pixel-specific inference tuning was introduced.

The primary comparison uses **the identical executable on both phones**:

```text
SHA-256 13287858ddbfc3178f0ac79b103e15edaeed732d2d911bdb387faf42fd0cd64f
```

This is the migrated `~/bin/ryk`, built for the application changes at
`8501a62`. The following `e03d0e3` commit changes documentation only.
It matches the executable recorded in
[the S10e validation](battery-validation-8501a62.md). Both runs use the split
Kokoro model, FP16, `af_heart`, `en-us` and speed 1.0. The recorded stage and
voice SHA-256 values are preserved with the results.

The Pixel was unplugged with Battery Saver off, as confirmed by the user.
Termux:API also reported `UNPLUGGED` / `DISCHARGING` before and after the
campaigns. Background apps and short-test screen state were not controlled.
For the continuous run, the user confirmed the screen stayed on in the same app. Release-test
compilation finished before timing began. The observer runs on CPU 0; private
benchmark daemons disable the headset helper, notifications and earcons.

## Identical-binary lighthouse comparison

The exact README two-sentence prompt produces 86 and 155 tokens and
**14.80 seconds** of audio on both phones. Measurements use real `pacat`
playback and also save the WAV, matching the S10e smoke-test procedure.
The Pixel has three independent process starts with a ten-second rest before
each. Each process recompiles its pipeline; filesystem caches were not cleared.
The historical S10e baseline is one recorded run, not a new repeated campaign.

| Metric | S10e recorded run | Pixel mean | Pixel range, three runs |
| --- | ---: | ---: | ---: |
| Inference time | 11.54 s | 4.53 s | 4.49–4.59 s |
| Infer RTF | 0.780 | **0.306** | 0.303–0.310 |
| Synthesis wall time | 9.95 s | 3.90 s | 3.87–3.94 s |
| Synthesis wall RTF | 0.672 | **0.263** | 0.261–0.266 |
| First-audio log, from synthesis start | 4.71 s | 1.80 s | 1.78–1.84 s |
| Entire command, including compilation and playback drain | 24.34 s | 18.61 s | 18.56–18.68 s |
| Reported synthesis gap | 0.00 s | 0.00 s | 0.00–0.00 s |

The 2.55× figure describes synthesis throughput on this prompt. Full command
completion improves about 24%, because playing 14.80 seconds of speech still
takes real time. First-audio values above exclude model compilation and do not
measure sound reaching the speaker. The Pixel observer saw the first-audio log
about 3.58 seconds after command start, including compilation, with nominal
100 ms polling resolution; no equivalent S10e observation was recorded.

The WAV check compares all **355,200 PCM16 samples**, mono at 24 kHz. Every
Pixel run has zero maximum sample difference against the migrated S10e WAV,
correlation 1.0, peak 0.6484 and RMS 0.07269. This verifies that the speedup did
not come from generating shorter, silent or different PCM for this test.

## Warm throughput with a matched worker count

The historical [Resize benchmark](profiling/resize-8f4cf95/README.md) supplies
six warm S10e trials of a fixed three-sentence, **22.80-second** corpus. The
Pixel repeats that text six times after a full-corpus warmup, with one stage-1
CPU, three stage-2 workers, FP16, the existing A55 dense tile, governor off and
a discard sink. Its CPU sets are mapped to this phone: stage 1 on `1`, stage 2
on `2-4`, instead of the S10e's `1` and `4-6`.

| Metric | S10e mean, six trials | Pixel mean, six trials | Pixel range |
| --- | ---: | ---: | ---: |
| Infer RTF | 0.790 | **0.390** | 0.381–0.403 |
| Synthesis wall RTF | 0.617 | **0.324** | 0.315–0.338 |
| Inference time | 18.02 s | 8.90 s | 8.70–9.18 s |
| Synthesis wall time | 14.07 s | 7.40 s | 7.19–7.70 s |
| Process / waited-child CPU time | 35.38 s | 18.71 s | 18.60–18.91 s |

That is approximately **2.02× faster inference** and **1.90× faster elapsed
generation** with the same worker count. This is a supporting comparison:
the S10e used the earlier Resize implementation later committed as `cd78298`,
before the daemon/preparation follow-ups, and was charging; the Pixel uses the
newer migrated binary while discharging. It is less tightly matched than the
identical-binary lighthouse comparison. Neither CPU time nor this discarded
audio run measures battery energy.

This deliberately uneven corpus begins with only 1.48 seconds of speech
("Hello there.") before much longer sentences. The Pixel reports an estimated
0.57–0.82-second gap at the next sentence despite average RTF below 1.0. This
shows why average RTF alone does not guarantee uninterrupted speech for every
sentence-length sequence. It is a synthesis-clock estimate with a discard
sink, not an observation of physical playback silence. The separate real
lighthouse and six-sentence playback trials report zero gaps.

## Default warm daemon with real playback

Three measured requests, after a full-text warmup, each generate **20.25
seconds** of audio from the six-sentence control-test passage. The default
Pixel topology and thermal/pacing settings are used; the sink grace is the
normal five seconds.

| Metric | Pixel mean | Range, three trials |
| --- | ---: | ---: |
| Infer RTF | **0.319** | 0.318–0.320 |
| Synthesis wall RTF | **0.258** | 0.257–0.259 |
| Inference time | 6.47 s | 6.45–6.49 s |
| Synthesis wall time | 5.23 s | 5.21–5.25 s |
| First-audio log from synthesis start | 1.06 s | 1.04–1.08 s |
| Reported gap | 0.00 s | 0.00–0.00 s |

The archived S10e first daemon request for this passage reports infer RTF
0.809 and wall RTF 0.647. It had no separate full-corpus warmup and its helper
was enabled, so it is context rather than a matched warm-daemon speed ratio.
The original committed prose gives wall RTF 0.644; the archived raw log gives
0.647, retained in the baseline evidence here.

## One long sentence

A separate warm test uses only the 242-token lighthouse sentence, generating
**14.62 seconds** of speech. With the Pixel's default four middle-core workers,
governor off and a discard sink, three trials give:

| Metric | Pixel mean | Range |
| --- | ---: | ---: |
| Infer RTF | **0.310** | 0.306–0.314 |
| Synthesis wall RTF | **0.315** | 0.311–0.319 |
| Time until the first audio chunk | 4.61 s | 4.55–4.67 s |

This exceeds realtime throughput without any cross-sentence overlap. It also
shows the remaining effect of sentence-sized synthesis: first audio waits for
the whole long sentence, even though its generation is much faster than its
playback duration. There is no matched historical single-sentence S10e trial
in this comparison.

## Continuous default playback

One private daemon request repeats the five-sentence lighthouse paragraph
eight times: **40 sentences / 144.00 seconds of audio**, with real `pacat`,
automatic core selection and the normal thermal/pacing governor. There is no
separate full-corpus warmup and no pause, rewind or cancel during measurement.

| Metric | Result |
| --- | ---: |
| Infer RTF | **0.534** |
| Inference time | 76.94 s |
| Paced synthesis wall RTF | 0.856 |
| Paced synthesis wall time | 123.25 s |
| First-audio log from synthesis start | 1.26 s |
| Sentence infer RTF range | 0.306–0.975 |
| Sentences at or above infer RTF 1.0 | **0 of 40** |
| Reported synthesis gap | **0.00 s** |

Each repeated paragraph generates about 18 seconds of audio. Its aggregate
infer RTF varies across the run:

| Paragraph | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Infer RTF | 0.323 | 0.330 | 0.338 | 0.653 | 0.772 | 0.849 | 0.588 | 0.425 |

Paragraph RTFs are derived from the rounded per-sentence log values.

A live observation in the slower stretch reported CPUs `2-5` at **1574 MHz**,
against a configured ceiling of **3379 MHz**. CPU performance varied during
playback; the cause was not isolated. The user confirmed the screen remained
on in the same app. No application thermal-pool switch was logged, and CPU
temperature was unavailable. This evidence does not establish thermal
throttling as the cause.

The normal pacing cap accounts for part of the 123.25-second generation
clock: a fast producer deliberately waits once roughly 20 seconds of audio
is queued. Whole-request observation through playback and sink closure took
150.22 seconds, including the five-second sink grace. The run met realtime
throughout despite its variation; it is one 2.4-minute playback test, not a
matched long-duration S10e experiment.

## Interpreting RTF

RTF is measured seconds divided by generated audio seconds; **below 1.0
means synthesis is faster than playback**.

- **Infer RTF** sums the two stages' inference durations, excluding espeak.
  Stage 1 and stage 2 overlap across sentences, so this sum differs from elapsed
  generation time.
- **Synthesis wall RTF** measures from the start of speaking through the last
  chunk arriving. It includes preparation, pipeline overlap, pacing and any
  playback backpressure, and excludes model compilation and final audio drain.
- **Whole-command time** additionally includes loading, compilation and audio
  drain. Dividing that by audio duration is not the synthesis RTF target.

With longer real playback, the default governor deliberately stops working
when roughly 20 seconds of speech is buffered. Its paced wall RTF can therefore
approach 1.0 while individual inference calls remain much faster than real
time. The discard-sink measurements disable that pacing to expose throughput.

## Validation and measurement limits

The fresh release library suite passed on the Pixel: **156 passed, zero
failed, one ignored**, in 1.27 seconds after compilation. The ignored case
builds the headset Java helper with real tools; physical headset and
notification controls were not tested in this performance comparison.

```sh
OPENSSL_DIR="$PREFIX" CARGO_BUILD_JOBS=2 cargo test --release --locked --offline --lib
```

The first build attempt could not find the existing Termux OpenSSL installation.
Setting `OPENSSL_DIR` to the Termux prefix resolved it without installing a
package or changing application source. Test metadata records the current Rust
toolchain; performance measurements deliberately retain the historical binary.

This Pixel does not expose readable CPU thermal zones to Termux. The app's
thermal hold consequently has no temperature signal, while Android's own
`sched_pixel` governor still manages the CPU. Battery temperature is a different
sensor from the S10e's old per-core temperature measurements. These tests do
not establish lower CPU temperature, power consumption or battery energy per
audio second.

The main comparison isolates application binary and output, while the default
worker count changes with hardware. Historical observations, different Android
versions, uncontrolled background activity and short trials limit claims about
universal hardware speedups. The additional worker-matched campaign uses an
older S10e inference benchmark and is reported separately.

## Reproduction and evidence

The runner is [`tools/bench_phone.py`](../tools/bench_phone.py). It records
binary/model/voice identities, exact text and environment, topology, per-run
timings, process CPU time, sampled RSS and reported CPU frequencies. It uses a
private socket and cleans up its own daemon. Detailed local logs, WAVs and
100 ms telemetry remain under `tools/bench_out/pixel11-pro-20261118/`.

For example, repeat the lighthouse test with the same binary and an unused
output directory:

```sh
python3 tools/bench_phone.py --binary "$HOME/bin/ryk" \
  --mode oneshot --sink real --repeats 3 --rest-secs 10 --wav \
  --text-file docs/profiling/pixel11-pro-20261118/lighthouse.txt \
  --out tools/bench_out/pixel11-repeat \
  --conditions 'Unplugged; Battery Saver off'
```

The portable result records and benchmark logs are in
[`docs/profiling/pixel11-pro-20261118/`](profiling/pixel11-pro-20261118/).
