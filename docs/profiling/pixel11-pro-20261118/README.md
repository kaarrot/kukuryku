# Pixel 11 Pro benchmark evidence, 2026-11-18

Interpretation: [Pixel 11 Pro versus Galaxy S10e](../../pixel11-pro-vs-s10e.md).
Runner: [`tools/bench_phone.py`](../../../tools/bench_phone.py).

- `results.json`: exact command, binary/model/voice hashes, hardware topology,
  environment, power observations, warmup policy and per-request results for
  five completed campaigns. Sixteen measured requests; warmups are excluded
  from the means. `source_commit` identifies the checked-out source; the
  migrated executable's identity is its SHA-256.
- `summary.json`: arithmetic means and observed ranges, comparison ratios and
  per-paragraph sustained inference RTF. The archived S10e lighthouse result
  has one observation, so these are not confidence intervals on a hardware
  speedup.
- `s10e-baselines.json`, `s10e-smoke.log`, `s10e-first-daemon-request.log`:
  migrated historical measurements, with provenance. The supporting warm
  baseline uses the B arms of the already-committed
  [`warm-bounded.json`](../resize-8f4cf95/warm-bounded.json).
- `smoke-*.log`: three fresh-process, real-playback lighthouse runs.
- `warm-three-workers-*.log`: one warmup and six three-worker, governor-off
  requests with a discard sink. CPU IDs map to the Pixel; the historical
  S10e corpus, stage-1 worker count and stage-2 worker count are retained.
- `warm-real-*.log`: one warmup and three default-topology, real-playback
  requests of the six-sentence passage.
- `long-single-*.log`: one warmup and three single-sentence requests, default
  topology, governor off and a discard sink.
- `sustained-*.log`: one 40-sentence real-playback request with the default
  pacing governor; `sustained-live-observation.json` records one CPU-frequency
  and Termux:API battery snapshot during its slower portion.
- `wav-validation.json`: all three lighthouse PCM16 outputs match the S10e
  reference sample for sample. WAV payloads are excluded from the repository.
- `test-validation.json`, `lib-tests.log`: the passing freshly built release
  library suite, compiler/lockfile identity and the Termux OpenSSL build setting.
- `human-feedback.json`: user-reported power, sound and screen conditions.
- `device.json`: directly reported model, device, SoC, Android release and SDK.
- The five `.txt` files preserve the exact input texts.

`results.json` preserves the invocation arguments. For reproduction, substitute
the text files here for their original `tools/bench_out/` locations and choose
unused output directories. Voice is `af_heart`, language `en-us`, speed 1.0.
The three-worker CPU set `2-4` and stage-1 CPU `1` are Pixel settings; the normal
default discovers the topology automatically. The discarded-audio experiment
reports estimated gaps for its tiny-first-sentence corpus; these are not
observations of physical playback silence.

CPU seconds, sampled RSS and reported CPU frequencies are not energy
measurements. CPU thermal sysfs was unavailable. Complete originals, WAVs and
nominal 100 ms CPU/frequency telemetry remain locally under
`tools/bench_out/pixel11-pro-20261118/`.
