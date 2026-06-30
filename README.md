# speak — CPU text-to-speech on Candle (Orpheus-3B + SNAC)

A small command-line tool that reads text aloud using a fully local, CPU-only pipeline:

```
text → Orpheus-3B (quantized_llama GGUF) → SNAC audio tokens
     → SNAC 24 kHz codec → f32 PCM → played aloud via ffplay
```

Orpheus is a Llama-3.2-3B fine-tune that emits SNAC audio tokens; the SNAC codec turns them into a
24 kHz waveform. Because it *is* a Llama-3.2-3B model, it runs through Candle's `quantized_llama`
GGUF decode path — and through the [kaarrot CPU-decode fork](https://github.com/kaarrot/candle/tree/feature/cpu-decode-optim-squashed)
this project is wired to, which gives a ~3.4× decode speedup on this workload.

## Prerequisites

- **Rust** (1.90+; edition 2024).
- **ffmpeg** — playback shells out to `ffplay`. Install with `apt install ffmpeg` (or `pkg install
  ffmpeg` on Termux).
- **A PulseAudio-compatible audio server.** On WSL2 this is provided automatically by **WSLg**
  (`PULSE_SERVER=unix:/mnt/wslg/PulseServer`). On bare desktop Linux it's PulseAudio or PipeWire.
- **An x86_64 CPU with AVX2 + FMA** to get the fast decode path (any modern CPU). Without it the
  tool still works, just on the slower fallback path.
- **~3 GB free disk** for the model weights, and **~3.5 GB RAM** to run.

## Build

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release
```

`target-cpu=native` is what enables the fork's AVX2 SIMD kernels. The first build compiles the full
Candle stack from the fork (a few minutes); subsequent builds are seconds.

## Run

```bash
CANDLE_CPU_DECODE_EXECUTOR=1 ./target/release/speak "Hello there, this is a local text to speech test."
```

- Text comes from the command-line arguments, or from **stdin** if none are given
  (`echo "hi" | speak`).
- `CANDLE_CPU_DECODE_EXECUTOR=1` turns on the fast decode path. Omit it and you silently get the
  slow tensor path — the tool prints `decode fast path: engaged` vs `NOT engaged` so you can tell.
- Generation is slower than realtime on CPU (~×0.10), so a ~3 s clip takes ~30–45 s to synthesize
  before it plays. This is expected; the printed realtime factor makes it measurable.

You do **not** need to run `ffplay` yourself — `speak` spawns it internally and pipes the audio to
it.

## Model assets — downloaded automatically

**There is no manual download step.** On first run the tool fetches everything via `hf-hub` and
caches it under `~/.cache/huggingface/hub` (override with `HF_HOME`). Subsequent runs are offline
from cache.

| Asset | HF repo | File | Size |
|-------|---------|------|------|
| Orpheus GGUF (default) | `dahara1/orpheus-3b-0.1-ft_gguf` | `orpheus-3b-Q4_K_L.gguf` | ~2.3 GB |
| Tokenizer | `unsloth/orpheus-3b-0.1-ft` | `tokenizer.json` | small |
| SNAC weights | `lmz/candle-snac` | `snac_24khz.safetensors` | small |
| SNAC config | `hubertsiuzdak/snac_24khz` | `config.json` | tiny |

> The tokenizer comes from the public **unsloth** mirror because the canonical
> `canopylabs/orpheus-3b-0.1-ft` repo is **gated** (HTTP 401 without an HF license token). If you
> have accepted that model's terms, point at it with `SPEAK_TOKENIZER_REPO=canopylabs/orpheus-3b-0.1-ft`.

### Pre-fetching or swapping the GGUF (optional)

To download the weights ahead of time (so the first `speak` run is instant), or to try a different
quantization, use the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download dahara1/orpheus-3b-0.1-ft_gguf orpheus-3b-Q4_K_L.gguf
```

It lands in the same `~/.cache/huggingface` cache the tool reads from. Available quants in that repo
(pick one and pass it via `SPEAK_MODEL`):

```
orpheus-3b-Q3_K_L.gguf   orpheus-3b-Q4_K-f16.gguf  orpheus-3b-Q4_K_L.gguf  (default)
orpheus-3b-Q5_K_L.gguf   orpheus-3b-Q6_K-f16.gguf  orpheus-3b-Q6_K_L.gguf  orpheus-3b-Q8_0.gguf
```

Note there is **no `Q4_K_M`** in this repo, which is why the default is `Q4_K_L`. Lower-bit quants
(Q3) move fewer bytes per token and can be marginally faster on this memory-bound workload, at some
quality cost; higher-bit quants (Q6/Q8) sound a touch better and are slower.

```bash
SPEAK_MODEL=orpheus-3b-Q6_K_L.gguf CANDLE_CPU_DECODE_EXECUTOR=1 ./target/release/speak "higher quality take"
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SPEAK_VOICE` | `tara` | Voice: `tara leah jess leo dan mia zac zoe` |
| `SPEAK_MODEL` | `orpheus-3b-Q4_K_L.gguf` | GGUF filename in the model repo |
| `SPEAK_MAX_TOKENS` | `1200` | Hard cap on generated tokens (~8 s of audio) |
| `SPEAK_TEMP` | `0.6` | Sampling temperature |
| `SPEAK_SEED` | `299792458` | RNG seed |
| `SPEAK_WAV` | _(unset)_ | If set, also write a 16-bit PCM WAV to this path |
| `SPEAK_STREAM` | _(unset)_ | If set, stream audio as it's generated (low latency, but choppy — see below) |
| `SPEAK_TOKENIZER_REPO` | `unsloth/orpheus-3b-0.1-ft` | Override the tokenizer source repo |

### Latency vs. smoothness (SPEAK_STREAM)

By default `speak` generates the whole utterance, then plays it as one smooth
buffer — so you wait the full generation time (~17 s load + ~N s decode) before
any sound, but playback is clean.

`SPEAK_STREAM=1` instead pipes audio to `ffplay` as each chunk is decoded, so you
hear the first words a few seconds into generation. **The catch:** CPU generation
is ~10× slower than realtime, so `ffplay` drains each chunk and then waits for the
next — the voice **breaks up** (audible gaps). This is buffer underrun, not a bug;
it's fundamental to sub-realtime generation. Use streaming only when you want the
fastest possible feedback and can tolerate choppy audio. For smooth *and* fast you
need faster generation (a smaller model or a GPU), not a setting.

Fork runtime gates (advanced): `CANDLE_CPU_DECODE_EXECUTOR=1` (required for the fast path),
`CANDLE_MATVEC_THREADS=N` (matvec pool size; defaults to 4 and plateaus there — the workload is
memory-bandwidth bound), `CANDLE_CPU_F16_KV_CACHE=1` (halves KV cache memory).

## Examples

```bash
# Different voice
SPEAK_VOICE=leo CANDLE_CPU_DECODE_EXECUTOR=1 ./target/release/speak "I can sound different too."

# Save a WAV (useful on a headless box to verify synthesis independent of playback)
SPEAK_WAV=out.wav CANDLE_CPU_DECODE_EXECUTOR=1 ./target/release/speak "writing a wav file"

# From stdin
echo "piping text in" | CANDLE_CPU_DECODE_EXECUTOR=1 ./target/release/speak
```

## Performance

Decode throughput on a 16-core WSL2 box (Orpheus-3B Q4_K_L, seed 42, 200 tokens):

| Build / config | Decode | Realtime factor |
|----------------|--------|-----------------|
| Stock candle, no `target-cpu=native` | ~1.8 t/s | ×0.02 |
| Stock candle + `target-cpu=native` | 2.5 t/s | ×0.03 |
| **Fork + native + executor** | **8.6 t/s** | **×0.10** |

The fork is ~3.4× faster than the native-stock baseline. Realtime TTS would need ~82 t/s (SNAC
emits ~82 audio tokens per second of speech), so even optimized this is ~8–10× from live — it makes
CPU TTS *usable*, not realtime.

## Troubleshooting

- **No sound / instant finish** — if the metrics line shows a tiny token count (e.g. `15 tokens |
  0.17s`), the model stopped early and there was nothing to play. Make sure you're on a current
  build (the prompt must include the BOS token). If audio is generated (multi-second) but you still
  hear nothing, check your OS/WSLg output device and volume — the synthesis is fine. Pass
  `SPEAK_WAV=out.wav` and inspect the file to confirm.
- **`decode fast path: NOT engaged`** — you're on the slow path. Build with
  `RUSTFLAGS='-C target-cpu=native'` and run with `CANDLE_CPU_DECODE_EXECUTOR=1` on an AVX2 host.
- **Too quiet** — voice loudness varies per take; re-run or nudge `SPEAK_TEMP`.
- **401 fetching the tokenizer** — the canonical repo is gated; the default unsloth mirror avoids
  this. Set `SPEAK_TOKENIZER_REPO` only if you intend to use a gated repo with an HF token.

## How it works

See [`plan.md`](plan.md) for the full design rationale and the model-options comparison. The
pipeline mirrors Candle's `candle-examples/examples/orpheus` (prompt format, token→SNAC parsing,
SNAC decode), adapted to load **GGUF** via `quantized_llama` and to play aloud via `ffplay` instead
of writing a WAV.
