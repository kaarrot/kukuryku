# kokoro-tract — pure-Rust CPU text-to-speech (Kokoro-82M via tract)

`kokoro-tract` synthesizes speech on the CPU with **no native libraries** — no onnxruntime,
no `.so` to ship — by running Kokoro-82M through the pure-Rust [tract](https://github.com/sonos/tract)
inference engine. It is **faster than realtime** on a desktop CPU and trivial to cross-compile,
which makes it the backend of choice for **Termux/aarch64** and other targets where an
onnxruntime build is a pain.

```
text → espeak-ng IPA phonemes → phoneme-id tokens
     → tract (Kokoro-82M, two-stage split + Rust length regulator, + per-voice style vector)
     → 24 kHz waveform → ffplay (or a .wav)
```

> This branch (`tract-prototype`) is focused on `kokoro-tract`. The repo also contains two other
> binaries — `kokoro` (the same model on **onnxruntime**, used as the speed/quality reference) and
> `speak-orpheus` (Orpheus-3B, more natural but ~10× slower than realtime). They are summarized at the
> [bottom](#other-binaries-in-this-repo); the full write-up for the tract work is in
> [`docs/tract-support-plan.md`](docs/tract-support-plan.md).

## How it compares to onnxruntime

Both backends run the identical pipeline and produce the same audio (waveform correlation
**~0.976**); they differ only in the inference engine. Measured on a 16-thread WSL2 box,
`af_heart`, two-sentence streamed run:

| Utterance | `kokoro-tract` (pure Rust) | `kokoro` (onnxruntime) |
|---|---|---|
| 242 tokens / 14.60 s audio | infer 7.39 s · **RTF 0.506** | infer 5.04 s · **RTF 0.345** |
| 221 tokens / 12.97 s audio | infer 6.60 s · **RTF 0.509** | infer 4.51 s · **RTF 0.347** |

Both are comfortably faster than realtime. tract is currently **~1.47× slower than onnxruntime**
(down from ~3.6× at the start of the optimization arc — see Tiers 1–7 in the plan doc). The
remaining gap is MLAS-class matmul-kernel work; onnxruntime's kernels are hard to beat. You trade
that ~1.5× for a **fully self-contained, dependency-free binary**.

## Quick start

Fresh checkout → spoken audio, the pure-Rust path (no onnxruntime). On Termux, swap
`sudo apt install -y` for `pkg install`.

```bash
# 1. System dependencies
sudo apt install -y espeak-ng ffmpeg          # phonemizer + ffplay playback
pip install numpy onnx                         # only for the one-time model split (step 3)

# 2. Build the pure-Rust binary.
#    First build compiles the whole dependency tree (the candle git fork + the vendored
#    tract crates) — several minutes on a clean cache; later builds are seconds.
cargo build --release --bin kokoro-tract

# 3. One-time: fetch Kokoro-82M, then split it into the two tract stages (-> kokoro-onyx/).
./target/release/kokoro-tract "warm up"        # first run downloads the model into the HF cache,
                                               # then reports the split stages are missing — that
                                               # download is all this step needs
python3 tools/split_kokoro.py                  # writes kokoro-onyx/stage1.onnx + stage2.onnx

# 4. Speak (point kokoro-tract at the local split dir)
KOKORO_TRACT_DIR=kokoro-onyx ./target/release/kokoro-tract \
    "Hello, this is a pure-Rust text to speech test."
```

Each run prints a metrics line (`… tokens | … audio | infer …s | RTF …`); RTF below 1.0 is faster
than realtime. The sections below expand on each step.

## Prerequisites

- **Rust** 1.90+ (edition 2024) and a C toolchain (the tract build compiles a small C allocator).
- **espeak-ng** — phonemizer. `apt install espeak-ng` (or `pkg install espeak-ng` on Termux).
- **ffmpeg** — playback shells out to `ffplay`. `apt install ffmpeg` (`pkg install ffmpeg` on
  Termux). Not needed if you only ever write WAVs (`KOKORO_WAV`).
- **A PulseAudio-compatible audio server** for playback (WSL2 provides this via WSLg; desktop
  Linux via PulseAudio/PipeWire).
- **~650 MB disk** (the fp32 `model.onnx` ≈ 311 MB plus the two split subgraphs ≈ 311 MB),
  **~80 MB RAM** to run.

`kokoro-tract` needs **no onnxruntime at all** — that is the whole point of the tract backend.

## Build (tract only)

```bash
# Pure-Rust, no onnxruntime linked — the Termux/portable build:
cargo build --release --bin kokoro-tract
```

The first build (clean cache) compiles the full dependency tree — the vendored tract crates plus
the candle git fork, which `speak_tts` currently depends on unconditionally even though
`kokoro-tract` doesn't use it — so budget several minutes; later builds are seconds. The binary
lands at `target/release/kokoro-tract`.

To build it **alongside** the onnxruntime `kokoro` binary for side-by-side comparison (this also
pulls in `ort`, so it needs an onnxruntime `.so` at runtime — see the reference binary below):

```bash
cargo build --release --features onnx           # builds BOTH kokoro-tract and kokoro
```

## Split the model into two stages (one-time)

tract cannot optimize Kokoro's **monolithic** graph: its length regulator expands phoneme-level
features to frame level via an alignment matrix whose frame-axis length is
`sum(round(durations))` — a *value*, not a static shape — which tract's shape inference can't
represent. `kokoro-tract` sidesteps this by **splitting the model at the length regulator** into
two subgraphs and rebuilding the alignment in Rust between them:

```
stage1.onnx : input_ids, style, speed → phoneme features [1,640,N] + [1,512,N] + durations [1,N]
   (Rust length regulator: round durations, build the [N, total_frames] alignment matrix)
stage2.onnx : the two feature tensors + alignment → decoder + iSTFTNet → waveform
```

Produce the two subgraphs once with the bundled script:

```bash
pip install numpy onnx                            # the script's only deps (no onnxruntime needed)
python3 tools/split_kokoro.py                     # writes kokoro-onyx/stage1.onnx + stage2.onnx
./target/release/kokoro-tract "Hello world."      # then run pointing at the local dir:
KOKORO_TRACT_DIR=kokoro-onyx ./target/release/kokoro-tract "Hello world."
```

- With no arguments it reads the HF-cached `onnx/model.onnx` for
  `onnx-community/Kokoro-82M-v1.0-ONNX` and writes `stage1.onnx` + `stage2.onnx` into the
  project-local **`kokoro-onyx/`** directory (a stable path that lives with the checkout, instead of
  the HF cache's snapshot-hashed dir). Run `kokoro-tract` once first to trigger the model download,
  or pass an explicit source/dest: `python3 tools/split_kokoro.py path/to/model.onnx [OUT_DIR]`.
- **Point `kokoro-tract` at that dir with `KOKORO_TRACT_DIR=kokoro-onyx`** (it otherwise looks next
  to the cached `model.onnx`). `kokoro-onyx/` is git-ignored — see below.

**Fully offline / self-contained.** If `kokoro-onyx/` also contains `model.onnx` and
`voices/<voice>.bin`, `kokoro-tract` uses those directly and **skips hf-hub entirely** — no network,
which is what makes an offline or Termux run work. In that case `KOKORO_TRACT_DIR` is optional when
you run from the project root: the local `model.onnx` makes `kokoro-onyx/` the default stage dir
too. The lookup dir is `KOKORO_TRACT_DIR` if set, else `./kokoro-onyx`; it falls back to the HF
cache when the local files aren't present. (The `voices/` dir is tiny — ~0.5 MB per voice.)

> **Skipping the split:** the subgraphs only depend on the model weights, not on your machine, so
> you can produce them once (or copy an existing `stage1.onnx` + `stage2.onnx`) into `kokoro-onyx/`
> and reuse them across checkouts — no need to re-run the Python step or hunt through the HF cache.

### The split files are not committed

`stage1.onnx` + `stage2.onnx` are fp32 and large (≈ 311 MB together) and don't meaningfully compress
(fp32 weights are near-incompressible), so they're kept out of the repo — the split step writes them
into the git-ignored **`kokoro-onyx/`** directory instead. Produce them once with
`tools/split_kokoro.py` (or drop in an existing pair) and keep `KOKORO_TRACT_DIR=kokoro-onyx` set.
The subgraphs are just the original weights re-partitioned, so a pair produced on any machine works
anywhere.

## Run

```bash
./target/release/kokoro-tract "what is for lunch today?"
KOKORO_VOICE=am_michael ./target/release/kokoro-tract "I could go for some pizza."

# Headless / verify without speakers — write a WAV instead of playing:
KOKORO_WAV=/tmp/k.wav ./target/release/kokoro-tract "testing one two three"
```

Text comes from the CLI arguments, or from **stdin** if none are given
(`echo "hi" | kokoro-tract`). Each run prints a metrics line (`… tokens | … audio | infer …s |
RTF …`); RTF below 1.0 means faster than realtime.

> Phonemization uses raw `espeak-ng` rather than Kokoro's reference phonemizer (misaki), so
> pronunciation is close but not identical on tricky words.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `KOKORO_VOICE` | `af_heart` | Voice (e.g. `am_michael`, `bf_emma`, …) |
| `KOKORO_MODEL` | `onnx/model.onnx` | Model file in the HF repo (fp32; used to locate the split dir) |
| `KOKORO_LANG` | `en-us` | espeak-ng language |
| `KOKORO_SPEED` | `1.0` | Speaking rate |
| `KOKORO_WAV` | _(unset)_ | If set, write a 16-bit PCM WAV here instead of / in addition to playing |
| `KOKORO_TRACT_DIR` | _(next to `model.onnx`)_ | Directory holding `stage1.onnx` + `stage2.onnx` |
| `KOKORO_TRACT_THREADS` | _(all cores)_ | Thread-pool size for the stage-2 vocoder |

Diagnostics (rarely needed): `KOKORO_TRACT_PROFILE=1` prints a per-op stage-2 profile,
`KOKORO_TRACT_PROFILE_NODES=N` the top-N individual nodes, `KOKORO_TRACT_DUMP=dir` dumps
stage-boundary tensors. `tools/bench_conv.sh <label>` runs a fixed-sentence best-of-N timing +
profile for A/B work.

### Long input, and streaming across sentences

Kokoro-82M has a **fixed ~510-phoneme context** (`MAX_PHONEMES` in `src/lib.rs`). Because the model
is **non-autoregressive** (it predicts the whole utterance in one pass), it cannot "continue" past
that window. `kokoro-tract` handles arbitrarily long text by **splitting the input into sentences**
(on `.!?;` and newlines; fragments merged, over-long runs wrapped on comma/word boundaries) and
synthesizing each as its own short utterance — always inside the window, and each with its own
clean prosody.

Playback streams with look-ahead buffering: one persistent `ffplay` plays sentences back-to-back
while the model works ahead. Since `kokoro-tract` is under realtime, its compute is masked behind
playback — first-audio latency is just model-load + the first sentence, and the rest is seamless.
(If you push it *over* realtime, e.g. on a slow phone CPU, you'll instead hear a short gap between
sentences while the next is synthesized.)

## Termux / Android (aarch64)

`kokoro-tract` is the intended Android backend precisely because it needs no native inference lib:

```bash
pkg install rust espeak-ng ffmpeg
cargo build --release --bin kokoro-tract
```

Provide the two split subgraphs (see [above](#the-split-files-are-not-committed)) in a directory,
point `KOKORO_TRACT_DIR` at it, and start PulseAudio (e.g. `module-sles-sink`) for playback or just
use `KOKORO_WAV`.

## How it works, fidelity, and performance

The full engineering log — the two-stage split, the Rust length regulator, the symbolic
compile-once plan, the vocoder atan2 branch-cut fix that took fidelity to ~0.976, and the Tier 1–7
run-speed arc (RTF 1.73 → ~0.50: lazy im2col, SIMD binary fusion, single-pass variance, Pad fold,
a mimalloc global allocator, `Square(Sin)`→`SinSq` fusion, and a vectorized `sin`) — is in
[`docs/tract-support-plan.md`](docs/tract-support-plan.md).

## Other binaries in this repo

All three share the sentence-splitting/streaming front-end but target different models/engines. Each
is gated behind a cargo feature; `cargo build --release` (default `tract`) builds only `kokoro-tract`.

| Binary | Model | Engine | Cargo feature |
|---|---|---|---|
| `kokoro-tract` | Kokoro-82M | tract (pure Rust) | `tract` *(default)* |
| `kokoro` | Kokoro-82M | onnxruntime (`ort`) | `onnx` |
| `speak-orpheus` | Orpheus-3B + SNAC | Candle | `orpheus` |

- **`kokoro`** — the *same* Kokoro-82M pipeline on **onnxruntime** (`ort` `load-dynamic`; the binary
  dlopens an onnxruntime `.so`, e.g. from `pip install onnxruntime`). This is the speed/quality
  reference the table above compares against.
  ```bash
  pip install onnxruntime                        # provides libonnxruntime.so (ORT_DYLIB_PATH to override)
  cargo build --release --features onnx --bin kokoro
  ./target/release/kokoro "Hello world."
  ```
- **`speak-orpheus`** — **Orpheus-3B** (a Llama-3.2-3B fine-tune) + SNAC codec via Candle. Most natural
  voice, but ~10× slower than realtime on CPU. Opt-in (`--features orpheus`) since it pulls the heavy
  candle git-fork dependency tree. Needs `RUSTFLAGS='-C target-cpu=native'` for its AVX2 decode kernels
  and `CANDLE_CPU_DECODE_EXECUTOR=1` at runtime. See [`plan.md`](plan.md) for its design.
  ```bash
  cargo build --release --features orpheus --bin speak-orpheus
  ./target/release/speak-orpheus "Hello world."
  ```

