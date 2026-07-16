# kukuryku — pure-Rust CPU text-to-speech (Kokoro-82M via tract)

`ryk`, kukuryku's main binary, synthesizes speech on the CPU with **no native libraries** — no
onnxruntime, no `.so` to ship — by running Kokoro-82M through the pure-Rust
[tract](https://github.com/sonos/tract) inference engine. It is **faster than realtime** on a
desktop CPU and trivial to cross-compile, which makes it the backend of choice for
**Termux/aarch64** and other targets where an onnxruntime build is a pain.

## Quick start

Fresh checkout → spoken audio, the pure-Rust path (no onnxruntime). On Termux, swap
`sudo apt install -y` for `pkg install`.

```bash
# 1. System dependencies
sudo apt install -y espeak-ng                  # phonemizer — required
sudo apt install -y ffmpeg                     # playback via ffplay — optional: without it,
                                               # playback falls back to pacat (PulseAudio)

# 2. Build the pure-Rust binary.
#    First build compiles the whole dependency tree (the candle git fork + the vendored tract crates)
cargo build --release --bin ryk

# 3. Speak — from the project root, ryk finds kokoro-onyx/ on its own
./target/release/ryk "Hello, this is a pure-Rust text to speech test."
```

Step 3 assumes a self-contained `kokoro-onyx/` — `stage1.onnx` + `stage2.onnx` alongside
`model.onnx` and `voices/`. That is what lets it run bare: `ryk` resolves the model from
`kokoro-onyx/` and then looks for the stages next to it, so no `KOKORO_TRACT_DIR` is needed. `ryk`
cannot run the unsplit model, so the pair must be there — see
[Split the model into two stages](#split-the-model-into-two-stages-one-time) for what that means and
how to get it.

Each run prints a metrics line (`… tokens | … audio | infer …s | RTF …`); RTF below 1.0 is faster
than realtime. The sections below expand on each step.

## Prerequisites

- **Rust** 1.90+ (edition 2024) and a C toolchain (the tract build compiles a small C allocator).
- **espeak-ng** — phonemizer. `apt install espeak-ng` (or `pkg install espeak-ng` on Termux).
- **Audio playback** — one of:
  - **ffmpeg** (preferred, cross-platform) — playback shells out to `ffplay`.
    `apt install ffmpeg` (`pkg install ffmpeg` on desktop Linux/WSL).
  - **pulseaudio-utils** (fallback, used on Termux where ffplay is unavailable) — playback
    shells out to `pacat`. `pkg install pulseaudio` on Termux.

  Not needed if you only ever write WAVs (`KOKORO_WAV`).
- **A PulseAudio-compatible audio server** for playback (WSL2 provides this via WSLg; desktop
  Linux via PulseAudio/PipeWire; Termux via `pulseaudio --start` with `module-sles-sink`).
- **~650 MB disk** (the fp32 `model.onnx` ≈ 311 MB plus the two split subgraphs ≈ 311 MB),
  **~80 MB RAM** to run.
- **Python with `numpy` + `onnx`** — *only* to produce the split files once, and only if you
  can't copy an existing pair. Not a runtime dependency. See
  [Obtaining the split files](#obtaining-the-split-files).

`ryk` needs **no onnxruntime at all** — that is the whole point of the tract backend.

## Binaries

`cargo build --release` builds the first two; the others are behind cargo features.

| Binary | What it is | Build with |
|---|---|---|
| **`ryk`** | The main binary — Kokoro-82M on tract, pure Rust, no native libs. | *(default)* |
| `kokoro-tract` | **The same program as `ryk`**, under the name it had before the project became kukuryku. Kept so existing scripts and docs keep working. | *(default)* |
| `kokoro-onyx` | The same model on **onnxruntime** — the speed/quality reference the table below compares against. Needs an onnxruntime `.so` at runtime. | `--features onnx` |
| `speak-orpheus` | **Orpheus-3B** + SNAC on Candle. More natural, but ~10× slower than realtime. | `--features orpheus` |

`kokoro-onyx` and `speak-orpheus` are covered at the [bottom](#other-binaries-in-this-repo); the full
write-up for the tract work is in [`docs/tract-support-plan.md`](docs/tract-support-plan.md).
This branch (`tract-prototype`) is focused on `ryk`.

## How it compares to onnxruntime

To build it **alongside** the onnxruntime `kokoro-onyx` binary for side-by-side comparison (this also
pulls in `ort`, so it needs an onnxruntime `.so` at runtime — see the reference binary below):

```bash
cargo build --release --features onnx           # builds BOTH ryk and kokoro-onyx
```

Both backends run the identical pipeline and produce the same audio (waveform correlation
**~0.976**); they differ only in the inference engine. Measured on a 16-thread WSL2 box,
`af_heart`, two-sentence streamed run:

| Utterance | `ryk` (pure Rust) | `kokoro-onyx` (onnxruntime) |
|---|---|---|
| 242 tokens / 14.60 s audio | infer 7.39 s · **RTF 0.506** | infer 5.04 s · **RTF 0.345** |
| 221 tokens / 12.97 s audio | infer 6.60 s · **RTF 0.509** | infer 4.51 s · **RTF 0.347** |

Both are comfortably faster than realtime. tract is currently **~1.47× slower than onnxruntime**
(down from ~3.6× at the start of the optimization arc — see Tiers 1–7 in the plan doc). The
remaining gap is MLAS-class matmul-kernel work; onnxruntime's kernels are hard to beat. You trade
that ~1.5× for a **fully self-contained, dependency-free binary**.

## Split the model into two stages (one-time)

Tract cannot optimize Kokoro's **monolithic** graph: its length regulator expands phoneme-level
features to frame level via an alignment matrix whose frame-axis length is
`sum(round(durations))` — a *value*, not a static shape — which tract's shape inference can't
represent. `ryk` sidesteps this by **splitting the model at the length regulator** into
two subgraphs and rebuilding the alignment in Rust between them:

```
stage1.onnx : input_ids, style, speed → phoneme features [1,640,N] + [1,512,N] + durations [1,N]
   (Rust length regulator: round durations, build the [N, total_frames] alignment matrix)
stage2.onnx : the two feature tensors + alignment → decoder + iSTFTNet → waveform
```

So `ryk` cannot run the stock `model.onnx` — it needs the two subgraphs, and they are **not shipped
with the repo**. Getting them is a one-time step, described next.

### Obtaining the split files

`stage1.onnx` + `stage2.onnx` are fp32 and large (≈ 311 MB together) and don't meaningfully compress
(fp32 weights are near-incompressible), so they are kept out of git — the split step writes them
into the git-ignored **`kokoro-onyx/`** directory instead. They are just the original Kokoro weights
re-partitioned: nothing about them is machine-specific, so a pair produced anywhere works
everywhere. That gives you two ways to get them.

**Either** produce them yourself with the bundled script:

```bash
pip install numpy onnx                            # the script's only deps (no onnxruntime needed)
python3 tools/split_kokoro.py                     # writes kokoro-onyx/stage1.onnx + stage2.onnx
```

`numpy` + `onnx` are needed **only for this step** — they are build-time tooling for the split, not
a runtime dependency of `ryk`, which stays pure Rust. With no arguments the script reads the
HF-cached `onnx/model.onnx` for `onnx-community/Kokoro-82M-v1.0-ONNX` and writes the pair into the
project-local **`kokoro-onyx/`** directory (a stable path that lives with the checkout, instead of
the HF cache's snapshot-hashed dir). If your `model.onnx` lives somewhere else, pass an explicit
source/dest: `python3 tools/split_kokoro.py path/to/model.onnx [OUT_DIR]`.

**Or** copy an existing `stage1.onnx` + `stage2.onnx` into `kokoro-onyx/` — from another checkout,
another machine, or a colleague. No Python, no `pip install`, no HF download.

Either way, **point `ryk` at the dir with `KOKORO_TRACT_DIR=kokoro-onyx`** (it otherwise looks next
to the cached `model.onnx`):

```bash
KOKORO_TRACT_DIR=kokoro-onyx ./target/release/ryk "Hello world."
```

**Fully offline / self-contained.** If `kokoro-onyx/` also contains `model.onnx` and
`voices/<voice>.bin`, `ryk` uses those directly and **skips hf-hub entirely** — no network,
which is what makes an offline run work. In that case `KOKORO_TRACT_DIR` is optional when
you run from the project root: the local `model.onnx` makes `kokoro-onyx/` the default stage dir
too. The lookup dir is `KOKORO_TRACT_DIR` if set, else `./kokoro-onyx`; it falls back to the HF
cache when the local files aren't present. (The `voices/` dir is tiny — ~0.5 MB per voice.)

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
that window. `ryk` handles arbitrarily long text by **splitting the input into sentences**
(on `.!?;` and newlines; fragments merged, over-long runs wrapped on comma/word boundaries) and
synthesizing each as its own short utterance — always inside the window, and each with its own
clean prosody.

Playback streams with look-ahead buffering: one persistent `ffplay` plays sentences back-to-back
while the model works ahead. Since `ryk` is under realtime, its compute is masked behind
playback — first-audio latency is just model-load + the first sentence, and the rest is seamless.
(If you push it *over* realtime, e.g. on a slow phone CPU, you'll instead hear a short gap between
sentences while the next is synthesized.)

## Termux / Android (aarch64)

`ryk` is the intended Android backend precisely because it needs no native inference lib:

```bash
pkg install rust espeak-ng pulseaudio
cargo build --release --bin ryk
```

(Termux's `ffmpeg` package ships without `ffplay`, so playback there uses `pacat` from
`pulseaudio-utils`; the binary auto-selects whichever is on `PATH`.)

Provide the two split subgraphs (see [above](#obtaining-the-split-files)) in a directory,
point `KOKORO_TRACT_DIR` at it, and start PulseAudio (e.g. `module-sles-sink`) for playback or just
use `KOKORO_WAV`.

## How it works, fidelity, and performance

The full engineering log — the two-stage split, the Rust length regulator, the symbolic
compile-once plan, the vocoder atan2 branch-cut fix that took fidelity to ~0.976, and the Tier 1–7
run-speed arc (RTF 1.73 → ~0.50: lazy im2col, SIMD binary fusion, single-pass variance, Pad fold,
a mimalloc global allocator, `Square(Sin)`→`SinSq` fusion, and a vectorized `sin`) — is in
[`docs/tract-support-plan.md`](docs/tract-support-plan.md).

