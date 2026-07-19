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
cargo build --release --bin ryk

# 3. Download the model assets (~576 MB, once) — unpacks into the OS-specific per-user data dir
#    (Linux: ~/.local/share/kukuryku/kokoro-onyx/,
#     macOS: ~/Library/Application Support/kukuryku/kokoro-onyx/,
#     Windows: %APPDATA%\kukuryku\kokoro-onyx\)
./target/release/ryk --install-assets

# 4. Speak
./target/release/ryk "Hello, this is a pure-Rust text to speech test."
```

Step 3 fetches a pinned release archive, checks its sha256, and unpacks `stage1.onnx` +
`stage2.onnx` alongside `model.onnx` and `voices/`.
See
[Split the model into two stages](#split-the-model-into-two-stages-one-time) for why the split
exists and [Obtaining the split files](#obtaining-the-split-files) for the alternatives.

If you already have a `kokoro-onyx/` in the project root — from a previous checkout or your own
split — step 3 is unnecessary: `ryk` prefers it over the user-data-dir copy.

## Install (users, no clone)

Once you don't need a checkout, install the published binary and assets straight from
crates.io / GitHub — no `cargo build`, no `target/`:

```bash
# 1. System deps (same as the Quick start)
sudo apt install -y espeak-ng ffmpeg     # or: pkg install espeak-ng ffmpeg  (Termux)

# 2. The binary — cargo installs it into ~/.cargo/bin (put that on your PATH)
cargo install --locked --git https://github.com/kaarrot/kukuryku ryk

# 3. The ~576 MB model bundle — into the OS per-user data dir (see step 3 in Quick start)
ryk --install-assets

# 4. Speak
ryk "Hello from an installed ryk."
```

Why the split? `cargo install` copies the compiled binary into `~/.cargo/bin` and nothing else —
no post-install hook, no asset placement. So `--install-assets` is a one-time follow-up that
fetches the ~576 MB weight bundle into the OS's per-user data dir (`~/.local/share/kukuryku/…`
on Linux, `~/Library/Application Support/kukuryku/…` on macOS, `%APPDATA%\kukuryku\…` on
Windows). Re-running is a no-op once the stages are in place; delete the directory to force a
re-install.

Overrides:

- `ryk --install-assets --dev` — installs beside the running binary (`target/debug/kokoro-onyx/`
  on a `cargo run` checkout) instead of the user data dir. Use this when iterating on the
  install logic so you don't pollute your real `~/.local/share`.
- `KUKURYKU_ASSET_DIR=/some/path ryk --install-assets` — write the bundle at an explicit path
  (packagers, CI, Nix). `KUKURYKU_ASSET_DIR=exe` selects the exe-adjacent layout without
  hard-coding a path.

## Prerequisites

- **Rust** 1.90+ (edition 2024) and a C toolchain (the tract build compiles a small C allocator).
- **espeak-ng** — phonemizer. `apt install espeak-ng` (or `pkg install espeak-ng` on Termux).
- **Audio playback** — one of:
  - **ffmpeg** (preferred, cross-platform) — playback shells out to `ffplay`.
    `apt install ffmpeg`.
  - **pulseaudio-utils** (fallback, used on Termux where ffplay is unavailable) — playback
    shells out to `pacat`. `pkg install pulseaudio` on Termux. On Termux, `ryk` auto-starts one
    (`pulseaudio --start` with `module-sles-sink`) when it uses the `pacat` sink, so no manual
    step is needed.

## Binaries

`cargo build --release` builds the first two; the others are behind cargo features.

| Binary | What it is | Build with |
|---|---|---|
| **`ryk`** | The main binary — Kokoro-82M on tract, pure Rust, no native libs. | *(default)* |
| `kokoro-tract` | **The same program as `ryk`**, under the name it had before the project became kukuryku. Kept so existing scripts and docs keep working. | *(default)* |
| `kokoro-ort` | The same model on **onnxruntime** — the speed/quality reference the table below compares against. Named for the `ort` crate it wraps; the `kokoro-onyx` name is the assets directory, not a binary. Needs an onnxruntime `.so` at runtime. | `--features onnx` |
| `speak-orpheus` | **Orpheus-3B** + SNAC on Candle. More natural, but ~10× slower than realtime. | `--features orpheus` |

The full write-up for the tract work is in
[`docs/tract-support-plan.md`](docs/tract-support-plan.md). This branch (`tract-prototype`) is
focused on `ryk`.

## How it compares to onnxruntime

To build it **alongside** the onnxruntime `kokoro-ort` binary for side-by-side comparison (this also
pulls in `ort`, so it needs an onnxruntime `.so` at runtime — see the reference binary below):

```bash
cargo build --release --features onnx           # builds BOTH ryk and kokoro-ort
```

Both backends run the identical pipeline and produce the same audio (waveform correlation
**~0.976**); they differ only in the inference engine. Measured on a 16-thread box,
`af_heart`, two-sentence streamed run:

| Utterance | `ryk` (pure Rust) | `kokoro-ort` (onnxruntime) |
|---|---|---|
| 242 tokens / 14.60 s audio | infer 7.39 s · **RTF 0.506** | infer 5.04 s · **RTF 0.345** |
| 221 tokens / 12.97 s audio | infer 6.60 s · **RTF 0.509** | infer 4.51 s · **RTF 0.347** |

Both are comfortably faster than realtime. Tract is currently **~1.47× slower than onnxruntime**
(down from ~3.6× at the start of the optimization arc — see Tiers 1–7 in the plan doc). The
remaining gap is MLAS-class matmul-kernel work; onnxruntime's kernels are hard to beat. You trade
that ~1.5× for a **fully self-contained, dependency-free binary**.

## Split the model into two stages (one-time)

Tract cannot optimize Kokoro's **monolithic** graph: its length regulator expands phoneme-level
features to frame level via an alignment matrix whose frame-axis length is
`sum(round(durations))` — a *value*, not a static shape — which tract's shape inference can't
represent. `ryk` sidesteps this by **splitting the model at the length regulator** into
two subgraphs and rebuilding the alignment in Rust between them.

So `ryk` cannot run the stock `model.onnx` — it needs the two subgraphs, and they are **not shipped
with the repo**. Getting them is a one-time step, described next.

### Obtaining the split files

`stage1.onnx` + `stage2.onnx` are fp32 and large (≈ 325 MB together). They live in the git-ignored
**`kokoro-onyx/`** directory instead. They are just the original Kokoro weights re-partitioned:
nothing about them is machine-specific, so one published pair works on every target. Hence the
easy path — download it.

#### Download the published assets (recommended)

```bash
ryk --install-assets
```

Pulls `kokoro-onyx.zip` from the [releases page](https://github.com/kaarrot/kukuryku/releases),
verifies its sha256, and unpacks it into `kokoro-onyx/` inside the **OS-specific per-user data
dir** — `~/.local/share/kukuryku/` on Linux, `~/Library/Application Support/kukuryku/` on macOS,
`%APPDATA%\kukuryku\` on Windows. Pass `--dev` to install beside the running executable
instead (see the [Install](#install-users-no-clone) section). The archive carries `model.onnx`
too, which is what makes the unpacked directory work **fully offline**, and includes the
`af_heart` + `am_michael` voices.

It targets a pinned asset release (`kokoro-onyx-model`); override with `$KUKURYKU_ASSET_TAG` or
`$KUKURYKU_REPO`. The sha256 check is enforced only for the pinned tag — an overridden tag is a
different archive, so the pinned hash says nothing about it. Re-running is a no-op once the stages
are in place; delete the directory to force a re-install.

#### Split the model yourself

Needed if you want a voice the archive doesn't carry, or a Kokoro variant of your own:

```bash
pip install numpy onnx                            # the script's only deps (no onnxruntime needed)
python3 tools/split_kokoro.py                     # writes kokoro-onyx/stage1.onnx + stage2.onnx
```

`numpy` + `onnx` are needed **only for this step** — they are tooling for the split, not a runtime
dependency. With no arguments the script reads the HF-cached
`onnx/model.onnx` for `onnx-community/Kokoro-82M-v1.0-ONNX` and writes the pair into the
project-local **`kokoro-onyx/`** directory (a stable path that lives with the checkout, instead of
the HF cache's snapshot-hashed dir). If your `model.onnx` lives somewhere else, pass an explicit
source/dest: `python3 tools/split_kokoro.py path/to/model.onnx [OUT_DIR]`.

The dir is resolved in order:

1. `KOKORO_TRACT_DIR`, if set — always wins.
2. `dirs::data_dir()/kukuryku/kokoro-onyx` — the OS-specific per-user data dir where
   `--install-assets` unpacks the bundle. Standard install target for an installed `ryk`.
3. `./kokoro-onyx` — running from the project root, as in the quick start.
4. `kokoro-onyx/` beside the `ryk` executable — the `--install-assets --dev` target, for
   iterating on a checkout without polluting the real user data dir. Also serves as a
   last-resort arm so old installs keep working.

If none of these holds `stage1.onnx` + `stage2.onnx`, `ryk` errors and points you at
`ryk --install-assets` — the split stages are *not* on Hugging Face, so there is no useful
network fallback for them. Only a missing **voice** is fetched from the HF cache.

> Phonemization uses raw `espeak-ng` rather than Kokoro's reference phonemizer (misaki), so
> pronunciation is close but not identical on tricky words.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `KOKORO_VOICE` | `af_heart` | Voice (e.g. `am_michael`, `bf_emma`, …) |
| `KOKORO_LANG` | `en-us` | espeak-ng language |
| `KOKORO_SPEED` | `1.0` | Speaking rate |
| `KOKORO_WAV` | _(unset)_ | If set, write a 16-bit PCM WAV here instead of / in addition to playing |
| `KOKORO_TRACT_DIR` | _(auto; see above)_ | Directory holding `stage1.onnx` + `stage2.onnx` + `voices/` |
| `KOKORO_TRACT_THREADS` | _(all cores)_ | Thread-pool size for the stage-2 vocoder |
| `RYK_SOCKET` | `$XDG_RUNTIME_DIR/ryk.sock` | Daemon socket for `--serve`/`--send` (see below) |

(`KOKORO_MODEL` — the HF-repo path of the monolithic model — applies to `kokoro-ort` only;
`ryk` ignores it, since it runs the split stages, not `model.onnx`.)

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

## Low-latency editor use (`--serve` / `--send`)

Every plain `ryk` invocation compiles the two tract stages before the first sentence (~4s on a
desktop, more on a phone). Fine once; painful if you speak text repeatedly. The **warm daemon**
pays that cost once and keeps the compiled pipeline hot:

```bash
# The daemon auto-starts on the first --send and stays warm; no separate step needed.
echo "Hello from the warm daemon." | ryk --send
ryk --send "This one is near-instant."
```

`ryk --send` reads text from its arguments (or stdin) and hands it to the daemon, which
synthesizes and plays it. The first `--send` starts the daemon if it isn't running and waits for
it to warm up; every send after that returns immediately. Requests are **queued** and played in
order, gaplessly. Voice/lang/speed are read per request (from `KOKORO_VOICE` / `KOKORO_LANG` /
`KOKORO_SPEED` on the *client*), so you can switch voice without restarting the daemon. Run
`ryk --serve` yourself if you'd rather manage the daemon explicitly (foreground, or as a service).

The socket path is `$RYK_SOCKET`, else `$XDG_RUNTIME_DIR/ryk.sock`, else `/tmp/ryk-$USER.sock`;
an auto-started daemon logs beside it (`…/ryk.log`). This is **Unix-only**; elsewhere use the
one-shot form. Plain `ryk "text"` / stdin is unchanged and needs no daemon.

See [`docs/ryk-cli-and-daemon.md`](docs/ryk-cli-and-daemon.md) for the design and open follow-ups.

## Termux / Android (aarch64)

`ryk` is the intended Android backend precisely because it needs no native inference lib:

```bash
pkg install rust espeak-ng pulseaudio
cargo build --release --bin ryk
```

(Termux's `ffmpeg` package ships without `ffplay`, so playback there uses `pacat` from
`pulseaudio-utils`; the binary auto-selects whichever is on `PATH`.)

Provide the two split subgraphs (see [above](#obtaining-the-split-files)) in a directory and
point `KOKORO_TRACT_DIR` at it. When it falls back to `pacat`, `ryk` **auto-starts PulseAudio**
(`pulseaudio --start`, loading `module-sles-sink` on Android; add args via `RYK_PULSE_ARGS`) if
none is running — so playback works without a manual `pulseaudio --start`, which matters for the
detached `--serve` daemon. Or just use `KOKORO_WAV`. (The `ffplay` path, used on desktop, instead
relies on the audio server your session already runs — PulseAudio, PipeWire, or ALSA via SDL.)

## How it works, fidelity, and performance

The full engineering log — the two-stage split, the Rust length regulator, the symbolic
compile-once plan, the vocoder atan2 branch-cut fix that took fidelity to ~0.976, and the Tier 1–7
run-speed arc (RTF 1.73 → ~0.50: lazy im2col, SIMD binary fusion, single-pass variance, Pad fold,
a mimalloc global allocator, `Square(Sin)`→`SinSq` fusion, and a vectorized `sin`) — is in
[`docs/tract-support-plan.md`](docs/tract-support-plan.md).

