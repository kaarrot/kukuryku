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

# 3. Speak. The first run downloads the model (~576 MB, once) into the OS-specific per-user
#    data dir, then speaks. Later runs use that copy and need no network.
#    (Linux: ~/.local/share/kukuryku/kokoro-onyx/,
#     macOS: ~/Library/Application Support/kukuryku/kokoro-onyx/,
#     Windows: %APPDATA%\kukuryku\kokoro-onyx\)
./target/release/ryk "Hello, this is a pure-Rust text to speech test."
```

The first run fetches a pinned release archive, checks its sha256, and unpacks `stage1.onnx` +
`stage2.onnx` alongside `model.onnx` and `voices/`. `ryk --install-assets` does the same fetch
ahead of time, without speaking.
See
[Split the model into two stages](#split-the-model-into-two-stages-one-time) for why the split
exists and [Obtaining the split files](#obtaining-the-split-files) for the alternatives.

If you already have a `kokoro-onyx/` in the project root — from a previous checkout or your own
split — `ryk` uses it and downloads nothing, as long as the user data dir has no bundle of its own.

## Install (users, no clone)

Once you don't need a checkout, install the published binary and assets straight from
crates.io / GitHub — no `cargo build`, no `target/`:

```bash
# 1. System deps (same as the Quick start)
sudo apt install -y espeak-ng ffmpeg     # or: pkg install espeak-ng ffmpeg  (Termux)

# 2. The binary — cargo installs it into ~/.cargo/bin (put that on your PATH)
cargo install --locked --git https://github.com/kaarrot/kukuryku ryk

# 3. Speak. The first run downloads the ~576 MB model bundle into the OS per-user data dir.
ryk "Hello from an installed ryk."
```

`cargo install` copies the compiled binary into `~/.cargo/bin` and nothing else — no
post-install hook, no asset placement. So `ryk` places the assets itself: whenever it needs the
model and finds no local copy, it fetches the ~576 MB weight bundle into the OS's per-user data
dir (`~/.local/share/kukuryku/…` on Linux, `~/Library/Application Support/kukuryku/…` on macOS,
`%APPDATA%\kukuryku\…` on Windows), then carries on. That covers one-shot `ryk`, `ryk --serve`
and `ryk --send`. `--send` downloads in the foreground, before it starts the daemon, so the
progress shows in your terminal. Two `ryk`s started at once on a fresh machine download once:
the second waits for the first. An interrupted download or unzip is never mistaken for a
complete bundle; the next run starts it again. Delete the directory to force a re-install.

`ryk --install-assets` runs the same fetch on its own: to pre-load a machine before you go
offline, or to keep the download out of the first speak.

Overrides:

- `ryk --install-assets --dev` — installs beside the running binary (`target/debug/kokoro-onyx/`
  on a `cargo run` checkout) instead of the user data dir. Use this when iterating on the
  install logic so you don't pollute your real `~/.local/share`.
- `KUKURYKU_ASSET_DIR=/some/path ryk --install-assets` — write the bundle at an explicit path
  (packagers, CI, Nix). `ryk` also looks there first, and downloads there on first use. `KUKURYKU_ASSET_DIR=exe` selects the exe-adjacent layout without
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
    step is needed. PulseAudio is **not** started with `--exit-idle-time=-1`; once `pacat`
    disconnects it can exit after its usual ~20s idle.

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

#### Download the published assets (recommended, automatic)

Nothing to run: `ryk` downloads them the first time it needs them. To do it ahead of time:

```bash
ryk --install-assets
```

Either way it pulls `kokoro-onyx.zip` from the [releases page](https://github.com/kaarrot/kukuryku/releases),
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

1. `KOKORO_TRACT_DIR`, if set — always wins, and is the only dir checked. If it lacks the
   stages, `ryk` downloads the bundle into it.
2. `dirs::data_dir()/kukuryku/kokoro-onyx` — the OS-specific per-user data dir where
   `ryk` unpacks the bundle. Standard install target for an installed `ryk`.
   `KUKURYKU_ASSET_DIR` replaces this arm when set.
3. `./kokoro-onyx` — running from the project root, as in the quick start.
4. `kokoro-onyx/` beside the `ryk` executable — the `--install-assets --dev` target, for
   iterating on a checkout without polluting the real user data dir. Also serves as a
   last-resort arm so old installs keep working.

A directory that exists but lacks one of the stages is skipped. If none of these holds
`stage1.onnx` + `stage2.onnx`, `ryk` downloads the published bundle itself (the same fetch as
`ryk --install-assets`) into arm 2 and then speaks. That first fetch needs GitHub. Once the
stages are on disk, later runs do not touch the network, so `ryk` keeps working where GitHub
is unreachable. To set up a machine that can't reach GitHub at all, copy a `kokoro-onyx/` from
another machine into arm 2 (or anywhere, with `KOKORO_TRACT_DIR`); the error from a failed
first fetch prints that path. The split stages are *not* on Hugging Face. Only a missing
**voice** is fetched from the HF cache; the published bundle already includes `af_heart` and
`am_michael`.

> Phonemization uses raw `espeak-ng` rather than Kokoro's reference phonemizer (misaki), so
> pronunciation is close but not identical on tricky words.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `KOKORO_VOICE` | `af_heart` | Voice (e.g. `am_michael`, `bf_emma`, …) |
| `KOKORO_LANG` | `en-us` | espeak-ng language |
| `KOKORO_SPEED` | `1.0` | Speaking rate |
| `KOKORO_RAW` | _(unset)_ | If set, skip the markdown cleanup and speak input verbatim (same as `--raw`) |
| `KOKORO_WAV` | _(unset)_ | If set, write a 16-bit PCM WAV here instead of / in addition to playing |
| `KOKORO_TRACT_DIR` | _(auto; see above)_ | Directory holding `stage1.onnx` + `stage2.onnx` + `voices/`; downloaded into on first use if empty |
| `KOKORO_TRACT_THREADS` | _(Android: the gold cluster minus the prime core; elsewhere all cores)_ | Thread-pool size for the stage-2 vocoder. Fewer threads do not save battery: the golds run at the same clock either way, so the same work just takes longer. |
| `KOKORO_TRACT_FP16` | on for aarch64 with fp16 SIMD, off elsewhere | `1` forces the f16 GEMMs of both stages on (errors if the CPU lacks `asimdhp`); `0` forces f32. Weights are cast in memory; the ONNX files stay f32. |
| `KOKORO_GOVERNOR` | `thermal` | `pace` stops synthesizing ahead once 20 s of audio is queued. `thermal` (default) also drops stage 2 to 2 threads while the hottest gold core zone is at or above `KOKORO_THERM_HOT_C`. `off` does neither. |
| `KOKORO_THERM_HOT_C` | `90` | Thermal-hold trip in °C (released 6 °C lower). Normal 3-gold load sits at 80–85 °C, so this only fires when the phone is abnormally hot. |
| `RYK_SOCKET` | `$XDG_RUNTIME_DIR/ryk.sock` | Daemon socket for `--serve`/`--send` (see below) |
| `RYK_IDLE_TIMEOUT` | `1800` (seconds) | `--serve` exits after this long with no jobs and no live audio sink. `0`/`off`/`none`/`-1` disables. `--send` auto-starts a replacement. |
| `RYK_SINK_IDLE_MS` | `30000` (30 seconds) | Audio-sink **grace period**. `--serve` keeps `pacat`/`ffplay` open this long after the last sample so nearby `--send`s reuse the same pipe. Then the sink closes and PulseAudio can idle-exit. Kept short because a connected `pacat` holds the Android audio path awake even when silent. |
| `RYK_HISTORY_SECS` | `180` | `--serve`: seconds of **already-played** audio kept so `back` / `rewind` can replay it (~17 MB at 180 s). |
| `RYK_REWIND_WINDOW_SECS` | `600` | `--serve`: how long after the last sample the history stays, so `ryk-ctl back` still works once speech has ended. Independent of the 30 s sink grace (the sink is respawned on demand). Keep `RYK_IDLE_TIMEOUT` at least this long. |
| `RYK_PAUSE_MAX_SECS` | `600` | `--serve`: a pause longer than this becomes a cancel. |
| `RYK_BACK_RESTART_SECS` | `1.5` | `back` restarts the sentence you are in when you are more than this far into it, else it goes to the previous sentence (previous-track behaviour). |
| `RYK_REWIND_SECS` | `10` | Default step for `ryk-ctl rewind` / `ryk-rewind`. |
| `RYK_SINK_PIPE_BYTES` | `4096` | Size of the pipe to `pacat` (Linux/Android). The 64 KiB default queues 0.68 s of audio ahead of the speaker, so a pause would stop that far late; 4 KiB cuts the lag from ~0.87 s to ~0.23 s. `0`/`off` keeps the system default. |
| `RYK_PULSE_ARGS` | _(unset)_ | Extra args for `pulseaudio --start` (Termux). Do **not** pass `--exit-idle-time=-1` unless you want the daemon to pin the audio HAL awake. |

(`KOKORO_MODEL` — the HF-repo path of the monolithic model — applies to `kokoro-ort` only;
`ryk` ignores it, since it runs the split stages, not `model.onnx`.)

Diagnostics (rarely needed): `KOKORO_TRACT_PROFILE=1` prints per-op profiles,
`KOKORO_TRACT_PROFILE_NODES=N` adds the top-N individual nodes with names and IDs.
`KOKORO_TRACT_PROFILE_JSON=dir` enables profiling and saves every node's timing,
tensor sizes/dtypes, graph edges, and op details such as selected matmul kernels
(inspect with `tools/analyze_fusion_profile.py`).
`KOKORO_TRACT_DUMP=dir` dumps
stage-boundary tensors. `tools/bench_conv.sh <label>` runs a fixed-sentence best-of-N timing +
profile for A/B work.

On ARM64 Android, stage-1 FP16 dense matrices default to the A55 16x8 kernel,
measured at 5–6% less total inference CPU time on this phone. Set
`KOKORO_TRACT_S1_DENSE_KERNEL=legacy` (or `=0`) to restore the previous tract
selection; `=default` uses the application default. Other platforms keep their
previous default. Stage-2 and known matrix-vector operations retain their usual
kernels. Extremely small N=3 dense matrices can be slower (about 0.4 ms per FFN
multiply in the probe); these results do not establish battery-energy savings.
See [the matmul investigation](docs/matmul-kernel-investigation.md) for measurements
and probe commands.

ARM64 Android also runs small matmuls directly on the caller, avoiding pool
dispatch for a single outer tile or at most 262,144 multiply-accumulates. Larger
calls with multiple outer tiles still use the pool. Six warm A/B pairs used
3.3% less inference-process CPU time, with roughly unchanged elapsed generation
time. `TRACT_CPU_MMM_MIN_WORK=0` restores the previous pool
dispatch; a positive integer sets the cutoff. The value is read once when the
process first uses a multithreaded matmul. AdaIN/Snake row dispatch is unchanged.
See [the dispatch investigation](docs/tiny-dispatch-investigation.md) for the
profiling data, latency results and measurement limits.

Contiguous FP32 Resize operations use a specialized last-axis path for
asymmetric nearest/floor and half-pixel linear interpolation. Six warm A/B pairs
used about 1% less total inference CPU time, with roughly unchanged elapsed
generation time and exact raw speech outputs. `TRACT_ONNX_RESIZE_FAST=0` restores
the previous generic evaluator; the setting is read once per process. The
linear position table is bounded at 16 KiB on this phone. See
[the Resize investigation](docs/resize-investigation.md) for profiling data and
validation. Battery-energy savings remain unmeasured.

### Markdown is stripped before speaking

Listening to a `.md` file used to mean hearing its syntax: a nested `** bullet **` had every
asterisk pronounced, headings announced their hashes, and tables and code fences turned into long
runs of punctuation names. `ryk` now runs a **markdown cleanup pass over all input by default**
(`src/markdown.rs`), before the sentence splitting below.

- **Dropped:** fenced code blocks, tables, horizontal rules, images, HTML tags, link URLs,
  link-reference definitions.
- **Kept, markers removed:** emphasis (`**bold**`, `_italic_`, `~~strike~~`), headings (which gain
  a period so they land as their own utterance), blockquotes, bullet markers, inline `` `code` ``
  (usually a function or file name that belongs in the sentence), and link *text*.
- **Left alone:** ordered-list numbers (`1.` — informative when listening), and anything that only
  looks like markup: `3 * 4 = 12`, `snake_case_name`, `Vec<String>`, `a < b > c`.

Pass **`--raw`** (or set `KOKORO_RAW=1`) to speak the input verbatim instead. To see what the pass
does to a file without listening to it:

```bash
ryk --show-text < notes.md
```

`docs/markdown-stripping.md` has the full rule table and the reasoning behind the false-positive
guards.

### Long input, and streaming across sentences

Kokoro-82M has a **fixed ~510-phoneme context** (`MAX_PHONEMES` in `src/lib.rs`). Because the model
is **non-autoregressive** (it predicts the whole utterance in one pass), it cannot "continue" past
that window. `ryk` handles arbitrarily long text by **splitting the input into sentences**
(on `.!?;` and newlines; fragments merged, over-long runs wrapped on comma/word boundaries) and
synthesizing each as its own short utterance — always inside the window, and each with its own
clean prosody.

**Chunks with nothing to say are skipped, not fatal.** Text pasted from an editor often carries
punctuation-only lines (`};`, a ``` ``` ``` fence, a `>>>>>>>>` separator) that espeak-ng phonemizes
to nothing. `ryk` drops those chunks, logs one line to stderr, and speaks the rest — it no longer
aborts the whole run partway through. If *every* chunk is unspeakable it warns
`nothing speakable in input` and exits cleanly. Only setup problems (a missing voice file, no
espeak-ng) are still hard errors.

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

The **compiled pipeline** stays hot between sends. The **audio sink** stays warm for a
**30-second grace period** after the last sample (`RYK_SINK_IDLE_MS`, default `30000`):
consecutive sentences of one utterance, and `--send`s that overlap or arrive within those
30 seconds, share one `pacat`/`ffplay` pipe so playback stays gapless. After the grace period
the sink exits and PulseAudio can idle-exit (~20s more); a later `--send` restarts PulseAudio
while its first sentence synthesizes. After `RYK_IDLE_TIMEOUT` seconds of no jobs *and* no
live sink (default 30 minutes after the grace) the daemon itself exits; the next `--send`
starts a fresh one (~4s compile).

### Controlling a running session

The daemon keeps a replayable copy of what it played, so a running session can be paused,
rewound and cancelled without re-synthesizing anything. The client is `ryk-ctl`
(`ryk-skills/bin`; it never starts a daemon):

| Command | Does |
|---|---|
| `ryk-ctl pause` / `resume` / `toggle` | Stop the sink where you heard it (the ~0.23 s still in the pipe is taken back), and continue later. Resume replays the last second, or the sentence start if you paused within 2 s of it. |
| `ryk-ctl back` | **One sentence back**, like a music player's previous-track: restarts the sentence you are in, or goes to the previous one if you only just started it. Repeat to walk back further. Works while paused (and resumes), and after speech has ended, for `RYK_REWIND_WINDOW_SECS`. |
| `ryk-ctl rewind [SECS]` | Go back `SECS` (default `RYK_REWIND_SECS`, 10). `ryk-rewind` is the same thing under the name the Helix `A-r` key runs. |
| `ryk-ctl cancel` | Stop everything: the sentence being spoken, the rest of that text, and every queued `--send`. What was already heard stays rewindable. A `--send` after the cancel plays normally. |
| `ryk-ctl status` | `ok playing`, `ok paused` or `ok idle`. |

Replies are one line (`ok playing back`, `err: nothing playing`); the exit status is 0 for `ok`, 1 for
`err`, 3 when no daemon is running. The same verbs are accepted on the socket as the header line
(`pause`, `resume`, `toggle`, `back`, `cancel`, `status`, `rewind[\tSECS]`) with an empty body.
A paused daemon never idle-exits. See [`docs/headset-control.md`](docs/headset-control.md) for
the plan to drive these from headset buttons.

The socket path is `$RYK_SOCKET`, else `$XDG_RUNTIME_DIR/ryk.sock`, else `/tmp/ryk-$USER.sock`;
an auto-started daemon logs beside it (`…/ryk.log`). This is **Unix-only**; elsewhere use the
one-shot form. Plain `ryk "text"` / stdin is unchanged and needs no daemon.

See [`docs/ryk-cli-and-daemon.md`](docs/ryk-cli-and-daemon.md) for the design and open follow-ups.
Live rewind of already-played audio (Helix `A-r`, Grok/Claude `/rr`, no re-inference) is
planned in [`docs/rewind-plan.md`](docs/rewind-plan.md).

### Helix

Do **not** pipe the selection to one-shot `ryk`: that blocks Helix until playback ends, so a
rewind key cannot fire. Use `--send` and a separate poke for rewind (same script as `/rr`):

```toml
[keys.normal]
A-r = ":sh ryk-rewind"

[keys.normal.space."."]
d = "@:: /path/to/ryk --send<ret>"
```

`ryk-rewind` talks to the live `--serve` socket; it is not a `ryk --rewind` flag. On this
phone the Helix bindings live in `~/.config/helix/config.toml` (outside this repo); the
canonical copy is in [`docs/ryk-cli-and-daemon.md`](docs/ryk-cli-and-daemon.md#editor-integration).

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
(`pulseaudio --start`; extra args via `RYK_PULSE_ARGS`) if none is running — so playback works
without a manual `pulseaudio --start`, which matters for the detached `--serve` daemon. On Android
it adds `--load=module-sles-sink` only when `default.pa` does not already load an OpenSL/AAudio
sink (Termux's stock one does, and loading it twice made a second OpenSL sink). PulseAudio is
started from the playback thread, which is pinned to the little cores, so the daemon does not
inherit the gold-core mask of stage 2; a one-shot run starts it and opens `pacat` while the
model compiles, and the open stream keeps the phone from suspending mid-synthesis with the screen
off. It does **not** pin PulseAudio awake
(`--exit-idle-time=-1`); `--serve` keeps `pacat` up for a **30-second grace period** after the
last sample (`RYK_SINK_IDLE_MS`), then closes the sink so PulseAudio can idle-exit. The next
utterance after that grace respawns both. Or just use `KOKORO_WAV`.
(The `ffplay` path, used on desktop, instead relies on the audio server your session already
runs — PulseAudio, PipeWire, or ALSA via SDL.)

**FP16 GEMMs.** On aarch64, `ryk` casts the GEMMs of both stages to f16 by default when the
CPU has fp16 SIMD (`asimdhp`): the encoder and predictor matmuls in stage 1, and the vocoder's
convolutions and matmuls in stage 2 (its transposed convolutions stay f32). It stays in f32 if
the CPU lacks it. Set `KOKORO_TRACT_FP16=1` to require the optimized path, so `ryk` errors out instead of
quietly falling back to f32:

```bash
KOKORO_TRACT_FP16=1 ryk -v "Hello from Termux."
```

`-v` should print `FP16 GEMM is on`. On the test phone this took wall RTF from 1.46 to 1.11 and
peak CPU temperature from 82 °C to at most 75 °C (75 °C is the floor of the old bench script's
peak column, which included the static `lmh-dcvs` sentinel). Desktop builds leave it off, because x86 has no f16 GEMM
kernels and f32 already runs well under 1.0 RTF there. Use `KOKORO_TRACT_FP16=0` to compare
against f32.

**Battery and heat.** On this class of phone the gold cores run at a fixed 2016 MHz cap while
`ryk` synthesizes, so fewer threads only make the same work slower; they save neither battery
nor, at realtime playback, heat. What saves both is less work per audio second.
[`docs/battery-and-heat.md`](docs/battery-and-heat.md) has the measurements, and
`tools/power_bench.sh` (run unplugged) measures a change in joules per audio second.

## How it works, fidelity, and performance

The full engineering log — the two-stage split, the Rust length regulator, the symbolic
compile-once plan, the vocoder atan2 branch-cut fix that took fidelity to ~0.976, and the Tier 1–7
run-speed arc (RTF 1.73 → ~0.50: lazy im2col, SIMD binary fusion, single-pass variance, Pad fold,
a mimalloc global allocator, `Square(Sin)`→`SinSq` fusion, and a vectorized `sin`) — is in
[`docs/tract-support-plan.md`](docs/tract-support-plan.md).

## Testing

Smoke-test RTF against the already-built binary (no `cargo run`, no rebuild) with a two-sentence
prompt so the pipeline reports a `[1/2]` and `[2/2]` line plus the `done:` summary:

```bash
target/release/ryk -v "The old lighthouse keeper climbed the winding staircase every single evening at dusk. Carrying his heavy brass lantern up the narrow stone steps to make certain the great lamp would burn steadily and brightly through the long and stormy night."
```

Read **`infer RTF`** for pure synthesis speed and **`wall RTF`** for synthesis + sink startup +
first-sample latency. Prefix with `KOKORO_GOVERNOR=off` to isolate full-pool speed from the
governor's thermal/pacing decisions.

Fastest on Android: override the 3-gold default to use all four gold cores (adds the prime core
`cpu7`, ~4% faster on SD855). It is not the default because it costs battery and heat: on a
3-sentence paragraph it used ~7% more CPU time than the 3-gold pool and peaked at 85 °C.

```bash
KOKORO_TRACT_CPUSET=4-7 KOKORO_TRACT_THREADS=4 target/release/ryk -v "The old lighthouse keeper climbed the winding staircase every single evening at dusk. Carrying his heavy brass lantern up the narrow stone steps to make certain the great lamp would burn steadily and brightly through the long and stormy night."
```

