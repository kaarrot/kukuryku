# `ryk` CLI & warm-daemon design

Status: **implemented** (2026-07-17). A living doc to iterate over — the design and open
questions for a low-latency editor-integration path, not a strict changelog. The
`--serve`/`--send` modes and the CLI fixes in "Already shipped" are all in `tract-prototype`.
See [Low-latency editor use](../README.md#low-latency-editor-use--serve--send) in the README
for usage; the open items are in "Follow-ups" below.

## Goal

Make "select text in an editor → hear it, repeatedly" feel instant, and keep `ryk` pleasant
to drive from vim/emacs and the shell — without changing how the plain one-shot synthesizer
already works.

## Already shipped (context)

Commit `0165d49` (branch `tract-prototype`):

- **Flags.** `ryk --help`/`-h` and `--version`/`-V`; an unknown `--flag` errors instead of
  being phonemized into gibberish. Previously only `--install-assets` was intercepted, so any
  other flag fell through to `read_text()` and got spoken.
- **Asset resolution decoupled from `model.onnx`.** The tract path loads `stage1.onnx` +
  `stage2.onnx` and never reads the monolithic `model.onnx`, yet the old `resolve_assets()`
  required it locally or downloaded it into `~/.cache/huggingface`. New
  `kokoro::resolve_assets_tract()` resolves the split-model dir from the stages. When
  none of the local lookup dirs has them, `ryk` downloads the published bundle (the same
  fetch as `ryk --install-assets`) and then continues. A later run that finds the stages
  does not touch the network. HF is used only for a genuinely missing voice. `kokoro-ort`
  still uses the model.onnx-centric `resolve_assets()` (correct there — it actually loads
  the model).

## The latency problem

Every `ryk` invocation pays startup before the first sentence is synthesized:

- Compiling the two tract stages (`into_optimized`): stage2 ≈ 3.9 s, stage1 ≈ 1.4 s, run in
  parallel ≈ 4 s wall.
- Loading ~325 MB of ONNX off disk.

For one-shot use that's fine. For an editor loop (select → speak, again and again) it means
~4 s of dead air on every call. Reading text as an **argument** or over **stdin** already
works well and is the right channel for editors (stdin avoids shell quoting/escaping and
`ARG_MAX` limits on large selections) — the missing piece is not paying the compile each time.

## What the current streaming already does (and doesn't)

The one-shot path is not "process everything, then play". It reads the whole (finite) input,
splits it into sentences up front (cheap), then synthesizes **one sentence at a time** and
pushes each to `StreamPlayer` (`src/lib.rs`). `push()` hands the chunk to a background thread
over a bounded channel and returns, so **sentence 1 plays while sentence 2+ synthesize**.
Time-to-first-audio is compile + first sentence, never the whole input; gaps appear only if a
sentence synthesizes slower than realtime (backend-dependent). The warm daemon keeps this
behavior and removes the per-call compile.

## Design: warm daemon + thin client

The compiled `Pipeline` is independent of text/voice/speed (those only feed
`prepare`/`synthesize` per utterance), so **one warm pipeline serves every request**. A
persistent daemon compiles once and owns audio output; a thin client pipes text to it.

Additive and opt-in — plain `ryk "text"` and `ryk < file` are unchanged and need no daemon.
Gated under the existing `tract` feature, `#[cfg(unix)]` (Linux/Termux/macOS), **no new
dependencies** (std `UnixListener`/`UnixStream`, `thread`, `mpsc`, `process`).

### Daemon — `ryk --serve`

- Resolve assets + `Pipeline::new()` once.
- One `StreamPlayer` for the daemon's life. The sink process (`ffplay`/`pacat`) stays warm
  across overlapping or nearby requests (gapless playback, including while the next sentence
  synthesizes). After a **30-second grace period** (`RYK_SINK_IDLE_MS`, default `30000`)
  with an empty queue it closes, so PulseAudio can idle-exit instead of pinning the OpenSL
  HAL. The next `push` respawns it. The grace used to be 10 minutes, but a connected
  `pacat` keeps the PulseAudio/OpenSL sink RUNNING even with nothing to play (~1.5% of a
  core plus the Android audio path), so every short `--send` cost 10 minutes of that.
  Each job calls `StreamPlayer::warm` first, which opens the sink on the playback thread
  (little cores), restarting PulseAudio if it idle-exited, while the first sentence
  synthesizes. A respawn after the grace then adds no ~0.7 s Pulse start to first audio,
  and the open stream keeps Android from suspending while the job synthesizes.
- A **listener** accepts connections; a single **worker thread** owns the `Pipeline` and
  drains an `mpsc` job queue FIFO. Per job: `split_sentences` → per sentence `prepare` +
  `synthesize` → `player.push` (backpressure paces synthesis to playback).

### Client — `ryk --send [TEXT...]`

- Read text from args (joined) or stdin — same rules as one-shot.
- Connect to the socket; if it's not there, **auto-start** the daemon (re-exec
  `current_exe --serve` detached) and poll-connect for ~20 s (covers the compile).
  Before spawning, the client resolves the assets itself, so a first-run model download
  happens in the foreground with progress on the user's terminal. Left to the daemon, the
  download would run silently in its log and outlast the connect window.
- Send the request, half-close, read the `ok`/`err` ack, exit. Fire-and-forget: the client
  returns once the text is queued, so the editor never blocks on playback.

### Concurrency: queue

A new selection arriving mid-playback is **queued** and played after the current one. Chosen
for simplicity and safety — nothing is lost, and it needs no mechanism to kill in-flight
audio. (Barge-in is a possible later mode; see below.)

### Wire protocol

One utterance per connection, EOF-delimited:

```
line 1:  "<voice>\t<lang>\t<speed>\n"     # per-request config header
body:    UTF-8 text until EOF
```

Per-request voice/lang/speed is cheap (voice path resolved on demand, cached in a
`HashMap<String, PathBuf>`) and avoids "restart the daemon to change voice". The client
always sends the header from its own env, so there's no ambiguity.

The **markdown cleanup happens client-side**, in `client_text` (see
`docs/markdown-stripping.md`): the body on the wire is already stripped, so `--raw` is honoured
per invocation without a fourth header field, and the daemon never re-strips what it receives.
A non-`ryk` client speaking this socket directly gets no cleanup and should strip its own text.

### Socket path

`$RYK_SOCKET` if set, else `$XDG_RUNTIME_DIR/ryk.sock`, else `/tmp/ryk-$USER.sock`. On
`--serve`: if the path exists and a connect succeeds, another daemon is live → exit
("already running"); a dead socket is unlinked, then bound. Best-effort unlink on exit via a
`Drop` guard (no signal-handling crate).

## Editor integration

Speak must go through `--send` so the editor is not blocked in one-shot `ryk` until
`finish()`. Rewind is a poke into that same `--serve` process (`ryk-rewind` on the
existing Unix socket), not a `ryk --rewind` flag. Full control-plane plan:
[rewind-plan.md](rewind-plan.md).

- **Helix** (this machine's `~/.config/helix/config.toml`; not in this git repo).
  `space+. d` used to pipe the selection to one-shot `ryk`, which held Helix until
  playback ended. It now `--send`s. `A-r` runs the same `ryk-rewind` poke Grok/Claude
  `/rr` will use:

  ```toml
  [keys.normal]
  # Rewind ~10s of currently playing ryk audio. Same script as Grok/Claude /rr.
  # Needs ~/.local/bin/ryk-rewind and a live ryk --serve (auto-started by --send).
  A-r = ":sh ryk-rewind"

  [keys.normal.space."."]
  # Selection → warm daemon. Helix returns immediately; --serve owns pacat.
  d = "@:: /data/data/com.termux/files/home/PRJ/kukuryku/target/release/ryk --send<ret>"
  ```

  Because `/spk` already `--send`s into the same daemon, Helix `A-r` also rewinds
  Grok/Claude speech if Helix can receive the key (split pane, or Helix focused).

- **Grok / Claude Code:** `/spk` → `speak-last-reply` → `ryk --send` (unchanged).
  Optional `/rr` skill runs **the same** `~/.local/bin/ryk-rewind` via Bash
  (`disable-model-invocation: true`, like `/spk`). Slash dispatch is the slow path
  (harness turn, seconds). Fast path is Helix `A-r`, a Termux widget, or SIGUSR1.
  Grok pager keys are not remappable.

- **vim** (visual selection, buffer unmodified): `xnoremap <leader>s :w !ryk --send<CR>` —
  writes the selection to the client's stdin.
- **emacs**: `(call-process-region (region-beginning) (region-end) "ryk" nil 0 nil "--send")`
  — the `0` runs it async so the editor doesn't block.

## Refactor needed

`tract_backend::Pipeline` currently lives as a private `mod` inside
`src/bin/kokoro_tract.rs`. Move it to `src/tract_backend.rs` (library) so both the binary and
a new `src/serve.rs` can use it. The bin slims to flag dispatch + the existing one-shot loop.

## Docs to update when it lands

- **README**: new "Low-latency editor use (`--serve`/`--send`)" section after "Long input, and
  streaming across sentences"; add `RYK_SOCKET` to the env table; note `--help`/`--version`.
- **README reconciliation** (already-stale from `0165d49`): the `KOKORO_MODEL` env row applies
  to `kokoro-ort` only now; the offline lookup no longer downloads `model.onnx` for
  `ryk`. Missing stages trigger one GitHub bundle download into the user data dir;
  once those files are present, `ryk` only fetches a voice that the bundle does not carry.

## Open questions / follow-ups

- **Rewind 10s of already-played audio** — planned in [rewind-plan.md](rewind-plan.md). Not a
  `ryk --rewind` flag: Helix `A-r` and Grok/Claude `/rr` poke `--serve` via `ryk-rewind`.
  Shares the future barge-in `kill_sink` path but keeps PCM history.
- **Barge-in** (stop current, speak new) — nicer for rapid re-selection, but needs a
  `StreamPlayer` kill path (kill the ffplay/pacat child + abort remaining sentences between
  utterances). Deferred in favor of queue. Rewind will land `kill_sink` first.
- **Idle auto-shutdown** (`RYK_IDLE_TIMEOUT`, default 1800s) so a forgotten daemon doesn't
  linger. Implemented: the accept loop sleeps in `poll(2)` (on the little cores) and exits
  once no jobs are pending and the audio sink has closed. While a job or the sink is live
  it re-checks every 5 s; while truly idle it sleeps until the deadline (it used to wake a
  gold core every 250 ms for the daemon's whole life). The sink itself uses a
  **30-second grace** (`RYK_SINK_IDLE_MS`, default `30000`) after the last sample before
  closing; the daemon idle clock starts only after that grace. `0`/`off`/`none`/`-1`
  disables the daemon timeout. `--send` already auto-starts a replacement.
- **Windows** — `AF_UNIX`/named-pipe transport; one-shot already works there.
- **Voice preloading** — should the daemon pre-resolve a set of voices, or lazily on first use?
