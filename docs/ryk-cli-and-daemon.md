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
  `kokoro::resolve_assets_tract()` resolves the split-model dir from the stages, errors
  toward `ryk --install-assets` when they're missing, and touches HF only for a genuinely
  missing voice. `kokoro-ort` still uses the model.onnx-centric `resolve_assets()` (correct
  there — it actually loads the model).

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
- One long-lived `StreamPlayer` (ffplay/pacat stays warm; playback is gapless *across*
  queued requests, not just within one).
- A **listener** accepts connections; a single **worker thread** owns the `Pipeline` and
  drains an `mpsc` job queue FIFO. Per job: `split_sentences` → per sentence `prepare` +
  `synthesize` → `player.push` (backpressure paces synthesis to playback).

### Client — `ryk --send [TEXT...]`

- Read text from args (joined) or stdin — same rules as one-shot.
- Connect to the socket; if it's not there, **auto-start** the daemon (re-exec
  `current_exe --serve` detached) and poll-connect for ~15 s (covers the compile).
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

### Socket path

`$RYK_SOCKET` if set, else `$XDG_RUNTIME_DIR/ryk.sock`, else `/tmp/ryk-$USER.sock`. On
`--serve`: if the path exists and a connect succeeds, another daemon is live → exit
("already running"); a dead socket is unlinked, then bound. Best-effort unlink on exit via a
`Drop` guard (no signal-handling crate).

## Editor integration

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
  to `kokoro-ort` only now; the offline lookup "arm 4" no longer downloads `model.onnx` for
  `ryk` — it needs the stages locally (`--install-assets`) and only fetches a missing voice.

## Open questions / follow-ups

- **Barge-in** (stop current, speak new) — nicer for rapid re-selection, but needs a
  `StreamPlayer` kill path (kill the ffplay/pacat child + abort remaining sentences between
  utterances). Deferred in favor of queue.
- **Idle auto-shutdown** (`RYK_IDLE_TIMEOUT`) so a forgotten daemon doesn't linger.
- **Windows** — `AF_UNIX`/named-pipe transport; one-shot already works there.
- **Voice preloading** — should the daemon pre-resolve a set of voices, or lazily on first use?
