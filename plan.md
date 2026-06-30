
# Plan: `speak` — a CPU TTS CLI on Candle (Orpheus-3B + SNAC)

## Status — resume here (last touched 2026-06-18)

**Branch:** `audio-prototype` (created off `main`; nothing committed yet — this plan is the only
artifact so far).

**Environment verified on the dev box:**
- PulseAudio server **already running** (`/run/user/1000/pulse/native`); `pacat`, `paplay`,
  `pactl`, `ffplay` present (`pulseaudio`, `aplay`, `sox` absent — not needed, the server is up).
  → the **play-aloud path is ready to use right now**.
- candle fork **not cloned** (`candle/` absent) → the project won't build as configured, because
  `Cargo.toml` `[patch]` points at `candle/candle-core` / `candle-transformers`.
- No model files at repo root. ~26 GB free disk. Rust 1.90.

**Open decision (we paused mid-question):** scope of the first prototype —
- **(A) audio-path first** — a small program synthesizes a test waveform → `pacat`; runnable
  immediately, no downloads. Proves play-aloud, then bolt on Orpheus. *(Recommended for a first
  prototype.)*
- **(B) full Orpheus now** — clone the fork, download ~2.3 GB GGUF + SNAC weights, long first CPU
  build, then it reads the hardcoded text for real.

**Prototype build shortcut (no fork clone needed):** the repo already vendors candle 0.10.1 at
`vendor/candle-core` + `vendor/candle-transformers`, and `Cargo.toml` has a commented-out `[patch]`
pointing there. Switching the patch to the `vendor/` paths lets the prototype build with stock
(unoptimized) candle; the fork's CPU speedups get wired in later. Note `vendor/` has no
`candle-nn` — pull `candle-nn = "0.10.1"` from crates.io (it resolves onto the patched candle-core).

**Immediate next steps:**
1. Decide A vs B above.
2. Backend: either `git clone https://github.com/kaarrot/candle.git ./candle && git -C candle
   switch feature/cpu-decode-optim-squashed`, **or** switch `[patch]` to `vendor/` for the
   no-fork prototype.
3. Add deps to `Cargo.toml`: `candle-nn`, `hf-hub`, `serde_json`, `bytemuck`.
4. Write `src/bin/speak.rs` (Orpheus → SNAC → `pacat`, hardcoded text).
5. Quick audio sanity check first:
   `cat /dev/urandom | head -c 96000 | pacat --rate=24000 --channels=1 --format=float32le`
   should make ~0.5 s of noise.

---

## Context

`candle-inference` exists to benchmark quantized **Llama-3.2-3B (Q4_K_M GGUF)** CPU decode in
Candle (a fork on `feature/cpu-decode-optim-squashed`) against llama.cpp. The fork adds an
optimized no-copy decode path (`forward_decode_token_into`) and Q4_K repack, exercised by
`src/main.rs`.

The goal is a command-line tool that takes input text and **reads it aloud** with the most
natural voice available, CPU-only, built on Candle, and reusing the CPU decode optimization.

**Chosen approach: Orpheus-3B + SNAC.** Orpheus is a single decoder-only **Llama-3.2-3B**
fine-tune that emits SNAC audio tokens (~150 tokens per second of speech, 7 per frame); the SNAC
24 kHz codec decodes them to a waveform. Because it *is* a Llama-3.2-3B model, it runs through the
exact `quantized_llama` decode path this repo already optimizes — the audio "tokens" flow through
`forward_decode_token_into` 1:1. It ships 8 consistent built-in voices (`tara`, `leah`, …) plus
trained emotion tags (`<laugh>`, `<sigh>`, …), needs no reference audio, and has Q4_K_M GGUF
weights. Decisions: **GGUF weights**, **play aloud only** — no WAV file — by piping raw PCM to a
running **PulseAudio** server via `pacat`, which works identically on desktop Linux and Termux.
Expect slower-than-realtime on
CPU initially — that is precisely what the optimization branch is meant to close, so the tool
prints a realtime factor to tie back into the benchmark.

Everything except the model load (`from_gguf` instead of safetensors) mirrors upstream candle's
`candle-examples/examples/orpheus/main.rs` — the prompt format, token→SNAC parsing, and SNAC
decode should be copied from there. `snac.rs` and `mimi/` are already vendored, so no model code
needs writing.

## What gets built

A new binary `src/bin/speak.rs` (binary name `speak`), leaving the benchmark `main.rs` intact.
Shared scaffolding is lifted into a tiny `src/lib.rs` so both binaries reuse it instead of
duplicating.

### 1. Reuse / refactor: `src/lib.rs`
Extract from `src/main.rs` (no behavior change) and re-export so `main.rs` and `speak.rs` share:
- `env_string`, `env_parse` (`src/main.rs:280-298`)
- `check_memory_headroom` + helpers (`src/main.rs:300-395`)
- a `load_quantized_llama(path, &device) -> ModelWeights` wrapper around
  `gguf_file::Content::read` + `quantized_llama::ModelWeights::from_gguf` (`src/main.rs:42-46`)

`main.rs` becomes a thin user of these. Keep it light; if extraction looks risky, `speak.rs` may
duplicate the ~15 helper lines instead — the benchmark must not change behavior.

### 2. `src/bin/speak.rs`
- **Input:** text from `argv[1..].join(" ")`, falling back to stdin if no args. Voice from
  `SPEAK_VOICE` (default `tara`).
- **Load (CPU):** `Device::Cpu`; load Orpheus GGUF via the shared `load_quantized_llama`; load
  `tokenizer.json`; load SNAC (`snac::Model::new` with `candle_nn::VarBuilder` over
  `snac_24khz.safetensors` + its `config.json`).
- **Prompt format (copy from upstream orpheus example):**
  encode `"{voice}: {text}"`, then build the id sequence
  `[128259] + ids + [128009, 128260, 128261, 128257]`. Stop token `128258`.
- **Decode loop:** replicate the optimized loop from `src/main.rs:102-165` — prompt forward via
  `model.forward`, then per-token `model.forward_decode_token_into(..)` no-copy path with
  `LogitsProcessor` (temp ~0.6, optional top-k/top-p, seed). Collect generated ids; stop on
  `128258` or a max-token cap sized to requested seconds (~150 tokens/sec → cap e.g. 30 s).
- **Token → SNAC codes (copy from upstream orpheus example):** map each audio token via
  `tok = tok - 10 - ((index % 7) * 4096)`, then `chunks_exact(7)` de-interleaved into 3 SNAC
  codebooks: `[0]`, `[1,4]`, `[2,3,5,6]`. Build three `(1, n)` code tensors.
- **SNAC decode:** `snac::Model::decode(&[&c0, &c1, &c2])` → `(1,1,samples)` f32 PCM at 24 kHz;
  flatten to `Vec<f32>` in [-1, 1].
- **Play aloud (no WAV, PulseAudio `pacat`):** ensure the daemon is up via an idempotent
  `pulseaudio --start` (no-op when already running), then spawn
  `pacat --rate=24000 --channels=1 --format=float32le` with a piped stdin and
  `write_all(bytemuck::cast_slice(&samples))` (x86/aarch64 are little-endian, so the f32 slice is
  already f32le). Drop stdin to signal EOF, then `child.wait()`. Identical code on Linux and
  Termux — no audio crate, no in-process device, so the Termux JNI/oboe problem never arises.
- **Metrics:** print generated audio-token count, audio seconds, decode t/s, and realtime factor
  (reusing the timing style in `main.rs`).

### 3. `Cargo.toml`
Add: `candle-nn = "0.10.1"` (for `VarBuilder`), `hf-hub` (fetch weights), `serde_json` (read SNAC
`config.json`), `bytemuck` (f32→bytes for the pipe). **No audio crate** — playback shells out to
`pacat`, so there is no `cpal`/`rodio`/oboe dependency (this is what avoids the Termux JNI/JavaVM
problem entirely). Add `candle-nn = { path = "candle/candle-nn" }` to the existing
`[patch.crates-io]` block so all candle crates resolve to the fork (version-aligned).

## Model assets (via hf-hub, with env overrides)
- **Orpheus GGUF (Q4_K_M):** `dahara1/orpheus-3b-0.1-ft_gguf` (or `isaiahbjork/...`); confirm the
  exact `*q4_k_m.gguf` filename at implementation time. Override via `CANDLE_MODEL_PATH`
  (reuses the existing default convention).
- **Tokenizer:** `canopylabs/orpheus-3b-0.1-ft` → `tokenizer.json`.
- **SNAC:** weights `lmz/candle-snac` → `snac_24khz.safetensors`; config `hubertsiuzdak/snac_24khz`
  → `config.json`.

## Notes / risks
- **Playback dep (both targets):** a running PulseAudio-compatible server + the `pacat` client.
  Desktop Linux already has it (PulseAudio, or PipeWire via `pipewire-pulse`); on a bare box
  `apt install pulseaudio`. Termux: `pkg install pulseaudio`, started once per session with the
  OpenSL ES sink and no idle-exit:
  `pulseaudio --start --load="module-sles-sink" --exit-idle-time=-1` (or persist it via
  `~/.config/pulse/{default.pa,daemon.conf}` + shell autostart). The daemon is a **session-level**
  cost, not per-invocation; `pulseaudio --start` is an idempotent no-op when already up.
- **Speed:** Orpheus-3B Q4_K_M on CPU is ~5–10× slower than realtime before the optimization;
  this is expected and is the motivation for the fork. The printed realtime factor makes it
  measurable.
- **Quality:** 4-bit can slightly flatten emotion; if expression suffers, try Q5_K_M / Q6_K
  (still loads through the same `quantized_llama` path).
- **Memory:** GGUF ~2.3 GB + SNAC (small); the reused `check_memory_headroom` guards low-RAM hosts
  (bypass with `CANDLE_INFERENCE_FORCE=1`).

## Verification
1. Confirm a PulseAudio server is reachable and `pacat` plays raw PCM:
   `cat /dev/urandom | pacat --rate=24000 --channels=1 --format=float32le` (Linux: usually already
   running; Termux: `pkg install pulseaudio` then
   `pulseaudio --start --load="module-sles-sink" --exit-idle-time=-1`).
2. Build: `RUSTFLAGS='-C target-cpu=native' cargo build --release --bin speak` (desktop). On
   Termux build natively (`pkg install rust`) and drop `target-cpu=native` (NEON is baseline on
   aarch64).
3. `./target/release/speak "Hello there. This is a Candle CPU text to speech test. <laugh>"`
   - Expect: assets download/load, audio tokens generate, SNAC decode, and the sentence plays
     through the speakers in the `tara` voice — no file written.
4. Voice + emotion check: `SPEAK_VOICE=leo ./target/release/speak "I can sound different too. <sigh>"`
5. Confirm the metrics line prints audio seconds, decode t/s, and realtime factor; re-run on the
   optimized branch to confirm the t/s improvement carries over to the TTS decode loop.

---

## Options considered (for reference)

| Model | In candle? | Reuses your quantized decode opt? | Naturalness | Notes |
|-------|-----------|-----------------------------------|-------------|-------|
| **Orpheus-3B + SNAC** (chosen) | example exists; `snac.rs` vendored | **Yes** — single Llama-3.2-3B → GGUF `quantized_llama` | Very natural/emotive | 8 named voices, emotion tags, no context needed |
| Sesame CSM-1B + Mimi | `csm.rs` + `mimi/` vendored | No — dual transformer, safetensors path | Highest isolated MOS | Needs reference-audio context or sounds flat; no named-voice UX |
| Kokoro-82M | No (ONNX only) | No | Very natural, #1 TTS Arena | Leaves candle entirely |
| Parler-TTS / MetaVoice | examples + `parler_tts.rs`/`metavoice.rs` | No | Lower | Don't reuse the Llama optimization |
