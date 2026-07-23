//! Shared, backend-agnostic pieces of the Kokoro-82M TTS pipeline, used by both
//! the `kokoro-ort` (onnxruntime) and `ryk` (pure-Rust) binaries. Everything
//! here is independent of the inference engine: text -> phonemes -> token ids +
//! style vector, asset resolution, and audio output (WAV + ffplay). Each binary
//! adds only its own model-execution step between `prepare` and `emit`.

/// `ryk --install-assets`. Gated on `tract` because that is the only binary that
/// needs the split stages, and it keeps the http/zip deps out of other builds.
#[cfg(feature = "tract")]
pub mod install;

/// The two-stage tract inference `Pipeline`. In the library so both the `ryk`
/// binary's one-shot path and the `serve` daemon can share one compiled pipeline.
#[cfg(feature = "tract")]
pub mod tract_backend;

/// `ryk --serve` (warm daemon) + `ryk --send` (thin client). Unix-only (uses
/// `std::os::unix::net`); additive to the one-shot path.
#[cfg(all(feature = "tract", unix))]
pub mod serve;

pub mod kokoro {
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    use anyhow::{Context, Result, bail};
    use hf_hub::api::sync::Api;

    pub const REPO: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
    pub const SAMPLE_RATE: u32 = 24000;
    pub const MAX_PHONEMES: usize = 510; // model context; pad token 0 wraps both ends
    pub const STYLE_DIM: usize = 256;

    pub fn env_or(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }

    /// Text from CLI args, else stdin. A leading `--` is the usual argv separator and
    /// is dropped, which is the only way to speak text that itself starts with a dash
    /// (`ryk -- "- first bullet"`) without it looking like a flag.
    pub fn read_text() -> Result<String> {
        let mut args: Vec<String> = std::env::args().skip(1).collect();
        if args.first().is_some_and(|a| a == "--") {
            args.remove(0);
        }
        if !args.is_empty() {
            return Ok(args.join(" "));
        }
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        let buf = buf.trim().to_string();
        if buf.is_empty() {
            bail!("no text provided (pass as args or pipe to stdin)");
        }
        Ok(buf)
    }

    /// Phonemize text to IPA via espeak-ng (stress marks via --ipa=3). The exact
    /// output differs slightly from Kokoro's reference phonemizer, so pronunciation
    /// is close but not identical — fine for a prototype. Unknown symbols (e.g. the
    /// ZWJ ties espeak emits) are dropped later by the vocab filter.
    ///
    /// The `--` before `text` matters: espeak-ng parses a leading `-` as a CLI option,
    /// so a markdown bullet (`- item`) or a chunk like `---` would otherwise fail the
    /// whole run with `unrecognized option`.
    pub fn phonemize(text: &str, lang: &str) -> Result<String> {
        let out = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "-v", lang, "--", text])
            .output()
            .context("running espeak-ng (is it installed?)")?;
        if !out.status.success() {
            bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.replace('\n', " ").trim().to_string())
    }

    /// Model + voice files, resolved from the HF cache (downloaded on first use).
    pub struct Assets {
        pub model_path: PathBuf,
        pub voice_path: PathBuf,
    }

    /// `kokoro-onyx/` beside the running executable. The dev-mode install
    /// target: `--install-assets --dev` writes here for local iteration on a
    /// checkout, so `cargo run` doesn't pollute the real user data dir. Also
    /// serves as a last-resort runtime lookup so old installs keep working.
    pub fn exe_assets_dir() -> Option<PathBuf> {
        let exe = std::env::current_exe().ok()?;
        Some(exe.parent()?.join("kokoro-onyx"))
    }

    /// The user data-dir install target: `dirs::data_dir()/kukuryku/kokoro-onyx`.
    /// Right place for ~600 MB of weights: `~/.local/share/...` on Linux,
    /// `~/Library/Application Support/...` on macOS, `%APPDATA%\...` on Windows.
    /// Never falls back to CWD — a `None` here means the platform has no data-dir
    /// (bare embedded, no `$HOME`), and callers should surface that as an error.
    pub fn user_assets_dir() -> Option<PathBuf> {
        Some(dirs::data_dir()?.join("kukuryku").join("kokoro-onyx"))
    }

    /// Directory checked for a local asset bundle before hitting the network, in order:
    ///
    /// 1. `KOKORO_TRACT_DIR` — explicit override, always wins.
    /// 2. `user_assets_dir()` — the standard install target for `--install-assets`.
    /// 3. `./kokoro-onyx` — the repo-root dev workflow (`cargo run` from the checkout).
    /// 4. `kokoro-onyx/` beside the executable — the `--install-assets --dev` target.
    ///
    /// Reuses `KOKORO_TRACT_DIR` (where the split stages live) so one env var can point
    /// at a fully self-contained dir. Arms 2–4 are probed with `is_dir` so a missing one
    /// falls through rather than short-circuiting; if none exist we return the user data
    /// path (or CWD if we couldn't compute one), letting `resolve_assets` report against
    /// the location `--install-assets` would target.
    pub fn local_assets_dir() -> PathBuf {
        if let Some(dir) = std::env::var_os("KOKORO_TRACT_DIR") {
            return PathBuf::from(dir);
        }
        let user_dir = user_assets_dir();
        if let Some(ref dir) = user_dir {
            if dir.is_dir() {
                return dir.clone();
            }
        }
        let cwd_dir = PathBuf::from("kokoro-onyx");
        if cwd_dir.is_dir() {
            return cwd_dir;
        }
        if let Some(dir) = exe_assets_dir() {
            if dir.is_dir() {
                return dir;
            }
        }
        user_dir.unwrap_or(cwd_dir)
    }

    /// Assets for the tract (`ryk`) binary: the directory holding the split stages
    /// plus the voice file. Deliberately *not* [`Assets`] — the tract path never
    /// reads the monolithic `model.onnx`, so requiring it (as [`resolve_assets`]
    /// does, correctly, for `kokoro-ort`) only forced a pointless 325 MB download
    /// into `~/.cache/huggingface` on any box whose bundle lacked it.
    pub struct TractAssets {
        pub dir: PathBuf,
        pub voice_path: PathBuf,
    }

    /// Resolve the split-model dir + voice for `ryk` without ever touching the HF
    /// cache for the model. The dir (see [`local_assets_dir`]) must already hold
    /// `stage1.onnx` + `stage2.onnx` — those come from `ryk --install-assets`, not
    /// HF — so a missing pair is a clear error, not a silent network fetch. Only a
    /// missing *voice* falls back to the HF cache (voices do live in [`REPO`]).
    pub fn resolve_assets_tract(voice: &str) -> Result<TractAssets> {
        let dir = local_assets_dir();
        let stage1 = dir.join("stage1.onnx");
        let stage2 = dir.join("stage2.onnx");
        if !stage1.is_file() || !stage2.is_file() {
            bail!(
                "split model not found in {} (need stage1.onnx + stage2.onnx).\n  \
                 run `ryk --install-assets` to fetch the bundle, or point KOKORO_TRACT_DIR \
                 at a directory that has them.",
                dir.display()
            );
        }
        let local_voice = dir.join("voices").join(format!("{voice}.bin"));
        let voice_path = if local_voice.is_file() {
            local_voice
        } else {
            let api = Api::new().context("creating hf-hub api")?;
            api.model(REPO.to_string())
                .get(&format!("voices/{voice}.bin"))
                .with_context(|| format!("fetching voice {voice}"))?
        };
        Ok(TractAssets { dir, voice_path })
    }

    pub fn resolve_assets(model_file: &str, voice: &str) -> Result<Assets> {
        // Project-local bundle first: if the assets dir already holds `<model>.onnx` and
        // `voices/<voice>.bin`, use them and skip hf-hub entirely — no network, which is
        // what makes an offline / Termux run possible. Falls back to the HF cache otherwise.
        let dir = local_assets_dir();
        let model_name = Path::new(model_file).file_name().unwrap_or(model_file.as_ref());
        let local_model = dir.join(model_name);
        let local_voice = dir.join("voices").join(format!("{voice}.bin"));
        if local_model.is_file() && local_voice.is_file() {
            return Ok(Assets { model_path: local_model, voice_path: local_voice });
        }

        let api = Api::new().context("creating hf-hub api")?;
        let model_path = api
            .model(REPO.to_string())
            .get(model_file)
            .with_context(|| format!("fetching {REPO}/{model_file}"))?;
        let voice_path = api
            .model(REPO.to_string())
            .get(&format!("voices/{voice}.bin"))
            .with_context(|| format!("fetching voice {voice}"))?;
        Ok(Assets { model_path, voice_path })
    }

    /// Phonemes -> padded token ids + the per-utterance style vector.
    pub struct Prepared {
        pub phonemes: String,
        pub ids: Vec<i64>,  // token ids, padded with 0 at both ends
        pub style: Vec<f32>, // STYLE_DIM row for this utterance length
        pub token_len: usize, // unpadded token count
    }

    /// Outcome of preparing one chunk. Neither non-`Ready` variant is fatal: a chunk
    /// with nothing to say is simply not spoken, and the rest of the input plays on.
    pub enum ChunkPrep {
        /// Ready to synthesize.
        Ready(Prepared),
        /// Phonemized fine, but nothing survived the vocab filter — a punctuation-only
        /// line (`};`, ```` ``` ````, `???`), which is what pasted code is full of.
        Unspeakable,
        /// espeak-ng itself failed on this chunk; already reported to stderr.
        Failed,
    }

    /// Phonemize + tokenize (Kokoro fixed vocab) + pick the voice style row.
    ///
    /// Returns `None` when the chunk phonemizes to nothing the model can say — see
    /// [`ChunkPrep::Unspeakable`]. Callers that want espeak-ng failures tolerated too
    /// should use [`prepare_or_skip`].
    pub fn prepare(text: &str, lang: &str, voice_path: &std::path::Path) -> Result<Option<Prepared>> {
        let phonemes = phonemize(text, lang)?;
        prepare_from_phonemes(phonemes, voice_path)
    }

    /// [`prepare`], with espeak-ng process failures downgraded to a skip. Errors that
    /// still propagate are setup problems (an unreadable voice file), not text ones.
    pub fn prepare_or_skip(
        text: &str,
        lang: &str,
        voice_path: &std::path::Path,
    ) -> Result<ChunkPrep> {
        let phonemes = match phonemize(text, lang) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[kokoro] skipping chunk (espeak-ng failed: {e:#})");
                return Ok(ChunkPrep::Failed);
            }
        };
        Ok(match prepare_from_phonemes(phonemes, voice_path)? {
            Some(p) => ChunkPrep::Ready(p),
            None => ChunkPrep::Unspeakable,
        })
    }

    /// Tokenize IPA against the fixed vocab and pick the style row. Symbols the vocab
    /// doesn't carry (e.g. espeak's ZWJ ties) are dropped silently; if *nothing* is
    /// left there is no utterance to make, hence `Ok(None)` rather than an error.
    fn prepare_from_phonemes(
        phonemes: String,
        voice_path: &std::path::Path,
    ) -> Result<Option<Prepared>> {
        let vocab: HashMap<String, i64> =
            serde_json::from_str(include_str!("kokoro_vocab.json")).context("parsing vocab")?;

        let mut tokens: Vec<i64> = phonemes
            .chars()
            .filter_map(|c| vocab.get(&c.to_string()).copied())
            .collect();
        tokens.truncate(MAX_PHONEMES);
        if tokens.is_empty() {
            return Ok(None);
        }
        let token_len = tokens.len();

        // Voice style vector: row indexed by (unpadded) token length.
        let voice_bytes = std::fs::read(voice_path).context("reading voice file")?;
        let voice_f32: &[f32] = bytemuck::cast_slice(&voice_bytes);
        let rows = voice_f32.len() / STYLE_DIM;
        let row = token_len.min(rows - 1);
        let style: Vec<f32> = voice_f32[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

        // Input ids, padded with 0 at both ends.
        let mut ids: Vec<i64> = Vec::with_capacity(token_len + 2);
        ids.push(0);
        ids.extend_from_slice(&tokens);
        ids.push(0);

        Ok(Some(Prepared { phonemes, ids, style, token_len }))
    }

    /// Split a paragraph into sentence-sized chunks for streaming synthesis.
    ///
    /// This is what lets long input play back smoothly: each chunk is synthesized
    /// and queued independently, so the model works on the *next* sentence while
    /// `ffplay` is still reading the current one aloud (see [`StreamPlayer`]). It
    /// also sidesteps the `MAX_PHONEMES` truncation that would otherwise clip a
    /// long paragraph to the first ~510 phonemes.
    ///
    /// Rules: break after sentence-final punctuation (`. ! ? ;`) when followed by
    /// whitespace/end, and on newlines; merge fragments shorter than `MIN_CHARS`
    /// into their predecessor (no micro-utterances); and hard-wrap any run longer
    /// than `MAX_CHARS` on word (preferably comma) boundaries so no chunk grossly
    /// overruns the model's phoneme budget. Abbreviations/decimals aren't special-
    /// cased — a stray split just adds a short pause, which is harmless here.
    pub fn split_sentences(text: &str) -> Vec<String> {
        const MAX_CHARS: usize = 300;
        const MIN_CHARS: usize = 16;

        // 1. Primary split on sentence-final punctuation / newlines.
        let chars: Vec<char> = text.chars().collect();
        let mut raw: Vec<String> = Vec::new();
        let mut cur = String::new();
        for (i, &c) in chars.iter().enumerate() {
            if c == '\n' {
                let t = cur.trim();
                if !t.is_empty() {
                    raw.push(t.to_string());
                }
                cur.clear();
                continue;
            }
            cur.push(c);
            let ends_sentence = matches!(c, '.' | '!' | '?' | ';')
                && chars.get(i + 1).map_or(true, |n| n.is_whitespace());
            if ends_sentence {
                let t = cur.trim();
                if !t.is_empty() {
                    raw.push(t.to_string());
                }
                cur.clear();
            }
        }
        let t = cur.trim();
        if !t.is_empty() {
            raw.push(t.to_string());
        }

        // 2. Hard-wrap over-long chunks on word/comma boundaries.
        let mut wrapped: Vec<String> = Vec::new();
        for chunk in raw {
            if chunk.chars().count() <= MAX_CHARS {
                wrapped.push(chunk);
            } else {
                wrapped.extend(wrap_long(&chunk, MAX_CHARS));
            }
        }

        // 3. Merge tiny fragments into the previous chunk.
        let mut out: Vec<String> = Vec::new();
        for chunk in wrapped {
            if chunk.chars().count() < MIN_CHARS {
                if let Some(last) = out.last_mut() {
                    last.push(' ');
                    last.push_str(&chunk);
                    continue;
                }
            }
            out.push(chunk);
        }
        if out.is_empty() {
            let t = text.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
        out
    }

    /// Greedily pack words up to `max` chars, breaking early right after a comma
    /// once past the halfway point (a natural prosodic pause).
    fn wrap_long(s: &str, max: usize) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        for word in s.split_whitespace() {
            if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > max {
                out.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(word);
            if cur.ends_with(',') && cur.chars().count() >= max / 2 {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Warn that a whole input produced no audio. Not an error: skipping every chunk is
    /// a legitimate outcome for, say, a pasted block of pure punctuation. `failed` counts
    /// [`ChunkPrep::Failed`] chunks, which point at a broken espeak-ng rather than at the text.
    pub fn warn_nothing_spoken(failed: usize) {
        if failed > 0 {
            eprintln!(
                "[kokoro] nothing speakable in input ({failed} chunk(s) failed to phonemize \
                 — is espeak-ng installed?)"
            );
        } else {
            eprintln!("[kokoro] nothing speakable in input");
        }
    }

    /// Per-chunk metrics line (audio seconds, synth time, realtime factor).
    pub fn report_chunk(idx: usize, total: usize, token_len: usize, audio_len: usize, infer_secs: f64) {
        let audio_secs = audio_len as f64 / SAMPLE_RATE as f64;
        let rtf = infer_secs / audio_secs.max(1e-9);
        eprintln!(
            "[kokoro] [{n}/{total}] {tok} tokens | {audio:.2}s audio | infer {inf:.2}s | RTF {rtf:.3}",
            n = idx + 1,
            tok = token_len,
            audio = audio_secs,
            inf = infer_secs,
            rtf = rtf,
        );
    }

    /// Streaming audio sink: one long-lived `ffplay` fed by a background thread
    /// over a bounded channel. [`push`](Self::push)ed chunks play back-to-back with
    /// no gap between sentences; because `ffplay` consumes at realtime and its
    /// stdin pipe applies backpressure, the writer paces itself while the caller
    /// races ahead synthesizing later sentences — masking their compute behind
    /// playback of the earlier ones. Memory is bounded to `STREAM_BUFFER` queued
    /// chunks; a faster-than-realtime backend fills that and then blocks on
    /// `push`, a slower one (e.g. tract) simply never gets ahead and may underrun.
    pub struct StreamPlayer {
        tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>>,
        thread: Option<std::thread::JoinHandle<Result<()>>>,
    }

    /// Max sentences of synthesized audio buffered ahead of playback.
    const STREAM_BUFFER: usize = 32;

    /// Pick a raw-PCM sink command. Prefer `ffplay` (portable, needs ffmpeg+SDL);
    /// fall back to `pacat` (PulseAudio, standard on Linux incl. Termux).
    fn build_sink_command() -> Result<Command> {
        let sr = SAMPLE_RATE.to_string();
        if which("ffplay") {
            let mut c = Command::new("ffplay");
            c.args([
                "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
                // Start playback the instant the first samples arrive. Without these,
                // ffmpeg buffers ~analyzeduration (~5s) of input before starting, so a
                // single short sentence into the persistent `--serve` pipe never plays
                // until more data (the next sentence) arrives — one-shot only worked
                // because EOF flushed the buffer. probesize's minimum is 32 bytes.
                "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer",
                "-f", "f32le", "-ar", &sr, "-ac", "1", "-i", "pipe:0",
            ]);
            return Ok(c);
        }
        if which("pacat") {
            // pacat needs a running PulseAudio server. Start one if none is up —
            // the `--serve` daemon is launched detached, so there's no interactive
            // shell to run `pulseaudio --start` first.
            ensure_pulseaudio();
            // --latency-msec keeps the buffer small: default prebuf is ~2s, which
            // means clips shorter than that never trigger auto-start and only play
            // via drain-on-EOF — a race that can drop samples on Android/OpenSL.
            let mut c = Command::new("pacat");
            c.args([
                "--raw", "--format=float32le", "--rate", &sr, "--channels=1",
                "--latency-msec=100",
            ]);
            return Ok(c);
        }
        bail!("no audio sink found: install ffmpeg (for ffplay) or pulseaudio (for pacat)");
    }

    fn which(cmd: &str) -> bool {
        Command::new(cmd)
            .arg("--help")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Make sure a PulseAudio server is running before we hand back a `pacat` sink.
    ///
    /// Idempotent: `pulseaudio --check` exits 0 when a server is already up, so on
    /// desktop Linux (which already has a per-session daemon) this is a no-op. When
    /// none is running we start one — on Android/Termux that also needs the OpenSL ES
    /// sink module, and extra start args can be passed via `RYK_PULSE_ARGS`. If the
    /// `pulseaudio` binary is absent there's nothing to do; `pacat` then fails with
    /// its own clear error.
    fn ensure_pulseaudio() {
        if !which("pulseaudio") {
            return;
        }
        let running = Command::new("pulseaudio")
            .arg("--check")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if running {
            return;
        }
        eprintln!("[kokoro] no PulseAudio server running; starting one (pulseaudio --start)");
        let mut c = Command::new("pulseaudio");
        c.arg("--start").arg("--exit-idle-time=-1");
        // Termux/Android has no default sink; load the OpenSL ES output.
        if cfg!(target_os = "android") {
            c.arg("--load=module-sles-sink");
        }
        if let Ok(extra) = std::env::var("RYK_PULSE_ARGS") {
            c.args(extra.split_whitespace());
        }
        match c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status() {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("[kokoro] `pulseaudio --start` exited with {s}; pacat may fail"),
            Err(e) => eprintln!("[kokoro] could not start pulseaudio: {e}; pacat may fail"),
        }
    }

    impl StreamPlayer {
        pub fn new() -> Result<Self> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(STREAM_BUFFER);
            let mut cmd = build_sink_command()?;
            let thread = std::thread::spawn(move || -> Result<()> {
                let mut child = cmd
                    .stdin(Stdio::piped())
                    .spawn()
                    .context("spawning audio sink (ffplay/pacat)")?;
                let mut stdin = child.stdin.take().context("ffplay stdin unavailable")?;
                for chunk in rx.iter() {
                    stdin
                        .write_all(bytemuck::cast_slice::<f32, u8>(&chunk))
                        .context("writing pcm to ffplay")?;
                }
                drop(stdin); // EOF -> ffplay drains and exits (autoexit)
                let status = child.wait().context("waiting for ffplay")?;
                if !status.success() {
                    bail!("ffplay exited with {status}");
                }
                Ok(())
            });
            Ok(StreamPlayer { tx: Some(tx), thread: Some(thread) })
        }

        /// Queue a finished chunk. Blocks if the buffer is full (backpressure).
        pub fn push(&self, chunk: Vec<f32>) -> Result<()> {
            self.tx
                .as_ref()
                .context("player already finished")?
                .send(chunk)
                .map_err(|_| anyhow::anyhow!("playback thread stopped early"))
        }

        /// Close the queue and wait for playback to finish draining.
        pub fn finish(mut self) -> Result<()> {
            self.tx.take(); // drop sender -> writer loop ends after the last chunk
            match self.thread.take() {
                Some(h) => h.join().map_err(|_| anyhow::anyhow!("playback thread panicked"))?,
                None => Ok(()),
            }
        }
    }

    pub fn write_wav(path: &str, samples: &[f32]) -> Result<()> {
        write_wav_i16(path, samples, SAMPLE_RATE)
    }

    fn write_wav_i16(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
        let mut out = Vec::with_capacity(44 + samples.len() * 2);
        let data_len = (samples.len() * 2) as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, out)?;
        Ok(())
    }
}

/// Chunk-skipping behaviour: text that can't be spoken must not abort the run.
///
/// These need `espeak-ng` on PATH but no model assets — the decision to skip a chunk
/// is made before the voice file is ever read, so a bogus voice path is enough to
/// prove which branch was taken.
#[cfg(test)]
mod tests {
    use super::kokoro::*;
    use std::path::Path;

    fn prep(text: &str) -> anyhow::Result<Option<Prepared>> {
        prepare(text, "en-us", Path::new("/nonexistent/voice.bin"))
    }

    /// Punctuation-only lines — what pasted code is full of — phonemize to nothing.
    /// Reaching the (missing) voice file would mean we tried to synthesize them.
    #[test]
    fn punctuation_only_chunks_are_unspeakable() {
        for text in ["};", "}", "|>", "```", "???", ". . ."] {
            assert!(matches!(prep(text), Ok(None)), "expected {text:?} to be unspeakable");
        }
    }

    /// Real code phonemizes fine and must still be spoken, not skipped. It gets past
    /// the vocab filter, so the bogus voice path is what stops it.
    #[test]
    fn code_with_letters_is_still_spoken() {
        for text in ["#[derive(Debug)]", "=>", "*/", "for x in xs {"] {
            assert!(prep(text).is_err(), "expected {text:?} to reach the voice-file read");
        }
    }

    /// espeak-ng parses a leading `-` as a CLI option unless argv carries `--`.
    #[test]
    fn leading_dash_text_phonemizes() {
        let out = phonemize("- first bullet", "en-us").expect("leading dash must not fail");
        assert!(!out.is_empty());
    }

    /// The reported bug, end to end at the chunk level: prose with code pasted into it.
    /// The `};` line used to abort the whole run, so the prose after it was never spoken.
    /// `split_sentences`' merge rule absorbs most stray punctuation into a speakable
    /// neighbour; what reaches `prepare` alone is a fragment with no speakable neighbour
    /// to merge into (a leading one) or a punctuation run too long to merge.
    #[test]
    fn code_interleaved_with_prose_keeps_the_prose() {
        for (text, want_skipped) in [
            ("};\nThe function returns early there.", "};"),
            (
                "The parser handles that case.\n>>>>>>>>>>>>>>>>>>>>\nThen it recurses downward.",
                ">>>>>>>>>>>>>>>>>>>>",
            ),
        ] {
            let chunks = split_sentences(text);
            let skipped: Vec<&String> =
                chunks.iter().filter(|c| matches!(prep(c), Ok(None))).collect();
            assert_eq!(skipped, vec![want_skipped], "wrong chunks skipped for {text:?}");
            // Every other chunk still gets synthesized, so no prose is lost.
            assert!(chunks.iter().filter(|c| !matches!(prep(c), Ok(None))).count() >= 1);
        }
    }

    /// An input that is nothing but punctuation: every chunk skipped, no audio, no error.
    #[test]
    fn all_punctuation_input_is_entirely_unspeakable() {
        let chunks = split_sentences("};\n```\n???");
        assert!(chunks.iter().all(|c| matches!(prep(c), Ok(None))));
    }

    /// A missing voice file is a setup problem and must stay fatal, never a skip.
    #[test]
    fn voice_errors_are_not_swallowed() {
        assert!(matches!(prepare_or_skip("hello there", "en-us", Path::new("/nope.bin")), Err(_)));
    }
}
