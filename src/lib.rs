//! Shared, backend-agnostic pieces of the Kokoro-82M TTS pipeline, used by both
//! the `kokoro` (onnxruntime) and `kokoro-tract` (pure-Rust) binaries. Everything
//! here is independent of the inference engine: text -> phonemes -> token ids +
//! style vector, asset resolution, and audio output (WAV + ffplay). Each binary
//! adds only its own model-execution step between `prepare` and `emit`.

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

    /// Text from CLI args, else stdin.
    pub fn read_text() -> Result<String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
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
    pub fn phonemize(text: &str, lang: &str) -> Result<String> {
        let out = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "-v", lang, text])
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

    /// Directory checked for a project-local asset bundle before hitting the network.
    /// Reuses `KOKORO_TRACT_DIR` (where the split stages live) so one env var can point
    /// at a fully self-contained dir; defaults to `kokoro-onyx/` relative to the CWD.
    fn local_assets_dir() -> PathBuf {
        std::env::var_os("KOKORO_TRACT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("kokoro-onyx"))
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

    /// Phonemize + tokenize (Kokoro fixed vocab) + pick the voice style row.
    pub fn prepare(text: &str, lang: &str, voice_path: &std::path::Path) -> Result<Prepared> {
        let vocab: HashMap<String, i64> =
            serde_json::from_str(include_str!("kokoro_vocab.json")).context("parsing vocab")?;

        let phonemes = phonemize(text, lang)?;
        let mut tokens: Vec<i64> = phonemes
            .chars()
            .filter_map(|c| vocab.get(&c.to_string()).copied())
            .collect();
        tokens.truncate(MAX_PHONEMES);
        if tokens.is_empty() {
            bail!("no recognizable phonemes produced for the input text");
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

        Ok(Prepared { phonemes, ids, style, token_len })
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
                "-f", "f32le", "-ar", &sr, "-ac", "1", "-i", "pipe:0",
            ]);
            return Ok(c);
        }
        if which("pacat") {
            let mut c = Command::new("pacat");
            c.args(["--raw", "--format=float32le", "--rate", &sr, "--channels=1"]);
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
