//! Shared, backend-agnostic pieces of the Kokoro-82M TTS pipeline, used by both
//! the `kokoro` (onnxruntime) and `kokoro-tract` (pure-Rust) binaries. Everything
//! here is independent of the inference engine: text -> phonemes -> token ids +
//! style vector, asset resolution, and audio output (WAV + ffplay). Each binary
//! adds only its own model-execution step between `prepare` and `emit`.

pub mod kokoro {
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
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

    pub fn resolve_assets(model_file: &str, voice: &str) -> Result<Assets> {
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

    /// Report timing/RTF, optionally dump a WAV (KOKORO_WAV), and play via ffplay.
    pub fn emit(
        audio: &[f32],
        phonemes: &str,
        token_len: usize,
        infer_secs: f64,
        total_secs: f64,
    ) -> Result<()> {
        let audio_secs = audio.len() as f64 / SAMPLE_RATE as f64;
        let rtf = infer_secs / audio_secs.max(1e-9);
        eprintln!(
            "[kokoro] {ph} phonemes | {tok} tokens | {audio:.2}s audio @ {sr} Hz | infer {inf:.2}s | RTF {rtf:.3} | total {tot:.2}s",
            ph = phonemes.chars().count(),
            tok = token_len,
            audio = audio_secs,
            sr = SAMPLE_RATE,
            inf = infer_secs,
            rtf = rtf,
            tot = total_secs,
        );

        if let Ok(wav_path) = std::env::var("KOKORO_WAV") {
            write_wav_i16(&wav_path, audio, SAMPLE_RATE)?;
            eprintln!("[kokoro] wrote {wav_path}");
        }
        play_via_ffplay(audio)?;
        Ok(())
    }

    fn play_via_ffplay(samples: &[f32]) -> Result<()> {
        let mut child = Command::new("ffplay")
            .args([
                "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
                "-f", "f32le", "-ar", &SAMPLE_RATE.to_string(), "-ac", "1", "-i", "pipe:0",
            ])
            .stdin(Stdio::piped())
            .spawn()
            .context("spawning ffplay (is ffmpeg installed?)")?;
        {
            let mut stdin = child.stdin.take().context("ffplay stdin unavailable")?;
            stdin
                .write_all(bytemuck::cast_slice::<f32, u8>(samples))
                .context("writing pcm to ffplay")?;
        }
        let status = child.wait().context("waiting for ffplay")?;
        if !status.success() {
            bail!("ffplay exited with {status}");
        }
        Ok(())
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
