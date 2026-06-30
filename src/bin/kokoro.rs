// kokoro — a soft-realtime CPU TTS prototype using Kokoro-82M (ONNX) via `ort`.
//
// Unlike the Orpheus path (`speak`), Kokoro is a small non-autoregressive model:
// it predicts the whole utterance's audio in one ONNX forward pass, so it runs
// faster than realtime on CPU — playback is smooth, no streaming tricks needed.
//
// Pipeline: text -> espeak-ng IPA phonemes -> phoneme-id tokens -> Kokoro ONNX
//           (with a per-voice style vector) -> 24 kHz f32 waveform -> ffplay.
//
// Config via env:
//   KOKORO_VOICE  voice name (default "af_heart"; e.g. am_michael, bf_emma, ...)
//   KOKORO_MODEL  onnx file in the HF repo (default "onnx/model.onnx";
//                 try "onnx/model_q8f16.onnx" for a smaller/faster variant)
//   KOKORO_LANG   espeak voice/language (default "en-us")
//   KOKORO_SPEED  speech speed multiplier (default 1.0)
//   KOKORO_WAV    if set, also write a 16-bit PCM WAV to this path

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::value::TensorRef;

const REPO: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
const SAMPLE_RATE: u32 = 24000;
const MAX_PHONEMES: usize = 510; // model context; pad token 0 wraps both ends
const STYLE_DIM: usize = 256;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn read_text() -> Result<String> {
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
fn phonemize(text: &str, lang: &str) -> Result<String> {
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

/// With ort's `load-dynamic`, onnxruntime is dlopened at runtime from
/// `ORT_DYLIB_PATH`. If unset, auto-detect the pip-installed onnxruntime .so
/// (manylinux build, glibc-compatible) so this works out of the box.
fn ensure_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let out = Command::new("python3")
        .args([
            "-c",
            "import onnxruntime,glob,os;d=os.path.dirname(onnxruntime.__file__);print(sorted(glob.glob(d+'/capi/libonnxruntime.so*'))[-1])",
        ])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !p.is_empty() {
                eprintln!("[kokoro] onnxruntime: {p}");
                // SAFE: single-threaded, before any ort call reads the var.
                unsafe { std::env::set_var("ORT_DYLIB_PATH", &p) };
            }
        }
    }
}

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();
    ensure_ort_dylib();
    let text = read_text()?;
    let voice = env_or("KOKORO_VOICE", "af_heart");
    let model_file = env_or("KOKORO_MODEL", "onnx/model.onnx");
    let lang = env_or("KOKORO_LANG", "en-us");
    let speed: f32 = env_or("KOKORO_SPEED", "1.0").parse().unwrap_or(1.0);

    // Phoneme -> id vocab (embedded; the Kokoro fixed vocab, 114 entries).
    let vocab: HashMap<String, i64> =
        serde_json::from_str(include_str!("../kokoro_vocab.json")).context("parsing vocab")?;

    // ---- fetch assets (cached under ~/.cache/huggingface) ----
    eprintln!("[kokoro] resolving assets...");
    let api = Api::new().context("creating hf-hub api")?;
    let model_path = api
        .model(REPO.to_string())
        .get(&model_file)
        .with_context(|| format!("fetching {REPO}/{model_file}"))?;
    let voice_path = api
        .model(REPO.to_string())
        .get(&format!("voices/{voice}.bin"))
        .with_context(|| format!("fetching voice {voice}"))?;

    // ---- phonemize + tokenize ----
    let phonemes = phonemize(&text, &lang)?;
    eprintln!("[kokoro] phonemes: {phonemes}");
    let mut tokens: Vec<i64> = phonemes
        .chars()
        .filter_map(|c| vocab.get(&c.to_string()).copied())
        .collect();
    tokens.truncate(MAX_PHONEMES);
    if tokens.is_empty() {
        bail!("no recognizable phonemes produced for the input text");
    }
    let token_len = tokens.len();

    // ---- voice style vector: row indexed by (unpadded) token length ----
    let voice_bytes = std::fs::read(&voice_path).context("reading voice file")?;
    let voice_f32: &[f32] = bytemuck::cast_slice(&voice_bytes);
    let rows = voice_f32.len() / STYLE_DIM;
    let row = token_len.min(rows - 1);
    let style: Vec<f32> = voice_f32[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

    // ---- input ids, padded with 0 at both ends ----
    let mut ids: Vec<i64> = Vec::with_capacity(token_len + 2);
    ids.push(0);
    ids.extend_from_slice(&tokens);
    ids.push(0);

    // ---- ONNX inference ----
    eprintln!("[kokoro] loading model ({model_file})...");
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("loading kokoro onnx model")?;

    let infer_start = std::time::Instant::now();
    let id_tensor = TensorRef::from_array_view((vec![1_i64, ids.len() as i64], ids.as_slice()))?;
    let style_tensor = TensorRef::from_array_view((vec![1_i64, STYLE_DIM as i64], style.as_slice()))?;
    let speed_vec = vec![speed];
    let speed_tensor = TensorRef::from_array_view((vec![1_i64], speed_vec.as_slice()))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => id_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor,
        ])
        .context("running kokoro inference")?;
    let (_shape, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extracting audio output")?;
    let infer_secs = infer_start.elapsed().as_secs_f64();

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
        tot = t0.elapsed().as_secs_f64(),
    );

    // ---- optional WAV dump ----
    if let Ok(wav_path) = std::env::var("KOKORO_WAV") {
        write_wav_i16(&wav_path, audio, SAMPLE_RATE)?;
        eprintln!("[kokoro] wrote {wav_path}");
    }

    // ---- play aloud (smooth: whole clip handed to ffplay at once) ----
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
