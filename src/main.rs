// kokoro — minimal CPU text-to-speech using Kokoro-82M (ONNX) via `ort`.
//
// Pipeline: text -> espeak-ng IPA phonemes -> phoneme-id tokens -> Kokoro ONNX
//           (with a per-voice style vector) -> 24 kHz f32 waveform -> ffplay.
//
// Text comes from the command-line args (joined) or stdin if none are given.
// Env: KOKORO_VOICE (default "af_heart"), KOKORO_WAV (also write a WAV there).

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::value::TensorRef;

const REPO: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
const MODEL_FILE: &str = "onnx/model.onnx";
const LANG: &str = "en-us";
const SPEED: f32 = 1.0;
const SAMPLE_RATE: u32 = 24000;
const MAX_PHONEMES: usize = 510; // model context (pad token 0 wraps both ends)
const STYLE_DIM: usize = 256;

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

/// Phonemize text to IPA via espeak-ng. Slightly different from Kokoro's
/// reference phonemizer (misaki), so pronunciation is close but not identical.
/// Symbols not in the vocab (e.g. ZWJ ties) are dropped during tokenization.
fn phonemize(text: &str) -> Result<String> {
    let out = Command::new("espeak-ng")
        .args(["-q", "--ipa=3", "-v", LANG, text])
        .output()
        .context("running espeak-ng (is it installed?)")?;
    if !out.status.success() {
        bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8_lossy(&out.stdout).replace('\n', " ").trim().to_string())
}

/// onnxruntime is dlopened at runtime from `ORT_DYLIB_PATH`. If unset, auto-detect
/// the pip-installed onnxruntime .so so this works out of the box.
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
                // SAFE: single-threaded, before any ort call reads the var.
                unsafe { std::env::set_var("ORT_DYLIB_PATH", &p) };
            }
        }
    }
}

fn main() -> Result<()> {
    ensure_ort_dylib();

    let text = read_text()?;
    let voice = std::env::var("KOKORO_VOICE").unwrap_or_else(|_| "af_heart".to_string());

    // Phoneme -> id vocab (the fixed Kokoro vocab, 114 entries, embedded).
    let vocab: HashMap<String, i64> =
        serde_json::from_str(include_str!("kokoro_vocab.json")).context("parsing vocab")?;

    // Fetch model + voice (cached under ~/.cache/huggingface).
    let api = Api::new().context("creating hf-hub api")?;
    let repo = api.model(REPO.to_string());
    let model_path = api.model(REPO.to_string()).get(MODEL_FILE).context("fetching model")?;
    let voice_path = repo
        .get(&format!("voices/{voice}.bin"))
        .with_context(|| format!("fetching voice {voice}"))?;

    // Phonemize -> tokens.
    let phonemes = phonemize(&text)?;
    let mut tokens: Vec<i64> = phonemes
        .chars()
        .filter_map(|c| vocab.get(&c.to_string()).copied())
        .collect();
    if tokens.is_empty() {
        bail!("no recognizable phonemes produced for the input text");
    }
    if tokens.len() > MAX_PHONEMES {
        eprintln!(
            "[kokoro] note: {} phonemes truncated to {} (single-shot limit)",
            tokens.len(),
            MAX_PHONEMES
        );
        tokens.truncate(MAX_PHONEMES);
    }
    let token_len = tokens.len();

    // Voice style vector: the row indexed by (unpadded) token length.
    let voice_bytes = std::fs::read(&voice_path).context("reading voice file")?;
    let voice_f32: &[f32] = bytemuck::cast_slice(&voice_bytes);
    let row = token_len.min(voice_f32.len() / STYLE_DIM - 1);
    let style: Vec<f32> = voice_f32[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

    // input_ids padded with 0 at both ends.
    let mut ids: Vec<i64> = Vec::with_capacity(token_len + 2);
    ids.push(0);
    ids.extend_from_slice(&tokens);
    ids.push(0);

    // ONNX inference.
    let mut session = Session::builder()?
        .commit_from_file(&model_path)
        .context("loading kokoro onnx model")?;
    let id_tensor = TensorRef::from_array_view((vec![1_i64, ids.len() as i64], ids.as_slice()))?;
    let style_tensor = TensorRef::from_array_view((vec![1_i64, STYLE_DIM as i64], style.as_slice()))?;
    let speed_vec = vec![SPEED];
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

    eprintln!(
        "[kokoro] voice={voice} | {tok} tokens | {audio:.2}s @ {SAMPLE_RATE} Hz",
        tok = token_len,
        audio = audio.len() as f64 / SAMPLE_RATE as f64,
    );

    if let Ok(wav_path) = std::env::var("KOKORO_WAV") {
        write_wav_i16(&wav_path, audio)?;
        eprintln!("[kokoro] wrote {wav_path}");
    }

    play_via_ffplay(audio)?;
    Ok(())
}

/// Pipe raw f32le mono PCM to ffplay (smooth: the whole clip at once).
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
    if !child.wait().context("waiting for ffplay")?.success() {
        bail!("ffplay failed");
    }
    Ok(())
}

/// Minimal 16-bit PCM mono WAV writer.
fn write_wav_i16(path: &str, samples: &[f32]) -> Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}
