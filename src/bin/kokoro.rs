// kokoro-onyx — soft-realtime CPU TTS using Kokoro-82M (ONNX) via onnxruntime (`ort`).
//
// Kokoro is a small non-autoregressive model: it predicts the whole utterance's
// audio in one ONNX forward pass, so it runs faster than realtime on CPU. This is
// the reference backend; the pure-Rust equivalent is the `ryk` binary.
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

use std::process::Command;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

use kukuryku::kokoro::{self, STYLE_DIM};

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
    eprintln!("[kokoro] backend: onnxruntime");

    let text = kokoro::read_text()?;
    let voice = kokoro::env_or("KOKORO_VOICE", "af_heart");
    let model_file = kokoro::env_or("KOKORO_MODEL", "onnx/model.onnx");
    let lang = kokoro::env_or("KOKORO_LANG", "en-us");
    let speed: f32 = kokoro::env_or("KOKORO_SPEED", "1.0").parse().unwrap_or(1.0);

    eprintln!("[kokoro] resolving assets...");
    let assets = kokoro::resolve_assets(&model_file, &voice)?;

    // Split into sentence chunks and stream: synthesize each, queue it, and let
    // playback of earlier sentences mask synthesis of later ones (see StreamPlayer).
    let sentences = kokoro::split_sentences(&text);
    eprintln!("[kokoro] {} sentence chunk(s)", sentences.len());

    // ---- ONNX inference: load the model once, run one forward pass per chunk ----
    eprintln!("[kokoro] loading model ({model_file})...");
    let mut session = Session::builder()?
        .commit_from_file(&assets.model_path)
        .context("loading kokoro onnx model")?;

    let player = kokoro::StreamPlayer::new()?;
    let want_wav = std::env::var("KOKORO_WAV").ok();
    let mut all: Vec<f32> = Vec::new();
    let (mut total_audio, mut total_infer) = (0usize, 0f64);

    for (i, sentence) in sentences.iter().enumerate() {
        let prep = kokoro::prepare(sentence, &lang, &assets.voice_path)?;

        let infer_start = std::time::Instant::now();
        let id_tensor =
            TensorRef::from_array_view((vec![1_i64, prep.ids.len() as i64], prep.ids.as_slice()))?;
        let style_tensor =
            TensorRef::from_array_view((vec![1_i64, STYLE_DIM as i64], prep.style.as_slice()))?;
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

        kokoro::report_chunk(i, sentences.len(), prep.token_len, audio.len(), infer_secs);
        total_audio += audio.len();
        total_infer += infer_secs;
        let audio = audio.to_vec();
        if want_wav.is_some() {
            all.extend_from_slice(&audio);
        }
        player.push(audio)?;
    }

    player.finish()?;
    if let Some(path) = want_wav {
        kokoro::write_wav(&path, &all)?;
        eprintln!("[kokoro] wrote {path}");
    }
    let audio_secs = total_audio as f64 / kokoro::SAMPLE_RATE as f64;
    eprintln!(
        "[kokoro] done: {audio_secs:.2}s audio | infer {total_infer:.2}s | RTF {:.3} | total {:.2}s",
        total_infer / audio_secs.max(1e-9),
        t0.elapsed().as_secs_f64(),
    );
    Ok(())
}
