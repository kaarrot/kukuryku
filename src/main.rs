// speak — a CPU text-to-speech CLI on Candle (Orpheus-3B GGUF -> SNAC 24 kHz).
//
// Pipeline: text -> Orpheus (quantized_llama GGUF) emits SNAC audio tokens
//           -> de-interleave into 3 SNAC codebooks -> SNAC decode -> f32 PCM @ 24 kHz
//           -> play aloud via `ffplay` (pipes raw f32le to the WSLg/PulseAudio server).
//
// Config via env:
//   SPEAK_VOICE      voice name (default "tara"; also: leah jess leo dan mia zac zoe)
//   SPEAK_MODEL      gguf filename in the HF repo (default "orpheus-3b-Q4_K_L.gguf")
//   SPEAK_MAX_TOKENS hard cap on generated tokens (default 1200, ~8 s of audio)
//   SPEAK_TEMP       sampling temperature (default 0.6)
//   SPEAK_SEED       RNG seed (default 299792458)
//   SPEAK_WAV        if set, also write a 16-bit PCM WAV to this path (headless verify)

use std::io::Write;

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::snac::{Config as SnacConfig, Model as SnacModel};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

// HF repos / files for the model assets.
const GGUF_REPO: &str = "dahara1/orpheus-3b-0.1-ft_gguf";
// canopylabs/orpheus-3b-0.1-ft is gated (401 without an HF license token); the
// unsloth mirror is public and ships the identical Orpheus tokenizer.
const TOKENIZER_REPO: &str = "unsloth/orpheus-3b-0.1-ft";
const SNAC_WEIGHTS_REPO: &str = "lmz/candle-snac";
const SNAC_CONFIG_REPO: &str = "hubertsiuzdak/snac_24khz";

// Orpheus prompt-wrap / control tokens (model-agnostic, copied from the upstream
// candle orpheus example).
const START_TOKEN: u32 = 128259;
const END_TOKENS: [u32; 4] = [128009, 128260, 128261, 128257];
const STOP_TOKEN: u32 = 128258;

const SAMPLE_RATE: u32 = 24000;
const KNOWN_VOICES: [&str; 8] = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_text_from_args_or_stdin() -> Result<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return Ok(args.join(" "));
    }
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading text from stdin")?;
    let buf = buf.trim().to_string();
    if buf.is_empty() {
        bail!("no text provided (pass as args or pipe to stdin)");
    }
    Ok(buf)
}

fn main() -> Result<()> {
    let total_start = std::time::Instant::now();

    let text = read_text_from_args_or_stdin()?;
    let voice = env_or("SPEAK_VOICE", "tara");
    if !KNOWN_VOICES.contains(&voice.as_str()) {
        eprintln!("[speak] warning: unknown voice {voice:?}; known voices are {KNOWN_VOICES:?}");
    }
    let model_file = env_or("SPEAK_MODEL", "orpheus-3b-Q4_K_L.gguf");
    let max_tokens: usize = env_parse("SPEAK_MAX_TOKENS", 1200);
    let temperature: f64 = env_parse("SPEAK_TEMP", 0.6);
    let seed: u64 = env_parse("SPEAK_SEED", 299792458);

    let device = Device::Cpu;
    let api = Api::new().context("creating hf-hub api")?;

    // ---- fetch assets (cached under ~/.cache/huggingface) ----
    eprintln!("[speak] resolving assets (first run downloads ~2.3 GB)...");
    let gguf_path = api
        .model(GGUF_REPO.to_string())
        .get(&model_file)
        .with_context(|| format!("fetching {GGUF_REPO}/{model_file}"))?;
    let tokenizer_repo = env_or("SPEAK_TOKENIZER_REPO", TOKENIZER_REPO);
    let tokenizer_path = api
        .model(tokenizer_repo.clone())
        .get("tokenizer.json")
        .with_context(|| format!("fetching {tokenizer_repo}/tokenizer.json"))?;
    let snac_weights_path = api
        .model(SNAC_WEIGHTS_REPO.to_string())
        .get("snac_24khz.safetensors")
        .context("fetching snac_24khz.safetensors")?;
    let snac_config_path = api
        .model(SNAC_CONFIG_REPO.to_string())
        .get("config.json")
        .context("fetching snac config.json")?;

    // ---- load Orpheus GGUF via quantized_llama ----
    eprintln!("[speak] loading orpheus ({model_file})...");
    let mut gguf_reader = std::fs::File::open(&gguf_path).context("opening gguf file")?;
    let content = gguf_file::Content::read(&mut gguf_reader)
        .map_err(|e| e.with_path(&gguf_path))
        .context("reading gguf content")?;
    let mut model = ModelWeights::from_gguf(content, &mut gguf_reader, &device)
        .context("building quantized_llama model from gguf")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(anyhow::Error::msg)
        .context("loading tokenizer")?;

    // ---- load SNAC ----
    eprintln!("[speak] loading snac 24khz codec...");
    let snac_cfg: SnacConfig = serde_json::from_reader(
        std::fs::File::open(&snac_config_path).context("opening snac config.json")?,
    )
    .context("parsing snac config.json")?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[snac_weights_path], DType::F32, &device)
            .context("mmapping snac weights")?
    };
    let snac = SnacModel::new(&snac_cfg, vb).context("building snac model")?;

    // ---- build prompt: [START] + ids("{voice}: {text}") + [END...] ----
    let prompt = format!("{voice}: {text}");
    // add_special_tokens = true so the tokenizer prepends <|begin_of_text|> (128000),
    // matching the upstream orpheus example. Without BOS the model starts in an
    // ill-defined state and emits the stop token prematurely on some prompts.
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)
        .context("encoding prompt")?;
    let mut ids: Vec<u32> = Vec::new();
    ids.push(START_TOKEN);
    ids.extend_from_slice(encoded.get_ids());
    ids.extend_from_slice(&END_TOKENS);

    // ---- decode loop ----
    eprintln!("[speak] generating audio tokens...");
    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(0.9));
    let mut audio_tokens: Vec<u32> = Vec::new();
    let decode_start = std::time::Instant::now();
    let mut generated = 0usize;

    let prompt_len = ids.len();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    // Prompt is m>1, so it always takes the standard tensor forward; its last-row
    // logits seed the first sample.
    let mut logits_tensor = Some(model.forward(&input, 0)?.squeeze(0)?);
    let mut logits_slice: Vec<f32> = Vec::new();
    let mut on_fast_path = false; // true once forward_decode_token_into engaged
    let mut index_pos = prompt_len;

    for _ in 0..max_tokens {
        let next = if on_fast_path {
            logits_processor.sample_f32_slice(&logits_slice)?
        } else {
            logits_processor.sample(logits_tensor.as_ref().unwrap())?
        };
        generated += 1;
        if next == STOP_TOKEN {
            eprintln!("[speak] reached stop token");
            break;
        }
        if let Some(tok) = tokenizer.id_to_token(next) {
            if let Some(rest) = tok.strip_prefix("<custom_token_") {
                if let Some(num) = rest.strip_suffix('>') {
                    let parsed = num.parse::<u32>().context("parsing custom token id")?;
                    let offset = 10 + (audio_tokens.len() as u32 % 7) * 4096;
                    if parsed >= offset {
                        audio_tokens.push(parsed - offset);
                    }
                }
            }
        }
        // No-copy decode: writes logits into the reused Vec and bypasses the
        // per-token Tensor alloc. Returns false when the CPU executor is
        // ineligible (e.g. no AVX2 / executor disabled) -> tensor fallback.
        on_fast_path = model.forward_decode_token_into(next, index_pos, &mut logits_slice)?;
        if !on_fast_path {
            let next_input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            logits_tensor = Some(model.forward(&next_input, index_pos)?.squeeze(0)?);
        }
        index_pos += 1;
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    if generated > 0 {
        eprintln!(
            "[speak] decode fast path: {}",
            if on_fast_path { "engaged (CPU executor)" } else { "NOT engaged (tensor fallback)" }
        );
    }

    if audio_tokens.len() < 7 {
        bail!(
            "model produced too few audio tokens ({}) to decode; try a longer prompt or a higher-precision SPEAK_MODEL",
            audio_tokens.len()
        );
    }

    // ---- de-interleave audio tokens into 3 SNAC codebooks (7 tokens per frame) ----
    let mut codes0: Vec<u32> = Vec::new();
    let mut codes1: Vec<u32> = Vec::new();
    let mut codes2: Vec<u32> = Vec::new();
    for frame in audio_tokens.chunks_exact(7) {
        codes0.push(frame[0]);
        for i in [1, 4] {
            codes1.push(frame[i]);
        }
        for i in [2, 3, 5, 6] {
            codes2.push(frame[i]);
        }
    }
    let codes0 = Tensor::new(codes0, &device)?.unsqueeze(0)?;
    let codes1 = Tensor::new(codes1, &device)?.unsqueeze(0)?;
    let codes2 = Tensor::new(codes2, &device)?.unsqueeze(0)?;

    // ---- SNAC decode -> f32 PCM ----
    eprintln!("[speak] snac decode...");
    let pcm = snac.decode(&[&codes0, &codes1, &codes2])?;
    // pcm shape is (1, 1, samples); flatten to Vec<f32>.
    let pcm = pcm.i(0)?.i(0)?;
    let samples: Vec<f32> = pcm.to_vec1::<f32>()?;

    let audio_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let rtf = if decode_secs > 0.0 {
        audio_secs / decode_secs
    } else {
        f64::INFINITY
    };
    eprintln!(
        "[speak] {gen} tokens | {frames} frames | {audio:.2}s audio @ {sr} Hz | decode {tps:.1} tok/s | realtime x{rtf:.2} | total {total:.1}s",
        gen = generated,
        frames = audio_tokens.len() / 7,
        audio = audio_secs,
        sr = SAMPLE_RATE,
        tps = generated as f64 / decode_secs,
        rtf = rtf,
        total = total_start.elapsed().as_secs_f64(),
    );

    // ---- optional WAV dump ----
    if let Ok(wav_path) = std::env::var("SPEAK_WAV") {
        write_wav_i16(&wav_path, &samples, SAMPLE_RATE)
            .with_context(|| format!("writing wav to {wav_path}"))?;
        eprintln!("[speak] wrote {wav_path}");
    }

    // ---- play aloud via ffplay ----
    play_via_ffplay(&samples)?;
    Ok(())
}

/// Pipe raw f32le mono PCM to `ffplay`, which renders it through the system
/// (PulseAudio/WSLg) audio server. No temp file, no audio crate.
fn play_via_ffplay(samples: &[f32]) -> Result<()> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("ffplay")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nodisp",
            "-autoexit",
            "-f",
            "f32le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            "1",
            "-i",
            "pipe:0",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .context("spawning ffplay (is ffmpeg installed?)")?;
    {
        let mut stdin = child.stdin.take().context("ffplay stdin unavailable")?;
        // x86/aarch64 are little-endian, so the f32 slice is already f32le.
        stdin
            .write_all(bytemuck::cast_slice::<f32, u8>(samples))
            .context("writing pcm to ffplay")?;
    } // drop stdin -> EOF
    let status = child.wait().context("waiting for ffplay")?;
    if !status.success() {
        bail!("ffplay exited with {status}");
    }
    Ok(())
}

/// Minimal 16-bit PCM mono WAV writer (no extra deps).
fn write_wav_i16(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}
