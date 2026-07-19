// ryk (aka kokoro-tract) — pure-Rust CPU TTS using Kokoro-82M via `tract` (no onnxruntime,
// no native .so), the Termux/aarch64-friendly backend. Same pipeline and output as
// the `kokoro-ort` binary; only the model-execution step differs.
//
// Kokoro's length regulator expands phoneme-level features to frame level via an
// alignment matrix whose length = sum(durations) — a value, not a static shape —
// so tract can't optimize the monolithic graph. We split it (tools/split_kokoro.py)
// into two subgraphs and rebuild the alignment in Rust between them. Each subgraph
// is optimized with a concrete phoneme count per utterance, which is what makes
// tract's static shape inference succeed. See docs/tract-support-plan.md.
//
// `ryk --install-assets` downloads the split stages + voices + model.onnx from the
// pinned GitHub release into the OS-specific per-user data dir (see install.rs);
// pass --dev to install beside the executable instead (for `cargo run` on a checkout).
//
// For the low-latency `--serve` (warm daemon) / `--send` (client) modes, see serve.rs
// and docs/ryk-cli-and-daemon.md. The one-shot path below is unchanged by them.
//
// Config via env (mostly shared with `kokoro-ort`): KOKORO_VOICE / KOKORO_LANG /
//   KOKORO_SPEED / KOKORO_WAV, plus:
//   KOKORO_TRACT_DIR   dir holding stage1.onnx + stage2.onnx + voices/ (default
//                      lookup: OS-specific user data dir, else ./kokoro-onyx, else
//                      kokoro-onyx/ beside the binary). Produce it with
//                      tools/split_kokoro.py or `ryk --install-assets`.
//   KOKORO_TRACT_DUMP        if set to a dir, dump stage-boundary tensors as raw f32.
//   KOKORO_TRACT_NAN_TRACE   step the plan node-by-node and print the first node whose
//                            output contains a non-finite value (see nan_trace_run).

use anyhow::Result;

use kukuryku::{install, kokoro, tract_backend};

// Retain and reuse the large intermediate-tensor segments instead of returning them
// to the OS after every op (glibc mmaps/munmaps big blocks, costing first-touch page
// faults on each fresh output). ~8% infer win, bit-identical. See Cargo.toml note.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// One-line usage + the env vars that configure a run. Kept here (not in `kokoro`)
/// because the flag set is per-binary: `ryk` is the only one with `--install-assets`.
fn print_help() {
    print!(
        "ryk {version} — pure-Rust Kokoro-82M TTS (tract backend)

USAGE:
    ryk [TEXT...]           speak TEXT (all args joined); reads stdin if none given
    ryk --serve             run the warm daemon (compile once, serve over a socket)
    ryk --send [TEXT...]    send TEXT/stdin to the daemon (auto-starts it); low-latency
    ryk --install-assets [--dev]
                            download the split model + voices into the OS-specific
                            per-user data dir (Linux: ~/.local/share/kukuryku,
                            macOS: ~/Library/Application Support/kukuryku,
                            Windows: %APPDATA%\\kukuryku). --dev writes beside the
                            binary instead — for iterating on a checkout without
                            polluting the real user data dir.
    ryk --help | -h         show this help
    ryk --version | -V      show version

ENV:
    KOKORO_VOICE   voice name (default af_heart)
    KOKORO_LANG    espeak-ng language (default en-us)
    KOKORO_SPEED   speaking rate multiplier (default 1.0)
    KOKORO_WAV     also write synthesized audio to this WAV path
    KOKORO_TRACT_DIR  directory holding stage1.onnx + stage2.onnx + voices/
    KUKURYKU_ASSET_DIR  override the --install-assets target (absolute path, or
                        the literal `exe` for the exe-adjacent dir)
    RYK_SOCKET     daemon socket path (default $XDG_RUNTIME_DIR/ryk.sock)
",
        version = env!("CARGO_PKG_VERSION"),
    );
}

fn main() -> Result<()> {
    // Handle flags before read_text(), which would otherwise treat a flag as text
    // to speak (espeak-ng phonemizes "--help" into gibberish audio, not usage).
    match std::env::args().nth(1).as_deref() {
        Some("--install-assets") => {
            // `--dev` is the only supported extra arg here: force the
            // exe-adjacent install target (for `cargo run` on a checkout)
            // instead of the OS-specific user data dir. Any other trailing arg
            // is a mistake, not a filename.
            let dev = match std::env::args().nth(2).as_deref() {
                None => false,
                Some("--dev") => true,
                Some(other) => anyhow::bail!(
                    "unknown argument `{other}` to --install-assets (only `--dev` accepted)"
                ),
            };
            return install::run(dev);
        }
        Some("--help" | "-h") => {
            print_help();
            return Ok(());
        }
        Some("--version" | "-V") => {
            println!("ryk {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        // Warm-daemon modes (Unix-only): keep the compiled pipeline hot across
        // requests so editor-driven TTS doesn't pay the stage-compile each call.
        Some("--serve") => {
            #[cfg(unix)]
            {
                return kukuryku::serve::serve();
            }
            #[cfg(not(unix))]
            {
                anyhow::bail!("--serve is only supported on Unix");
            }
        }
        Some("--send") => {
            #[cfg(unix)]
            {
                return kukuryku::serve::send();
            }
            #[cfg(not(unix))]
            {
                anyhow::bail!("--send is only supported on Unix; use one-shot `ryk TEXT`");
            }
        }
        // An unknown `--flag` is a mistake, not something to read aloud.
        Some(arg) if arg.starts_with("--") => {
            anyhow::bail!("unknown flag `{arg}` (see `ryk --help`)");
        }
        _ => {}
    }

    let t0 = std::time::Instant::now();
    eprintln!("[kokoro] backend: tract (pure Rust, two-stage split)");

    let text = kokoro::read_text()?;
    let voice = kokoro::env_or("KOKORO_VOICE", "af_heart");
    let lang = kokoro::env_or("KOKORO_LANG", "en-us");
    let speed: f32 = kokoro::env_or("KOKORO_SPEED", "1.0").parse().unwrap_or(1.0);

    eprintln!("[kokoro] resolving assets...");
    // The tract path loads stage1.onnx + stage2.onnx, never the monolithic
    // model.onnx — so resolve the split-model dir directly rather than deriving it
    // from a model.onnx we'd have to download but never read.
    let assets = kokoro::resolve_assets_tract(&voice)?;
    let dir = assets.dir.clone();
    eprintln!("[kokoro] loading split model (stage1.onnx + stage2.onnx) from {}", dir.display());

    // Compile both subgraphs once (symbolic length dims) and reuse the plans for
    // every sentence — so per-sentence cost is just `run`, not re-optimization.
    let mut pipeline = tract_backend::Pipeline::new(&dir)?;

    let sentences = kokoro::split_sentences(&text);
    eprintln!("[kokoro] {} sentence chunk(s)", sentences.len());

    let player = kokoro::StreamPlayer::new()?;
    let want_wav = std::env::var("KOKORO_WAV").ok();
    let mut all: Vec<f32> = Vec::new();
    let (mut total_audio, mut total_infer) = (0usize, 0f64);

    for (i, sentence) in sentences.iter().enumerate() {
        let prep = kokoro::prepare(sentence, &lang, &assets.voice_path)?;
        let infer_start = std::time::Instant::now();
        let audio = pipeline.synthesize(&prep.ids, &prep.style, speed)?;
        let infer_secs = infer_start.elapsed().as_secs_f64();

        kokoro::report_chunk(i, sentences.len(), prep.token_len, audio.len(), infer_secs);
        total_audio += audio.len();
        total_infer += infer_secs;
        if want_wav.is_some() {
            all.extend_from_slice(&audio);
        }
        player.push(audio)?;
        if i == 0 {
            eprintln!("[kokoro] first audio at {:.2}s", t0.elapsed().as_secs_f64());
        }
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
