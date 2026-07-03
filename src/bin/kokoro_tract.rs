// kokoro-tract — pure-Rust CPU TTS using Kokoro-82M via `tract` (no onnxruntime,
// no native .so), the Termux/aarch64-friendly backend. Same pipeline and output as
// the `kokoro` binary; only the model-execution step differs.
//
// Kokoro's length regulator expands phoneme-level features to frame level via an
// alignment matrix whose length = sum(durations) — a value, not a static shape —
// so tract can't optimize the monolithic graph. We split it (tools/split_kokoro.py)
// into two subgraphs and rebuild the alignment in Rust between them. Each subgraph
// is optimized with a concrete phoneme count per utterance, which is what makes
// tract's static shape inference succeed. See docs/tract-support-plan.md.
//
// Config via env (shared with `kokoro`): KOKORO_VOICE / KOKORO_MODEL / KOKORO_LANG
//   / KOKORO_SPEED / KOKORO_WAV, plus:
//   KOKORO_TRACT_DIR   dir holding stage1.onnx + stage2.onnx (default: beside the
//                      HF-cached model.onnx). Produce them with tools/split_kokoro.py.
//   KOKORO_TRACT_DUMP  if set to a dir, dump stage-boundary tensors as raw f32.

use anyhow::{Context, Result};

use speak_tts::kokoro;

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();
    eprintln!("[kokoro] backend: tract (pure Rust, two-stage split)");

    let text = kokoro::read_text()?;
    let voice = kokoro::env_or("KOKORO_VOICE", "af_heart");
    let model_file = kokoro::env_or("KOKORO_MODEL", "onnx/model.onnx");
    let lang = kokoro::env_or("KOKORO_LANG", "en-us");
    let speed: f32 = kokoro::env_or("KOKORO_SPEED", "1.0").parse().unwrap_or(1.0);

    eprintln!("[kokoro] resolving assets...");
    let assets = kokoro::resolve_assets(&model_file, &voice)?;

    // ---- two-stage tract inference with a Rust length regulator ----
    let dir = match std::env::var_os("KOKORO_TRACT_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => assets
            .model_path
            .parent()
            .context("model path has no parent dir")?
            .to_path_buf(),
    };
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

/// Pure-Rust two-stage Kokoro inference on top of `tract`.
///
///   Stage 1: input_ids, style, speed -> phoneme features (640-ch, 512-ch) + durations
///   [Rust]:  round durations, total_frames = sum, build the [N, total_frames]
///            alignment matrix (block expansion: frame t belongs to phoneme i)
///   Stage 2: the two feature tensors + style + alignment -> waveform
mod tract_backend {
    use anyhow::{Context, Result, bail};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use tract_onnx::prelude::*;

    // Stage-boundary tensor names (see tools/split_kokoro.py).
    const S1_FEAT_640: &str = "/encoder/Transpose_output_0";
    const S1_FEAT_512: &str = "/encoder/text_encoder/Transpose_2_output_0";
    const S2_ALIGNMENT: &str = "/encoder/Cast_4_output_0";

    /// A subgraph compiled for one concrete input shape. tract can't optimize the
    /// split subgraphs with a *symbolic* length dim (the style-broadcast
    /// Expand/Concat hits `Impossible to unify Sym(N) with Val(1)`), so each plan is
    /// shape-specialized. `Pipeline` caches these keyed by (bucketed) length so the
    /// ~1–2s `into_optimized()` is paid once per bucket, not once per sentence.
    struct Stage {
        runnable: TypedRunnableModel<TypedModel>,
        input_names: Vec<String>,
    }

    impl Stage {
        /// Parse the subgraph, pin each input to its concrete shape (matched by
        /// name so input order is robust), and optimize.
        fn build(path: &Path, spec: &[(&str, &[usize])]) -> Result<Stage> {
            let mut model = tract_onnx::onnx()
                .model_for_path(path)
                .with_context(|| format!("loading {}", path.display()))?;
            let outlets = model.input_outlets()?.to_vec();
            let input_names: Vec<String> =
                outlets.iter().map(|o| model.node(o.node).name.clone()).collect();

            for (ix, name) in input_names.iter().enumerate() {
                let shape = spec
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, d)| *d)
                    .with_context(|| format!("{}: no shape spec for input '{name}'", path.display()))?;
                let dt = model.outlet_fact(outlets[ix])?.datum_type().unwrap_or_else(f32::datum_type);
                model.set_input_fact(ix, InferenceFact::dt_shape(dt, shape))?;
            }

            let runnable = model
                .into_optimized()
                .with_context(|| format!("optimizing {}", path.display()))?
                .into_runnable()?;
            Ok(Stage { runnable, input_names })
        }

        /// Run the cached plan; tensors are matched to declared inputs by name.
        fn run(&self, inputs: &[(&str, Tensor)], stage: &str) -> Result<TVec<TValue>> {
            let mut ordered: TVec<TValue> = TVec::with_capacity(self.input_names.len());
            for name in &self.input_names {
                let (_, t) = inputs
                    .iter()
                    .find(|(n, _)| n == name)
                    .with_context(|| format!("{stage}: no tensor supplied for input '{name}'"))?;
                ordered.push(t.clone().into());
            }
            if std::env::var_os("KOKORO_TRACT_PROFILE").is_some() {
                profile_run(&self.runnable, ordered, stage)
            } else {
                self.runnable.run(ordered).with_context(|| format!("running {stage}"))
            }
        }
    }

    /// Per-op profiler (KOKORO_TRACT_PROFILE): run the plan node-by-node, timing
    /// each node's eval and accumulating wall-time by op type, then print the
    /// biggest cost centres. Shows where stage-2's runtime actually goes.
    fn profile_run(
        runnable: &TypedRunnableModel<TypedModel>,
        inputs: TVec<TValue>,
        stage: &str,
    ) -> Result<TVec<TValue>> {
        use std::collections::HashMap;
        use tract_onnx::tract_core::plan::{SimpleState, eval};
        let mut state = SimpleState::new(runnable)?;
        // (total secs, call count) keyed by op type name.
        let mut acc: HashMap<String, (f64, usize)> = HashMap::new();
        let out = state.run_plan_with_eval(inputs, |session, op_state, node, input| {
            let t = std::time::Instant::now();
            let r = eval(session, op_state, node, input);
            let e = acc.entry(node.op().name().into_owned());
            let slot = e.or_insert((0.0, 0));
            slot.0 += t.elapsed().as_secs_f64();
            slot.1 += 1;
            r
        })?;
        let mut rows: Vec<(String, f64, usize)> =
            acc.into_iter().map(|(k, (s, c))| (k, s, c)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = rows.iter().map(|r| r.1).sum();
        eprintln!("[kokoro]   {stage} profile (op: total_s  calls  %):");
        for (op, secs, calls) in rows.iter().take(12) {
            eprintln!("[kokoro]     {op:<28} {secs:7.3}s  {calls:5}  {:4.1}%", 100.0 * secs / total);
        }
        Ok(out)
    }

    /// Debug: if KOKORO_TRACT_DUMP=<dir> is set, write an f32 tensor as raw
    /// little-endian bytes (shape known by the caller) for offline diffing.
    fn dump(name: &str, v: &TValue) -> Result<()> {
        if let Some(dir) = std::env::var_os("KOKORO_TRACT_DUMP") {
            let t = v.cast_to::<f32>()?;
            let data: Vec<f32> = t.to_array_view::<f32>()?.iter().copied().collect();
            std::fs::write(
                std::path::Path::new(&dir).join(format!("{name}.f32")),
                bytemuck::cast_slice::<f32, u8>(&data),
            )?;
        }
        Ok(())
    }

    /// Copy an output tensor as an f32 Tensor (features cross the stage boundary
    /// as f32; casts down if a subgraph ran in f64).
    fn f32_tensor(v: &TValue) -> Result<Tensor> {
        let t = v.cast_to::<f32>()?;
        let view = t.to_array_view::<f32>()?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data: Vec<f32> = view.iter().copied().collect();
        Ok(Tensor::from_shape(&shape, &data)?)
    }

    /// Build the tract thread pool (KOKORO_TRACT_THREADS, else available cores).
    /// tract runs single-threaded by default (~1 core); we scope this pool to the
    /// conv/matmul-heavy stage 2 only, since stage 1's matmuls are tiny and the pool
    /// overhead would slow them down.
    fn build_executor() -> tract_linalg::multithread::Executor {
        use tract_linalg::multithread::Executor;
        let threads = std::env::var("KOKORO_TRACT_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0)
            .or_else(|| std::thread::available_parallelism().ok().map(|p| p.get()))
            .unwrap_or(1);
        if threads > 1 {
            eprintln!("[kokoro] tract executor: {threads} threads");
            Executor::multithread(threads)
        } else {
            Executor::SingleThread
        }
    }

    /// The two-stage tract pipeline with a per-shape plan cache.
    ///
    /// tract compiles a plan for one concrete shape, and the model's global
    /// normalization (decoder instance-norm over frames, encoder norm/attention
    /// over phonemes) means we *cannot* pad to a shared bucket — padding poisons
    /// the output (empirically corr 0.73 for phoneme padding, 0.02 for frame
    /// padding). So plans are cached at their exact shape: a repeated phoneme count
    /// reuses stage 1, a repeated (phoneme, frame) pair reuses stage 2. This helps
    /// recurring lengths and re-runs; distinct lengths still compile once each.
    /// (The clean one-plan-fits-all path needs *symbolic* dims, which tract can't
    /// optimize here — the style-broadcast Expand/Concat hits `unify Sym(N) with
    /// Val(1)`.)
    pub struct Pipeline {
        stage1_path: PathBuf,
        stage2_path: PathBuf,
        executor: tract_linalg::multithread::Executor,
        stage1: HashMap<usize, Stage>,          // key: phoneme count N
        stage2: HashMap<(usize, usize), Stage>, // key: (phoneme count N, frame count F)
    }

    impl Pipeline {
        pub fn new(dir: &Path) -> Result<Pipeline> {
            Ok(Pipeline {
                stage1_path: dir.join("stage1.onnx"),
                stage2_path: dir.join("stage2.onnx"),
                executor: build_executor(),
                stage1: HashMap::new(),
                stage2: HashMap::new(),
            })
        }

        pub fn synthesize(&mut self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
            let n = ids.len();
            let style_t = Tensor::from_shape(&[1, style.len()], style)?;

            // ---- Stage 1: encoder + duration predictor (single-threaded) ----
            if !self.stage1.contains_key(&n) {
                let st = Stage::build(
                    &self.stage1_path,
                    &[("input_ids", &[1, n]), ("style", &[1, 256]), ("speed", &[1])],
                )?;
                self.stage1.insert(n, st);
            }
            let s1 = self.stage1[&n].run(
                &[
                    ("input_ids", Tensor::from_shape(&[1, n], ids)?),
                    ("style", style_t.clone()),
                    ("speed", Tensor::from_shape(&[1], &[speed])?),
                ],
                "stage1",
            )?;
            // Outputs (split_kokoro.py order): [0] 640-ch [1,640,N] [1] 512-ch
            // [1,512,N] [2] durations [1,N].
            dump("s1_feat640", &s1[0])?;
            dump("s1_feat512", &s1[1])?;
            dump("s1_dur", &s1[2])?;
            let feat640 = f32_tensor(&s1[0])?;
            let feat512 = f32_tensor(&s1[1])?;
            if feat640.shape().get(1) != Some(&640) || feat512.shape().get(1) != Some(&512) {
                bail!("unexpected stage1 feature shapes: {:?}, {:?}", feat640.shape(), feat512.shape());
            }
            let dur_t = s1[2].cast_to::<f32>()?;
            let durations = dur_t.to_array_view::<f32>()?;

            // ---- Rust length regulator: durations -> alignment matrix -------
            // Round per-phoneme durations to frame counts and build A[N, total_frames]
            // with A[i,t] = 1 iff frame t belongs to phoneme i (block expansion).
            let durs: Vec<usize> = durations.iter().map(|&d| d.round().max(0.0) as usize).collect();
            let total_frames: usize = durs.iter().sum();
            if total_frames == 0 {
                bail!("length regulator produced 0 frames (all durations rounded to 0)");
            }
            let mut align = vec![0f32; n * total_frames];
            let mut t = 0usize;
            for (i, &d) in durs.iter().enumerate() {
                for _ in 0..d {
                    align[i * total_frames + t] = 1.0;
                    t += 1;
                }
            }
            let alignment = Tensor::from_shape(&[n, total_frames], &align)?;

            // ---- Stage 2: decoder + iSTFTNet vocoder (multithreaded) --------
            let key = (n, total_frames);
            if !self.stage2.contains_key(&key) {
                let st = Stage::build(
                    &self.stage2_path,
                    &[
                        (S1_FEAT_640, &[1, 640, n]),
                        (S1_FEAT_512, &[1, 512, n]),
                        (S2_ALIGNMENT, &[n, total_frames]),
                        ("style", &[1, 256]),
                    ],
                )?;
                self.stage2.insert(key, st);
            }
            let stage2 = &self.stage2[&key];
            // Scope the thread pool to this run; stage 1 above stays single-threaded.
            let s2 = tract_linalg::multithread::multithread_tract_scope(self.executor.clone(), || {
                stage2.run(
                    &[
                        (S1_FEAT_640, feat640),
                        (S1_FEAT_512, feat512),
                        (S2_ALIGNMENT, alignment),
                        ("style", style_t),
                    ],
                    "stage2",
                )
            })?;
            for (i, o) in s2.iter().enumerate() {
                dump(&format!("s2_out{i}"), o)?;
            }
            let wav = s2[0].cast_to::<f32>()?;
            Ok(wav.to_array_view::<f32>()?.iter().copied().collect())
        }
    }
}
