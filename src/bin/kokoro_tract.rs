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

// Retain and reuse the large intermediate-tensor segments instead of returning them
// to the OS after every op (glibc mmaps/munmaps big blocks, costing first-touch page
// faults on each fresh output). ~8% infer win, bit-identical. See Cargo.toml note.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    use tract_onnx::tract_hir::infer::ShapeFactoid;

    /// One input dimension in a plan spec: a fixed size, or a named symbol shared
    /// across inputs. Shared symbols let a *single* optimized plan serve every
    /// phoneme count N / frame count F, so `into_optimized()` is paid once total
    /// instead of once per distinct sentence length. See docs/tract-support-plan.md.
    #[derive(Clone, Copy)]
    enum Dim {
        Fixed(usize),
        Sym(&'static str),
    }
    use Dim::{Fixed, Sym};

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
        /// Parse the subgraph, pin each input to its (possibly symbolic) shape
        /// (matched by name so input order is robust), and optimize. Symbolic dims
        /// with a shared name are the same `Symbol`, so tract keeps the length axis
        /// free and one plan serves all lengths; fixed dims specialize as before.
        fn build(path: &Path, spec: &[(&str, &[Dim])]) -> Result<Stage> {
            let mut model = tract_onnx::onnx()
                .model_for_path(path)
                .with_context(|| format!("loading {}", path.display()))?;
            let outlets = model.input_outlets()?.to_vec();
            let input_names: Vec<String> =
                outlets.iter().map(|o| model.node(o.node).name.clone()).collect();

            for (ix, name) in input_names.iter().enumerate() {
                let dims = spec
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, d)| *d)
                    .with_context(|| format!("{}: no shape spec for input '{name}'", path.display()))?;
                let shape: Vec<TDim> = dims
                    .iter()
                    .map(|d| match d {
                        Fixed(v) => (*v as i64).to_dim(),
                        Sym(s) => model.sym(s).to_dim(),
                    })
                    .collect();
                let dt = model.outlet_fact(outlets[ix])?.datum_type().unwrap_or_else(f32::datum_type);
                model.set_input_fact(ix, InferenceFact::dt_shape(dt, ShapeFactoid::from(shape)))?;
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
    ///
    /// With KOKORO_TRACT_PROFILE_NODES=<N> also print the top-N *individual* nodes
    /// by time, tagged with their concrete input/output shapes — this is what pins
    /// which shapes an aggregated op bucket (e.g. the raw `Mul` bucket) actually is,
    /// so a fusion-gate fix can be scoped correctly. Shapes are read from the live
    /// tensors at eval, so they're concrete even under the symbolic plan.
    fn profile_run(
        runnable: &TypedRunnableModel<TypedModel>,
        inputs: TVec<TValue>,
        stage: &str,
    ) -> Result<TVec<TValue>> {
        use std::collections::HashMap;
        use tract_onnx::tract_core::plan::{SimpleState, eval};
        let top_nodes: Option<usize> = std::env::var("KOKORO_TRACT_PROFILE_NODES")
            .ok()
            .and_then(|v| v.parse().ok());
        let mut state = SimpleState::new(runnable)?;
        // (total secs, call count) keyed by op type name.
        let mut acc: HashMap<String, (f64, usize)> = HashMap::new();
        // Per-node accumulator (only populated when top_nodes is set): node id ->
        // (op name, total secs, calls, last-seen input shapes, last-seen out shapes).
        let mut per_node: HashMap<usize, (String, f64, usize, String, String)> = HashMap::new();
        let out = state.run_plan_with_eval(inputs, |session, op_state, node, input| {
            let in_shapes = top_nodes.map(|_| shape_tag(input.iter().map(|t| t.shape())));
            let t = std::time::Instant::now();
            let r = eval(session, op_state, node, input);
            let dt = t.elapsed().as_secs_f64();
            let e = acc.entry(node.op().name().into_owned());
            let slot = e.or_insert((0.0, 0));
            slot.0 += dt;
            slot.1 += 1;
            if let Some(in_shapes) = in_shapes {
                let out_shapes = r
                    .as_ref()
                    .map(|o| shape_tag(o.iter().map(|t| t.shape())))
                    .unwrap_or_default();
                let e = per_node.entry(node.id).or_insert_with(|| {
                    (node.op().name().into_owned(), 0.0, 0, in_shapes, out_shapes)
                });
                e.1 += dt;
                e.2 += 1;
            }
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
        if let Some(n) = top_nodes {
            let mut nodes: Vec<_> = per_node.into_values().collect();
            nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("[kokoro]   {stage} top-{n} nodes (op  total_s  calls  in -> out):");
            for (op, secs, calls, ins, outs) in nodes.iter().take(n) {
                eprintln!(
                    "[kokoro]     {op:<20} {secs:7.3}s  {calls:4}  {ins} -> {outs}",
                );
            }
        }
        Ok(out)
    }

    /// Render an iterator of tensor shapes as a compact tag like `[1,512,377]x[1,512,1]`.
    fn shape_tag<'a>(shapes: impl Iterator<Item = &'a [usize]>) -> String {
        shapes
            .map(|s| {
                let dims: Vec<String> = s.iter().map(|d| d.to_string()).collect();
                format!("[{}]", dims.join(","))
            })
            .collect::<Vec<_>>()
            .join("x")
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

    /// A compiled stage: either a single *symbolic* plan that serves every length,
    /// or — if the subgraph won't optimize symbolically — a lazily populated
    /// per-exact-shape cache (the previous behaviour, kept as a fallback).
    ///
    /// The symbolic plan is the win: tract's `into_optimized()` (~1–4 s) is paid
    /// once total, not once per distinct sentence length, so streaming a paragraph
    /// of differently-sized sentences no longer recompiles each one. It requires
    /// the split subgraphs produced by `tools/split_kokoro.py` (which rewires two
    /// `Expand` targets so the phoneme/frame axes stay symbolic) plus two small
    /// tract patches (symbolic `Resize` scale, symbolic `Slice` end-clamp). We
    /// still cannot *pad* to a shared bucket — the model's global normalization
    /// poisons padded output (corr 0.73 phoneme / 0.02 frame) — but a symbolic plan
    /// never pads; it resolves N/F from each run's real input shapes.
    enum StagePlan {
        Symbolic(Stage),
        PerShape(HashMap<Vec<usize>, Stage>),
    }

    impl StagePlan {
        /// Build a single symbolic plan; on optimize failure, degrade to per-shape.
        fn build(path: &Path, spec: &[(&str, &[Dim])], name: &str) -> StagePlan {
            // Debug lever: force the concrete per-shape path (which enables tract's
            // concrete-shape-gated conv fast paths — lazy im2col + depthwise) so we
            // can bench conv run-speed symbolic-vs-concrete. See docs conv section.
            if std::env::var_os("KOKORO_TRACT_FORCE_PERSHAPE").is_some() {
                eprintln!("[kokoro] {name}: FORCE_PERSHAPE — using per-exact-shape plans");
                return StagePlan::PerShape(HashMap::new());
            }
            match Stage::build(path, spec) {
                Ok(st) => {
                    eprintln!("[kokoro] {name}: compiled one symbolic plan (length-independent)");
                    StagePlan::Symbolic(st)
                }
                Err(e) => {
                    eprintln!(
                        "[kokoro] {name}: symbolic optimize failed ({e:#}); \
                         falling back to per-exact-shape plans"
                    );
                    StagePlan::PerShape(HashMap::new())
                }
            }
        }

        /// The plan to run for a given concrete shape: the symbolic plan as-is, or
        /// the cached exact-shape plan (compiled on first sight of that shape).
        fn get(&mut self, path: &Path, concrete: &[(&str, &[usize])]) -> Result<&Stage> {
            match self {
                StagePlan::Symbolic(st) => Ok(st),
                StagePlan::PerShape(cache) => {
                    let key: Vec<usize> =
                        concrete.iter().flat_map(|(_, d)| d.iter().copied()).collect();
                    if !cache.contains_key(&key) {
                        let owned: Vec<(&str, Vec<Dim>)> = concrete
                            .iter()
                            .map(|(nm, d)| (*nm, d.iter().map(|&v| Fixed(v)).collect()))
                            .collect();
                        let spec: Vec<(&str, &[Dim])> =
                            owned.iter().map(|(nm, d)| (*nm, d.as_slice())).collect();
                        cache.insert(key.clone(), Stage::build(path, &spec)?);
                    }
                    Ok(&cache[&key])
                }
            }
        }
    }

    /// The two-stage tract pipeline: one symbolic plan per stage (see [`StagePlan`]),
    /// with a Rust length regulator between them.
    pub struct Pipeline {
        stage1_path: PathBuf,
        stage2_path: PathBuf,
        executor: tract_linalg::multithread::Executor,
        stage1: StagePlan,
        stage2: StagePlan,
    }

    impl Pipeline {
        pub fn new(dir: &Path) -> Result<Pipeline> {
            let stage1_path = dir.join("stage1.onnx");
            let stage2_path = dir.join("stage2.onnx");
            let executor = build_executor();
            // Compile each stage once with shared symbolic length dims: N (phoneme
            // count) across stage 1 + the two stage-2 feature tensors, and F (frame
            // count) on the alignment's frame axis.
            let stage1 = StagePlan::build(
                &stage1_path,
                &[
                    ("input_ids", &[Fixed(1), Sym("N")]),
                    ("style", &[Fixed(1), Fixed(256)]),
                    ("speed", &[Fixed(1)]),
                ],
                "stage1",
            );
            let stage2 = StagePlan::build(
                &stage2_path,
                &[
                    (S1_FEAT_640, &[Fixed(1), Fixed(640), Sym("N")]),
                    (S1_FEAT_512, &[Fixed(1), Fixed(512), Sym("N")]),
                    (S2_ALIGNMENT, &[Sym("N"), Sym("F")]),
                    ("style", &[Fixed(1), Fixed(256)]),
                ],
                "stage2",
            );
            Ok(Pipeline { stage1_path, stage2_path, executor, stage1, stage2 })
        }

        pub fn synthesize(&mut self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
            let n = ids.len();
            let style_t = Tensor::from_shape(&[1, style.len()], style)?;

            // ---- Stage 1: encoder + duration predictor (single-threaded) ----
            // Tier 7 Lever 2 A/B'd wrapping this in the stage-2 thread pool: it
            // regressed +31% (8.97s -> 11.75s). The serial LSTM predictor and the
            // small per-op GEMMs pay more in thread-dispatch overhead than the BERT
            // encoder saves, so stage 1 stays single-threaded by design.
            let s1 = self
                .stage1
                .get(&self.stage1_path, &[("input_ids", &[1, n]), ("style", &[1, 256]), ("speed", &[1])])?
                .run(
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
            let executor = self.executor.clone();
            let stage2 = self.stage2.get(
                &self.stage2_path,
                &[
                    (S1_FEAT_640, &[1, 640, n]),
                    (S1_FEAT_512, &[1, 512, n]),
                    (S2_ALIGNMENT, &[n, total_frames]),
                    ("style", &[1, 256]),
                ],
            )?;
            // Scope the thread pool to this run; stage 1 above stays single-threaded.
            let s2 = tract_linalg::multithread::multithread_tract_scope(executor, || {
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
