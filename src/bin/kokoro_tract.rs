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
    let prep = kokoro::prepare(&text, &lang, &assets.voice_path)?;
    eprintln!("[kokoro] phonemes: {}", prep.phonemes);

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

    let infer_start = std::time::Instant::now();
    let audio = tract_backend::synthesize(&dir, &prep.ids, &prep.style, speed)?;
    let infer_secs = infer_start.elapsed().as_secs_f64();

    kokoro::emit(&audio, &prep.phonemes, prep.token_len, infer_secs, t0.elapsed().as_secs_f64())
}

/// Pure-Rust two-stage Kokoro inference on top of `tract`.
///
///   Stage 1: input_ids, style, speed -> phoneme features (640-ch, 512-ch) + durations
///   [Rust]:  round durations, total_frames = sum, build the [N, total_frames]
///            alignment matrix (block expansion: frame t belongs to phoneme i)
///   Stage 2: the two feature tensors + style + alignment -> waveform
mod tract_backend {
    use anyhow::{Context, Result, bail};
    use std::path::Path;
    use tract_onnx::prelude::*;

    // Stage-boundary tensor names (see tools/split_kokoro.py).
    const S1_FEAT_640: &str = "/encoder/Transpose_output_0";
    const S1_FEAT_512: &str = "/encoder/text_encoder/Transpose_2_output_0";
    const S2_ALIGNMENT: &str = "/encoder/Cast_4_output_0";

    /// Load an ONNX subgraph, pin each input's fact to the supplied tensor's
    /// concrete dtype+shape, optimize, and run. Inputs are matched to the model's
    /// declared inputs by name, so ordering is robust.
    fn run_stage(path: &Path, inputs: &[(&str, Tensor)]) -> Result<TVec<TValue>> {
        let t_load = std::time::Instant::now();
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading {}", path.display()))?;
        let load_secs = t_load.elapsed().as_secs_f64();
        let outlets = model.input_outlets()?.to_vec();
        let names: Vec<String> =
            outlets.iter().map(|o| model.node(o.node).name.clone()).collect();

        // Cast each supplied tensor to the model's declared input dtype, so an
        // f64-cast subgraph (KOKORO_TRACT_F64 experiment) runs transparently with
        // the same f32 features from stage 1.
        let mut ordered: Vec<Tensor> = Vec::with_capacity(names.len());
        for (ix, name) in names.iter().enumerate() {
            let (_, t) = inputs
                .iter()
                .find(|(n, _)| n == name)
                .with_context(|| format!("{}: no tensor supplied for input '{name}'", path.display()))?;
            let want = model.outlet_fact(outlets[ix])?.datum_type().unwrap_or_else(|| t.datum_type());
            let ct = t.cast_to_dt(want)?.into_owned();
            model.set_input_fact(ix, InferenceFact::dt_shape(want, ct.shape()))?;
            ordered.push(ct);
        }

        let t_opt = std::time::Instant::now();
        let runnable = model
            .into_optimized()
            .with_context(|| format!("optimizing {}", path.display()))?
            .into_runnable()?;
        let opt_secs = t_opt.elapsed().as_secs_f64();

        let run_inputs: TVec<TValue> = ordered.into_iter().map(|t| t.into()).collect();
        let stage = path.file_stem().and_then(|s| s.to_str()).unwrap_or("stage");
        let t_run = std::time::Instant::now();
        let out = if std::env::var_os("KOKORO_TRACT_PROFILE").is_some() {
            profile_run(&runnable, run_inputs, stage)?
        } else {
            runnable.run(run_inputs).with_context(|| format!("running {}", path.display()))?
        };
        eprintln!(
            "[kokoro]   {stage}: parse {load_secs:.2}s | optimize {opt_secs:.2}s | run {run:.2}s",
            run = t_run.elapsed().as_secs_f64(),
        );
        Ok(out)
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

    /// Point tract's matmul/conv kernels at a rayon thread pool. By default tract
    /// runs single-threaded (~1 core), which is the bulk of the gap vs onnxruntime
    /// (which saturates all cores). Thread count: KOKORO_TRACT_THREADS, else the
    /// machine's available parallelism.
    fn enable_multithreading() {
        use tract_linalg::multithread::{Executor, set_default_executor};
        let threads = std::env::var("KOKORO_TRACT_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0)
            .or_else(|| std::thread::available_parallelism().ok().map(|p| p.get()))
            .unwrap_or(1);
        if threads > 1 {
            set_default_executor(Executor::multithread(threads));
            eprintln!("[kokoro] tract executor: {threads} threads");
        }
    }

    pub fn synthesize(dir: &Path, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        let style_t = Tensor::from_shape(&[1, style.len()], style)?;

        // ---- Stage 1: encoder + duration predictor --------------------------
        let s1 = run_stage(
            &dir.join("stage1.onnx"),
            &[
                ("input_ids", Tensor::from_shape(&[1, n], ids)?),
                ("style", style_t.clone()),
                ("speed", Tensor::from_shape(&[1], &[speed])?),
            ],
        )?;
        // Outputs follow the graph's declared order (split_kokoro.py):
        //   [0] 640-ch features [1,640,N]  [1] 512-ch features [1,512,N]  [2] durations [1,N]
        dump("s1_feat640", &s1[0])?;
        dump("s1_feat512", &s1[1])?;
        dump("s1_dur", &s1[2])?;
        let feat640 = f32_tensor(&s1[0])?;
        let feat512 = f32_tensor(&s1[1])?;
        if feat640.shape().get(1) != Some(&640) || feat512.shape().get(1) != Some(&512) {
            bail!(
                "unexpected stage1 feature shapes: {:?}, {:?}",
                feat640.shape(),
                feat512.shape()
            );
        }
        let dur_t = s1[2].cast_to::<f32>()?;
        let durations = dur_t.to_array_view::<f32>()?;

        // ---- Rust length regulator: durations -> alignment matrix -----------
        // Round the (already clipped) per-phoneme durations to frame counts, then
        // build A[N, total_frames] where A[i,t] = 1 iff frame t belongs to phoneme
        // i (a block-diagonal expansion). Stage 2's MatMul does features @ A, so N
        // must be A's first (contracted) axis.
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
        eprintln!("[kokoro] length regulator: {n} phonemes -> {total_frames} frames");

        // ---- Stage 2: decoder + iSTFTNet vocoder ----------------------------
        // Threads help only the conv/matmul-heavy stage 2; stage 1's matmuls are
        // tiny and the rayon pool overhead makes it slower, so enable here only.
        enable_multithreading();
        let s2 = run_stage(
            &dir.join("stage2.onnx"),
            &[
                (S1_FEAT_640, feat640),
                (S1_FEAT_512, feat512),
                (S2_ALIGNMENT, alignment),
                ("style", style_t),
            ],
        )?;
        // s2[0] is the waveform; any extra outputs are debug probe points (added
        // to a stage2_dbg.onnx) — dump them all for offline diffing vs onnxruntime.
        for (i, o) in s2.iter().enumerate() {
            dump(&format!("s2_out{i}"), o)?;
        }
        let wav = s2[0].cast_to::<f32>()?;
        Ok(wav.to_array_view::<f32>()?.iter().copied().collect())
    }
}
