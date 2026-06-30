//! Step 0 of docs/tract-support-plan.md — bounded probe.
//!
//! Drives `tract` *directly* (bypassing the `ort` shim) against the Kokoro-82M
//! fp32 ONNX model with pinned input facts, and reports how far through the
//! load/optimize/run pipeline it gets. The point is to turn "unknown difficulty"
//! into a concrete answer:
//!   - loads & runs        -> shapes were the only blocker (~1 day of work)
//!   - dies at optimize    -> a static-shape / op-inference wall (the InferenceConcat)
//!   - dies at runnable/run -> a missing/unsupported op (likely the iSTFTNet vocoder)
//!
//! Run:
//!   cargo run --release --example tract_probe --features tract-probe -- [MODEL.onnx]
//!
//! Env:
//!   PROBE_LEN   if set, PIN input_ids dim1 to this fixed length (experiment B).
//!               if UNSET (default), leave input_ids symbolic — the model already
//!               declares it as [1, sequence_length] — and only feed a concrete
//!               length at run time (experiment A, the Step 1 approach).
//!   PROBE_RUN_LEN  concrete length used for the run() dummy tensors, default 32.
//!
//! Input contract (mirrors src/bin/kokoro.rs):
//!   input_ids : [1, N] i64   (phoneme ids, 0-padded both ends)
//!   style     : [1, 256] f32 (per-voice style row)
//!   speed     : [1] f32      (speed multiplier)

use tract_onnx::prelude::*;

const DEFAULT_MODEL: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--onnx-community--Kokoro-82M-v1.0-ONNX",
    "/snapshots/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx/model.onnx"
);

const STYLE_DIM: usize = 256;

fn main() -> TractResult<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    // When set, pin input_ids to a fixed length; otherwise keep it symbolic.
    let pin_len: Option<usize> = std::env::var("PROBE_LEN").ok().and_then(|s| s.parse().ok());
    let run_len: usize = std::env::var("PROBE_RUN_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    eprintln!("[probe] model       = {model_path}");
    match pin_len {
        Some(n) => eprintln!("[probe] input_ids   = PINNED to [1, {n}]"),
        None => eprintln!("[probe] input_ids   = SYMBOLIC (model's own sequence_length)"),
    }
    eprintln!("[probe] run length  = {run_len}");

    // --- load raw inference model -------------------------------------------
    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;
    eprintln!("[probe] OK: model_for_path (parsed {} nodes)", model.nodes().len());

    // --- discover input order + names, pin facts in that order --------------
    let inputs = model.input_outlets()?.to_vec();
    let names: Vec<String> = inputs.iter().map(|o| model.node(o.node).name.clone()).collect();
    eprintln!("[probe] inputs (in model order):");
    for (ix, name) in names.iter().enumerate() {
        eprintln!("  [{ix}] '{name}'  fact={:?}", model.outlet_fact(inputs[ix])?);
    }

    for (ix, name) in names.iter().enumerate() {
        let fact: InferenceFact = match name.as_str() {
            // Only pin input_ids when PROBE_LEN was given; otherwise leave the
            // model's symbolic sequence_length untouched.
            "input_ids" | "tokens" => match pin_len {
                Some(n) => i64::fact([1, n]).into(),
                None => continue,
            },
            "style" => f32::fact([1, STYLE_DIM]).into(),
            "speed" => f32::fact([1]).into(),
            other => {
                eprintln!("[probe] WARN: unrecognized input '{other}' — leaving its fact unpinned");
                continue;
            }
        };
        model.set_input_fact(ix, fact)?;
    }
    eprintln!("[probe] OK: pinned input facts");

    // --- optional: dump a node + its input producers (pre-analyse) ----------
    // Use PROBE_DUMP_NODE=1802 to inspect the failing Concat's neighbourhood.
    if let Some(id) = std::env::var("PROBE_DUMP_NODE").ok().and_then(|s| s.parse::<usize>().ok()) {
        eprintln!("[probe] --- dump of node #{id} and its inputs ---");
        dump_node(&model, id);
    }

    // --- shape inference: this is where the documented InferenceConcat dies --
    eprintln!("[probe] analysing (static shape inference)...");
    model.analyse(false)?;
    eprintln!("[probe] OK: analyse passed (shapes resolved through the whole graph)");

    // --- lower to typed, then optimize --------------------------------------
    eprintln!("[probe] into_optimized()...");
    let optimized = model.into_optimized()?;
    eprintln!("[probe] OK: into_optimized passed");

    eprintln!("[probe] into_runnable()...");
    let runnable = optimized.into_runnable()?;
    eprintln!("[probe] OK: into_runnable passed");

    // --- run with dummy inputs (values irrelevant; this exercises every op) --
    let mut tensors: TVec<TValue> = tvec!();
    for name in &names {
        let t = match name.as_str() {
            "input_ids" | "tokens" => Tensor::zero::<i64>(&[1, run_len])?,
            "style" => Tensor::zero::<f32>(&[1, STYLE_DIM])?,
            "speed" => {
                let mut s = Tensor::zero::<f32>(&[1])?;
                s.as_slice_mut::<f32>()?[0] = 1.0;
                s
            }
            _ => Tensor::zero::<f32>(&[1])?,
        };
        tensors.push(t.into());
    }

    eprintln!("[probe] run()...");
    let outputs = runnable.run(tensors)?;
    eprintln!("[probe] OK: run passed");
    for (i, o) in outputs.iter().enumerate() {
        eprintln!("  output[{i}] = {:?}", o.shape());
    }

    eprintln!("[probe] SUCCESS — tract loaded, optimized, and ran the full Kokoro graph.");
    Ok(())
}

/// Print a node, its op, and for each input the producing node (name/op) and the
/// pre-analyse fact on that wire. Helps see what shapes a failing node is fed.
fn dump_node(model: &InferenceModel, id: usize) {
    let node = model.node(id);
    eprintln!("  node #{id} '{}' op={}", node.name, node.op().name());
    for (i, inlet) in node.inputs.iter().enumerate() {
        let producer = model.node(inlet.node);
        let fact = model
            .outlet_fact(*inlet)
            .map(|f| format!("{f:?}"))
            .unwrap_or_else(|e| format!("<err: {e}>"));
        eprintln!(
            "    input[{i}] <- #{} '{}' op={} | fact={}",
            inlet.node,
            producer.name,
            producer.op().name(),
            fact
        );
    }
}
