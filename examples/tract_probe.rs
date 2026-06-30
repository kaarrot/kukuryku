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

    // --- PROBE_OPS=1: op-type inventory + missing-op scan, then exit ---------
    // Unregistered ONNX ops become UnimplementedOp placeholders (op name
    // "Unimplemented(<OpType>)"), so this enumerates exactly which ops tract
    // lacks for this model — independent of the shape-inference wall.
    if std::env::var("PROBE_OPS").is_ok() {
        op_inventory(&model);
        return Ok(());
    }

    // --- PROBE_FIND=<substr>: list node ids whose op name contains substr -----
    if let Ok(needle) = std::env::var("PROBE_FIND") {
        eprintln!("[probe] nodes with op name containing '{needle}':");
        for node in model.nodes() {
            if node.op().name().contains(&needle) {
                eprintln!("  #{} '{}' op={}", node.id, node.name, node.op().name());
            }
        }
        return Ok(());
    }

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
        let depth: usize = std::env::var("PROBE_DUMP_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        eprintln!("[probe] --- backtrace of node #{id} (depth {depth}) ---");
        dump_node(&model, id, depth, 0);
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

/// Histogram every node's op name and call out UnimplementedOp placeholders —
/// the ops tract has no parser for (the real "missing op" list for this model).
fn op_inventory(model: &InferenceModel) {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    for node in model.nodes() {
        let name = node.op().name().to_string();
        *counts.entry(name.clone()).or_default() += 1;
        if let Some(inner) = name.strip_prefix("Unimplemented(").and_then(|s| s.strip_suffix(")")) {
            *missing.entry(inner.to_string()).or_default() += 1;
        }
    }

    // sort histogram by count desc
    let mut by_count: Vec<(String, usize)> = counts.into_iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    eprintln!("[probe] op-type inventory ({} distinct ops):", by_count.len());
    for (name, c) in &by_count {
        eprintln!("  {c:>5}  {name}");
    }

    eprintln!();
    if missing.is_empty() {
        eprintln!("[probe] MISSING OPS: none — tract has a parser registered for every op in the model.");
    } else {
        let total: usize = missing.values().sum();
        eprintln!("[probe] MISSING OPS: {} node(s) across {} op type(s) tract cannot build:", total, missing.len());
        for (op, c) in &missing {
            eprintln!("  {c:>5}  {op}");
        }
    }
}

/// Recursively print a node and its input producers up to `depth` levels, with
/// the pre-analyse fact on each wire. Helps see the shape-source chain feeding a
/// failing node (e.g. what computes an Expand's target length).
fn dump_node(model: &InferenceModel, id: usize, depth: usize, indent: usize) {
    let pad = "  ".repeat(indent + 1);
    let node = model.node(id);
    eprintln!("{pad}#{id} '{}' op={}", node.name, node.op().name());
    for (i, inlet) in node.inputs.iter().enumerate() {
        let producer = model.node(inlet.node);
        let fact = model
            .outlet_fact(*inlet)
            .map(|f| format!("{f:?}"))
            .unwrap_or_else(|e| format!("<err: {e}>"));
        eprintln!(
            "{pad}  input[{i}] <- #{} '{}' op={} | fact={}",
            inlet.node,
            producer.name,
            producer.op().name(),
            fact
        );
        if depth > 0 {
            dump_node(model, inlet.node, depth - 1, indent + 2);
        }
    }
}
