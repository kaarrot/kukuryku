# Future work: pure-Rust (tract) backend for Kokoro

Status: **deferred** (2026-06-30). onnxruntime is the working backend; this doc captures the
goal, the known blocker, and a concrete plan for picking it up later.

## Purpose

Run the Kokoro-82M ONNX model through a **pure-Rust ONNX backend (tract)** instead of the native
ONNX Runtime — i.e. give tract enough shape/op support to load and execute *this specific model* so
the `kokoro` binary can synthesize and play audio with **no native `libonnxruntime.so` dependency**.

Why it matters:
- **Termux / aarch64 deployment.** ONNX Runtime needs a native `.so` built for the target libc.
  The glibc pip wheel won't load under Android's bionic libc; the workaround is extracting the
  `onnxruntime-android` AAR's `.so`. A pure-Rust backend removes that whole problem — it
  cross-compiles to any target with `cargo build`.
- **Static, dependency-free binaries** and simpler builds/CI everywhere.

Non-goal: replacing onnxruntime on glibc desktop, where it already works well (RTF ~0.4).

## Current state

- `kokoro` (`src/bin/kokoro.rs`) runs on **onnxruntime** via `ort` `load-dynamic` — works, smooth,
  soft-realtime (RTF ~0.4 on x86; verified audible).
- A `tract` cargo feature exists (`cargo build --features tract --bin kokoro`): `ort-tract`
  0.3.0+0.22 (tract-onnx 0.22), selected at runtime with `ort::set_api(ort_tract::api())`.
  It **builds and links**, but **fails at model load**:
  ```
  Failed to parse model: Failed analyse for node #1802
  "/encoder/predictor/text_encoder/Concat_1" InferenceConcat
  ```

## What we know / don't know

- The failure is **static shape inference**, not (necessarily) a missing op. tract resolves all
  shapes at load time and couldn't propagate shape through a `Concat` in the text encoder —
  almost certainly because of the model's **dynamic phoneme-length axis** and the fact that the
  `ort` shim calls `commit_from_file` with **no input-shape hints**.
- That node is **early** in the graph (text encoder). tract stopped there, so the duration
  predictor, decoder, and especially the **iSTFTNet vocoder** are unexamined. **The dominant
  unknown is what blockers lie behind the first one.**

## Plan

### Step 0 — Bounded probe (do this first; ~30–60 min)
Convert "unknown difficulty" into a concrete answer. Write a small standalone loader that drives
**tract directly** (bypassing the `ort` shim) with pinned input facts, and report how far through
the graph it gets:
```rust
let model = tract_onnx::onnx()
    .model_for_path("…/model.onnx")?
    .with_input_fact(0, i64::fact([1, sym]).into())?   // input_ids (phoneme len symbolic/fixed)
    .with_input_fact(1, f32::fact([1, 256]).into())?   // style
    .with_input_fact(2, f32::fact([1]).into())?        // speed
    .into_optimized()?
    .into_runnable()?;
```
Outcome decides everything:
- **Loads & runs** → shapes were the only issue; this is roughly a **1-day** job (Step 1 + 3).
- **Dies at op X** → we see the real wall (likely in the vocoder); estimate Step 2 accordingly.

### Step 0 results (2026-06-30)
Ran a direct-tract probe (`examples/tract_probe.rs`, `--features tract-probe`) against the fp32
`onnx/model.onnx`, bypassing the `ort` shim. tract parses all **3012 nodes**; the wall is **static
shape inference**, not a missing op (vocoder still unexamined behind it). Two cases:

- **Symbolic `input_ids` (`[1, sequence_length]`)** → reproduces the documented failure exactly:
  node **#1802 `/encoder/predictor/text_encoder/Concat_1`** `InferenceConcat`,
  `rule inputs[0].shape[1] == inputs[1].shape[1]`, *"Impossible to unify Sym(sequence_length) with
  Val(1)"*.
- **Pinned `input_ids` (e.g. `[1, 32]`)** → fails *earlier*, node **#550
  `/encoder/bert/embeddings/word_embeddings/Gather`**, unifying `Sym(sequence_length)` with
  `Val(32)`. Pinning a constant conflicts with the model's own `sequence_length` symbol baked into
  intermediate value_infos — so a fixed length is the *wrong* lever; keep it symbolic.

**Root cause of the #1802 Concat** (dumped via `PROBE_DUMP_NODE=1802`): it concatenates on the
feature axis (512+128→640):
- input[0] `?,?,512` ← `/encoder/bert_encoder/Add`
- input[1] `?,?,128` ← node **#1801 `/encoder/predictor/text_encoder/Expand` (`MultiBroadcastTo`)**

The `Expand`'s broadcast **target length is data-dependent** (computed from a runtime Shape chain),
so tract infers its seq axis as `1` instead of `sequence_length`, and the Concat's non-concat-axis
equality rule then can't unify `sequence_length` with `1`. This is the dynamic-shape pattern Step 2
warned about, surfacing already in the text encoder.

**Implication for next step:** the lever is making `Expand` (#1801) carry `sequence_length` on axis
1 — either by feeding tract symbolic shape info it can propagate through the Shape→…→Expand chain,
by graph surgery on that subgraph, or via the **static-shape re-export** track (which removes the
dynamic `Expand` target entirely and is probably the lowest-risk path to getting *past* #1802 and
finally seeing the vocoder).

### Step 1 — Shape resolution (likely hours–1 day)
Replace the `ort`-API inference path in `kokoro.rs` with a **direct tract** path so we control
input facts. Use a symbolic phoneme-length dim (or a fixed max, e.g. 510) so tract can analyse the
`Concat`. Keep the onnxruntime path as the default; make tract a parallel code path under the
`tract` feature.

### Step 2 — Op support (the real risk; days–weeks)
Once shapes resolve, every op must be supported. Expected-fine: LSTM, 1D/transposed conv, norms.
**Danger zone: the iSTFTNet vocoder** — if the ONNX export contains `STFT`/`ISTFT`/complex ops,
tract likely lacks them. For each unsupported op, choose:
- implement it in tract (DSP work; possibly upstream to sonos/tract), or
- **graph surgery**: cut the vocoder out of the ONNX and do the iSTFT in Rust after the model.

### Step 3 — Parity & perf check
Confirm tract output matches onnxruntime (compare WAVs, ~1e-3), and measure RTF. tract is usually
somewhat slower than onnxruntime; verify it's still < 1.0 on target hardware (esp. aarch64).

### Alternative track — static-shape re-export
Instead of bending tract, re-export Kokoro from PyTorch (model on HF) with **fixed input shapes**
+ `onnx-simplifier`/constant-folding. A static, simplified graph often loads in tract and avoids
some dynamic-op patterns. ~1 day, uncertain it removes *all* blockers; needs the torch model and
matching preprocessing.

## Effort & decision criteria

| Outcome | Effort |
|---|---|
| Shapes were the only issue | ~1 day |
| Shapes + a few unsupported ops | several days – ~2 weeks |
| Vocoder needs ops tract lacks | weeks / reimplement-a-chunk |

**Worth doing only if pure-Rust is a hard requirement** (fully static binary, or a target with no
onnxruntime build). For Termux specifically, the **onnxruntime-android `.so`** is a few hours of
known work and already solves deployment — prefer it unless the pure-Rust goal is itself the point.

## References

- Editable tract: the 0.22.3 tract crates are checked in under `third_party/tract/` and overridden
  via `[patch.crates-io]` in `Cargo.toml`, so edits to tract are picked up by *both* the probe and
  the `ort-tract` shim. Shape inference / `InferenceConcat` live in `third_party/tract/tract-hir`.
- Step 0 probe: `examples/tract_probe.rs` (`cargo run --release --example tract_probe --features tract-probe`; env `PROBE_LEN`, `PROBE_RUN_LEN`, `PROBE_DUMP_NODE`).
- Binary / backend wiring: `src/bin/kokoro.rs` (`ort::set_api(ort_tract::api())` under `#[cfg(feature = "tract")]`).
- Feature: `Cargo.toml` `[features] tract = ["dep:ort-tract", "ort/alternative-backend"]`.
- Versions: `ort` 2.0.0-rc.12, `ort-tract` 0.3.0+0.22 (tract-onnx 0.22).
- Model: `onnx-community/Kokoro-82M-v1.0-ONNX` → `onnx/model.onnx` (fp32; use for tract — quantized
  variants add QOperator ops tract is less likely to support).
- tract: https://github.com/sonos/tract — `with_input_fact`, `into_optimized`, `into_runnable`.
- Pipeline contract (phonemes, vocab, voice .bin, I/O) is in `README.md` (kokoro section).
