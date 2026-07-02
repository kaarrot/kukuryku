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

The `Expand`'s target shape is the PyTorch `tensor.expand(-1,-1,-1)` ONNX lowering:
`Where(Equal(Concat[Gather(Shape(x),0), Gather(Shape(x),1), -1], -1), [1,1,1], …)`. The seq element
is `Shape(x)[1]` — i.e. **`sequence_length`, *not* a duration-derived length** — but tract can't
carry the symbol through that `Where`/`Equal`/`ConstantOfShape` control-flow pattern, so axis 1
collapses to `1` and the Concat's non-concat-axis equality rule can't unify `sequence_length` with
`1`. This is a *symbol-propagation* failure, not a genuinely dynamic axis. See the next section.

### What needs to be done — investigation (2026-06-30)

Used the probe's op inventory (`PROBE_OPS=1`) and node backtraces (`PROBE_DUMP_NODE`/`PROBE_FIND`) to
scope the *whole* model, not just the first wall.

**Op support is NOT the problem.** The model uses **50 distinct op types across 3012 nodes, and
tract has a registered parser for every one of them** (`MISSING OPS: none`; unregistered ops would
show up as `Unimplemented(<op>)` placeholders — there are none). This **retires the doc's biggest
fear**: the iSTFTNet vocoder's `STFT` (×1) and `ConvTranspose` (×6) are both present, plus `LSTM`,
`Resize`, `LayerNorm`, `Gemm`, etc. There is no "reimplement a chunk of DSP" task lurking behind the
shape wall. The remaining risk is op *correctness/optimization*, not op *existence*.

**The real problem is shape inference, and it splits into two distinct classes:**

1. **Symbol-propagation failures (fixable).** The text encoder repeatedly uses the PyTorch
   `expand(-1,…)` lowering (`Shape → Gather → Unsqueeze → Concat → Equal → Where → Expand`) and
   similar `Reshape`/control-flow patterns. The dims involved are just `sequence_length` (a clean
   input symbol), but tract loses the symbol through the `Where`/`Equal`/`ConstantOfShape` value
   logic. #1802 (and #550 when pinned) are instances. These are removable by constant-folding /
   `onnx-simplifier` on re-export, by graph surgery on the subgraph, or by improving tract's
   symbolic propagation.

2. **A genuinely data-dependent axis (NOT fixable by symbol propagation).** Kokoro's **length
   regulator** turns predicted phoneme durations into a frame axis:
   `#1865 /encoder/CumSum → #1866 Gather → #1870 /encoder/Range(0, total_frames, 1)`, where
   `total_frames = sum(round(durations))` — a tensor **value**, not a shape. tract's static analyser
   cannot represent a value-derived length as a shape symbol, so no re-export or simplifier removes
   this; it is intrinsic to the model. Everything downstream (decoder + iSTFTNet vocoder) runs on
   this frame axis. (The other data-dependent ops — `NonZero` #2989, `ScatterND` #3005, `Range`
   #2985, `m_source` `CumSum` #2067 — live *inside* the STFT / harmonic source generator and operate
   on constant or now-concrete data; they are not outer-graph shape drivers.)

**Recommended approach: split the model at the length regulator (graph surgery / two-stage).**
- **Stage 1 (tract):** `input_ids, style, speed` → BERT/text encoder + duration predictor →
  per-phoneme **durations** + phoneme-level prosody/features. Shapes here key only on
  `sequence_length`.
- **Rust glue:** round/clamp durations, `total_frames = sum`, build the phoneme→frame alignment
  matrix (a few loops). This removes the entire data-dependent shape region (CumSum/Range/alignment)
  from any ONNX graph tract has to analyse.
- **Stage 2 (tract):** frame-level features (concrete `total_frames` per utterance, fed as the input
  dim) → decoder + iSTFTNet vocoder → 24 kHz waveform.

Both stages then have static-or-cleanly-symbolic shapes, which also sidesteps class (1) (each
subgraph is simpler and the worst `expand` patterns are around the alignment). The cost is producing
two ONNX subgraphs — cleanest via a re-export from the HF PyTorch model split at the alignment, or
ONNX graph surgery (`onnx.utils.extract_model`) on the existing file. This supersedes the
"static-shape re-export" alternative below: re-export alone won't remove the data-dependent frame
axis; **the split is what makes the model tract-able.**

Effort re-estimate: with op support already proven present, the work is (a) the two-stage split +
Rust length-regulator (~1–2 days incl. parity checks) and (b) shaking out op-correctness/optimize
issues per stage (uncertain but bounded — no missing ops). The earlier "weeks / reimplement-a-chunk"
vocoder scenario looks unlikely.

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

## Stage 2 validation in tract (2026-07-02)

Extracted the decoder + iSTFTNet vocoder subgraph (`onnx.utils.extract_model`, cut at the two
encoder frame-level feature tensors `/encoder/MatMul_output_0` [1,640,T] and
`/encoder/MatMul_1_output_0` [1,512,T]; output `waveform`) and drove it through the probe with a
**concrete** frame axis (`PROBE_RUN_LEN`, batch axis pinned to 1). Result: with a concrete frame
axis the whole vocoder shape-resolves. It took two changes to get there, then surfaced one more:

1. **Batch-axis concretization.** The cut tensors carry two symbolic dims (`?,640,?`); the probe now
   pins the leading (batch) axis to 1 and the frame axis to `run_len`. (Pinning both to `run_len`
   produced a spurious `64 vs 4096` mismatch at the decoder Concat.)

2. **tract STFT patch (`tract-onnx/src/ops/fft.rs`).** tract's STFT enforced the ONNX opset-17
   contract (rank-3 signal `[batch, length, 1|2]`), but Kokoro's iSTFTNet generator reshapes to a
   **rank-2** `[batch, length]` real signal that onnxruntime accepts. Downstream `Transpose perm=
   [0,2,1,3]` confirms it still expects a rank-4 STFT output. Patched: relax the rank rule to allow
   rank 2 or 3, and lift a rank-2 signal to `[batch, length, 1]` in `wire()` (`AxisOp::Add`). This
   is an additive change — rank-3 callers are unaffected. **This is the kind of small tract patch the
   shakeout predicted, not a missing op.**

3. **Stale symbolic value_info.** `extract_model` copied a `num_samples` `dim_param` onto the
   `waveform` value_info; the final `Reshape` target is a plain `[1, -1]`, so tract computed
   `[1, 38400]` (64 frames × 600) but refused to unify the concrete value with the free `num_samples`
   symbol. Stripping all intermediate value_info (they are only hints) fixed it — a clean
   re-export/surgery would not carry the symbol.

After (1)–(3), **`analyse` passes through the entire Stage 2 graph (1950/1950 nodes)** — STFT, every
Conv/ConvTranspose, the AdaIN blocks, the harmonic source, the iSTFT. The remaining wall is now in
`into_optimized()` at a `Range(0, Shape(signal)[0], 1)` (the iSTFT index range): tract's `Shape`
yields `TDim`, so `Range`'s input supertype is `TDim` while a downstream consumer pins the output to
`I64` → `Impossible to unify TDim with I64`. This is a tract type-coherence issue (how `Shape`→`TDim`
flows into `Range`→`Gather`), more invasive than the STFT patch and best fixed carefully (cast, or a
declutter rule) rather than force-patched.

**Bottom line:** the two-stage split is validated. With a concrete frame axis the vocoder's *shapes*
fully resolve; what remains is a short series of small, localized tract op/type patches (STFT done;
`Range`/`Shape`→`TDim` next), exactly the "bounded op-correctness shakeout" this plan anticipated.
The "weeks / reimplement-a-chunk" scenario is effectively ruled out for Stage 2.

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
