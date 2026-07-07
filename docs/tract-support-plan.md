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

- `kokoro-tract` (`src/bin/kokoro_tract.rs`) is the **pure-Rust** backend (tract, no native `.so`)
  and now runs **end-to-end, faster than realtime**. The monolithic-graph load failure below was
  solved by splitting the model at the length regulator into two subgraphs (`tools/split_kokoro.py`)
  with a Rust length regulator between them, then optimizing each as **one symbolic plan**; the
  remaining perf work is the Tier 1–7 arc documented in this file.
- `kokoro` (`src/bin/kokoro.rs`) still runs on **onnxruntime** via `ort` `load-dynamic` and remains
  the fast reference (RTF ~0.35 on x86).
- **Latest measured comparison** (same host, WSL2 16-thread; two-sentence streamed run, `af_heart`):

  | Utterance | tract (`kokoro-tract`) | onnxruntime (`kokoro`) |
  |---|---|---|
  | 242 tokens / 14.60 s audio | infer 7.39 s · **RTF 0.506** | infer 5.04 s · **RTF 0.345** |
  | 221 tokens / 12.97 s audio | infer 6.60 s · **RTF 0.509** | infer 4.51 s · **RTF 0.347** |

  Both are faster than realtime; tract is now **~1.47× onnxruntime** (down from ~3.6× at the start of
  the perf arc). Fidelity vs onnxruntime is corr ~0.976. See the Tier 1–7 sections below.

### Original blocker (2026-06-30, now solved — kept for history)

The first `tract` attempt drove tract-onnx through the `ort-tract` shim
(`ort::set_api(ort_tract::api())`) on the *whole* graph. It **built and linked** but **failed at
model load**:
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

## Stage 1 validated + finalized split (2026-07-02)

Finalized the cut and validated Stage 1. The split is now reproducible via
`tools/split_kokoro.py` (uses `onnx.utils.extract_model`, then strips stale
value_info + output shapes — see below):

- **Stage 1 (tract):** `input_ids, style, speed` → prosody features
  `/encoder/Transpose_output_0` [1,640,N], text features
  `/encoder/text_encoder/Transpose_2_output_0` [1,512,N], and per-phoneme
  durations `/encoder/Clip_output_0` [1,N]. Cut is *upstream* of the frame axis.
- **Rust length regulator:** round durations, `total_frames = sum`, rebuild the
  boolean alignment matrix (the model's `And(GreaterOrEqual, Less)` over
  `Range(0, total_frames)` vs `CumSum(durations)`) as the tensor the two MatMuls
  consume (`/encoder/Cast_4_output_0`, shape `[N, total_frames]`).
- **Stage 2 (tract):** the two phoneme feature tensors + `style` + the alignment
  matrix → the two alignment MatMuls → decoder + iSTFTNet → `waveform`.

**Stage 1 fully loads, optimizes, AND runs in tract** (probe: analyse → optimize →
runnable → run all pass; outputs `[1,640,64]`, `[1,512,64]`, `[1,64]`). **No tract
patch was needed.**

**#1802 dissolves.** The class-1 symbol-propagation blocker (#1802 `InferenceConcat`)
is *not* a real blocker for a per-utterance execution model: it was tract failing
to *propagate* the `sequence_length` symbol through `Where`/`Equal`, but when N is
pinned to a concrete value (which we always know per utterance) the symbol never
needs to propagate. The one catch: `extract_model` copies stale symbolic
value_info (`sequence_length`, `num_samples`) that *conflicts* with the concrete
pin — stripping all intermediate value_info + clearing output shapes fixes it (and
also restores the real `input_ids` i64 dtype). `tools/split_kokoro.py` does this.

**Range/`Shape`→`TDim` fix (done).** The finalized **Stage 2** (MatMuls included,
alignment fed as `[N, total_frames]`) initially hit `Range(0, Shape(signal)[0], 1)`
in the iSTFT → `TDim` vs `I64`. Root cause: tract's core `Range` already resolves a
TDim-bounded range to concrete **I64** indices (`tract-core` `Range::output_facts`),
but the **HIR** expansion's inference rule forced the output datum type to
`super_type_for([I64, TDim, I64])` = `TDim`, contradicting the core op and any
downstream consumer that pinned the indices to I64. Patched
`tract-hir/src/ops/array/range.rs` to mirror the core op: when the supertype is
TDim, the inferred output is I64. Small and low-risk — only changes the TDim-input
case, to the type the core op already produces.

**Status: BOTH stages fully load, optimize, AND run in tract.** Stage 1 →
`[1,640,N]`, `[1,512,N]`, `[1,N]`; Stage 2 → `waveform [1, total_frames*hop]`
(validated at N=frames=64 → 38400 samples). The two tract patches (STFT rank-2,
Range TDim→I64) are the entire tract-code footprint. Remaining work is pure Rust:
(a) the length regulator (durations → alignment matrix), (b) wiring the two tract
sessions behind the kokoro backend. No missing ops, no re-export required.

## Two-stage pipeline wired + validated end-to-end (2026-07-02)

The `kokoro-tract` binary (`cargo build --features tract`, or
`--no-default-features --features tract --bin kokoro-tract` for a no-onnxruntime
build) now runs the full pure-Rust two-stage pipeline. Key design decision: **it drives `tract-onnx`
directly, not the `ort-tract` shim.** The shim optimizes the model at session-load
from the ONNX-declared *symbolic* `sequence_length`, which re-hits the #1802 Concat
(`Sym(sequence_length)` vs `Val(1)`); driving tract directly lets us pin a concrete
N per utterance before `into_optimized()`, which is what makes shape inference
succeed. The `tract` feature was repointed from `ort-tract` to `tract-onnx`.

Implementation (`src/bin/kokoro.rs`, `mod tract_backend`): load `stage1.onnx`, pin
input facts to concrete shapes, optimize+run → features + durations; Rust length
regulator builds the `[N, total_frames]` block-expansion alignment; load
`stage2.onnx`, feed the two features + `style` + alignment, optimize+run → waveform.
Subgraph dir via `KOKORO_TRACT_DIR` (default: next to `model.onnx`); produce the
subgraphs with `tools/split_kokoro.py`.

**Numerical validation (input "Hello world.", vs onnxruntime full model):**

| Check | Result |
|---|---|
| Split fidelity: stage1+stage2 **in ORT** vs full model | **corr 1.0** (bit-exact) |
| Length regulator: Rust alignment vs model `Cast_4` | **exact** (identical [15,65]) |
| Stage 1 in **tract** vs ORT (features + durations) | **corr 1.0**, max|Δ| 0.0 |
| Stage 2 in **tract** vs ORT (same inputs) | **corr 0.942**, rel-RMS 0.36 |
| End-to-end tract vs ORT waveform | corr 0.941 |

So the split, the length regulator, and Stage 1 are exact; the **only** correctness
gap is Stage 2's decoder + iSTFTNet vocoder (~0.94 — recognizable speech with
spectral artifacts). The nonstandard tract behavior there is the STFT rank-2 patch,
so that (and the iSTFT reconstruction) is the prime suspect for the follow-up. Two
onnxruntime runs are bit-identical, so the gap is a real tract op difference, not
model stochasticity. `KOKORO_TRACT_DUMP=<dir>` dumps stage-boundary tensors for
diffing.

**Perf:** RTF ~0.62 after the stage-2 op work (multithread + STFT/im2col/sin, the
`Padded1d` im2col patcher, lazy im2col under a symbolic length, SIMD binary fusion under a
symbolic length (Tier 3), the AdaIN/Snake scale-mul swap gate + lazy-im2col threshold
(Tier 4), single-pass `SumOfSquares` for the symbolic InstanceNorm variance (Tier 5), and
folding the lazy-im2col zero-`Pad` into the gather (Tier 6)); onnxruntime is ~0.4. See the
caching investigation and the six conv/elementwise run-speed sections below.

### Plan caching investigation (2026-07-03) — exact-shape cache landed; symbolic first thought blocked

> **Superseded:** the "symbolic BLOCKED" verdict below turned out to be wrong — the
> symbolic single plan **landed** the same day. Kept for the reasoning trail; see
> "Symbolic single plan (2026-07-03) — DONE" immediately after this section.

Per-utterance `parse + into_optimized` is ~2.6 s (optimize ~2.3 s), pure overhead
onnxruntime avoids by compiling its session once. Three tiers were evaluated:

- **Symbolic single plan (the clean win): ~~BLOCKED by tract~~ → DONE (see below).**
  Optimizing either
  subgraph with a symbolic length dim fails during analyse at the style-broadcast
  `Concat` (`/encoder/predictor/text_encoder/Concat_1`, axis 2):
  `Impossible to unify Sym(N) with Val(1)`. Root cause: that Concat joins the text
  features `[1,N,C0]` with an `Expand` output that should be `[1,N,C1]`, but the
  Expand's target shape is a dynamic `Slice→Where` chain that collapses the sequence
  dim to 1 under symbolic analysis, so tract infers the Expand output as `[1,1,C1]`.
  A fix needs (a) graph surgery to feed the Expand a symbolic target derived from
  `Shape(text_features)`, **and** (b) a tract patch — `MultiBroadcastTo::wire*`
  (`tract-hir/.../array/broadcast.rs`) calls `shape.concretize()`, which bails on any
  symbolic dim, so symbolic-length Expand can't be lowered even if analyse passes.
  ~~Deferred (deep, multi-site).~~ **This assessment was wrong — see the DONE section
  below; part (b) was unfounded and only graph surgery was needed.**
- **Bucketing + padding (to reuse across lengths): UNSAFE.** The model normalizes
  globally (decoder instance-norm over frames, encoder norm over phonemes), so
  padding poisons the whole output — measured waveform corr vs ORT: phoneme padding
  0.73, frame padding 0.02 (exact = 0.977). Ruled out.
- **Exact per-shape cache: DONE.** `Pipeline` compiles a plan per exact shape (key:
  phoneme count for stage 1, `(phoneme, frame)` for stage 2) and reuses it. Correct
  (corr 0.977 preserved); helps repeated lengths / re-runs, but distinct-length
  sentences still each compile once. The length-independent win — the symbolic single
  plan — landed (next section) and now supersedes this as the default; the exact-shape
  cache remains as the graceful fallback. The orthogonal `run`-speedup lever is the
  quantized `model_q8f16.onnx`.

### Symbolic single plan (2026-07-03) — DONE, the "BLOCKED" verdict above was wrong

The symbolic single plan **landed**: each stage now optimizes **once** with symbolic
length dims and runs for any phoneme/frame count, so streaming a paragraph of
distinct-length sentences no longer recompiles each one. Parity is unchanged (full
paragraph corr **0.9756**, "Hello world." **0.9767** — identical to the concrete
path). The prior "BLOCKED / deep, multi-site" assessment was too pessimistic on two
counts; a probe-driven spike (`PROBE_SYM` shared-symbol mode in `examples/tract_probe.rs`)
mapped the real walls, all small:

1. **The tract-patch fear (b above) was unfounded.** `ShapeFactoid::concretize()`
   (`tract-hir/src/infer/factoid.rs:243`) returns `TVec<TDim>` and only bails when the
   shape is *open* (unknown rank/dim) — known-symbolic `TDim`s pass through. So once
   analyse infers a clean `[1,N,C1]`, the existing `MultiBroadcastTo` wiring lowers it
   with **no tract patch**. Only graph surgery was needed for the Concat.

2. **`#1802`/`Concat_1` fix = one graph surgery, not symbol propagation through
   `Where`.** The Expand's target `[1,N,1]` is already `Shape(text_features)[1]`; tract
   just can't carry it through the `Equal`/`Where(-1→1)` sentinel chain. `split_kokoro.py`
   `fix_expand_symbolic` feeds the Expand a direct `[1, N, 1]` built from the existing
   `Unsqueeze_1` (`[N]`), skipping the sentinel ops. Stage 1 then optimizes+runs
   symbolically as-is.

**Stage 2 needed coherent symbols + two more small fixes** (found by pushing the spike
wall-by-wall, each revealing the next):
- **Coherent shared symbols** (not a patch): `extract_model` gives each input dim its
  own `unk__` symbol, so a stray batch symbol leaked into a conv's frame axis. Pinning
  batch=1, one shared `N`, one shared `F` (what `Pipeline` does) fixes it.
- **`Resize` symbolic unit-fraction scale** (`tract-onnx/src/ops/resize.rs`): the
  harmonic source resamples `600*F` by `1/300`; tract couldn't compute `600F × 0.00333`.
  Patched: when the input dim is symbolic and the scale is a unit fraction `1/k`, output
  `= dim / k` (TDim division: `600F/300 = 2F`).
- **`Slice` symbolic-end clamp** (`tract-core/src/ops/array/strided_slice.rs`): the iSTFT
  slices the constant-length (20) window by the symbolic signal length; tract only
  clamped `end` to the axis size when *both* were concrete. Patched to `end = min(end,
  dim)` (ONNX clamp semantics) when `end` is symbolic and `dim` concrete. Plus a second
  `Expand`/`Where` identity-bypass surgery (`fix_istft_expand_symbolic`) — here `Where`
  wraps a genuine `Shape()` (no `-1`), so the Expand target points straight at it.

**Footprint (entire feature):** 2 graph surgeries + 1 backend-agnostic atan2 fix in
`tools/split_kokoro.py`; 2 small tract patches (`resize.rs`, `strided_slice.rs`); the
`Pipeline` now holds one `StagePlan::Symbolic` per stage with a `PerShape` fallback
(degrades gracefully on un-surgeried subgraphs — verified). Both surgeries are no-ops
under concrete N/F, so the same `stage1.onnx`/`stage2.onnx` serve concrete and symbolic.

**Measured (4 distinct-length sentences):** symbolic infer **17.65 s** vs per-shape
**21.77 s**; the gap *grows* with the number of distinct lengths (symbolic compiles once
total, per-shape compiles once per length). Per-stage optimize is ~1.4 s (stage 1) /
~3.9 s (stage 2). `run`-speed (RTF ~1.6) is unchanged and orthogonal (quantized model
is the lever there).

### Conv run-speed: `Padded1d` im2col patcher (2026-07-06) — DONE (Tier 1)

Profiling the (symbolic) plan (`KOKORO_TRACT_PROFILE`, harness `tools/bench_conv.sh`)
put **~49% of stage-2 run time in the `Im2col` op**, i.e. building the conv patch
matrix, not the matmul. Root cause: tract's `Im2Col` picks a "patcher" by shape and
had fast specialized paths only for `Valid1d`/`Valid2d`/`Padded2d` — **no `Padded1d`**.
Kokoro's vocoder is all rank-1 conv and **74 of its 91 convs are padded** (kernel 3/7/11
resblocks + upsampling), so they all fell through to the slow `Generic` patcher
(per-element coord-unravel + generic `patch.at()` validity probe + a separate serial
pack pass).

Added a `Padded1d` patcher (`tract-core/.../conv/im2col.rs`) — the rank-1 analogue of
`padded_2d` (one spatial axis, no y-loop; reuses padded_2d's inner valid/invalid x
loops), walking each (channel, kernel-tap) row with a precomputed valid x-range and
writing straight into the k-outer pack format. Bit-identical output to `Generic`.

**Measured** (fixed 242-token sentence, best-of-4, 16 threads): pure `infer`
**25.32 s → 22.28 s** (RTF 1.734 → 1.526, ~12% faster end-to-end); the instrumented
`Im2col` share **11.83 s (49.2%) → 8.60 s (41.8%)**, a ~27% cut. Parity unchanged
(waveform corr vs onnxruntime **0.9754**). The gain is capped because `Padded1d` is
single-threaded where `Generic` fanned the *fill* across cores (the k-outer pack writer
is sequential) — cheaper per-element work still wins net.

### Conv run-speed: lazy im2col under a symbolic length (2026-07-06) — DONE (Tier 2)

The symbolic single plan had silently disabled tract's fastest conv lowering: lazy
(virtual) im2col — which never materializes the patch matrix, feeding the matmul
microkernel packed panels gathered on demand — was gated on
`input_fact.shape.as_concrete()` in `conv.rs::codegen` (also `wire_as_depth_wise`), which
is `None` under symbolic dims, so every conv fell to eager `wire_as_im2col_pair`. A quick
`KOKORO_TRACT_FORCE_PERSHAPE` A/B (concrete per-shape path, lazy on) confirmed the upside:
concrete **16.78 s** vs symbolic+`Padded1d` **22.28 s** — the biggest single lever left.

The only thing forcing the concrete gate was that lazy im2col bakes two gather-offset
vectors at codegen from the concrete geometry (`n_byte_offsets`, one per output position;
`k_byte_offsets`, per input-channel×kernel-tap). Both depend on the concrete length. But
`LazyIm2colParams`'s `MMMInputFormat`/`OpaqueFact` impls are used only at *runtime* (inside
`LazyIm2colInput`); the *fact* level uses `DynPackedOpaqueFact`. So the fix keeps the
heavily-unrolled eval gather **untouched** and only **defers building the offsets to eval**:

- `lazy_im2col.rs`: `LazyIm2Col.params` becomes `LazyParams::{Ready, Deferred}`. `Deferred`
  stores the (symbolic) `PoolGeometry` + packer + channels + symbolic `mn`; at eval it
  resolves the geometry to the now-concrete input shape and builds the offsets (shared
  `build_lazy_params`, so the concrete `Ready` path is numerically identical). `output_facts`
  reports a symbolic `mn` when deferred.
- `conv.rs`: `wire_as_lazy_im2col` flows `TDim` shapes throughout (padding → explicit `Pad`
  node → Valid conv, all symbolic-safe since exported pads are concrete) and picks `Ready`
  vs `Deferred` by whether the post-pad length is concrete. `should_use_lazy` became a method
  that also fires for a symbolic length (batch must be concretely 1; pads must be concrete).

**Measured** (fixed 242-token sentence, best-of-4, 16 threads): pure `infer`
**22.28 s → 15.20 s** (RTF 1.526 → **1.041**), and **25.32 s → 15.20 s = ~40%** off the
pre-conv-work baseline; stage-2 `Im2col` drops to 36 calls / ~7% (the 38 kernel>5 convs are
now lazy, folded into the matmul gather). Parity unchanged across five utterances + a
streamed paragraph (corr 0.9705–0.9842). vs onnxruntime (RTF ~0.4) the gap is now ~2.6×,
down from ~4.3×. Compile-once is preserved (still one symbolic plan per stage), so a streamed
paragraph of distinct-length sentences gets this conv speed *and* pays optimize only once.
`KOKORO_TRACT_FORCE_PERSHAPE=1` forces the concrete path for A/B benching.

### Conv/elementwise run-speed: SIMD binary fusion under a symbolic length (2026-07-06) — DONE (Tier 3)

Same bug class as Tier 2, a third time. The symbolic profile ran the AdaIN/Snake
elementwise ops as **raw, single-threaded scalar** `Mul`/`Add`/`Sub` (~3.8 s), where the
concrete (`KOKORO_TRACT_FORCE_PERSHAPE`) path fused them into SIMD
`OptMulByScalar`/`OptAddUnicast`/`OptSubByScalar` (~1.75 s). Root cause: the codegen in
`tract-core/src/ops/binary.rs` picks the fused kernel only if `gt_tdim(num_elements, 32)`
is true, and `gt_tdim` computed `min(32, x).to_i64() == 32` — for a symbolic element count
(the affine broadcasts `[1,C,F]`×`[1,C,1]`, so `num = F`) that can't concretize, so it
returned `false` and fell back to the scalar op. (`check_input_shapes` already accepts the
broadcast pattern symbolically, so `gt_tdim` was the only blocker.)

Fix: `gt_tdim` returns `true` when `x` can't be concretized — the fused op is
correctness-equivalent for any size and the threshold is only a tiny-tensor guard; a
symbolic frame axis is never tiny. Concrete `x` is unchanged.

**Measured** (fixed 242-token sentence, best-of-4): pure `infer` **15.20 s → 13.55 s**
(RTF 1.041 → **0.928**) — now **under realtime** on this bench, and streamed paragraph
sentences run RTF ~0.83–0.85. `Add`/`Sub` became `OptAddUnicast`/`OptAddByScalar`/
`OptSubByScalar` as expected. Parity bit-identical (corr 0.9705–0.9842 unchanged — same
math, faster kernel). vs onnxruntime (RTF ~0.4) the gap is now ~2.3×.

**Still on the table:** ~169 `Mul` calls (12.7%) stay raw — a *different* fusion condition
than `gt_tdim` (neither by-scalar nor unicast-aligned as emitted); the `Sin` oscillator
(16%, already multithreaded, needs a vectorized `sin`, fidelity-sensitive); the lazy-im2col
`Pad` nodes (6%, could fold into the gather); and the sequential `Scan`/LSTM (13%). See the
run-speed plan for the ranked map.

### Conv/elementwise run-speed: Tier 4 (2026-07-06) — RTF 0.928 → 0.745

Added a **top-N per-node profiler** (`KOKORO_TRACT_PROFILE_NODES=<N>` in
`src/bin/kokoro_tract.rs`) that tags each node with its concrete input/output shapes.
That turned the aggregated op buckets into actionable shapes and settled three things:

- **Raw `Mul` (12.7%) — the swap gate, not the fusion gate.** The dominant raw mul is the
  AdaIN/Snake per-channel scale `[1,C,1] × [1,C,F] → [1,C,F]`. That's a trailing-unary
  *by-scalar* shape, which Tier 3's `gt_tdim` already unblocked — yet it stayed raw. The
  real blocker is one gate earlier: `TypedBinOp::codegen` decides whether to swap inputs so
  the full-size operand is operand_1 (the by-scalar kernel evals in place into operand_1, so
  it must equal the broadcast result → `can_eval_in_a`) via
  `(a_vol - b_vol).prove_strict_negative()`. That can't prove `C - C*F < 0` for a symbolic
  `F`, so it never swaps, the big tensor stays operand_2, and fusion is skipped. Same bug
  class as Tiers 2/3. **Fix:** decide the swap by comparing each operand's shape against the
  broadcast result (decidable symbolically) — swap when operand_2 is full-size and operand_1
  is broadcast; the swap is already paired with `flip()` so non-commutative ops stay correct.
  → raw `Mul` becomes `OptMulByScalar`. **infer 13.55 → 11.88 s, RTF 0.928 → 0.814**,
  bit-identical (elementwise fp multiply). Commit `9a2c1d3`.

- **Eager `Im2col` (7.8%) — the lazy threshold.** `should_use_lazy` rejected any conv with
  kernel product ≤ 5, forcing the vocoder's 6 kernel-3 128-channel convs at audio scale
  (70081 frames) onto the eager patcher, which materializes a ~108 MB patch matrix each.
  Lowered the threshold to ≤ 1 (only genuine pointwise convs, which are pure matmuls, stay
  eager). **infer 11.88 → 10.87 s, RTF 0.814 → 0.745**, bit-identical (lazy and eager im2col
  feed the matmul the same values). Side effect: `Pad` grew 41 → 77 nodes (each newly-lazy
  conv adds a Tier-2 explicit Pad), so Pad is now ~10% — a bigger fold target. Commit
  `f71cb9a`.

- **`Sin` (16%) is not anomalous — hypothesis disproven.** Per-node timing shows each Sin is
  `[1,128,70081]` = 8.97 M elements at ~0.058 s ≈ **6.5 ns/element** — normal-range scalar
  `sinf`, *not* the 30–50× glibc slow-reduction path the unbounded-phase theory predicted.
  So there's no cheap graph-surgery win; the only Sin lever left is genuine SIMD
  vectorization (~3× ceiling on 16% ≈ 5% total), which carries the ~0.965 branch-cut
  fidelity risk. **Deferred.**

- **Rejected by measurement:** threading `Square` via `par_elementwise` (like sin/cos)
  *regressed* (RTF 0.745 → 0.766; the op bucket itself grew) — the `par_chunks_mut`
  scheduling overhead across its many calls outweighs the parallelism (memory-bandwidth
  bound on WSL). Reverted.

**Remaining levers.** `Scan`/LSTM (15%) was **tested and closed — see the next section**
(hoisting the input projection gives no win; the scan is latency-bound on the recurrent
path). Still open, both invasive/risky (need a decision before spending risk budget):
**`Pad`** (10%) — folding the zero-pad into the lazy gather means adding per-element bounds
logic to the heavily-unrolled unsafe gather kernels (`input_8n`/`6n`/`4n`/`2n`), risking the
majority valid-read path; **`Sin`** (16%) — vectorized `sin`, fidelity-sensitive.
`OptMatMul` (23%) is MLAS-class and out of scope. **Cumulative Tier 1–4: RTF 1.734 → 0.745;
gap to onnxruntime (0.4) now ~1.9×.**

### Scan/LSTM input-projection hoisting (2026-07-06) — EXPERIMENT, negative, reverted

Tested the plan's ranked Scan/LSTM lever. tract lowers each ONNX `LSTM` into a `Scan` whose
body wires **8 per-step `EinSum` matmuls**: 4 recurrent `Ht₋₁·Rᵀ` and 4 input `Xt·Wᵀ`. The
4 input projections are **loop-invariant** (they depend only on the per-step input `Xt`, not
on the recurrent state), so in principle they can be lifted out: precompute `X @ Wᵀ` over the
*whole* sequence as one large multithreaded GEMM before the loop and slice it per step.

**The lowering made this clean.** `W` is already the `[4·h, input]` gate concatenation in
i,o,f,c order (`lstm.rs` slices it into `Wi/Wo/Wf/Wc`), so `X @ Wᵀ = [batch, seq, 4·h]` is
**directly sliceable per gate — no reconcatenation**. Implementation (all in
`third_party/tract/tract-onnx/src/ops/rec/`): added a `WireBody::hoist_input_projection()`
hook (default `false`, `LSTM` returns `true`); in `common.rs::wire_one_side` precomputed
`EinSum("bsi,gi->bsg")` at the outer level and fed it as an extra `Scan(axis:1, chunk)`
input (`"XWt"`); branched `lstm.rs::wire_body` to slice `XWt` per gate instead of doing the 4
per-step matmuls. It compiled and ran correctly.

**Result — no speedup, slight regression:**

| | baseline | hoisted |
|---|---|---|
| RTF (best of 4) | **0.745** | 0.757 |
| `Scan` bucket | 1.463 s | 1.568 s |
| `OptMatMul` bucket | 2.159 s | 2.228 s (the added outer GEMM) |

**Why it doesn't pay off:** the Scan is **latency-bound on the serial recurrent path** — each
step waits on `Ht₋₁·Rᵀ` plus the sigmoid/tanh chain, and *that* cannot be hoisted because it
depends on the evolving state. The input matmuls are tiny `m=1` gemvs that were already cheap;
replacing them with 4 slices of a 4×-wider scanned input added about as much memory-traffic and
slicing overhead as it removed. Hoisting only helps a loop that is throughput-bound on its
input projections, and this one isn't.

**Parity was fine** (the change is numerically safe): durations and output lengths were
identical everywhere — "Hello world." stayed 39000 samples, the paragraph stayed
"242 tokens | 14.60 s", and fox/sea were bit-identical. An early "hello parity broke" scare
was a **stale-reference-file artifact** (compared against a wrong-length `o_hello.wav`); fresh
onnxruntime references via `--features onnx` confirmed baseline "Hello world." is 1.62 s /
39000 samples at corr 0.9767, matching tract. Method note for next time: regenerate the ORT
reference for the *exact* sentence before trusting a corr number — a length mismatch tanks corr
and masquerades as a regression.

**Reverted** both files; tree back to the RTF-0.745 baseline. **Takeaway: don't hoist LSTM
input projections — the recurrent serial dependency dominates the scan, so the win isn't
there.** The only structural Scan win left would be attacking the per-iteration alloc/copy in
`scan/optimized.rs` (more invasive, and not shown to dominate).

### Tier 5: single-pass `SumOfSquares` for the symbolic InstanceNorm variance (2026-07-06)

A per-node graph inspection (temporary `KOKORO_TRACT_INSPECT` dump of `Square`/`Reduce`
producers) found that the 6% stage-2 `Square` bucket is two unrelated things: 48 Snake
`sin²` (`Sin → Square → Mul`, elementwise, not a reduce) and **65 InstanceNorm variances**
(`Sub → Square → Reduce<Sum> → Mul(1/N)`). tract *already* has a fused `Reduce<MeanOfSquares>`
and it fires for the stage-1 LayerNorm reductions — where the reduced axis is **concrete**
(128/768) — but **not** for the stage-2 InstanceNorm ones, where the reduced axis is the
symbolic frame count `F`. Same bug class as Tiers 2–4: the fusion (`declutter_mean_of_square`
in `tract-core/src/ops/nn/reduce.rs`) is gated on `norm.as_i64()` **and** on the `1/N`
divisor being a compile-time uniform const — but under the symbolic plan `1/F` is a *runtime*
`Recip`, so both conditions fail.

The clean fix is *not* to un-gate `MeanOfSquares` (its eval isn't even single-pass — it clones
+ squares a full temp, then sums; and matching the runtime `Recip(F)` divisor is fragile).
Instead, a new `Reducer::SumOfSquares` variant with a **genuinely single-pass** eval
(square-and-accumulate per contiguous slice, never materializing the ~36 MB squared temp that
is the whole cost), fused via `Reduce<Sum>(Square(x)) → Reduce<SumOfSquares>(x)`, leaving the
cheap `× Recip(F)` on the small reduced tensor untouched. The declutter is **scoped to the
symbolic reduced axis** (`norm.as_i64().is_none()`), so the concrete stage-1 `MeanOfSquares`
path is byte-for-byte unchanged (verified: 31 `MeanOfSquares` retained, exactly the 65
symbolic variances become `SumOfSquares`).

Kernel detail that mattered: the reduction squares each element in **f32** (matching the
fused-away `Square` op's rounding) but accumulates in **f64** across 8 vectorizable lanes.
The f64 headroom makes the reassociation error negligible — so this non-bit-identical change
actually lands *closer* to onnxruntime than tract's original f32 SIMD sum: parity **improved**
above the previous baseline (fox 0.9754 → **0.9760**, hello 0.9767 → **0.9774**, sea 0.9705
unchanged), rather than regressing. (An initial f32-accumulate kernel had regressed fox to
0.9726; f64 accumulation costs nothing measurable because the reduction is bound by the f32
read.) **Speed:** a same-session back-to-back A/B (WSL absolute RTF drifts with load, so only
the paired delta is trustworthy) gave **RTF 0.767 → 0.747, ~2.6%**, monotonic (every fused run
beat every baseline run). Both stages still "compiled one symbolic plan"; pure-Rust build
links. Modest but real, parity-positive, and low-risk.

### Tier 6: fold the lazy-im2col zero-Pad into the gather (2026-07-06) — RTF 0.70 → 0.62

The Tier-2 lazy im2col deliberately expresses a padded conv as an explicit zero-`Pad`
feeding a `Valid` conv, so its gather is a plain in-bounds windowed read. But that's a
full-tensor copy per conv, and Tier 4's lower lazy threshold multiplied them (41 → 77 `Pad`
nodes, ~8% of stage 2). This tier removes the copy by folding the padding into the gather.

Key structural facts that made it low-risk: (1) the gather runs **per panel** (`do_panel`) of
`r` consecutive output positions, and (2) for a rank-1 kernel-K conv over F ≈ 44 k frames,
only the **first and last panels** touch padding — the interior >99.99% is fully in-bounds.
So the hot unrolled kernels (`input_8n`/`6n`/`4n`/`2n`) stay **byte-for-byte** on interior
panels; only the ≤2 boundary panels use a new cold `write_checked` that zero-fills
out-of-range reads (exactly what the zero-`Pad` supplied). `build_lazy_params` records each
tap's in-bounds output range (`tap_valid`, same `div_ceil` formula as the eager `padded_1d`
patcher) plus their intersection (`interior`); `wire_as_lazy_im2col` skips the `Pad` for
rank-1 and keeps the padded pool spec (multi-dim convs keep the explicit `Pad` — fold is
rank-1 only). No change to `should_use_lazy` (pads are still concrete at build/eval time).

**Bit-identical** — same values gathered, boundaries zeroed exactly as before (fox 0.9760 /
hello 0.9774 / sea 0.9705, byte-for-byte equal to the pre-fold waveform). **Speed:** a
same-session back-to-back A/B gave **RTF 0.698 → 0.623 (~11%)**, monotonic (every folded run
beat every baseline run). The win *exceeds* the 8% `Pad` bucket because deleting 77
full-tensor allocations+copies per synth also relieves the allocator/cache pressure that was
slowing the surrounding memory-bound ops on WSL. The `Pad` bucket disappears from the stage-2
profile; `OptMatMul` (29%), `Sin` (23%), `Scan` (19%) now dominate. Highest-ROI, zero-fidelity
tier of the arc. **Cumulative Tier 1–6: RTF 1.734 → ~0.62; gap to onnxruntime (0.4) now ~1.55×.**

### Tier 7: allocator + SinSq fusion + vectorized sin (2026-07-07) — RTF ~0.62 → ~0.50

Four independent levers, each A/B'd with `tools/bench_conv.sh` same-session paired best-of-4
(WSL RTF drifts session-to-session; only paired deltas are trustworthy). This-session baseline
measured **infer 9.58 s / RTF 0.656** (drifted up from the 0.62 Tier-6 endpoint); all deltas
below are paired against it.

- **Lever 1 — allocator pressure (KEPT, mimalloc).** Every intermediate tensor is a fresh
  `uninitialized_aligned_dt` → glibc mmaps/munmaps each big block, so every op ate first-touch
  page faults on its output (this is also why threading `Square` regressed in Tier 4, and why
  Tier 6's win exceeded its op bucket). Two fixes A/B'd: (a) env-only
  `MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=-1` → **9.58 → 8.81 s (−8.0%)**;
  (b) `#[global_allocator]` **mimalloc** → **9.58 → 8.97 s (−6.4%)**. Both bit-identical (audio
  byte-for-byte unchanged); both shift the same buckets (`OptMulByScalar` 11.2 → 9.5%). Kept
  **mimalloc** despite the env vars' marginally larger win: self-contained (no runtime env),
  portable (helps the Termux/musl target where glibc `mallopt` doesn't apply). Env vars
  documented here as the glibc-only alternative. `Cargo.toml` (`mimalloc` behind `tract`,
  `default-features = false`) + `src/bin/kokoro_tract.rs` (`static GLOBAL`).

- **Lever 2 — stage-1 threading + thread count (NEGATIVE, reverted).** `synthesize` scopes the
  thread pool to stage 2 only. Wrapping stage-1 `run` in the same `multithread_tract_scope`
  **regressed +31% (8.97 → 11.75 s)** — the serial LSTM duration predictor and small per-op
  GEMMs pay more in thread-dispatch overhead than the BERT encoder saves; stage-2 profile
  unchanged, so the whole +2.8 s is stage 1. Thread-count sweep: `KOKORO_TRACT_THREADS=8`
  (physical cores) → **10.15 s**, worse than the default 16 (SMT). The existing
  stage-2-only / all-cores config is already optimal. No change.

- **Lever 3a — fuse `Square(Sin(x)) → SinSq` (KEPT, bit-identical).** Kokoro's 48 Snake
  activations each ran `Sin` then `Square` as two full memory passes over `[1,C,F]`. Added a
  `sin_sq`/`SinSq` element-wise op (`tract-core/src/ops/math/mod.rs`) plus a `declutter` on
  `Square` whose `linear_prec` is a `Sin` (mirrors `declutter_recip`; `linear_prec` guarantees
  the Sin feeds only this Square so rewiring is safe). One pass instead of two; **bit-identical**
  (`x.sin().powi(2)` per element — verified byte-for-byte fused-vs-unfused WAV, same MD5).
  Profile: `Sin` + `Square` (23% + 1.8%) collapse into one `SinSq` (23%). **8.97 → 8.72 s
  (−2.8%)** — smaller than the plan's ~4–6% guess because `Square` was only 1.8% here, not the
  estimated 7%.

- **Lever 3b — vectorized `sin` (KEPT; the big win).** After 3a, `SinSq` was still 23% and the
  per-node profile (Tier 6) confirmed normal-range phases (scalar `sinf` fast path), so a
  branchless minimax `sinf` (Julien Pommier / cephes, π/4-octant 3-part Cody–Waite, ~1e-7 abs
  ≈ 1 ulp — **not** fast-math) auto-vectorizes inside `par_elementwise` instead of calling scalar
  libm per element. `ssin_f32` used by both `sin` and `sin_sq` (f32 arm; f16/f64 keep exact
  `.sin()`). **`SinSq` 1.672 s → 0.232 s (7.2×)**, total **8.72 → 7.36 s (−15.6%)** — the single
  biggest lever, far above the plan's ~8–12% estimate (scalar `sinf` was the cost).
  **Fidelity:** `ssin_f32` verified ~7.8e-8 max abs error over `[-2000,2000]` dense (vs
  `np.sin`), so the sin itself is essentially exact; the end-to-end shift is the vocoder's known
  branch-cut sensitivity (any perturbation flips marginal bins; saturates ~0.965). Full gate vs
  fresh onnxruntime references (exact-tract reproduced the documented 0.9760/0.9774/0.9705
  exactly):

  | sentence | exact-tract vs ORT | vecsin vs ORT | Δ |
  |---|---|---|---|
  | fox | 0.9760 | 0.9737 | −0.0022 |
  | hello | 0.9774 | 0.9782 | +0.0008 |
  | sea | 0.9705 | 0.9703 | −0.0002 |

  Shift ≤0.0022 (2/3 flat-or-better), within branch-cut jitter. Kept by explicit user decision
  (speed win vs the plan's strict "revert if it moves").

- **Lever 4 — Scan overhead (STOP, no change).** Scan was 24% after 3b (its ~1.41 s absolute is
  unchanged; share grew as the total shrank). Temporary `eval` instrumentation (reverted) split
  body-run vs state-copy/alloc: the dominant stage-2 scan is **body=749 ms, overhead=1%**
  (prep+assign 4.8 ms). 99% is recurrent body math — exactly the hoisting post-mortem's verdict.
  Buffer-reuse would save ≤1% of scan ≈ 0.5% of infer, far under the 3% stop threshold. Not done.

**Cumulative Tier 1–7: RTF 1.734 → ~0.50** (paired infer 9.58 → 7.36 s, −23%); gap to
onnxruntime (0.4) now ~1.25×. Remaining stage-2 profile: `OptMatMul` 36% (MLAS-class, out of
scope), `Scan` 24% (recurrent body-bound), `OptMulByScalar` 12%; `SinSq` down to 4%. The
low-hanging levers are now exhausted — further parity to 0.4 needs matmul-kernel work.

### Where the ~0.94 vocoder gap comes from (2026-07-02)

Bisected Stage 2 tract-vs-ORT with intermediate probe points (add tensors as extra
graph outputs; `kokoro-tract` dumps all stage-2 outputs under `KOKORO_TRACT_DUMP`,
compared against onnxruntime fed the same stage-1 dumps). Result, for "Hello world.":

| Probe point | tract-vs-ORT corr |
|---|---|
| forward STFT (`generator/STFT`, my rank-2 patch) | **1.00000** |
| harmonic source (`m_source` Tanh) | **1.00000** |
| `ups.0` ConvTranspose | **1.0**, max\|Δ\| 0.0 (bit-exact) |
| resblocks.0 AdaIN variance | 0.99935 |
| `ups.1` ConvTranspose | 0.9975 |
| `conv_post` (log-magnitude) | 0.9989, max\|Δ\| **1.34** |
| `Exp` → magnitude spectrogram | **0.937** |
| waveform | 0.942 |

So the STFT rank-2 patch and the iSTFT are **not** the cause (both ~1.0).

**Root cause (2026-07-02, deeper bisection): the harmonic source's phase computation
is numerically unstable and tract's tiny STFT difference triggers it.** The source
module computes phase as `atan(imag/real)` + quadrant correction
(`Div → Atan → Where(±π)`), then feeds that *raw phase* straight into a conv
(`noise_convs.0`). Probing the source-analysis chain (tract vs ORT, same inputs):

| tensor | corr |
|---|---|
| source magnitude (`Sqrt(real²+imag²)`) | **1.00000** |
| `Div = imag/real` | nan/inf (real≈0 crossings) |
| `Atan` raw | 0.40, max\|Δ\| **π** |
| phase (`Where_1`, after quadrant fix) | **0.11**, max\|Δ\| **2π** |
| `Concat_3` source features → conv | 0.11 |

tract's `Atan` is exact libm; the divergence is in `Div`. tract's f32 STFT differs
from onnxruntime's by only ~0.003 (corr 1.0), but **near `real ≈ 0` that flips the
sign of `imag/real`** → `atan(±∞)` = ±π/2 (the π jump), which the `Where(±π)` quadrant
correction doubles to 2π. The corrupted *raw phase* (used directly as a conv feature,
not via sin/cos) poisons the whole source/noise path, which is added into the decoder
→ wrong magnitude/phase spectrograms → ringing. The transcendental-precision
hypothesis is **refuted** (`Sin`/`Exp`/`Atan` are libm-exact; `m_source` Tanh is
corr 1.0). This is an inherent instability amplifying a tiny STFT difference, not a
logic bug. Candidate fixes: higher-precision (f64) STFT / source-analysis so the sign
near zero-crossings matches; or stabilize the phase op. Non-trivial; uncertain payoff.

**f64 experiment (2026-07-02) — did not pan out.** Two attempts:
- *Targeted STFT-in-f64* (compute the forward DFT in f64 internally): **no effect on
  the waveform** (byte-identical 0.94205 vs ORT). The forward STFT is a 20-point DFT,
  already f32-accurate — the ~0.003 real-part error is *inherited* from the source
  signal (m_source is ~0.0007 off), not produced by the STFT. Reverted.
- *Full-stage-2 in f64* (cast all initializers/IO to double): **blocked by tract's
  incomplete f64 support.** `Resize` requires f32 scale/roi params even on an f64
  graph, and `Gemm`'s `beta` is materialized as f32 regardless of operand dtype
  (`tensor is F32, accessed as F64` at `resblocks.*/adain*/fc/Gemm.beta_c`).

Conclusion: the phase instability is driven by the **source path** differing from
onnxruntime by ~0.0007 (f32 accumulation), amplified near real≈0. Matching it needs
the source path in f64, which tract 0.22 can't do without first fixing its f32-hardcoded
`Gemm`/`Resize` internals — and f64 is ~2× slower. Realistic options: accept the
ringing; fix tract's f64 `Gemm`/`Resize` then run the source path in f64 (slow, proof
of fix); or graph-surgery a numerically-stable atan2 into the model.

**Remaining:** (a) Stage-2 vocoder numerical fidelity (diffuse decoder precision +
`Exp` amplification — above); (b) optional plan caching for perf; (c) ship/generate the
split subgraphs as part of the tract build flow.

### atan2 branch-cut surgery (2026-07-03) — FIXED, 0.949 → 0.977

The prior "f64 / stable-atan2 won't help" reasoning was **wrong**, and an empirical
bisection (ORT-only, tapping the source-analysis tensors as extra graph outputs)
found the real, fixable root cause. Method: run the full model in onnxruntime with
internal taps (`real`=`generator/Gather_4`, `imag`=`Gather_5`, `Div`, `Atan`,
`phase`=`Where_1`, `mag`=`Sqrt`), then **substitute** candidate phase/real-imag
tensors back in via an override input and measure the *waveform* corr. This isolates
exactly what each fix buys, without needing to rebuild tract for every hypothesis.

**Findings:**
- The damage is **entirely in the phase channel**. Overriding an ORT run's magnitude
  channel with tract's (corr-1.0-but-Δ0.003) values → waveform 0.99999; overriding the
  phase channel with tract's → 0.949. So the vocoder gap is the phase, not the
  magnitude / `conv_post`-`Exp` path as previously written.
- **tract's phase is NOT NaN-poisoned.** `Div` overflows to `inf` in the ~4000
  tiny-magnitude bins, but `atan(inf)=±π/2` absorbs it cleanly — the dumped phase has
  0 NaN / 0 Inf. So there was no NaN defect to fix.
- **The real bug: tract's `Div→Atan→Where` atan2 emulation ≠ true atan2 on its own
  inputs** (corr −0.17, max|Δ| 2π). The model selects the +π vs −π quadrant with a
  *strict* `imag > 0` (`Greater`). tract structurally produces **exact `imag == +0.0`**
  in ~30% of the source-STFT bins; at the negative-real branch cut those fail `>0` and
  the graph returns **−π**, whereas IEEE `atan2` (and onnxruntime, whose imag is a tiny
  nonzero residue there) returns **+π**. That 2π error in the raw phase — fed straight
  into `noise_convs.0` — is the ringing. (onnxruntime's own graph *does* equal atan2,
  corr 0.99994, precisely because it never hits exact `imag==0`.)

**Fix (one node): `Greater` → `GreaterOrEqual`** on
`/decoder/decoder/generator/Greater` (i.e. `imag > 0` → `imag >= 0`). This makes
tract's emulation equal true atan2 on its own inputs (verified corr 1.0 / max|Δ| 2e-7
vs `np.arctan2`). It only changes the `imag==0` boundary, so for onnxruntime's
nonzero-residue inputs it is a no-op (backend-agnostic correctness fix). Applied as
graph surgery in `tools/split_kokoro.py` (`fix_atan2_branch`), so it ships with the
regenerated `stage2.onnx` and needs **no tract patch** (tract already implements
`GreaterOrEqual`).

**Measured end-to-end (tract vs onnxruntime, af_heart):**

| Utterance | before (`Greater`) | after (`GreaterOrEqual`) |
|---|---|---|
| "Hello world." | 0.9491 (rel-RMS 0.366) | **0.9767** (rel-RMS 0.224) |
| "The quick brown fox…" | 0.9597 | **0.9754** |
| "She sells seashells…" | 0.9501 | **0.9705** |

**Remaining gap to 1.0 (now ~0.976) is genuinely inherent** and *not* worth chasing:
substituting a perfect `atan2` still yields 0.977, because tract's `real`/`imag` differ
from onnxruntime by ~0.003 (inherited f32 accumulation in the harmonic source), which
flips the *genuine* branch cut (real<0, imag crossing 0) in the remaining bins. That
part is precision-driven **and precision-independent at the same time**: injecting even
1e-4 Gaussian noise into the source spectrogram saturates the waveform at ~0.965
(a discontinuous branch function — any perturbation flips the marginal bins fully), so
f64 cannot help. This is the correct place to stop the fidelity chase.

If revisited, the highest-value work is **perf (plan caching per phoneme-count)** and
**packaging the split subgraphs into the build flow**, not further vocoder parity.

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
