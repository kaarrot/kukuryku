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

**Perf:** RTF ~2 after the stage-2 op work (multithread + STFT/im2col/sin);
onnxruntime is ~0.4. See the caching investigation below.

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
