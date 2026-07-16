#!/usr/bin/env python3
"""Split Kokoro-82M ONNX into two tract-friendly stages at the length regulator.

Kokoro's length regulator expands phoneme-level features to frame level via an
alignment matrix whose frame axis length = sum(round(durations)) — a *value*, not
a static shape, so tract's static shape inference can't represent the monolithic
graph (see docs/tract-support-plan.md). We split the graph there and rebuild the
alignment in Rust:

    Stage 1 (tract): input_ids, style, speed
                     -> prosody features [1,640,N]   (/encoder/Transpose_output_0)
                     -> text features    [1,512,N]   (/encoder/text_encoder/Transpose_2_output_0)
                     -> durations        [1,N]       (/encoder/Clip_output_0, per-phoneme)
    Rust length regulator: round durations, total_frames = sum, build the boolean
                     alignment matrix A [N, total_frames] (the model's
                     And(GreaterOrEqual, Less) over Range(0,total_frames) vs
                     CumSum(durations), cast to f32).
    Stage 2 (tract): the two phoneme feature tensors + alignment A
                     -> decoder + iSTFTNet -> waveform [1, total_frames*hop]

Both extracted subgraphs carry stale symbolic value_info (`sequence_length`,
`num_samples`) that conflicts with concrete-length execution, so we strip all
intermediate value_info and clear the output shapes; tract then infers everything
from the concrete input facts. With that, Stage 1 loads/optimizes/runs as-is, and
Stage 2 does too once tract's STFT accepts the rank-2 signal (patched in
third_party/tract/tract-onnx/src/ops/fft.rs) — modulo a remaining Range/Shape->TDim
optimize issue tracked in the plan doc.

Usage:  python3 tools/split_kokoro.py [MODEL.onnx] [OUT_DIR]
Default MODEL is the HF cache path for onnx-community/Kokoro-82M-v1.0-ONNX.
Default OUT_DIR is the project-local `kokoro-onyx/` directory, which is where
`ryk` expects the stages when run with `KOKORO_TRACT_DIR=kokoro-onyx`.
(That directory is git-ignored — the fp32 subgraphs are ~311 MB; see the README.)
"""
import os
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--onnx-community--Kokoro-82M-v1.0-ONNX"
    "/snapshots/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx/model.onnx"
)

# Write the split stages into the project-local kokoro-onyx/ dir by default so they
# live with the checkout (stable path) instead of the HF cache's snapshot-hashed dir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "kokoro-onyx")

# --- cut tensors (see module docstring / docs/tract-support-plan.md) ----------
MODEL_INPUTS = ["input_ids", "style", "speed"]

# Stage 1 emits phoneme-level features + per-phoneme durations (no frame axis).
STAGE1_OUT = [
    "/encoder/Transpose_output_0",              # prosody features -> MatMul   [1,640,N]
    "/encoder/text_encoder/Transpose_2_output_0",  # text features -> MatMul_1 [1,512,N]
    "/encoder/Clip_output_0",                   # predicted durations          [1,N]
]

# Stage 2 begins at the two alignment MatMuls; it takes the phoneme features and
# the Rust-built alignment matrix (the shared second MatMul input in the original
# graph, /encoder/Cast_4_output_0) and produces the waveform.
STAGE2_IN = [
    "/encoder/Transpose_output_0",
    "/encoder/text_encoder/Transpose_2_output_0",
    "/encoder/Cast_4_output_0",                 # alignment matrix [1,N,total_frames]
    "style",                                    # decoder AdaIN affine params
]
STAGE2_OUT = ["waveform"]


def strip_symbols(path):
    """Remove intermediate value_info and output shapes so tract infers shapes
    from concrete input facts (kills stale `sequence_length`/`num_samples`)."""
    m = onnx.load(path)
    del m.graph.value_info[:]
    for o in m.graph.output:
        o.type.tensor_type.ClearField("shape")
    onnx.save(m, path)


# Node that selects the +pi vs -pi quadrant in the harmonic-source atan2 emulation
# (imag>0 ? Atan+pi : Atan-pi, for the real<0 half-plane).
ATAN2_QUADRANT_GREATER = "/decoder/decoder/generator/Greater"


def fix_atan2_branch(path):
    """Stabilize the harmonic-source phase (`atan(imag/real)` + quadrant fix) for
    tract.

    The source module computes phase as an atan2 emulation:
        real<0 ? (imag>0 ? Atan+pi : Atan-pi) : Atan
    The quadrant selector uses a *strict* `imag > 0` (a `Greater` node). At the
    negative-real branch cut with `imag == +0.0`, that test is false, so the graph
    returns -pi -- whereas IEEE atan2 (and onnxruntime, whose imag is a tiny nonzero
    residue there) returns +pi. tract structurally produces exact `imag == +0.0` in
    ~30% of the source-STFT bins, so its raw phase (fed straight into `noise_convs.0`)
    diverges from onnxruntime by 2*pi in those bins -> the ringing artifact.

    Relaxing the selector to `imag >= 0` (GreaterOrEqual) makes tract's emulation
    equal true atan2 on its own inputs (verified corr 1.0 vs np.arctan2), which lifts
    the vocoder waveform corr from ~0.949 to ~0.977 vs onnxruntime. This is an
    additive, backend-agnostic correctness fix: it only changes the imag==0 boundary,
    which for onnxruntime's nonzero-residue inputs is a no-op. See
    docs/tract-support-plan.md ("atan2 branch-cut surgery")."""
    m = onnx.load(path)
    hits = [n for n in m.graph.node if n.name == ATAN2_QUADRANT_GREATER]
    if not hits:
        raise SystemExit(f"atan2 quadrant node {ATAN2_QUADRANT_GREATER!r} not found")
    for n in hits:
        if n.op_type != "Greater":
            raise SystemExit(f"expected Greater at {n.name}, found {n.op_type}")
        n.op_type = "GreaterOrEqual"
    onnx.save(m, path)
    print(f"  patched {len(hits)} atan2 quadrant selector: Greater -> GreaterOrEqual")


# --- symbolic-length surgery for the style-broadcast Expand (stage 1) ---------
# The duration predictor's text_encoder broadcasts the 128-d style slice across
# the phoneme axis and concatenates it with the 512-d text features. It does this
# with the PyTorch `expand(-1,-1,-1)` ONNX lowering, whose target shape is built
# by a Shape -> Gather -> Concat -> Equal -> Where(-1 -> 1) sentinel chain. The
# sequence dim N in that target is genuinely Shape(text_features)[1], but tract
# cannot carry the symbol through the Equal/Where value logic under symbolic
# analysis, so it collapses the Expand output's seq axis to 1 and the downstream
# Concat_1 fails to unify Sym(N) with Val(1) (docs/tract-support-plan.md #1802).
EXPAND_NODE = "/encoder/predictor/text_encoder/Expand"
# `Unsqueeze_1_output_0` is already [N] (= Unsqueeze(Gather(Shape(text_features),1)));
# we reuse it and skip the Equal/Where the model wraps around it.
EXPAND_SEQ_DIM_1D = "/encoder/predictor/text_encoder/Unsqueeze_1_output_0"


def fix_expand_symbolic(path):
    """Feed the style-broadcast Expand a direct `[1, N, 1]` target derived from
    the sibling text features' Shape, bypassing the Equal/Where sentinel chain
    tract can't fold symbolically.

    The Expand data input is the style slice `[1, 128]`; broadcasting it to the
    target `[1, N, 1]` yields `[1, N, 128]` (the axis-2 128 comes from the data,
    the seq axis N from Shape(text_features)[1]) — identical to the model's own
    `[d0, d1, 1]` result, but expressed with ops tract propagates symbolically.
    This lets stage 1 optimize *once* with a symbolic phoneme count instead of
    recompiling per length. See docs/tract-support-plan.md ("symbolic tract patch")."""
    m = onnx.load(path)
    g = m.graph
    hits = [n for n in g.node if n.name == EXPAND_NODE]
    if not hits:
        raise SystemExit(f"expand node {EXPAND_NODE!r} not found")
    if not any(EXPAND_SEQ_DIM_1D in n.output for n in g.node):
        raise SystemExit(f"seq-dim tensor {EXPAND_SEQ_DIM_1D!r} not found")
    exp = hits[0]

    one = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="/encoder/predictor/text_encoder/_sym_one"
    )
    g.initializer.append(one)
    target = "/encoder/predictor/text_encoder/_sym_expand_shape"
    concat = helper.make_node(
        "Concat",
        inputs=[one.name, EXPAND_SEQ_DIM_1D, one.name],
        outputs=[target],
        axis=0,
        name="/encoder/predictor/text_encoder/_sym_expand_shape_concat",
    )
    # Insert just before the Expand so the graph stays topologically ordered
    # (Unsqueeze_1 and the initializer both precede this point).
    g.node.insert(list(g.node).index(exp), concat)
    exp.input[1] = target  # target-shape input
    onnx.save(m, path)
    print(f"  rewired {EXPAND_NODE} target shape -> [1, N, 1] (symbolic-friendly)")


# --- symbolic-length surgery for the iSTFT framing Expands (stage 2) ----------
# The iSTFTNet generator's inverse-STFT builds a per-sample frame-index grid whose
# length is the (frame-count-derived) signal length. Two Expands broadcast the
# range grid and the window-tap offsets to that grid shape, taking their target
# from a `Where(Equal(Shape_1,-1), 1, Shape_1)` sentinel wrapper. `Shape_1` is a
# genuine Shape() (no -1 dims), so the Where is an identity here — but tract can't
# prove `Shape_1 != -1` for the symbolic signal length and collapses the axis,
# breaking the downstream Concat. Pointing the Expands straight at `Shape_1`
# reproduces the identical shape with an op tract propagates symbolically.
ISTFT = "/decoder/decoder/generator/istft/stft"
ISTFT_SHAPE1 = f"{ISTFT}/Shape_1_output_0"
ISTFT_WHERE = f"{ISTFT}/Where_output_0"
ISTFT_EXPANDS = [f"{ISTFT}/Expand", f"{ISTFT}/Expand_1"]


def fix_istft_expand_symbolic(path):
    """Bypass the identity Where wrapping the iSTFT framing Expands' target shape,
    pointing them straight at the genuine Shape() they wrap. Lets stage 2 optimize
    once with symbolic phoneme/frame counts. See docs/tract-support-plan.md."""
    m = onnx.load(path)
    g = m.graph
    if not any(ISTFT_SHAPE1 in n.output for n in g.node):
        raise SystemExit(f"iSTFT shape tensor {ISTFT_SHAPE1!r} not found")
    n_fixed = 0
    for node in g.node:
        if node.name in ISTFT_EXPANDS:
            for i, inp in enumerate(node.input):
                if inp == ISTFT_WHERE:
                    node.input[i] = ISTFT_SHAPE1
                    n_fixed += 1
    if n_fixed == 0:
        raise SystemExit(f"no iSTFT Expand referenced {ISTFT_WHERE!r}")
    onnx.save(m, path)
    print(f"  rewired {n_fixed} iSTFT Expand target(s): Where -> Shape_1 (symbolic-friendly)")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_MODEL
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    s1 = os.path.join(out_dir, "stage1.onnx")
    s2 = os.path.join(out_dir, "stage2.onnx")

    onnx.utils.extract_model(model, s1, MODEL_INPUTS, STAGE1_OUT)
    strip_symbols(s1)
    fix_expand_symbolic(s1)
    print(f"wrote {s1}  ({len(onnx.load(s1).graph.node)} nodes)")

    onnx.utils.extract_model(model, s2, STAGE2_IN, STAGE2_OUT)
    strip_symbols(s2)
    fix_atan2_branch(s2)
    fix_istft_expand_symbolic(s2)
    print(f"wrote {s2}  ({len(onnx.load(s2).graph.node)} nodes)")


if __name__ == "__main__":
    main()
