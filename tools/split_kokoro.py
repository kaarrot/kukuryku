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
"""
import os
import sys
import onnx

DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--onnx-community--Kokoro-82M-v1.0-ONNX"
    "/snapshots/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx/model.onnx"
)

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


def main():
    model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_MODEL
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(model))
    os.makedirs(out_dir, exist_ok=True)

    s1 = os.path.join(out_dir, "stage1.onnx")
    s2 = os.path.join(out_dir, "stage2.onnx")

    onnx.utils.extract_model(model, s1, MODEL_INPUTS, STAGE1_OUT)
    strip_symbols(s1)
    print(f"wrote {s1}  ({len(onnx.load(s1).graph.node)} nodes)")

    onnx.utils.extract_model(model, s2, STAGE2_IN, STAGE2_OUT)
    strip_symbols(s2)
    print(f"wrote {s2}  ({len(onnx.load(s2).graph.node)} nodes)")


if __name__ == "__main__":
    main()
