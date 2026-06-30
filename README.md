# kokoro — minimal CPU text-to-speech

Reads command-line text aloud using **Kokoro-82M** (ONNX), CPU-only and faster than realtime.

```
text → espeak-ng phonemes → Kokoro ONNX (+ voice) → 24 kHz audio → ffplay
```

## Setup

Needs `espeak-ng`, `ffmpeg`, and an **ONNX Runtime ≥ 1.24** shared library (`libonnxruntime.so`).
The build never downloads onnxruntime — it's loaded dynamically at runtime.

```bash
# Debian/Ubuntu/WSL
sudo apt install -y ffmpeg espeak-ng
pip install --user onnxruntime          # provides libonnxruntime.so (>= 1.24)
cargo build --release

# Termux (Android / aarch64)
pkg install -y onnxruntime espeak-ng ffmpeg
cargo build --release
```

## Run

```bash
./target/release/kokoro "Hello, this is a test."
echo "or pipe text in" | ./target/release/kokoro
KOKORO_VOICE=am_michael ./target/release/kokoro "a different voice"
KOKORO_WAV=out.wav ./target/release/kokoro "verify without speakers"
```

On first run the model and voice download from `onnx-community/Kokoro-82M-v1.0-ONNX` into
`~/.cache/huggingface` (cached afterwards). Default voice is `af_heart`; other voices are the
`voices/*.bin` names in that repo (e.g. `am_michael`, `bf_emma`).

**onnxruntime selection:** at startup it auto-detects a `libonnxruntime.so` (≥ 1.24) from common
locations — Termux `$PREFIX/lib`, system lib dirs, `LD_LIBRARY_PATH`, and pip's `onnxruntime/capi`.
No Python is involved. To pick a specific library (or if auto-detection finds none), set
`ORT_DYLIB_PATH=/path/to/libonnxruntime.so`.

> Note: input longer than ~510 phonemes is truncated (single-shot). Pronunciation uses raw
> espeak-ng, so it's close but not identical to Kokoro's reference phonemizer on tricky words.
