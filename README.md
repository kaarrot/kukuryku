# kokoro — minimal CPU text-to-speech

Reads command-line text aloud using **Kokoro-82M** (ONNX), CPU-only and faster than realtime.

```
text → espeak-ng phonemes → Kokoro ONNX (+ voice) → 24 kHz audio → ffplay
```

## Setup

```bash
sudo apt install -y ffmpeg espeak-ng
pip install --user onnxruntime
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
`voices/*.bin` names in that repo (e.g. `am_michael`, `bf_emma`). onnxruntime is loaded
dynamically — if auto-detection fails, set `ORT_DYLIB_PATH` to your `libonnxruntime.so`.

> Note: input longer than ~510 phonemes is truncated (single-shot). Pronunciation uses raw
> espeak-ng, so it's close but not identical to Kokoro's reference phonemizer on tricky words.
