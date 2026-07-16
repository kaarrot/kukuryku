#!/usr/bin/env bash
# Download the pre-split Kokoro assets into kokoro-onyx/.
#
# This is the "copy an existing pair" path from the README: no Python, no
# numpy/onnx, no model split — the stage1.onnx + stage2.onnx that `ryk` runs,
# plus the voices and model.onnx. They ship as one zip because GitHub's web
# uploader rejects .onnx attachments.
#
# model.onnx is in the zip on purpose, even though `ryk` never executes it:
# resolve_assets() only takes the offline path when model.onnx *and* the voice
# are both present locally, and the stage dir defaults to model.onnx's parent.
# Drop it and a plain `ryk "Hello"` re-downloads it from HF, then looks for the
# stages in the HF cache and fails. It also doubles as the reference copy for
# re-running tools/split_kokoro.py.
#
# Usage:  tools/fetch_assets.sh [URL]
#         URL defaults to kokoro-onyx.zip on the pinned asset release below.
#         Override with $KUKURYKU_REPO / $KUKURYKU_ASSET_TAG, or pass a URL.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${KUKURYKU_REPO:-kaarrot/kukuryku}"
# Pinned deliberately, not `latest`: the assets are versioned separately from the
# code (they only change if the Kokoro weights or split_kokoro.py do), and
# /releases/latest/ would point at the first code release published without the
# zip attached, 404ing this script. Bump when a new asset release goes up.
TAG="${KUKURYKU_ASSET_TAG:-kokoro-onyx-model}"
URL="${1:-https://github.com/$REPO/releases/download/$TAG/kokoro-onyx.zip}"
# sha256 of the zip on the pinned tag above; re-zipping the assets changes it.
# Skipped when a URL is passed explicitly, since that is some other zip.
SHA256="469f4a2425a57454bddb93cbe4dfdb6628f8f1de3a9d85fe6193f77e258de594"

command -v curl  >/dev/null || { echo "error: need curl on PATH"  >&2; exit 1; }
command -v unzip >/dev/null || { echo "error: need unzip on PATH" >&2; exit 1; }

echo "== downloading =="
echo "   $URL"
curl -fL --progress-bar -o kokoro-onyx.zip.part "$URL"

if [ $# -eq 0 ] && command -v sha256sum >/dev/null; then
    echo "== verifying =="
    got="$(sha256sum kokoro-onyx.zip.part | cut -d' ' -f1)"
    if [ "$got" != "$SHA256" ]; then
        rm -f kokoro-onyx.zip.part
        echo "error: sha256 mismatch" >&2
        echo "  expected $SHA256" >&2
        echo "  got      $got" >&2
        exit 1
    fi
fi

mv kokoro-onyx.zip.part kokoro-onyx.zip

# The zip contains a top-level kokoro-onyx/, so extracting at the repo root puts
# the files exactly where ryk looks for them.
echo "== extracting -> kokoro-onyx/ =="
unzip -q -o kokoro-onyx.zip
rm -f kokoro-onyx.zip

du -sh kokoro-onyx/* 2>/dev/null || true
echo
echo "done — try:  ./target/release/ryk \"Hello world.\""
