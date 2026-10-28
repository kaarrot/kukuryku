#!/usr/bin/env python3
"""Compare saved ryk binaries at raw FP32 stage boundaries (silent synthesis).

Each case is one sentence because KOKORO_TRACT_DUMP replaces its files per
sentence. This deliberately uses a discard sink; use the warm playback harness
for energy measurements.
"""

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fused", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--assets", type=Path,
        default=Path.home() / ".local/share/kukuryku/kokoro-onyx",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fakebin = out / "discard-sink"
    fakebin.mkdir()
    sink = fakebin / "ffplay"
    sink.write_text("#!/bin/sh\ncat >/dev/null\n")
    sink.chmod(0o755)
    binaries = {"A": args.baseline.resolve(), "B": args.fused.resolve()}
    texts = [
        "Hello there.",
        "She stood at the old lighthouse and watched the fishing boats return "
        "slowly to the quiet harbour before sunset.",
        "The old lighthouse keeper climbed the winding staircase every single "
        "evening at dusk, carrying his heavy brass lantern up the narrow stone "
        "steps to make certain the great lamp would burn steadily and brightly "
        "through the long and stormy night.",
    ]
    summary = {
        "binary_sha256": {k: sha256(v) for k, v in binaries.items()},
        "assets": str(args.assets.resolve()),
        "sink": "discard (parity only)", "cases": [],
    }
    for voice in ["af_heart", "am_michael"]:
        for speed, text in zip([0.8, 1.0, 1.2], texts):
            case = {"voice": voice, "speed": speed, "text": text, "runs": {}}
            case_dir = out / f"{voice}-{speed}"
            case_dir.mkdir()
            for label, binary in binaries.items():
                run_dir = case_dir / label
                run_dir.mkdir()
                env = os.environ.copy()
                # Avoid inherited diagnostics or scheduler overrides changing a run.
                for name in list(env):
                    if name.startswith(("KOKORO_", "RYK_")):
                        del env[name]
                env.update({
                    "PATH": str(fakebin) + os.pathsep + env["PATH"],
                    "KOKORO_TRACT_DIR": str(args.assets.resolve()),
                    "KOKORO_GOVERNOR": "off", "KOKORO_VERBOSE": "1",
                    "KOKORO_VOICE": voice, "KOKORO_SPEED": str(speed),
                    "KOKORO_TRACT_DUMP": str(run_dir),
                    "KOKORO_WAV": str(run_dir / "audio.wav"),
                })
                started = time.monotonic()
                with (run_dir / "run.log").open("wb") as log:
                    subprocess.run([str(binary), text], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=180, check=True)
                log = (run_dir / "run.log").read_text()
                if "1 sentence chunk(s)" not in log or "[1/1]" not in log:
                    raise RuntimeError(f"Expected one sentence: {run_dir}")
                case["runs"][label] = {"elapsed_s": time.monotonic() - started}
                print(f"{voice} speed={speed} {label} complete", flush=True)
            files = ["s1_feat640.f32", "s1_feat512.f32", "s1_dur.f32", "s2_out0.f32"]
            expected = set(files)
            for label in binaries:
                if {p.name for p in (case_dir / label).glob("*.f32")} != expected:
                    raise RuntimeError(f"Unexpected dump files in {case_dir / label}")
            case["tensors"] = {}
            for name in files:
                a, b = [(case_dir / label / name).read_bytes() for label in binaries]
                vals = array.array("f")
                vals.frombytes(a)
                if sys.byteorder != "little":
                    vals.byteswap()
                same = a == b
                finite = all(math.isfinite(v) for v in vals)
                case["tensors"][name] = {
                    "equal_bytes": same, "all_finite": finite, "elements": len(vals),
                    "A_sha256": hashlib.sha256(a).hexdigest(),
                    "B_sha256": hashlib.sha256(b).hexdigest(),
                }
                if not same or not finite:
                    raise RuntimeError(f"FP32 parity/finite check failed: {case_dir}/{name}")
            case["audio_s"] = case["tensors"]["s2_out0.f32"]["elements"] / 24000
            case["wav_equal"] = sha256(case_dir / "A/audio.wav") == sha256(case_dir / "B/audio.wav")
            summary["cases"].append(case)
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"PASS {voice} speed={speed}: {case['audio_s']:.3f}s audio; "
                  "all four raw FP32 tensors match", flush=True)


if __name__ == "__main__":
    main()
