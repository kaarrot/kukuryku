#!/usr/bin/env python3
"""Build Resize numeric tests/probe against cached release data libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    deps = repo / "target/release/deps"
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    data = list(deps.glob("libtract_data-*.rlib"))
    if len(data) != 1:
        raise RuntimeError("Expected one cached release tract-data library")

    def profile(path, crate):
        tag = path.stem.removeprefix("lib" + crate + "-")
        directory = deps.parent / ".fingerprint" / (crate.replace("_", "-") + "-" + tag)
        files = list(directory.glob("lib-*.json"))
        return json.loads(files[0].read_text())["profile"] if files else None

    wanted = profile(data[0], "tract_data")
    libraries = {"tract_data": str(data[0])}
    choices = list(deps.glob("liblibc-*.rlib"))
    if len(choices) > 1:
        choices = [p for p in choices if profile(p, "libc") == wanted]
    if len(choices) != 1:
        raise RuntimeError("Expected one matching libc library")
    libraries["libc"] = str(choices[0])
    files = ["tools/bench_resize.rs", "third_party/tract/tract-onnx/src/ops/resize/contiguous.rs"]
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "source_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files},
                "libraries": libraries, "commands": []}
    base = ["rustc", "--edition=2024", "-C", "opt-level=3", "-C", "lto=thin", "-C", "codegen-units=1",
            "-L", "dependency=" + str(deps)]
    for crate, library in libraries.items():
        base.extend(["--extern", crate + "=" + library])
    for name, extra in [("resize-tests", ["--test"]), ("resize-probe", [])]:
        command = base + extra + [str(repo / files[0]), "-o", str(out / name)]
        print("Building", name, flush=True)
        started = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        manifest["commands"].append({"command": command, "elapsed_s": time.monotonic() - started, "returncode": result.returncode})
        (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            print((out / (name + ".log")).read_text())
            raise SystemExit(result.returncode)
    print("Resize tests and probe built", flush=True)


if __name__ == "__main__":
    main()
