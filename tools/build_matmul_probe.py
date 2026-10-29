#!/usr/bin/env python3
"""Build the matmul diagnostic against the existing release rlibs, without Cargo.

Requires a prior release build of ryk. Writes a separate executable and source
snapshot under tools/bench_out; never overwrites ryk or installs a binary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


INVENTORY = r'''
pub fn inventory(dir: &Path, output: &Path) -> Result<()> {
    use tract_core::ops::matmul::optimized::{OptMatMul, ProtoFusedSpec};
    let stages = [
        ("stage1", Stage::build(&dir.join("stage1.onnx"), &[
            ("input_ids", &[Fixed(1), Sym("N")]),
            ("style", &[Fixed(1), Fixed(256)]), ("speed", &[Fixed(1)]),
        ], true)?),
        ("stage2", Stage::build(&dir.join("stage2.onnx"), &[
            (S1_FEAT_640, &[Fixed(1), Fixed(640), Sym("N")]),
            (S1_FEAT_512, &[Fixed(1), Fixed(512), Sym("N")]),
            (S2_ALIGNMENT, &[Sym("N"), Sym("F")]),
            ("style", &[Fixed(1), Fixed(256)]),
        ], true)?),
    ];
    let mut records = Vec::new();
    for (stage, plan) in &stages {
        let model = plan.runnable.model();
        for node in model.nodes() {
            if let Some(mm) = node.op_as::<OptMatMul>() {
                let products: Vec<_> = mm.micro_ops.iter().filter_map(|op| {
                    if let ProtoFusedSpec::AddMatMul { geo, a, b, .. } = op {
                        Some(serde_json::json!({"k":geo.k.to_string(),"a":a,"b":b}))
                    } else { None }
                }).collect();
                records.push(serde_json::json!({
                    "stage":stage, "id":node.id, "name":node.name,
                    "dtype":format!("{:?}",mm.c_fact.datum_type),
                    "m_axis":mm.c_m_axis, "n_axis":mm.c_n_axis,
                    "shape":mm.c_fact.shape.iter().map(|d|d.to_string()).collect::<Vec<_>>(),
                    "kernels":mm.mmm.iter().map(|k|k.name()).collect::<Vec<_>>(),
                    "products":products,"info":node.op().info()?,
                    "input_nodes":node.inputs.iter().map(|o|model.node(o.node).name.clone()).collect::<Vec<_>>(),
                }));
            }
        }
    }
    std::fs::write(output, serde_json::to_string_pretty(&records)?)?;
    eprintln!("Saved {} matmul nodes to {}",records.len(),output.display());
    Ok(())
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-inventory", action="store_true", help="omit the ONNX plan inventory mode")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dense-only", action="store_true", help="build the smaller stage-1 dense probe")
    mode.add_argument("--dispatch-only", action="store_true", help="build the small-operation dispatch probe")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    deps = repo / "target/release/deps"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source = (repo / "src/tract_backend.rs").read_text()
    # Reuse the exact plan build and precision translation; keep private access
    # inside a generated module. No inference code is edited in the repository.
    prefix = source.split("    /// Run the cached plan;", 1)[0] + "}\n"
    prefix = prefix.replace("use crate::info;", "")
    start = source.index("fn fp16_enabled()")
    end = source.index("\n}\n", source.index("fn apply_fp16(")) + 3
    backend = prefix + source[start:end] + INVENTORY
    backend = "\n".join("//" + s[3:] if s.startswith("//!") else s for s in backend.splitlines())
    generated = out / "backend_probe.rs"
    generated.write_text(backend + "\n")
    env["MATMUL_BACKEND_PROBE"] = str(generated)
    if args.dispatch_only:
        math_source = (repo / "third_party/tract/tract-core/src/ops/math/mod.rs").read_text()
        def function(name):
            start = math_source.index("fn " + name + "(")
            end = math_source.index("\n}\n", start) + 3
            return math_source[start:end]
        rows_start = math_source.index("const ADAIN_LANES:")
        rows_end = math_source.index("\nimpl EvalOp for AdaIn", rows_start)
        rows = function("sin_sq_f32") + function("snake_f32") + math_source[rows_start:rows_end]
        row_path = out / "row_probe.rs"
        row_path.write_text(rows)
        env["TRACT_DISPATCH_ROW_PROBE"] = str(row_path)

    def profile(path, crate):
        tag = path.stem.removeprefix("lib" + crate + "-")
        directory = deps.parent / ".fingerprint" / (crate.replace("_", "-") + "-" + tag)
        files = list(directory.glob("lib-*.json"))
        return json.loads(files[0].read_text())["profile"] if files else None

    data = list(deps.glob("libtract_data-*.rlib"))
    if len(data) != 1:
        raise RuntimeError("Expected one cached tract-data release build")
    wanted_profile = profile(data[0], "tract_data")
    small = args.dense_only or args.dispatch_only
    names = ["tract_data", "tract_linalg", "libc"] if small else ["tract_core", "tract_data", "tract_linalg", "libc", "num_integer", "anyhow", "serde_json"]
    inventory = not args.no_inventory and not small
    if inventory:
        names.append("tract_onnx")
    command = ["rustc", "--edition=2024", "-C", "opt-level=3", "-C", "lto=thin",
               "-C", "codegen-units=1", "-L", "dependency=" + str(deps)]
    if inventory:
        command.extend(["--cfg", "probe_inventory"])
    selected = {}
    for name in names:
        paths = list(deps.glob("lib" + name + "-*.rlib"))
        if len(paths) > 1:
            paths = [p for p in paths if profile(p, name) == wanted_profile]
        if len(paths) != 1:
            raise RuntimeError(f"Expected one matching cached release library for {name}: {paths}")
        selected[name] = str(paths[0])
        command.extend(["--extern", name + "=" + str(paths[0])])
    filename = "bench_tiny_dispatch.rs" if args.dispatch_only else "bench_dense_matmul.rs" if args.dense_only else "bench_matmul.rs"
    bench_source = repo / "tools" / filename
    command.extend([str(bench_source), "-o", str(out / "matmul-probe")])
    manifest = {"command": command, "libraries": selected,
                "backend_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "probe_source_sha256": hashlib.sha256(bench_source.read_bytes()).hexdigest(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    if args.dispatch_only:
        manifest["math_source_sha256"] = hashlib.sha256(math_source.encode()).hexdigest()
        manifest["row_source_sha256"] = hashlib.sha256(row_path.read_bytes()).hexdigest()
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Building separate matmul probe; log: {out / 'build.log'}", flush=True)
    start = time.monotonic()
    with (out / "build.log").open("w") as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update({"build_s": time.monotonic() - start, "returncode": result.returncode})
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        print((out / "build.log").read_text())
        raise SystemExit(result.returncode)
    print(f"Built {out / 'matmul-probe'} in {manifest['build_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
