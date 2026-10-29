#!/usr/bin/env python3
"""Run alternating serial/pool microbenchmarks; measure CPU work, not energy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from bench_warm_playback import isolate_supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--cool-c", type=float, default=50)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("use at least two rounds to alternate execution order")
    os.sched_setaffinity(0, {0})
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    probe = args.probe.resolve()
    cases = [(f"f16vec-{m}", ["matmul", "f16vec", m, 128, 1, 256])
             for m in [128, 256, 512, 1024, 1090]]
    cases += [
        ("f16vec-dynamic", ["matmul", "f16vec", 2340, 256, 1, 128]),
        ("f32vec-256-256", ["matmul", "f32vec", 256, 256, 1, 128]),
        ("f32vec-1024-256", ["matmul", "f32vec", 1024, 256, 1, 64]),
        ("f32vec-128-128", ["matmul", "f32vec", 128, 128, 1, 256]),
        ("f32vec-single-tile", ["matmul", "f32vec", 32, 128, 1, 512]),
        ("f16mat-one-row-tile", ["matmul", "f16mat", 64, 128, 13, 128]),
        ("f16mat-256-128-13", ["matmul", "f16mat", 256, 128, 13, 128]),
        ("f16mat-control", ["matmul", "f16mat", 512, 512, 59, 16]),
    ]
    cases += [(f"{kind}-{c}-{f}", ["rows", kind, c, f, 0, repeats])
              for kind, c, f, repeats in [
                  ("leaky16", 512, 59, 32), ("leaky16", 1024, 59, 32),
                  ("leaky16", 256, 118, 32), ("leaky16", 512, 13, 128),
                  ("adain16", 128, 13, 128), ("adain16", 256, 1180, 8),
                  ("adain16", 128, 7081, 4), ("snake", 256, 1180, 16),
                  ("snake", 128, 7081, 8), ("leaky", 512, 59, 64),
                  ("leaky16", 1, 59, 512),
              ]]
    sensors = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text().strip()
            if name.startswith("cpu-1-") and name.endswith("-usr"):
                sensors.append(zone / "temp")
        except OSError:
            pass
    if not sensors:
        raise RuntimeError("No live gold-cluster sensors")
    gold = lambda: max(int(p.read_text()) for p in sensors) / 1000
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KOKORO_", "RYK_", "TRACT_CPU_"))}
    # A rebuilt probe may link the candidate dispatcher. Keep the pool arm on
    # the original policy so this remains a serial-versus-pool comparison.
    env["TRACT_CPU_MMM_MIN_WORK"] = "0"
    manifest = {"probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                "pinning": "probe/caller/workers=4-6; observer/supervisor=0",
                "method": "100ms park before arm; 30ms worker tail counted in process CPU; prepacked matmuls",
                "environment": {"TRACT_CPU_MMM_MIN_WORK": "0"},
                "rounds": args.rounds, "cases": []}
    with isolate_supervisor(args.supervisor_pid, out):
        for label, parameters in cases:
            deadline = time.monotonic() + 240
            while gold() > args.cool_c:
                if time.monotonic() > deadline:
                    raise RuntimeError("Cooldown timed out")
                time.sleep(5)
            start_c = gold()
            with (out / (label + ".csv")).open("w") as stdout, (out / (label + ".log")).open("w") as stderr:
                subprocess.run([str(probe), *map(str, parameters), str(args.rounds)],
                               stdout=stdout, stderr=stderr, env=env, timeout=180, check=True)
            rows = list(csv.DictReader((out / (label + ".csv")).open()))
            assert len(rows) == args.rounds * 2 and all(r["bit_equal"] == "true" for r in rows)
            assert "parity: finite=true bit_equal=true" in (out / (label + ".log")).read_text()
            means = {}
            for mode in ["serial", "pool3"]:
                group = [r for r in rows if r["executor"] == mode]
                means[mode] = {key: statistics.median(float(r[key]) / int(r["repeats"]) for r in group)
                               for key in ["work_wall_s", "cpu_including_tail_s"]}
            change = {k: 100 * (means["serial"][k] / means["pool3"][k] - 1) for k in means["serial"]}
            case = {"label": label, "arguments": parameters + [args.rounds],
                    "start_gold_c": start_c, "end_gold_c": gold(),
                    "median_per_call_s": means, "serial_change_percent": change, "initial_bit_parity": True}
            manifest["cases"].append(case)
            (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(label, "serial CPU", round(change["cpu_including_tail_s"], 1), "%",
                  "wall", round(change["work_wall_s"], 1), "%", flush=True)


if __name__ == "__main__":
    main()
