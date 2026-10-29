#!/usr/bin/env python3
"""Benchmark actual Resize geometries from a saved per-node profile summary."""
import argparse
import csv
import hashlib
import json
import math
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
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("use at least two rounds to alternate execution order")
    os.sched_setaffinity(0, {0})
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    attrs = {r["name"]: r for r in json.loads(args.attributes.read_text())}
    profiles = json.loads(args.profiles.read_text())
    cases = {}
    for profile in profiles["rounds"][0]["profiles"]:
        for node in profile["resize_nodes"]:
            entry = attrs[node["name"]]
            shape = node["inputs"][0]["shape"]
            output = node["outputs"][0]["shape"]
            assert shape[:-1] == output[:-1]
            channels = math.prod(shape[:-1])
            mode = entry["attributes"]["mode"]
            a = entry["attributes"]
            assert (mode == "nearest" and a["coordinate_transformation_mode"] == "asymmetric"
                    and a["nearest_mode"] == "floor") or (mode == "linear"
                    and a["coordinate_transformation_mode"] == "half_pixel")
            assert all(scale == 1.0 for scale in entry["scales"][:-1])
            scale = entry["scales"][-1]
            key = (mode, channels, shape[-1], output[-1], scale)
            cases.setdefault(key, []).append(node["name"])
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
    probe = args.probe.resolve()
    manifest = {"probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                "profile_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
                "attributes_sha256": hashlib.sha256(args.attributes.read_bytes()).hexdigest(),
                "rounds": args.rounds, "cases": []}
    with isolate_supervisor(args.supervisor_pid, out):
        for (mode, channels, length, output, scale), names in cases.items():
            deadline = time.monotonic() + 240
            while gold() > 50:
                if time.monotonic() > deadline:
                    raise RuntimeError("Cooldown timed out")
                time.sleep(5)
            label = f"{mode}-c{channels}-in{length}-out{output}"
            size = channels * max(length, output)
            repeats = 1 if size >= 1_000_000 else 2 if size >= 200_000 else 8
            params = [mode, channels, length, output, scale, repeats, args.rounds]
            start_c = gold()
            with (out / (label + ".csv")).open("w") as stdout, (out / (label + ".log")).open("w") as stderr:
                subprocess.run([str(probe), *map(str, params)], stdout=stdout, stderr=stderr,
                               check=True, timeout=180)
            rows = list(csv.DictReader((out / (label + ".csv")).open()))
            assert len(rows) == 2 * args.rounds and all(r["bit_equal"] == "true" for r in rows)
            medians = {variant: {key: statistics.median(float(r[key]) / repeats for r in rows if r["variant"] == variant)
                                 for key in ["wall_s", "cpu_s"]} for variant in ["generic", "fast"]}
            change = {key: 100 * (medians["fast"][key] / medians["generic"][key] - 1) for key in ["wall_s", "cpu_s"]}
            case = {"label": label, "arguments": params, "model_nodes": names,
                    "start_gold_c": start_c, "end_gold_c": gold(),
                    "median_per_call_s": medians, "fast_change_percent": change,
                    "initial_bit_parity": True}
            manifest["cases"].append(case)
            (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(label, "CPU", round(change["cpu_s"], 1), "%", flush=True)


if __name__ == "__main__":
    main()
