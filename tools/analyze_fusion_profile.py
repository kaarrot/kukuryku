#!/usr/bin/env python3
"""Inspect KOKORO_TRACT_PROFILE_JSON files for direct AdaIN/activation chains.

Usage: python3 tools/analyze_fusion_profile.py PROFILE.json [PROFILE.json ...]
Reports measured operator time, graph eligibility and concrete activation sizes.
Operator time is a target budget, not time that fusion is guaranteed to remove.
"""

import argparse
import collections
import json
from pathlib import Path


def analyze(path):
    profile = json.loads(path.read_text())
    nodes = {n["id"]: n for n in profile["nodes"]}
    if len(nodes) != len(profile["nodes"]):
        raise ValueError(f"{path}: expected one execution per outer graph node")

    def successor(node):
        if node is None or len(node["outputs"]) != 1:
            return None
        output = node["outputs"][0]
        if output["model_output"] or len(output["successors"]) != 1:
            return None
        return nodes.get(output["successors"][0]["node"])

    def cast(node, source, dest):
        return (
            node is not None
            and node["op"] == "Cast"
            and len(node["inputs"]) == len(node["outputs"]) == 1
            and node["inputs"][0]["dtype"] == source
            and node["outputs"][0]["dtype"] == dest
        )

    op_time = collections.Counter()
    op_count = collections.Counter()
    for node in nodes.values():
        op_time[node["op"]] += node["seconds"]
        op_count[node["op"]] += 1

    chains = []
    for norm in nodes.values():
        if norm["op"] != "AdaIN":
            continue
        before = nodes.get(norm["inputs"][0]["node"])
        after = successor(norm)
        activation = after if after and after["op"] in ("Snake", "LeakyRelu") else None
        after_activation = successor(activation)
        input_cast = before if cast(before, "F16", "F32") and successor(before) == norm else None
        output_cast = after_activation if cast(after_activation, "F32", "F16") else None
        members = [n for n in (input_cast, norm, activation, output_cast) if n]
        chains.append({
            "norm_id": norm["id"],
            "name": norm["name"],
            "input_producer": before["op"] if before else "constant/source",
            "norm_successors": [
                nodes[s["node"]]["op"] for s in norm["outputs"][0]["successors"]
                if s["node"] in nodes
            ],
            "activation": activation["op"] if activation else None,
            "input_cast": input_cast["id"] if input_cast else None,
            "output_cast": output_cast["id"] if output_cast else None,
            "closed_f16_chain": bool(input_cast and activation and output_cast),
            "shape": norm["outputs"][0]["shape"],
            "fp32_activation_bytes": norm["outputs"][0]["bytes"],
            "nodes": [n["id"] for n in members],
            "seconds": sum(n["seconds"] for n in members),
            "cast_seconds": sum(n["seconds"] for n in (input_cast, output_cast) if n),
        })

    groups = {}
    for label, selected in (
        ("all_adain", chains),
        ("adain_activation", [c for c in chains if c["activation"]]),
        ("closed_f16_chain", [c for c in chains if c["closed_f16_chain"]]),
    ):
        ids = {i for c in selected for i in c["nodes"]}
        groups[label] = {
            "count": len(selected),
            "seconds": sum(nodes[i]["seconds"] for i in ids),
            "cast_seconds": sum(nodes[i]["seconds"] for i in ids if nodes[i]["op"] == "Cast"),
            "activation_bytes_sum": sum(c["fp32_activation_bytes"] for c in selected),
        }

    return {
        "profile": str(path),
        "stage": profile["stage"],
        "op_wall_seconds": profile["op_wall_seconds"],
        "operators": [
            {"op": op, "seconds": seconds, "count": op_count[op]}
            for op, seconds in op_time.most_common()
        ],
        "groups": groups,
        "chains": chains,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, help="also save complete machine-readable analysis")
    args = parser.parse_args()
    summaries = [analyze(path) for path in args.profiles]
    for summary in summaries:
        total = summary["op_wall_seconds"]
        print(f"{summary['profile']}: {summary['stage']} eval={total:.4f}s")
        for op in summary["operators"][:10]:
            print(f"  {op['op']:24} {op['count']:4} {op['seconds']:.4f}s {100 * op['seconds'] / total:5.1f}%")
        for label, group in summary["groups"].items():
            print(
                f"  {label}: count={group['count']} region={group['seconds']:.4f}s "
                f"casts={group['cast_seconds']:.4f}s "
                f"FP32 activation sum={group['activation_bytes_sum'] / 2**20:.2f} MiB"
            )
        print("  graph patterns:")
        patterns = collections.Counter(
            (c["input_producer"], tuple(c["norm_successors"]), c["input_cast"] is not None, c["output_cast"] is not None)
            for c in summary["chains"]
        )
        for pattern, count in patterns.most_common():
            print(f"    {count:2} {pattern}")
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
