#!/usr/bin/env python3
"""Warm-daemon A/B energy runs with a fast battery sampler (A/A noise floor by default).

Differences from tools/energy_ab.sh: one persistent, warmed daemon per arm (so
model load and compilation are outside the energy window), a discard sink, the
gold-core pinning of tools/bench_warm_inference.py, a longer request (the
paragraph repeated), and the battery current read at 20 Hz instead of 1 Hz (the
fuel gauge updates roughly every 0.17 s). The kernel charge_counter is recorded
per run as well, to learn whether it is usable. Rows go to rows.log in the
key=value format tools/energy_ab_analyze.py reads.

Run unplugged with the screen off and nothing else running:
python3 tools/energy_warm_ab.py --out DIR [--start-delay-s 90] [options].
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True
from bench_warm_playback import BAT, DONE, PARAGRAPH, integer, isolate_supervisor, sha256, send, wait_idle, proc_cpu

CURRENT = BAT / "batt_current_ua_now"
VOLTAGE = BAT / "voltage_now"


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read())
    except (OSError, ValueError):
        return 0


class Meter:
    """Background battery sampler: (monotonic_s, microamps, microvolts, gold_c)."""

    def __init__(self, hz, gold_zones):
        self.period = 1.0 / hz
        self.zones = gold_zones
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def gold_c(self):
        return max(read_int(z) for z in self.zones) / 1000

    def run(self):
        nxt = time.monotonic()
        last_temp, gold = 0.0, self.gold_c()
        while not self.stop_event.is_set():
            t = time.monotonic()
            if t - last_temp >= 1.0:
                gold, last_temp = self.gold_c(), t
            self.rows.append((t, read_int(CURRENT), read_int(VOLTAGE), gold))
            nxt += self.period
            self.stop_event.wait(max(0.0, nxt - time.monotonic()))

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.rows


def energy(rows):
    """Trapezoid joules drawn from the battery (discharge current is negative)."""
    joules = 0.0
    for (t0, i0, v0, _), (t1, i1, v1, _) in zip(rows, rows[1:]):
        joules += (t1 - t0) * (-(i0 * v0) - (i1 * v1)) / 2 / 1e12
    return joules, rows[-1][0] - rows[0][0]


ALLOW_PLUGGED = False


def require_unplugged():
    status = (BAT / "status").read_text().strip()
    if status.lower() != "discharging" and not ALLOW_PLUGGED:
        raise RuntimeError(f"Battery is {status}; measurement requires discharging")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-a", type=Path, default=Path("target/release/ryk"))
    parser.add_argument("--binary-b", type=Path)
    parser.add_argument("--env-a", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--env-b", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--variant-a", default="A", help="name for what starts in slot A (rows carry variant=)")
    parser.add_argument("--variant-b", default="B")
    parser.add_argument("--swap-after", type=int, default=0, metavar="PAIRS",
                        help="after this many pairs, restart both daemons with the binaries swapped between "
                             "slots, so a slot bias can be separated from the variant effect")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path.home() / ".local/share/kukuryku/kokoro-onyx")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--max-min", type=float, default=25, help="stop starting pairs after this many minutes")
    parser.add_argument("--repeats", type=int, default=4, help="paragraph repeats per request (~19 s audio each)")
    parser.add_argument("--sample-hz", type=float, default=20)
    parser.add_argument("--idle-s", type=float, default=8, help="idle baseline seconds before each run")
    parser.add_argument("--cool-c", type=float, default=52)
    parser.add_argument("--cool-max-s", type=int, default=240, help="then start anyway and flag the run hot")
    parser.add_argument("--start-delay-s", type=float, default=0)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--min-battery", type=int, default=20, help="stop starting pairs below this percentage")
    parser.add_argument("--allow-plugged", action="store_true", help="plumbing tests only; joules are meaningless")
    args = parser.parse_args()
    global ALLOW_PLUGGED
    ALLOW_PLUGGED = args.allow_plugged
    binaries = {"A": args.binary_a.resolve(), "B": (args.binary_b or args.binary_a).resolve()}
    envs = {"A": args.env_a, "B": args.env_b}
    slots = {"A": {"binary": binaries["A"], "env": args.env_a, "variant": args.variant_a},
             "B": {"binary": binaries["B"], "env": args.env_b, "variant": args.variant_b}}
    text = " ".join([PARAGRAPH] * args.repeats)
    # Keep observation work off the pinned stage-1 core and the gold workers.
    os.sched_setaffinity(0, {0})
    if args.start_delay_s > 0:
        print(f"waiting {args.start_delay_s:.0f}s: unplug and turn the screen off", flush=True)
        time.sleep(args.start_delay_s)
    require_unplugged()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fakebin = out / "discard-sink"
    fakebin.mkdir()
    (fakebin / "ffplay").write_text(f"#!{shutil.which('sh')}\ncat >/dev/null\n")
    (fakebin / "ffplay").chmod(0o755)
    zones = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (z / "type").read_text().strip().startswith("cpu-1-") and (z / "type").read_text().strip().endswith("-usr"):
                zones.append(z / "temp")
        except OSError:
            pass
    if not zones:
        raise RuntimeError("No live gold-cluster sensors")
    gold = lambda: max(read_int(z) for z in zones) / 1000
    manifest = {
        "binaries": {k: {"path": str(p), "sha256": sha256(p)} for k, p in binaries.items()},
        "env": envs, "pairs": args.pairs, "repeats": args.repeats, "sample_hz": args.sample_hz,
        "sink": "discard", "governor": "off", "stage1_cpuset": "1", "stage2_cpuset": "4-6",
        "battery_pct": integer(BAT / "capacity"), "charge_counter_uah": integer(BAT / "charge_counter"),
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["termux-wake-lock"], check=False)

    processes, sockets, logs = {}, {}, {}
    rows_path = out / "rows.log"

    def start_daemon(label):
        slot = slots[label]
        env = os.environ.copy()
        for name in list(env):
            if name.startswith(("KOKORO_", "RYK_", "TRACT_CPU_", "TRACT_ONNX_")):
                del env[name]
        sock = Path(tempfile.gettempdir()) / f"ryk-ew-{os.getpid()}-{label}.sock"
        logpath = out / f"{label}-{slot['variant']}-daemon.log"
        env.update({
            "PATH": str(fakebin) + os.pathsep + env["PATH"],
            "KOKORO_TRACT_DIR": str(args.assets.resolve()), "KOKORO_VERBOSE": "1",
            "KOKORO_TRACT_FP16": "1", "KOKORO_GOVERNOR": "off",
            "KOKORO_TRACT_S1_CPUSET": "1", "KOKORO_TRACT_CPUSET": "4-6",
            "KOKORO_TRACT_THREADS": "3", "RYK_SOCKET": str(sock),
            "RYK_IDLE_TIMEOUT": "off", "RYK_SINK_IDLE_MS": "200",
        })
        for setting in slot["env"]:
            key, _, value = setting.partition("=")
            env[key] = value
        with logpath.open("wb") as log:
            proc = subprocess.Popen([str(slot["binary"]), "--serve"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[label], sockets[label], logs[label] = proc, sock, logpath
        deadline = time.monotonic() + 120
        while not sock.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"Daemon failed to start: {logpath}")
            time.sleep(0.25)
        print(label, slot["variant"], "ready; warming on the measured text", flush=True)
        send(sock, text)
        wait_idle(proc, logpath, 0, 300)

    def stop_daemons():
        for proc in processes.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        for sock in sockets.values():
            sock.unlink(missing_ok=True)
        processes.clear(), sockets.clear(), logs.clear()

    try:
        with isolate_supervisor(args.supervisor_pid, out):
            for label in slots:
                start_daemon(label)

            t0 = time.monotonic()
            prev_end = time.time()
            for pair in range(1, args.pairs + 1):
                if (integer(BAT / "capacity") or 100) < args.min_battery:
                    print(f"battery below {args.min_battery}%; stopping after {pair - 1} pairs", flush=True)
                    break
                if pair > 1 and time.monotonic() - t0 >= args.max_min * 60:
                    print(f"time budget of {args.max_min:.0f} min reached after {pair - 1} pairs", flush=True)
                    break
                if args.swap_after and pair == args.swap_after + 1:
                    print("swapping the binaries between daemon slots", flush=True)
                    stop_daemons()
                    for key in ("binary", "env", "variant"):
                        slots["A"][key], slots["B"][key] = slots["B"][key], slots["A"][key]
                    for label in slots:
                        start_daemon(label)
                for pos, label in enumerate("AB" if pair % 2 else "BA", 1):
                    require_unplugged()
                    waited, hot = 0, 0
                    while gold() > args.cool_c:
                        if waited >= args.cool_max_s:
                            hot = 1
                            break
                        time.sleep(5)
                        waited += 5
                    proc, logpath = processes[label], logs[label]
                    idle_meter = Meter(args.sample_hz, zones)
                    idle_meter.start()
                    time.sleep(args.idle_s)
                    idle_rows = idle_meter.stop()
                    idle_j, idle_t = energy(idle_rows)
                    offset = len(logpath.read_text())
                    meter = Meter(args.sample_hz, zones)
                    cc0, cpu0 = integer(BAT / "charge_counter"), proc_cpu(proc.pid)
                    wall0 = time.time()
                    meter.start()
                    started = time.monotonic()
                    start_c = gold()
                    send(sockets[label], text)
                    tail, _ = wait_idle(proc, logpath, offset, 400)
                    rows = meter.stop()
                    wall1 = time.time()
                    cc1, cpu_s = integer(BAT / "charge_counter"), proc_cpu(proc.pid) - cpu0
                    matches = list(DONE.finditer(tail))
                    if len(matches) != 1 or "synth error" in tail or "write failed" in tail:
                        raise RuntimeError(f"Failed request: pair {pair} {label}")
                    audio, infer, rtf, wall, wall_rtf, gap = map(float, matches[0].groups())
                    joules, elapsed = energy(rows)
                    idle_w = idle_j / idle_t
                    # Wall-clock time the process did not see: a frozen or suspended Termux shows up
                    # as a gap between runs or a window longer than its monotonic length.
                    stall_s = max(0.0, (wall0 - prev_end) - (waited + args.idle_s) - 15,
                                  (wall1 - wall0) - elapsed - 2)
                    volts = sum(r[2] for r in rows) / len(rows) / 1e6
                    cc_j = (cc0 - cc1) * 3.6e-3 * volts if cc0 and cc1 else 0.0
                    name = f"p{pair}{label}"
                    (out / f"{name}.samples").write_text("".join(f"{t:.3f} {i} {v} {g}\n" for t, i, v, g in rows))
                    (out / f"{name}.log").write_text(tail)
                    line = (f"pair={pair} arm={label} variant={slots[label]['variant']} pos={pos} cooled_s={waited} hot_start={hot} config={name} "
                            f"hz={args.sample_hz:g} wall_rtf={wall_rtf} gap_s={gap} cpu_s={cpu_s:.3f} "
                            f"real_s={elapsed:.3f} start_c={start_c:.1f} peak_c={max(r[3] for r in rows):.1f} "
                            f"stall_s={stall_s:.1f} mean_w={joules / elapsed:.4f} idle_w={idle_w:.4f} "
                            f"j_per_audio_s={joules / audio:.4f} "
                            f"net_j_per_audio_s={(joules - idle_w * elapsed) / audio:.4f} "
                            f"cc_j_per_audio_s={cc_j / audio:.4f} cc_delta_uah={cc0 - cc1} "
                            f"samples={len(rows)} battery=Discharging rc=0")
                    with rows_path.open("a") as f:
                        f.write(line + "\n")
                    print(line, flush=True)
                    prev_end = time.time()
    finally:
        stop_daemons()
        subprocess.run(["termux-wake-unlock"], check=False)
    analyzer = Path(__file__).with_name("energy_ab_analyze.py")
    if rows_path.exists():
        result = subprocess.run([sys.executable, str(analyzer), str(rows_path)], capture_output=True, text=True)
        (out / "analysis.txt").write_text(result.stdout + result.stderr)
        print(result.stdout, result.stderr, flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
