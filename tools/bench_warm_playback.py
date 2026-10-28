#!/usr/bin/env python3
"""Unplugged, warm ryk playback comparison in A/B/B/A order on Termux.

Uses real audio, a private daemon/socket per run, and the thermal governor.
Compilation and a warmup utterance precede cooling and measurement. Measures
from request submission through audio-sink drain, not just synthesis completion.
The default text produces 180 seconds of audio with af_heart at speed 1.
"""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time


BAT = Path("/sys/class/power_supply/battery")
PARAGRAPH = (
    "She stood at the lighthouse and watched the boats come in. "
    "The wind was cold and the light turned slowly. "
    "A child on the pier waved once and then ran back to the road. "
    "By evening the fog had covered the rocks and the bell started up. "
    "They stayed until the last boat was only a sound."
)
DONE = re.compile(
    r"\[kokoro\] done: ([\d.]+)s audio \| infer ([\d.]+)s \| RTF ([\d.]+)"
    r" \| wall ([\d.]+)s \| wall RTF ([\d.]+) \| gap ([\d.]+)s"
)
HZ = os.sysconf("SC_CLK_TCK")


def read(path):
    return Path(path).read_text().strip()


def integer(path):
    try:
        return int(read(path))
    except (OSError, ValueError):
        return None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volume():
    if not shutil.which("termux-volume"):
        return None
    result = subprocess.run(["termux-volume"], capture_output=True, text=True, timeout=8)
    return json.loads(result.stdout) if result.returncode == 0 else None


def require_unplugged():
    status = read(BAT / "status")
    if status.lower() != "discharging":
        raise RuntimeError(f"Battery is {status}; measurement requires discharging")


def descendants(pid):
    # This Android kernel has no /proc/PID/task/TID/children files. Read PPIDs
    # from the visible process table instead, and fail if the root is hidden.
    children, visible = {}, set()
    for path in Path("/proc").glob("[0-9]*"):
        try:
            fields = read(path / "stat").rsplit(")", 1)[1].split()
            current, parent = int(path.name), int(fields[1])
            visible.add(current)
            children.setdefault(parent, []).append(current)
        except (OSError, ValueError):
            continue
    if pid not in visible:
        raise RuntimeError(f"Cannot observe daemon process {pid}")
    found, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(children.get(current, []))
    return found


def proc_cpu(pid, include_children=True):
    try:
        # Exclude comm, which can contain spaces. Include waited-for child CPU.
        fields = read(f"/proc/{pid}/stat").rsplit(")", 1)[1].split()
        indexes = [11, 12, 13, 14] if include_children else [11, 12]
        return sum(int(fields[i]) for i in indexes) / HZ
    except (OSError, ValueError):
        return 0.0


@contextmanager
def isolate_supervisor(pid, directory):
    """Temporarily pin an explicitly supplied, owned supervisor to CPU 0."""
    if pid is None:
        yield
        return
    process = Path(f"/proc/{pid}")
    if process.stat().st_uid != os.getuid():
        raise RuntimeError("Supervisor must belong to the current user")
    original = {int(t.name): sorted(os.sched_getaffinity(int(t.name)))
                for t in (process / "task").glob("*")}
    if pid not in original or 0 not in original[pid]:
        raise RuntimeError("Supervisor affinity does not include efficiency CPU 0")
    record = {"pid": pid, "comm": read(process / "comm"),
              "original": original, "temporary_cpu": 0, "restored": False}
    path = directory / "supervisor-affinity.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for tid in original:
            os.sched_setaffinity(tid, {0})
        print(f"Supervisor {pid} isolated on efficiency CPU 0", flush=True)
        yield
    finally:
        for task in (process / "task").glob("*"):
            tid = int(task.name)
            try:
                os.sched_setaffinity(tid, original.get(tid, original[pid]))
            except ProcessLookupError:
                pass
        record["restored"] = True
        path.write_text(json.dumps(record, indent=2) + "\n")


class Sampler:
    def __init__(self, root_pid, supervisor_pid=None):
        self.pid = root_pid
        self.supervisor_pid = supervisor_pid
        self.gold, self.little = [], []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                typ = read(zone / "type")
                if typ.startswith("cpu-1-") and typ.endswith("-usr"):
                    self.gold.append(zone / "temp")
                elif typ.startswith("cpu-0-") and typ.endswith("-usr"):
                    self.little.append(zone / "temp")
            except OSError:
                pass
        if not self.gold:
            raise RuntimeError("No live gold-cluster temperatures readable")
        self.current = BAT / "batt_current_ua_now"
        if integer(self.current) is None:
            raise RuntimeError("Explicit microamp battery-current file is required")
        self.seen_children = set()

    def gold_c(self):
        values = [integer(p) for p in self.gold]
        return max(v for v in values if v is not None) / 1000

    def sample(self):
        require_unplugged()
        pids = descendants(self.pid)
        for pid in pids - {self.pid}:
            try:
                self.seen_children.add(read(f"/proc/{pid}/comm"))
            except OSError:
                pass
        ua, uv = integer(self.current), integer(BAT / "voltage_now")
        if ua is None or uv is None or ua >= 0 or uv <= 0:
            raise RuntimeError(f"Invalid discharging battery sample: {ua} uA, {uv} uV")
        return {
            "monotonic_s": time.monotonic(), "epoch_s": time.time(),
            "battery_status": "Discharging", "current_ua": ua, "voltage_uv": uv,
            "power_w": -ua * uv / 1e12, "gold_c": self.gold_c(),
            "little_c": max(integer(p) or 0 for p in self.little) / 1000 if self.little else None,
            "battery_c": (integer(BAT / "temp") or 0) / 10,
            "battery_pct": integer(BAT / "capacity"),
            "cpu0_khz": integer("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
            "cpu4_khz": integer("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq"),
            "cpu7_khz": integer("/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq"),
            "process_cpu_s": sum(proc_cpu(pid) for pid in pids),
            "supervisor_cpu_s": proc_cpu(self.supervisor_pid, False) if self.supervisor_pid else 0,
        }


def integrate(samples, key):
    return sum((b["monotonic_s"] - a["monotonic_s"]) * (a[key] + b[key]) / 2
               for a, b in zip(samples, samples[1:]))


def send(sockpath, text):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(10)
        client.connect(str(sockpath))
        client.sendall(("af_heart\ten-us\t1.0\n" + text).encode())
        client.shutdown(socket.SHUT_WR)
        if client.recv(1024) != b"ok\n":
            raise RuntimeError("Daemon did not acknowledge the request")


def wait_idle(proc, logpath, offset, timeout, sampler=None, csvpath=None):
    deadline = time.monotonic() + timeout
    samples, last_notice = [], 0
    output = csvpath.open("w", newline="") if csvpath else None
    writer = None
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Daemon exited: {logpath}")
            if sampler:
                point = sampler.sample()
                samples.append(point)
                if output:
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(point))
                        writer.writeheader()
                    writer.writerow(point)
                    output.flush()
                elapsed = point["monotonic_s"] - samples[0]["monotonic_s"]
                if elapsed - last_notice >= 30:
                    print(f"  playback {elapsed:.0f}s, gold {point['gold_c']:.1f} C, "
                          f"battery {point['battery_pct']}%", flush=True)
                    last_notice = elapsed
            tail = logpath.read_text()[offset:]
            if DONE.search(tail) and descendants(proc.pid) == {proc.pid}:
                return tail, samples
            time.sleep(1)
        raise TimeoutError(f"Playback did not drain: {logpath}")
    finally:
        if output:
            output.close()


def run(args, label, index, binary):
    directory = args.out / f"{index}-{label}"
    directory.mkdir()
    sockpath = Path(tempfile.gettempdir()) / f"ryk-fusion-{os.getpid()}-{index}.sock"
    logpath = directory / "daemon.log"
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("KOKORO_", "RYK_")):
            del env[name]
    env.update({
        "KOKORO_TRACT_DIR": str(args.assets.resolve()), "KOKORO_VERBOSE": "1",
        "KOKORO_GOVERNOR": "thermal", "KOKORO_TRACT_FP16": "1",
        "RYK_SOCKET": str(sockpath), "RYK_IDLE_TIMEOUT": "off",
        "RYK_SINK_IDLE_MS": "200",
    })
    require_unplugged()
    print(f"{index}-{label}: compile and warm up", flush=True)
    before_volume = volume()
    proc = None
    try:
        with logpath.open("wb") as log:
            proc = subprocess.Popen([str(binary), "--serve"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 120
        while not sockpath.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"Daemon did not become ready: {logpath}")
            time.sleep(0.25)
        sampler = Sampler(proc.pid, args.supervisor_pid)
        send(sockpath, PARAGRAPH)
        wait_idle(proc, logpath, 0, 120)
        # No compilation, warmup, or cooldown work is in the measured interval.
        deadline = time.monotonic() + args.cool_max_s
        idle = []
        while True:
            require_unplugged()
            if sampler.gold_c() <= args.cool_c:
                idle = []
                for _ in range(args.idle_s + 1):
                    idle.append(sampler.sample())
                    time.sleep(1)
                if sampler.gold_c() <= args.cool_c:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError("Cooldown limit reached; refusing an unmatched hot start")
            print(f"{index}-{label}: cooling, gold {sampler.gold_c():.1f} C", flush=True)
            time.sleep(5)
        with (directory / "idle.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(idle[0]))
            writer.writeheader()
            writer.writerows(idle)
        print(f"{index}-{label}: measuring real playback, start {sampler.gold_c():.1f} C", flush=True)
        sampler.seen_children.clear()
        offset = len(logpath.read_text())
        first = sampler.sample()
        send(sockpath, " ".join([PARAGRAPH] * args.repeats))
        tail, samples = wait_idle(proc, logpath, offset, 420, sampler, directory / "samples.csv")
        samples.insert(0, first)
        # Include the request-to-first-poll interval in the saved observations.
        with (directory / "samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(first))
            writer.writeheader()
            writer.writerows(samples)
        (directory / "measured.log").write_text(tail)
        matches = list(DONE.finditer(tail))
        if len(matches) != 1 or "synth error" in tail or "write failed" in tail:
            raise RuntimeError(f"Invalid synthesis result: {logpath}")
        if "pacat" not in sampler.seen_children:
            raise RuntimeError("No real pacat process was observed")
        audio_s, infer_s, infer_rtf, wall_s, wall_rtf, gap_s = map(float, matches[0].groups())
        elapsed = samples[-1]["monotonic_s"] - first["monotonic_s"]
        joules = integrate(samples, "power_w")
        idle_w = integrate(idle, "power_w") / (idle[-1]["monotonic_s"] - idle[0]["monotonic_s"])
        cpu_s = samples[-1]["process_cpu_s"] - first["process_cpu_s"]
        after_volume = volume()
        result = {
            "label": label, "index": index, "binary_sha256": sha256(binary),
            "audio_s": audio_s, "elapsed_s": elapsed, "infer_s": infer_s,
            "infer_rtf": infer_rtf, "synthesis_wall_s": wall_s,
            "synthesis_wall_rtf": wall_rtf, "estimated_gap_s": gap_s,
            "joules": joules, "j_per_audio_s": joules / audio_s,
            "mean_w": joules / elapsed, "idle_w": idle_w,
            "net_j_per_audio_s": (joules - idle_w * elapsed) / audio_s,
            "cpu_s": cpu_s, "cpu_s_per_audio_s": cpu_s / audio_s,
            "start_gold_c": first["gold_c"],
            "end_gold_c": samples[-1]["gold_c"],
            "peak_gold_c": max(s["gold_c"] for s in samples),
            "mean_gold_c": integrate(samples, "gold_c") / elapsed,
            "start_battery_c": first["battery_c"],
            "end_battery_c": samples[-1]["battery_c"],
            "volume_before": before_volume, "volume_after": after_volume,
            "volume_unchanged": before_volume == after_volume,
            "observed_child_names": sorted(sampler.seen_children),
            "sample_count": len(samples),
            "supervisor_cpu_fraction": (samples[-1]["supervisor_cpu_s"] - first["supervisor_cpu_s"]) / elapsed,
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ["index", "label", "audio_s", "j_per_audio_s",
                         "cpu_s_per_audio_s", "estimated_gap_s", "start_gold_c"]}), flush=True)
        if not result["volume_unchanged"]:
            raise RuntimeError("Volume changed during the run; result is not controlled")
        return result
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        sockpath.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path.home() / ".local/share/kukuryku/kokoro-onyx")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--idle-s", type=int, default=15)
    parser.add_argument("--cool-c", type=float, default=46)
    parser.add_argument("--cool-max-s", type=int, default=300)
    parser.add_argument("--supervisor-pid", type=int,
                        help="Temporarily isolate this owned supervisor on efficiency CPU 0")
    args = parser.parse_args()
    if args.repeats < 1 or args.idle_s < 1:
        parser.error("repeats and idle-s must be positive")
    if shutil.which("ffplay"):
        parser.error("This harness expects real pacat playback, with no ffplay on PATH")
    if not shutil.which("pacat"):
        parser.error("pacat is required")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    binaries = {"A": args.baseline.resolve(), "B": args.fused.resolve()}
    manifest = {
        "binaries": {label: {"path": str(p), "sha256": sha256(p)} for label, p in binaries.items()},
        "order": "ABBA", "voice": "af_heart", "speed": 1, "paragraph": PARAGRAPH,
        "repeats": args.repeats, "governor": "thermal", "sink": shutil.which("pacat"),
        "measurement": "request submission through sink drain; compilation and warmup excluded",
        "cpu_scope": "daemon threads plus live and waited-for children; excludes detached PulseAudio",
        "screen_state": "not accessible to this session; user asked to keep it constant",
        "installed_ryk_sha256": sha256(Path.home() / "bin/ryk"),
        "runs": [],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with isolate_supervisor(args.supervisor_pid, args.out):
        for index, label in enumerate("ABBA", 1):
            manifest["runs"].append(run(args, label, index, binaries[label]))
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
