#!/usr/bin/env python3
"""Record repeatable one-shot or warm-daemon RTF measurements on an Android phone.

Uses a private daemon/socket, records the exact binary/assets/text and CPU
topology, and tolerates phones that hide thermal and battery sysfs files.
Real playback and a discard sink are separate measurement modes. CPU time is
not an energy measurement. No installed executable or existing daemon changes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from bench_warm_playback import DONE, proc_cpu, send


CHUNK = re.compile(
    r"\[(\d+)/(\d+)\] (\d+) tokens \| ([\d.]+)s audio"
    r" \| prep ([\d.]+)s \| infer ([\d.]+)s \| RTF ([\d.]+) \| gap ([\d.]+)s"
)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def topology():
    result = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        result[cpu.name] = {
            name: read(cpu / name) for name in [
                "cpu_capacity", "cpufreq/cpuinfo_max_freq", "cpufreq/scaling_max_freq",
                "cpufreq/related_cpus", "cpufreq/scaling_governor", "topology/cluster_id",
            ]
        }
    return result


def observe(pid, cores):
    fields = read(f"/proc/{pid}/stat")
    if fields is None:
        return None
    fields = fields.rsplit(")", 1)[1].split()
    return {
        "monotonic_s": time.monotonic(), "process_cpu_s": proc_cpu(pid),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        "reported_frequency_khz": {
            core: read(f"/sys/devices/system/cpu/{core}/cpufreq/scaling_cur_freq")
            for core in cores
        },
    }


def parse_result(log):
    matches = list(DONE.finditer(log))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completed utterance, found {len(matches)}")
    for failure in ["synth error", "write failed", "sched_setaffinity", "Error:"]:
        if failure in log:
            raise RuntimeError(f"Run contains {failure!r}; inspect the saved log")
    audio, infer, infer_rtf, wall, wall_rtf, gap = map(float, matches[0].groups())
    first = re.search(r"first audio at ([\d.]+)s", log)
    return {
        "audio_s": audio, "infer_s": infer, "infer_rtf": infer_rtf,
        "synthesis_wall_s": wall, "synthesis_wall_rtf": wall_rtf,
        "estimated_gap_s": gap,
        "first_audio_from_synthesis_start_s": float(first[1]) if first else None,
        "chunks": [dict(zip(
            ["index", "count", "tokens", "audio_s", "prep_s", "infer_s", "infer_rtf", "gap_s"],
            [int(v) if i < 3 else float(v) for i, v in enumerate(match.groups())],
        )) for match in CHUNK.finditer(log)],
    }


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def wait_run(proc, logpath, offset, mode, timeout, cores):
    started = time.monotonic()
    deadline = started + timeout
    points, first_observed, notice = [], None, started
    while time.monotonic() < deadline:
        log = logpath.read_text()[offset:]
        point = observe(proc.pid, cores)
        if point:
            points.append(point)
        if first_observed is None and "first audio at " in log:
            first_observed = time.monotonic() - started
        if mode == "oneshot" and proc.poll() is not None:
            if proc.returncode != 0:
                raise RuntimeError(f"Command exited {proc.returncode}: {logpath}")
            return log, points, first_observed
        if mode == "warm":
            if proc.poll() is not None:
                raise RuntimeError(f"Daemon exited: {logpath}")
            if DONE.search(log) and "audio sink idle; closing" in log:
                return log, points, first_observed
        if time.monotonic() - notice >= 25:
            print(f"  {time.monotonic() - started:.0f}s elapsed: {logpath.name}", flush=True)
            notice = time.monotonic()
        time.sleep(0.1)
    raise TimeoutError(f"Run timed out after {timeout}s: {logpath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path.home() / "bin/ryk")
    parser.add_argument("--assets", type=Path,
                        default=Path.home() / ".local/share/kukuryku/kokoro-onyx")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["oneshot", "warm"], default="warm")
    parser.add_argument("--sink", choices=["real", "discard"], default="real")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rest-secs", type=float, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--wav", action="store_true", help="save WAVs for one-shot runs")
    parser.add_argument("--conditions", default="not independently controlled")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.rest_secs < 0:
        parser.error("repeats must be positive; warmups and rest-secs must be nonnegative")
    if args.wav and args.mode != "oneshot":
        parser.error("--wav requires --mode oneshot")
    args.binary, args.assets, args.out = (
        path.resolve() for path in (args.binary, args.assets, args.out)
    )
    text = args.text_file.read_text().strip()
    if not text:
        parser.error("text-file must not be empty")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KOKORO_", "RYK_", "TRACT_CPU_", "TRACT_ONNX_"))}
    overrides = {
        "KOKORO_TRACT_DIR": str(args.assets), "KOKORO_VOICE": "af_heart",
        "KOKORO_LANG": "en-us", "KOKORO_SPEED": "1.0", "KOKORO_VERBOSE": "1",
        "RYK_HEADSET": "0", "RYK_NOTIFY": "0", "RYK_EARCONS": "0",
        "RYK_IDLE_TIMEOUT": "off",
    }
    for setting in args.env:
        key, sep, value = setting.partition("=")
        if not sep or not key:
            parser.error("--env expects KEY=VALUE")
        overrides[key] = value
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "corpus.txt").write_text(text + "\n")
    if args.sink == "discard":
        fakebin = args.out / "discard-sink"
        fakebin.mkdir()
        fakeplay = fakebin / "ffplay"
        fakeplay.write_text(f"#!{shutil.which('sh')}\nexec cat >/dev/null\n")
        fakeplay.chmod(0o755)
        env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
        overrides["RYK_SINK_IDLE_MS"] = "200"
    elif shutil.which("ffplay") or not shutil.which("pacat"):
        parser.error("real mode requires pacat with no ffplay ahead of it on PATH")
    env.update(overrides)
    sock = Path(tempfile.gettempdir()) / f"ryk-phone-{os.getpid()}.sock"
    env["RYK_SOCKET"] = str(sock)
    cpus = topology()
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "conditions": args.conditions,
        "source_commit": command(["git", "rev-parse", "HEAD"]),
        "binary": {"path": str(args.binary), "sha256": sha256(args.binary)},
        "assets": {name: sha256(args.assets / name) for name in
                   ["stage1.onnx", "stage2.onnx", "voices/af_heart.bin"]},
        "model": command(["getprop", "ro.product.model"]),
        "soc": command(["getprop", "ro.soc.model"]),
        "android": command(["getprop", "ro.build.version.release"]),
        "kernel": command(["uname", "-a"]), "topology": cpus,
        "original_observer_affinity": allowed, "observer_cpuset": [allowed[0]],
        "espeak": command(["espeak-ng", "--version"]),
        "battery_before": command(["termux-battery-status"]),
        "environment": overrides, "mode": args.mode, "sink": args.sink,
        "warmups": args.warmups if args.mode == "warm" else 0,
        "rest_secs": args.rest_secs, "text": text,
        "cpu_time_scope": "process plus waited-for children; excludes detached PulseAudio",
        "thermal_sysfs_readable": any(read(p / "temp") is not None
                                      for p in Path("/sys/class/thermal").glob("thermal_zone*")),
        "runs": [],
    }
    path = args.out / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    proc = None
    try:
        logpath = args.out / "daemon.log"
        if args.mode == "warm":
            with logpath.open("wb") as log:
                proc = subprocess.Popen([str(args.binary), "--serve", "-v"], env=env,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            deadline = time.monotonic() + 120
            while not sock.exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Daemon did not start: {logpath}")
                time.sleep(0.1)
            ready = re.search(r"pipeline ready in ([\d.]+)s", logpath.read_text())
            manifest["pipeline_compile_s"] = float(ready[1]) if ready else None
            for index in range(args.warmups):
                offset = logpath.stat().st_size
                send(sock, text)
                log, _, _ = wait_run(proc, logpath, offset, "warm", args.timeout, cpus)
                parse_result(log)
                (args.out / f"warmup-{index + 1}.log").write_text(log)
            print("Warmup complete", flush=True)
        for index in range(1, args.repeats + 1):
            time.sleep(args.rest_secs)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            offset = logpath.stat().st_size if args.mode == "warm" else 0
            cpu_before = proc_cpu(proc.pid) if args.mode == "warm" else None
            started = time.monotonic()
            if args.mode == "oneshot":
                logpath = args.out / f"run-{index}.log"
                if args.wav:
                    env["KOKORO_WAV"] = str(args.out / f"run-{index}.wav")
                with logpath.open("wb") as log:
                    proc = subprocess.Popen([str(args.binary), "-v", text], env=env,
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            else:
                send(sock, text)
            log, points, first_observed = wait_run(
                proc, logpath, offset, args.mode, args.timeout, cpus)
            elapsed = time.monotonic() - started
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = (after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
                   if args.mode == "oneshot" else proc_cpu(proc.pid) - cpu_before)
            result = parse_result(log)
            result.update({"run": index, "elapsed_through_drain_s": elapsed, "cpu_s": cpu,
                           "cpu_s_per_audio_s": cpu / result["audio_s"],
                           "observed_first_audio_log_from_request_s": first_observed,
                           "sampled_peak_rss_bytes": max(p["rss_bytes"] for p in points) if points else None})
            (args.out / f"run-{index}.log").write_text(log)
            (args.out / f"run-{index}-telemetry.json").write_text(json.dumps(points) + "\n")
            manifest["runs"].append(result)
            save()
            print(json.dumps(result), flush=True)
        manifest["battery_after"] = command(["termux-battery-status"])
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    finally:
        stop(proc)
        sock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
