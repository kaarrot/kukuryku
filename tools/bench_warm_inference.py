#!/usr/bin/env python3
"""Compare warm inference CPU work with a discard sink in alternating A/B pairs.

Compilation, warmup and audio-sink drain are outside the CPU endpoints except
for the measured request's own sink work. No battery-energy claim is made.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from bench_warm_playback import DONE, isolate_supervisor, proc_cpu, requested_affinity, send, wait_idle


TEXT = (
    "Hello there. She stood at the old lighthouse and watched the fishing boats "
    "return slowly to the quiet harbour before sunset. The old lighthouse keeper "
    "climbed the winding staircase every single evening at dusk, carrying his "
    "heavy brass lantern up the narrow stone steps to make certain the great lamp "
    "would burn steadily and brightly through the long and stormy night."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thread_stats(pid):
    """Persistent daemon threads only; child CPU remains in the process total."""
    result = {}
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            tid = int(task.name)
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
            cpus = requested_affinity(tid)
            group = "little" if cpus and cpus <= {0, 1, 2, 3} else "gold" if cpus and cpus <= {4, 5, 6} else "other"
            result[tid] = {
                "start": int(fields[19]), "group": group,
                "cpu_s": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                "voluntary_switches": int(status["voluntary_ctxt_switches"]),
                "involuntary_switches": int(status["nonvoluntary_ctxt_switches"]),
            }
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def thread_delta(before, after):
    totals = {group: {k: 0 for k in ["cpu_s", "voluntary_switches", "involuntary_switches"]}
              for group in ["little", "gold", "other"]}
    common = {tid for tid in before.keys() & after.keys()
              if before[tid]["start"] == after[tid]["start"]}
    for tid in common:
        group = after[tid]["group"] if before[tid]["group"] == after[tid]["group"] else "other"
        for key in totals[group]:
            totals[group][key] += after[tid][key] - before[tid][key]
    return {"groups": totals, "unmatched_threads": len(before.keys() ^ after.keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--baseline-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--variant-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path.home()/".local/share/kukuryku/kokoro-onyx")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--cool-c", type=float, default=50)
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--thread-stats", action="store_true",
                        help="also record persistent-thread CPU and context switches by affinity")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    # Keep observation work off the tested little core and gold workers.
    os.sched_setaffinity(0, {0})
    overrides = {"A": {}, "B": {}}
    for label, settings in [("A", args.baseline_env), ("B", args.variant_env)]:
        for setting in settings:
            key, sep, value = setting.partition("=")
            if not sep or not key:
                parser.error("environment overrides expect KEY=VALUE")
            overrides[label][key] = value
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text = args.text_file.read_text() if args.text_file else TEXT
    (out/"corpus.txt").write_text(text+"\n")
    fakebin = out/"discard-sink"
    fakebin.mkdir()
    fakeplay = fakebin/"ffplay"
    fakeplay.write_text(f"#!{shutil.which('sh')}\ncat >/dev/null\n")
    fakeplay.chmod(0o755)
    binaries = {"A": args.baseline.resolve(), "B": args.variant.resolve()}
    manifest = {
        "binaries": {k:{"path":str(p),"sha256":sha256(p)} for k,p in binaries.items()},
        "baseline_environment":overrides["A"], "variant_environment":overrides["B"],
        "sink":"discard; CPU comparison only",
        "stage1_cpuset":"1", "stage2_cpuset":"4-6", "governor":"off",
        "warmup":"full corpus once per daemon", "rounds":args.rounds, "runs":[],
        "thread_stats": args.thread_stats,
    }
    path = out/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    zones=[]
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name=(z/"type").read_text().strip()
            if name.startswith("cpu-1-") and name.endswith("-usr"):
                zones.append(z/"temp")
        except OSError:
            pass
    if not zones:
        raise RuntimeError("No live gold-cluster sensors")

    def gold():
        return max(int(p.read_text()) for p in zones)/1000

    processes, sockets, logs = {}, {}, {}
    with isolate_supervisor(args.supervisor_pid,out):
        try:
            for label,binary in binaries.items():
                env=os.environ.copy()
                for name in list(env):
                    if name.startswith(("KOKORO_","RYK_","TRACT_CPU_")):
                        del env[name]
                sock=Path(tempfile.gettempdir())/f"ryk-mm-{os.getpid()}-{label}.sock"
                logpath=out/(label+"-daemon.log")
                env.update({
                    "PATH":str(fakebin)+os.pathsep+env["PATH"],
                    "KOKORO_TRACT_DIR":str(args.assets.resolve()),"KOKORO_VERBOSE":"1",
                    "KOKORO_TRACT_FP16":"1","KOKORO_GOVERNOR":"off",
                    "KOKORO_TRACT_S1_CPUSET":"1","KOKORO_TRACT_CPUSET":"4-6",
                    "KOKORO_TRACT_THREADS":"3","RYK_SOCKET":str(sock),
                    "RYK_IDLE_TIMEOUT":"off","RYK_SINK_IDLE_MS":"200",
                })
                env.update(overrides[label])
                with logpath.open("wb") as log:
                    proc=subprocess.Popen([str(binary),"--serve"],env=env,stdin=subprocess.DEVNULL,
                                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[label],sockets[label],logs[label]=proc,sock,logpath
                deadline=time.monotonic()+120
                while not sock.exists():
                    if proc.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError(f"Daemon failed to start: {logpath}")
                    time.sleep(.25)
                print(label,"pipeline ready; warming full corpus",flush=True)
                send(sock,text)
                wait_idle(proc,logpath,0,180)
            for rnd in range(1,args.rounds+1):
                order="AB" if rnd%2 else "BA"
                for label in order:
                    deadline=time.monotonic()+240
                    while gold()>args.cool_c:
                        if time.monotonic()>deadline:
                            raise RuntimeError("Cooldown timed out")
                        print("cooling",round(gold(),1),"C",flush=True)
                        time.sleep(5)
                    proc=processes[label];log=logs[label]
                    offset=len(log.read_text())
                    threads_before = thread_stats(proc.pid) if args.thread_stats else None
                    before=proc_cpu(proc.pid); started=time.monotonic(); start_c=gold()
                    send(sockets[label],text)
                    tail,_=wait_idle(proc,log,offset,180)
                    cpu=proc_cpu(proc.pid)-before; elapsed=time.monotonic()-started
                    matches=list(DONE.finditer(tail))
                    if len(matches)!=1 or "synth error" in tail or "write failed" in tail:
                        raise RuntimeError(f"Failed request: {label} round {rnd}")
                    audio,infer,rtf,wall,wall_rtf,gap=map(float,matches[0].groups())
                    result={"label":label,"round":rnd,"audio_s":audio,"cpu_s":cpu,
                            "cpu_s_per_audio_s":cpu/audio,"infer_s":infer,"infer_rtf":rtf,
                            "synthesis_wall_s":wall,"elapsed_s":elapsed,"estimated_gap_s":gap,
                            "start_gold_c":start_c,"end_gold_c":gold()}
                    if args.thread_stats:
                        result["persistent_threads"] = thread_delta(threads_before, thread_stats(proc.pid))
                    (out/f"{label}-{rnd}.log").write_text(tail)
                    manifest["runs"].append(result)
                    path.write_text(json.dumps(manifest,indent=2)+"\n")
                    print(json.dumps(result),flush=True)
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            for sock in sockets.values():
                sock.unlink(missing_ok=True)
    print("COMPLETE",flush=True)


if __name__=="__main__":
    main()
