#!/data/data/com.termux/files/usr/bin/python3
"""Check the headset chain on the phone without a headset and without ryk:

    media key  ->  Android  ->  RykKeys' MediaSession  ->  verb on the watch socket

A fake daemon stands in for `ryk --serve`. The keys are dispatched the way any
app can (tools/headset-spike MediaSpike `key`), so what is tested is the real
routing, the real helper and the real wire protocol; only the daemon is fake.

    python3 tools/rykkeys/chain-test.py [--notify]

--notify also tests RYK_NOTIFY=1 (a real notification appears on the phone for a few
seconds; needs Termux:API).

Needs ecj, dx and pulseaudio (a quiet tone plays so Termux is the app that last
played audio, as it is while ryk speaks). Takes about 45 s. The first key sent
is STOP; if the helper does not see it, the run stops there so nothing else can
reach another media app.
"""
import math
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(os.path.dirname(HERE), "headset-spike", "MediaSpike.java")
APP_PROCESS = "/system/bin/app_process"

# (key name for MediaSpike, presses, verb RykKeys must send; None: consumed, ignored).
# PLAY_PAUSE and HOOK are a wired headset's raw clicks: one click toggles (after the
# click window), two or more go back. The others arrive already resolved.
KEYS = [
    ("STOP", 1, "cancel"),
    ("PAUSE", 1, "toggle"),
    ("PLAY", 1, "toggle"),
    ("NEXT", 1, "back"),
    ("PREVIOUS", 1, "back"),
    ("PLAY_PAUSE", 1, "toggle"),
    ("HOOK", 1, "toggle"),
    ("HOOK", 2, "back"),
    ("HOOK", 3, "back"),
    ("PLAY_PAUSE", 2, "back"),
    ("FAST_FORWARD", 1, None),
]


def java_env(jar, extra=None):
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("RYK_KEYS_MAP", None)
    env.pop("RYK_KEYS_CLICK_MS", None)
    env["CLASSPATH"] = jar
    env.update(extra or {})
    return env


def app(jar, *args, env=None, **kw):
    return subprocess.Popen(
        [APP_PROCESS, "-Xnoimage-dex2oat", "/", *args], env=java_env(jar, env), **kw
    )


class Lines:
    """A process's output, collected line by line."""

    def __init__(self, proc):
        self.lines = []
        self.q = queue.Queue()
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()

    def _pump(self, proc):
        for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            self.lines.append(line)
            self.q.put(line)

    def wait_for(self, text, timeout):
        end = time.time() + timeout
        for line in list(self.lines):
            if text in line:
                return line
        while time.time() < end:
            try:
                line = self.q.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                break
            if text in line:
                return line
        return None


class FakeDaemon:
    """The daemon's side of the protocol, as far as the helper needs it."""

    def __init__(self, path):
        self.path = path
        self.srv = socket.socket(socket.AF_UNIX)
        self.srv.bind(path)
        self.srv.listen(2)
        self.verbs = queue.Queue()
        self.state = "playing"
        self.conn = None
        self.header = None
        self.connected = threading.Event()
        self.oneshots = queue.Queue()
        threading.Thread(target=self._accept, daemon=True).start()

    def _send(self, text):
        self.conn.sendall((text + "\n").encode())

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile("rb")
        header = f.readline().decode().strip()
        if header != "watch":
            # a one-shot control request, as `ryk-ctl` sends it
            self.oneshots.put(header)
            conn.sendall(b"ok playing\n")
            conn.close()
            return
        self.conn = conn
        self.header = header
        self._send("ok")
        self._send("state " + self.state)
        self.connected.set()
        for raw in f:
            verb = raw.decode().strip()
            self.verbs.put(verb)
            if verb == "toggle":
                self.state = "paused" if self.state == "playing" else "playing"
                self._send("ok " + self.state)
                self._send("state " + self.state)
            elif verb == "cancel":
                self.state = "idle"
                self._send("ok idle cancelled")
                self._send("state idle")
            else:
                self.state = "playing"
                self._send("ok playing back")
                self._send("state playing")

    def push(self, state):
        self.state = state
        self._send("state " + state)

    def hang_up(self):
        if self.conn:
            self.conn.shutdown(socket.SHUT_RDWR)


def notify_scenario(tmp, keys_jar, check):
    """RYK_NOTIFY=1: a notification follows the state and is removed when the helper exits.

    Only the exit status of termux-notification / termux-notification-remove is checked
    (termux-notification-list needs a permission and hangs without it), so look at the
    phone to see it.
    """
    ctl = os.path.expanduser("~/PRJ/ryk-skills/bin/ryk-ctl")
    d = FakeDaemon(os.path.join(tmp, "notify.sock"))
    helper = app(keys_jar, "RykKeys", d.path, env={"RYK_NOTIFY": "1", "RYK_CTL": ctl},
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = Lines(helper)
    check(d.connected.wait(20), "connects")
    check(log.wait_for("notification playing: ok", 40) is not None, "posts the notification (playing)")
    d.push("paused")
    check(log.wait_for("notification paused: ok", 40) is not None, "updates it (paused)")
    # a button's action, run the way Termux:API runs it: a bare environment
    line = log.wait_for("buttons run: ", 1)
    check(line is not None, "logs what the buttons run")
    if line:
        action = line.split("buttons run: ", 1)[1]
        env = {"PATH": "/data/data/com.termux/files/usr/bin", "HOME": os.environ["HOME"]}
        r = subprocess.run(["sh", "-c", action], env=env, capture_output=True, text=True, timeout=30)
        try:
            got = d.oneshots.get(timeout=3)
        except queue.Empty:
            got = None
        check(r.returncode == 0 and got == "back", f"the Back button's command reaches the daemon ({got!r}, exit {r.returncode})")
    d.hang_up()
    try:
        helper.wait(timeout=30)
        check(helper.returncode == 0, "exits 0 when the daemon hangs up")
    except subprocess.TimeoutExpired:
        helper.kill()
        check(False, "exits when the daemon hangs up")
    check(log.wait_for("notification removed: ok", 1) is not None, "removes it on the way out")


def build(tmp):
    """The helper jar (via the real build script) and the key-dispatching spike."""
    jar = os.path.join(tmp, "keys.jar")
    subprocess.run(["sh", os.path.join(HERE, "build.sh"), tmp], check=True, stdout=subprocess.DEVNULL)
    classes = os.path.join(tmp, "spike-classes")
    os.makedirs(classes)
    subprocess.run(["ecj", "-nowarn", "-d", classes, SPIKE], check=True)
    spike = os.path.join(tmp, "spike.jar")
    subprocess.run(["dx", "--dex", "--output=" + spike, classes], check=True, stdout=subprocess.DEVNULL)
    return jar, spike


def start_tone(seconds):
    """A very quiet tone through pacat, so Termux is the app that last played audio."""
    subprocess.run("pulseaudio --check || pulseaudio --start --exit-idle-time=30", shell=True, check=False)
    time.sleep(1)
    period = b"".join(struct.pack("<f", 0.002 * math.sin(2 * math.pi * 240 * i / 24000)) for i in range(100))
    pacat = subprocess.Popen(
        ["pacat", "--raw", "--format=float32le", "--rate", "24000", "--channels=1", "--latency-msec=100"],
        stdin=subprocess.PIPE,
    )

    def feed():
        try:
            for _ in range(int(seconds)):
                pacat.stdin.write(period * 240)
                pacat.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    threading.Thread(target=feed, daemon=True).start()
    return pacat


def main():
    failures = []

    def check(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what)
        if not ok:
            failures.append(what)

    def scenario(tmp, keys_jar, spike_jar, name, keys, env=None):
        """One helper against one fake daemon; press each key and compare the verb."""
        d = FakeDaemon(os.path.join(tmp, name + ".sock"))
        helper = app(keys_jar, "RykKeys", d.path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = Lines(helper)
        check(d.connected.wait(20), "connects and sends `watch`")
        check(log.wait_for("session active", 20) is not None, "creates its media session")
        if failures:
            return False
        time.sleep(1)
        for i, (key, times, want) in enumerate(keys):
            subprocess.run(
                [APP_PROCESS, "-Xnoimage-dex2oat", "/", "MediaSpike", "key", key, str(times), "120"],
                env=java_env(spike_jar), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
            try:
                got = d.verbs.get(timeout=3 if want else 1.5)
            except queue.Empty:
                got = None
            label = f"{key} x{times}"
            check(got == want, f"{label:16s} -> {want or '(nothing)'}" + ("" if got == want else f"   (got {got})"))
            if i == 0 and got != want:
                print("the first key never arrived; stopping so no other media app gets the rest")
                return False
            time.sleep(0.6)
            extra = []
            while not d.verbs.empty():
                extra.append(d.verbs.get())
            check(not extra, f"{label:16s}    exactly one verb" + (f"   (also {extra})" if extra else ""))
        check(helper.poll() is None, "still running after all of that")
        d.hang_up()
        try:
            helper.wait(timeout=10)
            check(helper.returncode == 0, "exits 0 when the daemon hangs up")
        except subprocess.TimeoutExpired:
            helper.kill()
            check(False, "exits when the daemon hangs up")
        return d, log

    tmp = tempfile.mkdtemp(prefix="ryk-chain-")
    pacat = None
    try:
        print("building the helper and the key dispatcher ...")
        keys_jar, spike_jar = build(tmp)

        print("`RykKeys check` (the hidden-API steps):")
        p = app(keys_jar, "RykKeys", "check", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = p.communicate(timeout=30)
        check(p.returncode == 0 and b"headset check: OK" in out, "passes on this phone")

        print("helper with no daemon behind it:")
        p = app(keys_jar, "RykKeys", os.path.join(tmp, "nothing.sock"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = p.communicate(timeout=20)
        check(p.returncode == 0, "exits 0 (the daemon went away, not a crash)")
        check(b"no daemon" in out, "says so")

        print("helper with a (fake) daemon, default mapping:")
        pacat = start_tone(90)
        res = scenario(tmp, keys_jar, spike_jar, "default", KEYS)
        if not res:
            return report(failures)
        d, log = res
        check(log.wait_for("<- state paused", 1) is not None, "state lines are read (fake daemon paused on toggle)")

        print("helper with RYK_KEYS_MAP=tap:pause,stop:none,skip:rewind:")
        scenario(tmp, keys_jar, spike_jar, "mapped", [
            ("PAUSE", 1, "pause"),
            ("PLAY", 1, "pause"),
            ("HOOK", 1, "pause"),
            ("NEXT", 1, "rewind"),
            ("STOP", 1, None),
        ], env={"RYK_KEYS_MAP": "tap:pause,stop:none,skip:rewind"})
        if "--notify" in sys.argv:
            print("helper with RYK_NOTIFY=1 (a notification will flash on the phone):")
            notify_scenario(tmp, keys_jar, check)
        return report(failures)
    finally:
        if pacat:
            pacat.kill()
            try:
                pacat.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def report(failures):
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
