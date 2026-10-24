//! Starting the headset helper. `RykKeys` is a small Java program that owns a
//! `MediaSession` as the Termux uid, receives the headset's media keys and
//! turns them into the control verbs of [`crate::control`]. This module builds
//! it (on first use, from the source embedded in the binary), launches it, and
//! runs its self-check; the daemon calls [`Keys::ensure`] when an utterance
//! starts. See `docs/headset-control.md`.
//!
//! The helper connects back to the daemon socket with a `watch` request and
//! exits when the daemon closes it (end of the rewind window, daemon exit or
//! crash), so nothing here has to stop it. Pure std; off the phone, or with
//! `RYK_HEADSET=0`, it does nothing.

use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Android's launcher for Java programs outside an app. Its absence means we
/// are not on Android, and there is nothing to start.
const APP_PROCESS: &str = "/system/bin/app_process";

/// The helper's main class, in `keys.jar`.
const HELPER_CLASS: &str = "RykKeys";

/// The helper's source, compiled on first use (and again when this changes).
const SOURCE: &str = include_str!("../tools/rykkeys/RykKeys.java");

/// A helper that crashes this many times in a row, each within [`QUICK`] of its
/// start, is not tried again (a hidden API that changed would otherwise cost a
/// JVM start on every utterance).
const MAX_QUICK_FAILURES: u32 = 3;
const QUICK: Duration = Duration::from_secs(10);

// -------------------------------- build --------------------------------------

/// A command and its leading arguments (`["ecj"]`, or `["sh", "fake.sh"]` in tests).
pub type Tool = Vec<OsString>;

/// Compiles the helper's source into `keys.jar` the way `td install` builds
/// `TdAlarm`: `ecj -nowarn` then `dx --dex`. The Termux `ecj` wrapper already
/// has `android.jar`, so no SDK is needed.
#[derive(Clone, Debug, PartialEq)]
pub struct Builder {
    /// `~/.cache/ryk`: holds `keys.jar`, the `keys.src` stamp and `keys-build.log`.
    pub dir: PathBuf,
    pub source: &'static str,
    pub ecj: Tool,
    pub dx: Tool,
}

impl Builder {
    pub fn new(dir: PathBuf) -> Self {
        Builder { dir, source: SOURCE, ecj: vec!["ecj".into()], dx: vec!["dx".into()] }
    }

    pub fn jar(&self) -> PathBuf {
        self.dir.join("keys.jar")
    }

    /// No jar yet, or it was built from different source (a newer `ryk`).
    pub fn stale(&self) -> bool {
        !self.jar().exists()
            || std::fs::read_to_string(self.dir.join("keys.src")).map_or(true, |s| s != self.source)
    }

    /// Build the jar. It appears atomically (a rename), so a helper starting at
    /// the same moment never sees half of one. The compiler's complaints go to
    /// `keys-build.log` as well as into the error.
    pub fn build(&self) -> Result<(), String> {
        let r = self.build_inner();
        if let Err(e) = &r {
            let _ = std::fs::create_dir_all(&self.dir);
            let _ = std::fs::write(self.dir.join("keys-build.log"), e);
        }
        r
    }

    fn build_inner(&self) -> Result<(), String> {
        let io = |what: &str| {
            let what = what.to_string();
            move |e: io::Error| format!("{what}: {e}")
        };
        let work = self.dir.join("build");
        let classes = work.join("classes");
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&classes).map_err(io("creating the build directory"))?;
        let java = work.join(format!("{HELPER_CLASS}.java"));
        std::fs::write(&java, self.source).map_err(io("writing the source"))?;

        run_tool(&self.ecj, &["-nowarn".into(), "-d".into(), classes.clone().into(), java.into()])?;
        // dx insists on a .jar/.zip/.dex extension
        let tmp = self.dir.join(format!("keys.{}.tmp.jar", std::process::id()));
        let mut out = OsString::from("--output=");
        out.push(&tmp);
        if let Err(e) = run_tool(&self.dx, &["--dex".into(), out, classes.into()]) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, self.jar()).map_err(io("installing keys.jar"))?;
        std::fs::write(self.dir.join("keys.src"), self.source).map_err(io("writing the stamp"))?;
        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }
}

fn run_tool(tool: &Tool, args: &[OsString]) -> Result<(), String> {
    let name = tool[0].to_string_lossy().into_owned();
    let out = Command::new(&tool[0])
        .args(&tool[1..])
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                format!("`{name}` not found (pkg install ecj dx)")
            } else {
                format!("could not run `{name}`: {e}")
            }
        })?;
    if out.status.success() {
        return Ok(());
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Err(format!("`{name}` failed ({}):\n{}", out.status, text.trim_end()))
}

// -------------------------------- launch -------------------------------------

/// How to start the helper.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub env_remove: Vec<&'static str>,
    /// A file that must exist first (the jar), else the helper is not started.
    pub needs: Option<PathBuf>,
    /// How to make `needs` when it is missing or out of date.
    pub build: Option<Builder>,
    /// Where the helper's output goes.
    pub log: Option<PathBuf>,
}

/// `$XDG_CACHE_HOME/ryk` or `$HOME/.cache/ryk`: the same place `td` keeps its jar.
fn cache_dir(get: &impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let base = match get("XDG_CACHE_HOME").filter(|s| !s.is_empty()) {
        Some(x) => PathBuf::from(x),
        None => Path::new(&get("HOME").filter(|s| !s.is_empty())?).join(".cache"),
    };
    Some(base.join("ryk"))
}

impl Launch {
    /// Work out the launch command from the environment (`get`), or say why the
    /// helper is off.
    ///
    /// - `RYK_HEADSET=0|off|no|false` turns it off.
    /// - `RYK_KEYS_CMD` replaces the whole command with `PROGRAM SOCKET` (for
    ///   tests and development).
    /// - `RYK_KEYS_JAR` uses that jar as it is; without it the jar is
    ///   `~/.cache/ryk/keys.jar`, built from the embedded source when missing
    ///   or out of date.
    pub fn resolve(
        socket: &Path,
        get: impl Fn(&str) -> Option<String>,
        app_process: &Path,
    ) -> Result<Launch, String> {
        Self::resolve_with(socket.into(), true, get, app_process)
    }

    /// The launch for `RykKeys check`; the off switch does not apply to a diagnosis.
    pub fn for_check(
        get: impl Fn(&str) -> Option<String>,
        app_process: &Path,
    ) -> Result<Launch, String> {
        Self::resolve_with("check".into(), false, get, app_process)
    }

    fn resolve_with(
        last_arg: OsString,
        honour_off: bool,
        get: impl Fn(&str) -> Option<String>,
        app_process: &Path,
    ) -> Result<Launch, String> {
        if honour_off
            && get("RYK_HEADSET").is_some_and(|v| matches!(v.as_str(), "0" | "off" | "no" | "false"))
        {
            return Err("RYK_HEADSET is off".into());
        }
        let cache = cache_dir(&get);
        let log = cache.as_ref().map(|c| c.join("keys.log"));
        if let Some(cmd) = get("RYK_KEYS_CMD").filter(|s| !s.is_empty()) {
            return Ok(Launch {
                program: cmd.into(),
                args: vec![last_arg],
                env: Vec::new(),
                env_remove: Vec::new(),
                needs: None,
                build: None,
                log,
            });
        }
        if !app_process.exists() {
            return Err(format!("{} not found (not Android)", app_process.display()));
        }
        let explicit = get("RYK_KEYS_JAR").filter(|s| !s.is_empty()).map(PathBuf::from);
        let build = match (&explicit, &cache) {
            (None, Some(dir)) => Some(Builder::new(dir.clone())),
            _ => None,
        };
        let jar = explicit
            .or_else(|| build.as_ref().map(Builder::jar))
            .ok_or_else(|| "no cache directory ($HOME is not set)".to_string())?;
        Ok(Launch {
            program: app_process.into(),
            args: vec!["-Xnoimage-dex2oat".into(), "/".into(), HELPER_CLASS.into(), last_arg],
            env: vec![("CLASSPATH".into(), jar.clone().into())],
            // termux-exec's preload does not belong in a system process
            env_remove: vec!["LD_PRELOAD", "LD_LIBRARY_PATH"],
            needs: Some(jar),
            build,
            log,
        })
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args).stdin(Stdio::null());
        for k in &self.env_remove {
            cmd.env_remove(k);
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd
    }

    fn spawn(&self) -> io::Result<Child> {
        let mut cmd = self.command();
        let log = self.log.as_ref().and_then(|p| {
            std::fs::create_dir_all(p.parent()?).ok()?;
            let f = File::create(p).ok()?;
            Some((f.try_clone().ok()?, f))
        });
        match log {
            Some((out, err)) => cmd.stdout(out).stderr(err),
            None => cmd.stdout(Stdio::null()).stderr(Stdio::null()),
        };
        cmd.spawn()
    }
}

// -------------------------------- check --------------------------------------

/// `ryk --headset-check`: build the helper if needed and run `RykKeys check`,
/// which tries the hidden-API steps and prints one line each. Returns the exit
/// status for `main`: 0 when headset control can work on this phone.
pub fn check() -> i32 {
    check_with(|k| std::env::var(k).ok(), Path::new(APP_PROCESS))
}

fn check_with(get: impl Fn(&str) -> Option<String>, app_process: &Path) -> i32 {
    let launch = match Launch::for_check(get, app_process) {
        Ok(l) => l,
        Err(why) => {
            eprintln!("headset check: {why}");
            return 1;
        }
    };
    if let Some(b) = &launch.build {
        if b.stale() {
            eprintln!("building the helper into {} ...", b.jar().display());
            if let Err(e) = b.build() {
                eprintln!("headset check: {e}");
                return 1;
            }
        }
    }
    if let Some(jar) = &launch.needs {
        if !jar.exists() {
            eprintln!("headset check: {} does not exist", jar.display());
            return 1;
        }
    }
    match launch.command().status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("headset check: could not start {}: {e}", launch.program.display());
            1
        }
    }
}

// -------------------------------- keys ---------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum Building {
    #[default]
    No,
    Yes,
    Failed,
}

#[derive(Default)]
struct State {
    child: Option<Child>,
    started: Option<Instant>,
    /// Crashes in a row, each soon after the start.
    failures: u32,
    spawns: u32,
    off: bool,
    warned_missing: bool,
    building: Building,
}

/// Keeps one helper running while the daemon is in use.
pub struct Keys {
    launch: Result<Launch, String>,
    state: Arc<Mutex<State>>,
}

fn lock(m: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Keys {
    pub fn new(launch: Result<Launch, String>) -> Self {
        Keys { launch, state: Arc::new(Mutex::new(State::default())) }
    }

    /// From the process environment; `socket` is the daemon's socket path.
    pub fn from_env(socket: &Path) -> Self {
        Self::new(Launch::resolve(socket, |k| std::env::var(k).ok(), Path::new(APP_PROCESS)))
    }

    /// One line for the daemon's startup log.
    pub fn describe(&self) -> String {
        match &self.launch {
            Ok(l) => match (&l.needs, &l.build) {
                (Some(jar), Some(b)) if b.stale() => {
                    format!("headset control on; building {} (needs ecj and dx)", jar.display())
                }
                (Some(jar), None) if !jar.exists() => {
                    format!("headset control off until {} exists", jar.display())
                }
                (Some(jar), _) => format!("headset control on ({})", jar.display()),
                (None, _) => format!("headset control on ({})", l.program.display()),
            },
            Err(why) => format!("headset control off: {why}"),
        }
    }

    /// Start building the jar in the background when it is missing or out of
    /// date, so it is ready by the time the first utterance needs it. Call once
    /// at daemon start; [`ensure`](Self::ensure) also calls it, which only
    /// matters if the jar was deleted while the daemon ran.
    pub fn prepare(&self) {
        let Ok(launch) = &self.launch else { return };
        let Some(builder) = launch.build.clone() else { return };
        {
            let mut st = lock(&self.state);
            if st.building != Building::No || st.off || !builder.stale() {
                return;
            }
            st.building = Building::Yes;
        }
        let state = Arc::clone(&self.state);
        let spawned = std::thread::Builder::new().name("ryk-keys-build".into()).spawn(move || {
            crate::info!("[ryk-serve] building the headset helper ...");
            let t = Instant::now();
            let r = builder.build();
            let mut st = lock(&state);
            match r {
                Ok(()) => {
                    st.building = Building::No;
                    crate::info!("[ryk-serve] headset helper built in {:.1}s", t.elapsed().as_secs_f64());
                }
                Err(e) => {
                    st.building = Building::Failed;
                    eprintln!("[ryk-serve] headset control is off: could not build the helper: {e}");
                }
            }
        });
        if spawned.is_err() {
            lock(&self.state).building = Building::Failed;
        }
    }

    /// Start the helper unless one is already running. Cheap when it is; call
    /// it whenever an utterance starts.
    pub fn ensure(&self) {
        let Ok(launch) = &self.launch else { return };
        self.prepare();
        let mut st = lock(&self.state);
        if st.off || st.building != Building::No {
            return;
        }
        if let Some(mut child) = st.child.take() {
            match child.try_wait() {
                Ok(None) => {
                    st.child = Some(child);
                    return;
                }
                Ok(Some(status)) => {
                    let soon = st.started.is_some_and(|t| t.elapsed() < QUICK);
                    // A clean exit is the daemon hanging up on it, which is normal.
                    st.failures = if status.success() || !soon { 0 } else { st.failures + 1 };
                    if !status.success() {
                        eprintln!("[ryk-serve] headset helper exited with {status}");
                    }
                }
                Err(_) => {}
            }
        }
        if st.failures >= MAX_QUICK_FAILURES {
            st.off = true;
            let log = launch.log.as_ref().map(|p| format!(" (see {})", p.display())).unwrap_or_default();
            eprintln!("[ryk-serve] headset helper keeps crashing; headset control stays off{log}");
            return;
        }
        if let Some(need) = &launch.needs {
            if !need.exists() {
                if !st.warned_missing {
                    st.warned_missing = true;
                    eprintln!("[ryk-serve] headset control is off: {} does not exist", need.display());
                }
                return;
            }
        }
        st.spawns += 1;
        st.started = Some(Instant::now());
        match launch.spawn() {
            Ok(child) => st.child = Some(child),
            Err(e) => {
                st.failures += 1;
                eprintln!("[ryk-serve] could not start the headset helper: {e}");
            }
        }
    }

    /// The helper's pid while it is running.
    pub fn pid(&self) -> Option<u32> {
        let mut st = lock(&self.state);
        let alive = st.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)));
        if alive { st.child.as_ref().map(|c| c.id()) } else { None }
    }

    /// How many times the helper was started.
    pub fn spawn_count(&self) -> u32 {
        lock(&self.state).spawns
    }

    /// Given up after repeated crashes.
    pub fn gave_up(&self) -> bool {
        lock(&self.state).off
    }

    /// The jar is being built right now.
    pub fn building(&self) -> bool {
        lock(&self.state).building == Building::Yes
    }

    /// The jar could not be built.
    pub fn build_failed(&self) -> bool {
        lock(&self.state).building == Building::Failed
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        let mut st = lock(&self.state);
        if let Some(mut child) = st.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// -------------------------------- tests --------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| m.get(k).cloned()
    }

    fn sock() -> PathBuf {
        PathBuf::from("/run/ryk.sock")
    }

    /// A real file standing in for /system/bin/app_process.
    fn here() -> PathBuf {
        PathBuf::from(file!())
    }

    fn cmd(program: &str, args: &[&str]) -> Launch {
        Launch {
            program: program.into(),
            args: args.iter().map(OsString::from).collect(),
            env: Vec::new(),
            env_remove: Vec::new(),
            needs: None,
            build: None,
            log: None,
        }
    }

    fn settle() {
        std::thread::sleep(Duration::from_millis(150));
    }

    /// A fresh empty directory, unique per call (tests run in parallel).
    fn temp(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ryk-headset-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn wait_for(mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Fake `ecj` and `dx`: shell scripts run through `sh`, so no freshly
    /// written executable is ever exec'd (which can fail with ETXTBSY when
    /// tests run in parallel).
    fn fake_tools(dir: &Path, ecj_body: &str, dx_body: &str) -> (Tool, Tool) {
        let ecj = dir.join("ecj.sh");
        let dx = dir.join("dx.sh");
        std::fs::write(&ecj, ecj_body).unwrap();
        std::fs::write(&dx, dx_body).unwrap();
        (vec!["sh".into(), ecj.into()], vec!["sh".into(), dx.into()])
    }

    // ecj -nowarn -d CLASSES SRC
    const ECJ_OK: &str = "mkdir -p \"$3\"; : > \"$3/RykKeys.class\"\n";
    // dx --dex --output=PATH CLASSES
    const DX_OK: &str = "out=\"${2#--output=}\"; echo jar > \"$out\"\n";

    fn builder(dir: &Path, ecj: &str, dx: &str) -> Builder {
        let tools = temp("tools");
        let (ecj, dx) = fake_tools(&tools, ecj, dx);
        Builder { dir: dir.to_path_buf(), source: "class RykKeys {}", ecj, dx }
    }

    #[test]
    fn the_off_switch_wins() {
        for v in ["0", "off", "no", "false"] {
            let r = Launch::resolve(&sock(), env(&[("RYK_HEADSET", v), ("HOME", "/h")]), &here());
            assert!(r.is_err(), "{v}");
        }
        let r = Launch::resolve(&sock(), env(&[("RYK_HEADSET", "1"), ("HOME", "/h")]), &here());
        assert!(r.is_ok());
    }

    #[test]
    fn default_launch_is_app_process_with_the_cached_jar() {
        let l = Launch::resolve(&sock(), env(&[("HOME", "/home/u")]), &here()).unwrap();
        assert_eq!(l.program, here());
        let args: Vec<_> = l.args.iter().map(|a| a.to_str().unwrap()).collect();
        assert_eq!(args, ["-Xnoimage-dex2oat", "/", "RykKeys", "/run/ryk.sock"]);
        assert_eq!(l.needs, Some(PathBuf::from("/home/u/.cache/ryk/keys.jar")));
        assert_eq!(l.env, [(OsString::from("CLASSPATH"), OsString::from("/home/u/.cache/ryk/keys.jar"))]);
        assert!(l.env_remove.contains(&"LD_PRELOAD") && l.env_remove.contains(&"LD_LIBRARY_PATH"));
        assert_eq!(l.log, Some(PathBuf::from("/home/u/.cache/ryk/keys.log")));
        let b = l.build.expect("the default jar is built from the embedded source");
        assert_eq!(b.dir, PathBuf::from("/home/u/.cache/ryk"));
        assert_eq!(b.jar(), PathBuf::from("/home/u/.cache/ryk/keys.jar"));
        assert!(b.source.contains("class RykKeys"));
    }

    #[test]
    fn xdg_cache_and_an_explicit_jar_are_honoured() {
        let l = Launch::resolve(&sock(), env(&[("XDG_CACHE_HOME", "/c"), ("HOME", "/h")]), &here()).unwrap();
        assert_eq!(l.needs, Some(PathBuf::from("/c/ryk/keys.jar")));
        let l = Launch::resolve(&sock(), env(&[("RYK_KEYS_JAR", "/x/k.jar"), ("HOME", "/h")]), &here()).unwrap();
        assert_eq!(l.needs, Some(PathBuf::from("/x/k.jar")));
        assert_eq!(l.build, None, "an explicit jar is the user's, never rebuilt");
        assert!(Launch::resolve(&sock(), env(&[]), &here()).is_err(), "no HOME, no jar");
    }

    #[test]
    fn off_android_there_is_nothing_to_start() {
        let r = Launch::resolve(&sock(), env(&[("HOME", "/h")]), Path::new("/nonexistent/app_process"));
        assert!(r.unwrap_err().contains("not Android"));
    }

    #[test]
    fn a_command_override_gets_the_socket_and_skips_the_jar() {
        let l = Launch::resolve(&sock(), env(&[("RYK_KEYS_CMD", "/bin/fake"), ("HOME", "/h")]), Path::new("/no"))
            .unwrap();
        assert_eq!(l.program, PathBuf::from("/bin/fake"));
        assert_eq!(l.args, [OsString::from("/run/ryk.sock")]);
        assert_eq!((l.needs, l.build), (None, None));
    }

    #[test]
    fn the_check_launch_ends_in_check_and_ignores_the_off_switch() {
        let l = Launch::for_check(env(&[("RYK_HEADSET", "0"), ("HOME", "/h")]), &here()).unwrap();
        assert_eq!(l.args.last().unwrap(), "check");
        assert!(l.build.is_some());
        let l = Launch::for_check(env(&[("RYK_KEYS_CMD", "/bin/fake"), ("HOME", "/h")]), Path::new("/no")).unwrap();
        assert_eq!(l.args, [OsString::from("check")]);
        assert!(Launch::for_check(env(&[("HOME", "/h")]), Path::new("/nonexistent")).is_err());
    }

    // ---- build ----

    #[test]
    fn a_build_makes_the_jar_and_the_stamp_and_leaves_no_litter() {
        let dir = temp("build-ok");
        let b = builder(&dir, ECJ_OK, DX_OK);
        assert!(b.stale());
        b.build().unwrap();
        assert!(!b.stale());
        assert!(b.jar().exists());
        assert_eq!(std::fs::read_to_string(dir.join("keys.src")).unwrap(), "class RykKeys {}");
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert!(!names.iter().any(|n| n.to_string_lossy().contains("tmp") || n == "build"), "{names:?}");
    }

    #[test]
    fn changed_source_makes_the_jar_stale() {
        let dir = temp("build-stale");
        let mut b = builder(&dir, ECJ_OK, DX_OK);
        b.build().unwrap();
        b.source = "class RykKeys { int newer; }";
        assert!(b.stale());
        b.build().unwrap();
        assert!(!b.stale());
    }

    #[test]
    fn compiler_errors_are_reported_and_logged_and_leave_no_jar() {
        let dir = temp("build-ecj-fails");
        let b = builder(&dir, "echo 'RykKeys.java:3: error: boom' >&2; exit 1\n", DX_OK);
        let e = b.build().unwrap_err();
        assert!(e.contains("boom") && e.contains("sh"), "{e}");
        assert!(!b.jar().exists());
        assert!(std::fs::read_to_string(dir.join("keys-build.log")).unwrap().contains("boom"));
    }

    #[test]
    fn a_failing_dx_leaves_no_jar_and_no_temp_file() {
        let dir = temp("build-dx-fails");
        let b = builder(&dir, ECJ_OK, "out=\"${2#--output=}\"; echo half > \"$out\"; exit 1\n");
        assert!(b.build().is_err());
        assert!(!b.jar().exists());
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert!(!names.iter().any(|n| n.to_string_lossy().contains("tmp")), "{names:?}");
    }

    #[test]
    fn missing_tools_say_how_to_get_them() {
        let dir = temp("build-no-tools");
        let mut b = builder(&dir, ECJ_OK, DX_OK);
        b.ecj = vec!["/nonexistent/ecj".into()];
        let e = b.build().unwrap_err();
        assert!(e.contains("not found") && e.contains("pkg install"), "{e}");
    }

    #[test]
    fn prepare_builds_in_the_background_and_ensure_waits_for_it() {
        let dir = temp("prepare");
        let b = builder(&dir, "sleep 0.4; mkdir -p \"$3\"\n", DX_OK);
        let mut l = cmd("sleep", &["30"]);
        l.needs = Some(b.jar());
        l.build = Some(b);
        let k = Keys::new(Ok(l));
        assert!(k.describe().contains("building"));
        k.prepare();
        assert!(k.building());
        k.ensure();
        assert_eq!(k.spawn_count(), 0, "no helper while the jar is still being built");
        wait_for(|| !k.building());
        assert!(!k.build_failed());
        k.ensure();
        assert_eq!(k.spawn_count(), 1);
        assert!(k.pid().is_some());
        assert!(k.describe().starts_with("headset control on ("));
    }

    #[test]
    fn a_failed_build_keeps_headset_control_off_without_retrying() {
        let dir = temp("prepare-fails");
        let b = builder(&dir, "exit 1\n", DX_OK);
        let mut l = cmd("sleep", &["30"]);
        l.needs = Some(b.jar());
        l.build = Some(b);
        let k = Keys::new(Ok(l));
        k.prepare();
        wait_for(|| k.build_failed());
        for _ in 0..3 {
            k.ensure();
        }
        assert_eq!(k.spawn_count(), 0);
        assert!(k.build_failed());
    }

    #[test]
    fn an_up_to_date_jar_is_not_rebuilt() {
        let dir = temp("prepare-fresh");
        let b = builder(&dir, ECJ_OK, DX_OK);
        b.build().unwrap();
        let before = std::fs::metadata(b.jar()).unwrap().modified().unwrap();
        let mut l = cmd("sleep", &["30"]);
        l.needs = Some(b.jar());
        l.build = Some(b.clone());
        let k = Keys::new(Ok(l));
        k.prepare();
        assert!(!k.building());
        assert_eq!(std::fs::metadata(b.jar()).unwrap().modified().unwrap(), before);
    }

    /// The real thing: embedded source, real ecj and dx. Needs Termux's tools,
    /// so it runs only on request (`cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn the_embedded_source_compiles_with_the_real_tools() {
        let dir = temp("real-build");
        let b = Builder::new(dir.clone());
        b.build().unwrap_or_else(|e| panic!("{e}"));
        assert!(b.jar().metadata().unwrap().len() > 1000);
        assert!(!b.stale());
    }

    // ---- check ----

    #[test]
    fn check_fails_cleanly_off_android() {
        let code = check_with(env(&[("HOME", "/h")]), Path::new("/nonexistent/app_process"));
        assert_eq!(code, 1);
    }

    #[test]
    fn check_passes_on_the_helpers_exit_status() {
        for (prog, want) in [("true", 0), ("false", 1)] {
            let code = check_with(env(&[("RYK_KEYS_CMD", prog), ("HOME", "/h")]), Path::new("/no"));
            assert_eq!(code, want, "{prog}");
        }
        let code = check_with(env(&[("RYK_KEYS_CMD", "/nonexistent/x"), ("HOME", "/h")]), Path::new("/no"));
        assert_eq!(code, 1);
    }

    // ---- ensure ----

    #[test]
    fn ensure_starts_one_helper_and_leaves_it_alone() {
        let k = Keys::new(Ok(cmd("sleep", &["30"])));
        assert_eq!(k.pid(), None);
        k.ensure();
        let pid = k.pid().expect("started");
        k.ensure();
        k.ensure();
        assert_eq!(k.pid(), Some(pid));
        assert_eq!(k.spawn_count(), 1);
    }

    #[test]
    fn a_helper_that_exited_cleanly_is_started_again() {
        let k = Keys::new(Ok(cmd("true", &[])));
        for expected in 1..=5 {
            k.ensure();
            settle();
            assert_eq!(k.spawn_count(), expected);
        }
        assert!(!k.gave_up(), "clean exits are the daemon hanging up, not crashes");
    }

    #[test]
    fn a_helper_that_keeps_crashing_is_given_up_on() {
        let k = Keys::new(Ok(cmd("false", &[])));
        for _ in 0..8 {
            k.ensure();
            settle();
        }
        assert!(k.gave_up());
        assert_eq!(k.spawn_count(), MAX_QUICK_FAILURES);
    }

    #[test]
    fn a_program_that_cannot_start_counts_as_a_crash() {
        let k = Keys::new(Ok(cmd("/nonexistent/helper", &[])));
        for _ in 0..8 {
            k.ensure();
        }
        assert!(k.gave_up());
        assert_eq!(k.spawn_count(), MAX_QUICK_FAILURES);
    }

    #[test]
    fn a_missing_jar_means_no_helper_but_it_is_picked_up_later() {
        let dir = temp("missing-jar");
        let jar = dir.join("keys.jar");
        let mut l = cmd("sleep", &["30"]);
        l.needs = Some(jar.clone());
        let k = Keys::new(Ok(l));
        k.ensure();
        k.ensure();
        assert_eq!(k.spawn_count(), 0);
        std::fs::write(&jar, b"jar").unwrap();
        k.ensure();
        assert_eq!(k.spawn_count(), 1);
    }

    #[test]
    fn a_disabled_launch_never_spawns() {
        let k = Keys::new(Err("off".into()));
        k.ensure();
        k.prepare();
        assert_eq!(k.spawn_count(), 0);
        assert_eq!(k.describe(), "headset control off: off");
    }

    #[test]
    fn dropping_keys_stops_the_helper() {
        let k = Keys::new(Ok(cmd("sleep", &["30"])));
        k.ensure();
        let proc_dir = PathBuf::from(format!("/proc/{}", k.pid().unwrap()));
        assert!(proc_dir.exists());
        drop(k);
        assert!(!proc_dir.exists(), "the helper is gone");
    }

    #[test]
    fn the_helpers_output_goes_to_the_log() {
        let dir = temp("log");
        let log = dir.join("sub").join("keys.log");
        let mut l = cmd("sh", &["-c", "echo started; echo oops >&2"]);
        l.log = Some(log.clone());
        let k = Keys::new(Ok(l));
        k.ensure();
        settle();
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("started") && text.contains("oops"), "{text:?}");
    }
}
