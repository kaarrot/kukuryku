//! Starting the headset helper. `RykKeys` is a small Java program that owns a
//! `MediaSession` as the Termux uid, receives the headset's media keys and
//! turns them into the control verbs of [`crate::control`]. This module only
//! launches it; the daemon calls [`Keys::ensure`] when an utterance starts. See
//! `docs/headset-control.md`.
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
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Android's launcher for Java programs outside an app. Its absence means we
/// are not on Android, and there is nothing to start.
const APP_PROCESS: &str = "/system/bin/app_process";

/// The helper's main class, in `keys.jar`.
const HELPER_CLASS: &str = "RykKeys";

/// A helper that crashes this many times in a row, each within [`QUICK`] of its
/// start, is not tried again (a hidden API that changed would otherwise cost a
/// JVM start on every utterance).
const MAX_QUICK_FAILURES: u32 = 3;
const QUICK: Duration = Duration::from_secs(10);

/// How to start the helper.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub env_remove: Vec<&'static str>,
    /// A file that must exist first (the jar), else the helper is not started.
    pub needs: Option<PathBuf>,
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
    /// - `RYK_KEYS_JAR` replaces `~/.cache/ryk/keys.jar`.
    pub fn resolve(
        socket: &Path,
        get: impl Fn(&str) -> Option<String>,
        app_process: &Path,
    ) -> Result<Launch, String> {
        if get("RYK_HEADSET").is_some_and(|v| matches!(v.as_str(), "0" | "off" | "no" | "false")) {
            return Err("RYK_HEADSET is off".into());
        }
        let cache = cache_dir(&get);
        let log = cache.as_ref().map(|c| c.join("keys.log"));
        if let Some(cmd) = get("RYK_KEYS_CMD").filter(|s| !s.is_empty()) {
            return Ok(Launch {
                program: cmd.into(),
                args: vec![socket.into()],
                env: Vec::new(),
                env_remove: Vec::new(),
                needs: None,
                log,
            });
        }
        if !app_process.exists() {
            return Err(format!("{} not found (not Android)", app_process.display()));
        }
        let jar = get("RYK_KEYS_JAR")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| cache.map(|c| c.join("keys.jar")))
            .ok_or_else(|| "no cache directory ($HOME is not set)".to_string())?;
        Ok(Launch {
            program: app_process.into(),
            args: vec!["-Xnoimage-dex2oat".into(), "/".into(), HELPER_CLASS.into(), socket.into()],
            env: vec![("CLASSPATH".into(), jar.clone().into())],
            // termux-exec's preload does not belong in a system process
            env_remove: vec!["LD_PRELOAD", "LD_LIBRARY_PATH"],
            needs: Some(jar),
            log,
        })
    }

    fn spawn(&self) -> io::Result<Child> {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args).stdin(Stdio::null());
        for k in &self.env_remove {
            cmd.env_remove(k);
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
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

#[derive(Default)]
struct State {
    child: Option<Child>,
    started: Option<Instant>,
    /// Crashes in a row, each soon after the start.
    failures: u32,
    spawns: u32,
    off: bool,
    warned_missing: bool,
}

/// Keeps one helper running while the daemon is in use.
pub struct Keys {
    launch: Result<Launch, String>,
    state: Mutex<State>,
}

impl Keys {
    pub fn new(launch: Result<Launch, String>) -> Self {
        Keys { launch, state: Mutex::new(State::default()) }
    }

    /// From the process environment; `socket` is the daemon's socket path.
    pub fn from_env(socket: &Path) -> Self {
        Self::new(Launch::resolve(socket, |k| std::env::var(k).ok(), Path::new(APP_PROCESS)))
    }

    /// One line for the daemon's startup log.
    pub fn describe(&self) -> String {
        match &self.launch {
            Ok(l) => match &l.needs {
                Some(jar) if !jar.exists() => format!(
                    "headset control off until {} exists (tools/rykkeys/build.sh builds it)",
                    jar.display()
                ),
                Some(jar) => format!("headset control on ({})", jar.display()),
                None => format!("headset control on ({})", l.program.display()),
            },
            Err(why) => format!("headset control off: {why}"),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start the helper unless one is already running. Cheap when it is; call
    /// it whenever an utterance starts.
    pub fn ensure(&self) {
        let Ok(launch) = &self.launch else { return };
        let mut st = self.lock();
        if st.off {
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
                    eprintln!(
                        "[ryk-serve] headset control is off: {} does not exist (tools/rykkeys/build.sh builds it)",
                        need.display()
                    );
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
        let mut st = self.lock();
        let alive = st.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)));
        if alive { st.child.as_ref().map(|c| c.id()) } else { None }
    }

    /// How many times the helper was started.
    pub fn spawn_count(&self) -> u32 {
        self.lock().spawns
    }

    /// Given up after repeated crashes.
    pub fn gave_up(&self) -> bool {
        self.lock().off
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
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
            log: None,
        }
    }

    fn settle() {
        std::thread::sleep(Duration::from_millis(150));
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
    }

    #[test]
    fn xdg_cache_and_an_explicit_jar_are_honoured() {
        let l = Launch::resolve(&sock(), env(&[("XDG_CACHE_HOME", "/c"), ("HOME", "/h")]), &here()).unwrap();
        assert_eq!(l.needs, Some(PathBuf::from("/c/ryk/keys.jar")));
        let l = Launch::resolve(&sock(), env(&[("RYK_KEYS_JAR", "/x/k.jar"), ("HOME", "/h")]), &here()).unwrap();
        assert_eq!(l.needs, Some(PathBuf::from("/x/k.jar")));
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
        assert_eq!(l.needs, None);
    }

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
        let dir = std::env::temp_dir().join(format!("ryk-headset-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let jar = dir.join("keys.jar");
        let _ = std::fs::remove_file(&jar);
        let mut l = cmd("sleep", &["30"]);
        l.needs = Some(jar.clone());
        let k = Keys::new(Ok(l));
        k.ensure();
        k.ensure();
        assert_eq!(k.spawn_count(), 0);
        std::fs::write(&jar, b"jar").unwrap();
        k.ensure();
        assert_eq!(k.spawn_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_disabled_launch_never_spawns() {
        let k = Keys::new(Err("off".into()));
        k.ensure();
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
        let dir = std::env::temp_dir().join(format!("ryk-headset-log-{}", std::process::id()));
        let log = dir.join("sub").join("keys.log");
        let mut l = cmd("sh", &["-c", "echo started; echo oops >&2"]);
        l.log = Some(log.clone());
        let k = Keys::new(Ok(l));
        k.ensure();
        settle();
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(text.contains("started") && text.contains("oops"), "{text:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
