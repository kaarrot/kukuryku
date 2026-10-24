//! The control protocol on the daemon socket: the one-shot verbs (`pause`,
//! `back`, ...) and the long-lived `watch` connection the headset helper keeps.
//! See `docs/headset-control.md`.
//!
//! Pure std plus [`crate::playback`], so the tests need no model and no sound
//! device.
//!
//! A `watch` connection is line-oriented in both directions:
//!
//! ```text
//! watch            (client's header line; the daemon answers)
//! ok
//! state playing    pushed first, then on every change: idle | playing | paused
//! toggle           a verb, exactly as `ryk-ctl` would send it
//! ok paused        its answer (or `err: ...`); answers and `state` lines interleave
//! ```
//!
//! The daemon closes the connection when the rewind window is over (nothing
//! playing, nothing to replay, nothing queued), which is the helper's cue to
//! exit.

use std::io::{BufRead, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};

use crate::playback::{Control, Handle, PlayState, Snapshot, parse_secs};

/// Verbs that control the running session instead of speaking. An allowlist,
/// not "any first field that is not a voice": a voice is never named like this.
pub const VERBS: [&str; 7] = ["pause", "resume", "toggle", "back", "cancel", "rewind", "status"];

/// A longest line a `watch` client may send.
const MAX_LINE: u64 = 4096;

/// A client that stops reading must not wedge the writer for long.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// Re-check interval for a connection whose window might end without a
/// notification (a job that played nothing finishes with nothing to wake us).
const SETTLE_POLL: Duration = Duration::from_secs(1);

/// Re-check interval otherwise; every change wakes the watcher anyway.
const KEEPALIVE: Duration = Duration::from_secs(30);

/// Default step for a bare `rewind`: `$RYK_REWIND_SECS` or 10 s.
pub fn default_rewind_secs() -> f32 {
    parse_secs(std::env::var("RYK_REWIND_SECS").ok().as_deref()).map_or(10.0, |v| v as f32)
}

/// A verb and its optional argument as a [`Control`]. Only `rewind` takes one.
pub fn parse_control(verb: &str, arg: Option<&str>) -> Result<Control> {
    Ok(match verb {
        "pause" => Control::Pause,
        "resume" => Control::Resume,
        "toggle" => Control::Toggle,
        "back" => Control::Back,
        "cancel" => Control::Cancel,
        "status" => Control::Status,
        "rewind" => match arg {
            None => Control::Rewind(default_rewind_secs()),
            Some(v) => match parse_secs(Some(v)) {
                Some(secs) => Control::Rewind(secs as f32),
                None => bail!("invalid rewind seconds"),
            },
        },
        _ => bail!("unknown command {verb}"),
    })
}

/// Run a control and word the answer (`ok <state> [note]` or `err: ...`).
/// `queued_jobs` is whether utterances are waiting behind the running one: a
/// cancel with nothing playing still counts then, because the epoch bump
/// already dropped them.
pub fn control_reply(ctl: &Handle, c: Control, queued_jobs: bool) -> String {
    match ctl.control(c) {
        Ok(o) if c == Control::Cancel && o.note.as_deref() == Some("nothing") => {
            if queued_jobs { "ok idle cancelled".to_string() } else { "err: nothing playing".to_string() }
        }
        Ok(o) => format!("ok {}", o.line()),
        Err(e) => format!("err: {e}"),
    }
}

// -------------------------------- watch --------------------------------------

/// Serve a `watch` connection on two detached threads (it never blocks the
/// accept loop): one pushes `state` lines, one reads verbs. `reader` is whatever
/// already wraps the read side, so bytes it buffered past the header are kept.
pub fn serve_watch<R: BufRead + Send + 'static>(
    conn: UnixStream,
    reader: R,
    ctl: Handle,
    pending: Arc<AtomicUsize>,
) {
    let _ = conn.set_write_timeout(Some(WRITE_TIMEOUT));
    let out = Arc::new(Mutex::new(conn));
    if !send(&out, "ok") {
        return;
    }
    let done = Arc::new(AtomicBool::new(false));
    let pusher = {
        let (out, done, ctl, pending) =
            (Arc::clone(&out), Arc::clone(&done), ctl.clone(), Arc::clone(&pending));
        std::thread::Builder::new()
            .name("ryk-watch-state".into())
            .spawn(move || push_states(&out, &done, &ctl, &pending))
    };
    let verbs = {
        let out = Arc::clone(&out);
        std::thread::Builder::new()
            .name("ryk-watch-verbs".into())
            .spawn(move || read_verbs(reader, &out, &done, &ctl, &pending))
    };
    if pusher.is_err() || verbs.is_err() {
        eprintln!("[ryk-serve] could not start the watch threads");
        close(&out);
    }
}

fn send(out: &Mutex<UnixStream>, line: &str) -> bool {
    let mut s = out.lock().unwrap_or_else(|e| e.into_inner());
    s.write_all(format!("{line}\n").as_bytes()).is_ok()
}

fn close(out: &Mutex<UnixStream>) {
    let _ = out.lock().unwrap_or_else(|e| e.into_inner()).shutdown(Shutdown::Both);
}

/// Nothing is playing, nothing can be replayed and nothing is queued.
fn window_over(snap: &Snapshot, pending: &AtomicUsize) -> bool {
    snap.state == PlayState::Idle && !snap.history && pending.load(Ordering::Acquire) == 0
}

fn push_states(out: &Mutex<UnixStream>, done: &AtomicBool, ctl: &Handle, pending: &AtomicUsize) {
    let mut snap = ctl.snapshot();
    let mut sent: Option<PlayState> = None;
    loop {
        if done.load(Ordering::Acquire) {
            return;
        }
        if sent != Some(snap.state) {
            if !send(out, &format!("state {}", snap.state.as_str())) {
                break;
            }
            sent = Some(snap.state);
        }
        if window_over(&snap, pending) {
            break;
        }
        let settling = snap.state == PlayState::Idle && !snap.history;
        snap = ctl.wait_change(snap.seq, if settling { SETTLE_POLL } else { KEEPALIVE });
    }
    close(out);
}

fn read_verbs<R: BufRead>(
    mut reader: R,
    out: &Mutex<UnixStream>,
    done: &AtomicBool,
    ctl: &Handle,
    pending: &AtomicUsize,
) {
    let mut line = String::new();
    loop {
        line.clear();
        match (&mut reader).take(MAX_LINE).read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) if !line.ends_with('\n') => break, // over-long line, or EOF mid-line
            Ok(_) => {}
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let reply = watch_reply(text, ctl, pending);
        if !send(out, &reply) {
            break;
        }
    }
    done.store(true, Ordering::Release);
    ctl.nudge(); // wake the pusher so it notices
    close(out);
}

fn watch_reply(text: &str, ctl: &Handle, pending: &AtomicUsize) -> String {
    let mut words = text.split_whitespace();
    let verb = words.next().unwrap_or("");
    if !VERBS.contains(&verb) {
        return format!("err: unknown command {verb}");
    }
    match parse_control(verb, words.next()) {
        Ok(c) => control_reply(ctl, c, pending.load(Ordering::Acquire) > 0),
        Err(e) => format!("err: {e}"),
    }
}

// -------------------------------- tests --------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{Config, Session, Sink, SinkFactory};
    use std::io::{BufRead, BufReader};
    use std::time::Instant;

    struct Null;

    impl Sink for Null {
        fn write(&mut self, _pcm: &[u8]) -> std::io::Result<()> {
            Ok(())
        }
        fn finish(self: Box<Self>) -> anyhow::Result<()> {
            Ok(())
        }
        fn kill(self: Box<Self>) {}
    }

    struct NullFactory;

    impl SinkFactory for NullFactory {
        fn spawn(&mut self) -> anyhow::Result<Box<dyn Sink>> {
            Ok(Box::new(Null))
        }
        fn lag_samples(&self) -> u64 {
            0
        }
    }

    fn session() -> Session {
        // A short window so the "window ends" case does not take ten minutes, but
        // long enough that a loaded machine cannot outrun it mid-test.
        let cfg = Config {
            sink_idle: Duration::from_millis(100),
            window: Duration::from_millis(1000),
            ..Config::default()
        };
        Session::spawn(NullFactory, cfg)
    }

    /// A connected client with a read timeout, so a broken test fails instead of hanging.
    struct Client {
        reader: BufReader<UnixStream>,
        writer: UnixStream,
    }

    impl Client {
        fn line(&mut self) -> Option<String> {
            let mut l = String::new();
            match self.reader.read_line(&mut l) {
                Ok(0) => None,
                Ok(_) => Some(l.trim_end().to_string()),
                // the daemon closed with our unsent bytes still unread
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => None,
                Err(e) => panic!("read failed: {e}"),
            }
        }

        fn say(&mut self, text: &str) {
            self.writer.write_all(format!("{text}\n").as_bytes()).unwrap();
        }

        /// Lines until one starts with `prefix`.
        fn until(&mut self, prefix: &str) -> String {
            loop {
                let l = self.line().unwrap_or_else(|| panic!("closed while waiting for {prefix:?}"));
                if l.starts_with(prefix) {
                    return l;
                }
            }
        }
    }

    fn connect(ctl: &Handle, pending: &Arc<AtomicUsize>) -> Client {
        let (server, client) = UnixStream::pair().unwrap();
        client.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let reader = BufReader::new(server.try_clone().unwrap());
        serve_watch(server, reader, ctl.clone(), Arc::clone(pending));
        Client { reader: BufReader::new(client.try_clone().unwrap()), writer: client }
    }

    fn wait_for(mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn verbs_parse_with_their_argument() {
        for (v, c) in [
            ("pause", Control::Pause),
            ("resume", Control::Resume),
            ("toggle", Control::Toggle),
            ("back", Control::Back),
            ("cancel", Control::Cancel),
            ("status", Control::Status),
        ] {
            assert_eq!(parse_control(v, None).unwrap(), c);
        }
        assert_eq!(parse_control("rewind", Some("2.5")).unwrap(), Control::Rewind(2.5));
        assert!(matches!(parse_control("rewind", None).unwrap(), Control::Rewind(s) if s > 0.0));
        assert!(parse_control("rewind", Some("-1")).is_err());
        assert!(parse_control("rewind", Some("soon")).is_err());
        assert!(parse_control("dance", None).is_err());
        for v in VERBS {
            assert!(parse_control(v, None).is_ok(), "{v} is allowlisted but not parsed");
        }
    }

    #[test]
    fn cancel_with_nothing_playing_depends_on_the_queue() {
        let s = session();
        let h = s.handle();
        assert_eq!(control_reply(&h, Control::Cancel, false), "err: nothing playing");
        assert_eq!(control_reply(&h, Control::Cancel, true), "ok idle cancelled");
        assert_eq!(control_reply(&h, Control::Status, false), "ok idle");
        assert_eq!(control_reply(&h, Control::Back, false), "err: nothing playing");
        s.finish().unwrap();
    }

    #[test]
    fn watch_says_ok_then_the_state_and_runs_verbs() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold(); // a job is running, so there is something to watch
        let mut c = connect(&h, &pending);
        assert_eq!(c.line().as_deref(), Some("ok"));
        assert_eq!(c.line().as_deref(), Some("state playing"));

        c.say("status");
        assert_eq!(c.until("ok"), "ok playing");
        c.say("pause");
        // the answer and the pushed state may come in either order
        let mut got = vec![c.line().unwrap(), c.line().unwrap()];
        got.sort();
        assert_eq!(got, ["ok paused", "state paused"]);
        c.say("toggle");
        let mut got = vec![c.line().unwrap(), c.line().unwrap()];
        got.sort();
        assert_eq!(got, ["ok playing", "state playing"]);
        s.release();
        s.finish().unwrap();
    }

    #[test]
    fn watch_rejects_unknown_and_bad_lines_but_stays_open() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold();
        let mut c = connect(&h, &pending);
        c.until("state");
        c.say("dance");
        assert_eq!(c.until("err"), "err: unknown command dance");
        c.say("rewind soon");
        assert_eq!(c.until("err"), "err: invalid rewind seconds");
        c.say("");
        c.say("status");
        assert_eq!(c.until("ok"), "ok playing");
        s.release();
        s.finish().unwrap();
    }

    #[test]
    fn watch_closes_when_there_is_nothing_left_to_control() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold();
        let mut c = connect(&h, &pending);
        assert_eq!(c.until("state"), "state playing");
        s.release(); // no audio was ever pushed: idle, no history, nothing queued
        assert_eq!(c.until("state"), "state idle");
        assert_eq!(c.line(), None, "the daemon closes the connection");
        s.finish().unwrap();
    }

    #[test]
    fn watch_stays_while_a_job_is_still_queued() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(1));
        let mut c = connect(&h, &pending);
        assert_eq!(c.until("state"), "state idle");
        // idle and empty, but a queued job: not over yet
        c.say("status");
        assert_eq!(c.until("ok"), "ok idle");
        pending.store(0, Ordering::Release);
        assert_eq!(c.line(), None, "closes within the settle poll once the queue is empty");
        s.finish().unwrap();
    }

    #[test]
    fn watch_stays_through_the_window_then_closes() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold();
        s.start_job(h.epoch());
        s.push(vec![0.1; 2400]).unwrap();
        let mut c = connect(&h, &pending);
        c.until("state");
        s.release();
        assert_eq!(c.until("state"), "state idle");
        // idle with something to replay: `back` still works and the line stays open
        wait_for(|| h.snapshot().history);
        c.say("back");
        assert!(c.until("ok").starts_with("ok playing"));
        // after the 1 s window with no more audio the daemon hangs up (a quick
        // playing/idle flip may be coalesced, so only the hang-up is asserted)
        while c.line().is_some() {}
        assert!(!h.snapshot().history);
        s.finish().unwrap();
    }

    #[test]
    fn a_client_that_hangs_up_frees_the_watch_threads() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold();
        let mut c = connect(&h, &pending);
        c.until("state");
        assert!(Arc::strong_count(&pending) > 1);
        drop(c);
        wait_for(|| Arc::strong_count(&pending) == 1);
        s.release();
        s.finish().unwrap();
    }

    #[test]
    fn an_over_long_line_ends_the_connection() {
        let s = session();
        let h = s.handle();
        let pending = Arc::new(AtomicUsize::new(0));
        s.hold();
        let mut c = connect(&h, &pending);
        c.until("state");
        c.writer.write_all(&vec![b'x'; 5000]).unwrap();
        assert_eq!(c.line(), None);
        s.release();
        s.finish().unwrap();
    }
}
