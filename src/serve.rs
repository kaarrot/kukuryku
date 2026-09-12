//! `ryk --serve` (warm daemon) and `ryk --send` (thin client).
//!
//! The compiled tract `Pipeline` is text/voice/speed-independent (those only feed
//! `prepare`/`synthesize` per utterance), so one warm pipeline can serve every
//! request. `--serve` compiles it once and owns audio output; `--send` pipes the
//! selected text to it — so editor-driven TTS ("select text -> hear it, again and
//! again") never pays the ~4 s stage-compile per call.
//!
//! Unix-only and additive: plain `ryk "text"` / stdin are unchanged and need no
//! daemon. See docs/ryk-cli-and-daemon.md.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::info;
use crate::kokoro;
use crate::tract_backend::Pipeline;

/// A queued utterance: per-request config plus the text to speak.
struct Job {
    voice: String,
    lang: String,
    speed: f32,
    text: String,
}

/// Where the daemon listens and the client connects. `$RYK_SOCKET` wins; else
/// `$XDG_RUNTIME_DIR/ryk.sock`; else `$TMPDIR/ryk-$USER.sock` (Termux sets
/// `$TMPDIR` to `$PREFIX/tmp` and has no `/tmp`); else `/tmp/ryk-$USER.sock`.
pub fn socket_path() -> PathBuf {
    if let Some(p) = std::env::var_os("RYK_SOCKET") {
        return PathBuf::from(p);
    }
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        return Path::new(&dir).join("ryk.sock");
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
    if let Some(dir) = std::env::var_os("TMPDIR") {
        return Path::new(&dir).join(format!("ryk-{user}.sock"));
    }
    PathBuf::from(format!("/tmp/ryk-{user}.sock"))
}

/// Best-effort socket-file cleanup when the daemon exits normally.
struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ------------------------------- daemon --------------------------------------

/// Default `--serve` lifetime with no jobs and no live audio sink. `--send` already
/// auto-starts a replacement, so a forgotten daemon shouldn't sit for days. `0` /
/// `off` / `none` / `-1` disables. Override with `$RYK_IDLE_TIMEOUT` (seconds).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1800;

/// Parse `$RYK_IDLE_TIMEOUT`. `None` means "run until killed".
pub(crate) fn parse_idle_timeout(val: Option<&str>) -> Option<Duration> {
    match val {
        None => Some(Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS)),
        Some(s) if matches!(s, "" | "0" | "off" | "none" | "-1") => None,
        Some(s) => match s.parse::<u64>() {
            Ok(0) => None,
            Ok(secs) => Some(Duration::from_secs(secs)),
            Err(_) => Some(Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS)),
        },
    }
}

fn daemon_idle_timeout() -> Option<Duration> {
    parse_idle_timeout(std::env::var("RYK_IDLE_TIMEOUT").ok().as_deref())
}

/// Parse `$RYK_SINK_IDLE_MS`. Missing or unparseable values use the 10-minute default.
pub(crate) fn parse_sink_idle_ms(val: Option<&str>) -> Duration {
    match val {
        None => kokoro::DEFAULT_SINK_IDLE,
        Some(s) => match s.parse::<u64>() {
            Ok(ms) => Duration::from_millis(ms),
            Err(_) => kokoro::DEFAULT_SINK_IDLE,
        },
    }
}

fn sink_idle_duration() -> Duration {
    parse_sink_idle_ms(std::env::var("RYK_SINK_IDLE_MS").ok().as_deref())
}

/// Run the warm daemon: compile the pipeline once, then serve queued utterances
/// over a Unix socket until killed or idle.
pub fn serve() -> Result<()> {
    let path = socket_path();

    // Stale-socket handling: a live daemon means "already running"; a dead socket
    // file (leftover from a crash) is unlinked so we can bind fresh.
    if path.exists() {
        if UnixStream::connect(&path).is_ok() {
            eprintln!("[ryk-serve] already running at {}", path.display());
            return Ok(());
        }
        let _ = std::fs::remove_file(&path);
    }

    // Compile ONCE. Voice is per-request; the startup voice just resolves the
    // asset dir (and confirms the split stages are present).
    let voice0 = kokoro::env_or("KOKORO_VOICE", "af_heart");
    let assets = kokoro::resolve_assets_tract(&voice0)?;
    let dir = assets.dir.clone();
    info!("[ryk-serve] compiling pipeline from {} ...", dir.display());
    let t0 = Instant::now();
    let pipeline = Pipeline::new(&dir)?;
    info!("[ryk-serve] pipeline ready in {:.2}s", t0.elapsed().as_secs_f64());

    // Sink stays warm across overlapping / nearby utterances, then closes so
    // PulseAudio can idle-exit. The compiled pipeline stays hot until the
    // daemon idle timeout.
    let sink_idle = sink_idle_duration();
    info!(
        "[ryk-serve] audio sink grace {}s after last sample",
        sink_idle.as_secs()
    );
    let player = kokoro::StreamPlayer::with_sink_idle(sink_idle)?;
    let sink_live = player.sink_live_flag();

    // A single worker owns the pipeline (it's `&mut` / single-instance) and drains
    // an mpsc queue FIFO — that is the "queue" concurrency policy for free.
    let pending = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<Job>();
    let pending_w = Arc::clone(&pending);
    let worker = std::thread::spawn(move || worker_loop(pipeline, player, dir, rx, pending_w));

    let listener =
        UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;
    let _guard = SocketGuard(path.clone());
    let idle_timeout = daemon_idle_timeout();
    match idle_timeout {
        Some(t) => info!(
            "[ryk-serve] listening on {} (idle timeout {}s)",
            path.display(),
            t.as_secs()
        ),
        None => info!("[ryk-serve] listening on {} (no idle timeout)", path.display()),
    }

    if let Some(timeout) = idle_timeout {
        listener.set_nonblocking(true).context("setting listener nonblocking")?;
        accept_loop(&listener, &tx, &pending, &sink_live, timeout)?;
    } else {
        // Blocking accept: no 250ms poll when the user disabled idle-exit.
        for conn in listener.incoming() {
            match conn {
                Ok(conn) => {
                    if let Err(e) = handle_conn(conn, &tx, &pending) {
                        eprintln!("[ryk-serve] connection error: {e:#}");
                    }
                }
                Err(e) => eprintln!("[ryk-serve] accept error: {e}"),
            }
        }
    }

    drop(tx); // close the queue so the worker drains and exits
    let _ = worker.join();
    Ok(())
}

fn accept_loop(
    listener: &UnixListener,
    tx: &mpsc::Sender<Job>,
    pending: &AtomicUsize,
    sink_live: &AtomicBool,
    idle_timeout: Duration,
) -> Result<()> {
    let mut idle_since: Option<Instant> = None;
    loop {
        match listener.accept() {
            Ok((conn, _)) => {
                idle_since = None;
                if let Err(e) = handle_conn(conn, tx, pending) {
                    eprintln!("[ryk-serve] connection error: {e:#}");
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                let truly_idle =
                    pending.load(Ordering::Acquire) == 0 && !sink_live.load(Ordering::Acquire);
                if truly_idle {
                    let since = idle_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= idle_timeout {
                        eprintln!(
                            "[ryk-serve] idle for {}s; shutting down",
                            idle_timeout.as_secs()
                        );
                        return Ok(());
                    }
                } else {
                    idle_since = None;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(e) => eprintln!("[ryk-serve] accept error: {e}"),
        }
    }
}

/// Read one utterance from a connection (header line + body to EOF), enqueue it,
/// and ack the client.
fn handle_conn(mut conn: UnixStream, tx: &mpsc::Sender<Job>, pending: &AtomicUsize) -> Result<()> {
    // Blocking read on this connection; the listener itself is nonblocking.
    conn.set_nonblocking(false).ok();
    let mut reader = BufReader::new(conn.try_clone().context("cloning connection")?);
    let mut header = String::new();
    reader.read_line(&mut header).context("reading request header")?;
    let mut body = String::new();
    reader.read_to_string(&mut body).context("reading request body")?;

    match parse_request(&header, body) {
        Ok(job) => {
            let chars = job.text.chars().count();
            pending.fetch_add(1, Ordering::Release);
            if tx.send(job).is_err() {
                pending.fetch_sub(1, Ordering::Release);
                bail!("worker thread gone");
            }
            conn.write_all(b"ok\n").ok();
            info!("[ryk-serve] queued utterance ({chars} chars)");
            Ok(())
        }
        Err(e) => {
            conn.write_all(format!("err: {e}\n").as_bytes()).ok();
            Err(e)
        }
    }
}

/// Parse the wire request: a tab-separated `voice\tlang\tspeed` header line, then
/// the UTF-8 body text. Missing/short fields fall back to the usual defaults.
fn parse_request(header: &str, body: String) -> Result<Job> {
    let mut parts = header.trim_end_matches(['\n', '\r']).split('\t');
    let voice = parts.next().filter(|s| !s.is_empty()).unwrap_or("af_heart").to_string();
    let lang = parts.next().filter(|s| !s.is_empty()).unwrap_or("en-us").to_string();
    let speed: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let text = body.trim().to_string();
    if text.is_empty() {
        bail!("empty text");
    }
    Ok(Job { voice, lang, speed, text })
}

/// Decrements the pending-job count when a worker iteration finishes (including
/// on panic unwind), so a failed synth cannot pin the idle timer forever.
struct PendingGuard<'a>(&'a AtomicUsize);
impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Keeps `pacat`/`ffplay` open while the next sentence of this utterance is still
/// synthesizing — that gap can be longer than the sink idle, and closing the pipe
/// there is what produces a click between sentences.
struct SinkHold<'a>(&'a kokoro::StreamPlayer);
impl Drop for SinkHold<'_> {
    fn drop(&mut self) {
        self.0.release_sink();
    }
}

/// The worker: owns the pipeline, drains the queue, plays each utterance. Runs
/// until the queue is closed (daemon shutdown), then drains playback.
fn worker_loop(
    mut pipeline: Pipeline,
    player: kokoro::StreamPlayer,
    dir: PathBuf,
    rx: mpsc::Receiver<Job>,
    pending: Arc<AtomicUsize>,
) {
    let mut voices: HashMap<String, PathBuf> = HashMap::new();
    for job in rx {
        let _pending = PendingGuard(&pending);
        if let Err(e) = speak_job(&mut pipeline, &player, &dir, &mut voices, &job) {
            eprintln!("[ryk-serve] synth error: {e:#}");
        }
    }
    let _ = player.finish();
}

/// Synthesize one utterance sentence-by-sentence and stream it to the player.
fn speak_job(
    pipeline: &mut Pipeline,
    player: &kokoro::StreamPlayer,
    dir: &Path,
    voices: &mut HashMap<String, PathBuf>,
    job: &Job,
) -> Result<()> {
    player.hold_sink();
    let _hold = SinkHold(player);
    let voice_path = resolve_voice(dir, &job.voice, voices)?;
    let sentences = kokoro::split_sentences(&job.text);
    info!(
        "[ryk-serve] speak: voice={} lang={} speed={} {} sentence(s)",
        job.voice, job.lang, job.speed, sentences.len(),
    );
    // A chunk with nothing to say is skipped so the rest of the utterance still plays,
    // and an all-skipped job is a warning rather than an error — the daemon stays warm.
    let (mut spoken, mut failed) = (0usize, 0usize);
    for sentence in &sentences {
        let prep = match kokoro::prepare_or_skip(sentence, &job.lang, &voice_path)? {
            kokoro::ChunkPrep::Ready(prep) => prep,
            kokoro::ChunkPrep::Unspeakable => continue,
            kokoro::ChunkPrep::Failed => {
                failed += 1;
                continue;
            }
        };
        let audio = pipeline.synthesize(&prep.ids, &prep.style, job.speed)?;
        player.push(audio)?;
        spoken += 1;
    }
    if spoken == 0 {
        kokoro::warn_nothing_spoken(failed);
    }
    Ok(())
}

/// The voice file for `voice`, cached. Prefers the daemon's bundle dir; only falls
/// back to the HF cache for a voice the bundle doesn't carry (same policy as
/// `resolve_assets_tract`).
fn resolve_voice(dir: &Path, voice: &str, cache: &mut HashMap<String, PathBuf>) -> Result<PathBuf> {
    if let Some(p) = cache.get(voice) {
        return Ok(p.clone());
    }
    let local = dir.join("voices").join(format!("{voice}.bin"));
    let path = if local.is_file() {
        local
    } else {
        kokoro::resolve_assets_tract(voice)?.voice_path
    };
    cache.insert(voice.to_string(), path.clone());
    Ok(path)
}

// ------------------------------- client --------------------------------------

/// Send text to the daemon (auto-starting it if needed). Text comes from the args
/// after `--send`, else stdin — the editor entry point.
pub fn send() -> Result<()> {
    let text = client_text()?;
    let voice = kokoro::env_or("KOKORO_VOICE", "af_heart");
    let lang = kokoro::env_or("KOKORO_LANG", "en-us");
    let speed = kokoro::env_or("KOKORO_SPEED", "1.0");
    let path = socket_path();

    let mut stream = match UnixStream::connect(&path) {
        Ok(s) => s,
        Err(_) => {
            spawn_daemon(&path)?;
            connect_retry(&path)?
        }
    };

    // Header line + body; half-close so the daemon reads the body to EOF.
    stream
        .write_all(format!("{voice}\t{lang}\t{speed}\n").as_bytes())
        .context("sending request header")?;
    stream.write_all(text.as_bytes()).context("sending request body")?;
    stream.shutdown(Shutdown::Write).context("half-closing connection")?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp).context("reading daemon ack")?;
    let resp = resp.trim();
    if let Some(msg) = resp.strip_prefix("err:") {
        bail!("daemon rejected request:{msg}");
    }
    info!("[ryk] queued {} chars -> {}", text.chars().count(), path.display());
    Ok(())
}

/// Client text: args after `--send` (joined), else stdin, markdown-stripped.
///
/// Stripping happens here, client-side, so `--raw` is honoured per invocation and
/// the wire header stays `voice\tlang\tspeed`. The daemon speaks whatever body it
/// receives and never re-strips it.
fn client_text() -> Result<String> {
    // Drop the program name, `--send` itself, and any `-v`/`--verbose`/`--raw`
    // (which may appear on either side of `--send`, e.g. `ryk -v --send hi` or
    // `ryk --send -v hi`). What's left is user text plus an optional argv
    // separator `--` that read-verbatim callers use for leading-dash text.
    let mut args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--send" && a != "-v" && a != "--verbose" && a != "--raw")
        .collect();
    if args.first().is_some_and(|a| a == "--") {
        args.remove(0);
    }
    if !args.is_empty() {
        return Ok(kokoro::clean_input(&args.join(" ")));
    }
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).context("reading stdin")?;
    let buf = kokoro::clean_input(buf.trim());
    if buf.trim().is_empty() {
        bail!("no text provided (pass after --send or pipe to stdin)");
    }
    Ok(buf)
}

/// Launch `current_exe --serve` detached, with its own process group and stdio
/// redirected to a log file beside the socket, so the client can connect once the
/// pipeline is warm without the daemon's output leaking into the editor.
fn spawn_daemon(path: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("locating the ryk executable")?;
    let log = path.with_extension("log");
    info!(
        "[ryk] no daemon at {}; starting one (log: {}) ...",
        path.display(),
        log.display(),
    );
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--serve").process_group(0).stdin(std::process::Stdio::null());
    if let Ok(out) = std::fs::File::create(&log) {
        if let Ok(err) = out.try_clone() {
            cmd.stdout(out).stderr(err);
        }
    } else {
        cmd.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    }
    cmd.spawn().context("spawning `ryk --serve`")?;
    Ok(())
}

/// Poll-connect until the daemon accepts or we give up — the first connect covers
/// the one-time pipeline compile (a few seconds).
fn connect_retry(path: &Path) -> Result<UnixStream> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match UnixStream::connect(path) {
            Ok(s) => return Ok(s),
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(150)),
            Err(e) => {
                return Err(anyhow::Error::new(e).context(format!(
                    "daemon did not come up at {} (see {})",
                    path.display(),
                    path.with_extension("log").display(),
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_timeout_default_is_thirty_minutes() {
        assert_eq!(parse_idle_timeout(None), Some(Duration::from_secs(1800)));
    }

    #[test]
    fn idle_timeout_zero_or_off_disables() {
        for v in ["0", "off", "none", "-1", ""] {
            assert_eq!(parse_idle_timeout(Some(v)), None, "{v}");
        }
    }

    #[test]
    fn idle_timeout_parses_seconds() {
        assert_eq!(parse_idle_timeout(Some("60")), Some(Duration::from_secs(60)));
    }

    #[test]
    fn idle_timeout_garbage_falls_back_to_default() {
        assert_eq!(parse_idle_timeout(Some("nope")), Some(Duration::from_secs(1800)));
    }

    #[test]
    fn sink_idle_default_is_ten_minutes() {
        assert_eq!(parse_sink_idle_ms(None), Duration::from_secs(10 * 60));
        assert_eq!(kokoro::DEFAULT_SINK_IDLE, Duration::from_secs(10 * 60));
    }

    #[test]
    fn sink_idle_parses_milliseconds() {
        assert_eq!(parse_sink_idle_ms(Some("3000")), Duration::from_millis(3000));
    }

    #[test]
    fn sink_idle_garbage_falls_back_to_default() {
        assert_eq!(parse_sink_idle_ms(Some("nope")), Duration::from_secs(10 * 60));
    }
}
