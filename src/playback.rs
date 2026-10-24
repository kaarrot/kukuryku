//! Replayable playback for `ryk --serve`: a PCM history with a playhead, a sink
//! that can be killed and respawned, and the controls built on both (pause,
//! resume, back one sentence, rewind, cancel). See `docs/headset-control.md`
//! and `docs/rewind-plan.md`.
//!
//! Pure std. The audio sink is a trait, so the tests drive the whole thing with
//! a fake one and no sound device.
//!
//! Layout: [`PcmHistory`] (samples + playhead + sentence marks), [`Engine`]
//! (one thread's worth of state: history, sink, pause state, the controls), and
//! [`Session`] (the thread and the queue that feeds it).

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::kokoro::SAMPLE_RATE;

const SR: u64 = SAMPLE_RATE as u64;

/// Samples per pipe write (100 ms). The playhead moves one slice at a time, so a
/// control is noticed within a slice.
pub const WRITE_SLICE: usize = SAMPLE_RATE as usize / 10;

/// Max queued messages (sentences) between the synth thread and playback.
pub const STREAM_BUFFER: usize = 32;

/// How long a caller waits for the playback thread to answer a control.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);

/// Seconds to samples.
pub fn secs_to_samples(secs: f64) -> u64 {
    (secs.max(0.0) * SR as f64).round() as u64
}

// ------------------------------- history -------------------------------------

/// Samples that have been handed to the playback thread, with an explicit
/// playhead. Indices are absolute sample counts since the process started, so
/// they never rewind and `queued - playhead` is always "audio not yet played".
///
/// The store holds `[phys, end)`. Already-played audio is kept back to
/// `retain_from()` (the `played_cap` most recent seconds); older samples are
/// dropped lazily.
#[derive(Debug)]
pub struct PcmHistory {
    buf: Vec<f32>,
    phys: u64,
    playhead: u64,
    /// Absolute start sample of each appended chunk (= one sentence), ascending.
    marks: VecDeque<u64>,
    played_cap: u64,
    compact_after: u64,
}

impl PcmHistory {
    pub fn new(played_cap: u64, compact_after: u64) -> Self {
        Self {
            buf: Vec::new(),
            phys: 0,
            playhead: 0,
            marks: VecDeque::new(),
            played_cap,
            compact_after: compact_after.max(1),
        }
    }

    /// One past the last ingested sample.
    pub fn end(&self) -> u64 {
        self.phys + self.buf.len() as u64
    }

    /// Next sample to write to the sink.
    pub fn playhead(&self) -> u64 {
        self.playhead
    }

    /// Oldest sample a rewind may return to.
    pub fn retain_from(&self) -> u64 {
        self.phys.max(self.playhead.saturating_sub(self.played_cap))
    }

    /// Samples ingested but not yet written.
    pub fn unplayed(&self) -> u64 {
        self.end() - self.playhead
    }

    /// Anything a rewind could replay (or still has to play).
    pub fn has_audio(&self) -> bool {
        self.end() > self.retain_from()
    }

    /// Append one chunk (one sentence) and remember where it starts.
    pub fn append(&mut self, chunk: &[f32]) {
        if chunk.is_empty() {
            return;
        }
        self.marks.push_back(self.end());
        self.buf.extend_from_slice(chunk);
    }

    /// Up to `max` samples from the playhead, without moving it.
    pub fn peek(&self, max: usize) -> &[f32] {
        let start = (self.playhead - self.phys) as usize;
        let stop = (start + max).min(self.buf.len());
        &self.buf[start..stop]
    }

    /// Advance the playhead over `n` samples that were written, then trim.
    pub fn commit(&mut self, n: usize) {
        self.playhead += (n as u64).min(self.unplayed());
        self.compact();
    }

    /// Move the playhead to `target`, clamped to `[retain_from, end]`. Returns
    /// where it landed.
    pub fn seek(&mut self, target: u64) -> u64 {
        self.playhead = target.clamp(self.retain_from(), self.end());
        self.playhead
    }

    /// Start of the sentence containing `pos`, never before `retain_from`.
    pub fn sentence_start(&self, pos: u64) -> u64 {
        let floor = self.retain_from();
        self.marks
            .iter()
            .rev()
            .find(|&&m| m <= pos)
            .map_or(floor, |&m| m.max(floor))
    }

    /// Start of the sentence before the one starting at `s0`, if still retained.
    pub fn prev_sentence_start(&self, s0: u64) -> Option<u64> {
        let floor = self.retain_from();
        self.marks.iter().rev().find(|&&m| m < s0).copied().filter(|&m| m >= floor)
    }

    /// Drop everything after the chunk that contains `pos`. A cancel keeps the
    /// sentence being heard whole so `back` can replay it.
    pub fn truncate_after_chunk(&mut self, pos: u64) {
        let Some(&next) = self.marks.iter().find(|&&m| m > pos) else { return };
        self.buf.truncate((next - self.phys) as usize);
        while self.marks.back().is_some_and(|&m| m >= next) {
            self.marks.pop_back();
        }
    }

    /// Forget everything. Numbering continues, so absolute indices stay valid.
    pub fn clear(&mut self) {
        self.phys = self.end();
        self.playhead = self.phys;
        self.buf.clear();
        self.marks.clear();
    }

    /// Drop samples older than `retain_from`, but only once `compact_after` of
    /// them have piled up, so the memmove is rare.
    fn compact(&mut self) {
        let floor = self.retain_from();
        if floor - self.phys < self.compact_after {
            return;
        }
        self.buf.drain(..(floor - self.phys) as usize);
        self.phys = floor;
        while self.marks.front().is_some_and(|&m| m < floor) {
            self.marks.pop_front();
        }
    }
}

// ------------------------------- controls ------------------------------------

/// What a caller can ask the playback thread to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Control {
    Pause,
    Resume,
    Toggle,
    /// Previous-track rule: restart the sentence you are in, or go to the
    /// previous one when you have only just started it.
    Back,
    /// Go back this many seconds.
    Rewind(f32),
    /// Drop everything unplayed (the caller also invalidates queued work).
    Cancel,
    Status,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayState {
    Idle,
    Playing,
    Paused,
}

impl PlayState {
    pub fn as_str(self) -> &'static str {
        match self {
            PlayState::Idle => "idle",
            PlayState::Playing => "playing",
            PlayState::Paused => "paused",
        }
    }
}

/// The answer to a [`Control`]: the state afterwards plus a short note
/// (`back`, `at-start`, `rewound 4.0`, ...).
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub state: PlayState,
    pub note: Option<String>,
}

impl Outcome {
    fn new(state: PlayState, note: Option<&str>) -> Self {
        Self { state, note: note.map(str::to_string) }
    }

    /// `playing back` / `idle` -- the text after `ok ` on the wire.
    pub fn line(&self) -> String {
        match &self.note {
            Some(n) => format!("{} {n}", self.state.as_str()),
            None => self.state.as_str().to_string(),
        }
    }
}

/// Returned by a push after a cancel: the job should stop quietly.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

// -------------------------------- sink ---------------------------------------

/// A running audio sink (a `pacat`/`ffplay` child, or a fake in tests).
pub trait Sink: Send {
    /// Write raw f32le PCM. May block until the sink takes it.
    fn write(&mut self, pcm: &[u8]) -> io::Result<()>;
    /// Close stdin and let the sink play out what it has.
    fn finish(self: Box<Self>) -> anyhow::Result<()>;
    /// Stop now, dropping whatever the sink has buffered.
    fn kill(self: Box<Self>);
}

pub trait SinkFactory: Send {
    fn spawn(&mut self) -> anyhow::Result<Box<dyn Sink>>;
    /// Samples between what was written and what is audible (pipe + sink
    /// buffers). Used to put a pause or a cancel where the listener heard it.
    fn lag_samples(&self) -> u64;
}

// ------------------------------- engine --------------------------------------

/// Tunables, normally from `RYK_*` environment variables.
#[derive(Clone, Debug)]
pub struct Config {
    /// Close the sink this long after the last sample (unless held).
    pub sink_idle: Duration,
    /// Keep the history this long after the last sample so `back` still works.
    pub window: Duration,
    /// A pause longer than this becomes a cancel.
    pub pause_max: Duration,
    /// Already-played audio kept for rewinding.
    pub played_cap: u64,
    /// `back` restarts the current sentence when this far into it.
    pub restart_after: u64,
    /// Resume replays this much (a sentence start within `resume_snap` wins).
    pub resume_replay: u64,
    pub resume_snap: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sink_idle: Duration::from_secs(30),
            window: Duration::from_secs(600),
            pause_max: Duration::from_secs(600),
            played_cap: secs_to_samples(180.0),
            restart_after: secs_to_samples(1.5),
            resume_replay: secs_to_samples(1.0),
            resume_snap: secs_to_samples(2.0),
        }
    }
}

/// What the rest of the process may read without asking the playback thread.
#[derive(Debug, Default)]
pub struct Status {
    /// Absolute playhead: samples written to the sink.
    pub playhead: AtomicU64,
    pub paused: AtomicBool,
    /// History still retained (inside the rewind window).
    pub history: AtomicBool,
}

pub struct Engine<F: SinkFactory> {
    hist: PcmHistory,
    factory: F,
    sink: Option<Box<dyn Sink>>,
    /// Playhead when the current sink was spawned; nothing before it is audible.
    sink_start: u64,
    paused_at: Option<Instant>,
    /// Last time audio was ingested or written. Drives sink idle and the window.
    last_activity: Option<Instant>,
    fails: u8,
    cfg: Config,
    status: Arc<Status>,
}

impl<F: SinkFactory> Engine<F> {
    pub fn new(factory: F, cfg: Config, status: Arc<Status>) -> Self {
        let hist = PcmHistory::new(cfg.played_cap, secs_to_samples(10.0));
        Self {
            hist,
            factory,
            sink: None,
            sink_start: 0,
            paused_at: None,
            last_activity: None,
            fails: 0,
            cfg,
            status,
        }
    }

    pub fn end(&self) -> u64 {
        self.hist.end()
    }

    pub fn playhead(&self) -> u64 {
        self.hist.playhead()
    }

    pub fn is_paused(&self) -> bool {
        self.paused_at.is_some()
    }

    pub fn has_sink(&self) -> bool {
        self.sink.is_some()
    }

    /// Unwritten samples are waiting and nothing is stopping us.
    pub fn wants_write(&self) -> bool {
        !self.is_paused() && self.hist.unplayed() > 0
    }

    /// The next queued chunk may be taken. Only when the last one is fully
    /// written, so the lookahead stays in the queue and history stays small.
    pub fn can_ingest(&self) -> bool {
        self.hist.unplayed() == 0
    }

    /// A job started: open the sink now so it is up for the first sentence.
    pub fn warm(&mut self, now: Instant) {
        if !self.is_paused() {
            self.ensure_sink();
        }
        self.last_activity = Some(now);
    }

    pub fn ingest(&mut self, chunk: &[f32], now: Instant) {
        self.hist.append(chunk);
        self.last_activity = Some(now);
        self.publish();
    }

    /// Write one slice at the playhead. Blocks while the sink is full.
    pub fn write_slice(&mut self, now: Instant) {
        if !self.ensure_sink() {
            // No sink: drop the audio rather than retry forever.
            let rest = self.hist.unplayed() as usize;
            self.hist.commit(rest);
            self.publish();
            return;
        }
        let slice = self.hist.peek(WRITE_SLICE);
        let n = slice.len();
        let bytes: &[u8] = bytemuck::cast_slice(slice);
        let res = self.sink.as_mut().map(|s| s.write(bytes));
        match res {
            Some(Ok(())) => {
                self.fails = 0;
                self.hist.commit(n);
                self.last_activity = Some(now);
            }
            _ => {
                eprintln!("[kokoro] audio sink write failed; will respawn");
                if let Some(s) = self.sink.take() {
                    s.kill();
                }
                self.fails += 1;
                if self.fails >= 3 {
                    self.fails = 0;
                    let rest = self.hist.unplayed() as usize;
                    self.hist.commit(rest);
                }
            }
        }
        self.publish();
    }

    /// Record that audio was just written (called after a blocking write, with
    /// a fresh clock reading).
    pub fn touch(&mut self, now: Instant) {
        self.last_activity = Some(now);
    }

    /// Timers: close an idle sink, expire the history, end a long pause.
    pub fn tick(&mut self, now: Instant, held: bool) {
        if let Some(since) = self.paused_at {
            if now.duration_since(since) >= self.cfg.pause_max {
                self.cancel(now);
            }
            return;
        }
        if self.hist.unplayed() > 0 || held {
            return;
        }
        let Some(last) = self.last_activity else { return };
        let idle = now.duration_since(last);
        if self.sink.is_some() && idle >= self.cfg.sink_idle {
            crate::info!("[kokoro] audio sink idle; closing");
            if let Some(s) = self.sink.take() {
                if let Err(e) = s.finish() {
                    eprintln!("[kokoro] {e:#}; will respawn on next chunk");
                }
            }
        }
        if self.hist.has_audio() && idle >= self.cfg.window {
            self.hist.clear();
        }
        self.publish();
    }

    /// When `tick` next has something to do, or `None` to sleep until woken.
    pub fn next_deadline(&self, held: bool) -> Option<Instant> {
        if let Some(since) = self.paused_at {
            return Some(since + self.cfg.pause_max);
        }
        if self.hist.unplayed() > 0 || held {
            return None;
        }
        let last = self.last_activity?;
        let mut d: Option<Instant> = None;
        if self.sink.is_some() {
            d = Some(last + self.cfg.sink_idle);
        }
        if self.hist.has_audio() {
            let w = last + self.cfg.window;
            d = Some(d.map_or(w, |x| x.min(w)));
        }
        d
    }

    /// Close the sink at shutdown, letting it play out.
    pub fn shutdown(&mut self) {
        if let Some(s) = self.sink.take() {
            let _ = s.finish();
        }
    }

    /// Where the listener is: the playhead minus what is still in the pipe.
    fn audible(&self, now: Instant) -> u64 {
        let ph = self.hist.playhead();
        if self.sink.is_none() {
            return ph;
        }
        let in_flight = match self.last_activity {
            Some(t) => {
                let elapsed = secs_to_samples(now.saturating_duration_since(t).as_secs_f64());
                self.factory.lag_samples().saturating_sub(elapsed)
            }
            None => 0,
        };
        ph.saturating_sub(in_flight).max(self.sink_start).max(self.hist.retain_from())
    }

    fn state(&self, now: Instant, held: bool) -> PlayState {
        if self.is_paused() {
            return PlayState::Paused;
        }
        let tail = self.sink.is_some()
            && self.last_activity.is_some_and(|t| {
                secs_to_samples(now.saturating_duration_since(t).as_secs_f64())
                    < self.factory.lag_samples()
            });
        if self.hist.unplayed() > 0 || held || tail {
            PlayState::Playing
        } else {
            PlayState::Idle
        }
    }

    /// Run one control. `held` is whether a job is in progress (so a pause
    /// between two sentences still counts as playing).
    pub fn apply(&mut self, ctl: Control, now: Instant, held: bool) -> Result<Outcome, String> {
        const NOTHING: &str = "nothing playing";
        let state = self.state(now, held);
        match ctl {
            Control::Status => Ok(Outcome::new(state, None)),
            Control::Pause => match state {
                PlayState::Paused => Ok(Outcome::new(PlayState::Paused, None)),
                PlayState::Idle => Err(NOTHING.into()),
                PlayState::Playing => {
                    self.pause(now);
                    Ok(Outcome::new(PlayState::Paused, None))
                }
            },
            Control::Resume => match state {
                PlayState::Paused => {
                    self.resume(now);
                    Ok(Outcome::new(PlayState::Playing, None))
                }
                PlayState::Playing => Ok(Outcome::new(PlayState::Playing, None)),
                PlayState::Idle => Err(NOTHING.into()),
            },
            Control::Toggle => match state {
                PlayState::Paused => self.apply(Control::Resume, now, held),
                PlayState::Playing => self.apply(Control::Pause, now, held),
                PlayState::Idle => Err(NOTHING.into()),
            },
            Control::Back => {
                if !self.hist.has_audio() {
                    return Err(NOTHING.into());
                }
                let a = self.audible(now);
                let s0 = self.hist.sentence_start(a);
                let (target, at_start) = if a - s0 > self.cfg.restart_after {
                    (s0, false)
                } else {
                    match self.hist.prev_sentence_start(s0) {
                        Some(p) => (p, false),
                        None => (s0, true),
                    }
                };
                self.restart_at(target, now);
                Ok(Outcome::new(PlayState::Playing, Some(if at_start { "at-start" } else { "back" })))
            }
            Control::Rewind(secs) => {
                if !self.hist.has_audio() {
                    return Err(NOTHING.into());
                }
                let a = self.audible(now);
                let want = secs_to_samples(secs as f64);
                let target = a.saturating_sub(want).max(self.hist.retain_from());
                let applied = a - target.min(a);
                if applied == 0 {
                    return Ok(Outcome::new(state, Some("rewound 0.0")));
                }
                self.restart_at(target, now);
                let note = format!("rewound {:.1}", applied as f64 / SR as f64);
                Ok(Outcome::new(PlayState::Playing, Some(&note)))
            }
            Control::Cancel => {
                // "nothing" when there was nothing of ours to stop; the daemon
                // may still have queued jobs, which it cancels by epoch.
                let did = state != PlayState::Idle || self.hist.unplayed() > 0;
                self.cancel(now);
                Ok(Outcome::new(PlayState::Idle, Some(if did { "cancelled" } else { "nothing" })))
            }
        }
    }

    /// Stop the sink now and park the playhead where the listener was.
    fn pause(&mut self, now: Instant) {
        let a = self.audible(now);
        self.kill_sink();
        self.hist.seek(a);
        self.paused_at = Some(now);
        self.publish();
    }

    fn resume(&mut self, now: Instant) {
        let ph = self.hist.playhead();
        let s0 = self.hist.sentence_start(ph);
        let target = if ph - s0 <= self.cfg.resume_snap {
            s0
        } else {
            ph.saturating_sub(self.cfg.resume_replay).max(s0)
        };
        self.hist.seek(target);
        self.paused_at = None;
        self.last_activity = Some(now);
        self.publish();
    }

    /// Move to `target` and play from there (also ends a pause).
    fn restart_at(&mut self, target: u64, now: Instant) {
        self.kill_sink();
        self.hist.seek(target);
        self.paused_at = None;
        self.last_activity = Some(now);
        self.publish();
    }

    /// Drop the unplayed audio. The sentence being heard stays in the history,
    /// whole, with the playhead at its end, so `back` replays it.
    fn cancel(&mut self, now: Instant) {
        let a = self.audible(now);
        self.kill_sink();
        self.hist.seek(a);
        self.hist.truncate_after_chunk(a);
        self.hist.seek(self.hist.end());
        self.paused_at = None;
        self.last_activity = Some(now);
        self.publish();
    }

    fn kill_sink(&mut self) {
        if let Some(s) = self.sink.take() {
            s.kill();
        }
    }

    fn ensure_sink(&mut self) -> bool {
        if self.sink.is_some() {
            return true;
        }
        match self.factory.spawn() {
            Ok(s) => {
                self.sink = Some(s);
                self.sink_start = self.hist.playhead();
                true
            }
            Err(e) => {
                eprintln!("[kokoro] {e:#}; dropping audio");
                false
            }
        }
    }

    fn publish(&self) {
        self.status.playhead.store(self.hist.playhead(), Ordering::Release);
        self.status.paused.store(self.is_paused(), Ordering::Release);
        self.status.history.store(self.hist.has_audio(), Ordering::Release);
    }
}

// ------------------------------- session -------------------------------------

enum Msg {
    /// A job is starting: open the sink.
    Warm,
    Chunk(Vec<f32>),
}

struct Queued {
    msg: Msg,
    epoch: u64,
}

struct CtlReq {
    ctl: Control,
    reply: mpsc::Sender<Result<Outcome, String>>,
}

#[derive(Default)]
struct Queue {
    chunks: VecDeque<Queued>,
    ctls: VecDeque<CtlReq>,
    closed: bool,
    /// The playback thread has exited (normally or by panic).
    dead: bool,
}

struct Shared {
    q: Mutex<Queue>,
    /// Wakes the playback thread.
    consumer: Condvar,
    /// Wakes pushes blocked on a full queue.
    producer: Condvar,
    hold: AtomicBool,
    /// Samples accepted by `push` and not dropped by a cancel.
    queued: AtomicU64,
    /// Bumped by every cancel.
    cancel_epoch: AtomicU64,
    /// The epoch the running job was stamped with (see `start_job`).
    job_epoch: AtomicU64,
    status: Arc<Status>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The caller's side of the playback thread. Cheap to clone via [`Handle`].
pub struct Session {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// What the accept loop keeps after the [`Session`] has moved into the worker.
#[derive(Clone)]
pub struct Handle {
    shared: Arc<Shared>,
}

/// Marks the thread dead and wakes everyone waiting on it, even on panic.
struct DeadOnDrop(Arc<Shared>);

impl Drop for DeadOnDrop {
    fn drop(&mut self) {
        let mut q = self.0.lock();
        q.dead = true;
        for r in q.ctls.drain(..) {
            let _ = r.reply.send(Err("playback thread stopped".into()));
        }
        drop(q);
        self.0.producer.notify_all();
    }
}

impl Session {
    pub fn spawn<F: SinkFactory + 'static>(factory: F, cfg: Config) -> Self {
        Self::spawn_with(factory, cfg, || {})
    }

    /// Like [`spawn`](Self::spawn), running `init` first on the playback thread
    /// (the daemon pins it to the little cores there).
    pub fn spawn_with<F: SinkFactory + 'static>(
        factory: F,
        cfg: Config,
        init: impl FnOnce() + Send + 'static,
    ) -> Self {
        let status = Arc::new(Status::default());
        let shared = Arc::new(Shared {
            q: Mutex::new(Queue::default()),
            consumer: Condvar::new(),
            producer: Condvar::new(),
            hold: AtomicBool::new(false),
            queued: AtomicU64::new(0),
            cancel_epoch: AtomicU64::new(0),
            job_epoch: AtomicU64::new(0),
            status: Arc::clone(&status),
        });
        let engine = Engine::new(factory, cfg, status);
        let sh = Arc::clone(&shared);
        let thread = std::thread::spawn(move || {
            init();
            run(sh, engine)
        });
        Session { shared, thread: Some(thread) }
    }

    pub fn handle(&self) -> Handle {
        Handle { shared: Arc::clone(&self.shared) }
    }

    /// Queue a finished sentence. Blocks while the queue is full. Fails with
    /// [`Cancelled`] once a cancel has invalidated the running job.
    pub fn push(&self, chunk: Vec<f32>) -> anyhow::Result<()> {
        let n = chunk.len() as u64;
        self.enqueue(Msg::Chunk(chunk), n)
    }

    pub fn warm(&self) {
        // Never blocks a job on a full queue: a dropped Warm only delays the sink.
        let mut q = self.shared.lock();
        if q.chunks.len() < STREAM_BUFFER && !q.dead {
            let epoch = self.shared.job_epoch.load(Ordering::Acquire);
            q.chunks.push_back(Queued { msg: Msg::Warm, epoch });
            drop(q);
            self.shared.consumer.notify_one();
        }
    }

    fn enqueue(&self, msg: Msg, samples: u64) -> anyhow::Result<()> {
        let sh = &self.shared;
        let mut q = sh.lock();
        loop {
            if sh.job_epoch.load(Ordering::Acquire) != sh.cancel_epoch.load(Ordering::Acquire) {
                return Err(Cancelled.into());
            }
            if q.dead {
                anyhow::bail!("playback thread stopped early");
            }
            if q.chunks.len() < STREAM_BUFFER {
                break;
            }
            q = sh.producer.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        sh.queued.fetch_add(samples, Ordering::Release);
        let epoch = sh.job_epoch.load(Ordering::Acquire);
        q.chunks.push_back(Queued { msg, epoch });
        drop(q);
        sh.consumer.notify_one();
        Ok(())
    }

    pub fn hold(&self) {
        self.shared.hold.store(true, Ordering::Release);
    }

    pub fn release(&self) {
        self.shared.hold.store(false, Ordering::Release);
        // Take the lock so the thread cannot miss the wakeup between its check
        // of `hold` and its wait.
        drop(self.shared.lock());
        self.shared.consumer.notify_one();
    }

    /// Seconds queued but not yet written (includes audio a rewind put back).
    pub fn buffered_secs(&self) -> f64 {
        let q = self.shared.queued.load(Ordering::Acquire);
        let p = self.shared.status.playhead.load(Ordering::Acquire);
        q.saturating_sub(p) as f64 / SR as f64
    }

    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| t.is_finished())
    }

    /// Mark the running job with the current cancel epoch. The worker calls this
    /// when it starts a job; a later cancel makes that job's pushes fail.
    pub fn start_job(&self, epoch: u64) {
        self.shared.job_epoch.store(epoch, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.shared.job_epoch.load(Ordering::Acquire)
            != self.shared.cancel_epoch.load(Ordering::Acquire)
    }

    /// Close the queue, let playback finish, join the thread.
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.shared.hold.store(false, Ordering::Release);
        self.shared.lock().closed = true;
        self.shared.consumer.notify_all();
        if let Some(t) = self.thread.take() {
            t.join().map_err(|_| anyhow::anyhow!("playback thread panicked"))?;
        }
        Ok(())
    }
}

impl Handle {
    /// The current cancel epoch; stamp a job with it when it is queued.
    pub fn epoch(&self) -> u64 {
        self.shared.cancel_epoch.load(Ordering::Acquire)
    }

    /// Paused or holding history: the daemon must not idle-exit.
    pub fn busy(&self) -> bool {
        self.shared.status.paused.load(Ordering::Acquire)
            || self.shared.status.history.load(Ordering::Acquire)
    }

    /// Send a control to the playback thread and wait for its answer.
    pub fn control(&self, ctl: Control) -> Result<Outcome, String> {
        let sh = &self.shared;
        let (tx, rx) = mpsc::channel();
        {
            let mut q = sh.lock();
            if q.dead {
                return Err("playback thread stopped".into());
            }
            if ctl == Control::Cancel {
                // Invalidate the running and queued jobs before anything else,
                // under the lock so no push slips in between.
                sh.cancel_epoch.fetch_add(1, Ordering::AcqRel);
            }
            q.ctls.push_back(CtlReq { ctl, reply: tx });
        }
        if ctl == Control::Cancel {
            sh.producer.notify_all();
        }
        sh.consumer.notify_one();
        rx.recv_timeout(CONTROL_TIMEOUT).unwrap_or_else(|_| Err("playback busy".into()))
    }
}

fn run<F: SinkFactory>(sh: Arc<Shared>, mut eng: Engine<F>) {
    let _dead = DeadOnDrop(Arc::clone(&sh));
    enum Work {
        Ctl(CtlReq),
        Write,
        Ingest(Queued),
        Tick,
        Exit,
    }
    loop {
        let work = {
            let mut q = sh.lock();
            loop {
                if let Some(r) = q.ctls.pop_front() {
                    break Work::Ctl(r);
                }
                if eng.wants_write() {
                    break Work::Write;
                }
                if eng.can_ingest() {
                    if let Some(m) = q.chunks.pop_front() {
                        sh.producer.notify_one();
                        break Work::Ingest(m);
                    }
                }
                if q.closed && (eng.is_paused() || (q.chunks.is_empty() && !eng.wants_write())) {
                    break Work::Exit;
                }
                let held = sh.hold.load(Ordering::Acquire);
                match eng.next_deadline(held) {
                    None => q = sh.consumer.wait(q).unwrap_or_else(|e| e.into_inner()),
                    Some(d) => {
                        let now = Instant::now();
                        if d <= now {
                            break Work::Tick;
                        }
                        q = sh.consumer.wait_timeout(q, d - now).unwrap_or_else(|e| e.into_inner()).0;
                    }
                }
            }
        };
        match work {
            Work::Exit => {
                eng.shutdown();
                return;
            }
            Work::Tick => eng.tick(Instant::now(), sh.hold.load(Ordering::Acquire)),
            Work::Write => {
                eng.write_slice(Instant::now());
                eng.touch(Instant::now());
            }
            Work::Ingest(m) => match m.msg {
                Msg::Warm => eng.warm(Instant::now()),
                Msg::Chunk(c) => eng.ingest(&c, Instant::now()),
            },
            Work::Ctl(req) => {
                let held = sh.hold.load(Ordering::Acquire);
                let res = eng.apply(req.ctl, Instant::now(), held);
                if req.ctl == Control::Cancel {
                    // Drop what the cancelled jobs queued; keep newer ones.
                    let mut q = sh.lock();
                    let cur = sh.cancel_epoch.load(Ordering::Acquire);
                    q.chunks.retain(|c| c.epoch >= cur);
                    let kept: u64 = q
                        .chunks
                        .iter()
                        .map(|c| match &c.msg {
                            Msg::Chunk(v) => v.len() as u64,
                            Msg::Warm => 0,
                        })
                        .sum();
                    sh.queued.store(eng.end() + kept, Ordering::Release);
                    drop(q);
                    sh.producer.notify_all();
                }
                let _ = req.reply.send(res);
            }
        }
    }
}

// ------------------------------- settings ------------------------------------

/// Parse a positive number of seconds; anything else is `None`.
pub fn parse_secs(val: Option<&str>) -> Option<f64> {
    val.and_then(|s| s.trim().parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0)
}

impl Config {
    /// Defaults overridden by `RYK_SINK_IDLE_MS` (passed in), `RYK_HISTORY_SECS`,
    /// `RYK_REWIND_WINDOW_SECS`, `RYK_PAUSE_MAX_SECS` and `RYK_BACK_RESTART_SECS`.
    pub fn from_env(sink_idle: Duration) -> Self {
        let get = |k: &str| std::env::var(k).ok();
        let mut c = Config { sink_idle, ..Config::default() };
        if let Some(s) = parse_secs(get("RYK_HISTORY_SECS").as_deref()) {
            c.played_cap = secs_to_samples(s);
        }
        if let Some(s) = parse_secs(get("RYK_REWIND_WINDOW_SECS").as_deref()) {
            c.window = Duration::from_secs_f64(s);
        }
        if let Some(s) = parse_secs(get("RYK_PAUSE_MAX_SECS").as_deref()) {
            c.pause_max = Duration::from_secs_f64(s);
        }
        if let Some(s) = parse_secs(get("RYK_BACK_RESTART_SECS").as_deref()) {
            c.restart_after = secs_to_samples(s);
        }
        c
    }
}

// -------------------------------- tests --------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// `n` samples that all equal `v`, so a test can tell sentences apart.
    fn chunk(v: f32, n: usize) -> Vec<f32> {
        vec![v; n]
    }

    fn hist(cap_secs: f64) -> PcmHistory {
        PcmHistory::new(secs_to_samples(cap_secs), secs_to_samples(1.0))
    }

    // ---- PcmHistory ----

    #[test]
    fn append_records_one_mark_per_chunk() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 100));
        h.append(&chunk(2.0, 50));
        h.append(&[]);
        assert_eq!(h.end(), 150);
        assert_eq!(h.sentence_start(0), 0);
        assert_eq!(h.sentence_start(99), 0);
        assert_eq!(h.sentence_start(100), 100);
        assert_eq!(h.sentence_start(149), 100);
    }

    #[test]
    fn peek_and_commit_walk_forward() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 10));
        h.append(&chunk(2.0, 10));
        assert_eq!(h.peek(4), &[1.0; 4]);
        h.commit(4);
        assert_eq!(h.playhead(), 4);
        assert_eq!(h.peek(100).len(), 16);
        h.commit(100);
        assert_eq!(h.playhead(), 20);
        assert_eq!(h.unplayed(), 0);
    }

    #[test]
    fn seek_clamps_to_retained_range() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 100));
        h.commit(100);
        assert_eq!(h.seek(40), 40);
        assert_eq!(h.seek(10_000), 100);
    }

    #[test]
    fn played_cap_limits_how_far_back_you_can_go() {
        // cap = 1 s = 24000 samples; one 5 s chunk played fully
        let mut h = hist(1.0);
        h.append(&chunk(1.0, 120_000));
        h.commit(120_000);
        assert_eq!(h.retain_from(), 120_000 - 24_000);
        assert_eq!(h.seek(0), 96_000);
    }

    #[test]
    fn compaction_is_lazy_and_keeps_absolute_numbers() {
        let mut h = hist(1.0);
        h.append(&chunk(1.0, 24_000 * 5));
        h.commit(24_000); // retain_from still 0
        h.commit(24_000 * 2);
        // 3 s played, cap 1 s -> 2 s droppable, compact_after 1 s -> compacted
        let rf = h.retain_from();
        assert_eq!(rf, 24_000 * 2);
        assert_eq!(h.seek(0), rf);
        assert_eq!(h.end(), 24_000 * 5);
        assert_eq!(h.peek(3).len(), 3);
    }

    #[test]
    fn sentence_navigation_respects_the_floor() {
        let mut h = hist(1.0);
        for i in 0..4 {
            h.append(&chunk(i as f32, 24_000)); // four 1 s sentences
        }
        h.commit(24_000 * 4);
        // cap 1 s: only the last second is rewindable
        assert_eq!(h.retain_from(), 24_000 * 3);
        assert_eq!(h.sentence_start(24_000 * 4 - 1), 24_000 * 3);
        assert_eq!(h.prev_sentence_start(24_000 * 3), None);
    }

    #[test]
    fn prev_sentence_start_finds_the_one_before() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 100));
        h.append(&chunk(2.0, 100));
        h.append(&chunk(3.0, 100));
        assert_eq!(h.prev_sentence_start(200), Some(100));
        assert_eq!(h.prev_sentence_start(100), Some(0));
        assert_eq!(h.prev_sentence_start(0), None);
    }

    #[test]
    fn truncate_after_chunk_keeps_the_sentence_being_heard() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 100));
        h.append(&chunk(2.0, 100));
        h.append(&chunk(3.0, 100));
        h.commit(150);
        h.truncate_after_chunk(150);
        assert_eq!(h.end(), 200);
        assert_eq!(h.sentence_start(199), 100);
        h.append(&chunk(4.0, 10));
        assert_eq!(h.end(), 210);
        assert_eq!(h.sentence_start(205), 200);
    }

    #[test]
    fn clear_forgets_but_keeps_numbering() {
        let mut h = hist(60.0);
        h.append(&chunk(1.0, 100));
        h.commit(100);
        h.clear();
        assert!(!h.has_audio());
        assert_eq!(h.end(), 100);
        h.append(&chunk(2.0, 10));
        assert_eq!(h.playhead(), 100);
        assert_eq!(h.end(), 110);
        assert_eq!(h.peek(10), &[2.0; 10]);
    }

    // ---- fake sink ----

    /// Records every byte written and every kill/finish, in order.
    #[derive(Default)]
    struct Log {
        /// First sample of each write slice, as the value of its first f32.
        writes: Vec<f32>,
        spawns: usize,
        kills: usize,
        finishes: usize,
        fail_spawn: bool,
    }

    type SharedLog = Arc<StdMutex<Log>>;

    struct FakeSink(SharedLog);

    impl Sink for FakeSink {
        fn write(&mut self, pcm: &[u8]) -> io::Result<()> {
            let s: &[f32] = bytemuck::cast_slice(pcm);
            self.0.lock().unwrap().writes.push(s[0]);
            Ok(())
        }
        fn finish(self: Box<Self>) -> anyhow::Result<()> {
            self.0.lock().unwrap().finishes += 1;
            Ok(())
        }
        fn kill(self: Box<Self>) {
            self.0.lock().unwrap().kills += 1;
        }
    }

    struct FakeFactory {
        log: SharedLog,
        lag: u64,
    }

    impl SinkFactory for FakeFactory {
        fn spawn(&mut self) -> anyhow::Result<Box<dyn Sink>> {
            let mut l = self.log.lock().unwrap();
            if l.fail_spawn {
                anyhow::bail!("no sink");
            }
            l.spawns += 1;
            Ok(Box::new(FakeSink(Arc::clone(&self.log))))
        }
        fn lag_samples(&self) -> u64 {
            self.lag
        }
    }

    fn engine(lag: u64) -> (Engine<FakeFactory>, SharedLog, Instant) {
        engine_cfg(lag, Config::default())
    }

    fn engine_cfg(lag: u64, cfg: Config) -> (Engine<FakeFactory>, SharedLog, Instant) {
        let log: SharedLog = Arc::default();
        let eng = Engine::new(FakeFactory { log: Arc::clone(&log), lag }, cfg, Arc::default());
        (eng, log, Instant::now())
    }

    /// Ingest `values` as sentences of `len` samples each.
    fn feed(e: &mut Engine<FakeFactory>, values: &[f32], len: usize, t: Instant) {
        for &v in values {
            e.ingest(&chunk(v, len), t);
        }
    }

    /// Write slices until `n` samples have gone out.
    fn play(e: &mut Engine<FakeFactory>, samples: usize, t: Instant) {
        let slices = samples.div_ceil(WRITE_SLICE);
        for _ in 0..slices {
            if e.wants_write() {
                e.write_slice(t);
            }
        }
    }

    // ---- Engine: controls ----

    const S: usize = 24_000; // one second

    #[test]
    fn plays_a_sentence_in_slices_from_the_start() {
        let (mut e, log, t) = engine(0);
        feed(&mut e, &[1.0], 3 * S, t);
        while e.wants_write() {
            e.write_slice(t);
        }
        let l = log.lock().unwrap();
        assert_eq!(l.spawns, 1);
        assert_eq!(l.writes.len(), 30); // 3 s / 100 ms
        assert!(l.writes.iter().all(|&v| v == 1.0));
        assert_eq!(e.playhead(), 3 * S as u64);
    }

    #[test]
    fn back_in_the_middle_restarts_the_sentence() {
        let (mut e, log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 6 * S, t); // 2 s into sentence 2
        let out = e.apply(Control::Back, t, true).unwrap();
        assert_eq!(out.state, PlayState::Playing);
        assert_eq!(out.note.as_deref(), Some("back"));
        assert_eq!(e.playhead(), 4 * S as u64); // start of sentence 2
        assert_eq!(log.lock().unwrap().kills, 1);
    }

    #[test]
    fn back_right_after_a_sentence_started_goes_to_the_previous_one() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 4 * S + S / 2, t); // 0.5 s into sentence 2
        e.apply(Control::Back, t, true).unwrap();
        assert_eq!(e.playhead(), 0);
    }

    #[test]
    fn repeated_back_walks_one_sentence_each() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0, 3.0], 4 * S, t);
        play(&mut e, 12 * S, t); // all played
        // at the end of 3: 4 s in -> restart 3
        e.apply(Control::Back, t, false).unwrap();
        assert_eq!(e.playhead(), 8 * S as u64);
        // we are 0 s into 3 now -> previous
        e.apply(Control::Back, t, false).unwrap();
        assert_eq!(e.playhead(), 4 * S as u64);
        e.apply(Control::Back, t, false).unwrap();
        assert_eq!(e.playhead(), 0);
        let out = e.apply(Control::Back, t, false).unwrap();
        assert_eq!(out.note.as_deref(), Some("at-start"));
        assert_eq!(e.playhead(), 0);
    }

    #[test]
    fn back_after_everything_has_played_replays_the_last_sentence() {
        let (mut e, log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 8 * S, t);
        assert!(!e.wants_write());
        e.apply(Control::Back, t, false).unwrap();
        assert!(e.wants_write());
        while e.wants_write() {
            e.write_slice(t);
        }
        let l = log.lock().unwrap();
        // 80 slices of the first pass, then 40 of sentence 2 again
        assert_eq!(l.writes.len(), 80 + 40);
        assert!(l.writes[80..].iter().all(|&v| v == 2.0));
    }

    #[test]
    fn back_with_nothing_retained_is_an_error() {
        let (mut e, _log, t) = engine(0);
        assert_eq!(e.apply(Control::Back, t, false), Err("nothing playing".to_string()));
    }

    #[test]
    fn rewind_goes_back_by_seconds_and_clamps() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0], 10 * S, t);
        play(&mut e, 10 * S, t);
        let out = e.apply(Control::Rewind(4.0), t, false).unwrap();
        assert_eq!(out.note.as_deref(), Some("rewound 4.0"));
        assert_eq!(e.playhead(), 6 * S as u64);
        let out = e.apply(Control::Rewind(100.0), t, false).unwrap();
        assert_eq!(out.note.as_deref(), Some("rewound 6.0"));
        assert_eq!(e.playhead(), 0);
        // already at the start: no kill, no movement
        let kills = _log.lock().unwrap().kills;
        let out = e.apply(Control::Rewind(5.0), t, false).unwrap();
        assert_eq!(out.note.as_deref(), Some("rewound 0.0"));
        assert_eq!(_log.lock().unwrap().kills, kills);
    }

    #[test]
    fn pause_rolls_back_by_the_sink_lag_and_resume_snaps_to_the_sentence() {
        let lag = S as u64 / 4; // 0.25 s
        let (mut e, log, t) = engine(lag);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 4 * S + S, t); // 1 s into sentence 2, written
        let out = e.apply(Control::Pause, t, true).unwrap();
        assert_eq!(out.state, PlayState::Paused);
        assert_eq!(e.playhead(), (5 * S) as u64 - lag);
        assert_eq!(log.lock().unwrap().kills, 1);
        assert!(!e.wants_write()); // paused: nothing is written
        e.apply(Control::Resume, t, true).unwrap();
        // 0.75 s into the sentence -> within 2 s of its start -> snap to the start
        assert_eq!(e.playhead(), (4 * S) as u64);
        assert!(e.wants_write());
    }

    #[test]
    fn resume_far_into_a_sentence_replays_one_second() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0], 10 * S, t);
        play(&mut e, 6 * S, t);
        e.apply(Control::Pause, t, true).unwrap();
        assert_eq!(e.playhead(), 6 * S as u64);
        e.apply(Control::Resume, t, true).unwrap();
        assert_eq!(e.playhead(), 5 * S as u64);
    }

    #[test]
    fn pause_keeps_synthesized_audio_and_resume_loses_nothing() {
        let (mut e, log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, S, t);
        e.apply(Control::Pause, t, true).unwrap();
        // a sentence arrives while paused (playhead < end, so it is not ingested
        // by the real loop, but ingest itself must not break)
        e.apply(Control::Resume, t, true).unwrap();
        while e.wants_write() {
            e.write_slice(t);
        }
        let l = log.lock().unwrap();
        assert_eq!(l.spawns, 2); // killed on pause, respawned on resume
        assert_eq!(l.writes.iter().filter(|&&v| v == 2.0).count(), 40);
    }

    #[test]
    fn toggle_flips_between_pause_and_resume() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0], 4 * S, t);
        play(&mut e, S, t);
        assert_eq!(e.apply(Control::Toggle, t, true).unwrap().state, PlayState::Paused);
        assert_eq!(e.apply(Control::Toggle, t, true).unwrap().state, PlayState::Playing);
    }

    #[test]
    fn pause_and_toggle_when_idle_say_nothing_is_playing() {
        let (mut e, _log, t) = engine(0);
        for c in [Control::Pause, Control::Resume, Control::Toggle] {
            assert_eq!(e.apply(c, t, false), Err("nothing playing".to_string()), "{c:?}");
        }
        assert_eq!(e.apply(Control::Status, t, false).unwrap().state, PlayState::Idle);
    }

    #[test]
    fn a_pause_between_two_sentences_still_pauses() {
        // everything written, the job still synthesizing the next sentence
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0], 2 * S, t);
        play(&mut e, 2 * S, t);
        assert_eq!(e.apply(Control::Status, t + Duration::from_secs(5), true).unwrap().state, PlayState::Playing);
        assert_eq!(e.apply(Control::Pause, t, true).unwrap().state, PlayState::Paused);
        // the next sentence is ingested while paused and waits
        e.ingest(&chunk(2.0, S), t);
        assert!(!e.wants_write());
        e.apply(Control::Resume, t, true).unwrap();
        assert!(e.wants_write());
    }

    #[test]
    fn back_while_paused_resumes_from_the_earlier_sentence() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 4 * S + S / 2, t);
        e.apply(Control::Pause, t, true).unwrap();
        let out = e.apply(Control::Back, t, true).unwrap();
        assert_eq!(out.state, PlayState::Playing);
        assert!(!e.is_paused());
        assert_eq!(e.playhead(), 0);
    }

    #[test]
    fn cancel_drops_the_rest_but_keeps_the_heard_sentence_for_back() {
        let (mut e, log, t) = engine(0);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, 4 * S + S, t); // 1 s into sentence 2
        let out = e.apply(Control::Cancel, t, true).unwrap();
        assert_eq!(out.state, PlayState::Idle);
        assert!(!e.wants_write());
        assert_eq!(log.lock().unwrap().kills, 1);
        // nothing after sentence 2 exists; sentence 2 is whole and the playhead
        // sits at its end
        assert_eq!(e.end(), 8 * S as u64);
        assert_eq!(e.playhead(), 8 * S as u64);
        // back replays the sentence that was being heard, from its start
        e.apply(Control::Back, t, false).unwrap();
        assert_eq!(e.playhead(), 4 * S as u64);
        assert!(e.wants_write());
    }

    #[test]
    fn cancel_truncates_later_sentences() {
        let (mut e, _log, t) = engine(0);
        // three sentences ingested while the player is still on the first
        feed(&mut e, &[1.0, 2.0, 3.0], 4 * S, t);
        play(&mut e, S, t);
        e.apply(Control::Cancel, t, true).unwrap();
        assert_eq!(e.end(), 4 * S as u64);
    }

    #[test]
    fn long_pause_becomes_a_cancel() {
        let cfg = Config { pause_max: Duration::from_secs(60), ..Config::default() };
        let (mut e, _log, t) = engine_cfg(0, cfg);
        feed(&mut e, &[1.0, 2.0], 4 * S, t);
        play(&mut e, S, t);
        e.apply(Control::Pause, t, true).unwrap();
        e.tick(t + Duration::from_secs(30), true);
        assert!(e.is_paused());
        e.tick(t + Duration::from_secs(61), true);
        assert!(!e.is_paused());
        assert!(!e.wants_write());
    }

    // ---- Engine: sink lifetime and the rewind window ----

    #[test]
    fn idle_sink_closes_but_history_stays_for_the_window() {
        let cfg = Config {
            sink_idle: Duration::from_secs(30),
            window: Duration::from_secs(600),
            ..Config::default()
        };
        let (mut e, log, t) = engine_cfg(0, cfg);
        feed(&mut e, &[1.0], 2 * S, t);
        play(&mut e, 2 * S, t);
        e.tick(t + Duration::from_secs(10), false);
        assert!(e.has_sink());
        e.tick(t + Duration::from_secs(31), false);
        assert!(!e.has_sink());
        assert_eq!(log.lock().unwrap().finishes, 1);
        // rewind still works after the sink has closed, and respawns it
        e.apply(Control::Back, t + Duration::from_secs(60), false).unwrap();
        while e.wants_write() {
            e.write_slice(t + Duration::from_secs(60));
        }
        assert_eq!(log.lock().unwrap().spawns, 2);
        // window over: history is gone
        e.tick(t + Duration::from_secs(1000), false);
        assert_eq!(e.apply(Control::Back, t + Duration::from_secs(1000), false), Err("nothing playing".to_string()));
    }

    #[test]
    fn a_held_sink_stays_open() {
        let (mut e, _log, t) = engine(0);
        feed(&mut e, &[1.0], 2 * S, t);
        play(&mut e, 2 * S, t);
        e.tick(t + Duration::from_secs(120), true);
        assert!(e.has_sink());
        e.tick(t + Duration::from_secs(120), false);
        assert!(!e.has_sink());
    }

    #[test]
    fn deadlines_follow_the_state() {
        let (mut e, _log, t) = engine(0);
        assert_eq!(e.next_deadline(false), None);
        feed(&mut e, &[1.0], 2 * S, t);
        assert_eq!(e.next_deadline(false), None); // unplayed audio: busy writing
        play(&mut e, 2 * S, t);
        let d = e.next_deadline(false).unwrap();
        assert_eq!(d, t + Duration::from_secs(30)); // sink idle comes first
        assert_eq!(e.next_deadline(true), None); // held: wait for release
    }

    #[test]
    fn spawn_failure_drops_the_audio_instead_of_looping() {
        let (mut e, log, t) = engine(0);
        log.lock().unwrap().fail_spawn = true;
        feed(&mut e, &[1.0], 2 * S, t);
        e.write_slice(t);
        assert!(!e.wants_write());
    }

    // ---- Session: the thread and the queue ----

    struct SlowFactory(FakeFactory);

    impl SinkFactory for SlowFactory {
        fn spawn(&mut self) -> anyhow::Result<Box<dyn Sink>> {
            self.0.spawn()
        }
        fn lag_samples(&self) -> u64 {
            0
        }
    }

    fn session() -> (Session, SharedLog) {
        let log: SharedLog = Arc::default();
        let cfg = Config { sink_idle: Duration::from_millis(200), ..Config::default() };
        let s = Session::spawn(SlowFactory(FakeFactory { log: Arc::clone(&log), lag: 0 }), cfg);
        (s, log)
    }

    fn wait_for(mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn session_plays_everything_pushed_then_finishes() {
        let (s, log) = session();
        s.push(chunk(1.0, S / 2)).unwrap();
        s.push(chunk(2.0, S / 2)).unwrap();
        s.finish().unwrap();
        let l = log.lock().unwrap();
        assert_eq!(l.writes.len(), 10);
        assert_eq!(l.finishes, 1);
    }

    #[test]
    fn session_buffered_secs_tracks_the_playhead_and_a_rewind_replays() {
        let (s, log) = session();
        let h = s.handle();
        s.hold();
        s.push(chunk(1.0, 2 * S)).unwrap();
        wait_for(|| s.buffered_secs() < 0.01);
        let before = log.lock().unwrap().writes.len();
        h.control(Control::Rewind(1.0)).unwrap();
        // the replayed second (10 slices) goes out again, then the queue is empty
        wait_for(|| log.lock().unwrap().writes.len() >= before + 10);
        wait_for(|| s.buffered_secs() < 0.01);
        s.release();
        s.finish().unwrap();
    }

    #[test]
    fn session_control_round_trip() {
        let (s, _log) = session();
        let h = s.handle();
        assert_eq!(h.control(Control::Status).unwrap().state, PlayState::Idle);
        assert_eq!(h.control(Control::Back), Err("nothing playing".to_string()));
        s.hold();
        s.push(chunk(1.0, 2 * S)).unwrap();
        wait_for(|| s.buffered_secs() < 0.01);
        assert!(h.control(Control::Pause).unwrap().state == PlayState::Paused);
        assert!(h.busy());
        s.release();
        h.control(Control::Cancel).unwrap();
        s.finish().unwrap();
    }

    #[test]
    fn cancel_stops_the_running_job_and_drops_queued_chunks() {
        let (s, log) = session();
        let h = s.handle();
        s.hold();
        s.start_job(h.epoch());
        s.push(chunk(1.0, S)).unwrap();
        wait_for(|| s.buffered_secs() < 0.01);
        // pause so queued sentences cannot be taken yet
        h.control(Control::Pause).unwrap();
        s.push(chunk(2.0, S)).unwrap();
        s.push(chunk(3.0, S)).unwrap();
        assert!(!s.is_cancelled());
        h.control(Control::Cancel).unwrap();
        assert!(s.is_cancelled());
        let err = s.push(chunk(4.0, S)).unwrap_err();
        assert!(err.downcast_ref::<Cancelled>().is_some());
        // a job started after the cancel plays normally
        s.start_job(h.epoch());
        assert!(!s.is_cancelled());
        let before = log.lock().unwrap().writes.len();
        s.push(chunk(5.0, S / 2)).unwrap();
        wait_for(|| log.lock().unwrap().writes.len() >= before + 5);
        s.release();
        s.finish().unwrap();
        let l = log.lock().unwrap();
        assert!(!l.writes.contains(&2.0) && !l.writes.contains(&3.0) && !l.writes.contains(&4.0));
        assert!(l.writes.contains(&5.0));
    }

    #[test]
    fn push_blocked_on_a_full_queue_is_released_by_a_cancel() {
        let (s, _log) = session();
        let h = s.handle();
        s.hold();
        s.start_job(h.epoch());
        s.push(chunk(1.0, S)).unwrap();
        wait_for(|| s.buffered_secs() < 0.01);
        h.control(Control::Pause).unwrap();
        let s = Arc::new(s);
        // fill the queue (the engine takes one chunk, so STREAM_BUFFER + 1 pushes)
        for _ in 0..=STREAM_BUFFER {
            s.push(chunk(2.0, 10)).unwrap();
        }
        let s2 = Arc::clone(&s);
        let t = std::thread::spawn(move || s2.push(chunk(3.0, 10)));
        std::thread::sleep(Duration::from_millis(100));
        h.control(Control::Cancel).unwrap();
        let res = t.join().unwrap();
        assert!(res.unwrap_err().downcast_ref::<Cancelled>().is_some());
    }

    #[test]
    fn controls_after_the_thread_exits_fail_cleanly() {
        let (s, _log) = session();
        let h = s.handle();
        s.finish().unwrap();
        assert!(h.control(Control::Status).is_err());
    }

    #[test]
    fn parse_secs_accepts_only_positive_finite_numbers() {
        assert_eq!(parse_secs(Some("1.5")), Some(1.5));
        for bad in [None, Some(""), Some("0"), Some("-1"), Some("nan"), Some("inf"), Some("x")] {
            assert_eq!(parse_secs(bad), None, "{bad:?}");
        }
    }
}
