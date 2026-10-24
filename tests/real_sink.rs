//! The daemon's playback against the real `pacat`, with a quiet synthetic tone
//! instead of the model. Needs PulseAudio and an output, so it is ignored by
//! default:
//!
//!   cargo test --test real_sink -- --ignored --nocapture --test-threads=1
//!
//! It plays about ten seconds of quiet tone and checks, by counting `pacat`
//! processes, that pause/back/cancel really stop and restart the sink.

use std::process::Command;
use std::time::{Duration, Instant};

use kukuryku::kokoro::{SAMPLE_RATE, StreamPlayer};
use kukuryku::playback::{Control, PlayState};

/// `secs` of a quiet sine at `hz`.
fn tone(hz: f32, secs: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * secs) as usize;
    (0..n)
        .map(|i| 0.03 * (2.0 * std::f32::consts::PI * hz * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

fn pacat_count() -> usize {
    Command::new("pgrep")
        .args(["-x", "pacat"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
        .unwrap_or(0)
}

fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn timed(h: &kukuryku::playback::Handle, c: Control) -> (kukuryku::playback::Outcome, Duration) {
    let t = Instant::now();
    let out = h.control(c).unwrap_or_else(|e| panic!("{c:?}: {e}"));
    (out, t.elapsed())
}

#[test]
#[ignore = "plays a few seconds of quiet tone through pacat"]
fn pause_resume_back_cancel_drive_the_real_sink() {
    let base = pacat_count();
    let player = StreamPlayer::with_sink_idle(Duration::from_secs(30)).unwrap();
    let h = player.handle().unwrap();
    player.hold_sink();
    player.start_job(h.epoch());
    player.warm();
    for hz in [330.0, 440.0, 550.0] {
        player.push(tone(hz, 3.0)).unwrap();
    }

    wait_for("pacat to start", || pacat_count() == base + 1);
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(h.control(Control::Status).unwrap().state, PlayState::Playing);
    let before = player.buffered_secs();

    let (out, took) = timed(&h, Control::Pause);
    println!("pause  -> {} in {took:?}", out.line());
    assert_eq!(out.state, PlayState::Paused);
    assert_eq!(pacat_count(), base, "pause must kill the sink");
    let after = player.buffered_secs();
    println!("buffered {before:.2}s -> {after:.2}s (rolled back by the sink lag)");
    assert!(after > before, "the playhead must move back");
    assert!(h.busy(), "a paused session keeps the daemon alive");

    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(pacat_count(), base, "nothing plays while paused");

    let (out, took) = timed(&h, Control::Toggle);
    println!("toggle -> {} in {took:?}", out.line());
    assert_eq!(out.state, PlayState::Playing);
    wait_for("pacat after resume", || pacat_count() == base + 1);
    std::thread::sleep(Duration::from_millis(800));

    let (out, took) = timed(&h, Control::Back);
    println!("back   -> {} in {took:?}", out.line());
    assert_eq!(out.state, PlayState::Playing);
    wait_for("pacat after back", || pacat_count() == base + 1);

    let (out, took) = timed(&h, Control::Cancel);
    println!("cancel -> {} in {took:?}", out.line());
    assert_eq!(out.state, PlayState::Idle);
    assert_eq!(pacat_count(), base, "cancel must kill the sink");
    assert!(player.is_cancelled());
    assert!(player.push(tone(660.0, 1.0)).is_err(), "a cancelled job cannot push");

    // The sentence that was playing is still there to replay.
    player.release_sink();
    let (out, took) = timed(&h, Control::Back);
    println!("back   -> {} in {took:?}", out.line());
    assert_eq!(out.state, PlayState::Playing);
    wait_for("pacat after back", || pacat_count() == base + 1);
    h.control(Control::Cancel).unwrap();
    assert_eq!(pacat_count(), base);
    player.finish().unwrap();
}
