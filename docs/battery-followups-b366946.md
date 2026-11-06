# Battery review follow-ups

Changes selected from `battery-review-b366946.md`, reviewed against `b366946`.
These reduce idle work and fix playback tracking. Subsequent
[live validation](battery-validation-8501a62.md) passed the software and control
checks and confirmed earlier audio shutdown; battery-energy savings remain
unproven.

| Review item | Implemented behavior |
| --- | --- |
| #1, helper affinity | On Android, the background helper build pins itself to little cores. A separate pinned launcher thread starts the helper so its JVM inherits that mask while the synthesis caller keeps its existing mask. No `pre_exec` hook is used. |
| #2, watcher timers | The last pending job and failed enqueue notify watchers. State watchers block on the change counter rather than 1 s / 30 s timers. Disconnect and thread-start failure wake and close the watch. State refresh and snapshots share one mutex; snapshots expose only state published with the change counter, so concurrent transitions cannot strand an indefinite waiter. Shutdown also publishes the released synthesis hold. |
| #3, audio tail | The shared sink grace defaults to 5 s, overridable with `RYK_SINK_IDLE_MS`. Synthesis still holds the sink across sentence gaps and warms it at job start. |
| #7, preparation cache | The embedded vocabulary is parsed once with character keys. Each preparation thread keeps only its latest voice, validated by file metadata on every use. Unix identity/change timestamps also detect replacements that preserve size and mtime. Failed reloads discard the cached voice. Token IDs and style values retain their previous behavior. |
| #9, Rayon cleanup | The row-outer matmul path uses one `pool.install` call. Its arithmetic and parallel iteration stay the same. |
| #11, playback position | A separate last-successful-write timestamp drives the audible cursor and audio tail. Warming or ingesting cannot revive a drained tail; failed writes do not create one, and a new sink starts with no write timestamp. |

The 600 s rewind window, pause lifetime, and daemon idle-exit policy are retained.
The accept loop still rechecks busy playback/history every 5 s. Replacing that
polling needs descriptor-based notifications and is deferred, along with the
speculative inference, frequency and first-audio experiments. The proposed 89°C
thermal floor is not applied.

The regression tests cover playback cursor/tail behavior, sink closure with
history retained, notification-driven watch closure and disconnects, pending-job
completion and failed enqueue, exact token/style preparation, voice edits,
replacement, switching, deletion and repair. Android tests use real child
processes to verify helper/compiler affinity and the unchanged caller mask.

Validation on Android:

- `cargo test --release --locked --offline --lib`: 156 passed, 1 ignored.
- Twenty further consecutive runs of the completed test binary: all passed,
  with 156 passing tests and 1 ignored in each run.
- `cargo build --release --locked --offline --bin ryk`: passed. The executable
  at `target/release/ryk` reports `ryk 0.1.0` and a 5000 ms sink default in `--help`.
- The installed `~/bin/ryk` SHA-256 is unchanged; the new build is not installed.
- Local repetition record: `tools/bench_out/battery-followups-b366946/validation.json`.

The follow-up speech, real audio/Java integration, live daemon/headset controls
and unplugged paired energy run are recorded in
[`battery-validation-8501a62.md`](battery-validation-8501a62.md). The user confirmed
the wired-headset and notification controls work. A detailed perceptual rating
of restart clicks and cancel-tone loudness was not collected. The five-second
grace closed the audio client/server about 25 s sooner; four energy pairs were
too noisy to establish savings. The installed binary remains unchanged.
