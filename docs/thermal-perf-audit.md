# kukuryku — thermal/perf headroom audit (measurement-grounded rev)

## Context

Original ask: "deep research and look at the past commits and check what else can be improved to optimize overheating and keep performance on RTF ≈ 1.1 on the Android target." The first draft of this plan was based on code reading alone and anchored to an inherited baseline of "RTF ~1.1–1.2 on the 3-gold full pool." A measurement pass on the actual device (Galaxy S10, SD855) **falsified that baseline** and reshaped the recommendations. This rev replaces the plan with one anchored in numbers taken from the running binary on 2026-10-07/08.

### Device characterization (this S10 handset, measured)

- **SoC: Snapdragon 855** confirmed (`CPU implementer 0x51` Qualcomm; part `0x805` A55 on cpu0–3; part `0x804` A76 Kryo 485 Gold on cpu4–7). The plan's Exynos 9820 ambiguity is dropped.
- Every core advertises `asimdhp` + `asimddp` — FP16 and DOTPROD both available on both clusters.
- **Samsung firmware caps the gold cluster to 2016 MHz** permanently: `cpuinfo_max_freq` on cpu4=2419200, cpu7=2841600; `scaling_max_freq` on both=2016000. The current code's `gold_freq_capped()` returns `true` steady-state because `scaling_max < cpuinfo_max` — this is **not a thermal trip**, it is Samsung's always-on power policy. The real ceiling to design around is 2.016 GHz, not 2.42/2.84 GHz.
- **`lmh-dcvs-00/01` is a static sentinel = 75000** (75.0°C exact) regardless of actual silicon temperature. Live CPU-core zones (`cpu-1-*-usr`, `cpu-1-*-step`) read sensible 37–50°C. The code's `read_lmh_max_c()` reads only the sentinel path, so it reports 75°C always, equal to `THERM_HOT`.
- Combined, the two above make `update_thermal_hold` return `true` at the very first per-utterance check on this device. The governor logs this verbatim at idle on a cool phone (gold cores at 37–38°C):
  ```
  [kokoro] thermal hold on (lmh 75°C, gold capped true)
  [kokoro] governor Full -> Cool (buffered 0.00s)
  ```

### Measured baselines (same binary, same reference sentence, phone cool before each)

| Scenario | Infer RTF | Wall RTF | Notes |
|---|---|---|---|
| Default governor, short sentence (3.5s audio) | 1.555 | 1.700 | **Cool-pool floor fires at frame 0** |
| `KOKORO_GOVERNOR=off`, reference sentence 4× best-of | **1.349** | 1.362 | True full-pool, 3 threads on cpu4-6 |
| `KOKORO_GOVERNOR=off`, reference sentence 4× mean | 1.361 | 1.380 | σ ≈ 0.01 across the 4 runs |
| `KOKORO_GOVERNOR=off`, 5-sentence paragraph (power_bench default text) | — | **0.951** | Stage-1 lookahead overlaps across chunks |
| Thermal cost of 4× single-sentence runs | +5°C on `cpu-1-*-usr` (44 → 49°C) | — | Thermal is not the bottleneck |

### Stage-2 op share, `KOKORO_TRACT_PROFILE=1` on the reference sentence (gov off)

| Op | Time | Calls | % of stage-2 |
|---|---|---|---|
| OptMatMul | 6.435s | 254 | **58.3%** |
| Cast | 1.794s | 335 | **16.2%** |
| Snake | 0.706s | 48 | 6.4% |
| Resize | 0.337s | 6 | 3.0% |
| OptMulByScalar | 0.301s | 281 | 2.7% |
| OptSubByScalar | 0.256s | 66 | 2.3% |
| Scan | 0.191s | 3 | 1.7% |
| OptAddUnicast | 0.155s | 54 | 1.4% |
| OptAddByScalar | 0.152s | 210 | 1.4% |
| Reduce<SumOfSquares> | 0.119s | 65 | 1.1% |
| Reduce<Sum> | 0.092s | 65 | 0.8% |
| Gather | 0.076s | 5 | 0.7% |

Three ops account for **81%** of stage-2 time: GEMMs, Cast shuttles between f16 and f32, and Snake. The 16.2% going to Cast means the FP16 translate's narrow accept list is paying a per-boundary f32↔f16 shuttle tax that erodes the gain from f16 GEMM kernels.

### What's already landed (verified from source, not just the earlier plan's claims)

- FP16 cast at `into_optimized()` for stage-2 GEMMs (`src/tract_backend.rs:343-367`), gated by `tract_linalg::has_fp16()` and `KOKORO_TRACT_FP16`. **Runtime log confirms "FP16 GEMM is on"** on this device.
- Lazy im2col + Pad-fold + parallel patcher for vocoder convs (prior tiers).
- Snake / AdaIN scale-mul fusion before the FP16 translate.
- SinSq fusion + vectorised sin/cos.
- mimalloc global allocator.
- Stage-1 lookahead on the A55 little cluster (`src/tract_backend.rs:620`); stage-2 on A75/A76 gold mids; vocoder off the prime; 3-gold default via `KOKORO_TRACT_CPUSET=auto`. Stage-1 handoff via `sync_channel(1)` (`src/tract_backend.rs:1063-1072`).
- Thermal governor with hysteresis `THERM_HOT=75`, `THERM_COOL=68`, plus `cpu4 scaling_max_freq` cap detection (`src/tract_backend.rs:695-791`). **Reads a sensor that is static on this device** (see characterization above).
- Buffered-audio pacing `GOV_HI=6s` / `GOV_LO=2s` / `GOV_MAX_AHEAD=20s`.

### Build profile state (unchanged from earlier draft, re-verified)

- `[profile.release]` has only `opt-level = 3`. No LTO, no `codegen-units` pin.
- No `.cargo/config.toml` exists.
- No `rust-toolchain` pin.

---

## Headline reframe

- The RTF 1.1 target is **already met on paragraph-style input** (measured wall RTF 0.95 on a 5-sentence paragraph with governor off), because stage-1 lookahead on the little cluster overlaps with stage-2 on the golds across chunks.
- The RTF 1.1 target is **not met on single-chunk long utterances** (measured RTF 1.36 governor off, 1.5+ with the thermal governor's cool-pool trap). There is no stage-1 overlap available for a single chunk.
- The **largest single win on this device is fixing the governor's sensor path** so the default run of the binary isn't permanently stuck on the 2-thread cool pool. That alone takes default-run RTF from ~1.55 to ~1.36 with zero algorithmic change.
- The **next largest lever is Cast reduction** — 16.2% of stage-2 going to f16↔f32 shuttles is a direct, measurable target.
- Thermal is **not** the limiter here: 4 single-sentence runs produced +5°C on the gold cluster. The device's Samsung-firmware 2.016 GHz cap is already doing power management for us. Several items in the earlier draft (per-chunk polling, battery-saver detection) are deprioritized accordingly.

---

## Recommendations (measurement-grounded)

### (A) Must-fix — the thermal governor is misbehaving on this device

**A0. Replace `read_lmh_max_c()` with a hottest-core read of the live CPU zones.** `src/tract_backend.rs:751-775`. The `lmh-dcvs` zones are a static 75°C sentinel on this SoC — using them trips `thermal_hold` on a cool phone. Switch to the hottest of `cpu-1-*-usr` (the gold cluster; the ops that actually heat the chip run there), and verify with a 5× sample at idle that it varies before shipping. Live zones measured during this audit:
   ```
   cpu-1-0-usr=37.1   cpu-1-4-usr=36.7
   cpu-1-1-usr=37.5   cpu-1-5-usr=37.9
   cpu-1-2-usr=38.6   cpu-1-6-usr=38.6
   cpu-1-3-usr=37.5   cpu-1-7-usr=37.9
   ```
   Keep the fallback: if no `cpu-1-*-usr` zones exist, read `cpu-*-usr`, then nothing (return 0 → do not trip hold by itself).

**A1. Remove `gold_freq_capped()` from the trip condition** (`src/tract_backend.rs:742-748, 779-791`). On Samsung firmware the scaling ceiling is below `cpuinfo_max` **permanently**, not just when thermally capped — so the current logic forces `thermal_hold` on at boot and never releases. Either: (a) drop the signal entirely, or (b) baseline it: record `scaling_max_freq` at startup and only treat **further** reductions as a hot signal. (a) is simpler and loses nothing on devices where the kernel never writes it (Termux can't write it anyway).

**A2. Tune `THERM_HOT` / `THERM_COOL` for live core zones.** The current `75 / 68` numbers come from lmh-dcvs's trip semantics. For live `cpu-1-*-usr` reads, start with `THERM_HOT=78`, `THERM_COOL=72` and refine with power_bench runs. Rationale: this device's cores sit at 40–50°C under load, 55–65°C under sustained load; a 75°C trip would never fire from live zones (hiding real heat). The new hysteresis has to bracket the actual load range.

**Expected result after A0+A1+A2:** default-run RTF on single sentences improves from ~1.55 (cool-pool floor) to ~1.36 (true full pool), **matching the measured governor-off case**. This is the single largest win on this device.

### (B) Cast reduction — the next-largest lever

**B1. Widen the FP16 translator's accept list to cover scalar/elementwise neighbors of GEMMs.** `src/tract_backend.rs:343-367` + the `fp16_translate_name` filter. Right now the translate skips Snake / STFT / norms / trig, so every GEMM output has to Cast back to f32 before an adjacent `OptMulByScalar` / `OptSubByScalar` / `OptAddByScalar` runs, then Cast back to f16 for the next GEMM. 335 Cast calls dominate because of this. Candidates to add to the f16 accept list, in order of expected win:
   - `OptMulByScalar`, `OptAddByScalar`, `OptSubByScalar`, `OptAddUnicast` — scalar fusions already specialized by tract (together 7.8% of stage-2).
   - `Reduce<SumOfSquares>`, `Reduce<Sum>` — numerical risk; verify audio correlation after enabling.
   - **Keep in DENY:** Snake (branch cuts and trig need f32), STFT (phase accumulation), norm ε denominators (f16 ε can underflow).

   Each candidate must be validated independently against the reference WAV correlation (~0.976 vs ONNX Runtime). Add via name-pattern filter in `fp16_translate_name`; no need for a new mechanism.

**Expected result:** reclaiming even half of the Cast cost (16.2% → ~8%) drops stage-2 time by ~8%, which on a 20s single-sentence infer is ~1.6s → infer RTF 1.36 → ~1.25. Combined with A0–A2 this is the realistic path to RTF ~1.2 on single sentences.

### (C) Build-profile tuning — small wins, low risk

**C1. `[profile.release]` — add `lto = "thin"` and `codegen-units = 1`.** `Cargo.toml:107-108`. Keep `panic = "unwind"` (vendored tract is actively patched; backtraces matter). Expected 2–5% on cross-crate hot paths. Not `lto = "fat"` yet — ~3× compile time for marginal additional win.

**C2. `.cargo/config.toml` with `target-feature=+fp16,+dotprod` for aarch64-linux-android.** Both features already runtime-available on both clusters per `/proc/cpuinfo`; adding them at compile time lets the compiler emit SIMD without run-time dispatch overhead in non-hand-rolled paths. **Do not use `target-cpu=cortex-a76`** — stage-1 runs on A55; a cortex-a76 target would issue instructions the little cluster executes slowly. Expected 1–3% on code paths not already asm in tract-linalg.

**C3. `prctl(PR_SET_TIMERSLACK, 50_000, …)` on the speak thread.** `src/lib.rs` near line 831 (speak-thread setup). Android background apps default to 50 ms timerslack, coarsening the sleeps inside `wait_until_buffered_below`. 50 µs is standard foreground. One-liner, Android-only. Marginal effect on infer but tightens the pacing loop.

**C4. `madvise(MADV_WILLNEED)` on mmaped weights after load.** Prefaults ~160 MB of model weights so the first inference isn't paying first-touch costs under the stage-2 lock. One-shot startup effect only; doesn't change steady-state RTF.

### (D) Measurement-gated — do not ship blind

**D1. Alignment-matrix layout.** `src/tract_backend.rs:1039-1047` builds `align: [N, total_frames]` row-major. Alignment matmul is part of the 58.3% OptMatMul share. Whether a transpose helps depends on how the downstream OptMatMul packs its inputs. Dump the optimised plan (`TRACT_LOG=debug`) and look for an unpacked transpose or a non-strided pack before touching the builder. If tract already packs this input, a manual transpose is wasted work.

**D2. Pin stage-1 lookahead to one little core instead of the whole A55 cluster.** `src/tract_backend.rs:620`. EAS on kernel 4.14 migrates aggressively; a single pin preserves L1. Measure first — on this device the lookahead already overlaps enough to deliver RTF 0.95 on paragraphs, so the headroom here is small.

**D3. Opt-in `setpriority(PRIO_PROCESS, 0, -4)` on stage-2 workers.** Will `EACCES` on unrooted Termux without `CAP_SYS_NICE`. Behind warn-once. Only helps outlier chunks; of limited value given the device's hard Samsung cap.

### (E) Deferred — big project / wrong device for the measurement

- **PGO on-device.** cargo-pgo + NDK + representative on-device profile. Realistic 3–6%; most hot kernels are already asm.
- **Stage-1 int8 quantization with UDOT/SDOT.** 15–25% on stage 1 in theory; needs model re-training or QAT and new asset pipeline. High value / high cost. Reconsider after (A)+(B) land and sustained-RTF data motivates it.
- **BOLT post-link.** Same category as PGO.
- **Battery-saver detection.** The S10 Samsung firmware is already running a conservative power policy regardless (hence the 2.016 GHz cap). Informational-only; low value here.

---

## Explicit anti-recommendations (unchanged — still correct after measurement)

- **No `target-cpu=cortex-a76`** — stage-1 runs on A55.
- **No `panic = "abort"`** while vendored tract is actively patched.
- **No sidecar polling thread** for thermal — piggyback on existing per-chunk work when thermal signal is wired up correctly.
- **No `MALLOC_ARENA_MAX`** tuning — mimalloc ignores it.
- **No forced `MADV_HUGEPAGE`** on Samsung 4.14 kernel.
- **No default-on duty-cycle sleep** — would starve stage-1 lookahead via `sync_channel(1)`.
- **Do not revisit the 2-thread full-mid pool** — code's own comment says RTF ~1.45 vs ~1.22 on 3 golds (measured gap confirmed in this audit: 2-thread cool-pool runs at RTF 1.55+).
- **No f64 vocoder** — tried, produced ringing (commit `8ba589a`).

---

## Verification plan

Priority order for A/B after each change (all against this device, phone cool before each — gold cluster < 50°C per `cpu-1-*-usr`):

1. **Default-run RTF on the single reference sentence** — proves the governor fix lands without regressing anything else:
   ```
   target/release/ryk "<lighthouse reference sentence>" 2>&1 | grep '\[kokoro\] done:'
   ```
   Before A0–A2: wall RTF ~1.70 (cool-pool). After A0–A2: expected ~1.38 (matching governor-off case).

2. **Governor-off single-sentence infer** — proves (B)/(C) wins are real:
   ```
   KOKORO_GOVERNOR=off target/release/ryk "<lighthouse reference>" 2>&1 | grep done:
   ```
   Current: RTF 1.36. Target after (B1): ~1.25.

3. **Paragraph wall RTF with governor off** — sanity check that lookahead-overlap still delivers:
   ```
   KOKORO_GOVERNOR=off ./tools/power_bench.sh <label>
   ```
   Current: wall RTF 0.95. Expect to stay below 1.0.

4. **Stage-2 op profile** — verify Cast share actually drops after (B1):
   ```
   KOKORO_GOVERNOR=off KOKORO_TRACT_PROFILE=1 target/release/ryk "<reference>" 2>&1 | sed -n '/stage2 profile/,/\[1\/1\]/p'
   ```
   Current: Cast 16.2%. After each added fp16 accept-list entry, re-measure.

5. **Audio correctness** — reference WAV correlation ~0.976 target. Any PR that drops below ~0.97 needs a fidelity analysis (precedent: commit `a97f57f`, atan2 branch cut).

6. **Thermal cadence** — only once (A0) is in: run `power_bench.sh` on a long multi-sentence text and confirm the governor's "thermal hold on/off" log fires on real heat (gold zone > `THERM_HOT`) and not at idle. Not useful before A0 because the current sensor path can't produce this signal.

---

## Critical files

- `/data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs`
  - **751–775** — `read_lmh_max_c()` (A0 primary change: switch to `cpu-1-*-usr`).
  - **742–748, 779–791** — `update_thermal_hold()`, `gold_freq_capped()` (A1: drop gold cap from trip).
  - **695–701** — `THERM_HOT` / `THERM_COOL` constants (A2 retune).
  - **343–367** — `apply_fp16()` + `fp16_translate_name` filter (B1 accept-list widening).
  - **1039–1047** — alignment matrix builder (D1, pending plan dump).
  - **618–650** — stage-1 pinning helpers (D2).
- `/data/data/com.termux/files/home/PRJ/kukuryku/Cargo.toml` — `[profile.release]` (C1).
- `/data/data/com.termux/files/home/PRJ/kukuryku/.cargo/config.toml` — **to be created** for C2.
- `/data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs` — speak-thread setup near line 831 (C3).
- `/data/data/com.termux/files/home/PRJ/kukuryku/tools/power_bench.sh` — thermal + wall-RTF harness.
- `/data/data/com.termux/files/home/PRJ/kukuryku/tools/bench_conv.sh` — fixed-sentence infer harness (NB: triggers a tract-core rebuild if any tract crate's source mtime moves; for quick iteration run the binary directly).

---

## What this rev learned that the first draft didn't

- The quoted baseline "RTF 1.1–1.2 on the 3-gold full pool" is **wrong on this device**. Real governor-off single-sentence baseline is RTF 1.36.
- The thermal governor **does not actually fire from a thermal event on this SoC** — it fires from a sentinel sensor reading and a steady-state Samsung-firmware cap. On a cool phone it immediately traps every run to the cool pool.
- Thermal is **not** the current bottleneck — 4 passes warm the gold cluster only +5°C. The earlier plan's thermal-polling recommendations are solving a non-problem on this hardware (though they would still be correct on a device with a working lmh-dcvs sensor).
- The **Cast op share (16.2%)** was not visible until the profile was dumped. It is a bigger lever than any single build-flag tweak.
- RTF 1.1 **is already met on paragraphs** (wall RTF 0.95), via stage-1 lookahead overlap. The hard case is single-chunk long utterances where no overlap is available.
