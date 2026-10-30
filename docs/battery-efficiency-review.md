# Battery Efficiency Review & Recommendations (main @ 4ece158)

## 1. Baseline & Physics on Galaxy S10e (SD855)

On Galaxy S10e (Snapdragon 855), `main` currently runs paragraph synthesis at **infer RTF `1.00–1.04`** and **wall RTF `0.80–0.86`** (Stage 1 overlaps on `cpu0–3` A55 little cores; Stage 2 runs on `cpu4–6` A76 gold cores at `~0.70×` realtime).

Because Samsung’s `schedutil` + `/top-app` `schedtune` policy pins the gold cluster (`cpu4–7`) to its `2016 MHz` cap whenever any gold thread is active, reducing thread count (3 → 2 gold cores) only stretches execution time at the same voltage and clock (`58.0 s` vs `57.8 s` CPU, RTF `~1.35`). Therefore, reducing **Joules per audio second** while keeping wall RTF $\le 1.0$ requires:
1. **Doing less CPU and DRAM work per audio second** (so the gold cluster returns to deep idle sooner),
2. **Releasing audio/daemon wakelocks and polling timers sooner**, and/or
3. **Lowering the gold cluster’s voltage/frequency ($V^2 f$)** to spend the `0.80 → 1.00` wall-RTF headroom.

---

## 2. High-Impact Compute & DRAM Optimizations (Less Work per Audio Second)

### A. Fix Unfused Bias `Add` + `Cast` Ping-Pong After `EinSum` (Stage 1 & Stage 2)
- **Root cause**: In [`Stage::build`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L88-L96), `typed.declutter()` runs before [`apply_fp16`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L422-L458). During declutter:
  - Every $1\times 1$ `Conv` with bias splits into `EinSum` + `TypedBinOp(Add)` in [`Conv::declutter_as_einsum`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/cnn/conv/conv.rs#L758-L780).
  - Every `Gemm` splits its bias into a separate `Add` in [`Gemm::wire`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-onnx/src/ops/math/gemm.rs#L62-L90).
  - Then [`fp16_translate_node`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L383-L416) translates `EinSum` to `f16`, but leaves the immediately following bias `Add` and shape-only [`AxisOp`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/change_axes.rs#L47-L53) nodes in `f32`.
  - [`FloatPrecisionTranslator`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/floats.rs#L24-L115) therefore inserts a `Cast(f16 -> f32)` **between** `EinSum` and its bias `Add`.
  - Later, in `into_optimized()`, [`OptMatMul::fuse`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/matmul/optimized.rs#L434-L608) attempts to fuse successor `Add` (bias) and `AxisOp` nodes into `OptMatMul` micro-ops (`ProtoFusedSpec::BinPerRow` / `BinPerCol` / `AddUnicast`), **but is blocked by the intervening `Cast(f16 -> f32)`**.
- **Impact**: Instead of fusing bias addition in-register inside `f16` `OptMatMul` and passing `f16` activations straight into the next `EinSum`/`Conv`, the graph writes `f16` to memory, runs single-threaded `Cast(f16 -> f32)` (**353 calls = 7.2% of Stage 2 time**), executes a separate `f32` `OptAdd*` pass over memory, and casts `f32 -> f16` again at the next layer.
- **Fix**:
  - In [`fp16_translate_node`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L383-L416), also translate `AxisOp` (`AddAxis`, `RmAxis`, `MoveAxis`, `Reshape`) and `Add` nodes whose non-constant input comes from a translated `EinSum`/`Conv`/`Gemm` (unlike the earlier reverted experiment that translated *all* isolated binary ops in `f32` subgraphs, restricting translation to `EinSum`-attached bias `Add`s and `AxisOp` chains strictly removes `Cast` boundaries and enables [`OptMatMul::fuse`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/matmul/optimized.rs#L434-L608) to absorb the bias into the `f16` kernel epilogue).
  - Or teach [`Cast::declutter`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/cast.rs#L75-L120) to commute `Cast` past `AxisOp` and constant bias `Add`.

### B. Enable FP16 Inside `Scan` (BiLSTM)
- **Root cause**: `Scan` accounts for **16.4% of Stage 1** and **2.1% of Stage 2** ([docs/fp16-einsum.md](file:///data/data/com.termux/files/home/PRJ/kukuryku/docs/fp16-einsum.md#L247-L249)). [`FloatPrecisionTranslator::translate_node`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/floats.rs#L118-L165) already supports recursing into `Scan.body`, and [`arm64.rs`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-linalg/src/arm64.rs#L348-L375) already registers ARMv8.2 NEON FP16 kernels for `mmv_f16` (`128x1`), `mmm_f16`, `sigmoid_f16`, and `tanh_f16`. However, `"Scan"` is omitted from `ACCEPT_OPS` in [`fp16_translate_node`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L414), so `should_translate_node` returns `false` on the outer `Scan` node and every LSTM runs in `f32`.
- **Fix**: Add `"Scan"` to `ACCEPT_OPS` in [`fp16_translate_node`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L414) (or translate the inner `Scan.body` with the same `fp16_translate_node` predicate so gate `EinSum`s run in `f16` while keeping recurrence state accuracy checkable via `KOKORO_FP16=2` parity). This halves LSTM weight bandwidth and cuts Stage 1 little-core CPU time.

### C. Cache `LazyIm2colParams` Across Same-Shape Convolutions
- **Root cause**: Under the symbolic Stage 2 plan, [`LazyIm2Col::eval`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/cnn/conv/lazy_im2col.rs#L205-L208) calls [`LazyDeferred::build`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/cnn/conv/lazy_im2col.rs#L94-L168) on every conv execution (~74 times per sentence), heap-allocating fresh `n_byte_offsets: Vec<isize>` (up to ~70k `isize` = 560 KB) and `k_byte_offsets: Vec<isize>` every time, even when the input shape is unchanged across steps or shared across convs in a resblock.
- **Fix**: Cache the last `(SmallVec<[usize; 4]>, Arc<LazyIm2colParams>)` inside [`LazyIm2Col`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/cnn/conv/lazy_im2col.rs#L194-L200) (or a thread-local / mutex slot on `LazyDeferred`), eliminating ~74 offset table builds and multi-megabyte heap churn per sentence.

### D. Cache `kokoro_vocab.json` and Voice `.bin` in Memory
- **Root cause**: [`prepare_from_phonemes`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L355-L387) runs on **every sentence chunk** and:
  1. Re-parses `include_str!("kokoro_vocab.json")` via `serde_json::from_str::<HashMap<String, i64>>` and allocates `c.to_string()` per phoneme character ([src/lib.rs](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L360-L366)).
  2. Reads the ~522 KB voice `.bin` file from disk (`std::fs::read(voice_path)`) and converts bytes to `f32` on every chunk ([src/lib.rs](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L374-L378)).
- **Fix**: Store the parsed vocab in a `OnceLock<HashMap<char, i64>>` and cache the loaded voice `Vec<f32>` keyed by `PathBuf`.

---

## 3. Idle Audio Sink & Daemon Wakeup Efficiency (`--serve` / `--send`)

As documented in [docs/battery-and-heat.md](file:///data/data/com.termux/files/home/PRJ/kukuryku/docs/battery-and-heat.md#L183-L185), an open `pacat` stream keeps PulseAudio and Android’s OpenSL ES audio HAL in `RUNNING` state (`~1.5%` continuous little-core CPU + audio path power + Android partial wakelock preventing Doze).

1. **Eliminate the 200 ms `HOLD_POLL` wakeup loop in [`playback_sessions`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L1032-L1077)**:
   - While `hold` is set (`StreamPlayer::with_hold`), the playback thread wakes up every `HOLD_POLL = 200 ms` ([src/lib.rs](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L645)) via `rx.recv_timeout(HOLD_POLL)` just to check `hold.load(Ordering::Acquire)`.
   - Adding a `PlayerMsg::ReleaseHold` variant to [`PlayerMsg`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L647-L651) and sending it from [`StreamPlayer::release_sink`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L688-L690) lets the playback thread block indefinitely in `rx.recv()` while `hold` is active and only start the `idle_timeout` timer once `ReleaseHold` arrives—zero timer wakeups during synthesis.
2. **Lower [`DEFAULT_SINK_IDLE`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L642) from `30 s` to `5–8 s`**:
   - Currently, after a short `--send` notification finishes playing, `pacat` holds the audio sink open for `30 s`, and then PulseAudio waits another `20 s` (`--exit-idle-time=20`) before exiting—holding Android's audio wakelock for **50 seconds** after every utterance.
   - Because [`StreamPlayer::warm`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L680-L685) already pre-warms PulseAudio in parallel with Stage 1 + Stage 2 synthesis on the next utterance, dropping `DEFAULT_SINK_IDLE` to `5–8 s` cuts post-utterance audio-HAL drain by `~22–25 s` per invocation without adding first-chunk latency.
3. **Avoid 5 s `BUSY_RECHECK` wakeups in [`accept_loop`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/serve.rs#L185-L209)**:
   - While `player.is_sink_live()` is true, the daemon wakes every `5 s` (`BUSY_RECHECK`) until the sink closes. Lowering `DEFAULT_SINK_IDLE` (or sleeping for the remaining sink idle duration instead of polling every 5 s) removes those extra wakeups.

---

## 4. Spending the `0.80 → 1.00` Wall-RTF Headroom for Lower $V^2 f$

Because Stage 2 finishes in `~0.70×` realtime on 3 gold cores (wall RTF `~0.80–0.85` with Stage 1 pipelined), the gold cluster runs at `2016 MHz` (its highest voltage/frequency point) and then idles for ~15–20% of the audio duration:
- **OS-level 70% CPU speed limit (One UI Power Saving)**: Enabling Samsung’s Power Saving *"Limit CPU speed to 70%"* lowers the gold cluster’s `scaling_max_freq` from `2016 MHz` to `~1497–1708 MHz`. Unlike dropping from 3 threads to 2 (which stays at `2016 MHz` and high $V$), capping frequency drops the cluster operating voltage $V$, yielding a quadratic $V^2$ dynamic power reduction while spending the existing ~20% wall-RTF margin to stay right around **wall RTF `~0.95–1.00`**.
- **Combined with 2A + 2B above**: Once bias `Add` fusion and `Scan` FP16 reduce Stage 2/Stage 1 compute by another ~8–12%, the 3-gold-core pipeline will comfortably maintain wall RTF $\le 1.0$ even at lower clock caps or under sustained thermal load.

---

## 5. Remaining Correctness / Battery Edge Cases on `main` (`4ece158`)

1. **`KOKORO_THERM_HOT_C <= 87` permanently latches thermal hold at RTF `~1.35`**:
   - In [`ThermalWatch::from_env_with`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/tract_backend.rs#L815-L837), `hot` accepts `50..=120` and sets `cool = hot - 6`. Since the 2-thread gold pool stabilizes around `~82 °C`, setting `KOKORO_THERM_HOT_C=85` sets `cool = 79 °C`, which the 2-thread pool never cools down to during continuous reading—locking synthesis into the slower 2-thread mode (`RTF ~1.35`, audio gaps, no energy saved). Floor `KOKORO_THERM_HOT_C` at `88` (or make the hysteresis floor `max(hot - 6, 83_000)`).
2. **`declutter_pow` missing float check**:
   - In [`declutter_pow`](file:///data/data/com.termux/files/home/PRJ/kukuryku/third_party/tract/tract-core/src/ops/math/mod.rs#L489-L530), `exp == 3.0` rewrites `Pow(x, 3)` to `Square(x) * x` without checking `dt.is_float()`, which fails if an integer `Pow` has exponent `3`. Add `if !dt.is_float() { return Ok(None); }` alongside the `2.0` branch.
3. **PulseAudio config detection edge cases in [`src/lib.rs`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L783-L808)**:
   - [`pa_default_script_replaced`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L783-L794) checks `-n` and `PULSE_SCRIPT`, but not `-F` / `--file=`.
   - [`default_pa_loads_android_sink`](file:///data/data/com.termux/files/home/PRJ/kukuryku/src/lib.rs#L796-L808) does not follow `.include /data/data/com.termux/files/usr/etc/pulse/default.pa` in a user `~/.config/pulse/default.pa`, which would cause duplicate `module-sles-sink` loads.
