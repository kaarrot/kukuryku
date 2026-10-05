//! Pure-Rust two-stage Kokoro inference on top of `tract`. Lives in the library
//! (not the `ryk` binary) so both the binary's one-shot path and the `serve`
//! daemon can share a single compiled `Pipeline`.
//!
//!   Stage 1: input_ids, style, speed -> phoneme features (640-ch, 512-ch) + durations
//!   [Rust]:  round durations, total_frames = sum, build the [N, total_frames]
//!            alignment matrix (block expansion: frame t belongs to phoneme i)
//!   Stage 2: the two feature tensors + style + alignment -> waveform

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::ShapeFactoid;

use crate::info;

/// One input dimension in a plan spec: a fixed size, or a named symbol shared
/// across inputs. Shared symbols let a *single* optimized plan serve every
/// phoneme count N / frame count F, so `into_optimized()` is paid once total
/// instead of once per distinct sentence length. See docs/tract-support-plan.md.
#[derive(Clone, Copy)]
enum Dim {
    Fixed(usize),
    Sym(&'static str),
}
use Dim::{Fixed, Sym};

// Stage-boundary tensor names (see tools/split_kokoro.py).
const S1_FEAT_640: &str = "/encoder/Transpose_output_0";
const S1_FEAT_512: &str = "/encoder/text_encoder/Transpose_2_output_0";
const S2_ALIGNMENT: &str = "/encoder/Cast_4_output_0";

/// A subgraph compiled for one concrete input shape. tract can't optimize the
/// split subgraphs with a *symbolic* length dim (the style-broadcast
/// Expand/Concat hits `Impossible to unify Sym(N) with Val(1)`), so each plan is
/// shape-specialized. `Pipeline` caches these keyed by (bucketed) length so the
/// ~1–2s `into_optimized()` is paid once per bucket, not once per sentence.
struct Stage {
    runnable: TypedRunnableModel<TypedModel>,
    input_names: Vec<String>,
    /// When set, Rust feeds f16 inputs. The f32 ONNX files are not rewritten;
    /// weights are cast inside the compiled plan (see [`apply_fp16`]).
    fp16_inputs: bool,
}

impl Stage {
    /// Parse the subgraph, pin each input to its (possibly symbolic) shape
    /// (matched by name so input order is robust), and optimize. Symbolic dims
    /// with a shared name are the same `Symbol`, so tract keeps the length axis
    /// free and one plan serves all lengths; fixed dims specialize as before.
    ///
    /// `fp16` does not write a new weight file. `stage1.onnx` / `stage2.onnx`
    /// stay f32 on disk. When it is set, Snake is fused on the f32 graph first
    /// (declutter), then [`apply_fp16`] casts matching GEMM weights to f16
    /// inside this plan. The default path is one `into_optimized`, unchanged.
    fn build(path: &Path, spec: &[(&str, &[Dim])], fp16: bool) -> Result<Stage> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading {}", path.display()))?;
        let outlets = model.input_outlets()?.to_vec();
        let input_names: Vec<String> =
            outlets.iter().map(|o| model.node(o.node).name.clone()).collect();

        for (ix, name) in input_names.iter().enumerate() {
            let dims = spec
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, d)| *d)
                .with_context(|| format!("{}: no shape spec for input '{name}'", path.display()))?;
            let shape: Vec<TDim> = dims
                .iter()
                .map(|d| match d {
                    Fixed(v) => (*v as i64).to_dim(),
                    Sym(s) => model.sym(s).to_dim(),
                })
                .collect();
            let dt = model.outlet_fact(outlets[ix])?.datum_type().unwrap_or_else(f32::datum_type);
            model.set_input_fact(ix, InferenceFact::dt_shape(dt, ShapeFactoid::from(shape)))?;
        }

        let mut typed = model
            .into_typed()
            .with_context(|| format!("typing {}", path.display()))?;
        if fp16 {
            // Fuse Snake while the graph is still f32. A second declutter inside
            // `into_optimized` then runs on the translated graph.
            typed.declutter().with_context(|| format!("declutter {}", path.display()))?;
            apply_fp16(&mut typed)?;
        }
        let runnable = typed
            .into_optimized()
            .with_context(|| format!("optimizing {}", path.display()))?
            .into_runnable()?;
        Ok(Stage { runnable, input_names, fp16_inputs: fp16 })
    }

    /// Run the cached plan; tensors are matched to declared inputs by name.
    fn run(&self, inputs: &[(&str, Tensor)], stage: &str) -> Result<TVec<TValue>> {
        let mut ordered: TVec<TValue> = TVec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let (_, t) = inputs
                .iter()
                .find(|(n, _)| n == name)
                .with_context(|| format!("{stage}: no tensor supplied for input '{name}'"))?;
            // The translator rewrites sources to f16 even when the node filter
            // says no, so Rust-built f32 inputs have to be cast at run time.
            let owned = if self.fp16_inputs {
                t.cast_to::<f16>()
                    .with_context(|| format!("{stage}: casting input '{name}' to f16"))?
                    .into_owned()
            } else {
                t.clone()
            };
            ordered.push(owned.into());
        }
        if std::env::var_os("KOKORO_TRACT_NAN_TRACE").is_some() {
            nan_trace_run(&self.runnable, ordered, stage)
        } else if std::env::var_os("KOKORO_TRACT_PROFILE").is_some() {
            profile_run(&self.runnable, ordered, stage)
        } else {
            self.runnable.run(ordered).with_context(|| format!("running {stage}"))
        }
    }
}

/// Per-op profiler (KOKORO_TRACT_PROFILE): run the plan node-by-node, timing
/// each node's eval and accumulating wall-time by op type, then print the
/// biggest cost centres. Shows where stage-2's runtime actually goes.
///
/// With KOKORO_TRACT_PROFILE_NODES=<N> also print the top-N *individual* nodes
/// by time, tagged with their concrete input/output shapes — this is what pins
/// which shapes an aggregated op bucket (e.g. the raw `Mul` bucket) actually is,
/// so a fusion-gate fix can be scoped correctly. Shapes are read from the live
/// tensors at eval, so they're concrete even under the symbolic plan.
fn profile_run(
    runnable: &TypedRunnableModel<TypedModel>,
    inputs: TVec<TValue>,
    stage: &str,
) -> Result<TVec<TValue>> {
    use std::collections::HashMap;
    use tract_onnx::tract_core::plan::{SimpleState, eval};
    let top_nodes: Option<usize> = std::env::var("KOKORO_TRACT_PROFILE_NODES")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut state = SimpleState::new(runnable)?;
    // (total secs, call count) keyed by op type name.
    let mut acc: HashMap<String, (f64, usize)> = HashMap::new();
    // Per-node accumulator (only populated when top_nodes is set): node id ->
    // (op name, total secs, calls, last-seen input shapes, last-seen out shapes).
    let mut per_node: HashMap<usize, (String, f64, usize, String, String)> = HashMap::new();
    let out = state.run_plan_with_eval(inputs, |session, op_state, node, input| {
        let in_shapes = top_nodes.map(|_| shape_tag(input.iter().map(|t| t.shape())));
        let t = std::time::Instant::now();
        let r = eval(session, op_state, node, input);
        let dt = t.elapsed().as_secs_f64();
        let e = acc.entry(node.op().name().into_owned());
        let slot = e.or_insert((0.0, 0));
        slot.0 += dt;
        slot.1 += 1;
        if let Some(in_shapes) = in_shapes {
            let out_shapes = r
                .as_ref()
                .map(|o| shape_tag(o.iter().map(|t| t.shape())))
                .unwrap_or_default();
            let e = per_node.entry(node.id).or_insert_with(|| {
                (node.op().name().into_owned(), 0.0, 0, in_shapes, out_shapes)
            });
            e.1 += dt;
            e.2 += 1;
        }
        r
    })?;
    let mut rows: Vec<(String, f64, usize)> =
        acc.into_iter().map(|(k, (s, c))| (k, s, c)).collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    eprintln!("[kokoro]   {stage} profile (op: total_s  calls  %):");
    for (op, secs, calls) in rows.iter().take(12) {
        eprintln!("[kokoro]     {op:<28} {secs:7.3}s  {calls:5}  {:4.1}%", 100.0 * secs / total);
    }
    if let Some(n) = top_nodes {
        let mut nodes: Vec<_> = per_node.into_values().collect();
        nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("[kokoro]   {stage} top-{n} nodes (op  total_s  calls  in -> out):");
        for (op, secs, calls, ins, outs) in nodes.iter().take(n) {
            eprintln!(
                "[kokoro]     {op:<20} {secs:7.3}s  {calls:4}  {ins} -> {outs}",
            );
        }
    }
    Ok(out)
}

/// NaN-trace (KOKORO_TRACT_NAN_TRACE): step the plan node-by-node and, the first
/// time any f32 output contains a non-finite value, print the culprit node's op /
/// id / shapes plus each input's nan/inf/min/max. That pins the failing op class
/// (conv, InstanceNorm, atan2, tract-linalg kernel, …) without further rebuilds.
/// Cast-to-f32 lets the check see f16 tensors after widening; non-numeric ops are
/// skipped silently. Only the first bad node is reported (further NaNs propagate).
fn nan_trace_run(
    runnable: &TypedRunnableModel<TypedModel>,
    inputs: TVec<TValue>,
    stage: &str,
) -> Result<TVec<TValue>> {
    use std::collections::HashMap;
    use tract_onnx::tract_core::plan::{SimpleState, eval};
    // Snapshot: op names + first-output stats accumulated as we go, so when
    // the trace fires we can walk any number of hops upstream from the culprit.
    let plan_model = runnable.model().clone();
    let mut stats: HashMap<usize, (String, Vec<usize>, usize, usize, usize, f32, f32)> =
        HashMap::new();
    let mut fired = false;
    let mut state = SimpleState::new(runnable)?;
    let out = state.run_plan_with_eval(inputs, |session, op_state, node, input| {
        let r = eval(session, op_state, node, input);
        if let Ok(ref outputs) = r {
            if let Some(o) = outputs.first() {
                let (nan, inf, n, zeros, mn, mx) = summarize(o);
                stats.insert(
                    node.id,
                    (node.op().name().into_owned(), o.shape().to_vec(), nan, inf, zeros, mn, mx),
                );
                if !fired && (nan > 0 || inf > 0) {
                    fired = true;
                    eprintln!(
                        "[nan-trace] {stage}: FIRST bad node #{} op={} shape={:?} nan={nan} inf={inf} zeros={zeros} of {n}",
                        node.id, node.op().name(), o.shape(),
                    );
                    // Walk upstream chain (BFS bounded by depth) from node.id.
                    let max_depth: usize = std::env::var("KOKORO_TRACT_NAN_HOPS")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(5);
                    let mut frontier: Vec<(usize, usize)> = node.inputs.iter()
                        .map(|o| (o.node, 1)).collect();
                    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
                    seen.insert(node.id);
                    while let Some((nid, depth)) = frontier.pop() {
                        if !seen.insert(nid) { continue }
                        if depth > max_depth { continue }
                        let indent = "  ".repeat(depth);
                        let pred = plan_model.node(nid);
                        if let Some(s) = stats.get(&nid) {
                            eprintln!(
                                "[nan-trace] {indent}#{nid}/{} shape={:?} nan={} inf={} zeros={} min={:.4e} max={:.4e}",
                                s.0, s.1, s.2, s.3, s.4, s.5, s.6,
                            );
                        } else {
                            eprintln!(
                                "[nan-trace] {indent}#{nid}/{} (no runtime stats — likely graph input/const)",
                                pred.op().name(),
                            );
                        }
                        for o in &pred.inputs {
                            frontier.push((o.node, depth + 1));
                        }
                    }
                }
            }
        }
        r
    })?;
    if !fired {
        eprintln!("[nan-trace] {stage}: no non-finite outputs observed");
    }
    Ok(out)
}

/// (nan, inf, n, zeros, finite_min, finite_max) for an f32-castable tensor. Zero
/// count is important because `Recip` upstream of a NaN needs exact zeros (not
/// just small values) to emit Inf. Non-numeric tensors report all zeros.
fn summarize(v: &TValue) -> (usize, usize, usize, usize, f32, f32) {
    let Ok(t) = v.cast_to::<f32>() else { return (0, 0, 0, 0, 0.0, 0.0) };
    let Ok(s) = t.as_slice::<f32>() else { return (0, 0, 0, 0, 0.0, 0.0) };
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut zeros = 0usize;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in s {
        if x.is_nan() {
            nan += 1;
        } else if x.is_infinite() {
            inf += 1;
        } else {
            if x == 0.0 { zeros += 1 }
            if x < mn { mn = x }
            if x > mx { mx = x }
        }
    }
    if !mn.is_finite() {
        mn = 0.0;
        mx = 0.0;
    }
    (nan, inf, s.len(), zeros, mn, mx)
}

/// Render an iterator of tensor shapes as a compact tag like `[1,512,377]x[1,512,1]`.
fn shape_tag<'a>(shapes: impl Iterator<Item = &'a [usize]>) -> String {
    shapes
        .map(|s| {
            let dims: Vec<String> = s.iter().map(|d| d.to_string()).collect();
            format!("[{}]", dims.join(","))
        })
        .collect::<Vec<_>>()
        .join("x")
}

/// Debug: if KOKORO_TRACT_DUMP=<dir> is set, write an f32 tensor as raw
/// little-endian bytes (shape known by the caller) for offline diffing.
fn dump(name: &str, v: &TValue) -> Result<()> {
    if let Some(dir) = std::env::var_os("KOKORO_TRACT_DUMP") {
        let t = v.cast_to::<f32>()?;
        let data: Vec<f32> = t.to_array_view::<f32>()?.iter().copied().collect();
        std::fs::write(
            std::path::Path::new(&dir).join(format!("{name}.f32")),
            bytemuck::cast_slice::<f32, u8>(&data),
        )?;
    }
    Ok(())
}

/// `KOKORO_TRACT_FP16=1`. Opt-in; the default graph stays f32.
fn fp16_enabled() -> bool {
    std::env::var("KOKORO_TRACT_FP16").ok().as_deref() == Some("1")
}

/// Translate a node only when it is a GEMM-like op and not on the f32 keep
/// list (Snake, STFT, norms, elementwise trig, the harmonic source).
fn fp16_translate_name(name: &str) -> bool {
    const DENY: &[&str] = &[
        "m_source", "stft", "STFT", "istft", "iSTFT", "Greater", "Atan", "Exp", "InstanceNorm",
        "SinSq", "Snake", "Sin",
    ];
    if DENY.iter().any(|d| name.contains(d)) {
        return false;
    }
    name.contains("Conv") || name.contains("MatMul") || name.contains("Gemm")
}

/// Cast matching f32 weights to f16 inside `model`. Does not touch the ONNX
/// files. No-op (with an error) when the CPU has no fp16 SIMD.
fn apply_fp16(model: &mut TypedModel) -> Result<()> {
    use std::sync::OnceLock;
    if !tract_linalg::has_fp16() {
        bail!("KOKORO_TRACT_FP16=1 but this CPU has no fp16 SIMD (asimdhp)");
    }
    static ONCE: OnceLock<()> = OnceLock::new();
    if ONCE.set(()).is_ok() {
        eprintln!(
            "[kokoro] FP16 GEMM is on: casting f32 weights to f16 in memory; stage1.onnx/stage2.onnx are not rewritten"
        );
    }
    let translator =
        tract_onnx::tract_core::floats::FloatPrecisionTranslator::<f32, f16>::with_filter(|node| {
            fp16_translate_name(&node.name)
        });
    model.transform(&translator)?;
    Ok(())
}

/// Copy an output tensor as an f32 Tensor (features cross the stage boundary
/// as f32; casts down if a subgraph ran in f64).
fn f32_tensor(v: &TValue) -> Result<Tensor> {
    let t = v.cast_to::<f32>()?;
    let view = t.to_array_view::<f32>()?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<f32> = view.iter().copied().collect();
    Ok(Tensor::from_shape(&shape, &data)?)
}

/// One online core and its capacity (sysfs `cpu_capacity`, else max freq).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CpuCore {
    id: usize,
    capacity: u32,
}

/// `$KOKORO_TRACT_CPUSET`: keyword or an explicit list (`4-6`, `0-3,7`).
#[derive(Clone, Debug, PartialEq, Eq)]
enum CpusetSpec {
    Auto,
    Mid,
    Little,
    All,
    OffPrime,
    Explicit(Vec<usize>),
}

/// Parse `0-3,7` / `4-6` style lists. Empty or garbage → `None`.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().ok()?;
            let end: usize = b.trim().parse().ok()?;
            if start > end {
                return None;
            }
            out.extend(start..=end);
        } else {
            out.push(part.parse().ok()?);
        }
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() { None } else { Some(out) }
}

fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &c in &cpus[1..] {
        if c == prev + 1 {
            prev = c;
            continue;
        }
        parts.push(fmt_cpu_range(start, prev));
        start = c;
        prev = c;
    }
    parts.push(fmt_cpu_range(start, prev));
    parts.join(",")
}

fn fmt_cpu_range(a: usize, b: usize) -> String {
    if a == b { format!("{a}") } else { format!("{a}-{b}") }
}

fn parse_cpuset_spec(s: &str) -> Option<CpusetSpec> {
    match s.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Some(CpusetSpec::Auto),
        "mid" => Some(CpusetSpec::Mid),
        "little" => Some(CpusetSpec::Little),
        "all" => Some(CpusetSpec::All),
        "off-prime" | "off_prime" => Some(CpusetSpec::OffPrime),
        other => parse_cpu_list(other).map(CpusetSpec::Explicit),
    }
}

fn cpuset_from_env() -> CpusetSpec {
    match std::env::var("KOKORO_TRACT_CPUSET") {
        Err(_) => CpusetSpec::Auto,
        Ok(s) => parse_cpuset_spec(&s).unwrap_or_else(|| {
            eprintln!("[kokoro] ignoring invalid KOKORO_TRACT_CPUSET={s:?}");
            CpusetSpec::Auto
        }),
    }
}

/// Capacity groups, ascending. Each group's cpu ids are sorted.
fn groups_by_capacity(cpus: &[CpuCore]) -> Vec<(u32, Vec<usize>)> {
    let mut map = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for c in cpus {
        map.entry(c.capacity).or_default().push(c.id);
    }
    map.into_iter()
        .map(|(cap, mut ids)| {
            ids.sort_unstable();
            (cap, ids)
        })
        .collect()
}

fn nth_highest_group(cpus: &[CpuCore], n: usize) -> Option<Vec<usize>> {
    let groups = groups_by_capacity(cpus);
    let idx = groups.len().checked_sub(n + 1)?;
    Some(groups[idx].1.clone())
}

/// Unique max-capacity core (the prime), if that group has size 1.
fn unique_prime(cpus: &[CpuCore]) -> Option<usize> {
    let groups = groups_by_capacity(cpus);
    let last = groups.last()?;
    if last.1.len() == 1 { Some(last.1[0]) } else { None }
}

/// Drop a singleton prime, then take the highest remaining cluster.
/// Homogeneous (one capacity) → `None` (do not pin).
fn auto_android_cpuset(cpus: &[CpuCore]) -> Option<Vec<usize>> {
    if groups_by_capacity(cpus).len() < 2 {
        return None;
    }
    let mut remaining: Vec<CpuCore> = cpus.to_vec();
    if let Some(prime) = unique_prime(cpus) {
        remaining.retain(|c| c.id != prime);
    }
    nth_highest_group(&remaining, 0)
}

fn off_prime_cpuset(cpus: &[CpuCore]) -> Option<Vec<usize>> {
    let prime = unique_prime(cpus)?;
    let mut ids: Vec<usize> = cpus.iter().map(|c| c.id).filter(|&id| id != prime).collect();
    if ids.is_empty() {
        return None;
    }
    ids.sort_unstable();
    Some(ids)
}

fn pick_cpuset(spec: &CpusetSpec, topo: &[CpuCore], android: bool) -> Option<Vec<usize>> {
    let all: Vec<usize> = {
        let mut ids: Vec<usize> = topo.iter().map(|c| c.id).collect();
        ids.sort_unstable();
        ids
    };
    let chosen = match spec {
        CpusetSpec::All => None,
        CpusetSpec::Auto => {
            if android { auto_android_cpuset(topo) } else { None }
        }
        CpusetSpec::Mid => nth_highest_group(topo, 1).or_else(|| nth_highest_group(topo, 0)),
        CpusetSpec::Little => groups_by_capacity(topo).into_iter().next().map(|(_, ids)| ids),
        CpusetSpec::OffPrime => off_prime_cpuset(topo).or_else(|| {
            if all.is_empty() { None } else { Some(all.clone()) }
        }),
        CpusetSpec::Explicit(ids) => {
            if topo.is_empty() {
                Some(ids.clone())
            } else {
                let online: std::collections::HashSet<usize> = all.iter().copied().collect();
                let v: Vec<usize> = ids.iter().copied().filter(|id| online.contains(id)).collect();
                if v.is_empty() { None } else { Some(v) }
            }
        }
    };
    match chosen {
        Some(cs) if !all.is_empty() && cs == all => None, // pin-to-all is a no-op
        other => other,
    }
}

fn read_online_cpus() -> Option<Vec<usize>> {
    let s = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    parse_cpu_list(s.trim())
}

fn read_capacity(cpu: usize) -> Option<u32> {
    let cap = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
    if let Ok(s) = std::fs::read_to_string(&cap) {
        if let Ok(v) = s.trim().parse() {
            return Some(v);
        }
    }
    let freq = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
    std::fs::read_to_string(freq).ok()?.trim().parse().ok()
}

fn read_topology() -> Vec<CpuCore> {
    let Some(online) = read_online_cpus() else {
        return Vec::new();
    };
    online
        .into_iter()
        .filter_map(|id| Some(CpuCore { id, capacity: read_capacity(id)? }))
        .collect()
}

/// Pin the calling thread. New threads (rayon pool, compile, playback) inherit.
fn apply_affinity(cpus: &[usize]) -> std::io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        unsafe {
            let mut set = std::mem::zeroed::<libc::cpu_set_t>();
            for &cpu in cpus {
                libc::CPU_SET(cpu, &mut set);
            }
            let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            if rc == 0 { Ok(()) } else { Err(std::io::Error::last_os_error()) }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = cpus;
        Ok(())
    }
}

fn nproc() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// Thread count: env wins, else cpuset size, else nproc.
/// Always at least 1; never more workers than CPUs in the pin set.
///
/// Android auto is the full mid cluster (3 golds on this S10e). A 2-thread cap
/// was measured at RTF ~1.45 vs ~1.22 on 3 golds — too slow to keep the ~1.2
/// rate. `KOKORO_TRACT_THREADS=2` is still the cooler override.
fn resolve_threads(
    env_threads: Option<usize>,
    cpuset: Option<&[usize]>,
    nproc: usize,
) -> usize {
    let mut threads = env_threads.unwrap_or_else(|| cpuset.map(|c| c.len()).unwrap_or(nproc));
    if let Some(cs) = cpuset {
        threads = threads.min(cs.len());
    }
    threads.max(1)
}

/// Where espeak + stage 1 run. Android `auto` is the little cluster, not the
/// golds the vocoder is pinned to. Desktop `auto` does not pin.
fn pick_s1(spec: &CpusetSpec, topo: &[CpuCore], android: bool) -> Option<Vec<usize>> {
    match spec {
        CpusetSpec::Auto if android => pick_cpuset(&CpusetSpec::Little, topo, true),
        CpusetSpec::Auto => None,
        other => pick_cpuset(other, topo, android),
    }
}

fn s1_spec_from_env() -> CpusetSpec {
    match std::env::var("KOKORO_TRACT_S1_CPUSET") {
        Err(_) => CpusetSpec::Auto,
        Ok(s) => parse_cpuset_spec(&s).unwrap_or_else(|| {
            eprintln!("[kokoro] ignoring invalid KOKORO_TRACT_S1_CPUSET={s:?}");
            CpusetSpec::Auto
        }),
    }
}

/// Pin the calling thread to the stage-1 set. Called from the lookahead thread
/// so it does not keep the gold mask inherited from main.
pub fn pin_stage1_thread() {
    let topo = read_topology();
    let Some(cs) = pick_s1(&s1_spec_from_env(), &topo, cfg!(target_os = "android")) else {
        return;
    };
    if let Err(e) = apply_affinity(&cs) {
        eprintln!(
            "[kokoro] stage1 sched_setaffinity({}) failed: {e}",
            format_cpu_list(&cs)
        );
    } else {
        info!("[kokoro] stage1 lookahead on cpus {}", format_cpu_list(&cs));
    }
}

/// Pin playback (and any pacat / pulseaudio it spawns) to the little cores.
/// No-op off Android. Does not touch the calling thread when it is not the
/// playback thread — callers invoke it from that thread only.
pub fn pin_playback_thread() {
    if !cfg!(target_os = "android") {
        return;
    }
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    let topo = read_topology();
    let Some(cs) = pick_cpuset(&CpusetSpec::Little, &topo, true) else {
        return;
    };
    if let Err(e) = apply_affinity(&cs) {
        if ONCE.set(()).is_ok() {
            eprintln!(
                "[kokoro] playback sched_setaffinity({}) failed: {e}",
                format_cpu_list(&cs)
            );
        }
    } else if ONCE.set(()).is_ok() {
        info!("[kokoro] playback pinned to cpus {}", format_cpu_list(&cs));
    }
}

/// Stage-2 pool the governor may pick. `Full` is the whole pin set (3 golds
/// on this phone). `Cool` is the first two of that set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoolKind {
    Full,
    Cool,
}

/// `KOKORO_GOVERNOR`. Default is thermal (pace + heat). `off` is today's
/// always-full behaviour. `pace` uses the buffer thresholds only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GovernorMode {
    Off,
    Pace,
    Thermal,
}

const GOV_HI: f64 = 6.0;
const GOV_LO: f64 = 2.0;
const GOV_MAX_AHEAD: f64 = 20.0;
/// lmh-dcvs trips at 85°C. Start shedding a gold before that slam.
const THERM_HOT: i32 = 75;
/// Release the hold only after the sensor has come back down.
const THERM_COOL: i32 = 68;

fn governor_mode() -> GovernorMode {
    match std::env::var("KOKORO_GOVERNOR") {
        Err(_) => GovernorMode::Thermal,
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "" | "thermal" => GovernorMode::Thermal,
            "off" => GovernorMode::Off,
            "pace" => GovernorMode::Pace,
            other => {
                eprintln!("[kokoro] ignoring invalid KOKORO_GOVERNOR={other:?}; using thermal");
                GovernorMode::Thermal
            }
        },
    }
}

/// Pure pool choice. Does not read sysfs.
///
/// `Off` stays on the full pool. A thermal hold forces cool. Otherwise ≥6 s
/// of queued audio drops to cool, <2 s goes back to full, and the band in
/// between keeps the previous choice.
fn choose_pool(mode: GovernorMode, buffered: f64, thermal_hold: bool, prev: PoolKind) -> PoolKind {
    if mode == GovernorMode::Off {
        return PoolKind::Full;
    }
    if mode == GovernorMode::Thermal && thermal_hold {
        return PoolKind::Cool;
    }
    if buffered >= GOV_HI {
        PoolKind::Cool
    } else if buffered < GOV_LO {
        PoolKind::Full
    } else {
        prev
    }
}

/// Pure hysteresis. Missing sensor reads pass `lmh_c == 0` and do not trip
/// the hold by themselves. `gold_capped` means the kernel already lowered
/// cpu4's ceiling, which is treated as hot.
fn update_thermal_hold(hold: bool, lmh_c: i32, gold_capped: bool) -> bool {
    if hold {
        !(lmh_c < THERM_COOL && !gold_capped)
    } else {
        lmh_c >= THERM_HOT || gold_capped
    }
}

/// Hottest `lmh-dcvs*` thermal zone, in °C. 0 when the zones are unreadable.
fn read_lmh_max_c() -> i32 {
    let Ok(rd) = std::fs::read_dir("/sys/class/thermal") else {
        return 0;
    };
    let mut max_c = 0i32;
    let mut found = false;
    for ent in rd.flatten() {
        let p = ent.path();
        let Ok(typ) = std::fs::read_to_string(p.join("type")) else {
            continue;
        };
        if !typ.trim().starts_with("lmh-dcvs") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(p.join("temp")) else {
            continue;
        };
        let Ok(milli) = raw.trim().parse::<i32>() else {
            continue;
        };
        found = true;
        max_c = max_c.max(milli / 1000);
    }
    if found { max_c } else { 0 }
}

/// True when cpu4's scaling ceiling is already below its hardware max.
/// Reads only — writing `scaling_max_freq` is EPERM from Termux.
fn gold_freq_capped() -> bool {
    let base = "/sys/devices/system/cpu/cpu4/cpufreq";
    let max = std::fs::read_to_string(format!("{base}/scaling_max_freq"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let info = std::fs::read_to_string(format!("{base}/cpuinfo_max_freq"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    match (max, info) {
        (Some(m), Some(i)) => m + 1000 < i,
        _ => false,
    }
}

fn pin_or_warn(cpus: &[usize]) {
    if let Err(e) = apply_affinity(cpus) {
        eprintln!(
            "[kokoro] sched_setaffinity({}) failed: {e}; continuing unpinned",
            format_cpu_list(cpus)
        );
    }
}

fn make_pool(threads: usize) -> tract_linalg::multithread::Executor {
    use tract_linalg::multithread::Executor;
    if threads > 1 { Executor::multithread(threads) } else { Executor::SingleThread }
}

struct Executors {
    full: tract_linalg::multithread::Executor,
    cool: Option<tract_linalg::multithread::Executor>,
    /// False when the env picked a thread count, the governor is off, or the
    /// pin set is smaller than 3. Speak then always uses `full`.
    switch: bool,
    mode: GovernorMode,
}

/// Build the stage-2 pools.
///
/// Android auto pins main to the mid cluster (this S10e: cpu4–6) and never
/// puts it back on all cores — a later unpinned spawn would let EAS park
/// stage 1 on the prime. When the governor is allowed to switch, the cool
/// pool is spawned *while* main is pinned to the first two of those cpus, then
/// main is re-pinned to the full set and the full pool is spawned. Rayon
/// workers keep the mask they inherited. `KOKORO_TRACT_THREADS` forces one
/// pool and disables the switch.
fn build_executors() -> Executors {
    let spec = cpuset_from_env();
    let topo = read_topology();
    let cpuset = pick_cpuset(&spec, &topo, cfg!(target_os = "android"));
    let env_threads = std::env::var("KOKORO_TRACT_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t > 0);
    let mode = governor_mode();

    if let Some(req) = env_threads {
        let threads = resolve_threads(Some(req), cpuset.as_deref(), nproc());
        if let Some(cs) = cpuset.as_ref() {
            pin_or_warn(cs);
            eprintln!(
                "[kokoro] KOKORO_TRACT_THREADS={threads}; governor pool switch disabled (cpuset {})",
                format_cpu_list(cs)
            );
        } else {
            eprintln!(
                "[kokoro] KOKORO_TRACT_THREADS={threads}; governor pool switch disabled"
            );
        }
        return Executors { full: make_pool(threads), cool: None, switch: false, mode };
    }

    if let Some(cs) = cpuset.as_ref() {
        if cs.len() >= 3 && mode != GovernorMode::Off {
            let cool_cpus = &cs[..2];
            pin_or_warn(cool_cpus);
            let cool = tract_linalg::multithread::Executor::multithread(2);
            pin_or_warn(cs);
            let full = tract_linalg::multithread::Executor::multithread(cs.len());
            info!(
                "[kokoro] tract executor: {} threads on {} (governor {:?}), cool 2 threads on {}",
                cs.len(),
                format_cpu_list(cs),
                mode,
                format_cpu_list(cool_cpus),
            );
            return Executors { full, cool: Some(cool), switch: true, mode };
        }
        pin_or_warn(cs);
        let threads = resolve_threads(None, Some(cs), nproc());
        info!(
            "[kokoro] tract executor: {threads} threads, cpuset {}",
            format_cpu_list(cs)
        );
        return Executors { full: make_pool(threads), cool: None, switch: false, mode };
    }

    let threads = resolve_threads(None, None, nproc());
    if threads > 1 {
        info!("[kokoro] tract executor: {threads} threads");
    }
    Executors { full: make_pool(threads), cool: None, switch: false, mode }
}

/// A compiled stage: either a single *symbolic* plan that serves every length,
/// or — if the subgraph won't optimize symbolically — a lazily populated
/// per-exact-shape cache (the previous behaviour, kept as a fallback).
///
/// The symbolic plan is the win: tract's `into_optimized()` (~1–4 s) is paid
/// once total, not once per distinct sentence length, so streaming a paragraph
/// of differently-sized sentences no longer recompiles each one. It requires
/// the split subgraphs produced by `tools/split_kokoro.py` (which rewires two
/// `Expand` targets so the phoneme/frame axes stay symbolic) plus two small
/// tract patches (symbolic `Resize` scale, symbolic `Slice` end-clamp). We
/// still cannot *pad* to a shared bucket — the model's global normalization
/// poisons padded output (corr 0.73 phoneme / 0.02 frame) — but a symbolic plan
/// never pads; it resolves N/F from each run's real input shapes.
enum StagePlan {
    Symbolic(Stage),
    PerShape { cache: HashMap<Vec<usize>, Stage>, fp16: bool },
}

impl StagePlan {
    /// Build a single symbolic plan; on optimize failure, degrade to per-shape.
    fn build(path: &Path, spec: &[(&str, &[Dim])], name: &str, fp16: bool) -> StagePlan {
        // Debug lever: force the concrete per-shape path (which enables tract's
        // concrete-shape-gated conv fast paths — lazy im2col + depthwise) so we
        // can bench conv run-speed symbolic-vs-concrete. See docs conv section.
        if std::env::var_os("KOKORO_TRACT_FORCE_PERSHAPE").is_some() {
            eprintln!("[kokoro] {name}: FORCE_PERSHAPE — using per-exact-shape plans");
            return StagePlan::PerShape { cache: HashMap::new(), fp16 };
        }
        match Stage::build(path, spec, fp16) {
            Ok(st) => {
                info!("[kokoro] {name}: compiled one symbolic plan (length-independent)");
                StagePlan::Symbolic(st)
            }
            Err(e) => {
                // Warning (not `info!`): symbolic-plan failure means every distinct
                // length re-pays the ~1–4 s optimize cost — surprising perf cliff
                // that users should see even in silent mode.
                eprintln!(
                    "[kokoro] {name}: symbolic optimize failed ({e:#}); \
                     falling back to per-exact-shape plans"
                );
                StagePlan::PerShape { cache: HashMap::new(), fp16 }
            }
        }
    }

    /// The plan to run for a given concrete shape: the symbolic plan as-is, or
    /// the cached exact-shape plan (compiled on first sight of that shape).
    fn get(&mut self, path: &Path, concrete: &[(&str, &[usize])]) -> Result<&Stage> {
        match self {
            StagePlan::Symbolic(st) => Ok(st),
            StagePlan::PerShape { cache, fp16 } => {
                let fp16 = *fp16;
                let key: Vec<usize> =
                    concrete.iter().flat_map(|(_, d)| d.iter().copied()).collect();
                if !cache.contains_key(&key) {
                    let owned: Vec<(&str, Vec<Dim>)> = concrete
                        .iter()
                        .map(|(nm, d)| (*nm, d.iter().map(|&v| Fixed(v)).collect()))
                        .collect();
                    let spec: Vec<(&str, &[Dim])> =
                        owned.iter().map(|(nm, d)| (*nm, d.as_slice())).collect();
                    cache.insert(key.clone(), Stage::build(path, &spec, fp16)?);
                }
                Ok(&cache[&key])
            }
        }
    }
}

/// Stage 1 plus the Rust length regulator. Owned by the lookahead thread so
/// espeak and the encoder overlap the previous sentence's vocoder.
struct Stage1Runner {
    path: PathBuf,
    plan: StagePlan,
}

/// What stage 2 needs, plus the clocks `speak` reports. `prep_secs` is espeak
/// and is not part of infer RTF. `s1_secs` is.
struct S1Out {
    feat640: Tensor,
    feat512: Tensor,
    alignment: Tensor,
    style: Tensor,
    n: usize,
    frames: usize,
    token_len: usize,
    prep_secs: f64,
    s1_secs: f64,
}

enum AheadOut {
    Ready(S1Out),
    Unspeakable,
    Failed,
}

enum S1Job {
    Text {
        sentence: String,
        lang: String,
        voice: PathBuf,
        speed: f32,
        reply: std::sync::mpsc::SyncSender<Result<AheadOut>>,
    },
    Ids {
        ids: Vec<i64>,
        style: Vec<f32>,
        speed: f32,
        reply: std::sync::mpsc::SyncSender<Result<AheadOut>>,
    },
}

impl Stage1Runner {
    /// Single-threaded on purpose. Putting stage 1 on the stage-2 pool was
    /// measured at +31% (the serial LSTM predictor costs more than the encoder
    /// saves). The lookahead thread is a different core, not that pool.
    fn infer(
        &mut self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
        token_len: usize,
        prep_secs: f64,
    ) -> Result<S1Out> {
        let n = ids.len();
        let style_t = Tensor::from_shape(&[1, style.len()], style)?;
        let t0 = std::time::Instant::now();
        let s1 = self
            .plan
            .get(&self.path, &[("input_ids", &[1, n]), ("style", &[1, 256]), ("speed", &[1])])?
            .run(
                &[
                    ("input_ids", Tensor::from_shape(&[1, n], ids)?),
                    ("style", style_t.clone()),
                    ("speed", Tensor::from_shape(&[1], &[speed])?),
                ],
                "stage1",
            )?;
        dump("s1_feat640", &s1[0])?;
        dump("s1_feat512", &s1[1])?;
        dump("s1_dur", &s1[2])?;
        let feat640 = f32_tensor(&s1[0])?;
        let feat512 = f32_tensor(&s1[1])?;
        if feat640.shape().get(1) != Some(&640) || feat512.shape().get(1) != Some(&512) {
            bail!("unexpected stage1 feature shapes: {:?}, {:?}", feat640.shape(), feat512.shape());
        }
        let dur_t = s1[2].cast_to::<f32>()?;
        let durations = dur_t.to_array_view::<f32>()?;
        // Round per-phoneme durations and build A[N, total_frames] with
        // A[i, t] = 1 iff frame t belongs to phoneme i.
        let durs: Vec<usize> = durations.iter().map(|&d| d.round().max(0.0) as usize).collect();
        let total_frames: usize = durs.iter().sum();
        if total_frames == 0 {
            bail!("length regulator produced 0 frames (all durations rounded to 0)");
        }
        let mut align = vec![0f32; n * total_frames];
        let mut t = 0usize;
        for (i, &d) in durs.iter().enumerate() {
            for _ in 0..d {
                align[i * total_frames + t] = 1.0;
                t += 1;
            }
        }
        let alignment = Tensor::from_shape(&[n, total_frames], &align)?;
        Ok(S1Out {
            feat640,
            feat512,
            alignment,
            style: style_t,
            n,
            frames: total_frames,
            token_len,
            prep_secs,
            s1_secs: t0.elapsed().as_secs_f64(),
        })
    }
}

/// One in-flight stage-1 job. The reply channel holds one message; submitting
/// a second job before that reply is received deadlocks, so callers recv first.
struct Lookahead {
    tx: Option<std::sync::mpsc::SyncSender<S1Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Lookahead {
    fn spawn(runner: Stage1Runner) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || stage1_thread(rx, runner));
        Self { tx: Some(tx), thread: Some(thread) }
    }

    fn submit_text(
        &self,
        sentence: &str,
        lang: &str,
        voice: &Path,
        speed: f32,
    ) -> Result<std::sync::mpsc::Receiver<Result<AheadOut>>> {
        let (reply, rx) = std::sync::mpsc::sync_channel(1);
        self.send(S1Job::Text {
            sentence: sentence.to_string(),
            lang: lang.to_string(),
            voice: voice.to_path_buf(),
            speed,
            reply,
        })?;
        Ok(rx)
    }

    fn submit_ids(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<std::sync::mpsc::Receiver<Result<AheadOut>>> {
        let (reply, rx) = std::sync::mpsc::sync_channel(1);
        self.send(S1Job::Ids {
            ids: ids.to_vec(),
            style: style.to_vec(),
            speed,
            reply,
        })?;
        Ok(rx)
    }

    fn send(&self, job: S1Job) -> Result<()> {
        self.tx
            .as_ref()
            .context("stage1 lookahead stopped")?
            .send(job)
            .map_err(|_| anyhow::anyhow!("stage1 lookahead stopped"))
    }
}

impl Drop for Lookahead {
    fn drop(&mut self) {
        // Drop the sender before joining, or the worker blocks forever in recv.
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn stage1_thread(rx: std::sync::mpsc::Receiver<S1Job>, mut runner: Stage1Runner) {
    pin_stage1_thread();
    while let Ok(job) = rx.recv() {
        let (reply, out) = match job {
            S1Job::Text { sentence, lang, voice, speed, reply } => {
                let t = std::time::Instant::now();
                let prepared = crate::kokoro::prepare_or_skip(&sentence, &lang, &voice);
                let prep_secs = t.elapsed().as_secs_f64();
                let out = match prepared {
                    Ok(crate::kokoro::ChunkPrep::Ready(p)) => {
                        runner.infer(&p.ids, &p.style, speed, p.token_len, prep_secs).map(AheadOut::Ready)
                    }
                    Ok(crate::kokoro::ChunkPrep::Unspeakable) => Ok(AheadOut::Unspeakable),
                    Ok(crate::kokoro::ChunkPrep::Failed) => Ok(AheadOut::Failed),
                    Err(e) => Err(e),
                };
                (reply, out)
            }
            S1Job::Ids { ids, style, speed, reply } => {
                let out = runner.infer(&ids, &style, speed, ids.len(), 0.0).map(AheadOut::Ready);
                (reply, out)
            }
        };
        if reply.send(out).is_err() {
            break;
        }
    }
}

/// One utterance, after [`Pipeline::speak`]. `infer_secs` is stage 1 + stage 2
/// and does not include espeak. `wall_secs` does: it runs from the start of
/// `speak` to the moment the last chunk was ready.
pub struct SpeakReport {
    pub audio_samples: usize,
    pub infer_secs: f64,
    pub wall_secs: f64,
    pub gap_secs: f64,
    pub spoken: usize,
    pub failed: usize,
}

/// The two-stage tract pipeline. Stage 1 lives on a lookahead thread (little
/// cores on Android). Stage 2 uses the gold pool, optionally dropping from 3
/// threads to 2 when audio is buffered or the SoC is hot.
pub struct Pipeline {
    stage2_path: PathBuf,
    execs: Executors,
    stage2: StagePlan,
    ahead: Lookahead,
    thermal_hold: bool,
    pool: PoolKind,
}

impl Pipeline {
    pub fn new(dir: &Path) -> Result<Pipeline> {
        let stage1_path = dir.join("stage1.onnx");
        let stage2_path = dir.join("stage2.onnx");
        // Pins main to the golds before either compile thread or the lookahead
        // thread is spawned, so they inherit that mask (the lookahead then
        // re-pins itself to the littles).
        let execs = build_executors();
        let stage2_path_bg = stage2_path.clone();
        let stage2_fp16 = fp16_enabled();
        // The two compiles are independent. Stage 2 is the long pole; building
        // it on a background thread while stage 1 compiles here drops startup
        // to about max(the two) instead of the sum.
        let stage2_handle = std::thread::spawn(move || {
            StagePlan::build(
                &stage2_path_bg,
                &[
                    (S1_FEAT_640, &[Fixed(1), Fixed(640), Sym("N")]),
                    (S1_FEAT_512, &[Fixed(1), Fixed(512), Sym("N")]),
                    (S2_ALIGNMENT, &[Sym("N"), Sym("F")]),
                    ("style", &[Fixed(1), Fixed(256)]),
                ],
                "stage2",
                stage2_fp16,
            )
        });
        // Stage 1 stays f32. The fp16 experiment is the vocoder GEMMs.
        let stage1 = StagePlan::build(
            &stage1_path,
            &[
                ("input_ids", &[Fixed(1), Sym("N")]),
                ("style", &[Fixed(1), Fixed(256)]),
                ("speed", &[Fixed(1)]),
            ],
            "stage1",
            false,
        );
        let stage2 = stage2_handle
            .join()
            .map_err(|_| anyhow::anyhow!("stage2 compile thread panicked"))?;
        let ahead = Lookahead::spawn(Stage1Runner { path: stage1_path, plan: stage1 });
        Ok(Pipeline {
            stage2_path,
            execs,
            stage2,
            ahead,
            thermal_hold: false,
            pool: PoolKind::Full,
        })
    }

    /// One sentence, no overlap. Stage 2 still honours a thermal hold when the
    /// governor is active. Prefer [`speak`](Self::speak) for a paragraph.
    pub fn synthesize(&mut self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let rx = self.ahead.submit_ids(ids, style, speed)?;
        let ahead = rx.recv().map_err(|_| anyhow::anyhow!("stage1 lookahead stopped"))??;
        let pool = self.select_pool_buffered(0.0);
        match ahead {
            AheadOut::Ready(s1) => self.run_stage2(s1, pool),
            AheadOut::Unspeakable | AheadOut::Failed => bail!("nothing to synthesize"),
        }
    }

    /// Speak `sentences` in order. Espeak and stage 1 of sentence N+1 run on the
    /// lookahead thread while stage 2 of sentence N runs here. One sentence of
    /// lookahead, never two (a second submit before the reply is taken deadlocks).
    ///
    /// `wav`, when set, receives every sample in utterance order.
    pub fn speak(
        &mut self,
        sentences: &[String],
        lang: &str,
        voice: &Path,
        speed: f32,
        player: &crate::kokoro::StreamPlayer,
        mut wav: Option<&mut Vec<f32>>,
    ) -> Result<SpeakReport> {
        let t0 = std::time::Instant::now();
        let mut clock = crate::kokoro::GapClock::new();
        let mut audio_samples = 0usize;
        let mut infer_secs = 0f64;
        let mut spoken = 0usize;
        let mut failed = 0usize;
        let mut wall_secs = 0f64;
        if sentences.is_empty() {
            crate::kokoro::warn_nothing_spoken(0);
            return Ok(SpeakReport {
                audio_samples,
                infer_secs,
                wall_secs,
                gap_secs: 0.0,
                spoken,
                failed,
            });
        }

        let mut pending = Some(self.ahead.submit_text(&sentences[0], lang, voice, speed)?);
        for i in 0..sentences.len() {
            let msg = pending
                .take()
                .context("internal: missing stage1 reply")?
                .recv()
                .map_err(|_| anyhow::anyhow!("stage1 lookahead stopped"))??;
            // Start the next sentence before this vocoder so the two overlap,
            // unless we are already far enough ahead that another sentence
            // would only heat the phone. In that case the vocoder runs first
            // and the next sentence waits until the queue drains.
            let start_next = i + 1 < sentences.len()
                && (self.execs.mode == GovernorMode::Off
                    || player.buffered_secs() < GOV_MAX_AHEAD);
            let mut next_rx = if start_next {
                Some(self.ahead.submit_text(&sentences[i + 1], lang, voice, speed)?)
            } else {
                None
            };
            match msg {
                AheadOut::Ready(s1) => {
                    let prep_secs = s1.prep_secs;
                    let token_len = s1.token_len;
                    let s1_secs = s1.s1_secs;
                    let pool = self.select_pool_buffered(player.buffered_secs());
                    let t_s2 = std::time::Instant::now();
                    let audio = self.run_stage2(s1, pool)?;
                    let s2_secs = t_s2.elapsed().as_secs_f64();
                    let arrival = t0.elapsed().as_secs_f64();
                    let gap = clock.observe(
                        arrival,
                        audio.len() as f64 / crate::kokoro::SAMPLE_RATE as f64,
                    );
                    let chunk_infer = s1_secs + s2_secs;
                    infer_secs += chunk_infer;
                    audio_samples += audio.len();
                    wall_secs = arrival;
                    crate::kokoro::report_chunk(
                        i,
                        sentences.len(),
                        token_len,
                        audio.len(),
                        chunk_infer,
                        prep_secs,
                        gap,
                    );
                    if let Some(buf) = wav.as_mut() {
                        buf.extend_from_slice(&audio);
                    }
                    player.push(audio)?;
                    spoken += 1;
                    if spoken == 1 {
                        info!("[kokoro] first audio at {:.2}s", t0.elapsed().as_secs_f64());
                    }
                }
                AheadOut::Failed => failed += 1,
                AheadOut::Unspeakable => {}
            }
            if i + 1 < sentences.len() && next_rx.is_none() {
                if self.execs.mode != GovernorMode::Off {
                    player.wait_until_buffered_below(GOV_MAX_AHEAD);
                }
                next_rx = Some(self.ahead.submit_text(&sentences[i + 1], lang, voice, speed)?);
            }
            pending = next_rx;
        }
        if spoken == 0 {
            crate::kokoro::warn_nothing_spoken(failed);
        }
        Ok(SpeakReport {
            audio_samples,
            infer_secs,
            wall_secs,
            gap_secs: clock.gap(),
            spoken,
            failed,
        })
    }

    fn select_pool_buffered(&mut self, buffered: f64) -> PoolKind {
        if !self.execs.switch {
            return PoolKind::Full;
        }
        if self.execs.mode == GovernorMode::Thermal {
            let lmh = read_lmh_max_c();
            let capped = gold_freq_capped();
            let next_hold = update_thermal_hold(self.thermal_hold, lmh, capped);
            if next_hold != self.thermal_hold {
                info!(
                    "[kokoro] thermal hold {} (lmh {lmh}°C, gold capped {capped})",
                    if next_hold { "on" } else { "off" }
                );
                self.thermal_hold = next_hold;
            }
        }
        let next = choose_pool(self.execs.mode, buffered, self.thermal_hold, self.pool);
        if next != self.pool {
            info!(
                "[kokoro] governor {:?} -> {:?} (buffered {buffered:.2}s)",
                self.pool, next
            );
            self.pool = next;
        }
        self.pool
    }

    fn executor_for(&self, pool: PoolKind) -> tract_linalg::multithread::Executor {
        if pool == PoolKind::Cool {
            if let Some(cool) = &self.execs.cool {
                return cool.clone();
            }
        }
        self.execs.full.clone()
    }

    fn run_stage2(&mut self, s1: S1Out, pool: PoolKind) -> Result<Vec<f32>> {
        let n = s1.n;
        let frames = s1.frames;
        let executor = self.executor_for(pool);
        let stage2 = self.stage2.get(
            &self.stage2_path,
            &[
                (S1_FEAT_640, &[1, 640, n]),
                (S1_FEAT_512, &[1, 512, n]),
                (S2_ALIGNMENT, &[n, frames]),
                ("style", &[1, 256]),
            ],
        )?;
        let feat640 = s1.feat640;
        let feat512 = s1.feat512;
        let alignment = s1.alignment;
        let style = s1.style;
        let s2 = tract_linalg::multithread::multithread_tract_scope(executor, || {
            stage2.run(
                &[
                    (S1_FEAT_640, feat640),
                    (S1_FEAT_512, feat512),
                    (S2_ALIGNMENT, alignment),
                    ("style", style),
                ],
                "stage2",
            )
        })?;
        for (i, o) in s2.iter().enumerate() {
            dump(&format!("s2_out{i}"), o)?;
        }
        let wav = s2[0].cast_to::<f32>()?;
        Ok(wav.to_array_view::<f32>()?.iter().copied().collect())
    }
}

#[cfg(test)]
mod cpuset_tests {
    use super::*;

    fn s10e() -> Vec<CpuCore> {
        // Snapdragon 855: 4×A55 + 3×A76 + prime A76
        let mut v = Vec::new();
        v.extend((0..4).map(|id| CpuCore { id, capacity: 378 }));
        v.extend((4..7).map(|id| CpuCore { id, capacity: 871 }));
        v.push(CpuCore { id: 7, capacity: 1024 });
        v
    }

    fn big_little_4_4() -> Vec<CpuCore> {
        let mut v = Vec::new();
        v.extend((0..4).map(|id| CpuCore { id, capacity: 378 }));
        v.extend((4..8).map(|id| CpuCore { id, capacity: 1024 }));
        v
    }

    fn homogeneous() -> Vec<CpuCore> {
        (0..8).map(|id| CpuCore { id, capacity: 1024 }).collect()
    }

    #[test]
    fn parse_lists() {
        assert_eq!(parse_cpu_list("4-6"), Some(vec![4, 5, 6]));
        assert_eq!(parse_cpu_list("0-3,7"), Some(vec![0, 1, 2, 3, 7]));
        assert_eq!(parse_cpu_list(" 0-3, 4-6 "), Some((0..=6).collect()));
        assert_eq!(parse_cpu_list("7"), Some(vec![7]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("foo"), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(format_cpu_list(&[4, 5, 6]), "4-6");
        assert_eq!(format_cpu_list(&[0, 1, 2, 3, 7]), "0-3,7");
        assert_eq!(format_cpu_list(&[7]), "7");
    }

    #[test]
    fn s10e_auto_is_three_golds() {
        let topo = s10e();
        assert_eq!(auto_android_cpuset(&topo), Some(vec![4, 5, 6]));
        assert_eq!(
            pick_cpuset(&CpusetSpec::Auto, &topo, true),
            Some(vec![4, 5, 6])
        );
        assert_eq!(pick_cpuset(&CpusetSpec::Auto, &topo, false), None);
        assert_eq!(pick_cpuset(&CpusetSpec::Mid, &topo, true), Some(vec![4, 5, 6]));
        assert_eq!(
            pick_cpuset(&CpusetSpec::Little, &topo, true),
            Some(vec![0, 1, 2, 3])
        );
        assert_eq!(
            pick_cpuset(&CpusetSpec::OffPrime, &topo, true),
            Some(vec![0, 1, 2, 3, 4, 5, 6])
        );
        assert_eq!(pick_cpuset(&CpusetSpec::All, &topo, true), None);
    }

    #[test]
    fn four_plus_four_auto_keeps_bigs() {
        let topo = big_little_4_4();
        // Max cluster is not a singleton, so auto does not drop it.
        assert_eq!(auto_android_cpuset(&topo), Some(vec![4, 5, 6, 7]));
        assert_eq!(
            pick_cpuset(&CpusetSpec::Auto, &topo, true),
            Some(vec![4, 5, 6, 7])
        );
        assert_eq!(
            pick_cpuset(&CpusetSpec::Mid, &topo, true),
            Some(vec![0, 1, 2, 3])
        );
        assert_eq!(pick_cpuset(&CpusetSpec::OffPrime, &topo, true), None);
    }

    #[test]
    fn homogeneous_does_not_pin() {
        let topo = homogeneous();
        assert_eq!(auto_android_cpuset(&topo), None);
        assert_eq!(pick_cpuset(&CpusetSpec::Auto, &topo, true), None);
        assert_eq!(pick_cpuset(&CpusetSpec::Mid, &topo, true), None);
        assert_eq!(pick_cpuset(&CpusetSpec::Little, &topo, true), None);
    }

    #[test]
    fn android_auto_threads_use_full_mid_cluster() {
        let golds = [4, 5, 6];
        assert_eq!(resolve_threads(None, Some(&golds), 8), 3);
        assert_eq!(resolve_threads(Some(2), Some(&golds), 8), 2);
        assert_eq!(resolve_threads(Some(8), Some(&golds), 8), 3);
        assert_eq!(resolve_threads(Some(1), Some(&golds), 8), 1);
        assert_eq!(resolve_threads(None, None, 8), 8);
    }

    #[test]
    fn explicit_filters_to_online() {
        let topo = s10e();
        assert_eq!(
            pick_cpuset(&CpusetSpec::Explicit(vec![4, 5, 6, 99]), &topo, true),
            Some(vec![4, 5, 6])
        );
        assert_eq!(
            pick_cpuset(&CpusetSpec::Explicit(vec![99]), &topo, true),
            None
        );
    }

    #[test]
    fn s1_auto_is_little_on_android_only() {
        let topo = s10e();
        assert_eq!(pick_s1(&CpusetSpec::Auto, &topo, true), Some(vec![0, 1, 2, 3]));
        assert_eq!(pick_s1(&CpusetSpec::Auto, &topo, false), None);
        assert_eq!(pick_s1(&CpusetSpec::Mid, &topo, true), Some(vec![4, 5, 6]));
    }

    #[test]
    fn governor_choose_pool() {
        use GovernorMode::*;
        use PoolKind::*;
        assert_eq!(choose_pool(Off, 100.0, true, Cool), Full);
        assert_eq!(choose_pool(Thermal, 0.0, true, Full), Cool);
        // Pace ignores a thermal hold and follows the buffer.
        assert_eq!(choose_pool(Pace, 0.0, true, Full), Full);
        assert_eq!(choose_pool(Thermal, 6.0, false, Full), Cool);
        // 2.0 s is inside the band: keep the previous pool.
        assert_eq!(choose_pool(Thermal, 2.0, false, Full), Full);
        assert_eq!(choose_pool(Thermal, 3.0, false, Cool), Cool);
        assert_eq!(choose_pool(Thermal, 1.9, false, Cool), Full);
        assert_eq!(choose_pool(Pace, 6.0, false, Full), Cool);
    }

    #[test]
    fn thermal_hold_hysteresis() {
        assert!(update_thermal_hold(false, 80, false));
        assert!(update_thermal_hold(true, 70, false));
        assert!(!update_thermal_hold(true, 60, false));
        assert!(update_thermal_hold(true, 60, true));
        assert!(!update_thermal_hold(false, 70, false));
        assert!(update_thermal_hold(false, 0, true));
        assert!(!update_thermal_hold(false, 0, false));
    }

    #[test]
    fn fp16_filter_is_gemm_only() {
        assert!(fp16_translate_name("/decoder/Conv"));
        assert!(fp16_translate_name("MatMul"));
        assert!(fp16_translate_name("Gemm_1"));
        assert!(!fp16_translate_name("/decoder/Sin"));
        assert!(!fp16_translate_name("Snake"));
        assert!(!fp16_translate_name("SinSq"));
        assert!(!fp16_translate_name("InstanceNorm"));
        assert!(!fp16_translate_name("stft"));
        assert!(!fp16_translate_name("iSTFT"));
        assert!(!fp16_translate_name("Greater"));
        assert!(!fp16_translate_name("Atan"));
        assert!(!fp16_translate_name("Exp"));
        assert!(!fp16_translate_name("m_source"));
        assert!(!fp16_translate_name("Add"));
    }
}
