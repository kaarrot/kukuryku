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
}

impl Stage {
    /// Parse the subgraph, pin each input to its (possibly symbolic) shape
    /// (matched by name so input order is robust), and optimize. Symbolic dims
    /// with a shared name are the same `Symbol`, so tract keeps the length axis
    /// free and one plan serves all lengths; fixed dims specialize as before.
    fn build(path: &Path, spec: &[(&str, &[Dim])]) -> Result<Stage> {
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

        let runnable = model
            .into_optimized()
            .with_context(|| format!("optimizing {}", path.display()))?
            .into_runnable()?;
        Ok(Stage { runnable, input_names })
    }

    /// Run the cached plan; tensors are matched to declared inputs by name.
    fn run(&self, inputs: &[(&str, Tensor)], stage: &str) -> Result<TVec<TValue>> {
        let mut ordered: TVec<TValue> = TVec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let (_, t) = inputs
                .iter()
                .find(|(n, _)| n == name)
                .with_context(|| format!("{stage}: no tensor supplied for input '{name}'"))?;
            ordered.push(t.clone().into());
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

/// Build the tract thread pool (`KOKORO_TRACT_THREADS`, else available cores).
///
/// On Android heterogeneous SoCs the default is the mid cluster (this S10e:
/// 3 threads on cpu4–6), pinned so EAS cannot park a worker on the prime core.
/// Desktop keeps all-cores unless `KOKORO_TRACT_CPUSET` is set. Affinity is
/// applied on this thread *before* the rayon pool is created so workers inherit
/// it. The pool is scoped to stage 2 only; stage 1 stays single-threaded.
fn build_executor() -> tract_linalg::multithread::Executor {
    use tract_linalg::multithread::Executor;
    let spec = cpuset_from_env();
    let topo = read_topology();
    let cpuset = pick_cpuset(&spec, &topo, cfg!(target_os = "android"));
    let env_threads = std::env::var("KOKORO_TRACT_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t > 0);
    let mut threads = env_threads.unwrap_or_else(|| cpuset.as_ref().map(|c| c.len()).unwrap_or_else(nproc));
    if let Some(cs) = cpuset.as_ref() {
        threads = threads.min(cs.len()).max(1);
    }
    if let Some(cs) = cpuset.as_ref() {
        if let Err(e) = apply_affinity(cs) {
            eprintln!(
                "[kokoro] sched_setaffinity({}) failed: {e}; continuing unpinned",
                format_cpu_list(cs)
            );
        }
    }
    if threads > 1 {
        match cpuset.as_ref() {
            Some(cs) => info!(
                "[kokoro] tract executor: {threads} threads, cpuset {}",
                format_cpu_list(cs)
            ),
            None => info!("[kokoro] tract executor: {threads} threads"),
        }
        Executor::multithread(threads)
    } else {
        if let Some(cs) = cpuset.as_ref() {
            info!(
                "[kokoro] tract executor: 1 thread, cpuset {}",
                format_cpu_list(cs)
            );
        }
        Executor::SingleThread
    }
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
    PerShape(HashMap<Vec<usize>, Stage>),
}

impl StagePlan {
    /// Build a single symbolic plan; on optimize failure, degrade to per-shape.
    fn build(path: &Path, spec: &[(&str, &[Dim])], name: &str) -> StagePlan {
        // Debug lever: force the concrete per-shape path (which enables tract's
        // concrete-shape-gated conv fast paths — lazy im2col + depthwise) so we
        // can bench conv run-speed symbolic-vs-concrete. See docs conv section.
        if std::env::var_os("KOKORO_TRACT_FORCE_PERSHAPE").is_some() {
            eprintln!("[kokoro] {name}: FORCE_PERSHAPE — using per-exact-shape plans");
            return StagePlan::PerShape(HashMap::new());
        }
        match Stage::build(path, spec) {
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
                StagePlan::PerShape(HashMap::new())
            }
        }
    }

    /// The plan to run for a given concrete shape: the symbolic plan as-is, or
    /// the cached exact-shape plan (compiled on first sight of that shape).
    fn get(&mut self, path: &Path, concrete: &[(&str, &[usize])]) -> Result<&Stage> {
        match self {
            StagePlan::Symbolic(st) => Ok(st),
            StagePlan::PerShape(cache) => {
                let key: Vec<usize> =
                    concrete.iter().flat_map(|(_, d)| d.iter().copied()).collect();
                if !cache.contains_key(&key) {
                    let owned: Vec<(&str, Vec<Dim>)> = concrete
                        .iter()
                        .map(|(nm, d)| (*nm, d.iter().map(|&v| Fixed(v)).collect()))
                        .collect();
                    let spec: Vec<(&str, &[Dim])> =
                        owned.iter().map(|(nm, d)| (*nm, d.as_slice())).collect();
                    cache.insert(key.clone(), Stage::build(path, &spec)?);
                }
                Ok(&cache[&key])
            }
        }
    }
}

/// The two-stage tract pipeline: one symbolic plan per stage (see [`StagePlan`]),
/// with a Rust length regulator between them.
pub struct Pipeline {
    stage1_path: PathBuf,
    stage2_path: PathBuf,
    executor: tract_linalg::multithread::Executor,
    stage1: StagePlan,
    stage2: StagePlan,
}

impl Pipeline {
    pub fn new(dir: &Path) -> Result<Pipeline> {
        let stage1_path = dir.join("stage1.onnx");
        let stage2_path = dir.join("stage2.onnx");
        let executor = build_executor();
        // Compile each stage once with shared symbolic length dims: N (phoneme
        // count) across stage 1 + the two stage-2 feature tensors, and F (frame
        // count) on the alignment's frame axis.
        //
        // The two compiles are independent, so run them concurrently: stage 2's
        // `into_optimized()` (~3.9s) is the long pole, and building it on a
        // background thread while stage 1 (~1.4s) compiles here drops startup
        // compile wall-time to ~max(the two) instead of the sum. tract's
        // optimizer holds no shared mutable state across models, and the result
        // is bit-identical to sequential compilation — only wall-time changes.
        let stage2_path_bg = stage2_path.clone();
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
            )
        });
        let stage1 = StagePlan::build(
            &stage1_path,
            &[
                ("input_ids", &[Fixed(1), Sym("N")]),
                ("style", &[Fixed(1), Fixed(256)]),
                ("speed", &[Fixed(1)]),
            ],
            "stage1",
        );
        let stage2 = stage2_handle
            .join()
            .map_err(|_| anyhow::anyhow!("stage2 compile thread panicked"))?;
        Ok(Pipeline { stage1_path, stage2_path, executor, stage1, stage2 })
    }

    pub fn synthesize(&mut self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        let style_t = Tensor::from_shape(&[1, style.len()], style)?;

        // ---- Stage 1: encoder + duration predictor (single-threaded) ----
        // Tier 7 Lever 2 A/B'd wrapping this in the stage-2 thread pool: it
        // regressed +31% (8.97s -> 11.75s). The serial LSTM predictor and the
        // small per-op GEMMs pay more in thread-dispatch overhead than the BERT
        // encoder saves, so stage 1 stays single-threaded by design.
        let s1 = self
            .stage1
            .get(&self.stage1_path, &[("input_ids", &[1, n]), ("style", &[1, 256]), ("speed", &[1])])?
            .run(
                &[
                    ("input_ids", Tensor::from_shape(&[1, n], ids)?),
                    ("style", style_t.clone()),
                    ("speed", Tensor::from_shape(&[1], &[speed])?),
                ],
                "stage1",
            )?;
        // Outputs (split_kokoro.py order): [0] 640-ch [1,640,N] [1] 512-ch
        // [1,512,N] [2] durations [1,N].
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

        // ---- Rust length regulator: durations -> alignment matrix -------
        // Round per-phoneme durations to frame counts and build A[N, total_frames]
        // with A[i,t] = 1 iff frame t belongs to phoneme i (block expansion).
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

        // ---- Stage 2: decoder + iSTFTNet vocoder (multithreaded) --------
        let executor = self.executor.clone();
        let stage2 = self.stage2.get(
            &self.stage2_path,
            &[
                (S1_FEAT_640, &[1, 640, n]),
                (S1_FEAT_512, &[1, 512, n]),
                (S2_ALIGNMENT, &[n, total_frames]),
                ("style", &[1, 256]),
            ],
        )?;
        // Scope the thread pool to this run; stage 1 above stays single-threaded.
        let s2 = tract_linalg::multithread::multithread_tract_scope(executor, || {
            stage2.run(
                &[
                    (S1_FEAT_640, feat640),
                    (S1_FEAT_512, feat512),
                    (S2_ALIGNMENT, alignment),
                    ("style", style_t),
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
}
