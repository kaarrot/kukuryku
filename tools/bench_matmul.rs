//! Standalone probe for the cached tract libraries. Build with build_matmul_probe.py.
//! Uses the repository's lazy im2col implementation and normal MMM executor.
#![allow(dead_code, unused_imports)]

pub use tract_core::{internal, ops};
use internal::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul, MMMInputValue, ScratchSpace};
use tract_linalg::multithread::{Executor, multithread_tract_scope};
use tract_linalg::pack::PackedFormat;

// Keep gather behavior identical to the current repository. These types are
// private in tract-core, so compile its source into this diagnostic executable.
#[path = "../third_party/tract/tract-core/src/ops/cnn/conv/lazy_im2col.rs"]
mod lazy_im2col;

#[cfg(probe_inventory)]
mod backend {
    include!(env!("MATMUL_BACKEND_PROBE"));
}

fn cpu_seconds() -> f64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) }, 0);
    t.tv_sec as f64 + t.tv_nsec as f64 / 1e9
}

fn pin(threads: usize) {
    assert!((1..=3).contains(&threads), "use 1..3 gold workers");
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for cpu in 4..4 + threads {
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    assert_eq!(unsafe {
        libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set)
    }, 0, "could not pin probe to the gold cores");
}

fn values(len: usize, seed: u64, scale: f32) -> Vec<f16> {
    let mut state = seed;
    (0..len).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        f16::from_f32((((state >> 32) as u32 % 2001) as f32 - 1000.0) * scale)
    }).collect()
}

struct Case {
    kernel: Box<dyn MatMatMul>,
    packed_a: Box<dyn MMMInputValue>,
    lazy_b: TValue,
    out: Tensor,
    scratch: Box<dyn ScratchSpace>,
    m: usize,
    n: usize,
}

impl Case {
    fn new(kernel: Box<dyn MatMatMul>, a: &Tensor, x: &TValue,
           m: usize, ci: usize, kw: usize, dilation: usize, n: usize) -> TractResult<Self> {
        use lazy_im2col::{LazyIm2Col, LazyIm2colParams, LazyParams};
        let packed_a = kernel.packings()[0].0.prepare_one(a, 1, 0)?;
        let packer = kernel.packings()[0].1.downcast_ref::<PackedFormat>().unwrap().clone();
        let pad = (kw - 1) * dilation / 2;
        let tap_valid: Vec<std::ops::Range<isize>> = (0..kw).map(|tap| {
            let dx = (tap * dilation) as isize - pad as isize;
            (-dx).max(0).min(n as isize)..(n as isize - dx).max(0).min(n as isize)
        }).collect();
        let lo = tap_valid.iter().map(|r| r.start).max().unwrap();
        let hi = tap_valid.iter().map(|r| r.end).min().unwrap();
        let params = LazyIm2colParams {
            packer,
            n_byte_offsets: (0..n).map(|i| (i * 2) as isize).collect(),
            k_byte_offsets: (0..ci).flat_map(|c| {
                (0..kw).map(move |t| ((c * n + t * dilation) as isize - pad as isize) * 2)
            }).collect(),
            tap_valid, interior: lo..hi.max(lo),
        };
        let op = LazyIm2Col { params: LazyParams::Ready(Arc::new(params)) };
        let lazy_b = op.eval(tvec!(x.clone()))?.remove(0);
        let out = Tensor::zero::<f16>(&[m, n])?;
        let scratch = unsafe { kernel.allocate_scratch_space() };
        Ok(Self { kernel, packed_a, lazy_b, out, scratch, m, n })
    }

    fn run(&mut self, bias: &Tensor, residual: Option<&Tensor>) -> TractResult<()> {
        let opaque = &self.lazy_b.as_slice::<Opaque>()?[0];
        let b = opaque.downcast_ref::<Box<dyn MMMInputValue>>().unwrap();
        unsafe {
            let store = self.kernel.c_view(Some(0), Some(1));
            let output = store.wrap(&self.out.view_mut());
            let mut ops = vec![
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&*self.packed_a),
                    b: AsInputValue::Borrowed(&**b), packing: 0,
                },
                FusedSpec::BinPerRow(bias.view(), tract_linalg::BinOp::Add),
            ];
            if let Some(r) = residual {
                ops.push(FusedSpec::AddUnicast(store.wrap(&r.view())));
            }
            ops.push(FusedSpec::Store(output));
            self.kernel.run_with_scratch_space(self.m, self.n, &mut *self.scratch, &ops)
        }
    }
}

fn bench(args: &[String]) -> TractResult<()> {
    ensure!(args.len() == 8, "bench M CI KW DILATION N THREADS ROUNDS RESIDUAL(0|1)");
    let p: Vec<usize> = args.iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let [m, ci, kw, dilation, n, threads, rounds, residual] = p[..] else { unreachable!() };
    ensure!(kw > 0 && kw % 2 == 1 && m > 0 && ci > 0 && n > 0 && dilation > 0 && rounds > 0);
    pin(threads);
    let executor = if threads == 1 { Executor::SingleThread } else { Executor::multithread(threads) };
    let a = Tensor::from_shape(&[m, ci * kw], &values(m * ci * kw, 17, 0.0002))?;
    let x = Tensor::from_shape(&[1, ci, n], &values(ci * n, 41, 0.001))?.into_tvalue();
    let bias = Tensor::from_shape(&[m], &values(m, 91, 0.0001))?;
    let residual = if residual != 0 {
        Some(Tensor::from_shape(&[m, n], &values(m * n, 71, 0.0003))?)
    } else { None };
    let names = [
        "arm64fp16_mmm_f16_16x8_gen", "arm64fp16_mmm_f16_32x4_gen",
        "arm64fp16_mmm_f16_32x6_gen", "arm64fp16_mmm_f16_64x3_gen",
        "arm64fp16_mmm_f16_16x8_a55", "arm64fp16_mmm_f16_32x4_a55",
    ];
    let mut cases = Vec::new();
    for name in names {
        let kernel = tract_linalg::ops().mmm_impls().iter()
            .find(|k| k.name() == name).with_context(|| format!("kernel missing: {name}"))?.clone();
        cases.push(Case::new(kernel, &a, &x, m, ci, kw, dilation, n)?);
    }
    // Warm every variant, including its per-thread scratch, before measuring.
    let warm = Instant::now();
    for c in &mut cases {
        multithread_tract_scope(executor.clone(), || c.run(&bias, residual.as_ref()))?;
    }
    let expected = cases[0].out.as_slice::<f16>()?.to_vec();
    let mut parity = Vec::new();
    for c in &cases {
        let got = c.out.as_slice::<f16>()?;
        let unequal = expected.iter().zip(got).filter(|(a,b)| a.to_bits()!=b.to_bits()).count();
        let max_abs = expected.iter().zip(got)
            .map(|(a,b)| (a.to_f32()-b.to_f32()).abs()).fold(0f32, f32::max);
        let finite = got.iter().all(|v| v.is_finite());
        eprintln!("parity {} unequal={unequal}/{} max_abs={max_abs} finite={finite}",
                  c.kernel.name(), expected.len());
        parity.push(unequal == 0 && finite);
    }
    eprintln!("setup_and_warm_s={:.3}",warm.elapsed().as_secs_f64());
    // Rotating the order gives every variant each position once in six rounds.
    println!("round,position,kernel,m,ci,kw,dilation,n,threads,residual,wall_s,cpu_s,bit_equal");
    for round in 0..rounds {
        for position in 0..cases.len() {
            let index = (round + position) % cases.len();
            let c = &mut cases[index];
            let cpu = cpu_seconds();
            let start = Instant::now();
            multithread_tract_scope(executor.clone(), || c.run(&bias, residual.as_ref()))?;
            let wall_s = start.elapsed().as_secs_f64();
            let cpu_s = cpu_seconds() - cpu;
            println!("{round},{position},{},{m},{ci},{kw},{dilation},{n},{threads},{},{wall_s:.9},{cpu_s:.9},{}",
                     c.kernel.name(),residual.is_some(),parity[index]);
        }
    }
    Ok(())
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        #[cfg(probe_inventory)]
        Some("inventory") if args.len() == 3 => backend::inventory(Path::new(&args[1]), Path::new(&args[2])),
        _ => bail!("use bench M CI KW DILATION N THREADS ROUNDS RESIDUAL, or inventory ASSETS OUTPUT_JSON"),
    }
}
