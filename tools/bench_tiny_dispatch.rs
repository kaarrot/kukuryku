//! Compare real matmul/row work on the caller versus the existing three-worker pool.
//! Build with build_matmul_probe.py --dispatch-only. Times include a worker tail.
#![allow(dead_code)]
use std::time::{Duration, Instant};
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul, MMMInputValue, ScratchSpace};
use tract_linalg::multithread::{Executor, multithread_tract_scope, par_chunks_mut};

// Exact private row functions extracted from the current math/mod.rs by the builder.
include!(env!("TRACT_DISPATCH_ROW_PROBE"));

fn cpu_seconds() -> f64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) }, 0);
    t.tv_sec as f64 + t.tv_nsec as f64 / 1e9
}

fn pin() {
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for core in 4..7 { unsafe { libc::CPU_SET(core, &mut mask) }; }
    assert_eq!(unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) }, 0);
}

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((state >> 32) as u32 % 2001) as f32 - 1000.0) * 0.0002
    }).collect()
}

fn bits(t: &Tensor) -> TractResult<Vec<u32>> {
    if t.datum_type() == DatumType::F16 {
        let values = t.as_slice::<f16>()?;
        ensure!(values.iter().all(|v| v.is_finite()));
        Ok(values.iter().map(|v| v.to_bits() as u32).collect())
    } else {
        let values = t.as_slice::<f32>()?;
        ensure!(values.iter().all(|v| v.is_finite()));
        Ok(values.iter().map(|v| v.to_bits()).collect())
    }
}

trait Task {
    fn reset(&mut self) -> TractResult<()> { Ok(()) }
    fn run(&mut self) -> TractResult<()>;
    fn output(&self) -> &Tensor;
}

struct Matmul {
    kernel: Box<dyn MatMatMul>,
    a: Box<dyn MMMInputValue>,
    b: Box<dyn MMMInputValue>,
    bias: Tensor,
    out: Tensor,
    scratch: Box<dyn ScratchSpace>,
    m: usize,
    n: usize,
}

impl Matmul {
    fn new(name: &str, m: usize, k: usize, n: usize) -> TractResult<Self> {
        let name = match name {
            "f16vec" => "arm64fp16_mmm_f16_128x1_gen",
            "f16mat" => "arm64fp16_mmm_f16_64x3_gen",
            "f32vec" => "arm64simd_mmm_f32_32x1_gen",
            "f32vec64" => "arm64simd_mmm_f32_64x1_gen",
            name => name,
        };
        let kernel = tract_linalg::ops().mmm_impls().iter()
            .find(|v| v.name() == name).with_context(|| format!("Missing kernel {name}"))?.clone();
        let dt = kernel.internal_type();
        let a = Tensor::from_shape(&[m, k], &values(m * k, 17))?.cast_to_dt(dt)?.into_owned();
        let b = Tensor::from_shape(&[n, k], &values(n * k, 41))?.cast_to_dt(dt)?.into_owned();
        let a = kernel.packings()[0].0.prepare_one(&a, 1, 0)?;
        let b = kernel.packings()[0].1.prepare_one(&b, 1, 0)?;
        let bias = Tensor::from_shape(&[m], &values(m, 91))?.cast_to_dt(dt)?.into_owned();
        let out = Tensor::zero_dt(dt, &[n, m])?;
        let scratch = unsafe { kernel.allocate_scratch_space() };
        eprintln!("kernel={} mr={} nr={}", kernel.name(), kernel.mr(), kernel.nr());
        Ok(Self { kernel, a, b, bias, out, scratch, m, n })
    }
}

impl Task for Matmul {
    fn run(&mut self) -> TractResult<()> {
        unsafe {
            let out = self.kernel.c_view(Some(1), Some(0)).wrap(&self.out.view_mut());
            let ops = [
                FusedSpec::AddMatMul { a: AsInputValue::Borrowed(&*self.a),
                    b: AsInputValue::Borrowed(&*self.b), packing: 0 },
                FusedSpec::BinPerRow(self.bias.view(), tract_linalg::BinOp::Add),
                FusedSpec::Store(out),
            ];
            self.kernel.run_with_scratch_space(self.m, self.n, &mut *self.scratch, &ops)
        }
    }
    fn output(&self) -> &Tensor { &self.out }
}

struct Rows {
    kind: String,
    frames: usize,
    initial: Tensor,
    data: Tensor,
}

impl Rows {
    fn new(kind: &str, channels: usize, frames: usize) -> TractResult<Self> {
        ensure!(["snake", "adain", "adain16", "leaky", "leaky16"].contains(&kind));
        let dt = if kind.ends_with("16") { DatumType::F16 } else { DatumType::F32 };
        let initial = Tensor::from_shape(&[channels, frames], &values(channels * frames, 113))?
            .cast_to_dt(dt)?.into_owned();
        Ok(Self { kind: kind.into(), frames, data: initial.clone(), initial })
    }
}

impl Task for Rows {
    fn reset(&mut self) -> TractResult<()> {
        if self.data.datum_type() == DatumType::F16 {
            self.data.as_slice_mut::<f16>()?.copy_from_slice(self.initial.as_slice::<f16>()?);
        } else {
            self.data.as_slice_mut::<f32>()?.copy_from_slice(self.initial.as_slice::<f32>()?);
        }
        Ok(())
    }
    fn run(&mut self) -> TractResult<()> {
        use tract_data::half::slice::HalfFloatSliceExt;
        let f = self.frames;
        let act = if self.kind.starts_with("leaky") { RowAct::LeakyRelu(0.2) }
                  else { RowAct::Snake { a: 1.25, inv_a: 0.8 } };
        if self.data.datum_type() == DatumType::F16 {
            par_chunks_mut(self.data.as_slice_mut::<f16>()?, f, |r, row| {
                with_row_scratch(f, (r * f) % 4, |scratch| {
                    row.convert_to_f32_slice(scratch);
                    adain_row(scratch, 1.0 / f as f32, 1.0 / f as f32, 1e-5, 1.1, 0.1, act);
                    row.convert_from_f32_slice(scratch);
                });
            });
        } else if self.kind == "snake" {
            par_chunks_mut(self.data.as_slice_mut::<f32>()?, f, |_, row| {
                for x in row { *x = snake_f32(*x, 1.25, 0.8); }
            });
        } else {
            par_chunks_mut(self.data.as_slice_mut::<f32>()?, f, |_, row| {
                adain_row(row, 1.0 / f as f32, 1.0 / f as f32, 1e-5, 1.1, 0.1, act);
            });
        }
        Ok(())
    }
    fn output(&self) -> &Tensor { &self.data }
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(args.len() == 7, "matmul KERNEL M K N REPEATS ROUNDS | rows KIND C F UNUSED REPEATS ROUNDS");
    let numbers: Vec<usize> = args[2..].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let [a, b, c, repeats, rounds] = numbers[..] else { unreachable!() };
    ensure!(a > 0 && b > 0 && repeats > 0 && rounds > 0);
    pin();
    let serial = Executor::SingleThread;
    let pool = Executor::multithread(3);
    let mut task: Box<dyn Task> = match args[0].as_str() {
        "matmul" => { ensure!(c > 0); Box::new(Matmul::new(&args[1], a, b, c)?) },
        "rows" => Box::new(Rows::new(&args[1], a, b)?),
        _ => bail!("Expected matmul or rows"),
    };
    task.reset()?;
    multithread_tract_scope(serial.clone(), || task.run())?;
    let expected = bits(task.output())?;
    task.reset()?;
    multithread_tract_scope(pool.clone(), || task.run())?;
    ensure!(bits(task.output())? == expected, "Serial/pool output bits differ");
    eprintln!("parity: finite=true bit_equal=true elements={}", expected.len());
    // Warm each executor, including worker-local row and matmul scratch.
    for executor in [&serial, &pool] {
        task.reset()?;
        multithread_tract_scope(executor.clone(), || -> TractResult<()> {
            for _ in 0..3 { task.run()?; }
            Ok(())
        })?;
    }
    println!("round,position,executor,kind,label,a,b,c,repeats,work_wall_s,cpu_including_tail_s,bit_equal");
    for round in 0..rounds {
        for position in 0..2 {
            let is_pool = (round + position) % 2 == 1;
            let executor = if is_pool { pool.clone() } else { serial.clone() };
            task.reset()?;
            // Allow workers from the previous arm to park before starting CPU accounting.
            std::thread::sleep(Duration::from_millis(100));
            let cpu = cpu_seconds();
            let started = Instant::now();
            multithread_tract_scope(executor, || -> TractResult<()> {
                for _ in 0..repeats { task.run()?; }
                Ok(())
            })?;
            let work_wall = started.elapsed().as_secs_f64();
            // Count spinning/searching after the last task as well as its worker compute.
            std::thread::sleep(Duration::from_millis(30));
            let cpu = cpu_seconds() - cpu;
            let name = if is_pool { "pool3" } else { "serial" };
            println!("{round},{position},{name},{},{},{a},{b},{c},{repeats},{work_wall:.9},{cpu:.9},true", args[0], args[1]);
        }
    }
    Ok(())
}
