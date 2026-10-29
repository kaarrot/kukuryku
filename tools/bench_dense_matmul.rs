//! Stage-1 dense-matmul probe, linked to cached tract-linalg/data release rlibs.
use tract_data::internal::*;
use tract_linalg::mmm::{AsInputValue, FusedSpec, MatMatMul, MMMInputValue, ScratchSpace};
use std::time::Instant;

fn cpu_seconds() -> f64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) }, 0);
    t.tv_sec as f64 + t.tv_nsec as f64 / 1e9
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
    a: Box<dyn MMMInputValue>,
    out: Tensor,
    scratch: Box<dyn ScratchSpace>,
    m: usize,
    n: usize,
}

impl Case {
    fn new(kernel: Box<dyn MatMatMul>, a: &Tensor, m: usize, n: usize) -> TractResult<Self> {
        let a = kernel.packings()[0].0.prepare_one(a, 1, 0)?;
        let out = Tensor::zero::<f16>(&[n, m])?;
        let scratch = unsafe { kernel.allocate_scratch_space() };
        Ok(Self { kernel, a, out, scratch, m, n })
    }

    fn run(&mut self, b: &dyn MMMInputValue, bias: &Tensor) -> TractResult<()> {
        unsafe {
            // BERT's output is [tokens, channels], with M on axis 1 and N on 0.
            let store = self.kernel.c_view(Some(1), Some(0)).wrap(&self.out.view_mut());
            let ops = [
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&*self.a),
                    b: AsInputValue::Borrowed(b), packing: 0,
                },
                FusedSpec::BinPerRow(bias.view(), tract_linalg::BinOp::Add),
                FusedSpec::Store(store),
            ];
            self.kernel.run_with_scratch_space(self.m, self.n, &mut *self.scratch, &ops)
        }
    }
}

fn main() -> TractResult<()> {
    let args: Vec<usize> = std::env::args().skip(1).map(|s|s.parse()).collect::<Result<_, _>>()?;
    ensure!(args.len()==5, "M K N CORE ROUNDS");
    let [m,k,n,core,rounds] = args[..] else { unreachable!() };
    ensure!(m>0 && k>0 && n>0 && core<8 && rounds>0);
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::CPU_SET(core,&mut mask) };
    assert_eq!(unsafe { libc::sched_setaffinity(0,std::mem::size_of_val(&mask),&mask) },0);
    let a = Tensor::from_shape(&[m,k], &values(m*k,17,0.0002))?;
    let b = Tensor::from_shape(&[n,k], &values(n*k,41,0.001))?;
    let bias = Tensor::from_shape(&[m], &values(m,91,0.0001))?;
    let names = [
        "arm64fp16_mmm_f16_64x3_gen", "arm64fp16_mmm_f16_16x8_gen",
        "arm64fp16_mmm_f16_16x8_a55", "arm64fp16_mmm_f16_32x4_gen",
        "arm64fp16_mmm_f16_32x6_gen", "arm64fp16_mmm_f16_32x4_a55",
    ];
    let mut cases=Vec::new();
    for name in names {
        let kernel=tract_linalg::ops().mmm_impls().iter().find(|c|c.name()==name).unwrap().clone();
        let mut case=Case::new(kernel,&a,m,n)?;
        let pb=case.kernel.packings()[0].1.prepare_one(&b,1,0)?;
        case.run(&*pb,&bias)?;
        cases.push(case);
    }
    let expected=cases[0].out.as_slice::<f16>()?.to_vec();
    let mut parity=Vec::new();
    for c in &cases {
        let got=c.out.as_slice::<f16>()?;
        let unequal=expected.iter().zip(got).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        let max_abs=expected.iter().zip(got).map(|(a,b)|(a.to_f32()-b.to_f32()).abs()).fold(0f32,f32::max);
        let finite=got.iter().all(|v|v.is_finite());
        eprintln!("parity {} unequal={unequal}/{} max_abs={max_abs} finite={finite}",c.kernel.name(),expected.len());
        parity.push(unequal==0 && finite);
    }
    println!("round,position,kernel,m,k,n,core,pack_wall_s,pack_cpu_s,matmul_wall_s,matmul_cpu_s,total_wall_s,total_cpu_s,bit_equal");
    for round in 0..rounds {
        for position in 0..cases.len() {
            let index=(round+position)%cases.len();
            let c=&mut cases[index];
            let total_cpu=cpu_seconds(); let total_t=Instant::now();
            let pb=c.kernel.packings()[0].1.prepare_one(&b,1,0)?;
            let pack_wall=total_t.elapsed().as_secs_f64();
            let pack_cpu=cpu_seconds()-total_cpu;
            let compute_cpu=cpu_seconds(); let compute_t=Instant::now();
            c.run(&*pb,&bias)?;
            let matmul_wall=compute_t.elapsed().as_secs_f64();
            let matmul_cpu=cpu_seconds()-compute_cpu;
            let total_wall=total_t.elapsed().as_secs_f64();
            let total_cpu=cpu_seconds()-total_cpu;
            println!("{round},{position},{},{m},{k},{n},{core},{pack_wall:.9},{pack_cpu:.9},{matmul_wall:.9},{matmul_cpu:.9},{total_wall:.9},{total_cpu:.9},{}",c.kernel.name(),parity[index]);
        }
    }
    Ok(())
}
