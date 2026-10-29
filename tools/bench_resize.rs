//! Standalone tests/benchmark of the exact numeric Resize module.
#![allow(dead_code)]
use std::time::Instant;
pub use tract_data as tract_core;
use tract_data::internal::*;

#[path = "../third_party/tract/tract-onnx/src/ops/resize/contiguous.rs"]
mod contiguous;
use contiguous::{CoordTransformer, Interpolator, Nearest, resize_axis_generic, resize_last_axis};

fn cpu_seconds() -> f64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as f64 + t.tv_nsec as f64 / 1e9
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 7,
        "nearest|linear CHANNELS IN OUT SCALE REPEATS ROUNDS"
    );
    let channels: usize = args[1].parse()?;
    let len_in: usize = args[2].parse()?;
    let len_out: usize = args[3].parse()?;
    let scale: f32 = args[4].parse()?;
    let repeats: usize = args[5].parse()?;
    let rounds: usize = args[6].parse()?;
    ensure!(channels > 0 && len_in > 0 && repeats > 0 && rounds >= 2);
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for core in 4..7 {
        unsafe { libc::CPU_SET(core, &mut mask) };
    }
    assert_eq!(
        unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) },
        0
    );
    let linear = args[0] == "linear";
    ensure!(linear || args[0] == "nearest");
    let coord = if linear {
        CoordTransformer::HalfPixel
    } else {
        CoordTransformer::Asymmetric
    };
    let mode = if linear {
        Interpolator::Linear
    } else {
        Interpolator::Nearest
    };
    let mut state = 17u64;
    let input = tract_ndarray::ArrayD::from_shape_fn(&[1, channels, len_in][..], |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((state >> 32) % 2001) as f32 - 1000.0) * 0.003
    });
    let shape = [1, channels, len_out];
    let run = |fast| -> TractResult<tract_ndarray::ArrayD<f32>> {
        if fast {
            Ok(resize_last_axis(
                &input.view(),
                &shape,
                2,
                scale,
                &coord,
                &mode,
                Nearest::Floor,
            )?
            .unwrap())
        } else {
            // Include the input copy performed by the original Resize::eval.
            let copied = input.clone();
            Ok(resize_axis_generic(
                &copied,
                &shape,
                2,
                scale,
                &coord,
                &mode,
                Nearest::Floor,
            ))
        }
    };
    let old = run(false)?;
    let new = run(true)?;
    ensure!(
        old.iter()
            .zip(&new)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Resize bits differ"
    );
    ensure!(new.iter().all(|v| v.is_finite()));
    eprintln!("parity finite=true bit_equal=true samples={}", new.len());
    println!(
        "round,position,variant,mode,channels,input_len,output_len,scale,repeats,wall_s,cpu_s,bit_equal"
    );
    for round in 0..rounds {
        for position in 0..2 {
            let fast = (round + position) % 2 == 1;
            let cpu = cpu_seconds();
            let started = Instant::now();
            for _ in 0..repeats {
                std::hint::black_box(run(fast)?);
            }
            let wall = started.elapsed().as_secs_f64();
            let cpu = cpu_seconds() - cpu;
            let name = if fast { "fast" } else { "generic" };
            println!(
                "{round},{position},{name},{},{channels},{len_in},{len_out},{scale},{repeats},{wall:.9},{cpu:.9},true",
                args[0]
            );
        }
    }
    Ok(())
}
