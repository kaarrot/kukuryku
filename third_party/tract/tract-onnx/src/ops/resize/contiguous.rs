//! Numeric Resize paths, kept independent of model parsing for direct checks.
use crate::tract_core::internal::*;

type F32Array = tract_ndarray::ArrayD<f32>;

#[derive(Clone, Debug, Hash)]
pub(super) enum CoordTransformer {
    HalfPixel,
    AlignCorners,
    Asymmetric,
}

impl CoordTransformer {
    pub(super) fn transform(&self, x_out: usize, scale: f32, len_in: usize, len_out: usize) -> f32 {
        match self {
            CoordTransformer::HalfPixel => (x_out as f32 + 0.5) / scale - 0.5,
            CoordTransformer::AlignCorners => {
                (x_out as f32 * (len_in as f32 - 1.0)) / (len_out as f32 - 1.0)
            }
            CoordTransformer::Asymmetric => (x_out as f32) / scale,
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub(super) enum Interpolator {
    Linear,
    Nearest,
}

impl Interpolator {
    pub(super) fn interpolate(
        &self,
        y_left: f32,
        y_right: f32,
        x_ratio: f32,
        nearest_mode: Nearest,
    ) -> f32 {
        match self {
            Interpolator::Linear => y_left * (1.0 - x_ratio) + y_right * x_ratio,
            Interpolator::Nearest => match nearest_mode {
                Nearest::Floor => y_left,
                Nearest::Ceil => y_right,
                Nearest::RoundPreferFloor => {
                    if x_ratio <= 0.5 {
                        y_left
                    } else {
                        y_right
                    }
                }
                Nearest::RoundPreferCeil => {
                    if x_ratio < 0.5 {
                        y_left
                    } else {
                        y_right
                    }
                }
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Hash)]
pub(super) enum Nearest {
    Floor,
    Ceil,
    RoundPreferFloor,
    RoundPreferCeil,
}

/// The previous per-axis evaluator, including its existing edge semantics.
pub(super) fn resize_axis_generic(
    data: &F32Array,
    new_shape: &[usize],
    axis: usize,
    scale: f32,
    coord: &CoordTransformer,
    interpolator: &Interpolator,
    nearest: Nearest,
) -> F32Array {
    tract_ndarray::ArrayD::from_shape_fn(new_shape, |co_o| -> f32 {
        let x_out = co_o[axis];
        let x_in = coord.transform(x_out, scale, data.shape()[axis], new_shape[axis]);
        let mut co_i = co_o;
        let x_left = (x_in as usize).clamp(0, data.shape()[axis] - 1);
        co_i[axis] = x_left;
        let y_left = data[&co_i];
        let x_right = (x_left + 1).min(data.shape()[axis] - 1);
        co_i[axis] = x_right;
        let y_right = data[&co_i];
        let x_frac = x_in - x_left as f32;
        interpolator.interpolate(y_left, y_right, x_frac, nearest)
    })
}

pub(super) fn fast_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRACT_ONNX_RESIZE_FAST").as_deref() != Ok("0"))
}

/// Specialize the two mode combinations used by Kokoro, on contiguous last-axis
/// rows only. Other modes/layouts retain the generic implementation.
pub(super) fn resize_last_axis<S: tract_ndarray::Data<Elem = f32>>(
    data: &tract_ndarray::ArrayBase<S, tract_ndarray::IxDyn>,
    new_shape: &[usize],
    axis: usize,
    scale: f32,
    coord: &CoordTransformer,
    interpolator: &Interpolator,
    nearest: Nearest,
) -> TractResult<Option<F32Array>> {
    let supported = matches!(
        (coord, interpolator, nearest),
        (
            CoordTransformer::Asymmetric,
            Interpolator::Nearest,
            Nearest::Floor
        ) | (CoordTransformer::HalfPixel, Interpolator::Linear, _)
    );
    if !supported
        || axis + 1 != data.ndim()
        || new_shape.len() != data.ndim()
        || !scale.is_finite()
        || scale <= 0.0
        || data.shape()[..axis] != new_shape[..axis]
    {
        return Ok(None);
    }
    let Some(source) = data.as_slice() else {
        return Ok(None);
    };
    let len_in = data.shape()[axis];
    if len_in == 0 {
        return Ok(None);
    }
    let len_out = new_shape[axis];
    let rows = source.len() / len_in;
    let total = rows
        .checked_mul(len_out)
        .context("Resize output size overflow")?;
    let mut output = Vec::with_capacity(total);
    if total != 0 {
        match interpolator {
            Interpolator::Nearest => {
                // Repeated nearest samples form runs. This keeps the large 300x
                // pitch upsample's map small and lets each run use slice filling.
                let mut runs = Vec::<(usize, usize)>::new();
                for x_out in 0..len_out {
                    let x_in = coord.transform(x_out, scale, len_in, len_out);
                    let left = (x_in as usize).clamp(0, len_in - 1);
                    if let Some((previous, end)) = runs.last_mut() {
                        if *previous == left {
                            *end = x_out + 1;
                            continue;
                        }
                    }
                    runs.push((left, x_out + 1));
                }
                for row in source.chunks_exact(len_in) {
                    let mut start = 0;
                    for &(left, end) in &runs {
                        output.resize(output.len() + (end - start), row[left]);
                        start = end;
                    }
                }
            }
            Interpolator::Linear => {
                let position = |x_out| {
                    let x_in = coord.transform(x_out, scale, len_in, len_out);
                    let left = (x_in as usize).clamp(0, len_in - 1);
                    (left, x_in - left as f32)
                };
                let sample = |row: &[f32], (left, fraction): (usize, f32)| {
                    let right = (left + 1).min(len_in - 1);
                    // Keep the original arithmetic association, including negative
                    // half-pixel fractions and the right-edge zero-weight multiply.
                    row[left] * (1.0 - fraction) + row[right] * fraction
                };
                if rows == 1 {
                    output.extend((0..len_out).map(|x| sample(source, position(x))));
                } else {
                    // Reuse a small map across all rows before moving to the next
                    // block. A 300x upsample must not allocate a multi-megabyte
                    // position table or stream it repeatedly through the caches.
                    const BLOCK: usize = 1024;
                    let mut positions = Vec::with_capacity(len_out.min(BLOCK));
                    output.resize(total, 0.0);
                    for start in (0..len_out).step_by(BLOCK) {
                        let width = (len_out - start).min(BLOCK);
                        let end = start + width;
                        positions.clear();
                        positions.extend((start..end).map(position));
                        for (r, row) in source.chunks_exact(len_in).enumerate() {
                            let offset = r * len_out + start;
                            for (dst, &p) in
                                output[offset..offset + width].iter_mut().zip(&positions)
                            {
                                *dst = sample(row, p);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Some(tract_ndarray::ArrayD::from_shape_vec(
        new_shape, output,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(shape: &[usize]) -> F32Array {
        let mut state = 17u64;
        F32Array::from_shape_fn(shape, |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((state >> 32) % 2001) as f32 - 1000.0) * 0.003
        })
    }

    fn assert_bits(actual: &F32Array, expected: &F32Array) {
        assert_eq!(actual.shape(), expected.shape());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "sample {i}: {a} versus {b}");
        }
    }

    fn check(shape: &[usize], out_len: usize, scale: f32, linear: bool) {
        let data = input(shape);
        let axis = shape.len() - 1;
        let mut output_shape = shape.to_vec();
        output_shape[axis] = out_len;
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
        let reference = resize_axis_generic(
            &data,
            &output_shape,
            axis,
            scale,
            &coord,
            &mode,
            Nearest::Floor,
        );
        let actual = resize_last_axis(
            &data,
            &output_shape,
            axis,
            scale,
            &coord,
            &mode,
            Nearest::Floor,
        )
        .unwrap()
        .unwrap();
        assert_bits(&actual, &reference);
    }

    #[test]
    fn short_rows_and_non_integer_scales_are_bit_exact() {
        for rows in [1, 3, 9] {
            for len in [1, 2, 3, 13] {
                for scale in [0.25, 0.5, 1.1, 2.0, 3.7, 300.0] {
                    let out = (len as f32 * scale) as usize;
                    for linear in [false, true] {
                        check(&[rows, len], out, scale, linear);
                    }
                }
            }
        }
    }

    #[test]
    fn actual_model_resize_shapes_are_bit_exact() {
        for frames in [59, 268, 585] {
            check(&[1, 512, frames], 2 * frames, 2.0, false);
            check(&[1, 1090, frames], 2 * frames, 2.0, false);
            check(&[1, 1, 2 * frames], 600 * frames, 300.0, false);
            check(&[1, 9, 600 * frames], 2 * frames, 1.0 / 300.0, true);
            check(&[1, 9, 2 * frames], 600 * frames, 300.0, true);
        }
    }

    #[test]
    fn linear_block_boundaries_are_bit_exact() {
        for out_len in [1023, 1024, 1025, 2048, 2049] {
            check(&[3, 7], out_len, out_len as f32 / 7.0, true);
        }
    }

    #[test]
    fn batch_rank_and_size_derived_scales_are_bit_exact() {
        check(&[2, 3, 7], 19, 19.0 / 7.0, true);
        check(&[2, 3, 7], 19, 19.0 / 7.0, false);
        check(&[2, 2, 3, 5], 3, 3.0 / 5.0, true);
        check(&[13], 7, 7.0 / 13.0, true);
    }

    #[test]
    fn preserve_legacy_negative_half_pixel_fraction() {
        let data = F32Array::from_shape_vec(&[1, 2][..], vec![2.0, 10.0]).unwrap();
        let actual = resize_last_axis(
            &data,
            &[1, 4],
            1,
            2.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        )
        .unwrap()
        .unwrap();
        // Existing behavior extrapolates at x=-0.25 rather than clamping x to zero.
        assert_eq!(actual.as_slice().unwrap(), &[0.0, 4.0, 8.0, 10.0]);
    }

    #[test]
    fn nearest_copies_signed_zero_subnormals_and_nan_payloads() {
        let values = [0x80000000, 0, 1, 0x7fc01234, 0x7f800000, 0xff800000]
            .map(f32::from_bits)
            .to_vec();
        let data = F32Array::from_shape_vec(&[1, values.len()][..], values).unwrap();
        let reference = resize_axis_generic(
            &data,
            &[1, 18],
            1,
            3.0,
            &CoordTransformer::Asymmetric,
            &Interpolator::Nearest,
            Nearest::Floor,
        );
        let actual = resize_last_axis(
            &data,
            &[1, 18],
            1,
            3.0,
            &CoordTransformer::Asymmetric,
            &Interpolator::Nearest,
            Nearest::Floor,
        )
        .unwrap()
        .unwrap();
        assert_bits(&actual, &reference);
    }

    #[test]
    fn linear_keeps_zero_weight_multiply_and_non_finite_classification() {
        for values in [
            vec![1.0, f32::INFINITY],
            vec![-0.0, 0.0],
            vec![f32::NAN, -1.0],
        ] {
            let data = F32Array::from_shape_vec(&[1, 2][..], values).unwrap();
            let reference = resize_axis_generic(
                &data,
                &[1, 4],
                1,
                2.0,
                &CoordTransformer::HalfPixel,
                &Interpolator::Linear,
                Nearest::Floor,
            );
            let actual = resize_last_axis(
                &data,
                &[1, 4],
                1,
                2.0,
                &CoordTransformer::HalfPixel,
                &Interpolator::Linear,
                Nearest::Floor,
            )
            .unwrap()
            .unwrap();
            for (a, b) in actual.iter().zip(&reference) {
                if b.is_nan() {
                    assert!(a.is_nan());
                } else {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }

    #[test]
    fn empty_output_and_empty_outer_dimensions_are_preserved() {
        for linear in [false, true] {
            check(&[2, 3], 0, 0.1, linear);
            check(&[0, 3], 6, 2.0, linear);
        }
    }

    #[test]
    fn unsupported_modes_layouts_and_axes_decline_fast_path() {
        let data = input(&[2, 3, 4]);
        for (coord, mode, nearest) in [
            (
                CoordTransformer::AlignCorners,
                Interpolator::Linear,
                Nearest::Floor,
            ),
            (
                CoordTransformer::Asymmetric,
                Interpolator::Linear,
                Nearest::Floor,
            ),
            (
                CoordTransformer::HalfPixel,
                Interpolator::Nearest,
                Nearest::Floor,
            ),
            (
                CoordTransformer::Asymmetric,
                Interpolator::Nearest,
                Nearest::Ceil,
            ),
            (
                CoordTransformer::Asymmetric,
                Interpolator::Nearest,
                Nearest::RoundPreferFloor,
            ),
            (
                CoordTransformer::Asymmetric,
                Interpolator::Nearest,
                Nearest::RoundPreferCeil,
            ),
        ] {
            assert!(
                resize_last_axis(&data, &[2, 3, 8], 2, 2.0, &coord, &mode, nearest)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            resize_last_axis(
                &data,
                &[2, 6, 4],
                1,
                2.0,
                &CoordTransformer::Asymmetric,
                &Interpolator::Nearest,
                Nearest::Floor
            )
            .unwrap()
            .is_none()
        );
        let strided = data.permuted_axes(vec![0, 2, 1]);
        assert!(strided.as_slice().is_none());
        assert!(
            resize_last_axis(
                &strided,
                &[2, 4, 6],
                2,
                2.0,
                &CoordTransformer::Asymmetric,
                &Interpolator::Nearest,
                Nearest::Floor
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn generic_earlier_axis_then_fast_last_axis_preserves_order() {
        let data = input(&[1, 2, 3]);
        let middle = resize_axis_generic(
            &data,
            &[1, 4, 3],
            1,
            2.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        );
        let reference = resize_axis_generic(
            &middle,
            &[1, 4, 6],
            2,
            2.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        );
        let actual = resize_last_axis(
            &middle,
            &[1, 4, 6],
            2,
            2.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        )
        .unwrap()
        .unwrap();
        assert_bits(&actual, &reference);
    }

    #[test]
    fn borrowed_input_view_is_bit_exact() {
        let data = input(&[1, 9, 118]);
        let reference = resize_axis_generic(
            &data,
            &[1, 9, 35400],
            2,
            300.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        );
        let actual = resize_last_axis(
            &data.view(),
            &[1, 9, 35400],
            2,
            300.0,
            &CoordTransformer::HalfPixel,
            &Interpolator::Linear,
            Nearest::Floor,
        )
        .unwrap()
        .unwrap();
        assert_bits(&actual, &reference);
    }
}
