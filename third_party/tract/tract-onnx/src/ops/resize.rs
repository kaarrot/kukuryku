use crate::model::ParsingContext;
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;
use tract_nnef::tract_num_traits::Zero;

mod contiguous;
use contiguous::{CoordTransformer, Interpolator, Nearest, resize_axis_generic, resize_last_axis};

pub fn resize(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = match ctx.onnx_operator_set_version {
        10 => resize_10(node)?,
        11..=12 => resize_11(node)?,
        13..=17 => resize_13(node)?,
        18.. => resize_18(node)?,
        v => bail!("Unsupported operator set for Resize operator ({v})"),
    };
    Ok((Box::new(op), vec![]))
}

fn resize_10(node: &NodeProto) -> TractResult<Resize> {
    Ok(Resize {
        axes: None,
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_11(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(3);
    Ok(Resize {
        axes: None,
        optional_roi_input: Some(1),
        optional_scales_input: Some(2),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_13(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: None,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_18(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: node.get_attr_opt_vec("axes")?,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

impl CoordTransformer {
    fn from_node(node: &NodeProto) -> TractResult<CoordTransformer> {
        Ok(
            match node
                .get_attr_opt("coordinate_transformation_mode")?
                .unwrap_or("half_pixel")
            {
                "align_corners" => CoordTransformer::AlignCorners,
                "half_pixel" => CoordTransformer::HalfPixel,
                "asymmetric" => CoordTransformer::Asymmetric,
                s => bail!("coordinate_transformation_mode: {}", s),
            },
        )
    }
}

impl Interpolator {
    fn from_node(node: &NodeProto) -> TractResult<Interpolator> {
        Ok(match node.get_attr_opt("mode")?.unwrap_or("nearest") {
            "nearest" => Interpolator::Nearest,
            "linear" => Interpolator::Linear,
            s => bail!("mode: {}", s),
        })
    }
}

impl Nearest {
    fn from_node(node: &NodeProto) -> TractResult<Nearest> {
        Ok(
            match node
                .get_attr_opt("nearest_mode")?
                .unwrap_or("round_prefer_floor")
            {
                "floor" => Nearest::Floor,
                "ceil" => Nearest::Ceil,
                "round_prefer_floor" => Nearest::RoundPreferFloor,
                "round_prefer_ceil" => Nearest::RoundPreferCeil,
                s => bail!("nearest_mode: {}", s),
            },
        )
    }
}

#[derive(Clone, new, Debug, Hash)]
struct Resize {
    axes: Option<Vec<i64>>,
    coord_transformer: CoordTransformer,
    interpolator: Interpolator,
    nearest: Nearest,
    optional_roi_input: Option<usize>,
    optional_scales_input: Option<usize>,
    optional_sizes_input: Option<usize>,
}

impl Op for Resize {
    fn name(&self) -> StaticName {
        "Resize".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "coord={:?} mode={:?} nearest={:?} fast_last_axis={}",
            self.coord_transformer,
            self.interpolator,
            self.nearest,
            contiguous::fast_enabled()
        )])
    }

    op_as_typed_op!();
}

impl Resize {
    fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        input_scale: Option<&Tensor>,
        input_sizes: Option<&Tensor>,
    ) -> TractResult<TVec<D>> {
        if let Some(scale) = input_scale {
            if scale.len() == input_shape.len() {
                let mut shape = tvec!();
                for (i, s) in input_shape
                    .iter()
                    .zip(scale.cast_to::<f32>()?.as_slice::<f32>()?.iter())
                {
                    if s.round() == *s {
                        shape.push(i.clone() * (*s as usize));
                    } else if let Ok(i) = i.to_usize() {
                        shape.push(((i as f32 * s) as usize).into());
                    } else if *s < 1.0 && (1.0 / s).round() == 1.0 / s {
                        // Symbolic input dim downsampled by a unit fraction 1/k:
                        // the output is i/k, which TDim division simplifies exactly
                        // (Kokoro's harmonic source resamples 600*F -> (600*F)/300
                        // = 2*F). Keeps the length axis symbolic so one optimized
                        // plan serves all frame counts. See docs/tract-support-plan.md.
                        shape.push(i.clone() / (1.0 / s).round() as usize);
                    } else {
                        bail!(
                            "Can not compute output shape. inputs are {input_shape:?} and scale {scale:?}"
                        )
                    }
                }
                return Ok(shape);
            }
        }
        if let Some(sizes) = input_sizes {
            if sizes.len() == input_shape.len() {
                return sizes
                    .cast_to::<TDim>()?
                    .as_slice::<TDim>()?
                    .iter()
                    .map(|i| i.try_into())
                    .collect();
            }
        }
        bail!(
            "Neither sizes nor scales makes sense: input_shape: {:?}, scale: {:?}, sizes: {:?}",
            input_shape,
            input_scale,
            input_sizes,
        );
    }
}

impl EvalOp for Resize {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            inputs[0].shape(),
            scales.map(|t| &**t),
            sizes.map(|t| &**t),
        )?;
        let scales: TVec<f32> = if let Some(scales) = scales {
            scales.as_slice::<f32>()?.into()
        } else {
            output_shape
                .iter()
                .zip(inputs[0].shape())
                .map(|(o, i)| *o as f32 / *i as f32)
                .collect()
        };
        let input = inputs.remove(0);
        if contiguous::fast_enabled() {
            let mut changed = scales
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, s)| *s != 1.0);
            if let Some((axis, scale)) = changed.next() {
                if changed.next().is_none() && axis + 1 == input.rank() {
                    // A single eligible axis can borrow the input directly. In
                    // particular, downsampling the harmonic source need not copy
                    // its much larger input tensor into an intermediate ndarray.
                    if let Some(resized) = resize_last_axis(
                        &input.to_array_view::<f32>()?,
                        &output_shape,
                        axis,
                        scale,
                        &self.coord_transformer,
                        &self.interpolator,
                        self.nearest,
                    )? {
                        return Ok(tvec!(resized.into_tvalue()));
                    }
                }
            }
        }
        let mut data = input.into_tensor().into_array::<f32>()?;
        for (axis, scale) in scales.into_iter().enumerate().filter(|(_, s)| *s != 1.0) {
            let mut new_shape: TVec<usize> = data.shape().into();
            new_shape[axis] = output_shape[axis];
            if contiguous::fast_enabled() {
                if let Some(resized) = resize_last_axis(
                    &data,
                    &new_shape,
                    axis,
                    scale,
                    &self.coord_transformer,
                    &self.interpolator,
                    self.nearest,
                )? {
                    data = resized;
                    continue;
                }
            }
            data = resize_axis_generic(
                &data,
                &new_shape,
                axis,
                scale,
                &self.coord_transformer,
                &self.interpolator,
                self.nearest,
            );
        }
        Ok(tvec!(data.into_tvalue()))
    }
}

impl InferenceRulesOp for Resize {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if let Some(scales) = self.optional_scales_input {
            s.given(&inputs[scales].shape[0], move |s, len| {
                if len.is_zero() {
                    rules_with_sizes(self, s, inputs, outputs)
                } else {
                    rules_with_scales(self, s, inputs, outputs)
                }
            })
        } else if self.optional_sizes_input.is_some() {
            rules_with_sizes(self, s, inputs, outputs)
        } else {
            /*
            // bogus 4 inputs case
            s.given_2(
            &inputs[0].rank,
            &inputs[self.optional_scales_input.unwrap()].shape,
            move |s, input_rank, scale_shape| {
            if scale_shape.len() == 0 || scale_shape[0] != input_rank.to_dim() {
            rules_with_sizes(self, s, inputs, outputs)
            } else {
            rules_with_scales(self, s, inputs, outputs)
            }
            },
            )
            */
            todo!()
        }
    }

    as_op!();
    to_typed!();
}

fn rules_with_scales<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let scales = &inputs[op.optional_scales_input.unwrap()];
    s.equals(&scales.datum_type, f32::datum_type())?;
    s.equals(&scales.rank, 1)?;
    s.equals(&scales.shape[0], inputs[0].rank.bex().to_dim())?;
    s.given_2(
        &inputs[0].shape,
        &inputs[op.optional_scales_input.unwrap()].value,
        move |s, input_shape, scales| {
            let output_size = op.compute_output_shape(&input_shape, Some(scales.as_ref()), None)?;
            let rank = input_shape.len();
            for i in 0..rank {
                s.equals(&outputs[0].shape[i], output_size[i].to_dim())?;
            }
            Ok(())
        },
    )
}

fn rules_with_sizes<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let sizes = &inputs[op.optional_sizes_input.unwrap()];
    s.equals(&sizes.rank, 1)?;
    s.equals(&sizes.shape[0], inputs[0].rank.bex().to_dim())?;
    s.given(&inputs[0].rank, move |s, rank| {
        for i in 0..(rank as usize) {
            s.equals(&outputs[0].shape[i], sizes.value[i].bex().to_dim())?;
        }
        Ok(())
    })
}

impl TypedOp for Resize {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let _roi = self.optional_roi_input.and_then(|ix| inputs.get(ix));
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            &inputs[0].shape,
            scales.and_then(|f| f.konst.as_deref()),
            sizes.and_then(|f| f.konst.as_deref()),
        )?;
        Ok(tvec!(inputs[0].datum_type.fact(&output_shape)))
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}
