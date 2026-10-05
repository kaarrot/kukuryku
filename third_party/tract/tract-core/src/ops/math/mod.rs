#![allow(clippy::clone_on_copy)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::blocks_in_conditions)]

use super::array::MultiBroadcastTo;
use super::binary::TypedBinOp;
use crate::internal::*;
use crate::ops::quant::scale_by;
use num_traits::bounds::Bounded;
use num_traits::int::PrimInt;
use num_traits::{Float, Zero};
use tract_data::internal::ClampCast;
use tract_data::itertools::Itertools;
pub use tract_data::prelude::round_ties_to_even;
use tract_linalg::{ScaleShiftAndRound, Scaler};
use tract_num_traits::AsPrimitive;

#[cfg(feature = "complex")]
mod complex;
#[cfg(feature = "complex")]
pub use complex::{ComplexToInnerDim, InnerDimToComplex};

bin_to_super_type!(add, Add,
                   declutter: try_fuse_snake,
                   linalg: Add,
                   neutral_element: 0,
                   validation: Validation::Rounding,
                   q: [i8, u8, i32, i32] => add_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a+b},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim, String] => |c, a, b| *c = a.clone() + b);

fn add_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i64> + Datum,
    i64: AsPrimitive<T>,
{
    *c = (a.as_() + b.as_() - zp as i64).clamp_cast()
}

bin_to_super_type!(sub, Sub,
                   linalg:Sub,
                   is_commutative: false,
                   neutral_element: 0,
                   q: [i8, u8, i32, i32] => sub_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a-b},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);

bin_to_super_type!(subf, SubF,
                   linalg:SubF,
                   is_commutative: false,
                   neutral_element: 0,
                   q: [i8, u8, i32, i32] => subf_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {b - a},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = b.clone() - a);

fn sub_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16> + Datum,
    i16: AsPrimitive<T>,
{
    *c = (a.as_() - b.as_() + zp as i16).clamp_cast()
}

fn subf_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16> + Datum,
    i16: AsPrimitive<T>,
{
    *c = (b.as_() - a.as_() + zp as i16).clamp_cast()
}

bin_to_super_type!(mul, Mul,
                   cost: |dt| tvec!((Cost::FMA(dt), 1)),
                   declutter: declutter_mul,
                   eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                    // we apply only if type is QU8 zp_scale datum type
                    if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                        (a.datum_type(), b.datum_type(), c_dt)
                    {
                           let multiplier = a_scale  * b_scale * (1.0/ c_scale);
                           let a = a.to_array_view::<u8>()?;
                           let b = b.to_array_view::<u8>()?;
                           let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                           let mut c = Tensor::zero_dt(c_dt, &c_shape)?;
                           let view = c.to_array_view_mut::<u8>()?;
                           crate::ndarray::Zip::from(view)
                               .and_broadcast(a)
                               .and_broadcast(b)
                               .for_each(|c,a,b| *c = (scale_by((*a as i32 - a_zp as i32) * (*b as i32 - b_zp as i32), multiplier) + c_zp as i32).clamp_cast());
                           Ok(c)
                        } else {
                            Mul.generic_eval(a, b, c_dt)
                        }
                    },
                   linalg: Mul,
                   neutral_element: 1,
                   out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                       if c.datum_type() == TDim::datum_type() &&
                           a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                               let a = a.to_array_view::<TDim>()?;
                               let b = b.cast_to::<i32>()?;
                               let b = b.to_array_view::<i32>()?;
                               let c = c.to_array_view_mut::<TDim>()?;
                               crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() * *b);
                               Ok(true)
                           }
                       else {
                           match c.datum_type() {
                               DatumType::QI8(params) => {
                                   let (zp, scale) = params.zp_scale();
                                   let a = a.to_array_view::<i8>()?;
                                   let b = b.to_array_view::<i8>()?;
                                   let c = c.to_array_view_mut::<i8>()?;
                                   crate::ndarray::Zip::from(c)
                                       .and_broadcast(a)
                                       .and_broadcast(b)
                                       .for_each(|c,a,b| *c = (scale_by((*a as i16 - zp as i16) * (*b as i16 - zp as i16), scale) + zp as i16).clamp_cast());
                                   Ok(true)
                               }
                               DatumType::QU8(params) => {
                                   let (zp, scale) = params.zp_scale();
                                   let a = a.to_array_view::<u8>()?;
                                   let b = b.to_array_view::<u8>()?;
                                   let c = c.to_array_view_mut::<u8>()?;
                                   crate::ndarray::Zip::from(c)
                                       .and_broadcast(a)
                                       .and_broadcast(b)
                                       .for_each(|c,a,b| *c = (scale_by((*a as i32 - zp as i32) * (*b as i32 - zp as i32), scale) + zp as i32).clamp_cast());
                                   Ok(true)
                               }
                               _ => Ok(false)
                           }
                       }
                   },
                   q: [i8, u8, i32] => |c, a, b, zp, scale| {
                    *c = (scale_by((a.clone() as i32 - zp as i32) * (*b as i32 - zp as i32) , scale) + zp as i32).clamp_cast()
                   };
                   q_op_on_f32: |a: f32, b: f32| a * b,
[f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() * b
);

bin_to_super_type!(div, Div,
cost: |dt| tvec!((Cost::Div(dt), 1)),
declutter: declutter_div,
eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
    if
        a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
            let a = a.to_array_view::<TDim>()?;
            let b = b.to_array_view::<TDim>()?;
            let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
            unsafe {
                let a = a.broadcast(&*c_shape).unwrap();
                let b = b.broadcast(&*c_shape).unwrap();
                let mut c = Tensor::uninitialized_dt(DatumType::TDim, &c_shape)?;
                let mut view = c.to_array_view_mut::<TDim>()?;
                for coords in crate::ndarray::indices(&*c_shape) {
                    let (p, q) = a[&coords].maybe_div(&b[&coords])?;
                    view[&coords] = p/q;
                }
                Ok(c)
            }
        } else if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                       DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                       DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                (a.datum_type(), b.datum_type(), c_dt) {

               let multiplier = a_scale / (b_scale * c_scale);
                let a = a.to_array_view::<u8>()?;
                let b = b.to_array_view::<u8>()?;
                let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                let mut c = Tensor::zero_dt(c_dt, &c_shape)?;
                let view = c.to_array_view_mut::<u8>()?;
                crate::ndarray::Zip::from(view)
                    .and_broadcast(a)
                    .and_broadcast(b)
                    // maintain division in f32 before rescale to maintain high accuracy
                    .for_each(|c,a,b| *c = (
                            scale_by(
                                (*a as i32 - a_zp as i32) as f32 / (*b as i32 - b_zp as i32) as f32, multiplier
                            ) as i32 + c_zp as i32
                        ).clamp_cast());
                Ok(c)
        } else {
            Div.generic_eval(a, b, c_dt)
        }
},
is_commutative: false,
neutral_element: 1,
out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
    if c.datum_type() == TDim::datum_type() &&
        a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
            let a = a.to_array_view::<TDim>()?;
            let b = b.cast_to::<i32>()?;
            let b = b.to_array_view::<i32>()?;
            let c = c.to_array_view_mut::<TDim>()?;
            crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() / *b);
            Ok(true)
        } else if c.datum_type().is_quantized() || b.datum_type().is_quantized() || a.datum_type().is_quantized() {
            let a_f32 = a.cast_to::<f32>()?;
            let a_f32 = a_f32.to_array_view::<f32>()?;
            let b_f32 = b.cast_to::<f32>()?;
            let b_f32 = b_f32.to_array_view::<f32>()?;
            let c_f32 = &a_f32 / &b_f32;
            *c = c_f32.into_tensor().cast_to_dt(c.datum_type())?.into_owned();
            Ok(true)
        } else {
            Ok(false)
        }
},
q_op_on_f32: |a: f32, b: f32| a / b,
[f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() / b
);

bin_to_super_type!(rem, Rem,
                                      eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                                          if
                                              a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                                                  let a = a.to_array_view::<TDim>()?;
                                                  let b = b.cast_to::<i32>()?;
                                                  let b = b.to_array_view::<i32>()?;
                                                  let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                                                  unsafe {
                                                      let mut c = Tensor::uninitialized_dt(DatumType::TDim, &c_shape)?;
                                                      let view = c.to_array_view_mut::<TDim>()?;
                                                      crate::ndarray::Zip::from(view).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() % *b);
                                                      Ok(c)
                                                  }
                                              } else {
                                                  Rem.generic_eval(a,b, c_dt)
                                              }
                                      },
                                      out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                                          if c.datum_type() == TDim::datum_type() &&
                                              a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                                                  let a = a.to_array_view::<TDim>()?;
                                                  let b = b.cast_to::<i32>()?;
                                                  let b = b.to_array_view::<i32>()?;
                                                  let c = c.to_array_view_mut::<TDim>()?;
                                                  crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() % *b);
                                                  Ok(true)
                                              } else {
                                                  Ok(false)
                                              }
                                      },
                                      [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() % b);

bin_to_super_type!(min, Min, linalg:Min,
                   q: [i8, u8, i32] => |c, a, b, _, _| *c = if a < b { *a } else { *b };
                   q_op_on_f32: |a: f32, b: f32| a.min(b),
                   [f16, f32, f64] => |c,a,b| *c = a.min(*b),
                   [TDim] => |c,a,b| *c = a.clone().mini(b.clone()),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.min(b));

bin_to_super_type!(max, Max,
                   eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                   // Attempt to optimize relu case
                    if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                        (a.datum_type(), b.datum_type(), c_dt)
                    {
                        if a.is_uniform() || b.is_uniform() {
                            // select e between a and b as uniform if exist
                            // and d remaining a or b
                            let (d, d_zp, d_scale, e, e_zp, e_scale) = if a.is_uniform() && !b.is_uniform() {
                                (&b, &b_zp, &b_scale, &a, &a_zp, &a_scale)
                            } else {
                                (&a, &a_zp, &a_scale, &b, &b_zp, &b_scale)
                            };
                            if e.is_uniform() { // may be relu or any scalar
                                let e = e.cast_to::<u8>()?.as_slice::<u8>()?[0];
                                let e_val_as_d_aligned: i32 = scale_by(e as i32 - e_zp, e_scale / d_scale);
                                let multiplier = d_scale  * (1.0/ c_scale);
                                let d = d.to_array_view::<u8>()?;
                                let mut c = Tensor::zero_dt(c_dt, d.shape())?;
                                let view = c.to_array_view_mut::<u8>()?;
                                crate::ndarray::Zip::from(view)
                                    .and_broadcast(d)
                                    .for_each(|c,d| {
                                        let d_min_zp = *d as i32 - *d_zp as i32;
                                        let c_val: i32 = if d_min_zp < e_val_as_d_aligned {
                                            e_val_as_d_aligned
                                        } else {
                                            d_min_zp
                                        };
                                        *c = (scale_by(c_val, multiplier) + c_zp as i32).clamp_cast();
                                    });
                                return Ok(c)
                            }
                        }
                    }
                    Max.generic_eval(a, b, c_dt)
                   },
                   linalg:Max,
                   q: [i8, u8, i32] => |c, a, b, _, _| *c = if a < b { *b } else { *a };
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a.max(b)},
                   [f16, f32, f64] => |c,a,b| *c = a.max(*b),
                   [TDim] => |c,a,b| *c = a.clone().maxi(b.clone()),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.max(b));

bin_to_super_type!(pow, Pow,
                   declutter: declutter_pow,
                   is_commutative: false,
                   neutral_element: 1,
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a.powf(b)},
                   [f16, f32, f64] => |c,a,b| *c = a.powf(*b),
                   [i32, i64] => |c,a,b| *c = a.pow(*b as u32));

bin_to_super_type!(shift_left, ShiftLeft,
                   is_commutative: false,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a << *b);
bin_to_super_type!(shift_right, ShiftRight,
                   is_commutative: false,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a >> *b);

fn declutter_mul(
    _op: &Mul,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs[0] == node.inputs[1] && !node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[0..1],
            square(),
        )?));
    }

    if let Some(uniform) = crate::ops::binary::one_input_is_uniform(model, node)? {
        let var_fact = model.outlet_fact(uniform.var)?;
        if uniform.uni.cast_to_scalar::<f64>()? == 0.0 {
            let shapes =
                model.node_input_facts(node.id)?.iter().map(|f| &f.shape).collect::<TVec<_>>();
            let shape: ShapeFact =
                crate::broadcast::multi_broadcast(&shapes).context("Failed to broadcast")?.into();
            return Ok(Some(TypedModelPatch::rewire(
                model,
                &[],
                &[node.id.into()],
                &|patch, _| {
                    let scalar = patch.add_const(
                        format!("{}.zero", node.name),
                        if uniform.uni.datum_type().is_quantized() {
                            let output_dt = node.outputs[0].fact.datum_type;
                            Arc::new(uniform.uni.clone().cast_to_dt(output_dt)?.into_owned())
                        } else {
                            uniform.uni.clone()
                        },
                    )?;
                    let op = MultiBroadcastTo::new(shape.clone());
                    patch.wire_node(&node.name, op, &[scalar])
                },
            )?));
        }
        let dt = uniform.uni.datum_type();
        if !dt.is_quantized() {
            // avoid cast potential with Q tensor
            let integer = uniform.uni.cast_to_scalar::<i64>()?;
            if tensor0(integer)
                .cast_to_dt(uniform.uni.datum_type())?
                .close_enough(&uniform.uni, false)
                .is_ok()
                && uniform.uni.cast_to_scalar::<i64>()?.count_ones() == 1
                && dt.is_integer()
            {
                let shift = integer.trailing_zeros();
                return Ok(Some(TypedModelPatch::rewire(
                    model,
                    &[uniform.var],
                    &[node.id.into()],
                    &|patch, taps| {
                        let shift = patch.add_const(
                            format!("{}.shift", node.name),
                            tensor0(shift)
                                .cast_to_dt(dt)?
                                .into_owned()
                                .broadcast_into_rank(var_fact.rank())?,
                        )?;
                        patch.wire_node(&node.name, shift_left(), &[taps[0], shift])
                    },
                )?));
            }
        }
    }
    if let Some(patch) = declutter_mul_const_mul_const(model, node)? {
        return Ok(Some(patch));
    }
    Ok(None)
}

fn declutter_mul_const_mul_const(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let Some(const_slot) = input_facts.iter().position(|f| f.konst.is_some()) else {
        return Ok(None);
    };
    let prec = model.node(node.inputs[1 - const_slot].node);
    let Some(prec_mul) = prec.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    if prec.outputs[0].successors.len() > 1 {
        return Ok(None);
    };
    if !prec_mul.0.is::<Mul>() {
        return Ok(None);
    }
    let prec_input_facts = model.node_input_facts(prec.id)?;
    let Some(prec_const_slot) = prec_input_facts.iter().position(|f| f.konst.is_some()) else {
        return Ok(None);
    };

    let const_fact = model.outlet_fact(node.inputs[const_slot])?;
    let prec_const_fact = model.outlet_fact(prec.inputs[prec_const_slot])?;
    // todo: extend to anything broadcast compatible
    if !const_fact.shape.volume().is_one() && !prec_const_fact.shape.volume().is_one() {
        return Ok(None);
    }
    if !const_fact.datum_type.is_float() {
        return Ok(None);
    }
    let result = mul()
        .eval(tvec!(
            const_fact.konst.clone().unwrap().into_tvalue(),
            prec_const_fact.konst.clone().unwrap().into_tvalue()
        ))?
        .remove(0)
        .into_arc_tensor();
    let mut patch = TypedModelPatch::default();
    let konst = patch.add_const(&prec.name, result)?;
    let input_tap = patch.tap_model(model, prec.inputs[1 - prec_const_slot])?;
    let wire = patch.wire_node(&node.name, mul(), &[konst, input_tap])?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn declutter_div(
    _op: &Div,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if let &[p, q] = &*model.node_input_facts(node.id)? {
        let dt = q.datum_type;
        if let Some(q) = &q.uniform {
            if let Ok(integer) = q.cast_to_scalar::<i64>() {
                if tensor0(integer).cast_to_dt(dt)?.close_enough(q, false).is_ok()
                    && dt.is_integer()
                    && q.cast_to_scalar::<i64>()?.count_ones() == 1
                {
                    let shift = integer.trailing_zeros();
                    return Ok(Some(TypedModelPatch::rewire(
                        model,
                        &[node.inputs[0]],
                        &[node.id.into()],
                        &|patch, taps| {
                            let shift = patch.add_const(
                                format!("{}.shift", node.name),
                                tensor0(shift)
                                    .cast_to_dt(dt)?
                                    .into_owned()
                                    .broadcast_into_rank(p.rank())?,
                            )?;
                            patch.wire_node(&node.name, shift_right(), &[taps[0], shift])
                        },
                    )?));
                }
            }
        }
        if dt.is_float() {
            return Ok(Some(TypedModelPatch::rewire(
                model,
                &node.inputs,
                &[node.id.into()],
                &|patch, taps| {
                    let q =
                        patch.wire_node(format!("{}-recip", node.name), recip(), &[taps[1]])?[0];
                    patch.wire_node(&node.name, mul(), &[taps[0], q])
                },
            )?));
        }
    }
    Ok(None)
}

fn declutter_pow(
    _op: &Pow,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let b = model.outlet_fact(node.inputs[1])?;
    if let Some(b) = &b.uniform {
        let b = b.cast_to_scalar::<f32>()?;
        if b == 2.0 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                square(),
            )?));
        } else if b == 0.5 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                sqrt(),
            )?));
        }
    }
    Ok(None)
}

element_wise!(abs, Abs, [i8, i16, i32, i64, f16, f32, i32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.abs());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::abs;
operating_datum_type: |dt| if dt == TDim::datum_type() { i64::datum_type() } else { dt }
);

element_wise!(exp, Exp, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.exp());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::exp;
validation: Validation::Rounding
);

element_wise!(ln, Ln, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ln());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::ln;
validation: Validation::Rounding
);

element_wise!(square, Square, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.powi(2));
    Ok(())
};
q: [i8, u8, i32, i32] => |f : f32| f.powi(2);
declutter: declutter_square;
validation: Validation::Rounding
);

// Fuse `Square(Sin(x))` into a single SinSq pass. `linear_prec` guarantees the Sin
// feeds only this Square, so rewiring is safe. Both must be plain (no datum-type cast)
// so the fused op sees the same in/out type. Mirrors declutter_recip's shape.
fn declutter_square(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    use super::element_wise::*;
    let Some(sq) = node.op_as::<ElementWiseOp>() else { return Ok(None) };
    if sq.1.is_some() {
        return Ok(None); // Square carries an output cast; don't fold across it.
    }
    if let Some(prec) = model.linear_prec(node.id)? {
        if let Some(ew) = prec.op_as::<ElementWiseOp>() {
            if ew.0.is::<Sin>() && ew.1.is_none() {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, prec.inputs[0])?;
                wire = patch.wire_node(&node.name, sin_sq(), &[wire])?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
    }
    Ok(None)
}

element_wise!(sqrt, Sqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::sqrt;
validation: Validation::Rounding
);

element_wise!(recip, Recip,
    // f32 path: clamp exact-zero inputs to the smallest normal magnitude (sign
    // preserved) before reciprocating, so 1/0 becomes a huge *finite* value
    // instead of Inf. Needed on aarch64: Kokoro's iSTFTNet vocoder feeds this op
    // an STFT complex slice whose imaginary component hits exact zeros where the
    // same graph on x86 held tiny non-zeros; the resulting Infs turn the whole
    // stage-2 output into NaN via Inf-Inf / 0*Inf downstream. Huge-but-finite
    // sentinels flow through the vocoder's masking correctly. See
    // docs/debug-silent-tts.md for the full trace and root-cause discussion.
    [f32] => |_, xs| {
        xs.iter_mut().for_each(|x: &mut f32| {
            let denom = if *x == 0.0 {
                if x.is_sign_negative() { -f32::MIN_POSITIVE } else { f32::MIN_POSITIVE }
            } else {
                *x
            };
            *x = 1.0 / denom;
        });
        Ok(())
    },
    [f16, f64] => |_, xs| {
        xs.iter_mut().for_each(|x| *x = x.recip());
        Ok(())
    }
;
q: [i8, u8, i32, i32] => f32::recip;
cost: |dt| {tvec!((Cost::Div(dt), 1))};
declutter: declutter_recip;
validation: Validation::Rounding
);

fn declutter_recip(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    use super::element_wise::*;
    if let Some(prec) = model.linear_prec(node.id)? {
        if let Some(ew) = prec.op_as::<ElementWiseOp>() {
            let repl = if ew.0.is::<Sqrt>() {
                Some(rsqrt())
            } else if ew.0.is::<Rsqrt>() {
                Some(sqrt())
            } else {
                None
            };
            if let Some(repl) = repl {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, prec.inputs[0])?;
                wire = patch.wire_node(&node.name, repl, &[wire])?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
    }
    Ok(None)
}

element_wise!(rsqrt, Rsqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt().recip());
    Ok(())
};
q: [i8, u8, i32] => |x : f32| x.sqrt().recip();
validation: Validation::Rounding
);

element_wise!(ceil, Ceil, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ceil());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::recip);

element_wise!(floor, Floor, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.floor());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::floor);

element_wise!(round, Round, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.round());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::round);

element_wise!(q_scale, QScale{scaler: Scaler},[i32] => |op, xs| {
    xs.iter_mut().for_each(|x| *x = x.q_scale(op.scaler));
    Ok(())
});

element_wise!(round_half_to_even, RoundHalfToEven,
[f32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = round_ties_to_even(*x));
    Ok(())
},
[f16] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = f16::from_f32(round_ties_to_even(x.to_f32())));
    Ok(())
};
q: [i8, u8, i32] => round_ties_to_even);

/// Apply a scalar op elementwise, fanning large buffers across the tract thread
/// pool. Transcendental ops (sin/cos) are heavy enough that this pays off; the
/// threshold keeps the common small-tensor case on the sequential path (no rayon
/// overhead). Chunked so each element is touched once -> result is independent of
/// thread count.
fn par_elementwise<T: Datum + Send>(xs: &mut [T], f: impl Fn(&mut T) + Send + Sync) {
    const PAR_THRESHOLD: usize = 1 << 14;
    if xs.len() < PAR_THRESHOLD {
        xs.iter_mut().for_each(f);
    } else {
        let chunk = (xs.len() / 64).max(4096);
        tract_linalg::multithread::par_chunks_mut(xs, chunk, |_, c| c.iter_mut().for_each(&f));
    }
}

element_wise!(cos, Cos, [f16, f32, f64] => |_, xs| {
    par_elementwise(xs, |x| *x = x.cos());
    Ok(())
};
q: [i8, u8, i32] => f32::cos);

// Faithful branchless f32 sine (Julien Pommier / cephes minimax, ~1e-7 abs error,
// ~1 ulp — NOT fast-math). Fully branchless so `par_elementwise`'s `iter_mut` body
// auto-vectorizes to SIMD instead of calling scalar libm `sinf` per element. Reduces
// by pi/4 octants with a 3-part Cody-Waite pi/4, accurate over Kokoro's normal
// oscillator phase range (per-node profile confirms the sinf fast path, no huge-arg
// reduction). Drives the 48 Snake/oscillator Sin passes (Tier 7 Lever 3b).
#[inline(always)]
fn ssin_f32(xin: f32) -> f32 {
    const FOPI: f32 = 1.27323954473516; // 4/pi
    const DP1: f32 = -0.78515625; // pi/4 in 3 parts
    const DP2: f32 = -2.4187564849853515625e-4;
    const DP3: f32 = -3.77489497744594108e-8;
    const SINCOF_P0: f32 = -1.9515295891e-4;
    const SINCOF_P1: f32 = 8.3321608736e-3;
    const SINCOF_P2: f32 = -1.6666654611e-1;
    const COSCOF_P0: f32 = 2.443315711809948e-5;
    const COSCOF_P1: f32 = -1.388731625493765e-3;
    const COSCOF_P2: f32 = 4.166664568298827e-2;

    let sign_bit = xin.to_bits() & 0x8000_0000; // sin is odd; carry input sign
    let x = xin.abs();
    let mut j = (x * FOPI) as i32; // octant (floor via truncation, x >= 0)
    j = (j + 1) & !1; // round up to even
    let y = j as f32;
    let swap = ((j & 4) as u32) << 29; // sign flip for octants 4..7 (0 or 0x8000_0000)
    let use_cos = (j & 2) != 0; // octants 2,3,6,7 evaluate the cos polynomial
    let sign = sign_bit ^ swap;
    let r = x + y * DP1 + y * DP2 + y * DP3; // extended-precision reduced angle
    let z = r * r;
    let cos = {
        let p = COSCOF_P0;
        let p = p * z + COSCOF_P1;
        let p = p * z + COSCOF_P2;
        p * z * z - 0.5 * z + 1.0
    };
    let sin = {
        let p = SINCOF_P0;
        let p = p * z + SINCOF_P1;
        let p = p * z + SINCOF_P2;
        p * z * r + r
    };
    // Branchless select (LLVM lowers to vblendvps) + sign via XOR.
    let mag = if use_cos { cos } else { sin };
    f32::from_bits(mag.to_bits() ^ sign)
}

element_wise!(sin, Sin,
    [f32] => |_, xs| { par_elementwise(xs, |x| *x = ssin_f32(*x)); Ok(()) },
    [f16, f64] => |_, xs| { par_elementwise(xs, |x| *x = x.sin()); Ok(()) }
;
q: [i8, u8, i32] => f32::sin);

// Fused sin-then-square: `Square(Sin(x))`, decluttered from that pair (see
// declutter_square). One memory pass over the large [1,C,F] tensors instead of two.
// The f32 path uses the vectorized ssin_f32 (Lever 3b); f16/f64 keep exact `.sin()`.
// Kokoro's 48 Snake activations (`x + (1/a)*sin(a*x)^2`) drive this.
element_wise!(sin_sq, SinSq,
    [f32] => |_, xs| { par_elementwise(xs, |x| { let s = ssin_f32(*x); *x = s * s; }); Ok(()) },
    [f16, f64] => |_, xs| { par_elementwise(xs, |x| *x = x.sin().powi(2)); Ok(()) }
);

/// `x + (1/α) · sin(α · x)²` in one pass over `[1, C, F]`.
///
/// α is per channel (the Snake const of shape `[1, C, 1]`). Fusing deletes the
/// four memory passes `Mul → SinSq → Mul → Add` that otherwise stream the
/// vocoder activation. f16 is intentionally unsupported: f16 `Sin` is scalar
/// libm, and the fp16 experiment keeps Snake in f32.
#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    inv_alpha: Vec<f32>,
}

impl ElementWiseMiniOp for Snake {
    fn name(&self) -> String {
        "Snake".to_string()
    }

    fn same_as(&self, other: &dyn ElementWiseMiniOp) -> bool {
        other.downcast_ref::<Snake>().is_some_and(|o| o.alpha == self.alpha && o.inv_alpha == self.inv_alpha)
    }

    fn eval_in_place(&self, t: &mut Tensor, out_dt: Option<DatumType>) -> TractResult<()> {
        if t.datum_type() != f32::datum_type() || out_dt.is_some_and(|d| d != f32::datum_type()) {
            bail!("Snake only runs on f32");
        }
        let shape = t.shape().to_vec();
        let xs = t.as_slice_mut::<f32>()?;
        let alpha = &self.alpha;
        let inv = &self.inv_alpha;
        if alpha.len() == 1 && inv.len() == 1 {
            let (a, inv_a) = (alpha[0], inv[0]);
            par_elementwise(xs, |x| {
                let s = ssin_f32(a * *x);
                *x += inv_a * s * s;
            });
            return Ok(());
        }
        if shape.len() == 3 && shape[1] == alpha.len() && alpha.len() == inv.len() && shape[2] > 0 {
            let (n, c, f) = (shape[0], shape[1], shape[2]);
            for ni in 0..n {
                let start = ni * c * f;
                let block = &mut xs[start..start + c * f];
                tract_linalg::multithread::par_chunks_mut(block, f, |ci, row| {
                    if ci >= c || row.len() != f {
                        return;
                    }
                    let (a, inv_a) = (alpha[ci], inv[ci]);
                    for x in row.iter_mut() {
                        let s = ssin_f32(a * *x);
                        *x += inv_a * s * s;
                    }
                });
            }
            return Ok(());
        }
        bail!("Snake shape {:?} does not match {} channels", shape, alpha.len());
    }
}

/// Fold `Add(x, Mul(SinSq(Mul(x, α)), 1/α))` when α and 1/α are finite consts
/// and each intermediate has a single consumer. Runs from `Add`'s declutter,
/// after `Square(Sin)` has already become `SinSq` (the pass repeats).
fn try_fuse_snake(
    _op: &Add,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 2 {
        return Ok(None);
    }
    let dt = node.outputs.first().map(|o| o.fact.datum_type);
    if dt != Some(f32::datum_type()) {
        return Ok(None);
    }
    let fused = match_snake(model, node.inputs[0], node.inputs[1])
        .or_else(|| match_snake(model, node.inputs[1], node.inputs[0]));
    let Some((x, alpha, inv_alpha)) = fused else {
        return Ok(None);
    };
    let mut patch = TypedModelPatch::default();
    let tap = patch.tap_model(model, x)?;
    let op = crate::ops::element_wise::ElementWiseOp(Box::new(Snake { alpha, inv_alpha }), None);
    let wire = patch.wire_node(&node.name, op, &[tap])?[0];
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

fn match_snake(
    model: &TypedModel,
    tail: OutletId,
    x_expected: OutletId,
) -> Option<(OutletId, Vec<f32>, Vec<f32>)> {
    let outer = model.node(tail.node);
    if !sole_consumer(model, outer.id) {
        return None;
    }
    let (inv_t, sq_out) = mul_const_and_var(model, outer)?;
    let sq = model.node(sq_out.node);
    if sq_out.slot != 0 || !sole_consumer(model, sq.id) {
        return None;
    }
    let ew = sq.op_as::<crate::ops::element_wise::ElementWiseOp>()?;
    ew.0.downcast_ref::<SinSq>()?;
    let inner_out = *sq.inputs.first()?;
    let inner = model.node(inner_out.node);
    if inner_out.slot != 0 || !sole_consumer(model, inner.id) {
        return None;
    }
    let (alpha_t, x) = mul_const_and_var(model, inner)?;
    if x != x_expected {
        return None;
    }
    let alpha = const_f32s(&alpha_t)?;
    let inv = const_f32s(&inv_t)?;
    if alpha.len() != inv.len() || alpha.is_empty() {
        return None;
    }
    let reciprocal = alpha.iter().zip(&inv).all(|(a, i)| {
        a.is_finite() && i.is_finite() && *a != 0.0 && (*a * *i - 1.0).abs() <= 1e-2
    });
    if !reciprocal {
        return None;
    }
    Some((x, alpha, inv))
}

fn sole_consumer(model: &TypedModel, id: usize) -> bool {
    let n = model.node(id).outputs.iter().map(|o| o.successors.len()).sum::<usize>();
    n == 1
}

fn mul_const_and_var(model: &TypedModel, node: &TypedNode) -> Option<(Arc<Tensor>, OutletId)> {
    let bin = node.op_as::<TypedBinOp>()?;
    if !bin.0.is::<Mul>() {
        return None;
    }
    let c0 = model.outlet_fact(node.inputs[0]).ok()?.konst.clone();
    let c1 = model.outlet_fact(node.inputs[1]).ok()?.konst.clone();
    match (c0, c1) {
        (Some(c), None) => Some((c, node.inputs[1])),
        (None, Some(c)) => Some((c, node.inputs[0])),
        _ => None,
    }
}

fn const_f32s(t: &Tensor) -> Option<Vec<f32>> {
    let cast = t.cast_to::<f32>().ok()?;
    let s = cast.as_slice::<f32>().ok()?;
    Some(s.to_vec())
}

element_wise!(tan, Tan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.tan());
    Ok(())
};
q: [i8, u8, i32] => f32::tan);

element_wise!(acos, Acos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acos());
    Ok(())
};
q: [i8, u8, i32] => f32::acos);

element_wise!(asin, Asin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asin());
    Ok(())
};
q: [i8, u8, i32] => f32::asin);

element_wise!(atan, Atan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atan());
    Ok(())
};
q: [i8, u8, i32] => f32::atan);

element_wise!(cosh, Cosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cosh());
    Ok(())
};
q: [i8, u8, i32] => f32::cosh);

element_wise!(sinh, Sinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sinh());
    Ok(())
};
q: [i8, u8, i32] => f32::sinh);

element_wise!(tanh, Tanh,
 [f16] => |_, xs| { (tract_linalg::ops().tanh_f16)().run(xs) },
 [f32] => |_, xs| { (tract_linalg::ops().tanh_f32)().run(xs) },
 [f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.tanh()); Ok(()) };
 q: [i8, u8, i32] => f32::tanh;
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(erf, Erf,
 [f32] => |_, xs| { (tract_linalg::ops().erf_f32)().run(xs) },
 [f16] => |_, xs| {
     let mut f32s = xs.iter().map(|x| x.to_f32()).collect_vec();
     (tract_linalg::ops().erf_f32)().run(&mut f32s)?;
     xs.iter_mut().zip(f32s.into_iter()).for_each(|(x, f)| *x = f16::from_f32(f));
     Ok(())
};
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(acosh, Acosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acosh());
    Ok(())
};
q: [i8, u8, i32] => f32::acosh);
element_wise!(asinh, Asinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asinh());
    Ok(())
};
q: [i8, u8, i32] => f32::asinh);
element_wise!(atanh, Atanh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atanh());
    Ok(())
};
q: [i8, u8, i32] => f32::atanh);

element_wise!(neg, Neg, [i8, i16, i32, i64, f16, f32, f64, TDim] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = -x.clone());
    Ok(())
};
q: [i8, u8, i32] => |x: f32| -x);

element_wise!(sign, Sign, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = if x.is_zero() { *x } else { x.signum() });
    Ok(())
};
q: [i8, u8, i32] => f32::signum);

#[cfg(test)]
mod tests {
    use crate::ops::binary::TypedBinOp;

    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mul() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a * b, arr2(&[[1., 0.], [0., 0.]]));
    }

    #[test]
    fn dot() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a.dot(&b), arr2(&[[1., 0.], [3., 0.]]));
    }

    #[test]
    fn mul_as_shift_left() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", i32::fact([2usize, 2]))?;
        let a = model.add_const("a", tensor0(4i32).broadcast_into_rank(2)?.into_arc_tensor())?;
        let y = model.wire_node("y", mul(), &[x, a])?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[1, 2], [3, 4]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [12, 16]]));
        let decluttered = model.into_decluttered()?;
        let result =
            SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[1, 2], [3, 4]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [12, 16]]));
        let op = decluttered
            .node(decluttered.output_outlets()?[0].node)
            .op()
            .downcast_ref::<TypedBinOp>()
            .unwrap();
        assert!(op.0.downcast_ref::<ShiftLeft>().is_some());
        Ok(())
    }

    #[test]
    fn div_as_shift() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("a", i32::fact([2usize, 2]))?;
        let s = model.add_const("shift", tensor2(&[[4]]))?;
        let y = model.wire_node("c", div(), [x, s].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[16, 32], [64, 68]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [16, 17]]));
        let decluttered = model.into_decluttered()?;
        let result =
            SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[16, 32], [64, 68]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [16, 17]]));
        let op = decluttered
            .node(decluttered.output_outlets()?[0].node)
            .op()
            .downcast_ref::<TypedBinOp>()
            .unwrap();
        assert!(op.0.downcast_ref::<ShiftRight>().is_some());
        Ok(())
    }

    /// `Add(x, Mul(SinSq(Mul(x, α)), 1/α))` becomes one Snake pass, and the
    /// values match the f32 formula (not a looser fast-math sin).
    #[test]
    fn snake_fuses_and_matches_ssin() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 2, 4]))?;
        let alpha = model.add_const("alpha", Tensor::from_shape(&[1, 2, 1], &[0.5f32, 2.0])?)?;
        let inv = model.add_const("inv", Tensor::from_shape(&[1, 2, 1], &[2.0f32, 0.5])?)?;
        let scaled = model.wire_node("scaled", mul(), &[x, alpha])?[0];
        let sq = model.wire_node("sq", sin_sq(), &[scaled])?[0];
        let scaled_sq = model.wire_node("scaled_sq", mul(), &[sq, inv])?[0];
        let y = model.wire_node("y", add(), &[x, scaled_sq])?[0];
        model.set_output_outlets(&[y])?;

        let decluttered = model.clone().into_decluttered()?;
        let names: Vec<String> = decluttered.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(names.iter().any(|n| n.contains("Snake")), "not fused: {names:?}");

        let input = Tensor::from_shape(
            &[1, 2, 4],
            &[0.1f32, 0.2, -0.3, 0.4, 1.0, -1.0, 0.5, 0.25],
        )?;
        let out = SimplePlan::new(&decluttered)?.run(tvec!(input.into()))?;
        let got = out[0].as_slice::<f32>()?;
        let xs = [0.1f32, 0.2, -0.3, 0.4, 1.0, -1.0, 0.5, 0.25];
        let alphas = [0.5f32, 2.0];
        let invs = [2.0f32, 0.5];
        for c in 0..2 {
            for f in 0..4 {
                let x = xs[c * 4 + f];
                let s = super::ssin_f32(alphas[c] * x);
                let expect = x + invs[c] * s * s;
                let g = got[c * 4 + f];
                assert!((g - expect).abs() < 1e-5, "c={c} f={f} got {g} expect {expect}");
            }
        }
        Ok(())
    }

    /// A mul that is not actually 1/α must stay a plain add. Snake is not a
    /// general mul-sin-add fold.
    #[test]
    fn snake_skips_non_reciprocal_scale() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, 2, 4]))?;
        let alpha = model.add_const("alpha", Tensor::from_shape(&[1, 2, 1], &[0.5f32, 2.0])?)?;
        let inv = model.add_const("inv", Tensor::from_shape(&[1, 2, 1], &[0.5f32, 0.5])?)?;
        let scaled = model.wire_node("scaled", mul(), &[x, alpha])?[0];
        let sq = model.wire_node("sq", sin_sq(), &[scaled])?[0];
        let scaled_sq = model.wire_node("scaled_sq", mul(), &[sq, inv])?[0];
        let y = model.wire_node("y", add(), &[x, scaled_sq])?[0];
        model.set_output_outlets(&[y])?;
        let decluttered = model.into_decluttered()?;
        let names: Vec<String> = decluttered.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(!names.iter().any(|n| n.contains("Snake")), "fused unexpectedly: {names:?}");
        Ok(())
    }
}
