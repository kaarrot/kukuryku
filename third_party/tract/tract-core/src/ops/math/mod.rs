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
                   declutter: try_fuse_add,
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
        } else if b == 3.0
            // Dropping the exponent input must not change the output shape.
            && model.outlet_fact(node.inputs[0])?.shape == node.outputs[0].fact.shape
        {
            // x * Square(x): gelu_new's x³ (ALBERT, every stage-1 layer) was a
            // generic powf per element, ~8% of stage 1.
            return Ok(Some(TypedModelPatch::rewire(
                model,
                &[node.inputs[0]],
                &[node.id.into()],
                &|patch, taps| {
                    let sq =
                        patch.wire_node(format!("{}.square", node.name), square(), &[taps[0]])?[0];
                    patch.wire_node(&node.name, mul(), &[taps[0], sq])
                },
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

// `sin(r)²` for the Snake path, ~2.8e-7 abs error (ssin_f32 + a square is ~1.1e-7;
// f16 noise downstream is ~1e-3). sin² has period pi, so one reduction to
// w in [-pi/2, pi/2] and a degree-8 polynomial in w² replace ssin_f32's octant
// select and two polynomials. The Taylor series of sin²w is evaluated with Estrin's
// scheme: Horner's 8-deep FMA chain is latency-bound once vectorized, Estrin's is 4.
// 3.1 ns/element against 5.2 for ssin_f32 on an A76, single thread.
//
// `n = round(r/pi)` uses the 1.5*2^23 trick (round-half-even, valid for |r/pi| < 2^22)
// so it vectorizes without libm `roundf` or SSE4.1. pi is split in two so `n * PI_HI`
// is exact for |n| < 4096, so the error stays under 5e-7 for |r| < 12000. Past ~1e5 an
// f32 phase has an ulp above 0.008 rad, so the input carries no phase to preserve;
// the result is then garbage (as ssin_f32's is, its i32 octant overflows), and NaN
// stays NaN. Snake's inputs are alpha * activation, far inside the range. A clamp
// to keep the result finite cost 1.1 of the 4.2 ns/element, so there is none.
#[inline(always)]
fn sin_sq_f32(r: f32) -> f32 {
    const INV_PI: f32 = 0.318_309_9;
    const PI_HI: f32 = 3.140_625; // 12 significant bits
    const PI_LO: f32 = 9.676_536e-4; // pi - PI_HI
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23
    // Taylor coefficients of sin²w in w²: (-1)^(k+1) 2^(2k-1) / (2k)!, k = 1..8. The
    // dropped k = 9 term is 7e-8 at w = pi/2.
    const C1: f32 = 1.0;
    const C2: f32 = -3.333_333_3e-1;
    const C3: f32 = 4.444_444_4e-2;
    const C4: f32 = -3.174_603_2e-3;
    const C5: f32 = 1.410_934_7e-4;
    const C6: f32 = -4.275_559_8e-6;
    const C7: f32 = 9.396_834_8e-8;
    const C8: f32 = -1.566_139_1e-9;

    let n = (r * INV_PI + ROUND) - ROUND;
    let w = (r - n * PI_HI) - n * PI_LO;
    let y = w * w;
    let y2 = y * y;
    let y4 = y2 * y2;
    let lo = (C1 + C2 * y) + y2 * (C3 + C4 * y);
    let hi = (C5 + C6 * y) + y2 * (C7 + C8 * y);
    (lo + y4 * hi) * y
}

element_wise!(sin, Sin,
    [f32] => |_, xs| { par_elementwise(xs, |x| *x = ssin_f32(*x)); Ok(()) },
    [f16, f64] => |_, xs| { par_elementwise(xs, |x| *x = x.sin()); Ok(()) }
;
q: [i8, u8, i32] => f32::sin);

// Fused sin-then-square: `Square(Sin(x))`, decluttered from that pair (see
// declutter_square). One memory pass over the large [1,C,F] tensors instead of two.
// The f32 path uses the vectorized sin_sq_f32; f16/f64 keep exact `.sin()`.
// Kokoro's 48 Snake activations (`x + (1/a)*sin(a*x)^2`) drive this.
element_wise!(sin_sq, SinSq,
    [f32] => |_, xs| { par_elementwise(xs, |x| *x = sin_sq_f32(*x)); Ok(()) },
    [f16, f64] => |_, xs| { par_elementwise(xs, |x| *x = x.sin().powi(2)); Ok(()) }
);

/// One Snake activation, `x + (1/a) * sin(a * x)^2`. Shared by [`Snake`] and the fused
/// [`AdaIn`] apply loop so the two cannot drift apart.
#[inline(always)]
fn snake_f32(x: f32, a: f32, inv_a: f32) -> f32 {
    x + inv_a * sin_sq_f32(a * x)
}

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
            par_elementwise(xs, |x| *x = snake_f32(*x, a, inv_a));
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
                        *x = snake_f32(*x, a, inv_a);
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

/// Instance norm with a per-channel affine, `((x - mean) * rsqrt(var + eps)) * g + beta`,
/// over each `(n, c)` row of an `[N, C, F]` tensor, in one op. Bit-identical to the
/// unfused graph (see [`adain_row`]).
///
/// Kokoro's vocoder has 62 of these (AdaIN: `g` is `1 + gamma`, `beta` comes from the
/// style vector). Tract lowers each to `Reduce<Sum>`, `Mul`, `Sub`,
/// `Reduce<SumOfSquares>`, `Mul`, `Add`, `Rsqrt`, `Mul`, `Mul`, `Add`: six full passes
/// over the activation, memory-bound at 11-15 GB/s. Here a row (274 KiB for the longest
/// benchmark sentence) is read for the mean, again for the variance, and written once while
/// it is still in L2.
///
/// The `1/F` factors and eps arrive as inputs and are multiplied exactly where the
/// unfused graph multiplies them, so the fusion does not have to prove anything about
/// F (under the symbolic plan they are runtime `Recip` nodes).
///
/// Every one of the 62 feeds a single Snake or LeakyRelu, and under the FP16 translation
/// most sit between a `Cast(f16 -> f32)` and a `Cast(f32 -> f16)` (41 of them between
/// both). Declutter folds those neighbours in ([`AdaIn::declutter`]): the activation runs
/// in the apply loop, and the casts become a widen into / a narrow out of a row buffer, so
/// the f32 tensors the casts and the activation used to read and write never exist. Only
/// the storage changes. The values entering and leaving each f32 step are the ones the
/// separate nodes produced, and `half`'s slice conversions are the ones `Cast` uses.
///
/// Inputs: x `[N, C, F]` (f16 when `in_f16`), g `[1, C, 1]`, beta `[1, C, 1]`, d_mean,
/// d_var, eps (one element each, f32). Output f16 when `out_f16`.
#[derive(Debug, Clone, PartialEq)]
struct AdaIn {
    act: AdaInAct,
    in_f16: bool,
    out_f16: bool,
}

/// The activation the AdaIN node absorbed. Same arithmetic as the standalone ops.
#[derive(Debug, Clone, PartialEq)]
enum AdaInAct {
    Identity,
    LeakyRelu(f32),
    /// Per-channel (or one shared) `alpha` and `1/alpha`, as in [`Snake`].
    Snake { alpha: Vec<f32>, inv_alpha: Vec<f32> },
}

impl AdaIn {
    fn plain() -> AdaIn {
        AdaIn { act: AdaInAct::Identity, in_f16: false, out_f16: false }
    }
}

impl Op for AdaIn {
    fn name(&self) -> StaticName {
        match self.act {
            AdaInAct::Identity => "AdaIN",
            AdaInAct::LeakyRelu(_) => "AdaIN+LeakyRelu",
            AdaInAct::Snake { .. } => "AdaIN+Snake",
        }
        .into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let dt = |f16: bool| if f16 { "f16" } else { "f32" };
        Ok(vec![format!("{} -> {}", dt(self.in_f16), dt(self.out_f16))])
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<AdaIn>().is_some_and(|o| o == self)
    }

    op_as_typed_op!();
}

const ADAIN_LANES: usize = 8;

/// What a row applies after normalising, with the channel's constants picked out.
#[derive(Clone, Copy)]
enum RowAct {
    Identity,
    LeakyRelu(f32),
    Snake { a: f32, inv_a: f32 },
}

/// One row, in place. The two reductions are exactly what the unfused graph runs, so
/// the result is bit-identical to it: `Reduce<Sum>` is tract-linalg's `sum_f32` kernel,
/// and `Reduce<SumOfSquares>` (`sum_sq_f32` in nn/reduce.rs) squares in f32 and
/// accumulates in 8 f64 lanes, remainder first. The centred value is formed on the fly
/// instead of materialising `Sub`'s output. The apply keeps the unfused association,
/// `((x - mean) * rs) * g + beta`. An f32 accumulation here moved the audio by 0.04
/// spectral convergence for no speed worth that.
///
/// `sum_f32` groups the row by the address it starts at (a 16-byte-aligned middle with
/// padded head and tail, see `reduce_slice_with_alignment`), so a row's address modulo 16
/// is part of what it computes. Callers must hand over rows with the address phase the
/// unfused graph's row had ([`with_row_scratch`]).
///
/// The activation follows the apply: Snake inside the same loop (the unfused Snake is
/// the same scalar expression per element), LeakyRelu through the very kernel the
/// standalone op calls.
fn adain_row(
    row: &mut [f32],
    d_mean: f32,
    d_var: f32,
    eps: f32,
    g: f32,
    beta: f32,
    act: RowAct,
) {
    let sum = (tract_linalg::ops().sum_f32)().run_with_params(&*row, ()).unwrap();
    let mean = sum * d_mean;

    let mut acc = [0f64; ADAIN_LANES];
    let mut chunks = row.chunks_exact(ADAIN_LANES);
    for c in &mut chunks {
        for l in 0..ADAIN_LANES {
            let d = c[l] - mean;
            acc[l] += (d * d) as f64;
        }
    }
    let mut sum_sq = 0f64;
    for &x in chunks.remainder() {
        let d = x - mean;
        sum_sq += (d * d) as f64;
    }
    for l in 0..ADAIN_LANES {
        sum_sq += acc[l];
    }
    let rs = ((sum_sq as f32) * d_var + eps).sqrt().recip();

    match act {
        RowAct::Identity => {
            for x in row.iter_mut() {
                *x = ((*x - mean) * rs) * g + beta;
            }
        }
        RowAct::Snake { a, inv_a } => {
            for x in row.iter_mut() {
                *x = snake_f32(((*x - mean) * rs) * g + beta, a, inv_a);
            }
        }
        RowAct::LeakyRelu(alpha) => {
            for x in row.iter_mut() {
                *x = ((*x - mean) * rs) * g + beta;
            }
            (tract_linalg::ops().leaky_relu_f32)().run_with_params(row, alpha).unwrap();
        }
    }
}

thread_local! {
    static ADAIN_ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on a reusable per-thread f32 row of `len` elements that starts `phase` f32s
/// (0..4) past a 16-byte boundary, so `sum_f32` groups it the way it grouped the row of
/// the f32 tensor this scratch stands in for.
fn with_row_scratch<R>(len: usize, phase: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    debug_assert!(phase < 4);
    ADAIN_ROW.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < len + 8 {
            buf.resize(len + 8, 0.0);
        }
        let start = buf.as_ptr().align_offset(16) + phase;
        f(&mut buf[start..start + len])
    })
}

impl EvalOp for AdaIn {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        use tract_data::half::slice::HalfFloatSliceExt;
        ensure!(inputs.len() == 6, "AdaIN takes x, g, beta, d_mean, d_var, eps");
        let x = inputs.remove(0);
        let in_dt = if self.in_f16 { f16::datum_type() } else { f32::datum_type() };
        ensure!(x.datum_type() == in_dt, "AdaIN expects {in_dt:?}, got {:?}", x.datum_type());
        ensure!(x.rank() == 3, "AdaIN needs [N, C, F], got {:?}", x.shape());
        let (n, c, f) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let scalar = |t: &TValue| -> TractResult<f32> { t.cast_to_scalar::<f32>() };
        let (d_mean, d_var, eps) = (scalar(&inputs[2])?, scalar(&inputs[3])?, scalar(&inputs[4])?);
        let g = inputs[0].cast_to::<f32>()?;
        let g = g.as_slice::<f32>()?;
        let beta = inputs[1].cast_to::<f32>()?;
        let beta = beta.as_slice::<f32>()?;
        ensure!(
            (g.len() == c || g.len() == n * c) && beta.len() == g.len(),
            "AdaIN: g/beta have {}/{} values for {n} x {c} rows",
            g.len(),
            beta.len()
        );
        if let AdaInAct::Snake { alpha, inv_alpha } = &self.act {
            ensure!(
                alpha.len() == inv_alpha.len() && (alpha.len() == 1 || alpha.len() == c),
                "AdaIN+Snake: {} alphas for {c} channels",
                alpha.len()
            );
        }
        let out_dt = if self.out_f16 { f16::datum_type() } else { f32::datum_type() };
        if n * c * f == 0 {
            return Ok(tvec!(Tensor::zero_dt(out_dt, x.shape())?.into_tvalue()));
        }
        // Row `r` is channel `r % c` of batch `r / c`. Every row is independent.
        let run = |r: usize, row: &mut [f32]| {
            let (ni, ci) = (r / c, r % c);
            let off = if g.len() == c { 0 } else { ni * c };
            let act = match &self.act {
                AdaInAct::Identity => RowAct::Identity,
                AdaInAct::LeakyRelu(alpha) => RowAct::LeakyRelu(*alpha),
                AdaInAct::Snake { alpha, inv_alpha } => {
                    let i = if alpha.len() == 1 { 0 } else { ci };
                    RowAct::Snake { a: alpha[i], inv_a: inv_alpha[i] }
                }
            };
            adain_row(row, d_mean, d_var, eps, g[off + ci], beta[off + ci], act);
        };
        use tract_linalg::multithread::par_chunks_mut;
        match (self.in_f16, self.out_f16) {
            // f32 in place.
            (false, false) => {
                let mut x = x.into_tensor();
                par_chunks_mut(x.as_slice_mut::<f32>()?, f, |r, row| run(r, row));
                Ok(tvec!(x.into_tvalue()))
            }
            // f16 in place; the widened row only ever exists in the scratch. The f32
            // tensor `Cast` would have made starts 16-byte aligned (allocator), so row
            // `r` sits `r * f` f32s into it.
            (true, true) => {
                let mut x = x.into_tensor();
                par_chunks_mut(x.as_slice_mut::<f16>()?, f, |r, row| {
                    with_row_scratch(f, (r * f) % 4, |s| {
                        row.convert_to_f32_slice(s);
                        run(r, s);
                        row.convert_from_f32_slice(s);
                    })
                });
                Ok(tvec!(x.into_tvalue()))
            }
            // Widen straight into the output rows, which are then normalised in place.
            (true, false) => {
                let src = x.as_slice::<f16>()?;
                // SAFETY: every row of `out` is written by `convert_to_f32_slice`.
                let mut out = unsafe { Tensor::uninitialized::<f32>(x.shape())? };
                par_chunks_mut(out.as_slice_mut::<f32>()?, f, |r, row| {
                    src[r * f..][..f].convert_to_f32_slice(row);
                    run(r, row);
                });
                Ok(tvec!(out.into_tvalue()))
            }
            // The f32 input row stays untouched (it may be shared); the scratch takes its
            // address phase, and is narrowed into the output.
            (false, true) => {
                let src = x.as_slice::<f32>()?;
                // SAFETY: every row of `out` is written by `convert_from_f32_slice`.
                let mut out = unsafe { Tensor::uninitialized::<f16>(x.shape())? };
                par_chunks_mut(out.as_slice_mut::<f16>()?, f, |r, row| {
                    let from = &src[r * f..][..f];
                    with_row_scratch(f, (from.as_ptr() as usize & 15) / 4, |s| {
                        s.copy_from_slice(from);
                        run(r, s);
                        row.convert_from_f32_slice(s);
                    })
                });
                Ok(tvec!(out.into_tvalue()))
            }
        }
    }
}

impl TypedOp for AdaIn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 6, "AdaIN takes x, g, beta, d_mean, d_var, eps");
        let in_dt = if self.in_f16 { f16::datum_type() } else { f32::datum_type() };
        ensure!(inputs[0].datum_type == in_dt, "AdaIN expects {in_dt:?} input");
        ensure!(
            inputs[1..].iter().all(|f| f.datum_type == f32::datum_type()),
            "AdaIN side inputs must be f32"
        );
        ensure!(inputs[0].rank() == 3, "AdaIN needs [N, C, F]");
        let out_dt = if self.out_f16 { f16::datum_type() } else { f32::datum_type() };
        Ok(tvec!(out_dt.fact(inputs[0].shape.clone())))
    }

    /// Absorb the neighbours, one per pass: the activation first, then the output cast,
    /// then the input cast. Each needs its edge to have no other reader and not to be a
    /// model output, or the value it would drop is still wanted.
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let me = OutletId::new(node.id, 0);
        let c = model.outlet_fact(node.inputs[0])?.shape.get(1).and_then(|d| d.as_i64());
        if !self.out_f16 {
            if let Some(next) = sole_reader(model, me) {
                if self.act == AdaInAct::Identity {
                    if let Some(act) = absorbable_act(next, c) {
                        let op = AdaIn { act, ..self.clone() };
                        return rewire_adain(model, node, node.inputs[0], op, next.id);
                    }
                }
                if is_cast_to(next, f16::datum_type()) {
                    let op = AdaIn { out_f16: true, ..self.clone() };
                    return rewire_adain(model, node, node.inputs[0], op, next.id);
                }
            }
        }
        if !self.in_f16 {
            let prev = model.node(node.inputs[0].node);
            if is_cast_to(prev, f32::datum_type())
                && model.outlet_fact(prev.inputs[0])?.datum_type == f16::datum_type()
                && sole_reader(model, node.inputs[0]).is_some_and(|r| r.id == node.id)
            {
                let op = AdaIn { in_f16: true, ..self.clone() };
                return rewire_adain(model, node, prev.inputs[0], op, node.id);
            }
        }
        Ok(None)
    }

    as_op!();
}

/// The one node reading `outlet`, unless it is a model output or has several readers.
fn sole_reader(model: &TypedModel, outlet: OutletId) -> Option<&TypedNode> {
    if model.output_outlets().ok()?.contains(&outlet) {
        return None;
    }
    match model.node(outlet.node).outputs.get(outlet.slot)?.successors.as_slice() {
        [only] => Some(model.node(only.node)),
        _ => None,
    }
}

/// A `Cast` to `to`.
fn is_cast_to(node: &TypedNode, to: DatumType) -> bool {
    node.op_as::<crate::ops::cast::Cast>().is_some_and(|cast| cast.to == to)
}

/// The activation `node` applies, if AdaIN can run it: a Snake or LeakyRelu on f32 whose
/// alphas fit the `c` channels.
fn absorbable_act(node: &TypedNode, c: Option<i64>) -> Option<AdaInAct> {
    let ew = node.op_as::<crate::ops::element_wise::ElementWiseOp>()?;
    if ew.1.is_some() || node.outputs[0].fact.datum_type != f32::datum_type() {
        return None;
    }
    if let Some(s) = ew.0.downcast_ref::<Snake>() {
        let fits = s.alpha.len() == s.inv_alpha.len()
            && (s.alpha.len() == 1 || c == Some(s.alpha.len() as i64));
        return fits.then(|| AdaInAct::Snake {
            alpha: s.alpha.clone(),
            inv_alpha: s.inv_alpha.clone(),
        });
    }
    let leaky = ew.0.downcast_ref::<crate::ops::nn::LeakyRelu>()?;
    Some(AdaInAct::LeakyRelu(leaky.alpha))
}

/// Replace `shunt` (the AdaIN itself, or the node absorbed after it) by `op` fed with `x`
/// and the AdaIN's five side inputs.
fn rewire_adain(
    model: &TypedModel,
    node: &TypedNode,
    x: OutletId,
    op: AdaIn,
    shunt: usize,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    let mut taps = tvec!(patch.tap_model(model, x)?);
    for o in &node.inputs[1..] {
        taps.push(patch.tap_model(model, *o)?);
    }
    let wire = patch.wire_node(&node.name, op, &taps)?[0];
    patch.shunt_outside(model, shunt.into(), wire)?;
    Ok(Some(patch))
}

fn is_mul(node: &TypedNode) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.is::<Mul>())
}

fn is_sub(node: &TypedNode) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.is::<Sub>())
}

fn is_add(node: &TypedNode) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.is::<Add>())
}

/// A reduction over the last axis of a rank-3 tensor.
fn is_reduce_f(node: &TypedNode, reducer: crate::ops::nn::Reducer) -> bool {
    node.op_as::<crate::ops::nn::Reduce>()
        .is_some_and(|r| r.reducer == reducer && r.axes.len() == 1 && r.axes[0] == 2)
}

/// An f32 outlet holding exactly one element.
fn is_f32_scalar(model: &TypedModel, o: OutletId) -> bool {
    model.outlet_fact(o).is_ok_and(|f| {
        f.datum_type == f32::datum_type() && f.shape.iter().all(|d| d.as_i64() == Some(1))
    })
}

/// The instance-norm half of the pattern: `Mul(Sub(x, mean), Rsqrt(var + eps))`.
struct NormParts {
    x: OutletId,
    d_mean: OutletId,
    d_var: OutletId,
    eps: OutletId,
    /// Every node the fusion removes, bar the final `Mul(g, ·)` and `Add`.
    inner: [usize; 8],
}

fn adain_norm(model: &TypedModel, n_out: OutletId) -> Option<NormParts> {
    let n = model.node(n_out.node);
    if n_out.slot != 0 || !is_mul(n) || !sole_consumer(model, n.id) {
        return None;
    }
    [(n.inputs[0], n.inputs[1]), (n.inputs[1], n.inputs[0])]
        .into_iter()
        .find_map(|(xc_out, rs_out)| adain_norm_with(model, n, xc_out, rs_out))
}

fn adain_norm_with(
    model: &TypedModel,
    n: &TypedNode,
    xc_out: OutletId,
    rs_out: OutletId,
) -> Option<NormParts> {
    use crate::ops::nn::Reducer;
    // xc = x - mean, read by the variance reduction and by `n` and nothing else.
    let xc = model.node(xc_out.node);
    if xc_out.slot != 0 || !is_sub(xc) {
        return None;
    }
    let x = xc.inputs[0];
    // mean = Sum(x) * d_mean
    let mean_out = xc.inputs[1];
    let mean = model.node(mean_out.node);
    if mean_out.slot != 0 || !is_mul(mean) || !sole_consumer(model, mean.id) {
        return None;
    }
    let (sum_id, d_mean) = [(mean.inputs[0], mean.inputs[1]), (mean.inputs[1], mean.inputs[0])]
        .into_iter()
        .find_map(|(s_out, d)| {
            let s = model.node(s_out.node);
            (s_out.slot == 0
                && is_reduce_f(s, Reducer::Sum)
                && s.inputs[0] == x
                && sole_consumer(model, s.id))
            .then_some((s.id, d))
        })?;
    // rs = Rsqrt(SumOfSquares(xc) * d_var + eps)
    let rs = model.node(rs_out.node);
    let ew = rs.op_as::<crate::ops::element_wise::ElementWiseOp>()?;
    if rs_out.slot != 0 || !ew.0.is::<Rsqrt>() || ew.1.is_some() || !sole_consumer(model, rs.id) {
        return None;
    }
    let e_out = rs.inputs[0];
    let e = model.node(e_out.node);
    if e_out.slot != 0 || !is_add(e) || !sole_consumer(model, e.id) {
        return None;
    }
    let (v_out, eps) = [(e.inputs[0], e.inputs[1]), (e.inputs[1], e.inputs[0])]
        .into_iter()
        .find(|(v, eps)| {
            is_mul(model.node(v.node))
                && model.outlet_fact(*eps).is_ok_and(|f| f.konst.is_some())
        })?;
    let v = model.node(v_out.node);
    if v_out.slot != 0 || !sole_consumer(model, v.id) {
        return None;
    }
    let (ss_id, d_var) = [(v.inputs[0], v.inputs[1]), (v.inputs[1], v.inputs[0])]
        .into_iter()
        .find_map(|(ss_out, d)| {
            let ss = model.node(ss_out.node);
            (ss_out.slot == 0
                && is_reduce_f(ss, Reducer::SumOfSquares)
                && ss.inputs[0] == xc_out
                && sole_consumer(model, ss.id))
            .then_some((ss.id, d))
        })?;
    let readers = &xc.outputs[0].successors;
    if readers.len() != 2
        || !readers.iter().any(|r| r.node == ss_id)
        || !readers.iter().any(|r| r.node == n.id)
    {
        return None;
    }
    Some(NormParts {
        x,
        d_mean,
        d_var,
        eps,
        inner: [xc.id, mean.id, sum_id, rs.id, e.id, v.id, ss_id, n.id],
    })
}

/// Fold `Add(Mul(g, Mul(Sub(x, mean), Rsqrt(var + eps))), beta)` into [`AdaIn`].
/// Runs from `Add`'s declutter, once tract has lowered `InstanceNormalization`'s
/// `ReduceMean`/`Div`/`Sqrt` to `Reduce`, `Mul(Recip)` and `Rsqrt`.
fn try_fuse_adain(model: &TypedModel, add: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    if add.inputs.len() != 2 || add.outputs[0].fact.datum_type != f32::datum_type() {
        return Ok(None);
    }
    for (m_out, beta) in [(add.inputs[0], add.inputs[1]), (add.inputs[1], add.inputs[0])] {
        let m = model.node(m_out.node);
        if m_out.slot != 0 || !is_mul(m) || !sole_consumer(model, m.id) {
            continue;
        }
        for (n_out, g) in [(m.inputs[0], m.inputs[1]), (m.inputs[1], m.inputs[0])] {
            let Some(p) = adain_norm(model, n_out) else { continue };
            // Gates the eval relies on: f32 x of rank 3 with a leading 1 and a known C,
            // per-channel g and beta, one-element d_mean/d_var/eps.
            let xf = model.outlet_fact(p.x)?;
            let (gf, bf) = (model.outlet_fact(g)?, model.outlet_fact(beta)?);
            let per_channel = |f: &TypedFact, c: i64| {
                f.datum_type == f32::datum_type()
                    && f.rank() == 3
                    && f.shape[0].as_i64() == Some(1)
                    && f.shape[1].as_i64() == Some(c)
                    && f.shape[2].as_i64() == Some(1)
            };
            if xf.datum_type != f32::datum_type() || xf.rank() != 3 {
                continue;
            }
            let Some(c) = xf.shape[1].as_i64() else { continue };
            if xf.shape[0].as_i64() != Some(1)
                || !per_channel(gf, c)
                || !per_channel(bf, c)
                || ![p.d_mean, p.d_var, p.eps].iter().all(|o| is_f32_scalar(model, *o))
            {
                continue;
            }
            // A removed node that is also a model output would lose its value.
            let outputs = model.output_outlets()?;
            if p.inner.iter().chain([&m.id]).any(|id| outputs.iter().any(|o| o.node == *id)) {
                continue;
            }
            let mut patch = TypedModelPatch::default();
            let mut taps = tvec!();
            for o in [p.x, g, beta, p.d_mean, p.d_var, p.eps] {
                taps.push(patch.tap_model(model, o)?);
            }
            let wire = patch.wire_node(&add.name, AdaIn::plain(), &taps)?[0];
            patch.shunt_outside(model, add.id.into(), wire)?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
}

/// `Add`'s declutter: Snake first, then AdaIN. Each only matches its own shape.
fn try_fuse_add(
    op: &Add,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(patch) = try_fuse_snake(op, model, node)? {
        return Ok(Some(patch));
    }
    try_fuse_adain(model, node)
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

    /// `sin_sq_f32` stays within 5e-7 of the f64 value for |r| < 12000, including
    /// multiples of pi/2 where the reduction flips, and never leaves [0, 1] there.
    #[test]
    fn sin_sq_f32_matches_f64() {
        let mut worst = 0f64;
        let mut check = |r: f32| {
            let got = super::sin_sq_f32(r);
            let expect = (r as f64).sin().powi(2);
            worst = worst.max((got as f64 - expect).abs());
            assert!((0.0..=1.0).contains(&got), "sin_sq_f32({r}) = {got} outside [0, 1]");
        };
        let mut r = -300.0f32;
        while r < 300.0 {
            check(r);
            r += 0.00137;
        }
        let mut r = -12_000.0f32;
        while r < 12_000.0 {
            check(r);
            r += 0.031;
        }
        for k in -200..=200 {
            let half_pi = k as f32 * std::f32::consts::FRAC_PI_2;
            for d in [-1e-3f32, 0.0, 1e-3] {
                check(half_pi + d);
            }
        }
        assert!(worst < 5e-7, "worst abs error {worst:e}");
        assert!(super::sin_sq_f32(f32::NAN).is_nan());
        assert_eq!(super::sin_sq_f32(0.0), 0.0);
    }

    /// The chain tract leaves for InstanceNormalization plus a style affine, on a
    /// small concrete tensor. `d` (1/F) is a source so declutter cannot fold it, as
    /// in the symbolic plan where it is a runtime `Recip`.
    fn adain_chain(xc_is_output: bool) -> TractResult<TypedModel> {
        adain_chain_io(xc_is_output, false, false)
    }

    /// With `f16_io`, the chain sits between `Cast(f16 -> f32)` and `LeakyRelu`,
    /// `Cast(f32 -> f16)`, the shape the FP16 translation leaves in the vocoder.
    /// `act_is_output` also makes the activation's output a model output.
    fn adain_chain_io(
        xc_is_output: bool,
        f16_io: bool,
        act_is_output: bool,
    ) -> TractResult<TypedModel> {
        use crate::ops::cast::cast;
        use crate::ops::nn::{leaky_relu, Reduce, Reducer};
        let mut model = TypedModel::default();
        let x = if f16_io {
            let x16 = model.add_source("x", f16::fact([1, 2, 8]))?;
            model.wire_node("up", cast(f32::datum_type()), &[x16])?[0]
        } else {
            model.add_source("x", f32::fact([1, 2, 8]))?
        };
        let d = model.add_source("d", f32::fact([1, 1, 1]))?;
        let g = model.add_source("g", f32::fact([1, 2, 1]))?;
        let b = model.add_source("b", f32::fact([1, 2, 1]))?;
        let eps = model.add_const("eps", Tensor::from_shape(&[1, 1, 1], &[1e-5f32])?)?;
        let sum = model.wire_node("sum", Reduce::new(tvec!(2), Reducer::Sum), &[x])?[0];
        let mean = model.wire_node("mean", mul(), &[sum, d])?[0];
        let xc = model.wire_node("xc", sub(), &[x, mean])?[0];
        let ss = model.wire_node("ss", Reduce::new(tvec!(2), Reducer::SumOfSquares), &[xc])?[0];
        let var = model.wire_node("var", mul(), &[ss, d])?[0];
        let ve = model.wire_node("ve", add(), &[var, eps])?[0];
        let rs = model.wire_node("rs", rsqrt(), &[ve])?[0];
        let norm = model.wire_node("norm", mul(), &[xc, rs])?[0];
        let scaled = model.wire_node("scaled", mul(), &[g, norm])?[0];
        let mut y = model.wire_node("y", add(), &[scaled, b])?[0];
        let mut outputs = vec![];
        if f16_io {
            y = model.wire_node("act", leaky_relu(0.2), &[y])?[0];
            if act_is_output {
                outputs.push(y);
            }
            y = model.wire_node("down", cast(f16::datum_type()), &[y])?[0];
        }
        outputs.insert(0, y);
        if xc_is_output {
            outputs.push(xc);
        }
        model.set_output_outlets(&outputs)?;
        Ok(model)
    }

    /// The whole chain becomes one AdaIN op, and it computes what the chain did.
    #[test]
    fn adain_fuses_and_matches_the_unfused_chain() -> TractResult<()> {
        let model = adain_chain(false)?;
        let fused = model.clone().into_decluttered()?;
        let names: Vec<String> = fused.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(names.iter().any(|n| n == "AdaIN"), "not fused: {names:?}");
        for gone in ["Reduce<Sum>", "Reduce<SumOfSquares>", "Rsqrt"] {
            assert!(!names.iter().any(|n| n == gone), "{gone} survived: {names:?}");
        }
        let xs: Vec<f32> = (0..16).map(|i| ((i * 7 % 11) as f32) * 0.3 - 1.2).collect();
        let inputs = || -> TractResult<TVec<TValue>> {
            Ok(tvec!(
                Tensor::from_shape(&[1, 2, 8], &xs)?.into(),
                Tensor::from_shape(&[1, 1, 1], &[1.0f32 / 8.0])?.into(),
                Tensor::from_shape(&[1, 2, 1], &[1.3f32, 0.7])?.into(),
                Tensor::from_shape(&[1, 2, 1], &[-0.2f32, 0.4])?.into(),
            ))
        };
        let want = SimplePlan::new(&model)?.run(inputs()?)?;
        let got = SimplePlan::new(&fused)?.run(inputs()?)?;
        let (want, got) = (want[0].as_slice::<f32>()?, got[0].as_slice::<f32>()?);
        for (i, (w, g)) in want.iter().zip(got).enumerate() {
            assert!((w - g).abs() < 1e-5, "i={i} fused {g} unfused {w}");
        }
        Ok(())
    }

    /// If something else needs an intermediate (here `xc` is a model output), the
    /// chain stays as it is.
    #[test]
    fn adain_skips_when_an_intermediate_is_an_output() -> TractResult<()> {
        let fused = adain_chain(true)?.into_decluttered()?;
        let names: Vec<String> = fused.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(!names.iter().any(|n| n == "AdaIN"), "fused although xc is an output: {names:?}");
        Ok(())
    }

    /// Between f16 casts the chain, its LeakyRelu and both casts become one op, and it
    /// reproduces the chain (to f16 rounding: the unfused reductions can differ in the last
    /// f32 bits from the AdaIN op's).
    #[test]
    fn adain_absorbs_activation_and_casts() -> TractResult<()> {
        let model = adain_chain_io(false, true, false)?;
        let fused = model.clone().into_decluttered()?;
        let names: Vec<String> = fused.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(names.iter().any(|n| n == "AdaIN+LeakyRelu"), "not fused: {names:?}");
        for gone in ["Cast", "LeakyRelu", "AdaIN"] {
            assert!(!names.iter().any(|n| n == gone), "{gone} survived: {names:?}");
        }
        let xs: Vec<f16> =
            (0..16).map(|i| f16::from_f32(((i * 7 % 11) as f32) * 0.3 - 1.2)).collect();
        let inputs = || -> TractResult<TVec<TValue>> {
            Ok(tvec!(
                Tensor::from_shape(&[1, 2, 8], &xs)?.into(),
                Tensor::from_shape(&[1, 1, 1], &[1.0f32 / 8.0])?.into(),
                Tensor::from_shape(&[1, 2, 1], &[1.3f32, 0.7])?.into(),
                Tensor::from_shape(&[1, 2, 1], &[-0.2f32, 0.4])?.into(),
            ))
        };
        let want = SimplePlan::new(&model)?.run(inputs()?)?;
        let got = SimplePlan::new(&fused)?.run(inputs()?)?;
        assert_eq!(got[0].datum_type(), f16::datum_type());
        let (want, got) = (want[0].as_slice::<f16>()?, got[0].as_slice::<f16>()?);
        for (i, (w, g)) in want.iter().zip(got).enumerate() {
            assert!((w.to_f32() - g.to_f32()).abs() < 2e-3, "i={i} fused {g} unfused {w}");
        }
        Ok(())
    }

    /// A model output on the activation's result keeps the output cast out of the fusion
    /// (its input is still wanted), but the activation itself still folds in.
    #[test]
    fn adain_keeps_a_cast_whose_input_is_an_output() -> TractResult<()> {
        let fused = adain_chain_io(false, true, true)?.into_decluttered()?;
        let names: Vec<String> = fused.nodes.iter().map(|n| n.op().name().to_string()).collect();
        assert!(names.iter().any(|n| n == "AdaIN+LeakyRelu"), "activation not folded: {names:?}");
        assert!(names.iter().any(|n| n == "Cast"), "output cast was removed: {names:?}");
        Ok(())
    }

    /// Every storage combination, activation and row length gives the very bits the
    /// separate nodes (`Cast`, plain AdaIN, Snake / LeakyRelu, `Cast`) give. The lengths
    /// walk every address phase of a row, which `sum_f32`'s grouping depends on.
    #[test]
    fn adain_fused_forms_match_the_separate_nodes_bit_for_bit() -> TractResult<()> {
        use crate::ops::element_wise::ElementWiseOp;
        use crate::ops::nn::leaky_relu;
        let bits = |t: &Tensor| -> Vec<u32> {
            let t = t.cast_to::<f32>().unwrap();
            t.as_slice::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        let (n, c) = (2usize, 3usize);
        let alpha = vec![0.5f32, 2.0, 1.5];
        let inv_alpha: Vec<f32> = alpha.iter().map(|a| 1.0 / a).collect();
        let acts = [
            AdaInAct::Identity,
            AdaInAct::LeakyRelu(0.2),
            AdaInAct::Snake { alpha: alpha.clone(), inv_alpha: inv_alpha.clone() },
            AdaInAct::Snake { alpha: vec![0.75], inv_alpha: vec![1.0 / 0.75] },
        ];
        for f in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100, 257, 1001] {
            let xs: Vec<f32> = (0..n * c * f)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.07 + (i / f) as f32 * 0.3)
                .collect();
            let x32 = Tensor::from_shape(&[n, c, f], &xs)?;
            let x16 = x32.cast_to::<f16>()?.into_owned();
            // The reference is built from the same f16-rounded values.
            let x_up = x16.cast_to::<f32>()?.into_owned();
            let side = || -> TVec<TValue> {
                tvec!(
                    Tensor::from_shape(&[n, c, 1], &[1.3f32, 0.7, 1.1, 0.9, 1.2, 0.8]).unwrap().into(),
                    Tensor::from_shape(&[n, c, 1], &[-0.2f32, 0.4, 0.1, -0.3, 0.2, 0.0]).unwrap().into(),
                    Tensor::from_shape(&[1, 1, 1], &[1.0f32 / f as f32]).unwrap().into(),
                    Tensor::from_shape(&[1, 1, 1], &[1.0f32 / f as f32]).unwrap().into(),
                    Tensor::from_shape(&[1, 1, 1], &[1e-5f32]).unwrap().into(),
                )
            };
            let with_x = |x: &Tensor| -> TVec<TValue> {
                let mut v = tvec!(x.clone().into_tvalue());
                v.extend(side());
                v
            };
            for act in &acts {
                // Separate nodes, on the widened values.
                let mut want = AdaIn::plain().eval(with_x(&x_up))?;
                match act {
                    AdaInAct::Identity => {}
                    AdaInAct::LeakyRelu(a) => want = leaky_relu(*a).eval(want)?,
                    AdaInAct::Snake { alpha, inv_alpha } => {
                        let op = Snake { alpha: alpha.clone(), inv_alpha: inv_alpha.clone() };
                        want = ElementWiseOp(Box::new(op), None).eval(want)?;
                    }
                }
                let want32 = want[0].clone().into_tensor();
                let want16 = want32.cast_to::<f16>()?.into_owned();
                for (in_f16, out_f16) in [(false, false), (false, true), (true, false), (true, true)]
                {
                    let op = AdaIn { act: act.clone(), in_f16, out_f16 };
                    let got = op.eval(with_x(if in_f16 { &x16 } else { &x_up }))?;
                    let want = if out_f16 { &want16 } else { &want32 };
                    assert_eq!(got[0].datum_type(), want.datum_type());
                    assert_eq!(
                        bits(&got[0]),
                        bits(want),
                        "f={f} {act:?} in_f16={in_f16} out_f16={out_f16}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The scratch row starts where it is told to, whatever the thread's buffer is.
    #[test]
    fn row_scratch_has_the_requested_address_phase() {
        for len in [1usize, 5, 64, 1001, 70_201, 17] {
            for phase in 0..4 {
                super::with_row_scratch(len, phase, |s| {
                    assert_eq!(s.len(), len);
                    assert_eq!(s.as_ptr() as usize % 16, phase * 4, "len={len} phase={phase}");
                });
            }
        }
    }

    /// The premise of the phase handling: `adain_row` really does depend on the address
    /// its row starts at (through `sum_f32`'s grouping), so a fused kernel that moved rows
    /// to differently-aligned scratch would not reproduce the unfused bits. If this ever
    /// fails, the phase bookkeeping in `AdaIn::eval` is unnecessary on that CPU.
    #[test]
    fn adain_row_depends_on_the_row_address_phase() {
        let f = 1001usize;
        let xs: Vec<f32> = (0..f).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.0713 + 0.3).collect();
        let run = |phase: usize| -> Vec<u32> {
            super::with_row_scratch(f, phase, |s| {
                s.copy_from_slice(&xs);
                super::adain_row(s, 1.0 / f as f32, 1.0 / f as f32, 1e-5, 1.3, -0.2, super::RowAct::Identity);
                s.iter().map(|v| v.to_bits()).collect()
            })
        };
        let by_phase: Vec<Vec<u32>> = (0..4).map(run).collect();
        let distinct = (1..4).filter(|p| by_phase[*p] != by_phase[0]).count();
        assert!(distinct > 0, "all four address phases gave identical bits");
    }

    /// `Add(x, Mul(SinSq(Mul(x, α)), 1/α))` becomes one Snake pass, and the
    /// values match the exact formula.
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
                let s = (alphas[c] as f64 * x as f64).sin();
                let expect = (x as f64 + invs[c] as f64 * s * s) as f32;
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
