use crate::internal::*;
use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry};
use crate::ops::matmul::pack::DynPackedOpaqueFact;
use std::fmt::{Debug, Display};
use std::ops::Range;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue};
use tract_linalg::pack::{PackedFormat, PackingWriter};
use tract_linalg::WeightType;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LazyIm2colParams {
    pub packer: PackedFormat,
    pub n_byte_offsets: Vec<isize>,
    pub k_byte_offsets: Vec<isize>,
    /// Padding fold (rank-1). For each kernel tap, the output-position range whose
    /// read lands in-bounds of the *unpadded* input; a read at an output position
    /// outside it is zero — exactly what an explicit zero-`Pad` would have supplied,
    /// but without the full-tensor copy. `interior` is the intersection over all
    /// taps: the output range where *every* tap is in-bounds, so panels fully inside
    /// it use the unchecked gather. Empty `tap_valid` (with a full `interior`) means
    /// the input was pre-padded (multi-dim path) — the checked gather is never used.
    pub tap_valid: Vec<Range<isize>>,
    pub interior: Range<isize>,
}

/// Build the lazy im2col gather offsets from a *concrete* pool geometry. The
/// `n_byte_offsets` (one per output position) and `k_byte_offsets` (one per
/// input-channel × kernel-tap) both depend on the concrete spatial length, so
/// this is called either at codegen (concrete input) or per-eval (symbolic
/// length — see [`LazyDeferred`]).
///
/// When the geometry carries padding (rank-1 pad-fold path), the offsets can point
/// out of bounds at the first/last output positions; `tap_valid`/`interior` capture
/// exactly where, so the gather can zero-fill those reads instead of a `Pad` copy.
pub fn build_lazy_params(
    geo: &ConcretePoolGeometry,
    input_channels: usize,
    datum_type: DatumType,
    packer: PackedFormat,
) -> LazyIm2colParams {
    let size_of_b = datum_type.size_of() as isize;
    let c_stride = geo.input_shape.c_stride();
    let n_byte_offsets: Vec<isize> =
        geo.patch.centers_offsets().into_iter().map(|x| x * size_of_b).collect();
    let k_byte_offsets: Vec<isize> = (0..input_channels)
        .flat_map(|ici| {
            geo.patch
                .standard_layout_data_field
                .iter()
                .map(move |x| (x + (ici * c_stride) as isize) * size_of_b)
        })
        .collect();

    // Per-tap in-bounds output range (same formula as the eager `padded_1d`
    // patcher): output position `n` reads valid input iff `n*stride + dx` lands in
    // `[0, input_width)`, where `dx` is the tap's signed input coord. Only rank-1;
    // the multi-dim path keeps its explicit Pad, so it reports no padding here.
    let n_len = n_byte_offsets.len() as isize;
    let (tap_valid, interior) = if geo.patch.rank() == 1 {
        use num_integer::Integer;
        let stride = geo.patch.spec.strides[0] as isize;
        let input_width = geo.patch.spec.input_shape[0] as isize;
        let output_width = geo.patch.output_shape[0] as isize;
        let kernel_len = geo.patch.standard_layout_data_field.len();
        let mut tap_valid = Vec::with_capacity(kernel_len);
        let (mut lo, mut hi) = (0isize, output_width);
        for t in 0..kernel_len {
            let dx = geo.patch.data_field[[t, 0]];
            let start = Integer::div_ceil(&(-dx), &stride).max(0).min(output_width);
            let end = Integer::div_ceil(&(input_width - dx), &stride).min(output_width);
            lo = lo.max(start);
            hi = hi.min(end);
            tap_valid.push(start..end);
        }
        (tap_valid, lo..hi.max(lo))
    } else {
        (vec![], 0..n_len)
    };
    LazyIm2colParams { packer, n_byte_offsets, k_byte_offsets, tap_valid, interior }
}

/// A lazy im2col whose gather offsets can't be baked at codegen because the
/// conv runs on a symbolic length axis (one compiled plan serving every phoneme
/// / frame count — see `src/bin/kokoro_tract.rs`). Everything here is
/// length-independent; the offsets are rebuilt per-eval from the concrete input.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LazyDeferred {
    pub pool_geo: PoolGeometry,
    pub packer: PackedFormat,
    pub input_channels: usize,
    pub k: usize,
    pub mn: TDim,
    pub datum_type: DatumType,
}

impl LazyDeferred {
    fn build(&self, input_full_shape: &[usize]) -> TractResult<LazyIm2colParams> {
        let geo = self.pool_geo.to_concrete(input_full_shape)?;
        Ok(build_lazy_params(&geo, self.input_channels, self.datum_type, self.packer.clone()))
    }
}

/// Source of a [`LazyIm2Col`]'s offsets: baked at codegen (concrete input) or
/// deferred to eval (symbolic length).
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum LazyParams {
    Ready(Arc<LazyIm2colParams>),
    Deferred(Arc<LazyDeferred>),
}

impl MMMInputFormat for LazyIm2colParams {
    fn r(&self) -> usize {
        self.packer.r
    }

    fn precursor(&self) -> WeightType {
        self.packer.precursor()
    }

    fn prepare_tensor(&self, _t: &Tensor, _k_axis: usize, _mn_axis: usize) -> TractResult<Tensor> {
        bail!("Unexpected call to prepare_tensor on LazyIm2Col")
    }

    fn k_alignment(&self) -> usize {
        1
    }

    fn same_as(&self, other: &dyn MMMInputFormat) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| self == other)
    }

    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        k * mn * self.packer.dt.size_of()
    }

    fn extract_at_mn_f16(
        &self,
        _data: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _slice: &mut [f16],
    ) -> TractResult<()> {
        unimplemented!()
    }
    fn extract_at_mn_f32(
        &self,
        _data: &tract_linalg::mmm::EagerPackedInput,
        _mn: usize,
        _slice: &mut [f32],
    ) -> TractResult<()> {
        unimplemented!()
    }

    fn prepare_one(
        &self,
        _t: &Tensor,
        _k_axis: usize,
        _mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        bail!("Unexpected call to prepare_one on LazyIm2Col")
    }
}

impl Display for LazyIm2colParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LazyIm2Col")
    }
}

impl OpaqueFact for LazyIm2colParams {
    fn mem_size(&self) -> TDim {
        MMMInputFormat::mem_size(
            self,
            self.k_byte_offsets.len().to_dim(),
            self.n_byte_offsets.len().to_dim(),
        )
    }

    fn same_as(&self, _other: &dyn OpaqueFact) -> bool {
        _other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LazyIm2Col {
    pub params: LazyParams,
}

impl Op for LazyIm2Col {
    fn name(&self) -> StaticName {
        "LazyIm2col".into()
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for LazyIm2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let tensor = args_1!(inputs);
        // Symbolic length: rebuild the gather offsets now that the shape is known.
        let params = match &self.params {
            LazyParams::Ready(p) => p.clone(),
            LazyParams::Deferred(d) => Arc::new(d.build(tensor.shape())?),
        };
        let input: Box<dyn MMMInputValue> = Box::new(LazyIm2colInput { tensor, im2col: params });
        let input = Opaque(Arc::new(input));
        Ok(tvec!(tensor2(&[[input]]).into_tvalue()))
    }
}

impl TypedOp for LazyIm2Col {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // k is length-independent; mn (output positions) is symbolic when deferred.
        let (k, mn, packer) = match &self.params {
            LazyParams::Ready(p) => (
                p.k_byte_offsets.len().to_dim(),
                p.n_byte_offsets.len().to_dim(),
                p.packer.clone(),
            ),
            LazyParams::Deferred(d) => (d.k.to_dim(), d.mn.clone(), d.packer.clone()),
        };
        let opaque_fact = DynPackedOpaqueFact { k, mn, packers: vec![packer] };
        Ok(tvec!(Opaque::fact([1, 1]).with_opaque_fact(opaque_fact)))
    }

    as_op!();
}

#[derive(Clone, Debug)]
struct LazyIm2colInput {
    tensor: TValue,
    im2col: Arc<LazyIm2colParams>,
}

impl Display for LazyIm2colInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Hash for LazyIm2colInput {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.tensor.as_bytes(), &self.im2col).hash(state);
    }
}

unsafe impl Send for LazyIm2colInput {}
unsafe impl Sync for LazyIm2colInput {}

impl LazyIm2colInput {
    fn input_8n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
        let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            let o7 = *n_byte_offsets.offset(n + 6);
            let o8 = *n_byte_offsets.offset(n + 7);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                let v3 = *(ptr.offset(o3) as *const T);
                let v4 = *(ptr.offset(o4) as *const T);
                let v5 = *(ptr.offset(o5) as *const T);
                let v6 = *(ptr.offset(o6) as *const T);
                let v7 = *(ptr.offset(o7) as *const T);
                let v8 = *(ptr.offset(o8) as *const T);
                writer.write(v1);
                writer.write(v2);
                writer.write(v3);
                writer.write(v4);
                writer.write(v5);
                writer.write(v6);
                writer.write(v7);
                writer.write(v8);
            }
        }
    }

    fn input_6n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            let o5 = *n_byte_offsets.offset(n + 4);
            let o6 = *n_byte_offsets.offset(n + 5);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                let v3 = *(ptr.offset(o3) as *const T);
                let v4 = *(ptr.offset(o4) as *const T);
                let v5 = *(ptr.offset(o5) as *const T);
                let v6 = *(ptr.offset(o6) as *const T);
                writer.write(v1);
                writer.write(v2);
                writer.write(v3);
                writer.write(v4);
                writer.write(v5);
                writer.write(v6);
            }
        }
    }

    fn input_4n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            let o3 = *n_byte_offsets.offset(n + 2);
            let o4 = *n_byte_offsets.offset(n + 3);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                let v3 = *(ptr.offset(o3) as *const T);
                let v4 = *(ptr.offset(o4) as *const T);
                writer.write(v1);
                writer.write(v2);
                writer.write(v3);
                writer.write(v4);
            }
        }
    }

    fn input_2n<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: Range<isize>,
        n: isize,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let o1 = *n_byte_offsets.offset(n);
            let o2 = *n_byte_offsets.offset(n + 1);
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let v1 = *(ptr.offset(o1) as *const T);
                let v2 = *(ptr.offset(o2) as *const T);
                writer.write(v1);
                writer.write(v2);
            }
        }
    }

    fn write<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: std::ops::Range<isize>,
        mn_range: std::ops::Range<isize>,
    ) {
        let mn_end = mn_range.end.min(self.im2col.n_byte_offsets.len() as isize);
        let n_range = mn_range.start..mn_end;
        match n_range.len() {
            8 => return self.input_8n(writer, k_range, n_range.start),
            6 => return self.input_6n(writer, k_range, n_range.start),
            4 => return self.input_4n(writer, k_range, n_range.start),
            2 => return self.input_2n(writer, k_range, n_range.start),
            _ => (),
        }
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            for k in k_range.start..k_range.end {
                let ptr = ptr.offset(*k_byte_offsets.offset(k));
                let mut n = n_range.start;
                while n + 8 <= n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
                    let o5 = *n_byte_offsets.offset(n + 4);
                    let o6 = *n_byte_offsets.offset(n + 5);
                    let o7 = *n_byte_offsets.offset(n + 6);
                    let o8 = *n_byte_offsets.offset(n + 7);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    let v5 = *(ptr.offset(o5) as *const T);
                    let v6 = *(ptr.offset(o6) as *const T);
                    let v7 = *(ptr.offset(o7) as *const T);
                    let v8 = *(ptr.offset(o8) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    writer.write(v5);
                    writer.write(v6);
                    writer.write(v7);
                    writer.write(v8);
                    n += 8;
                }
                while n + 6 <= n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
                    let o5 = *n_byte_offsets.offset(n + 4);
                    let o6 = *n_byte_offsets.offset(n + 5);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    let v5 = *(ptr.offset(o5) as *const T);
                    let v6 = *(ptr.offset(o6) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    writer.write(v5);
                    writer.write(v6);
                    n += 6;
                }
                while n + 4 <= n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let o2 = *n_byte_offsets.offset(n + 1);
                    let o3 = *n_byte_offsets.offset(n + 2);
                    let o4 = *n_byte_offsets.offset(n + 3);
                    let v1 = *(ptr.offset(o1) as *const T);
                    let v2 = *(ptr.offset(o2) as *const T);
                    let v3 = *(ptr.offset(o3) as *const T);
                    let v4 = *(ptr.offset(o4) as *const T);
                    writer.write(v1);
                    writer.write(v2);
                    writer.write(v3);
                    writer.write(v4);
                    n += 4;
                }
                while n < n_range.end {
                    let o1 = *n_byte_offsets.offset(n);
                    let v1 = *(ptr.offset(o1) as *const T);
                    writer.write(v1);
                    n += 1;
                }
            }
        }
    }

    /// Bounds-checked gather for the ≤2 boundary panels of a padded rank-1 conv:
    /// each read is emitted only if the output position is in the tap's valid range
    /// (see `LazyIm2colParams::tap_valid`), otherwise a zero is written — replacing
    /// the explicit zero-`Pad` copy. k-outer/n-inner, matching `write`'s ordering.
    /// Cold path (interior panels use the unchecked `write`), so it isn't unrolled.
    fn write_checked<T: Datum + Copy>(
        &self,
        writer: &mut impl PackingWriter<T>,
        k_range: std::ops::Range<isize>,
        mn_range: std::ops::Range<isize>,
    ) {
        unsafe {
            let ptr = self.tensor.as_ptr_unchecked::<u8>();
            let k_byte_offsets = self.im2col.k_byte_offsets.as_ptr();
            let n_byte_offsets = self.im2col.n_byte_offsets.as_ptr();
            let kernel_len = self.im2col.tap_valid.len();
            // Pad value is always the constant zero (see `wire_as_lazy_im2col`); the
            // all-zero bit pattern is 0 for every numeric datum type gathered here.
            let zero: T = std::mem::zeroed();
            for k in k_range.start..k_range.end {
                let base = ptr.offset(*k_byte_offsets.offset(k));
                let valid = self.im2col.tap_valid.get_unchecked((k as usize) % kernel_len);
                for n in mn_range.start..mn_range.end {
                    if n >= valid.start && n < valid.end {
                        writer.write(*(base.offset(*n_byte_offsets.offset(n)) as *const T));
                    } else {
                        writer.write(zero);
                    }
                }
            }
        }
    }
}

impl MMMInputValue for LazyIm2colInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        let k = self.im2col.k_byte_offsets.len();
        Some(self.im2col.packer.single_panel_layout(k, self.tensor.datum_type().size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        Ok(dispatch_copy!(Self::do_panel(self.tensor.datum_type())(self, i, buffer)))
    }

    fn k(&self) -> usize {
        self.im2col.k_byte_offsets.len()
    }

    fn mn(&self) -> usize {
        self.im2col.n_byte_offsets.len()
    }

    fn format(&self) -> &dyn MMMInputFormat {
        &*self.im2col
    }

    fn opaque_fact(&self) -> &dyn OpaqueFact {
        &*self.im2col
    }

    fn same_as(&self, other: &dyn MMMInputValue) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| {
            o.tensor == self.tensor && (&*o.im2col as &dyn MMMInputFormat).same_as(&*self.im2col)
        })
    }
    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        unimplemented!()
    }
    fn extract_at_mn_f32(&self, _mn: usize, _slice: &mut [f32]) -> TractResult<()> {
        unimplemented!()
    }
}

impl LazyIm2colInput {
    fn do_panel<T: Datum + Copy>(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let r = self.im2col.packer.r;
        let mn_start = i * r;
        let mn_end = (mn_start + self.im2col.packer.r).min(self.im2col.n_byte_offsets.len());
        let k = self.im2col.k_byte_offsets.len();
        let mn_range = mn_start as isize..mn_end as isize;
        let k_range = 0..k as isize;
        let packed = buffer.unwrap();
        // A panel needs the bounds-checked gather only if the pad-fold is active
        // (`tap_valid` non-empty) and it isn't fully inside the all-in-bounds
        // interior — i.e. only the first/last panel of a padded rank-1 conv.
        let interior = &self.im2col.interior;
        let checked = !self.im2col.tap_valid.is_empty()
            && !(mn_range.start >= interior.start && mn_range.end <= interior.end);
        if mn_range.len() == r && mn_start % r == 0 {
            let mut writer = self.im2col.packer.write_single_panel_with_k_outer(packed as *mut T);
            if checked {
                self.write_checked(&mut writer, k_range, mn_range);
            } else {
                self.write(&mut writer, k_range, mn_range);
            }
        } else {
            let mut writer = self.im2col.packer.write_with_k_outer(
                packed as *mut T,
                k_range.len(),
                mn_range.len(),
            );
            if checked {
                self.write_checked(&mut writer, k_range, mn_range);
            } else {
                self.write(&mut writer, k_range, mn_range);
            }
        }
        packed
    }
}
