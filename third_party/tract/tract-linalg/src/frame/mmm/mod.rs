#[macro_use]
mod macros;

pub mod cost_model;
#[macro_use]
pub(crate) mod fuse;
pub(crate) mod input_store;
pub(crate) mod kernel;
#[macro_use]
pub(crate) mod panel_extract;
mod scratch;
mod storage;

#[cfg(test)]
#[macro_use]
pub mod tests;

use crate::multithread::Executor;
#[cfg(feature = "multithread-mm")]
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Debug;
use tract_data::internal::*;

pub use cost_model::*;
pub use fuse::*;
pub use input_store::*;
pub use kernel::*;
pub use panel_extract::*;
pub use scratch::*;
pub use storage::*;

pub fn no_prefetch(_ptr: *const u8, _len: usize) {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ImplementationQuality {
    /// Individual operations are emulated by individual conversion (f16->f32->f16)
    Dreadful,
    /// Rust scalar operation (with whatever optimisation the compiler manages)
    Generic,
    /// Implicit vectorization (e.g. Rust code, some unrolled loops, explicit template instantiations for small constant)
    RustOptimized,
    /// Explicit vectorization (e.g. intrinsics vector code)
    TargetOptimized,
    /// Hand optimized (assembly)
    ManuallyOptimized,
}

impl ImplementationQuality {
    pub fn best_to_worst() -> &'static [ImplementationQuality] {
        use ImplementationQuality::*;
        &[ManuallyOptimized, TargetOptimized, RustOptimized, Generic, Dreadful]
    }

    pub fn cost(&self) -> usize {
        ImplementationQuality::best_to_worst().iter().position(|x| x == self).unwrap()
    }
}

impl PartialOrd for ImplementationQuality {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(usize::from(*self).cmp(&usize::from(*other)))
    }
}

impl From<ImplementationQuality> for usize {
    fn from(value: ImplementationQuality) -> Self {
        value.cost()
    }
}

pub trait MatMatMul: Debug + dyn_clone::DynClone + Send + Sync + std::any::Any {
    fn name(&self) -> &str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn quality(&self) -> ImplementationQuality;
    fn dynamic_boost(&self) -> isize;

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)];

    fn internal_type(&self) -> DatumType;

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec;

    fn can_fuse(&self, spec: &FusedSpec) -> bool;

    fn stores(&self) -> Cow<'_, [DatumType]>;

    unsafe fn run(&self, m: usize, n: usize, non_linear: &[FusedSpec]) -> TractResult<()> {
        unsafe {
            let mut scratch = self.allocate_scratch_space();
            self.run_with_scratch_space(m, n, &mut *scratch, non_linear)
        }
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace>;
    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool;
    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

impl PartialEq for Box<dyn MatMatMul> {
    fn eq(&self, other: &Box<dyn MatMatMul>) -> bool {
        self.name() == other.name()
    }
}

impl std::hash::Hash for Box<dyn MatMatMul> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<K: MatMatMulKer> MatMatMul for K {
    fn name(&self) -> &str {
        self.name()
    }
    fn mr(&self) -> usize {
        self.mr()
    }
    fn nr(&self) -> usize {
        self.nr()
    }

    fn quality(&self) -> ImplementationQuality {
        MatMatMulKer::quality(self)
    }

    fn dynamic_boost(&self) -> isize {
        MatMatMulKer::dynamic_boost(self)
    }

    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)] {
        self.packings()
    }

    fn internal_type(&self) -> DatumType {
        K::Acc::datum_type()
    }

    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        self.can_fuse(spec)
    }

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec {
        OutputStoreSpec::View { m_axis, n_axis, mr: self.mr(), nr: self.nr() }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec {
        OutputStoreSpec::Strides {
            row_byte_stride: row_stride * item_size as isize,
            col_byte_stride: col_stride * item_size as isize,
            mr: self.mr(),
            nr: self.nr(),
        }
    }

    fn stores(&self) -> Cow<'_, [DatumType]> {
        self.stores()
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::<ScratchSpaceImpl<K::Acc>>::default()
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceImpl<K::Acc>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()> {
        unsafe {
            let scratch = scratch
                .downcast_mut::<ScratchSpaceImpl<K::Acc>>()
                .context("Wrong scratch space type")?;
            scratch.prepare(self, m, n, non_linear)?;
            if n == 1 && self.nr() == 1 {
                run_with_scratch_space_vec(self, m, scratch, non_linear)
            } else {
                let (mut prefer_col, mut prefer_row) = (0, 0);
                for uop in non_linear.iter() {
                    if let Some(col) = uop.prefer_col_outer() {
                        prefer_col = col as usize;
                        prefer_row = (!col) as usize;
                    }
                }
                if prefer_col > prefer_row {
                    run_with_scratch_space_col_outer(self, m, n, scratch, non_linear)
                } else {
                    run_with_scratch_space_row_outer(self, m, n, scratch, non_linear)
                }
            }
        }
    }
}

/// Keep small matrix work on the calling core rather than waking the pool.
/// The estimate counts multiply-accumulates, including every fused product.
/// A zero override restores the previous dispatch path for controlled A/B runs.
fn matmul_executor(m: usize, n: usize, outer_tiles: usize, ops: &[FusedSpec]) -> Executor {
    let executor = crate::multithread::current_tract_executor();
    #[cfg(feature = "multithread-mm")]
    if let Executor::MultiThread(_) = &executor {
        struct Policy { minimum_work: usize, trace: bool }
        static POLICY: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
        let policy = POLICY.get_or_init(|| Policy {
            minimum_work: std::env::var("TRACT_CPU_MMM_MIN_WORK").ok()
                .and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    if cfg!(all(target_arch = "aarch64", target_os = "android")) { 1 << 18 } else { 0 }
                }),
            trace: std::env::var("TRACT_CPU_MMM_TRACE").is_ok_and(|v| v == "1"),
        });
        if policy.minimum_work != 0 || policy.trace {
            let k = ops.iter().filter_map(|op| match op {
                FusedSpec::AddMatMul { a, .. } => Some(a.k()),
                _ => None,
            }).fold(0usize, usize::saturating_add);
            let work = m.saturating_mul(n).saturating_mul(k);
            let serial = policy.minimum_work != 0
                && (outer_tiles <= 1 || (k > 0 && work <= policy.minimum_work));
            if policy.trace {
                type Shape = (usize, usize, usize, usize, bool);
                static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<Shape>>> = std::sync::OnceLock::new();
                let first = SEEN.get_or_init(Default::default).lock().unwrap()
                    .insert((m, n, k, outer_tiles, serial));
                if first {
                    eprintln!("tract-mmm-dispatch m={m} n={n} k={k} outer={outer_tiles} work={work} route={} threshold={}",
                        if serial { "serial" } else { "pool" }, policy.minimum_work);
                }
            }
            if serial { return Executor::SingleThread; }
        }
    }
    #[cfg(not(feature = "multithread-mm"))]
    let _ = (m, n, outer_tiles, ops);
    executor
}

unsafe fn run_with_scratch_space_vec<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match matmul_executor(m, 1, m.div_ceil(ker.mr()), non_linear) {
            Executor::SingleThread => {
                for ia in 0..m.divceil(ker.mr()) {
                    scratch.run(ker, non_linear, ia, 0)?;
                }
                Ok(())
            }
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => pool.install(|| {
                (0..m.div_ceil(ker.mr()))
                    .into_par_iter()
                    .try_for_each(|ia| scratch.run(ker, non_linear, ia, 0))
            }),
        }
    }
}

unsafe fn run_with_scratch_space_col_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match matmul_executor(m, n, n.div_ceil(ker.nr()), non_linear) {
            Executor::SingleThread => {
                for ib in 0..n.divceil(ker.nr()) {
                    for ia in 0..m.divceil(ker.mr()) {
                        scratch.run(ker, non_linear, ia, ib)?;
                    }
                }
                Ok(())
            }
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => pool.install(|| {
                (0..n.div_ceil(ker.nr())).into_par_iter().try_for_each(|ib| {
                    for ia in 0..m.divceil(ker.mr()) {
                        scratch.run(ker, non_linear, ia, ib)?;
                    }
                    Ok(())
                })
            }),
        }
    }
}

unsafe fn run_with_scratch_space_row_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match matmul_executor(m, n, m.div_ceil(ker.mr()), non_linear) {
            Executor::SingleThread => {
                for ia in 0..m.divceil(ker.mr()) {
                    for ib in 0..n.divceil(ker.nr()) {
                        scratch.run(ker, non_linear, ia, ib)?;
                    }
                }
                Ok(())
            }
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => pool.install(|| {
                pool.install(|| {
                    (0..m.div_ceil(ker.mr())).into_par_iter().try_for_each(|ia| {
                        for ib in 0..n.divceil(ker.nr()) {
                            scratch.run(ker, non_linear, ia, ib)?;
                        }
                        Ok(())
                    })
                })
            }),
        }
    }
}
