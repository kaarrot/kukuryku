use std::cell::RefCell;
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};

#[cfg(feature = "multithread-mm")]
use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Debug, Clone, Default)]
pub enum Executor {
    #[default]
    SingleThread,
    #[cfg(feature = "multithread-mm")]
    MultiThread(Arc<ThreadPool>),
}

impl Executor {
    #[cfg(feature = "multithread-mm")]
    pub fn multithread(n: usize) -> Executor {
        Executor::multithread_with_name(n, "tract-default")
    }

    #[cfg(feature = "multithread-mm")]
    pub fn multithread_with_name(n: usize, name: &str) -> Executor {
        let name = name.to_string();
        let pool = ThreadPoolBuilder::new()
            .thread_name(move |n| format!("{name}-{n}"))
            .num_threads(n)
            .build()
            .unwrap();
        Executor::MultiThread(Arc::new(pool))
    }
}

static DEFAULT_EXECUTOR: Mutex<Executor> = Mutex::new(Executor::SingleThread);

thread_local! {
    static TLS_EXECUTOR_OVERRIDE: RefCell<Option<Executor>> = Default::default();
}

pub fn current_tract_executor() -> Executor {
    if let Some(over_ride) = TLS_EXECUTOR_OVERRIDE.with_borrow(|tls| tls.clone()) {
        over_ride
    } else {
        DEFAULT_EXECUTOR.lock().unwrap().clone()
    }
}

pub fn set_default_executor(executor: Executor) {
    *DEFAULT_EXECUTOR.lock().unwrap() = executor;
}

pub fn multithread_tract_scope<R, F: FnOnce() -> R>(pool: Executor, f: F) -> R {
    let previous = TLS_EXECUTOR_OVERRIDE.replace(Some(pool));
    let result = f();
    TLS_EXECUTOR_OVERRIDE.set(previous);
    result
}

/// Map `0..n` to a `Vec<R>` on the current tract executor: parallel across the
/// pool when one is set (multithread-mm), sequential otherwise. Lets tract-core
/// ops (STFT, Im2col, ...) fan out compute-heavy loops onto the same pool the
/// matmul kernels use, without depending on rayon directly.
pub fn par_map<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Send + Sync,
{
    match current_tract_executor() {
        Executor::SingleThread => (0..n).map(f).collect(),
        #[cfg(feature = "multithread-mm")]
        Executor::MultiThread(pool) => pool.install(|| {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(f).collect()
        }),
    }
}

/// Apply `f(chunk_index, chunk)` over disjoint mutable `chunk`-sized windows of
/// `data`, parallel on the current tract executor (sequential otherwise). The
/// final chunk may be shorter than `chunk`. Used to parallelize elementwise ops.
pub fn par_chunks_mut<T, F>(data: &mut [T], chunk: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    debug_assert!(chunk > 0);
    match current_tract_executor() {
        Executor::SingleThread => {
            data.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c))
        }
        #[cfg(feature = "multithread-mm")]
        Executor::MultiThread(pool) => pool.install(|| {
            use rayon::prelude::*;
            data.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
        }),
    }
}
