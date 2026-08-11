//! A persistent spin-barrier worker pool for CPU decode.
//!
//! # Why this exists
//!
//! Decode is one activation against every weight matrix in the model, so
//! it is latency-bound: ~7 matvecs per layer, ~200 per token on a small
//! model. Ferrox opened one Rayon fork-join *per matvec*, and the cost of
//! that region — split the range, wake whichever workers had gone cold,
//! join — is paid ~200 times per token no matter how little arithmetic
//! each matvec carries.
//!
//! Measured on Host B with thread count as the only variable (both
//! engines back to back, no forced `-t` except as the diagnostic):
//!
//! | model | ferrox 1→6 threads | llama.cpp 1→6 threads |
//! |---|---|---|
//! | TinyLlama-1.1B Q8_0 | 1.40× | 1.99× |
//! | Mistral-7B Q4_K_M | 2.93× | 4.39× |
//!
//! Ferrox is *ahead* of llama.cpp at one thread on Mistral-7B (5.87 vs
//! 4.75 tok/s), so the kernels are not the deficit — the scheduling is.
//!
//! llama.cpp runs one pool for the whole graph and never tears a parallel
//! region down between ops: workers sit in `ggml_barrier` spinning on
//! `ggml_thread_cpu_relax` and only park after a long spin
//! (`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_barrier` /
//! `ggml_graph_compute_thread`). This module is the same shape: N-1
//! workers parked on a generation counter, woken by a store, pulling
//! tasks off one atomic index, acknowledging into a completion counter
//! the submitter spins on.
//!
//! # Contract
//!
//! [`CpuPool::dispatch`] runs `f(0..n_tasks)` exactly once each, on the
//! submitting thread plus the workers, and does not return until every
//! task has completed. `f` may therefore borrow the submitter's stack —
//! the barrier is what makes that sound.
//!
//! Nested dispatch (a task that itself dispatches) runs **serially** on
//! the calling thread; the pool is one flat region, not a work-stealing
//! tree.
//!
//! # Living next to Rayon
//!
//! Rayon still owns prefill (`apply_batch`, prefill attention) and the
//! two decode sites that deliberately overlap independent matrices:
//! `WeightMatrix::apply_three` (q/k/v) and `ferrox_moe::run_expert`
//! (gate/up), plus the MoE `outs.par_iter_mut()` in
//! `ferrox_models::decoder`. Those call `apply*` from Rayon workers, so
//! several threads can reach [`CpuPool::dispatch`] at once. They
//! serialize on a submit lock, which gives each of them a full-width
//! region back to back — no worse than running them in sequence, and no
//! oversubscription, because a submitter waiting on the lock is parked
//! rather than spinning.
//!
//! The overlap those sites were landed for is still real where it pays:
//! a projection too small for a parallel region (SmolLM2's 576×192 k/v)
//! takes the serial path inside `apply` and never touches the pool at
//! all, so it genuinely runs alongside the pooled region for `q`.
//!
//! What must **not** happen is a Rayon region and a pool region running
//! the same work at the same time on the same cores. It cannot: a thread
//! either dispatches to the pool (and blocks in the barrier) or runs the
//! Rayon fallback, never both.
//!
//! # Environment
//!
//! - `FERROX_CPU_POOL=0|false|off` — fall back to Rayon everywhere
//!   (the pre-pool behaviour, byte for byte). Default on.
//! - `FERROX_CPU_POOL_SPIN=<n>` — spin iterations a worker burns before
//!   parking on the condvar. Default 50000.

use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Type-erased job. Written by the submitter *before* the generation
/// counter is released, read by workers *after* they acquire it, so the
/// generation store/load pair is what orders these fields.
#[derive(Clone, Copy)]
struct Job {
    call: Option<unsafe fn(usize, usize)>,
    ctx: usize,
}

struct Shared {
    /// Bumped once per dispatch; the wake signal and the release fence.
    generation: AtomicU64,
    /// Next unclaimed task index (llama's `current_chunk`).
    next_task: AtomicUsize,
    n_tasks: AtomicUsize,
    /// Workers that finished the current generation.
    done: AtomicUsize,
    /// Set if any task unwound; re-raised on the submitter.
    panicked: AtomicBool,
    shutdown: AtomicBool,
    /// Workers currently blocked on `wake` (or about to be). Lets the
    /// submitter skip the mutex entirely on the hot, all-spinning path.
    parked: AtomicUsize,
    park_lock: Mutex<()>,
    wake: Condvar,
    spin: u64,
    job: std::cell::UnsafeCell<Job>,
}

// SAFETY: `job` is only written by the submitter while it holds the
// submit lock and before the `generation` release-store, and only read by
// workers after the matching acquire-load and before they acknowledge
// `done`. The previous generation's workers have all acknowledged before
// a new job is written, so writer and readers never overlap.
unsafe impl Sync for Shared {}

thread_local! {
    /// True while this thread is inside a pool region (worker for its
    /// whole life, submitter for the duration of one dispatch). Nested
    /// dispatch from such a thread runs serially instead of deadlocking
    /// on the submit lock.
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

struct ActiveGuard(bool);

impl ActiveGuard {
    fn enter() -> Self {
        ActiveGuard(IN_POOL.with(|c| c.replace(true)))
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        IN_POOL.with(|c| c.set(self.0));
    }
}

/// Monomorphized bridge from the erased `Job` back to the closure. The
/// pointer is valid for the whole dispatch because `dispatch` barriers
/// before returning.
unsafe fn trampoline<F: Fn(usize) + Sync>(ctx: usize, task: usize) {
    let f = unsafe { &*(ctx as *const F) };
    f(task);
}

impl Shared {
    /// Blocks until `generation` differs from `my_gen`, spinning first.
    fn wait_for_generation(&self, my_gen: u64) -> u64 {
        for _ in 0..self.spin {
            let g = self.generation.load(Ordering::Acquire);
            if g != my_gen {
                return g;
            }
            std::hint::spin_loop();
        }
        // Announce the park *before* re-reading the generation, and have
        // the submitter store the generation before reading `parked`.
        // Both sides are SeqCst, so at least one of them observes the
        // other and a wakeup cannot be lost.
        self.parked.fetch_add(1, Ordering::SeqCst);
        let g = {
            let mut guard = self.park_lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                let g = self.generation.load(Ordering::SeqCst);
                if g != my_gen {
                    break g;
                }
                guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
        };
        self.parked.fetch_sub(1, Ordering::SeqCst);
        g
    }

    /// Drains the task counter. Never unwinds: a panicking task is
    /// recorded and re-raised on the submitter, because a worker that
    /// unwound without acknowledging `done` would hang the barrier.
    fn run_tasks(&self) {
        // SAFETY: see the `unsafe impl Sync for Shared` justification —
        // this read is ordered after the submitter's release-store of
        // `generation` and before this thread's `done` acknowledgement.
        let job = unsafe { *self.job.get() };
        let Some(call) = job.call else {
            return;
        };
        let n = self.n_tasks.load(Ordering::Relaxed);
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| loop {
            let task = self.next_task.fetch_add(1, Ordering::Relaxed);
            if task >= n {
                break;
            }
            // SAFETY: `call`/`ctx` were published together by the
            // submitter, whose closure outlives the barrier below.
            unsafe { call(job.ctx, task) };
        }));
        if outcome.is_err() {
            self.panicked.store(true, Ordering::Release);
        }
    }
}

fn worker_loop(shared: Arc<Shared>, index: usize) {
    crate::threads::promote_worker_qos("pool", index);
    IN_POOL.with(|c| c.set(true));
    let mut my_gen = 0u64;
    loop {
        let g = shared.wait_for_generation(my_gen);
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        if g == my_gen {
            continue;
        }
        my_gen = g;
        shared.run_tasks();
        shared.done.fetch_add(1, Ordering::Release);
    }
}

/// A persistent set of worker threads plus the submitting thread.
pub struct CpuPool {
    shared: Arc<Shared>,
    /// Worker threads, i.e. `width - 1`.
    n_workers: usize,
    submit: Mutex<()>,
}

impl CpuPool {
    fn new(width: usize, spin: u64) -> Self {
        let n_workers = width.saturating_sub(1);
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            next_task: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            park_lock: Mutex::new(()),
            wake: Condvar::new(),
            spin,
            job: std::cell::UnsafeCell::new(Job { call: None, ctx: 0 }),
        });
        for i in 0..n_workers {
            let s = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("ferrox-pool-{i}"))
                .spawn(move || worker_loop(s, i));
            if spawned.is_err() {
                // Fewer workers than requested is a throughput question,
                // not a correctness one — but the barrier counts workers,
                // so the pool must know how many actually started.
                return CpuPool {
                    shared,
                    n_workers: i,
                    submit: Mutex::new(()),
                };
            }
        }
        CpuPool {
            shared,
            n_workers,
            submit: Mutex::new(()),
        }
    }

    /// Total participants: workers plus the submitting thread.
    pub fn width(&self) -> usize {
        self.n_workers + 1
    }

    /// Runs `f(0)..f(n_tasks - 1)` exactly once each and returns only
    /// after all of them have finished.
    ///
    /// Runs serially when there are no workers, when there is a single
    /// task, or when the caller is already inside a pool region.
    pub fn dispatch<F>(&self, n_tasks: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        if n_tasks == 0 {
            return;
        }
        if n_tasks == 1 || self.n_workers == 0 || IN_POOL.with(|c| c.get()) {
            for task in 0..n_tasks {
                f(task);
            }
            return;
        }
        let _submit = self.submit.lock().unwrap_or_else(|e| e.into_inner());
        let s = &self.shared;
        s.n_tasks.store(n_tasks, Ordering::Relaxed);
        s.next_task.store(0, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        // SAFETY: no worker reads `job` until the release-store below,
        // and every worker of the previous generation has acknowledged.
        unsafe {
            *s.job.get() = Job {
                call: Some(trampoline::<F>),
                ctx: &f as *const F as usize,
            };
        }
        s.generation.fetch_add(1, Ordering::SeqCst);
        if s.parked.load(Ordering::SeqCst) > 0 {
            let _g = s.park_lock.lock().unwrap_or_else(|e| e.into_inner());
            s.wake.notify_all();
        }

        {
            let _active = ActiveGuard::enter();
            s.run_tasks();
        }

        let mut spins = 0u32;
        while s.done.load(Ordering::Acquire) < self.n_workers {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(8192) {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: every worker has acknowledged, so nobody reads `job`.
        unsafe {
            *s.job.get() = Job { call: None, ctx: 0 };
        }
        if s.panicked.swap(false, Ordering::AcqRel) {
            panic!("ferrox cpu pool: a parallel task panicked");
        }
    }
}

fn pool_spin() -> u64 {
    std::env::var("FERROX_CPU_POOL_SPIN")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(50_000)
}

/// Whether the persistent pool is used at all. `FERROX_CPU_POOL=0`
/// restores the Rayon fork-join path exactly. Read once.
pub fn enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("FERROX_CPU_POOL").ok().as_deref(),
            Some("0") | Some("false") | Some("off")
        )
    })
}

static POOL: OnceLock<CpuPool> = OnceLock::new();

/// The process-wide pool, built on first use. `None` when disabled.
pub fn pool() -> Option<&'static CpuPool> {
    if !enabled() {
        return None;
    }
    Some(POOL.get_or_init(|| CpuPool::new(crate::threads::resolve_cpu_threads(), pool_spin())))
}

/// Builds the pool eagerly (no-op when disabled or already built).
pub fn init() {
    let _ = pool();
}

/// The pool to dispatch a *new* parallel region onto: `None` when
/// disabled, single-threaded, or when the caller is already inside one.
pub fn pool_for_dispatch() -> Option<&'static CpuPool> {
    if IN_POOL.with(|c| c.get()) {
        return None;
    }
    pool().filter(|p| p.n_workers > 0)
}

/// Participants in a pool region, or 0 when the pool is not in use.
pub fn width() -> usize {
    pool().map(|p| p.width()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn test_pool() -> &'static CpuPool {
        static P: OnceLock<CpuPool> = OnceLock::new();
        P.get_or_init(|| CpuPool::new(4, 2_000))
    }

    #[test]
    fn every_task_runs_exactly_once() {
        let n = 1_000usize;
        let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        test_pool().dispatch(n, |t| {
            counts[t].fetch_add(1, Ordering::Relaxed);
        });
        for (t, c) in counts.iter().enumerate() {
            assert_eq!(
                c.load(Ordering::Relaxed),
                1,
                "task {t} ran the wrong number of times"
            );
        }
    }

    #[test]
    fn repeated_dispatches_reuse_the_same_workers() {
        // Exercises the generation handshake many times over, which is
        // where a lost wakeup or a stale `done` count would show up.
        for round in 0..200u32 {
            let n = 37usize;
            let sum = AtomicU32::new(0);
            test_pool().dispatch(n, |t| {
                sum.fetch_add(t as u32 + round, Ordering::Relaxed);
            });
            let expect = (0..n as u32).map(|t| t + round).sum::<u32>();
            assert_eq!(sum.load(Ordering::Relaxed), expect, "round {round}");
        }
    }

    #[test]
    fn dispatch_borrows_the_submitters_stack() {
        let data: Vec<usize> = (0..512).collect();
        let out: Vec<AtomicU32> = (0..512).map(|_| AtomicU32::new(0)).collect();
        test_pool().dispatch(512, |t| {
            out[t].store(data[t] as u32 * 3, Ordering::Relaxed);
        });
        for (t, slot) in out.iter().enumerate() {
            assert_eq!(slot.load(Ordering::Relaxed), t as u32 * 3);
        }
    }

    #[test]
    fn workers_park_and_are_woken_again() {
        // Spin budget is small in tests, so a sleep longer than it
        // guarantees the workers are on the condvar, not spinning.
        let pool = test_pool();
        pool.dispatch(8, |_| {});
        std::thread::sleep(std::time::Duration::from_millis(50));
        let sum = AtomicU32::new(0);
        pool.dispatch(64, |t| {
            sum.fetch_add(t as u32, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..64u32).sum::<u32>());
    }

    #[test]
    fn nested_dispatch_runs_serially_instead_of_deadlocking() {
        let inner_total = AtomicU32::new(0);
        test_pool().dispatch(16, |_| {
            test_pool().dispatch(4, |t| {
                inner_total.fetch_add(t as u32, Ordering::Relaxed);
            });
        });
        assert_eq!(inner_total.load(Ordering::Relaxed), 16 * (1 + 2 + 3));
    }

    #[test]
    fn concurrent_submitters_are_serialized_not_interleaved() {
        let pool = test_pool();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..25 {
                        let sum = AtomicU32::new(0);
                        pool.dispatch(64, |t| {
                            sum.fetch_add(t as u32, Ordering::Relaxed);
                        });
                        assert_eq!(sum.load(Ordering::Relaxed), (0..64u32).sum::<u32>());
                    }
                });
            }
        });
    }

    #[test]
    fn a_panicking_task_surfaces_on_the_submitter_and_leaves_the_pool_usable() {
        let pool = test_pool();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.dispatch(64, |t| {
                if t == 63 {
                    panic!("boom");
                }
            });
        }));
        std::panic::set_hook(hook);
        assert!(caught.is_err(), "a panicking task must not be swallowed");

        let sum = AtomicU32::new(0);
        pool.dispatch(32, |t| {
            sum.fetch_add(t as u32, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..32u32).sum::<u32>());
    }

    #[test]
    fn a_single_task_or_no_tasks_is_a_no_op_region() {
        let hits = AtomicU32::new(0);
        test_pool().dispatch(0, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 0);
        test_pool().dispatch(1, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 1);
    }
}
