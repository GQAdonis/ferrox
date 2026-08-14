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
//! [`CpuPool::dispatch`] either runs `f(0..n_tasks)` exactly once each —
//! on the submitting thread plus the workers, returning only after every
//! task has completed, so `f` may borrow the submitter's stack — **or**
//! runs nothing and returns `false` because another thread owns the
//! pool. It never waits for another submitter.
//!
//! Nested dispatch (a task that itself dispatches) runs **serially** on
//! the calling thread; the pool is one flat region, not a work-stealing
//! tree.
//!
//! # Living next to Rayon
//!
//! Rayon still owns prefill (`apply_batch`, prefill attention) and the
//! decode sites that overlap independent matrices:
//! `WeightMatrix::apply_three` (q/k/v) and `ferrox_moe::run_expert`
//! (gate/up), plus the MoE `outs.par_iter_mut()` in
//! `ferrox_models::decoder`. Those call `apply*` from Rayon workers, so
//! several threads can reach [`CpuPool::dispatch`] at once.
//!
//! **The first version of this module had them block on a submit mutex,
//! and that was a deadlock.** A pool task that reaches Rayon parks its
//! pool worker on a Rayon latch; the Rayon worker that would run that
//! job is parked on the submit mutex; the submitter holding the mutex is
//! spinning for the pool worker's acknowledgement. Reproduced under
//! `sample`, with all three stacks, in
//! `docs/plans/llama-cpp-parity-push.md`.
//!
//! So the lock is `try_lock`, and losing it is not an error: the caller
//! runs the Rayon path it would have run anyway. One decode thread —
//! the case this pool exists for — always wins it uncontended. Every
//! thread is therefore always in exactly one runtime that can make
//! progress on its own, which is the property the previous design
//! asserted and did not have.
//!
//! The overlap those sites were landed for is still real where it pays:
//! a projection too small for a parallel region (SmolLM2's 576×192 k/v)
//! takes the serial path inside `apply` and never touches the pool at
//! all, so it genuinely runs alongside the pooled region for `q`.
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
    /// after all of them have finished — **or** returns `false` without
    /// running anything, when another thread already owns the pool.
    ///
    /// Runs serially (returning `true`) when there are no workers, when
    /// there is a single task, or when the caller is already inside a
    /// pool region.
    ///
    /// # Why this never blocks
    ///
    /// The first version of this module took the submit lock with
    /// `lock()`, and that is what got it rejected: a pool task that
    /// reaches Rayon parks its pool worker on a Rayon latch, the Rayon
    /// worker that would have run that job is parked on *this* mutex,
    /// and the submitter holding the mutex is spinning for the pool
    /// worker's ack. Reproduced and sampled — see
    /// `docs/plans/llama-cpp-parity-push.md`.
    ///
    /// `try_lock` deletes the `Rayon worker → submit lock` edge, which
    /// is the only edge in that cycle that needs another runtime to make
    /// progress. A caller that loses the race runs the Rayon path
    /// instead, which is exactly what it did before this module existed.
    /// Serializing on the pool was never worth a deadlock class: the
    /// pool exists to make *one* decode thread's regions cheap, not to
    /// arbitrate between several.
    #[must_use = "a false return means the caller must run the Rayon path itself"]
    pub fn dispatch<F>(&self, n_tasks: usize, f: F) -> bool
    where
        F: Fn(usize) + Sync,
    {
        if n_tasks == 0 {
            return true;
        }
        if n_tasks == 1 || self.n_workers == 0 || IN_POOL.with(|c| c.get()) {
            for task in 0..n_tasks {
                f(task);
            }
            return true;
        }
        let _submit = match self.submit.try_lock() {
            Ok(guard) => guard,
            // The guard protects no data — it is a `Mutex<()>` used only
            // as a "one submitter at a time" token — and a task panic is
            // already re-raised on its own submitter below. Treating
            // poison as permanent contention would silently retire the
            // pool for the rest of the process after the first panic.
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
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
        true
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
    // Inside a Rayon worker the caller is *already* in a parallel
    // region — decode fans out over q/k/v (`apply_three`), gate/up
    // (`run_expert`) and routed experts. Opening a pool region from
    // there puts pool workers and Rayon workers on the same cores at
    // the same time; measured on OLMoE decode, that costs 21% (55.67
    // against 70.09 tok/s with the pool off). Rayon nesting inside an
    // existing region is cheap — it is the *top-level* fork-join this
    // pool exists to replace — so the rule is one runtime per thread:
    // the decode thread pools, Rayon workers stay on Rayon.
    if rayon::current_thread_index().is_some() {
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

    /// Tests share one pool and Cargo runs them concurrently, so a
    /// `dispatch` can legitimately lose the `try_lock`. Retry until it
    /// runs: the property under test is what the region *computes*, not
    /// who won the race (that is
    /// `concurrent_submitters_never_block_each_other`'s job).
    fn dispatch_eventually<F>(pool: &CpuPool, n: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        for _ in 0..10_000 {
            if pool.dispatch(n, &f) {
                return;
            }
            std::thread::yield_now();
        }
        panic!("pool stayed contended for 10k attempts");
    }

    fn test_pool() -> &'static CpuPool {
        static P: OnceLock<CpuPool> = OnceLock::new();
        P.get_or_init(|| CpuPool::new(4, 2_000))
    }

    #[test]
    fn every_task_runs_exactly_once() {
        let n = 1_000usize;
        let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        dispatch_eventually(test_pool(), n, |t| {
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
            dispatch_eventually(test_pool(), n, |t| {
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
        dispatch_eventually(test_pool(), 512, |t| {
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
        dispatch_eventually(pool, 8, |_| {});
        std::thread::sleep(std::time::Duration::from_millis(50));
        let sum = AtomicU32::new(0);
        dispatch_eventually(pool, 64, |t| {
            sum.fetch_add(t as u32, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..64u32).sum::<u32>());
    }

    #[test]
    fn nested_dispatch_runs_serially_instead_of_deadlocking() {
        let inner_total = AtomicU32::new(0);
        dispatch_eventually(test_pool(), 16, |_| {
            let _ = test_pool().dispatch(4, |t| {
                inner_total.fetch_add(t as u32, Ordering::Relaxed);
            });
        });
        assert_eq!(inner_total.load(Ordering::Relaxed), 16 * (1 + 2 + 3));
    }

    #[test]
    fn concurrent_submitters_never_block_each_other() {
        // Two properties at once: a submitter that wins the pool
        // computes the whole region, and a submitter that loses says so
        // instead of waiting. Both outcomes are correct; what would be
        // incorrect is a partially-run region, or a caller parked behind
        // another runtime (the rejected design's deadlock).
        // A private pool: the shared one is being hammered by the other
        // tests' retry loops, and "did anyone win?" is only meaningful
        // against a known set of contenders.
        let pool = CpuPool::new(4, 2_000);
        let pool = &pool;
        let won = AtomicU32::new(0);
        let lost = AtomicU32::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..25 {
                        let sum = AtomicU32::new(0);
                        if pool.dispatch(64, |t| {
                            sum.fetch_add(t as u32, Ordering::Relaxed);
                        }) {
                            assert_eq!(sum.load(Ordering::Relaxed), (0..64u32).sum::<u32>());
                            won.fetch_add(1, Ordering::Relaxed);
                        } else {
                            assert_eq!(
                                sum.load(Ordering::Relaxed),
                                0,
                                "a refused dispatch must not run any task"
                            );
                            lost.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        assert_eq!(
            won.load(Ordering::Relaxed) + lost.load(Ordering::Relaxed),
            100
        );
        assert!(won.load(Ordering::Relaxed) > 0, "nobody ever got the pool");
    }

    #[test]
    fn a_panicking_task_surfaces_on_the_submitter_and_leaves_the_pool_usable() {
        let pool = test_pool();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.dispatch(64, |t| {
                if t == 63 {
                    panic!("boom");
                }
            });
        }));
        std::panic::set_hook(hook);
        assert!(caught.is_err(), "a panicking task must not be swallowed");

        let sum = AtomicU32::new(0);
        dispatch_eventually(pool, 32, |t| {
            sum.fetch_add(t as u32, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..32u32).sum::<u32>());
    }

    /// The regression test for the deadlock that got the first version
    /// of this module rejected.
    ///
    /// Rayon workers submit pool regions while a pool task itself calls
    /// back into Rayon. With a blocking submit lock this hangs forever:
    /// pool worker parked on a Rayon latch -> Rayon worker parked on the
    /// submit mutex -> submitter holding it, spinning for the pool
    /// worker's ack. With `try_lock` the losing submitters run Rayon
    /// instead, so every edge of that cycle still has somewhere to go.
    ///
    /// It is a *hang* test: it either finishes or the suite times out,
    /// so it is deliberately small enough to be fast and repeated enough
    /// to be reliable.
    #[test]
    fn two_runtimes_cannot_deadlock_each_other() {
        use rayon::prelude::*;
        let pool = test_pool();
        for _ in 0..20 {
            (0..16).into_par_iter().for_each(|outer| {
                let acc = AtomicU32::new(0);
                let ran = pool.dispatch(64, |t| {
                    // A pool task reaching into Rayon: the edge that
                    // parks a pool worker on a Rayon latch.
                    let s: u32 = (0..64u32).into_par_iter().sum();
                    acc.fetch_add(s + t as u32 + outer as u32, Ordering::Relaxed);
                });
                if !ran {
                    // Lost the pool to another submitter: that is the
                    // designed outcome, not a failure.
                    (0..64).into_par_iter().for_each(|t| {
                        let s: u32 = (0..64u32).into_par_iter().sum();
                        acc.fetch_add(s + t as u32 + outer as u32, Ordering::Relaxed);
                    });
                }
                assert!(acc.load(Ordering::Relaxed) > 0);
            });
        }
    }

    #[test]
    fn a_single_task_or_no_tasks_is_a_no_op_region() {
        let hits = AtomicU32::new(0);
        let _ = test_pool().dispatch(0, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 0);
        let _ = test_pool().dispatch(1, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 1);
    }
}
