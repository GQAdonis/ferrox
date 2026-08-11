//! CPU worker-pool policy, shared by `ferrox` (CLI) and `ferrox-server`.
//!
//! Two things this module exists to control, both of which were measured
//! to matter far more than any kernel change on Apple Silicon:
//!
//! 1. **Thread count.** `available_parallelism()` on an M2 Pro reports 10
//!    (6 performance + 4 efficiency cores). Splitting a decode GEMV
//!    across all 10 makes every fork-join wait on the slowest E-core
//!    slice. llama.cpp defaults to `hw.perflevel0.physicalcpu` for
//!    exactly this reason (`common_cpu_get_num_math`), and collapses when
//!    forced above it -- 346 -> 176 tok/s on SmolLM2-135M Q8_0 going from
//!    `-t 4` to `-t 10`. So default to the performance-core count, not
//!    the logical-core count.
//!
//! 2. **Thread QoS.** macOS schedules threads onto E-cores based on their
//!    Quality-of-Service class, and QoS is *inherited* from whichever
//!    thread spawned them. Rayon builds its global pool lazily, on first
//!    use -- which inside `ferrox-server` is a Tokio `spawn_blocking`
//!    task, not the main thread. If that blocking thread carries a
//!    demoted QoS, every rayon worker inherits it and the whole matvec
//!    runs on efficiency cores. [`init_cpu_pool`] pins the workers to
//!    `USER_INTERACTIVE` explicitly so the pool's placement does not
//!    depend on who happened to touch rayon first.
//!
//! 3. **Dedicated GEMV pool.** Row-parallel matvec runs on a
//!    crate-owned [`rayon::ThreadPool`], not rayon's global pool, so
//!    library consumers and Tokio blocking threads do not fight over
//!    thread count or scheduling. See [`for_each_row`].

use rayon::prelude::*;
use std::sync::OnceLock;

/// macOS QoS classes (`sys/qos.h`). Only the ones we name are listed.
#[cfg(target_os = "macos")]
mod qos {
    pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    pub const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    pub const QOS_CLASS_DEFAULT: u32 = 0x15;
    pub const QOS_CLASS_UTILITY: u32 = 0x11;
    pub const QOS_CLASS_BACKGROUND: u32 = 0x09;

    extern "C" {
        pub fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
        pub fn qos_class_self() -> u32;
    }

    pub fn name(class: u32) -> &'static str {
        match class {
            QOS_CLASS_USER_INTERACTIVE => "user-interactive",
            QOS_CLASS_USER_INITIATED => "user-initiated",
            QOS_CLASS_DEFAULT => "default",
            QOS_CLASS_UTILITY => "utility",
            QOS_CLASS_BACKGROUND => "background",
            _ => "unspecified",
        }
    }
}

/// The calling thread's macOS QoS class, as a human-readable name.
/// `None` off macOS, where the concept does not exist.
pub fn current_qos_name() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        Some(qos::name(unsafe { qos::qos_class_self() }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Number of *performance* cores, which is the useful width for a decode
/// GEMV. On macOS this is `hw.perflevel0.physicalcpu` (llama.cpp reads
/// the same key). Elsewhere, and if the query fails, falls back to
/// `available_parallelism`.
pub fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_usize("hw.perflevel0.physicalcpu") {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    use std::ffi::CString;
    extern "C" {
        fn sysctlbyname(
            name: *const std::os::raw::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> std::os::raw::c_int;
    }
    let key = CString::new(name).ok()?;
    let mut out: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `key` is NUL-terminated, and `out`/`len` describe a live
    // i32 of exactly `len` bytes, which is what these keys return.
    let rc = unsafe {
        sysctlbyname(
            key.as_ptr(),
            &mut out as *mut i32 as *mut std::ffi::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && out > 0 {
        Some(out as usize)
    } else {
        None
    }
}

/// How many rayon workers to run: `FERROX_CPU_THREADS`, else
/// `RAYON_NUM_THREADS`, else [`perf_core_count`].
pub fn resolve_cpu_threads() -> usize {
    for key in ["FERROX_CPU_THREADS", "RAYON_NUM_THREADS"] {
        if let Ok(v) = std::env::var(key) {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    perf_core_count()
}

/// GEMV pool width: `FERROX_GEMV_THREADS`, else [`resolve_cpu_threads`].
pub fn resolve_gemv_threads() -> usize {
    if let Ok(v) = std::env::var("FERROX_GEMV_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    resolve_cpu_threads()
}

static GEMV_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Pins the *calling* worker thread to `USER_INTERACTIVE` on macOS, so a
/// pool built lazily from a demoted thread (Tokio `spawn_blocking`) does
/// not inherit that demotion and run the whole matvec on E-cores. `kind`
/// only names the pool in the `FERROX_QOS_LOG` trace.
pub(crate) fn promote_worker_qos(kind: &str, idx: usize) {
    #[cfg(target_os = "macos")]
    {
        let log = std::env::var_os("FERROX_QOS_LOG").is_some();
        let before = unsafe { qos::qos_class_self() };
        // SAFETY: sets only the calling thread's QoS class.
        let rc = unsafe { qos::pthread_set_qos_class_self_np(qos::QOS_CLASS_USER_INTERACTIVE, 0) };
        if log {
            eprintln!(
                "ferrox: {kind} worker {idx} qos {} -> {} (rc={rc})",
                qos::name(before),
                qos::name(unsafe { qos::qos_class_self() }),
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (kind, idx);
    }
}

fn gemv_pool_qos_start_handler(idx: usize) {
    promote_worker_qos("gemv", idx);
}

fn gemv_pool() -> Option<&'static rayon::ThreadPool> {
    GEMV_POOL
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(resolve_gemv_threads())
                .thread_name(|i| format!("ferrox-gemv-{i}"))
                .start_handler(gemv_pool_qos_start_handler)
                .build()
                .ok()
        })
        .as_ref()
}

/// Eagerly build the dedicated GEMV pool (no-op if already built).
pub fn init_gemv_pool() {
    let _ = gemv_pool();
}

/// Active GEMV pool width, or `1` when the pool is unavailable.
pub fn gemv_num_threads() -> usize {
    match gemv_pool() {
        Some(pool) => pool.current_num_threads(),
        None => 1,
    }
}

/// Element-ops below which a matvec is cheaper run serially than split
/// across a parallel region at all.
///
/// This is the **small-model thread heuristic**: nothing else in the
/// engine caps pool width by problem size, and both ferrox and llama.cpp
/// get *slower* with threads on SmolLM2-135M (ferrox 148.79 tok/s at
/// `-t 1` against 105.46 at `-t 6`). At hidden 576 a projection is
/// 576×576 ≈ 331k element-ops, roughly 30 µs of arithmetic — the same
/// order as one Rayon fork-join, so splitting it loses.
///
/// The default (256k) is deliberately left where it was: the persistent
/// pool ([`crate::cpu_pool`]) attacks the *cost* of the region rather
/// than the decision to open one, and moving both at once would make the
/// A/B unreadable. `FERROX_MIN_PARALLEL_OPS` exists so the threshold can
/// be swept without a rebuild — the right value is a measurement, and
/// this crate deliberately does not guess one.
pub fn min_parallel_ops() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FERROX_MIN_PARALLEL_OPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(256_000)
    })
}

/// Prefer serial when fork-join overhead exceeds the matvec work.
/// See [`min_parallel_ops`].
pub fn should_parallelize(n_rows: usize, n_cols: usize) -> bool {
    n_rows > 1 && n_rows.saturating_mul(n_cols) >= min_parallel_ops()
}

/// Participants in a parallel region: the persistent pool's width when
/// it is in use, otherwise Rayon's. Task sizing must agree with whoever
/// actually runs the tasks.
pub fn effective_num_threads() -> usize {
    let pooled = crate::cpu_pool::width();
    if pooled > 0 {
        pooled
    } else {
        rayon::current_num_threads().max(1)
    }
}

/// A `*mut f32` handed to pool tasks that own disjoint sub-slices.
struct RowsPtr(*mut f32);
// SAFETY: every task derives a sub-slice from a disjoint chunk range, so
// no two tasks ever alias. `par_chunks_indexed` is what guarantees that.
unsafe impl Send for RowsPtr {}
unsafe impl Sync for RowsPtr {}

impl RowsPtr {
    /// # Safety
    /// `off` must be in bounds of the original slice.
    #[inline]
    unsafe fn at(&self, off: usize) -> *mut f32 {
        unsafe { self.0.add(off) }
    }
}

/// `out.par_chunks_mut(chunk_len).with_min_len(m).enumerate().for_each(body)`,
/// but routed through the persistent [`crate::cpu_pool`] when it is
/// available.
///
/// This is the single seam between the decode matvec kernels and the
/// scheduler. The task *shape* is identical to what Rayon was handed
/// before — `ceil(n_chunks / min_chunks_per_task)` tasks, chunk index
/// passed through unchanged — so the only variable that moves is which
/// mechanism opens and closes the region.
///
/// `body` is called exactly once per chunk, with the chunk's index and
/// its slice (the final chunk is short when `chunk_len` does not divide
/// `out.len()`, matching `chunks_mut`).
pub fn par_chunks_indexed<F>(out: &mut [f32], chunk_len: usize, min_chunks_per_task: usize, body: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    if chunk_len == 0 || out.is_empty() {
        return;
    }
    let total = out.len();
    let n_chunks = total.div_ceil(chunk_len);
    let per_task = min_chunks_per_task.max(1);
    if n_chunks == 1 {
        body(0, out);
        return;
    }
    if let Some(pool) = crate::cpu_pool::pool_for_dispatch() {
        let n_tasks = n_chunks.div_ceil(per_task);
        if n_tasks > 1 {
            let base = RowsPtr(out.as_mut_ptr());
            let body = &body;
            pool.dispatch(n_tasks, move |task| {
                let c0 = task * per_task;
                let c1 = ((task + 1) * per_task).min(n_chunks);
                for c in c0..c1 {
                    let off = c * chunk_len;
                    let len = chunk_len.min(total - off);
                    // SAFETY: chunk ranges are disjoint across tasks and
                    // `base` stays valid for the whole dispatch, which
                    // barriers before `out`'s borrow ends.
                    let slice = unsafe { std::slice::from_raw_parts_mut(base.at(off), len) };
                    body(c, slice);
                }
            });
            return;
        }
    }
    out.par_chunks_mut(chunk_len)
        .with_min_len(per_task)
        .enumerate()
        .for_each(|(i, chunk)| body(i, chunk));
}

/// One output row per slot; parallel when [`should_parallelize`] and the
/// GEMV pool has more than one worker. Rows are never split.
pub fn for_each_row<F>(output: &mut [f32], n_rows: usize, n_cols: usize, row_fn: F)
where
    F: Fn(usize, &mut f32) + Send + Sync,
{
    let n = n_rows.min(output.len());
    if !should_parallelize(n, n_cols) {
        for (row, out) in output.iter_mut().enumerate().take(n) {
            row_fn(row, out);
        }
        return;
    }
    // Default: global rayon (same pool as act-quant). Dedicated pool via
    // FERROX_GEMV_DEDICATED=1 once callers stop using global rayon mid-matvec.
    let use_dedicated = matches!(
        std::env::var("FERROX_GEMV_DEDICATED").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    let rows = &mut output[..n];
    if use_dedicated {
        if let Some(pool) = gemv_pool() {
            if pool.current_num_threads() > 1 {
                pool.install(move || {
                    rows.par_iter_mut()
                        .enumerate()
                        .for_each(|(row, out)| row_fn(row, out));
                });
                return;
            }
        }
    }
    rows.par_iter_mut()
        .enumerate()
        .for_each(|(row, out)| row_fn(row, out));
}

/// Chunk-parallel sibling of [`for_each_row`]; chunks are never split.
pub fn for_each_chunk_init<S, I, F>(
    output: &mut [f32],
    chunk_len: usize,
    work_per_chunk: usize,
    init: I,
    f: F,
) where
    I: Fn() -> S + Send + Sync,
    S: Send,
    F: Fn(&mut S, usize, &mut [f32]) + Send + Sync,
{
    if chunk_len == 0 {
        return;
    }
    let n_chunks = output.len() / chunk_len;
    if !should_parallelize(n_chunks, work_per_chunk) {
        let mut state = init();
        for (i, chunk) in output[..n_chunks * chunk_len]
            .chunks_mut(chunk_len)
            .enumerate()
        {
            f(&mut state, i, chunk);
        }
        return;
    }
    let use_dedicated = matches!(
        std::env::var("FERROX_GEMV_DEDICATED").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    let chunks = &mut output[..n_chunks * chunk_len];
    let init = &init;
    let f = &f;
    if use_dedicated {
        if let Some(pool) = gemv_pool() {
            if pool.current_num_threads() > 1 {
                pool.install(move || {
                    chunks
                        .par_chunks_mut(chunk_len)
                        .enumerate()
                        .for_each_init(init, |state, (i, c)| f(state, i, c));
                });
                return;
            }
        }
    }
    chunks
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each_init(init, |state, (i, c)| f(state, i, c));
}

/// Builds the global rayon pool with an explicit width and an explicit
/// QoS, so neither depends on which thread first touched rayon. Safe to
/// call more than once and from either binary; a pool that already
/// exists is left alone.
///
/// Returns the thread count the pool was built with, or `None` if the
/// global pool already existed.
/// Builds the dedicated GEMV pool (and, for legacy callers that still
/// touch `rayon::prelude` on the global pool, a matching global pool).
/// Prefer [`for_each_row`] / [`for_each_chunk_init`] so matvecs stay on
/// the dedicated pool — mixing both pools oversubscribes P-cores.
///
/// Returns the thread count the **global** pool was built with, or
/// `None` if the global pool already existed.
pub fn init_cpu_pool() -> Option<usize> {
    // The persistent decode pool ([`crate::cpu_pool`]) is built here,
    // from the process's own startup thread, so its workers inherit an
    // undemoted QoS and their spawn cost is not charged to the first
    // token. Unlike the GEMV pool below, an idle one costs nothing: its
    // workers park on a condvar after a short spin, so a Metal run or a
    // prefill-only run is not paying for a second set of live threads.
    crate::cpu_pool::init();
    // Do not eagerly build the dedicated GEMV pool here — a second idle
    // rayon pool of P-core width was measured to regress CPU pp512 on
    // Host B (~40 → ~14 tok/s). Callers opt in via `for_each_*` (lazy) or
    // `init_gemv_pool()` + `FERROX_GEMV_DEDICATED=1`.
    let threads = resolve_cpu_threads();
    let log = std::env::var_os("FERROX_QOS_LOG").is_some();
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |idx| {
            #[cfg(target_os = "macos")]
            {
                let before = unsafe { qos::qos_class_self() };
                // SAFETY: sets only the calling thread's QoS class.
                let rc = unsafe {
                    qos::pthread_set_qos_class_self_np(qos::QOS_CLASS_USER_INTERACTIVE, 0)
                };
                if log {
                    eprintln!(
                        "ferrox: rayon worker {idx} qos {} -> {} (rc={rc})",
                        qos::name(before),
                        qos::name(unsafe { qos::qos_class_self() }),
                    );
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (idx, log);
            }
        })
        .build_global()
        .is_ok();
    if built {
        Some(threads)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_core_count_is_at_least_one_and_no_more_than_logical_cores() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let perf = perf_core_count();
        assert!(perf >= 1, "perf core count must be positive, got {perf}");
        assert!(
            perf <= logical,
            "perf cores ({perf}) cannot exceed logical cores ({logical})"
        );
    }

    #[test]
    fn resolved_thread_count_falls_back_to_perf_cores_without_env_overrides() {
        // `resolve_cpu_threads` reads process-global env, and tests share
        // a process, so assert the fallback only when nothing is set.
        if std::env::var_os("FERROX_CPU_THREADS").is_none()
            && std::env::var_os("RAYON_NUM_THREADS").is_none()
        {
            assert_eq!(resolve_cpu_threads(), perf_core_count());
        }
    }

    #[test]
    fn current_qos_name_is_reported_on_macos_and_absent_elsewhere() {
        let qos = current_qos_name();
        #[cfg(target_os = "macos")]
        assert!(qos.is_some(), "macOS must report a QoS class");
        #[cfg(not(target_os = "macos"))]
        assert!(qos.is_none(), "QoS is a macOS-only concept");
    }

    /// The whole decode matvec path funnels through
    /// [`par_chunks_indexed`], so "every chunk exactly once, with the
    /// index it would have had serially" is the property that keeps
    /// output bit-identical whichever scheduler runs it.
    #[test]
    fn par_chunks_indexed_matches_the_serial_chunking_it_replaces() {
        for (len, chunk_len, per_task) in [
            (4096usize, 8usize, 1usize),
            (4096, 8, 7),
            (4096, 1, 32),
            (4099, 8, 3),    // short final chunk
            (4096, 4096, 1), // one chunk
            (5, 8, 1),       // shorter than one chunk
            (0, 8, 1),
        ] {
            let f = |chunk_index: usize, slot: usize| {
                (chunk_index as f32) * 1000.0 + (slot as f32) * 0.5 - 3.0
            };

            let mut got = vec![f32::NAN; len];
            par_chunks_indexed(&mut got, chunk_len, per_task, |i, chunk| {
                for (slot, v) in chunk.iter_mut().enumerate() {
                    *v = f(i, slot);
                }
            });

            let mut want = vec![f32::NAN; len];
            for (i, chunk) in want.chunks_mut(chunk_len.max(1)).enumerate() {
                for (slot, v) in chunk.iter_mut().enumerate() {
                    *v = f(i, slot);
                }
            }
            assert_eq!(
                got, want,
                "len={len} chunk_len={chunk_len} per_task={per_task}"
            );
        }
    }

    /// Same region, run enough times to shake out a lost wakeup or a
    /// stale completion count in the pool handshake.
    #[test]
    fn par_chunks_indexed_is_stable_across_many_back_to_back_regions() {
        let len = 2048usize;
        let want: Vec<f32> = (0..len).map(|i| i as f32 * 2.0).collect();
        for round in 0..300 {
            let mut got = vec![f32::NAN; len];
            par_chunks_indexed(&mut got, 4, 2, |i, chunk| {
                for (slot, v) in chunk.iter_mut().enumerate() {
                    *v = ((i * 4 + slot) as f32) * 2.0;
                }
            });
            assert_eq!(got, want, "round {round}");
        }
    }

    #[test]
    fn effective_thread_count_is_positive_and_agrees_with_whoever_runs_the_tasks() {
        let n = effective_num_threads();
        assert!(n >= 1);
        if let Some(pool) = crate::cpu_pool::pool_for_dispatch() {
            assert_eq!(n, pool.width());
        }
    }

    #[test]
    fn for_each_row_parallel_matches_serial() {
        let n = 4097usize;
        let f = |row: usize| ((row % 97) as f32) * 0.25 - 3.0;

        let mut par = vec![0.0f32; n];
        for_each_row(&mut par, n, 4096, |row, slot| *slot = f(row));

        let mut serial = vec![0.0f32; n];
        for (row, slot) in serial.iter_mut().enumerate() {
            *slot = f(row);
        }
        assert_eq!(par, serial);
    }
}
