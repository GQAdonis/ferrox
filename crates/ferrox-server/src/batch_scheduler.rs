//! Continuous-batching decode scheduler: many in-flight sequences share
//! one `Decoder::forward_multi_seq` step per tick instead of each
//! request owning a private `forward_token` loop.
//!
//! Opt-in via `FERROX_CONTINUOUS_BATCHING=1`; mutually exclusive with
//! the KV pool and prefix cache (those paths keep the private-loop
//! `generate`). Stop sequences go through the same
//! [`StopMatcher`](crate::stop::StopMatcher) as
//! `generate::sample_until_stop`: a token-level match on the id before
//! the token is detokenized, plus output-suffix buffering that
//! withholds any tail that could still complete a stop string. One
//! implementation, so a row in a batch and a row on its own cannot
//! disagree about where an answer ends.
//!
//! **Chunked prefill.** A prompt is not prefilled in one uninterruptible
//! `forward_token` loop. Each accepted job first becomes a
//! [`PrefillState`] -- a resumable state machine over (`caches`,
//! `tokens_processed`, `tokens_remaining`) whose `step_chunk` runs at
//! most `prefill_chunk` tokens and reports whether the prompt is
//! finished. That is what makes a prompt a *bounded* unit of work: the
//! worker interleaves **one** prefill chunk with **one** batched decode
//! step per tick, so a long prompt joining the batch delays in-flight
//! decodes by one chunk rather than by its whole length. Chunks are
//! taken round-robin from the waiting prompts, so N concurrent long
//! prompts still cost decode one chunk per tick, not N.
//!
//! Chunking is a *scheduling* boundary, not a numerical one: a chunk is
//! still the same per-token `forward_token` sequence at the same
//! positions, so chunk size never changes the logits or the sampled
//! tokens (asserted by `prefill_chunking_does_not_change_logits`).
//!
//! **Keyed row state.** In-flight rows live in a `HashMap<Uid, Slot>`
//! with a separate admission-ordered `Vec<Uid>`, never in a `Vec<Slot>`
//! addressed by batch position. Batch membership changes on almost
//! every tick -- a row finishes on EOS, on a stop string, on its token
//! budget -- and a positional table renumbers its survivors when that
//! happens (`swap_remove` moves the last row into the removed slot). A
//! `Uid` captured before a removal still names its own row afterwards,
//! or nothing at all; a batch index captured before a removal quietly
//! names a *different request*, whose sampler, stop strings and reply
//! channel are not the ones the caller asked for. That is the bug class
//! oMLX had to monkey-patch around, and it is silent: no panic, no
//! error, just the wrong constraints applied to the wrong row.
//!
//! Per-request sampler state (`Slot::sampler`, seeded per request) lives
//! in the row for the same reason -- a shared or global RNG would make
//! one request's output depend on how many others were in flight.
//!
//! **Knobs.** `FERROX_CB_MAX_SEQS`: cap on concurrent in-flight
//! sequences, counting prompts still prefilling (default: unlimited).
//! At the cap, new jobs stay queued in the channel until a slot frees;
//! only a completely idle worker blocks on `recv`.
//! `FERROX_CB_PREFILL_CHUNK`: prompt tokens per prefill chunk (default
//! [`DEFAULT_PREFILL_CHUNK`]). `FERROX_CB_MAX_QUEUE`: how many jobs may
//! wait for admission before new ones are refused with
//! [`DecodeError::QueueFull`] (default [`DEFAULT_MAX_QUEUE`]).
//!
//! **Block admission.** A sequence count is not a memory budget: 8
//! concurrent 200-token chats and 8 concurrent 100k-token documents
//! are both "8 sequences" and are three orders of magnitude apart in
//! KV. So admission is on an *integer block budget* --
//! `blocks_needed <= blocks_free`, where a block is a fixed run of
//! `FERROX_CB_KV_BLOCK_SIZE` token positions and a request needs
//! `ceil((prompt + max_tokens) / block_size)` of them, reserved for its
//! whole lifetime at admission and released when it finishes.
//!
//! Integer blocks rather than a byte watermark, and this is the one
//! place the source study is deliberately *not* copied: oMLX charges
//! admission against sampled process footprint because MLX's allocator
//! will not tell it what a cache costs. ferrox reads its own KV layout
//! out of the GGUF header, so the exact question -- will this fit --
//! has an exact integer answer, and an exact answer does not need a
//! watermark, a hysteresis band, or a background pressure enforcer.
//!
//! Refusals are typed and split, because they send an operator to
//! different knobs -- an OOM, or a single "server busy", sends them to
//! the wrong one:
//!
//! - **Longer than any one request may be.** `FERROX_CB_MAX_CONTEXT`
//!   positions, if set. `context_length_exceeded` -> 400. The
//!   request's own size is the problem.
//! - **Bigger than the whole KV budget.** `blocks_needed >
//!   blocks_total`: an idle server refuses it identically.
//!   `device_memory_budget_exceeded` -> 400.
//! - **Does not fit right now.** Not an error at all: the job waits at
//!   the head of the admission queue until enough blocks come back.
//! - **Too many already waiting.** [`DecodeError::QueueFull`] -> 503
//!   with `Retry-After`, because that one really does clear.
//!
//! Both 400s name `estimated_bytes` and `limit_bytes` -- the KV cost of
//! the request and of the ceiling it hit, priced from the model's own
//! `KvShape`, so the arithmetic is checkable rather than asserted --
//! and carry separate counters, so "one request too big" and "too many
//! requests" are never read as the same event.
//!
//! Admission is strict FIFO. A skip-ahead policy ("admit the next job
//! that fits") raises utilization and can starve a large request
//! indefinitely behind a stream of small ones -- a queue that reorders
//! by size systematically punishes exactly the requests that already
//! wait longest. FIFO cannot starve, so FIFO it is until there is a
//! measured reason to change it.
//!
//! `FERROX_CB_KV_BLOCKS` unset means no block budget at all, which is
//! today's behaviour: `max_seqs` alone. The two caps compose -- a job
//! must satisfy both.
//!
//! **Deferred abort.** A cancellation never mutates the batch from the
//! canceller's thread. `CancelToken::on_cancel` gives the scheduler a
//! callback that does one thing -- push an [`AbortId`] into
//! [`AbortInbox`] -- and the worker drains that inbox at the top of a
//! tick, *between* forward passes, and does the removal itself.
//!
//! The indirection is the point. Dropping a sequence means dropping the
//! KV buffers a forward pass may be mid-way through reading, and the
//! `std::mem::take` that lifts a row's caches into the batch leaves the
//! row holding empty vectors for the duration of the step -- so a
//! removal landing inside that window would either free live buffers or
//! scatter results back into a row that is no longer there. ferrox-metal
//! has the same hazard in its own terms: its kernels
//! `waitUntilCompleted` per dispatch, so no command buffer outlives a
//! `forward_multi_seq` call, and the step boundary really is a point
//! where nothing is in flight -- but only for the thread that owns the
//! step. That is why the mutation is deferred onto it rather than done
//! wherever the cancel arrived.
//!
//! A cancel is honoured wherever the request has got to: still queued
//! (never prefilled at all), mid-prefill (the remaining chunks are
//! never run), or decoding (it stops at the next step). All three reply
//! [`FinishReason::Cancelled`] with the tokens produced so far, because
//! a cancelled generation has partial output and throwing it away
//! serves nobody.
//!
//! **Queue cap.** The job channel is unbounded, so without a cap a
//! client retry storm turns straight into unbounded memory: every
//! retry parks another prompt (and its reply channel) in the queue,
//! and the server's only signal that it is drowning is the RSS graph.
//! [`QueueGate`] bounds the *waiting* jobs -- in-flight sequences are
//! `FERROX_CB_MAX_SEQS`'s business -- and a refusal is a fast, cheap
//! 503 with `Retry-After` rather than a slow, expensive timeout. A job
//! holds its queue slot until it is *admitted*, not until the worker
//! pulls it off the channel: with a block budget a job can sit in the
//! worker's own admission queue for a while, and a cap that stopped
//! counting it there would stop bounding anything.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use ferrox_core::cache::KvCache;
use ferrox_models::sampling::Sampler;
use ferrox_models::Decoder;
use ferrox_models::{Ceiling, KvElem, KvShape};

use crate::generate::{DecodeError, FinishReason, GenerationParams, Usage};
use crate::stop::{StopMatcher, StopStep};

type DecodeFn = Arc<dyn Fn(&[usize]) -> String + Send + Sync>;

/// Finish reason, generated token ids, detokenized text (stop-trimmed),
/// and usage. Callers should prefer `text` for the response body when
/// stop sequences may have cut the decoded string short of a full
/// `decode(ids)`.
type JobResult = Result<(FinishReason, Vec<usize>, String, Usage), DecodeError>;

/// Prompt tokens run per prefill chunk when `FERROX_CB_PREFILL_CHUNK`
/// is unset. Large enough that a short prompt still prefills in one
/// tick, small enough that a long one cannot monopolize the worker.
pub const DEFAULT_PREFILL_CHUNK: usize = 128;

/// Token positions per KV block when `FERROX_CB_KV_BLOCK_SIZE` is
/// unset. The block is the admission quantum: smaller wastes less on
/// the rounding-up of each request, larger keeps the ledger cheap.
pub const DEFAULT_KV_BLOCK_SIZE: usize = 256;

/// Jobs allowed to wait for admission when `FERROX_CB_MAX_QUEUE` is
/// unset. Deep enough that a normal burst queues instead of failing,
/// shallow enough that a retry storm is refused while the server can
/// still refuse cheaply.
pub const DEFAULT_MAX_QUEUE: usize = 512;

/// Scheduler knobs, read from the environment by `from_env` and passed
/// explicitly by tests (which must not race each other over process
/// environment).
#[derive(Clone, Copy, Debug)]
pub struct BatcherConfig {
    /// Cap on in-flight sequences, counting prompts still prefilling.
    pub max_seqs: usize,
    /// Prompt tokens per `PrefillState::step_chunk` call.
    pub prefill_chunk: usize,
    /// Jobs that may wait for admission before new ones are refused.
    pub max_queue: usize,
    /// Token positions per KV block, the admission quantum.
    pub kv_block_size: usize,
    /// Total KV blocks the scheduler may hand out, or `None` for no
    /// block budget (sequence count alone, the pre-budget behaviour).
    pub kv_blocks: Option<usize>,
    /// Token positions (prompt + `max_tokens`) any single request may
    /// ask for, or `None` for no per-request ceiling.
    pub max_context: Option<usize>,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        BatcherConfig {
            max_seqs: usize::MAX,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            max_queue: DEFAULT_MAX_QUEUE,
            kv_block_size: DEFAULT_KV_BLOCK_SIZE,
            kv_blocks: None,
            max_context: None,
        }
    }
}

impl BatcherConfig {
    pub fn from_env() -> Self {
        BatcherConfig {
            max_seqs: env_positive("FERROX_CB_MAX_SEQS").unwrap_or(usize::MAX),
            prefill_chunk: env_positive("FERROX_CB_PREFILL_CHUNK").unwrap_or(DEFAULT_PREFILL_CHUNK),
            max_queue: env_positive("FERROX_CB_MAX_QUEUE").unwrap_or(DEFAULT_MAX_QUEUE),
            kv_block_size: env_positive("FERROX_CB_KV_BLOCK_SIZE").unwrap_or(DEFAULT_KV_BLOCK_SIZE),
            kv_blocks: env_positive("FERROX_CB_KV_BLOCKS"),
            max_context: env_positive("FERROX_CB_MAX_CONTEXT"),
        }
    }
}

/// Bounds the number of jobs waiting for admission.
///
/// The reservation is a compare-and-swap loop, not a load followed by a
/// fetch_add: with N threads submitting at once, "read the depth, then
/// increment it" admits every thread that read a value below the cap,
/// which is precisely the retry storm the cap exists to stop.
struct QueueGate {
    depth: AtomicUsize,
    cap: usize,
    rejected: AtomicU64,
}

impl QueueGate {
    fn new(cap: usize) -> Self {
        QueueGate {
            depth: AtomicUsize::new(0),
            cap,
            rejected: AtomicU64::new(0),
        }
    }

    /// Claims one queue slot, or reports the depth that refused it.
    fn try_reserve(&self) -> Result<(), usize> {
        let mut current = self.depth.load(Ordering::Acquire);
        loop {
            if current >= self.cap {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(current);
            }
            match self.depth.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => current = actual,
            }
        }
    }

    /// Frees a slot: the worker has taken the job off the channel, or
    /// the send failed and the job never joined the queue at all.
    fn release(&self) {
        let previous = self.depth.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "queue depth underflow");
    }

    fn depth(&self) -> usize {
        self.depth.load(Ordering::Relaxed)
    }

    fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }
}

/// Identifies one submitted job for cancellation. Handed out by
/// [`ContinuousBatcher::generate`] before the job is sent, so a cancel
/// racing the submission has something to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AbortId(u64);

/// Ids whose requests have been asked to stop, waiting for the worker
/// to act on them at a step boundary.
///
/// A set rather than a channel: cancelling twice is the same fact
/// stated twice, and the worker should do the work once.
#[derive(Default)]
struct AbortInbox {
    pending: Mutex<HashSet<AbortId>>,
    next_id: AtomicU64,
    /// Requests actually stopped by a cancellation.
    aborted: AtomicU64,
}

impl AbortInbox {
    fn next_id(&self) -> AbortId {
        AbortId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Called from whichever thread cancelled -- an HTTP handler,
    /// usually. Deliberately does nothing but record the id.
    fn enqueue(&self, id: AbortId) {
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id);
    }

    /// Takes everything pending. Worker thread, at a step boundary.
    fn drain(&self) -> HashSet<AbortId> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn aborted(&self) -> u64 {
        self.aborted.load(Ordering::Relaxed)
    }
}

/// The integer KV-block ledger admission is decided on.
///
/// Blocks are *positions*, not bytes: the scheduler knows its own KV
/// layout, so `blocks_free` is an exact statement about capacity rather
/// than an estimate that needs a safety margin. All mutation happens on
/// the worker thread; the atomics exist so `/metrics` can read the
/// gauge without taking a lock, not to make reservation concurrent.
struct BlockBudget {
    block_size: usize,
    /// Positions any one request may ask for, or `None`.
    max_context: Option<usize>,
    /// What this model's KV really costs, so a refusal can state bytes
    /// rather than an opaque block count. Priced from the GGUF header
    /// via `KvShape`, sliding-window cap included.
    shape: KvShape,
    /// `None` means no block budget is configured -- every request is
    /// admissible as far as this ledger is concerned.
    total: Option<usize>,
    free: AtomicUsize,
    /// Requests refused because they exceed the whole server's KV
    /// budget. Split from `QueueGate::rejected` on purpose: one says
    /// "come back later", the other says "this will never work", and an
    /// operator sent to the wrong one of those tunes the wrong knob.
    rejected_too_large: AtomicU64,
    /// Requests refused for asking for more context than any single
    /// request may have. Split again from the above, because the fix is
    /// in the request rather than on the machine.
    rejected_context_length: AtomicU64,
}

impl BlockBudget {
    fn new(
        block_size: usize,
        total: Option<usize>,
        max_context: Option<usize>,
        shape: KvShape,
    ) -> Self {
        assert!(block_size > 0, "kv block size must be positive");
        BlockBudget {
            block_size,
            max_context,
            shape,
            total,
            free: AtomicUsize::new(total.unwrap_or(0)),
            rejected_too_large: AtomicU64::new(0),
            rejected_context_length: AtomicU64::new(0),
        }
    }

    /// Prices `positions` of context in real KV bytes, sliding-window
    /// cap included.
    fn bytes_for(&self, positions: usize) -> u64 {
        self.shape.kv_bytes_for_tokens(positions)
    }

    /// The typed refusal for a request of `positions` tokens, or `None`
    /// when no immovable ceiling binds.
    ///
    /// Order matters: the context ceiling is checked first because it
    /// is the request's own size, and telling a client "the machine is
    /// too small" when the real answer is "your prompt is too long"
    /// sends it to a knob it does not have.
    fn immovable_refusal(&self, positions: usize) -> Option<DecodeError> {
        if let Some(limit) = self.max_context {
            if positions > limit {
                self.rejected_context_length.fetch_add(1, Ordering::Relaxed);
                return Some(DecodeError::KvBudgetExceeded {
                    binding: Ceiling::ContextLength.code(),
                    estimated_bytes: self.bytes_for(positions),
                    limit_bytes: self.bytes_for(limit),
                    positions,
                    positions_limit: limit,
                    detail: format!(
                        "request asks for {positions} token positions (prompt + max_tokens) \
                         but this deployment admits {limit} per request; shorten the prompt \
                         or lower max_tokens"
                    ),
                });
            }
        }
        let total = self.total?;
        let blocks = self.blocks_for(positions);
        if blocks <= total {
            return None;
        }
        self.rejected_too_large.fetch_add(1, Ordering::Relaxed);
        let limit_positions = total * self.block_size;
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::DeviceMemory.code(),
            estimated_bytes: self.bytes_for(positions),
            limit_bytes: self.bytes_for(limit_positions),
            positions,
            positions_limit: limit_positions,
            detail: format!(
                "request needs {blocks} KV blocks ({positions} token positions at {} per \
                 block) but this server's whole KV budget is {total} blocks; an idle server \
                 would refuse it identically",
                self.block_size
            ),
        })
    }

    /// Blocks a sequence of `positions` tokens occupies. At least one:
    /// even a single-token request holds a block.
    fn blocks_for(&self, positions: usize) -> usize {
        positions.div_ceil(self.block_size).max(1)
    }

    /// Takes `blocks` if they are there. Worker thread only.
    fn try_reserve(&self, blocks: usize) -> bool {
        if self.total.is_none() {
            return true;
        }
        let free = self.free.load(Ordering::Relaxed);
        if blocks > free {
            return false;
        }
        self.free.store(free - blocks, Ordering::Relaxed);
        true
    }

    /// Gives `blocks` back when a request ends, however it ends. Worker
    /// thread only.
    fn release(&self, blocks: usize) {
        let Some(total) = self.total else {
            return;
        };
        let free = self.free.load(Ordering::Relaxed);
        debug_assert!(
            free + blocks <= total,
            "released more blocks than were ever reserved"
        );
        self.free
            .store((free + blocks).min(total), Ordering::Relaxed);
    }

    fn free(&self) -> usize {
        self.total
            .map(|_| self.free.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}

fn env_positive(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let value: usize = raw
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer"));
    assert!(value > 0, "{name} must be a positive integer");
    Some(value)
}

/// Counters the worker keeps as it runs, exposed through
/// `ContinuousBatcher::stats` so prefill/decode interleaving is
/// *observable* rather than merely intended.
#[derive(Default)]
struct Counters {
    prefill_chunks: AtomicU64,
    prefill_tokens: AtomicU64,
    decode_steps: AtomicU64,
    /// High-water mark of KV blocks actually held by in-flight work.
    ///
    /// Deliberately measured from the rows and prefills themselves
    /// rather than derived from `BlockBudget::free`: a ledger-derived
    /// peak cannot exceed the budget however broken admission is, so it
    /// would be a gauge that reports the invariant instead of checking
    /// it.
    peak_blocks: AtomicUsize,
}

/// A snapshot of the worker's counters.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BatcherStats {
    /// `PrefillState::step_chunk` calls the worker has made.
    pub prefill_chunks: u64,
    /// Prompt tokens run through prefill.
    pub prefill_tokens: u64,
    /// Batched decode steps (one per tick that had an active row,
    /// regardless of how many rows that step covered).
    pub decode_steps: u64,
    /// Jobs currently waiting for admission.
    pub queue_depth: usize,
    /// Jobs refused because the queue was full.
    pub queue_rejected: u64,
    /// Total KV blocks in the budget; 0 when none is configured.
    pub kv_blocks_total: usize,
    /// KV blocks not currently reserved by an in-flight request.
    pub kv_blocks_free: usize,
    /// Token positions per KV block.
    pub kv_block_size: usize,
    /// Jobs refused because they exceed the whole KV block budget.
    /// Distinct from `queue_rejected`, which is momentary pressure.
    pub kv_rejected_too_large: u64,
    /// Jobs refused for asking for more context than one request may
    /// have. Distinct again: the fix is in the request, not the box.
    pub kv_rejected_context_length: u64,
    /// Most KV blocks ever held by in-flight work at one moment,
    /// counted from the rows themselves.
    pub kv_blocks_peak: usize,
    /// Requests the scheduler actually stopped because they were
    /// cancelled.
    pub aborted: u64,
}

/// One request's prefill as a resumable state machine.
///
/// Holds the KV `caches` being built, how many prompt tokens have been
/// processed, and the logits produced by the most recent token.
/// `step_chunk` advances by at most `chunk_size` tokens and reports
/// whether the prompt is finished, which converts an unbounded prefill
/// into a bounded unit of work the scheduler can interleave with
/// decode.
pub struct PrefillState {
    decoder: Arc<Decoder>,
    caches: Vec<KvCache>,
    /// The tokens this prefill must run. An *empty* prompt is stored as
    /// a single token 0, matching the private `generate` loop: one
    /// forward pass is still required to produce the logits the first
    /// sampled token comes from.
    tokens: Vec<usize>,
    tokens_processed: usize,
    logits: Vec<f32>,
    chunk_size: usize,
}

impl PrefillState {
    pub fn new(decoder: Arc<Decoder>, prompt_tokens: &[usize], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "prefill chunk size must be positive");
        let caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let tokens = if prompt_tokens.is_empty() {
            vec![0]
        } else {
            prompt_tokens.to_vec()
        };
        PrefillState {
            decoder,
            caches,
            tokens,
            tokens_processed: 0,
            logits: Vec::new(),
            chunk_size,
        }
    }

    /// Prompt tokens already through the model.
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed
    }

    /// Prompt tokens still to run.
    pub fn tokens_remaining(&self) -> usize {
        self.tokens.len() - self.tokens_processed
    }

    pub fn is_done(&self) -> bool {
        self.tokens_remaining() == 0
    }

    /// Runs at most `chunk_size` further prompt tokens. Returns `true`
    /// once the whole prompt has been processed. Calling it again after
    /// that is a no-op that still returns `true`.
    ///
    /// The KV position of each token is its index in the prompt, so
    /// resuming across chunk boundaries is exactly the sequential
    /// `forward_token` loop it replaces, split at different points.
    pub fn step_chunk(&mut self) -> bool {
        let end = (self.tokens_processed + self.chunk_size).min(self.tokens.len());
        for pos in self.tokens_processed..end {
            self.logits = self
                .decoder
                .forward_token(self.tokens[pos], pos, &mut self.caches);
        }
        self.tokens_processed = end;
        self.is_done()
    }

    /// Consumes a finished prefill into the pieces a decode row needs:
    /// KV caches, the logits the first token is sampled from, and the
    /// position the first generated token occupies.
    fn into_decode_start(self) -> (Vec<KvCache>, Vec<f32>, usize) {
        debug_assert!(self.is_done(), "prefill must finish before decoding");
        (self.caches, self.logits, self.tokens_processed)
    }
}

struct Job {
    prompt_tokens: Vec<usize>,
    params: GenerationParams,
    eos_id: Option<usize>,
    reply: Sender<JobResult>,
    /// Cancellation handle for this job, from submission onwards.
    abort: AbortId,
    /// KV blocks this job will need for its whole lifetime, computed
    /// once by the submitter (which already knows the prompt length and
    /// `max_tokens`) so the worker's admission check is a comparison
    /// rather than arithmetic.
    blocks: usize,
}

/// Stable identity for one in-flight request, handed out once at
/// admission and never reused. Unlike a batch index it does not move
/// when another row leaves the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Uid(u64);

/// The in-flight rows: state keyed by [`Uid`], plus the admission order
/// the batch is built in. Removing a row cannot renumber another one --
/// see the keyed-row-state note in the module docs for what that
/// prevents.
#[derive(Default)]
struct Rows {
    state: HashMap<Uid, Slot>,
    /// Admission order. Kept explicit so batch composition is
    /// deterministic; `HashMap` iteration order is not.
    order: Vec<Uid>,
    next_uid: u64,
}

impl Rows {
    fn insert(&mut self, slot: Slot) -> Uid {
        let uid = Uid(self.next_uid);
        self.next_uid += 1;
        self.state.insert(uid, slot);
        self.order.push(uid);
        uid
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    /// KV blocks the rows in this table are holding right now.
    fn blocks_held(&self) -> usize {
        self.state.values().map(|slot| slot.blocks).sum()
    }

    /// Marks any row whose abort id is in `ids` as cancelled, and
    /// reports which ids were consumed.
    ///
    /// Marking, not removing: the row leaves through the same
    /// `flush_finished` path as every other finished row, so its blocks
    /// are released and its caller is replied to exactly once. A second
    /// removal path is a second place to forget one of those.
    fn mark_cancelled(&mut self, ids: &HashSet<AbortId>) -> Vec<AbortId> {
        let mut consumed = Vec::new();
        for uid in &self.order {
            if let Some(slot) = self.state.get_mut(uid) {
                if ids.contains(&slot.abort) && slot.finish.is_none() {
                    slot.finish = Some(FinishReason::Cancelled);
                    consumed.push(slot.abort);
                }
            }
        }
        consumed
    }

    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// `None` for a uid that has already left the batch. A stale uid
    /// resolves to nothing -- never to whichever row happens to sit
    /// where it used to.
    fn get(&self, uid: Uid) -> Option<&Slot> {
        self.state.get(&uid)
    }

    fn get_mut(&mut self, uid: Uid) -> Option<&mut Slot> {
        self.state.get_mut(&uid)
    }

    fn remove(&mut self, uid: Uid) -> Option<Slot> {
        self.order.retain(|&u| u != uid);
        self.state.remove(&uid)
    }

    /// Rows that should take a decode step this tick, in admission
    /// order.
    fn ready(&self) -> Vec<Uid> {
        self.order
            .iter()
            .copied()
            .filter(|uid| {
                self.state
                    .get(uid)
                    .is_some_and(|s| s.finish.is_none() && s.generated_ids.len() < s.max_tokens)
            })
            .collect()
    }

    /// Replies to and removes every row that has finished, returning
    /// each row's KV blocks to the budget as it goes. Release happens
    /// here, on the one path every finished row takes, so a row cannot
    /// leave the table without giving its capacity back.
    fn flush_finished(&mut self, budget: &BlockBudget) {
        let finished: Vec<Uid> = self
            .order
            .iter()
            .copied()
            .filter(|uid| self.state.get(uid).is_some_and(|s| s.finish.is_some()))
            .collect();
        for uid in finished {
            if let Some(slot) = self.remove(uid) {
                budget.release(slot.blocks);
                reply_finished(slot);
            }
        }
    }
}

struct Slot {
    caches: Vec<KvCache>,
    pos: usize,
    logits: Vec<f32>,
    sampler: Sampler,
    generated_ids: Vec<usize>,
    /// Detokenized text already safe to expose (past the stop
    /// hold-back).
    visible: String,
    /// Both stop layers for this row.
    stops: StopMatcher,
    prompt_tokens: usize,
    max_tokens: usize,
    eos_id: Option<usize>,
    params: GenerationParams,
    reply: Sender<JobResult>,
    finish: Option<FinishReason>,
    abort: AbortId,
    /// KV blocks this row reserved at admission, returned when it ends.
    blocks: usize,
}

/// Owns a dedicated worker thread that batches decode steps. Cheap to
/// clone (`Sender` only); the worker stays alive as long as any clone
/// (or the original) exists.
#[derive(Clone)]
pub struct ContinuousBatcher {
    tx: Sender<Job>,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
    budget: Arc<BlockBudget>,
    aborts: Arc<AbortInbox>,
}

struct WorkerGuard {
    _join: JoinHandle<()>,
}

impl ContinuousBatcher {
    /// Spawns the worker. Holds `decoder` and a detokenize callback for
    /// the worker's lifetime. Returns the shareable handle; the worker
    /// exits when the last `ContinuousBatcher` clone is dropped.
    pub fn spawn(decoder: Arc<Decoder>, decode: DecodeFn) -> Self {
        Self::spawn_with_config(decoder, decode, BatcherConfig::from_env())
    }

    /// `spawn` with the scheduler knobs passed in rather than read from
    /// the environment. Tests use this: two tests setting
    /// `FERROX_CB_*` in one process would race each other.
    pub fn spawn_with_config(
        decoder: Arc<Decoder>,
        decode: DecodeFn,
        config: BatcherConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let counters = Arc::new(Counters::default());
        let queue = Arc::new(QueueGate::new(config.max_queue));
        let budget = Arc::new(BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            config.max_context,
            // Prefill runs one token at a time here, so a sliding layer
            // needs `window + 1 - 1` positions live: chunk = 1.
            KvShape::from_config(&decoder.config, KvElem::F32, 1),
        ));
        let aborts = Arc::new(AbortInbox::default());
        let worker_counters = Arc::clone(&counters);
        let worker_queue = Arc::clone(&queue);
        let worker_budget = Arc::clone(&budget);
        let worker_aborts = Arc::clone(&aborts);
        let _join = thread::Builder::new()
            .name("ferrox-continuous-batch".into())
            .spawn(move || {
                worker_loop(
                    decoder,
                    decode,
                    rx,
                    config,
                    worker_counters,
                    worker_queue,
                    worker_budget,
                    worker_aborts,
                )
            })
            .expect("spawn continuous-batch worker");
        // Detach join handle intentionally: dropping the last Sender
        // closes `rx` and ends the loop. Keep a process-lifetime leak
        // of the JoinHandle via Box::leak so dropping batcher clones
        // does not join (callers may still be mid-generate).
        let _guard: &'static WorkerGuard = Box::leak(Box::new(WorkerGuard { _join }));
        ContinuousBatcher {
            tx,
            counters,
            queue,
            budget,
            aborts,
        }
    }

    /// Live scheduler counters. Cheap (a handful of relaxed atomic
    /// loads) and safe to call from any thread while the worker runs.
    pub fn stats(&self) -> BatcherStats {
        BatcherStats {
            prefill_chunks: self.counters.prefill_chunks.load(Ordering::Relaxed),
            prefill_tokens: self.counters.prefill_tokens.load(Ordering::Relaxed),
            decode_steps: self.counters.decode_steps.load(Ordering::Relaxed),
            queue_depth: self.queue.depth(),
            queue_rejected: self.queue.rejected(),
            kv_blocks_total: self.budget.total.unwrap_or(0),
            kv_blocks_free: self.budget.free(),
            kv_block_size: self.budget.block_size,
            kv_rejected_too_large: self.budget.rejected_too_large.load(Ordering::Relaxed),
            kv_rejected_context_length: self.budget.rejected_context_length.load(Ordering::Relaxed),
            kv_blocks_peak: self.counters.peak_blocks.load(Ordering::Relaxed),
            aborted: self.aborts.aborted(),
        }
    }

    /// Submit one generation job and block until it finishes. Safe to
    /// call from many `spawn_blocking` tasks concurrently -- they
    /// serialize only on the shared decode worker, which is the point.
    pub fn generate(
        &self,
        prompt_tokens: Vec<usize>,
        params: GenerationParams,
        eos_id: Option<usize>,
    ) -> Result<(FinishReason, Vec<usize>, String, Usage), DecodeError> {
        // A request that could never fit is refused first, and refused
        // with the ceiling named: queueing it would only make it wait
        // for capacity that will never be enough. This is the
        // immovable half of the rejection split -- 400, not 503.
        let positions = prompt_tokens.len().saturating_add(params.max_tokens);
        let blocks = self.budget.blocks_for(positions);
        if let Some(refusal) = self.budget.immovable_refusal(positions) {
            return Err(refusal);
        }
        // Refuse before allocating a queue slot for the prompt, so a
        // retry storm costs a rejection rather than memory.
        self.queue
            .try_reserve()
            .map_err(|queued| DecodeError::QueueFull {
                queued,
                cap: self.queue.cap,
            })?;
        let (reply_tx, reply_rx) = mpsc::channel();
        let abort = self.aborts.next_id();
        let cancel = params.cancel.clone();
        if self
            .tx
            .send(Job {
                prompt_tokens,
                params,
                eos_id,
                reply: reply_tx,
                abort,
                blocks,
            })
            .is_err()
        {
            // The worker is gone, so nothing will ever dequeue this
            // reservation: release it here or the gate leaks a slot.
            self.queue.release();
            return Err(DecodeError::KvPoolExhausted);
        }
        // Registered after the send, so an id only ever reaches the
        // inbox for a job the worker will actually see. `on_cancel`
        // fires immediately if the token is already cancelled, so the
        // window between the send and this line loses nothing.
        if let Some(token) = cancel {
            let inbox = Arc::clone(&self.aborts);
            token.on_cancel(move || inbox.enqueue(abort));
        }
        reply_rx.recv().unwrap_or(Err(DecodeError::KvPoolExhausted))
    }
}

/// Moves everything currently on the channel into the worker's own
/// admission queue. Both are "waiting for admission" as far as
/// [`QueueGate`] is concerned, so nothing is released here.
fn drain_channel(rx: &Receiver<Job>, waiting: &mut VecDeque<Job>) {
    while let Ok(job) = rx.try_recv() {
        waiting.push_back(job);
    }
}

/// Admits as many waiting jobs as *both* caps allow, turning each into
/// a `Prefill`.
///
/// The sequence cap counts prompts that are still prefilling as well as
/// rows already decoding: a prefilling prompt holds a full set of KV
/// caches, so not counting it would let the worker exceed `max_seqs` by
/// however many prompts happen to be in flight.
///
/// The block cap is the real memory statement:
/// `blocks_needed <= blocks_free`, reserved here for the request's
/// whole lifetime and released in `Rows::flush_finished`.
///
/// Strict FIFO: a head job that does not fit stops the line rather than
/// being skipped over. See the module note on why the skip-ahead
/// alternative is a starvation bug, not an optimization.
fn admit(
    decoder: &Arc<Decoder>,
    waiting: &mut VecDeque<Job>,
    prefills: &mut VecDeque<Prefill>,
    decoding: usize,
    config: &BatcherConfig,
    queue: &QueueGate,
    budget: &BlockBudget,
) {
    while let Some(job) = waiting.front() {
        if decoding + prefills.len() >= config.max_seqs {
            break;
        }
        let blocks = job.blocks;
        if !budget.try_reserve(blocks) {
            break;
        }
        let job = waiting.pop_front().expect("front() just succeeded");
        // Admitted: the job has stopped waiting, so its queue slot goes
        // back now rather than when it was pulled off the channel.
        queue.release();
        match accept(decoder, job, config.prefill_chunk) {
            Some(prefill) => prefills.push_back(prefill),
            // Rejected at validation (a token outside the vocabulary):
            // it never becomes a row, so nothing else would ever give
            // its reservation back.
            None => budget.release(blocks),
        }
    }
}

/// Applies every pending cancellation, at a step boundary, on the
/// worker thread.
///
/// `carried` holds ids that arrived before their job did -- a cancel
/// can beat its own submission through the channel -- so they are kept
/// and retried rather than dropped, which would leave a request running
/// after the client asked it to stop.
///
/// The three states a cancelled request can be in are handled where
/// they live, because they cost different things to abandon:
///
/// - **waiting**: no KV, no reservation, no work done. Replied to and
///   dropped; its queue slot goes back.
/// - **prefilling**: holds KV and a reservation but has produced no
///   tokens. Dropped, reservation released.
/// - **decoding**: marked finished, and left to leave through
///   `flush_finished` like any other completed row, so there is exactly
///   one path that releases blocks and replies.
fn apply_aborts(
    inbox: &AbortInbox,
    carried: &mut HashSet<AbortId>,
    waiting: &mut VecDeque<Job>,
    prefills: &mut VecDeque<Prefill>,
    rows: &mut Rows,
    queue: &QueueGate,
    budget: &BlockBudget,
) {
    carried.extend(inbox.drain());
    if carried.is_empty() {
        return;
    }

    let mut stopped = 0u64;

    // Queued, never started.
    let mut still_waiting = VecDeque::with_capacity(waiting.len());
    while let Some(job) = waiting.pop_front() {
        if carried.remove(&job.abort) {
            queue.release();
            let _ = job.reply.send(Ok((
                FinishReason::Cancelled,
                Vec::new(),
                String::new(),
                Usage::new(job.prompt_tokens.len(), 0),
            )));
            stopped += 1;
        } else {
            still_waiting.push_back(job);
        }
    }
    *waiting = still_waiting;

    // Mid-prefill: the remaining chunks are never run.
    let mut still_prefilling = VecDeque::with_capacity(prefills.len());
    while let Some(prefill) = prefills.pop_front() {
        if carried.remove(&prefill.abort) {
            budget.release(prefill.blocks);
            let _ = prefill.reply.send(Ok((
                FinishReason::Cancelled,
                Vec::new(),
                String::new(),
                Usage::new(prefill.prompt_tokens, 0),
            )));
            stopped += 1;
        } else {
            still_prefilling.push_back(prefill);
        }
    }
    *prefills = still_prefilling;

    // Decoding: marked here, removed by `flush_finished` below.
    for id in rows.mark_cancelled(carried) {
        carried.remove(&id);
        stopped += 1;
    }

    if stopped > 0 {
        inbox.aborted.fetch_add(stopped, Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_loop(
    decoder: Arc<Decoder>,
    decode: DecodeFn,
    rx: Receiver<Job>,
    config: BatcherConfig,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
    budget: Arc<BlockBudget>,
    aborts: Arc<AbortInbox>,
) {
    let mut rows = Rows::default();
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    // Cancellations that arrived before the job they name.
    let mut carried_aborts: HashSet<AbortId> = HashSet::new();
    loop {
        // Only a completely idle worker blocks: with a prompt still
        // chunking, or a job waiting for capacity that an in-flight row
        // will return, there is always work to do on the next tick.
        if rows.is_empty() && prefills.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(job) => waiting.push_back(job),
                Err(_) => break,
            }
        }
        drain_channel(&rx, &mut waiting);
        // Before anything else this tick, and before any forward pass:
        // the one window in which mutating the batch is safe.
        apply_aborts(
            &aborts,
            &mut carried_aborts,
            &mut waiting,
            &mut prefills,
            &mut rows,
            &queue,
            &budget,
        );
        rows.flush_finished(&budget);
        admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            rows.len(),
            &config,
            &queue,
            &budget,
        );
        let held = rows.blocks_held() + prefills.iter().map(|p| p.blocks).sum::<usize>();
        counters.peak_blocks.fetch_max(held, Ordering::Relaxed);
        // With nothing in flight the whole budget is free, and a job
        // that could not fit an empty server was refused at submission
        // -- so an idle worker with a non-empty queue would be a spin,
        // and this is where it would show up.
        debug_assert!(
            !(rows.is_empty() && prefills.is_empty() && !waiting.is_empty()),
            "idle worker cannot admit its queue: {} jobs stuck",
            waiting.len()
        );

        // One bounded prefill chunk per tick, round-robin across the
        // waiting prompts. Round-robin rather than "finish the head
        // first" so a long prompt cannot starve a short one behind it;
        // one chunk rather than "advance every pending prefill" so N
        // concurrent long prompts cost decode one chunk per tick, not N.
        if let Some(mut prefill) = prefills.pop_front() {
            let before = prefill.state.tokens_processed();
            let done = prefill.state.step_chunk();
            counters.prefill_chunks.fetch_add(1, Ordering::Relaxed);
            counters.prefill_tokens.fetch_add(
                (prefill.state.tokens_processed() - before) as u64,
                Ordering::Relaxed,
            );
            if done {
                rows.insert(prefill.into_slot());
            } else {
                prefills.push_back(prefill);
            }
        }

        if rows.is_empty() {
            continue;
        }

        let ready = rows.ready();
        if ready.is_empty() {
            rows.flush_finished(&budget);
            continue;
        }

        // Sample one token per ready row. A row that finishes here (EOS
        // or a stop match) simply does not join `active`.
        let mut active: Vec<Uid> = Vec::with_capacity(ready.len());
        for uid in ready {
            let Some(slot) = rows.get_mut(uid) else {
                continue;
            };
            let next =
                slot.sampler
                    .sample(&slot.logits, &slot.params.sampling, &slot.generated_ids);
            // EOS and a token-level stop are the same fact: this token
            // ends the answer and is not part of it.
            if Some(next) == slot.eos_id || slot.stops.is_stop_token(next) {
                slot.finish = Some(FinishReason::Stop);
                continue;
            }
            slot.generated_ids.push(next);
            let piece = decode(&[next]);
            if apply_stop_buffer(slot, &piece) {
                continue;
            }
            active.push(uid);
        }

        if !active.is_empty() {
            // `active[j]` names the row that owns `logits_batch[j]`.
            // The kernel takes slices, so the batch itself is
            // positional -- but the position maps to a *uid*, so the
            // scatter below cannot land on the wrong request even if
            // the table changes shape between steps.
            let tokens: Vec<usize> = active
                .iter()
                .map(|&uid| *rows.get(uid).unwrap().generated_ids.last().unwrap())
                .collect();
            let positions: Vec<usize> = active
                .iter()
                .map(|&uid| rows.get(uid).unwrap().pos)
                .collect();
            let mut cache_refs: Vec<Vec<KvCache>> = active
                .iter()
                .map(|&uid| std::mem::take(&mut rows.get_mut(uid).unwrap().caches))
                .collect();
            let logits_batch = decoder.forward_multi_seq(&tokens, &positions, &mut cache_refs);
            counters.decode_steps.fetch_add(1, Ordering::Relaxed);
            for (j, &uid) in active.iter().enumerate() {
                let slot = rows
                    .get_mut(uid)
                    .expect("an active row cannot vanish mid-step");
                slot.caches = std::mem::take(&mut cache_refs[j]);
                slot.logits = logits_batch[j].clone();
                slot.pos += 1;
                if slot.generated_ids.len() >= slot.max_tokens {
                    slot.finish = Some(FinishReason::Length);
                }
            }
        }

        rows.flush_finished(&budget);
    }
}

/// Feeds `piece` through this row's stop matcher. Returns true when a
/// stop matched and the row should leave the active batch.
fn apply_stop_buffer(slot: &mut Slot, piece: &str) -> bool {
    match slot.stops.push(piece) {
        StopStep::Emit(text) => {
            slot.visible.push_str(&text);
            false
        }
        StopStep::Matched(text) => {
            slot.visible.push_str(&text);
            slot.finish = Some(FinishReason::Stop);
            true
        }
    }
}

/// An accepted job whose prompt is still being prefilled: the
/// resumable `PrefillState` plus everything the decode row will need
/// once the prompt is through.
struct Prefill {
    state: PrefillState,
    /// The *real* prompt length, for `Usage`. Deliberately not
    /// `state.tokens.len()`, which is 1 for an empty prompt.
    prompt_tokens: usize,
    params: GenerationParams,
    eos_id: Option<usize>,
    reply: Sender<JobResult>,
    abort: AbortId,
    /// Blocks reserved at admission; carried into the `Slot` so the
    /// reservation survives the prefill-to-decode handover.
    blocks: usize,
}

impl Prefill {
    fn into_slot(self) -> Slot {
        let Prefill {
            state,
            prompt_tokens,
            params,
            eos_id,
            reply,
            abort,
            blocks,
        } = self;
        let (caches, logits, pos) = state.into_decode_start();
        Slot {
            caches,
            pos,
            logits,
            sampler: Sampler::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens),
            visible: String::new(),
            stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
            prompt_tokens,
            max_tokens: params.max_tokens,
            eos_id,
            params,
            reply,
            finish: None,
            abort,
            blocks,
        }
    }
}

/// Validates a job and turns it into a waiting `Prefill`. No model work
/// happens here -- every prompt token is run by `step_chunk` on the
/// scheduler's own tick, which is the whole point of chunked prefill.
/// Returns `None` (having replied with the error) for a prompt this
/// model cannot accept at all.
fn accept(decoder: &Arc<Decoder>, job: Job, chunk_size: usize) -> Option<Prefill> {
    let vocab_size = decoder.config.vocab_size;
    if let Some(&bad) = job.prompt_tokens.iter().find(|&&t| t >= vocab_size) {
        let _ = job.reply.send(Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        }));
        return None;
    }

    Some(Prefill {
        state: PrefillState::new(Arc::clone(decoder), &job.prompt_tokens, chunk_size),
        prompt_tokens: job.prompt_tokens.len(),
        params: job.params,
        eos_id: job.eos_id,
        reply: job.reply,
        abort: job.abort,
        blocks: job.blocks,
    })
}

/// Sends one finished row's result to its own waiting caller. Takes the
/// `Slot` by value, so a row's reply channel travels with its state and
/// cannot be paired with another row's output.
fn reply_finished(mut slot: Slot) {
    let finish = slot.finish.expect("only a finished row is replied to");
    // Text withheld against a stop that never arrived is ordinary
    // output; dropping it would truncate every answer whose tail looks
    // like the start of a stop string.
    let tail = slot.stops.flush();
    slot.visible.push_str(&tail);
    let usage = Usage::new(slot.prompt_tokens, slot.generated_ids.len());
    let _ = slot
        .reply
        .send(Ok((finish, slot.generated_ids, slot.visible, usage)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::CancelToken;
    use ferrox_models::config::test_dense_fixture;
    use ferrox_models::sampling::SamplingParams;
    use std::sync::{Barrier, Mutex};

    fn tiny_decoder() -> Arc<Decoder> {
        let cfg = test_dense_fixture();
        let vocab = cfg.vocab_size;
        Arc::new(Decoder::new_random_small(cfg, 2, vocab))
    }

    fn greedy_params(max_tokens: usize, seed: u64) -> GenerationParams {
        GenerationParams {
            max_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            },
            seed,
            stop: vec![],
            stop_token_ids: Vec::new(),
            json_object: false,
            cancel: None,
        }
    }

    fn identity_decode() -> DecodeFn {
        Arc::new(|ids: &[usize]| {
            ids.iter()
                .map(|id| char::from_u32(65 + (*id as u32 % 26)).unwrap_or('?'))
                .collect()
        })
    }

    fn sequential_ids(
        decoder: &Decoder,
        prompt: &[usize],
        params: &GenerationParams,
    ) -> Vec<usize> {
        let mut caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut pos = 0;
        let mut logits = Vec::new();
        for &tok in prompt {
            logits = decoder.forward_token(tok, pos, &mut caches);
            pos += 1;
        }
        let mut sampler = Sampler::new(params.seed);
        let mut generated = Vec::new();
        for _ in 0..params.max_tokens {
            let next = sampler.sample(&logits, &params.sampling, &generated);
            generated.push(next);
            logits = decoder.forward_token(next, pos, &mut caches);
            pos += 1;
        }
        generated
    }

    /// Two concurrent jobs through the batcher must match two sequential
    /// private-loop generates token-for-token.
    #[test]
    fn continuous_batch_matches_sequential_generate_token_ids() {
        let decoder = tiny_decoder();
        let prompts: [Vec<usize>; 2] = [vec![1, 2, 3], vec![4, 5]];
        let params = [greedy_params(8, 7), greedy_params(5, 11)];
        let sequential: Vec<Vec<usize>> = prompts
            .iter()
            .zip(params.iter())
            .map(|(p, par)| sequential_ids(&decoder, p, par))
            .collect();

        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            // Chunk 1: every prompt token is its own scheduling unit, the
            // most aggressive split, and the sampled ids must not move.
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let barrier = Arc::new(Barrier::new(3));
        let results = Arc::new(Mutex::new(vec![None, None]));
        let mut threads = Vec::new();
        for i in 0..2 {
            let batcher = batcher.clone();
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            let prompt = prompts[i].clone();
            let par = GenerationParams {
                max_tokens: params[i].max_tokens,
                sampling: SamplingParams {
                    temperature: params[i].sampling.temperature,
                    top_p: params[i].sampling.top_p,
                    top_k: params[i].sampling.top_k,
                    repetition_penalty: params[i].sampling.repetition_penalty,
                    presence_penalty: params[i].sampling.presence_penalty,
                    frequency_penalty: params[i].sampling.frequency_penalty,
                },
                seed: params[i].seed,
                stop: vec![],
                stop_token_ids: Vec::new(),
                json_object: params[i].json_object,
                cancel: params[i].cancel.clone(),
            };
            threads.push(thread::spawn(move || {
                barrier.wait();
                let out = batcher.generate(prompt, par, None).expect("batch generate");
                results.lock().unwrap()[i] = Some(out.1);
            }));
        }
        barrier.wait();
        for t in threads {
            t.join().unwrap();
        }
        let got = results.lock().unwrap();
        assert_eq!(got[0].as_ref().unwrap(), &sequential[0]);
        assert_eq!(got[1].as_ref().unwrap(), &sequential[1]);
    }

    #[test]
    fn continuous_batch_honors_stop_sequence_in_decoded_text() {
        let decoder = tiny_decoder();
        // Map every token id to a fixed letter so a stop string is easy
        // to force once we know the first few sequential ids.
        let decode: DecodeFn = Arc::new(|ids: &[usize]| {
            ids.iter()
                .map(|id| match id % 3 {
                    0 => 'X',
                    1 => 'Y',
                    _ => 'Z',
                })
                .collect()
        });
        let prompt = vec![1usize, 2, 3];
        let mut params = greedy_params(32, 3);
        // First generate without stop to learn the decoded stream.
        let ids = sequential_ids(&decoder, &prompt, &params);
        let full: String = ids
            .iter()
            .map(|id| match id % 3 {
                0 => 'X',
                1 => 'Y',
                _ => 'Z',
            })
            .collect();
        // Pick a two-char substring that appears mid-stream when long enough.
        assert!(
            full.len() >= 4,
            "need enough tokens to place a mid-stream stop"
        );
        let stop = full[2..4].to_string();
        params.stop = vec![stop.clone()];

        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            decode,
            BatcherConfig {
                prefill_chunk: 2,
                ..BatcherConfig::default()
            },
        );
        let (finish, _ids, text, _usage) = batcher
            .generate(prompt, params, None)
            .expect("batch generate");
        assert_eq!(finish, FinishReason::Stop);
        assert!(
            !text.contains(&stop),
            "stop string must be trimmed from visible text: text={text:?} stop={stop:?}"
        );
        assert_eq!(&full[..full.find(&stop).unwrap()], text);
    }
    /// The state machine itself: each `step_chunk` is bounded by the
    /// chunk size, is resumable, and reports done exactly once the
    /// prompt is exhausted. This is the property the whole scheduler
    /// rests on -- an unbounded prefill has no safe interleaving point.
    #[test]
    fn prefill_step_chunk_is_bounded_and_resumable() {
        let decoder = tiny_decoder();
        let prompt: Vec<usize> = (1..=7).collect();
        let mut state = PrefillState::new(Arc::clone(&decoder), &prompt, 3);
        assert_eq!(state.tokens_remaining(), 7);
        assert_eq!(state.tokens_processed(), 0);

        assert!(!state.step_chunk());
        assert_eq!(state.tokens_processed(), 3, "a chunk may not overrun");
        assert_eq!(state.tokens_remaining(), 4);

        assert!(!state.step_chunk());
        assert_eq!(state.tokens_processed(), 6);

        assert!(state.step_chunk(), "final short chunk finishes the prompt");
        assert_eq!(state.tokens_processed(), 7);
        assert_eq!(state.tokens_remaining(), 0);
        assert!(state.is_done());
        assert!(state.step_chunk(), "stepping a finished prefill is a no-op");
        assert_eq!(state.tokens_processed(), 7);
    }

    /// An empty prompt still needs one forward pass to have logits to
    /// sample from -- the case the pre-chunking `admit` special-cased.
    #[test]
    fn empty_prompt_prefills_one_stand_in_token() {
        let decoder = tiny_decoder();
        let mut state = PrefillState::new(Arc::clone(&decoder), &[], 4);
        assert_eq!(state.tokens_remaining(), 1);
        assert!(state.step_chunk());
        let (_caches, logits, pos) = state.into_decode_start();
        assert_eq!(pos, 1);
        assert_eq!(logits.len(), decoder.config.vocab_size);
    }

    /// Chunking is a scheduling boundary, not a numerical one: whatever
    /// the chunk size, the prompt runs through the same `forward_token`
    /// sequence at the same positions, so the logits are bit-identical
    /// to the sequential prefill this replaced. If this ever fails,
    /// every sampled token downstream is suspect.
    #[test]
    fn prefill_chunking_does_not_change_logits() {
        let decoder = tiny_decoder();
        let prompt: Vec<usize> = (0..11).map(|i| (i * 3 + 1) % 16).collect();

        let mut sequential: Vec<f32> = Vec::new();
        let mut caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        for (pos, &tok) in prompt.iter().enumerate() {
            sequential = decoder.forward_token(tok, pos, &mut caches);
        }

        for chunk in [1usize, 2, 5, 11, 64] {
            let mut state = PrefillState::new(Arc::clone(&decoder), &prompt, chunk);
            while !state.step_chunk() {}
            let (_caches, logits, pos) = state.into_decode_start();
            assert_eq!(pos, prompt.len());
            assert_eq!(
                logits, sequential,
                "chunk size {chunk} changed the prefill logits"
            );
        }
    }

    /// The scheduling property chunking exists for, in two claims that
    /// both fail under an unbounded prefill:
    ///
    /// 1. A long prompt is *observable in partial states* -- it is a
    ///    sequence of bounded units, not one uninterruptible call. The
    ///    pre-chunking scheduler ran the whole prompt inside `admit`,
    ///    where `prefill_tokens` could only ever jump 0 -> len.
    /// 2. Decode keeps stepping while those partial states go by. A
    ///    prompt joining the batch costs an in-flight decode one chunk,
    ///    not the whole prompt.
    #[test]
    fn long_prefill_does_not_freeze_an_in_flight_decode() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );

        // A long-running decode: enough tokens that it is still
        // generating while the second job's prompt is chunked through.
        let decode_job = {
            let batcher = batcher.clone();
            thread::spawn(move || batcher.generate(vec![1, 2], greedy_params(90, 5), None))
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while batcher.stats().decode_steps < 2 {
            assert!(std::time::Instant::now() < deadline, "decode never started");
            thread::yield_now();
        }

        let long_prompt: Vec<usize> = (0..40).map(|i| (i % 16) + 1).collect();
        let total = long_prompt.len() as u64;
        let prefill_at_submit = batcher.stats().prefill_tokens;
        let prefill_job = {
            let batcher = batcher.clone();
            thread::spawn(move || batcher.generate(long_prompt, greedy_params(1, 9), None))
        };

        // Claim 1: catch the long prompt mid-prefill. An unbounded
        // prefill is never observable here -- it goes straight to done.
        let decode_before = loop {
            assert!(
                std::time::Instant::now() < deadline,
                "never observed the long prompt mid-prefill"
            );
            let st = batcher.stats();
            let progressed = st.prefill_tokens - prefill_at_submit;
            assert!(
                progressed < total,
                "the whole prompt was prefilled without ever being observed \
                 partially done: prefill ran as one unbounded unit of work"
            );
            if progressed > 0 {
                break st.decode_steps;
            }
            thread::yield_now();
        };

        // Claim 2: decode advances before that prefill finishes.
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "decode stalled while a long prompt prefilled"
            );
            let st = batcher.stats();
            if st.decode_steps > decode_before {
                break;
            }
            assert!(
                st.prefill_tokens - prefill_at_submit < total,
                "the prompt finished prefilling before the in-flight decode \
                 took a single step: prefill froze decode"
            );
            thread::yield_now();
        }

        let (_finish, ids, _text, _usage) = prefill_job.join().unwrap().expect("prefill job");
        assert_eq!(ids.len(), 1);
        let (_finish, ids, _text, _usage) = decode_job.join().unwrap().expect("decode job");
        assert_eq!(ids.len(), 90);
    }

    /// The in-flight cap counts prompts that are still prefilling, not
    /// just rows already decoding -- a prefilling prompt holds a full
    /// set of KV caches. Two jobs under `max_seqs: 1` must both still
    /// complete correctly (the second waits in the channel).
    #[test]
    fn max_seqs_cap_counts_prefilling_prompts_and_still_serves_both() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                max_seqs: 1,
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let expected: Vec<Vec<usize>> = [(vec![1usize, 2, 3], 6u64), (vec![4usize, 5], 6)]
            .iter()
            .map(|(p, seed)| sequential_ids(&decoder, p, &greedy_params(6, *seed)))
            .collect();

        let handles: Vec<_> = [(vec![1usize, 2, 3], 6u64), (vec![4usize, 5], 6)]
            .into_iter()
            .map(|(prompt, seed)| {
                let batcher = batcher.clone();
                thread::spawn(move || {
                    batcher
                        .generate(prompt, greedy_params(6, seed), None)
                        .expect("generate")
                        .1
                })
            })
            .collect();
        let got: Vec<Vec<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got[0], expected[0]);
        assert_eq!(got[1], expected[1]);
    }
    /// The KV shape of `tiny_decoder`, for pricing a refusal in bytes.
    fn test_shape() -> KvShape {
        KvShape::from_config(&test_dense_fixture(), KvElem::F32, 1)
    }

    /// A ledger with no budget configured, for tests that are not
    /// about admission.
    fn no_budget() -> BlockBudget {
        BlockBudget::new(DEFAULT_KV_BLOCK_SIZE, None, None, test_shape())
    }

    fn budget(block_size: usize, total: Option<usize>) -> BlockBudget {
        BlockBudget::new(block_size, total, None, test_shape())
    }

    fn test_slot(max_tokens: usize, seed: u64) -> (Slot, mpsc::Receiver<JobResult>) {
        let (tx, rx) = mpsc::channel();
        let params = greedy_params(max_tokens, seed);
        (
            Slot {
                caches: Vec::new(),
                pos: 0,
                logits: Vec::new(),
                sampler: Sampler::new(seed),
                generated_ids: Vec::new(),
                visible: String::new(),
                stops: StopMatcher::new(&params.stop, &params.stop_token_ids),
                prompt_tokens: 0,
                max_tokens,
                eos_id: None,
                params,
                reply: tx,
                abort: AbortId(0),
                blocks: 1,
                finish: None,
            },
            rx,
        )
    }

    /// The invariant behind keying rows by uid: a row leaving the batch
    /// must not renumber the rows that stay. A stale id resolves to
    /// nothing; a live id still resolves to its *own* state.
    #[test]
    fn removing_a_row_never_reassigns_another_rows_state() {
        let mut rows = Rows::default();
        let (a, _ra) = test_slot(3, 11);
        let (b, _rb) = test_slot(5, 22);
        let (c, _rc) = test_slot(7, 33);
        let a = rows.insert(a);
        let b = rows.insert(b);
        let c = rows.insert(c);
        assert_eq!(rows.order, vec![a, b, c]);

        let removed = rows.remove(b).expect("b was present");
        assert_eq!(removed.max_tokens, 5);

        assert!(
            rows.get(b).is_none(),
            "a stale uid must resolve to nothing, never to another request's row"
        );
        assert_eq!(rows.get(a).expect("a still in flight").max_tokens, 3);
        assert_eq!(
            rows.get(c).expect("c still in flight").max_tokens,
            7,
            "c must still be c after b left"
        );
        assert_eq!(rows.order, vec![a, c], "admission order is preserved");
        assert_eq!(rows.len(), 2);

        // The positional equivalent, spelled out: `swap_remove` moves
        // the last row into the removed slot, so an index captured for
        // C before the removal now addresses B's old position -- or
        // nothing. Same removal, silently wrong answer.
        let mut positional = vec![3usize, 5, 7];
        let c_index = 2;
        positional.swap_remove(1);
        assert_eq!(positional[1], 7, "C moved into B's index");
        assert!(
            positional.get(c_index).is_none(),
            "C's index now names nothing"
        );
    }

    /// A new row joining the table must not disturb the rows already in
    /// it, and uids are never reused -- so a reply channel and a
    /// sampler always travel with the request that owns them.
    #[test]
    fn uids_are_unique_and_insertion_does_not_disturb_existing_rows() {
        let mut rows = Rows::default();
        let (a, _ra) = test_slot(3, 11);
        let a = rows.insert(a);
        let (b, _rb) = test_slot(5, 22);
        let b = rows.insert(b);
        rows.remove(a);
        let (c, _rc) = test_slot(7, 33);
        let c = rows.insert(c);
        assert_ne!(c, a, "a uid is never reused after its row leaves");
        assert_ne!(c, b);
        assert_eq!(rows.get(b).expect("b untouched").max_tokens, 5);
        assert_eq!(rows.get(c).expect("c inserted").max_tokens, 7);
    }

    /// `ready` skips finished rows, and `flush_finished` replies to and
    /// removes exactly those -- each on its own channel.
    #[test]
    fn flush_replies_on_each_rows_own_channel() {
        let mut rows = Rows::default();
        let (a, ra) = test_slot(3, 11);
        let (mut b, rb) = test_slot(5, 22);
        b.finish = Some(FinishReason::Stop);
        b.visible.push_str("bee");
        b.generated_ids.push(7);
        let a = rows.insert(a);
        let b = rows.insert(b);
        let (c, _rc) = test_slot(7, 33);
        let c = rows.insert(c);

        assert_eq!(rows.ready(), vec![a, c], "a finished row takes no step");
        rows.flush_finished(&no_budget());
        assert!(rows.get(b).is_none());
        assert_eq!(rows.order, vec![a, c]);

        let (finish, ids, text, usage) =
            rb.try_recv().expect("b's caller got a reply").expect("ok");
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(ids, vec![7]);
        assert_eq!(text, "bee");
        assert_eq!(usage.completion_tokens, 1);
        assert!(
            ra.try_recv().is_err(),
            "an unfinished row's caller must not be replied to"
        );
    }

    /// End to end, with the batch mutation that renumbers a positional
    /// table: three concurrent rows, one of which trips a stop sequence
    /// mid-batch and leaves while the other two keep decoding. In that
    /// tick the batch is narrower than the row table, which is exactly
    /// when a batch index stops meaning what a row id means. Each
    /// caller must still get its own output.
    #[test]
    fn a_row_leaving_mid_batch_does_not_shift_its_neighbours_output() {
        let decoder = tiny_decoder();
        let prompts = [vec![1usize, 2, 3], vec![4usize, 5], vec![6usize]];
        let budgets = [25usize, 25, 20];
        let refs: Vec<Vec<usize>> = prompts
            .iter()
            .zip(budgets.iter())
            .map(|(p, &n)| sequential_ids(&decoder, p, &greedy_params(n, 4)))
            .collect();

        // The middle row stops on a two-character run from its own
        // stream, so it leaves the batch while its neighbours decode on.
        let letter = |id: &usize| char::from_u32(65 + (*id as u32 % 26)).unwrap_or('?');
        let middle_text: String = refs[1].iter().map(letter).collect();
        assert!(middle_text.len() >= 4);
        let stop = middle_text[2..4].to_string();

        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let barrier = Arc::new(Barrier::new(prompts.len()));
        let handles: Vec<_> = (0..prompts.len())
            .map(|i| {
                let batcher = batcher.clone();
                let barrier = Arc::clone(&barrier);
                let prompt = prompts[i].clone();
                let mut params = greedy_params(budgets[i], 4);
                if i == 1 {
                    params.stop = vec![stop.clone()];
                }
                thread::spawn(move || {
                    barrier.wait();
                    batcher.generate(prompt, params, None).expect("generate").1
                })
            })
            .collect();
        let got: Vec<Vec<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(got[0], refs[0], "row 0 received another row's output");
        assert_eq!(got[2], refs[2], "row 2 received another row's output");
        assert!(
            got[1].len() < refs[1].len() && refs[1].starts_with(&got[1]),
            "the stopped row must be a strict prefix of its own stream"
        );
    }
    #[test]
    fn queue_gate_admits_up_to_its_cap_and_frees_slots_on_release() {
        let gate = QueueGate::new(2);
        assert!(gate.try_reserve().is_ok());
        assert!(gate.try_reserve().is_ok());
        assert_eq!(gate.depth(), 2);
        assert_eq!(gate.try_reserve(), Err(2), "the refusal reports the depth");
        assert_eq!(gate.rejected(), 1);
        gate.release();
        assert_eq!(gate.depth(), 1);
        assert!(
            gate.try_reserve().is_ok(),
            "a released slot must be reusable"
        );
        assert_eq!(gate.depth(), 2);
    }

    /// The cap is only a cap if it holds under the exact condition it
    /// exists for: many clients submitting at once. A check-then-act
    /// gate ("read the depth, then increment") lets every thread that
    /// read a value below the cap through, which is how a retry storm
    /// gets past a limit that looks correct when read in isolation.
    ///
    /// Repeated rounds because a lost race is probabilistic: one round
    /// can get lucky, sixty-four rounds of thirty-two racing threads do
    /// not.
    #[test]
    fn queue_gate_never_exceeds_its_cap_under_concurrent_submitters() {
        const THREADS: usize = 32;
        const CAP: usize = 4;
        for round in 0..64 {
            let gate = Arc::new(QueueGate::new(CAP));
            let barrier = Arc::new(Barrier::new(THREADS));
            let admitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let gate = Arc::clone(&gate);
                    let barrier = Arc::clone(&barrier);
                    let admitted = Arc::clone(&admitted);
                    thread::spawn(move || {
                        barrier.wait();
                        if gate.try_reserve().is_ok() {
                            admitted.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(
                admitted.load(Ordering::Relaxed),
                CAP,
                "round {round}: exactly the cap may be admitted"
            );
            assert_eq!(gate.depth(), CAP, "round {round}: depth matches admissions");
            assert_eq!(gate.rejected(), (THREADS - CAP) as u64);
        }
    }

    /// End to end: a full queue is refused with a typed error naming
    /// the depth and the cap, and the refusal costs nothing -- no
    /// prompt is queued, no reply channel is parked.
    #[test]
    fn a_full_queue_refuses_new_jobs_with_queue_full() {
        let decoder = tiny_decoder();
        // cap 0 is degenerate on purpose: it makes "the queue is full"
        // deterministic in a test, where a real cap would be drained by
        // the worker before a second submission could ever see it.
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                max_queue: 0,
                ..BatcherConfig::default()
            },
        );
        let err = batcher
            .generate(vec![1, 2, 3], greedy_params(4, 1), None)
            .expect_err("a full queue must refuse");
        assert!(
            matches!(err, DecodeError::QueueFull { queued: 0, cap: 0 }),
            "expected QueueFull, got {err:?}"
        );
        assert_eq!(err.retry_after_secs(), Some(1), "a queue drains; say so");
        let stats = batcher.stats();
        assert_eq!(stats.queue_rejected, 1);
        assert_eq!(stats.queue_depth, 0, "a refused job holds nothing");
    }

    /// The gate must not leak slots: a job that is accepted, queued,
    /// dequeued and served leaves the queue empty again.
    #[test]
    fn queue_depth_returns_to_zero_after_a_served_request() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                max_queue: 1,
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        for _ in 0..3 {
            batcher
                .generate(vec![1, 2, 3], greedy_params(2, 1), None)
                .expect("a cap of 1 still serves requests one after another");
        }
        assert_eq!(batcher.stats().queue_depth, 0);
        assert_eq!(batcher.stats().queue_rejected, 0);
    }

    // ----------------------------------------------------------------
    // Block admission (`sched-block-admission`)
    // ----------------------------------------------------------------

    fn budget_config(block_size: usize, blocks: usize) -> BatcherConfig {
        BatcherConfig {
            prefill_chunk: 1,
            kv_block_size: block_size,
            kv_blocks: Some(blocks),
            ..BatcherConfig::default()
        }
    }

    #[test]
    fn blocks_are_counted_in_positions_and_always_round_up() {
        let budget = budget(4, Some(10));
        // A single-token request still holds a block.
        assert_eq!(budget.blocks_for(0), 1);
        assert_eq!(budget.blocks_for(1), 1);
        assert_eq!(budget.blocks_for(4), 1);
        // Rounding up, not down: 5 positions do not fit in one block.
        assert_eq!(budget.blocks_for(5), 2);
        assert_eq!(budget.blocks_for(8), 2);
        assert_eq!(budget.blocks_for(9), 3);
    }

    #[test]
    fn an_unconfigured_budget_admits_everything() {
        let budget = budget(4, None);
        assert!(budget.immovable_refusal(usize::MAX).is_none());
        assert!(budget.try_reserve(1_000_000));
        budget.release(1_000_000);
    }

    /// The whole point of an integer budget: a request that needs more
    /// blocks than the server owns is refused as a *request* problem,
    /// before it is allowed to occupy a queue slot. Answering 503 here
    /// would send a client into a retry loop that cannot ever succeed.
    #[test]
    fn a_request_larger_than_the_whole_budget_is_refused_rather_than_queued() {
        let decoder = tiny_decoder();
        // 2 blocks of 4 positions = 8 positions in the entire server.
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            budget_config(4, 2),
        );

        let err = batcher
            .generate(vec![1, 2, 3, 4, 5, 6], greedy_params(8, 1), None)
            .expect_err("14 positions cannot fit an 8-position server");
        let shape = test_shape();
        match &err {
            DecodeError::KvBudgetExceeded {
                binding,
                estimated_bytes,
                limit_bytes,
                positions,
                positions_limit,
                detail,
            } => {
                assert_eq!(*binding, "device_memory_budget_exceeded");
                assert_eq!(*positions, 14);
                assert_eq!(*positions_limit, 8, "2 blocks x 4 positions");
                // The bytes are the model's real KV cost, not a
                // restatement of the block count: an operator has to be
                // able to check the arithmetic against `inspect-plan`.
                assert_eq!(*estimated_bytes, shape.kv_bytes_for_tokens(14));
                assert_eq!(*limit_bytes, shape.kv_bytes_for_tokens(8));
                assert!(estimated_bytes > limit_bytes);
                assert!(detail.contains("14"), "{detail}");
            }
            other => panic!("expected KvBudgetExceeded, got {other:?}"),
        }
        // Retrying it is pointless, and the error says so.
        assert_eq!(err.retry_after_secs(), None);

        let stats = batcher.stats();
        assert_eq!(stats.kv_rejected_too_large, 1);
        assert_eq!(
            stats.kv_rejected_context_length, 0,
            "no per-request context ceiling is configured here"
        );
        assert_eq!(
            stats.queue_rejected, 0,
            "too-big and under-pressure are different counters"
        );
        assert_eq!(
            stats.queue_depth, 0,
            "an impossible request must not occupy a queue slot"
        );
        assert_eq!(stats.kv_blocks_free, 2, "nothing was reserved");

        // A request that does fit still works on the same server.
        batcher
            .generate(vec![1, 2], greedy_params(2, 1), None)
            .expect("4 positions fit");
    }

    /// The other immovable ceiling, and the reason there are two: this
    /// request is not too big for the machine, it is too big for what
    /// any one request is allowed to be. An operator reading
    /// `device_memory_budget_exceeded` here would go looking for a
    /// bigger box for a problem a shorter prompt solves.
    ///
    /// Confirmed to FAIL when the `max_context` branch is removed from
    /// `immovable_refusal` (the request is admitted and runs).
    #[test]
    fn a_request_longer_than_the_context_ceiling_names_that_ceiling() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                max_context: Some(6),
                // A generous block budget, so the *only* thing that can
                // bind is the per-request context ceiling.
                kv_block_size: 4,
                kv_blocks: Some(1024),
                ..BatcherConfig::default()
            },
        );

        let err = batcher
            .generate(vec![1, 2, 3, 4], greedy_params(4, 1), None)
            .expect_err("8 positions against a 6-position ceiling");
        let shape = test_shape();
        match &err {
            DecodeError::KvBudgetExceeded {
                binding,
                estimated_bytes,
                limit_bytes,
                positions,
                positions_limit,
                detail,
            } => {
                assert_eq!(*binding, "context_length_exceeded");
                assert_eq!(*positions, 8);
                assert_eq!(*positions_limit, 6);
                assert_eq!(*estimated_bytes, shape.kv_bytes_for_tokens(8));
                assert_eq!(*limit_bytes, shape.kv_bytes_for_tokens(6));
                assert!(detail.contains("max_tokens"), "{detail}");
            }
            other => panic!("expected KvBudgetExceeded, got {other:?}"),
        }
        assert_eq!(err.retry_after_secs(), None);

        let stats = batcher.stats();
        assert_eq!(stats.kv_rejected_context_length, 1);
        assert_eq!(
            stats.kv_rejected_too_large, 0,
            "the machine's budget was never the binding ceiling"
        );
        assert_eq!(stats.queue_rejected, 0);

        // Exactly at the ceiling is admitted: the check is `>`, not
        // `>=`, or the advertised limit would be off by one.
        batcher
            .generate(vec![1, 2, 3, 4], greedy_params(2, 1), None)
            .expect("6 positions is 6 positions");
    }

    /// With both ceilings configured and both exceeded, the request's
    /// own size is reported -- it is the one the client can act on.
    #[test]
    fn the_context_ceiling_is_reported_before_the_device_ceiling() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                max_context: Some(6),
                kv_block_size: 4,
                kv_blocks: Some(2),
                ..BatcherConfig::default()
            },
        );
        let err = batcher
            .generate(vec![1, 2, 3, 4, 5, 6], greedy_params(8, 1), None)
            .expect_err("14 positions breaks both ceilings");
        assert!(
            matches!(
                &err,
                DecodeError::KvBudgetExceeded { binding, .. }
                    if *binding == "context_length_exceeded"
            ),
            "got {err:?}"
        );
        let stats = batcher.stats();
        assert_eq!(stats.kv_rejected_context_length, 1);
        assert_eq!(stats.kv_rejected_too_large, 0);
    }

    /// No ceilings configured is the default, and it must refuse
    /// nothing.
    #[test]
    fn without_ceilings_nothing_is_refused_as_too_large() {
        let budget = BlockBudget::new(4, None, None, test_shape());
        assert!(budget.immovable_refusal(1_000_000).is_none());
        assert_eq!(budget.rejected_too_large.load(Ordering::Relaxed), 0);
        assert_eq!(budget.rejected_context_length.load(Ordering::Relaxed), 0);
    }

    /// The invariant, under real contention: however many requests are
    /// in flight, the blocks they hold together never exceed the
    /// budget. Six requests each needing two blocks against a
    /// four-block server can be at most two at a time.
    ///
    /// Confirmed to FAIL when `admit` reserves unconditionally (peak
    /// climbs to 12 -- every request admitted at once).
    #[test]
    fn concurrent_requests_never_hold_more_blocks_than_the_budget() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            // 4 positions per block, 4 blocks. Each request below is
            // 3 prompt + 5 generated = 8 positions = 2 blocks.
            budget_config(4, 4),
        );

        let start = Arc::new(Barrier::new(6));
        let handles: Vec<_> = (0..6)
            .map(|i| {
                let batcher = batcher.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    batcher
                        .generate(vec![1, 2, 3], greedy_params(5, i as u64), None)
                        .expect("every request fits the budget on its own")
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no submitter panicked");
        }

        let stats = batcher.stats();
        assert!(
            stats.kv_blocks_peak <= stats.kv_blocks_total,
            "admission handed out {} blocks from a budget of {}",
            stats.kv_blocks_peak,
            stats.kv_blocks_total
        );
        assert_eq!(stats.kv_rejected_too_large, 0, "all six fit individually");
    }

    /// Every reservation comes back. A row that leaves without
    /// releasing is a leak that shows up as a server which slowly stops
    /// admitting anything, with no error anywhere -- so the ledger must
    /// be exactly restored once the work is done.
    ///
    /// Confirmed to FAIL when the `budget.release` in
    /// `Rows::flush_finished` is removed.
    #[test]
    fn every_admitted_request_gives_its_blocks_back() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            budget_config(4, 4),
        );
        for i in 0..8 {
            batcher
                .generate(vec![1, 2, 3], greedy_params(4, i), None)
                .expect("generate");
        }
        // The last reply is sent before the row is removed, so give the
        // worker its moment to finish the release.
        for _ in 0..200 {
            if batcher.stats().kv_blocks_free == 4 {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        let stats = batcher.stats();
        assert_eq!(
            stats.kv_blocks_free, stats.kv_blocks_total,
            "an idle server must own its whole budget again"
        );
        assert!(stats.kv_blocks_peak > 0, "something was actually reserved");
    }

    /// A rejected job must not leak its reservation either: `accept`
    /// refuses a prompt this model cannot tokenize, and that job never
    /// becomes a row, so nothing downstream would ever release it.
    #[test]
    fn a_job_rejected_at_validation_gives_its_blocks_back() {
        let decoder = tiny_decoder();
        let vocab = decoder.config.vocab_size;
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            budget_config(4, 4),
        );
        assert!(matches!(
            batcher.generate(vec![vocab + 1], greedy_params(2, 1), None),
            Err(DecodeError::TokenOutOfVocab { .. })
        ));
        for _ in 0..200 {
            if batcher.stats().kv_blocks_free == 4 {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(batcher.stats().kv_blocks_free, 4);
        // And the server still works afterwards.
        batcher
            .generate(vec![1, 2], greedy_params(2, 1), None)
            .expect("a bad request must not poison the budget");
    }

    /// Admission is strict FIFO: a head job that does not fit stops the
    /// line rather than being skipped over by a smaller one behind it.
    /// Skip-ahead would raise utilization and let a stream of small
    /// requests starve a large one indefinitely.
    #[test]
    fn a_head_job_that_does_not_fit_holds_the_line() {
        let decoder = tiny_decoder();
        let config = budget_config(4, 4);
        let budget = BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            config.max_context,
            test_shape(),
        );
        let queue = QueueGate::new(config.max_queue);
        // Two blocks already out to an in-flight request.
        assert!(budget.try_reserve(2));

        let mut waiting: VecDeque<Job> = VecDeque::new();
        let (big_tx, _big_rx) = mpsc::channel();
        let (small_tx, _small_rx) = mpsc::channel();
        waiting.push_back(Job {
            prompt_tokens: vec![1, 2, 3],
            params: greedy_params(2, 1),
            eos_id: None,
            reply: big_tx,
            abort: AbortId(0),
            blocks: 3,
        });
        waiting.push_back(Job {
            prompt_tokens: vec![1],
            params: greedy_params(2, 2),
            eos_id: None,
            reply: small_tx,
            abort: AbortId(1),
            blocks: 1,
        });
        // The gate is counting both of them.
        queue.try_reserve().expect("cap 512");
        queue.try_reserve().expect("cap 512");

        let mut prefills: VecDeque<Prefill> = VecDeque::new();
        admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            0,
            &config,
            &queue,
            &budget,
        );

        assert!(
            prefills.is_empty(),
            "the 1-block job must not jump the 3-block job that cannot fit"
        );
        assert_eq!(waiting.len(), 2);
        assert_eq!(queue.depth(), 2, "neither job has stopped waiting");

        // Once the in-flight request finishes, the line moves -- both,
        // in order.
        budget.release(2);
        admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            0,
            &config,
            &queue,
            &budget,
        );
        assert_eq!(prefills.len(), 2);
        assert!(waiting.is_empty());
        assert_eq!(queue.depth(), 0);
        assert_eq!(budget.free(), 0, "3 + 1 blocks are now out");
    }

    /// The sequence cap and the block cap are separate statements and
    /// both must hold. A budget with room for four requests does not
    /// override `max_seqs = 1`.
    #[test]
    fn the_sequence_cap_and_the_block_cap_compose() {
        let decoder = tiny_decoder();
        let config = BatcherConfig {
            max_seqs: 1,
            ..budget_config(4, 8)
        };
        let budget = BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            config.max_context,
            test_shape(),
        );
        let queue = QueueGate::new(config.max_queue);
        let mut waiting: VecDeque<Job> = VecDeque::new();
        // The receivers stay alive for the test: a dropped receiver
        // would make the reply channel closed, which is a different
        // situation from the one under test.
        let mut receivers = Vec::new();
        for i in 0..3 {
            let (tx, rx) = mpsc::channel();
            receivers.push(rx);
            waiting.push_back(Job {
                prompt_tokens: vec![1, 2],
                params: greedy_params(2, i),
                eos_id: None,
                reply: tx,
                abort: AbortId(i),
                blocks: 1,
            });
            queue.try_reserve().expect("cap 512");
        }
        let mut prefills: VecDeque<Prefill> = VecDeque::new();
        admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            0,
            &config,
            &queue,
            &budget,
        );
        assert_eq!(prefills.len(), 1, "max_seqs still binds");
        assert_eq!(budget.free(), 7, "only the admitted job reserved");
        assert_eq!(receivers.len(), 3);
    }

    // ----------------------------------------------------------------
    // Deferred abort (`sched-deferred-abort`)
    // ----------------------------------------------------------------

    fn cancellable_params(max_tokens: usize, seed: u64) -> (GenerationParams, CancelToken) {
        let token = CancelToken::new();
        let mut params = greedy_params(max_tokens, seed);
        params.cancel = Some(token.clone());
        (params, token)
    }

    fn abortable_job(abort: AbortId, prompt: Vec<usize>) -> (Job, mpsc::Receiver<JobResult>) {
        let (tx, rx) = mpsc::channel();
        (
            Job {
                prompt_tokens: prompt,
                params: greedy_params(4, 1),
                eos_id: None,
                reply: tx,
                abort,
                blocks: 1,
            },
            rx,
        )
    }

    /// The end-to-end fact the item exists for, and the gap the
    /// `cancel` module used to state as unfixable: a request decoding
    /// on the shared batcher thread stops when the client cancels it.
    ///
    /// Confirmed to FAIL (finishes with `Length` after all 4000 tokens)
    /// when the `apply_aborts` call is removed from the worker loop.
    #[test]
    fn a_decoding_request_stops_when_it_is_cancelled() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let (params, token) = cancellable_params(4000, 9);
        let worker = {
            let batcher = batcher.clone();
            thread::spawn(move || batcher.generate(vec![1, 2, 3], params, None))
        };

        // Wait until it is genuinely decoding, so the cancel exercises
        // the decode path rather than racing the prefill.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while batcher.stats().decode_steps < 3 {
            assert!(
                std::time::Instant::now() < deadline,
                "never started decoding"
            );
            thread::sleep(std::time::Duration::from_millis(1));
        }
        token.cancel();

        let (finish, ids, _text, usage) = worker.join().expect("no panic").expect("generate");
        assert_eq!(finish, FinishReason::Cancelled);
        assert!(
            ids.len() < 4000,
            "cancelling did not shorten the decode: {} tokens",
            ids.len()
        );
        assert!(
            !ids.is_empty(),
            "the tokens produced before the cancel must survive it"
        );
        assert_eq!(usage.completion_tokens, ids.len());
        assert_eq!(batcher.stats().aborted, 1);
    }

    /// A cancelled request must leave the batch through the same exit
    /// every finished row uses -- marked at the step boundary, removed
    /// by `flush_finished` -- so its blocks are released once and its
    /// caller is replied to once.
    ///
    /// Removing the row inside `apply_aborts` instead would be a second
    /// exit path, and the reply this asserts arrives exactly once would
    /// arrive twice.
    #[test]
    fn a_cancelled_row_leaves_through_the_one_exit_every_row_uses() {
        let mut rows = Rows::default();
        let budget = budget(4, Some(4));
        assert!(budget.try_reserve(2));

        let (mut a, ra) = test_slot(9, 11);
        a.abort = AbortId(7);
        a.blocks = 2;
        a.generated_ids.push(3);
        a.visible.push_str("hi");
        let (b, rb) = test_slot(9, 22);
        let a_uid = rows.insert(a);
        let b_uid = rows.insert(b);

        let consumed = rows.mark_cancelled(&HashSet::from([AbortId(7)]));
        assert_eq!(consumed, vec![AbortId(7)]);
        assert!(
            rows.get(a_uid).is_some(),
            "the row must still be in the table until the flush: its KV \
             buffers are what the batch is built from"
        );
        assert_eq!(
            budget.free(),
            2,
            "marking must not release blocks -- the flush does"
        );
        assert!(ra.try_recv().is_err(), "no reply until the row leaves");

        rows.flush_finished(&budget);
        assert!(rows.get(a_uid).is_none());
        assert!(rows.get(b_uid).is_some(), "only the cancelled row left");
        assert_eq!(budget.free(), 4, "blocks came back exactly once");

        let (finish, ids, text, _usage) = ra.try_recv().expect("one reply").expect("ok");
        assert_eq!(finish, FinishReason::Cancelled);
        assert_eq!(ids, vec![3], "partial output survives the cancel");
        assert_eq!(text, "hi");
        assert!(
            ra.try_recv().is_err(),
            "a cancelled row must be replied to exactly once"
        );
        assert!(rb.try_recv().is_err(), "the other row is untouched");
    }

    /// A cancel can beat its own job through the channel. Dropping the
    /// id because nothing matches it yet would leave the request
    /// running after the client asked it to stop.
    ///
    /// Confirmed to FAIL when `apply_aborts` drops unmatched ids
    /// instead of carrying them.
    #[test]
    fn a_cancel_that_arrives_before_its_job_is_not_lost() {
        let config = budget_config(4, 4);
        let inbox = AbortInbox::default();
        let queue = QueueGate::new(config.max_queue);
        let budget = BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            config.max_context,
            test_shape(),
        );
        let mut carried = HashSet::new();
        let mut waiting = VecDeque::new();
        let mut prefills = VecDeque::new();
        let mut rows = Rows::default();

        // The cancel lands first, naming nothing.
        inbox.enqueue(AbortId(42));
        apply_aborts(
            &inbox,
            &mut carried,
            &mut waiting,
            &mut prefills,
            &mut rows,
            &queue,
            &budget,
        );
        assert_eq!(inbox.aborted(), 0, "nothing to stop yet");

        // Now the job it names shows up.
        let (job, rx) = abortable_job(AbortId(42), vec![1, 2]);
        waiting.push_back(job);
        queue.try_reserve().expect("cap");
        apply_aborts(
            &inbox,
            &mut carried,
            &mut waiting,
            &mut prefills,
            &mut rows,
            &queue,
            &budget,
        );

        assert!(waiting.is_empty(), "the late job must still be cancelled");
        assert_eq!(inbox.aborted(), 1);
        assert_eq!(queue.depth(), 0, "its queue slot came back");
        let (finish, ids, _, usage) = rx.try_recv().expect("reply").expect("ok");
        assert_eq!(finish, FinishReason::Cancelled);
        assert!(ids.is_empty(), "it never ran a token");
        assert_eq!(usage.prompt_tokens, 2);
    }

    /// A cancelled prompt must not be prefilled: that is the cheapest
    /// possible moment to stop, and the whole point of checking before
    /// the chunk rather than after it.
    #[test]
    fn a_cancelled_prefill_is_abandoned_and_gives_its_blocks_back() {
        let decoder = tiny_decoder();
        let config = budget_config(4, 4);
        let inbox = AbortInbox::default();
        let queue = QueueGate::new(config.max_queue);
        let budget = BlockBudget::new(
            config.kv_block_size,
            config.kv_blocks,
            config.max_context,
            test_shape(),
        );
        let mut carried = HashSet::new();
        let mut waiting = VecDeque::new();
        let mut prefills = VecDeque::new();
        let mut rows = Rows::default();

        let (job, rx) = abortable_job(AbortId(5), vec![1, 2, 3, 4]);
        waiting.push_back(job);
        queue.try_reserve().expect("cap");
        admit(
            &decoder,
            &mut waiting,
            &mut prefills,
            0,
            &config,
            &queue,
            &budget,
        );
        assert_eq!(prefills.len(), 1);
        assert_eq!(budget.free(), 3, "the prefill holds its reservation");

        inbox.enqueue(AbortId(5));
        apply_aborts(
            &inbox,
            &mut carried,
            &mut waiting,
            &mut prefills,
            &mut rows,
            &queue,
            &budget,
        );
        assert!(prefills.is_empty(), "the remaining chunks never run");
        assert_eq!(budget.free(), 4, "an abandoned prefill releases its blocks");
        assert_eq!(inbox.aborted(), 1);
        assert_eq!(
            rx.try_recv().expect("reply").expect("ok").0,
            FinishReason::Cancelled
        );
    }

    /// Cancelling one request must not disturb any other -- the failure
    /// mode a positional row table would make easy.
    #[test]
    fn cancelling_one_request_leaves_its_neighbours_running() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let expected = sequential_ids(&decoder, &[4, 5], &greedy_params(6, 3));

        let (doomed_params, token) = cancellable_params(4000, 9);
        let doomed = {
            let batcher = batcher.clone();
            thread::spawn(move || batcher.generate(vec![1, 2, 3], doomed_params, None))
        };
        let survivor = {
            let batcher = batcher.clone();
            thread::spawn(move || batcher.generate(vec![4, 5], greedy_params(6, 3), None))
        };

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while batcher.stats().decode_steps < 3 {
            assert!(
                std::time::Instant::now() < deadline,
                "never started decoding"
            );
            thread::sleep(std::time::Duration::from_millis(1));
        }
        token.cancel();

        let (finish, _, _, _) = doomed.join().expect("no panic").expect("generate");
        assert_eq!(finish, FinishReason::Cancelled);
        let (finish, ids, _, _) = survivor.join().expect("no panic").expect("generate");
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(
            ids, expected,
            "an uncancelled request must produce exactly what it would have alone"
        );
    }

    // ----------------------------------------------------------------
    // Stop sequences (`sched-stop-buffering`)
    // ----------------------------------------------------------------

    /// Layer 1 in the batched path. A batched row and a row decoding on
    /// its own must agree about where an answer ends, so both go
    /// through the same `StopMatcher`.
    ///
    /// Confirmed to FAIL (runs to all 8 tokens) when the
    /// `stops.is_stop_token` check is removed from the worker's
    /// sampling step.
    #[test]
    fn a_token_level_stop_ends_a_batched_row() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
        assert!(baseline.len() > 1, "need something to stop before the end");
        let stop_token = baseline[1];

        let (finish, ids, _text, usage) = batcher
            .generate(
                vec![1, 2, 3],
                GenerationParams {
                    stop_token_ids: vec![stop_token],
                    ..greedy_params(8, 4)
                },
                None,
            )
            .expect("generate");

        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(
            ids,
            baseline[..1].to_vec(),
            "the stop token itself is not part of the answer"
        );
        assert_eq!(usage.completion_tokens, 1);
    }

    /// Layer 2 in the batched path: a stop string spanning several
    /// tokens is matched, and nothing past it is reported.
    #[test]
    fn a_multi_token_stop_string_truncates_a_batched_row() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
        let decode = identity_decode();
        let full = decode(&baseline);
        if full.chars().count() < 4 {
            return;
        }
        // Two characters out of the middle, so the match spans a token
        // boundary and the first character is a partial for a while.
        let cut: Vec<char> = full.chars().collect();
        let stop_str: String = cut[1..3].iter().collect();
        let expected: String = cut[..1].iter().collect();

        let (finish, _ids, text, _usage) = batcher
            .generate(
                vec![1, 2, 3],
                GenerationParams {
                    stop: vec![stop_str.clone()],
                    ..greedy_params(8, 4)
                },
                None,
            )
            .expect("generate");

        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(
            text, expected,
            "a stop spanning two tokens must cut where it starts"
        );
        assert!(!text.contains(&stop_str));
    }

    /// Text withheld against a stop that never arrives is ordinary
    /// output. Dropping it would truncate every answer whose tail looks
    /// like the start of a stop string.
    #[test]
    fn a_batched_row_that_never_matches_loses_no_output() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let baseline = sequential_ids(&decoder, &[1, 2, 3], &greedy_params(8, 4));
        let expected = identity_decode()(&baseline);

        let (finish, ids, text, _usage) = batcher
            .generate(
                vec![1, 2, 3],
                GenerationParams {
                    stop: vec!["ZZ_NEVER_MATCHES_ZZ".to_string()],
                    ..greedy_params(8, 4)
                },
                None,
            )
            .expect("generate");
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(ids, baseline);
        assert_eq!(
            text, expected,
            "buffering is about when text is released, never whether"
        );
    }

    /// A request nobody cancels must be untouched by the machinery that
    /// exists for the ones that are.
    #[test]
    fn an_uncancelled_request_is_unaffected_by_the_abort_path() {
        let decoder = tiny_decoder();
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            identity_decode(),
            BatcherConfig {
                prefill_chunk: 1,
                ..BatcherConfig::default()
            },
        );
        let (params, _token) = cancellable_params(6, 3);
        let (finish, ids, _, _) = batcher
            .generate(vec![4, 5], params, None)
            .expect("generate");
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(ids, sequential_ids(&decoder, &[4, 5], &greedy_params(6, 3)));
        assert_eq!(batcher.stats().aborted, 0);
    }
}
