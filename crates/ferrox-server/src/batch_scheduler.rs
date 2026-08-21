//! Continuous-batching decode scheduler: many in-flight sequences share
//! one `Decoder::forward_multi_seq` step per tick instead of each
//! request owning a private `forward_token` loop.
//!
//! Opt-in via `FERROX_CONTINUOUS_BATCHING=1`; mutually exclusive with
//! the KV pool and prefix cache (those paths keep the private-loop
//! `generate`). Stop sequences use the same pending-buffer logic as
//! `generate::sample_until_stop` (decode each new token, hold back
//! `longest_stop - 1` bytes, finish on match).
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
//! **Queue cap.** The job channel is unbounded, so without a cap a
//! client retry storm turns straight into unbounded memory: every
//! retry parks another prompt (and its reply channel) in the queue,
//! and the server's only signal that it is drowning is the RSS graph.
//! [`QueueGate`] bounds the *waiting* jobs -- in-flight sequences are
//! `FERROX_CB_MAX_SEQS`'s business -- and a refusal is a fast, cheap
//! 503 with `Retry-After` rather than a slow, expensive timeout.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use ferrox_core::cache::KvCache;
use ferrox_models::sampling::Sampler;
use ferrox_models::tokenizer::StopTokens;
use ferrox_models::Decoder;

use crate::generate::{
    earliest_stop_match, floor_char_boundary, DecodeError, FinishReason, GenerationParams, Usage,
};

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
}

impl Default for BatcherConfig {
    fn default() -> Self {
        BatcherConfig {
            max_seqs: usize::MAX,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            max_queue: DEFAULT_MAX_QUEUE,
        }
    }
}

impl BatcherConfig {
    pub fn from_env() -> Self {
        BatcherConfig {
            max_seqs: env_positive("FERROX_CB_MAX_SEQS").unwrap_or(usize::MAX),
            prefill_chunk: env_positive("FERROX_CB_PREFILL_CHUNK").unwrap_or(DEFAULT_PREFILL_CHUNK),
            max_queue: env_positive("FERROX_CB_MAX_QUEUE").unwrap_or(DEFAULT_MAX_QUEUE),
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
    stop_tokens: StopTokens,
    reply: Sender<JobResult>,
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

    /// Replies to and removes every row that has finished.
    fn flush_finished(&mut self) {
        let finished: Vec<Uid> = self
            .order
            .iter()
            .copied()
            .filter(|uid| self.state.get(uid).is_some_and(|s| s.finish.is_some()))
            .collect();
        for uid in finished {
            if let Some(slot) = self.remove(uid) {
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
    /// Detokenized text already safe to expose (past the stop hold-back).
    visible: String,
    /// Tail that might still complete a stop match.
    pending: String,
    prompt_tokens: usize,
    max_tokens: usize,
    stop_tokens: StopTokens,
    params: GenerationParams,
    reply: Sender<JobResult>,
    finish: Option<FinishReason>,
}

/// Owns a dedicated worker thread that batches decode steps. Cheap to
/// clone (`Sender` only); the worker stays alive as long as any clone
/// (or the original) exists.
#[derive(Clone)]
pub struct ContinuousBatcher {
    tx: Sender<Job>,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
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
        let worker_counters = Arc::clone(&counters);
        let worker_queue = Arc::clone(&queue);
        let _join = thread::Builder::new()
            .name("ferrox-continuous-batch".into())
            .spawn(move || worker_loop(decoder, decode, rx, config, worker_counters, worker_queue))
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
        }
    }

    /// Submit one generation job and block until it finishes. Safe to
    /// call from many `spawn_blocking` tasks concurrently -- they
    /// serialize only on the shared decode worker, which is the point.
    pub fn generate(
        &self,
        prompt_tokens: Vec<usize>,
        params: GenerationParams,
        stop_tokens: StopTokens,
    ) -> Result<(FinishReason, Vec<usize>, String, Usage), DecodeError> {
        // Refuse before allocating a queue slot for the prompt, so a
        // retry storm costs a rejection rather than memory.
        self.queue
            .try_reserve()
            .map_err(|queued| DecodeError::QueueFull {
                queued,
                cap: self.queue.cap,
            })?;
        let (reply_tx, reply_rx) = mpsc::channel();
        if self
            .tx
            .send(Job {
                prompt_tokens,
                params,
                stop_tokens,
                reply: reply_tx,
            })
            .is_err()
        {
            // The worker is gone, so nothing will ever dequeue this
            // reservation: release it here or the gate leaks a slot.
            self.queue.release();
            return Err(DecodeError::KvPoolExhausted);
        }
        reply_rx.recv().unwrap_or(Err(DecodeError::KvPoolExhausted))
    }
}

/// Accepts as many queued jobs as the in-flight cap allows, turning
/// each into a waiting `Prefill`. The cap counts prompts that are still
/// prefilling as well as rows already decoding: a prefilling prompt
/// holds a full set of KV caches, so not counting it would let the
/// worker exceed `max_seqs` by however many prompts happen to be in
/// flight.
fn drain_pending_jobs(
    decoder: &Arc<Decoder>,
    rx: &Receiver<Job>,
    prefills: &mut VecDeque<Prefill>,
    decoding: usize,
    config: &BatcherConfig,
    queue: &QueueGate,
) {
    while decoding + prefills.len() < config.max_seqs {
        match rx.try_recv() {
            Ok(job) => {
                queue.release();
                if let Some(prefill) = accept(decoder, job, config.prefill_chunk) {
                    prefills.push_back(prefill);
                }
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => break,
        }
    }
}

fn worker_loop(
    decoder: Arc<Decoder>,
    decode: DecodeFn,
    rx: Receiver<Job>,
    config: BatcherConfig,
    counters: Arc<Counters>,
    queue: Arc<QueueGate>,
) {
    let mut rows = Rows::default();
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    loop {
        // Only a completely idle worker blocks: with a prompt still
        // chunking there is always work to do on the next tick.
        if rows.is_empty() && prefills.is_empty() {
            match rx.recv() {
                Ok(job) => {
                    // Off the queue and into the scheduler: free the
                    // slot the submitter reserved.
                    queue.release();
                    if let Some(prefill) = accept(&decoder, job, config.prefill_chunk) {
                        prefills.push_back(prefill);
                    }
                }
                Err(_) => break,
            }
        }
        drain_pending_jobs(&decoder, &rx, &mut prefills, rows.len(), &config, &queue);

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
            rows.flush_finished();
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
            if slot.stop_tokens.contains(next) {
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

        rows.flush_finished();
    }
}

/// Appends `piece` into the stop-sequence pending buffer. Returns true
/// when a stop matched and the slot should leave the active batch.
fn apply_stop_buffer(slot: &mut Slot, piece: &str) -> bool {
    if slot.params.stop.is_empty() {
        slot.visible.push_str(piece);
        return false;
    }
    slot.pending.push_str(piece);
    if let Some(cut) = earliest_stop_match(&slot.pending, &slot.params.stop) {
        slot.visible.push_str(&slot.pending[..cut]);
        slot.pending.clear();
        slot.finish = Some(FinishReason::Stop);
        return true;
    }
    let max_stop_len = slot.params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
    let hold_back = max_stop_len.saturating_sub(1);
    if slot.pending.len() > hold_back {
        let boundary = floor_char_boundary(&slot.pending, slot.pending.len() - hold_back);
        if boundary > 0 {
            slot.visible.push_str(&slot.pending[..boundary]);
            slot.pending.drain(..boundary);
        }
    }
    false
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
    stop_tokens: StopTokens,
    reply: Sender<JobResult>,
}

impl Prefill {
    fn into_slot(self) -> Slot {
        let Prefill {
            state,
            prompt_tokens,
            params,
            stop_tokens,
            reply,
        } = self;
        let (caches, logits, pos) = state.into_decode_start();
        Slot {
            caches,
            pos,
            logits,
            sampler: Sampler::new(params.seed),
            generated_ids: Vec::with_capacity(params.max_tokens),
            visible: String::new(),
            pending: String::new(),
            prompt_tokens,
            max_tokens: params.max_tokens,
            stop_tokens,
            params,
            reply,
            finish: None,
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
        stop_tokens: job.stop_tokens,
        reply: job.reply,
    })
}

/// Sends one finished row's result to its own waiting caller. Takes the
/// `Slot` by value, so a row's reply channel travels with its state and
/// cannot be paired with another row's output.
fn reply_finished(mut slot: Slot) {
    let finish = slot.finish.expect("only a finished row is replied to");
    if !slot.pending.is_empty() {
        slot.visible.push_str(&slot.pending);
        slot.pending.clear();
    }
    let usage = Usage::new(slot.prompt_tokens, slot.generated_ids.len());
    let _ = slot
        .reply
        .send(Ok((finish, slot.generated_ids, slot.visible, usage)));
}

#[cfg(test)]
mod tests {
    use super::*;
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
                json_object: params[i].json_object,
                cancel: params[i].cancel.clone(),
            };
            threads.push(thread::spawn(move || {
                barrier.wait();
                let out = batcher
                    .generate(prompt, par, StopTokens::default())
                    .expect("batch generate");
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
            .generate(prompt, params, StopTokens::default())
            .expect("batch generate");
        assert_eq!(finish, FinishReason::Stop);
        assert!(
            !text.contains(&stop),
            "stop string must be trimmed from visible text: text={text:?} stop={stop:?}"
        );
        assert_eq!(&full[..full.find(&stop).unwrap()], text);
    }
    /// The continuous batcher carried the same single `eos_id` every
    /// other server decode loop did, so a Llama-3 or gemma checkpoint
    /// served through it ran past its own turn ender to `max_tokens`.
    /// Here the stop set holds the third token this prompt would
    /// otherwise generate and nothing else: a loop honouring the set
    /// stops with exactly two tokens, one comparing against a lone
    /// metadata EOS runs all 32.
    #[test]
    fn continuous_batch_stops_on_any_member_of_the_stop_set() {
        let decoder = tiny_decoder();
        let decode: DecodeFn = Arc::new(|_: &[usize]| String::new());
        let prompt = vec![1usize, 2, 3];
        let params = greedy_params(32, 3);
        let ids = sequential_ids(&decoder, &prompt, &params);
        assert!(ids.len() > 3, "need a mid-stream token to stop on");
        let turn_ender = ids[2];

        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            decode,
            BatcherConfig {
                prefill_chunk: 2,
                ..BatcherConfig::default()
            },
        );
        let (finish, got, _text, usage) = batcher
            .generate(prompt, params, StopTokens::from_eos(Some(turn_ender)))
            .expect("batch generate");
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(got, ids[..2].to_vec());
        assert_eq!(usage.completion_tokens, 2);
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
            thread::spawn(move || {
                batcher.generate(vec![1, 2], greedy_params(90, 5), StopTokens::default())
            })
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
            thread::spawn(move || {
                batcher.generate(long_prompt, greedy_params(1, 9), StopTokens::default())
            })
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
                        .generate(prompt, greedy_params(6, seed), StopTokens::default())
                        .expect("generate")
                        .1
                })
            })
            .collect();
        let got: Vec<Vec<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got[0], expected[0]);
        assert_eq!(got[1], expected[1]);
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
                pending: String::new(),
                prompt_tokens: 0,
                max_tokens,
                stop_tokens: StopTokens::default(),
                params,
                reply: tx,
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
        rows.flush_finished();
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
                    batcher
                        .generate(prompt, params, StopTokens::default())
                        .expect("generate")
                        .1
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
            .generate(vec![1, 2, 3], greedy_params(4, 1), StopTokens::default())
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
                .generate(vec![1, 2, 3], greedy_params(2, 1), StopTokens::default())
                .expect("a cap of 1 still serves requests one after another");
        }
        assert_eq!(batcher.stats().queue_depth, 0);
        assert_eq!(batcher.stats().queue_rejected, 0);
    }
}
