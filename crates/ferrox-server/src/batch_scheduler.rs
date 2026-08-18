//! Continuous-batching decode scheduler: many in-flight sequences share
//! one `Decoder::forward_multi_seq` step per tick instead of each
//! request owning a private `forward_token` loop.
//!
//! Prefill stays per-sequence (sequential `forward_token`) for this
//! first wiring -- continuous batching's win is the decode phase, where
//! membership can change every step. Opt-in via
//! `FERROX_CONTINUOUS_BATCHING=1`; mutually exclusive with the KV pool
//! and prefix cache (those paths keep the private-loop `generate`).
//! Stop sequences use the same pending-buffer logic as
//! `generate::sample_until_stop` (decode each new token, hold back
//! `longest_stop - 1` bytes, finish on match).
//!
//! **Prefill priority (Crane-inspired):** when the job channel has
//! pending work and active decode slots already exist, the worker drains
//! `try_recv` and runs prefill (`admit`) before the next batched decode
//! step. If any pending job was admitted and no slot is finishing, one
//! decode tick is skipped so new prompts are not starved behind an
//! in-flight batch.
//!
//! **`FERROX_CB_MAX_SEQS`:** optional cap on concurrent in-flight
//! sequences (default: unlimited). At the cap, new jobs stay queued in
//! the channel until a slot frees; only an empty worker blocks on
//! `recv`.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use ferrox_core::cache::KvCache;
use ferrox_models::sampling::Sampler;
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

struct Job {
    prompt_tokens: Vec<usize>,
    params: GenerationParams,
    eos_id: Option<usize>,
    reply: Sender<JobResult>,
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
    eos_id: Option<usize>,
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
}

struct WorkerGuard {
    _join: JoinHandle<()>,
}

impl ContinuousBatcher {
    /// Spawns the worker. Holds `decoder` and a detokenize callback for
    /// the worker's lifetime. Returns the shareable handle; the worker
    /// exits when the last `ContinuousBatcher` clone is dropped.
    pub fn spawn(decoder: Arc<Decoder>, decode: DecodeFn) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let _join = thread::Builder::new()
            .name("ferrox-continuous-batch".into())
            .spawn(move || worker_loop(decoder, decode, rx))
            .expect("spawn continuous-batch worker");
        // Detach join handle intentionally: dropping the last Sender
        // closes `rx` and ends the loop. Keep a process-lifetime leak
        // of the JoinHandle via Box::leak so dropping batcher clones
        // does not join (callers may still be mid-generate).
        let _guard: &'static WorkerGuard = Box::leak(Box::new(WorkerGuard { _join }));
        ContinuousBatcher { tx }
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
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Job {
                prompt_tokens,
                params,
                eos_id,
                reply: reply_tx,
            })
            .map_err(|_| DecodeError::KvPoolExhausted)?;
        reply_rx.recv().unwrap_or(Err(DecodeError::KvPoolExhausted))
    }
}

fn max_concurrent_seqs() -> usize {
    std::env::var("FERROX_CB_MAX_SEQS")
        .ok()
        .map(|v| {
            v.parse()
                .expect("FERROX_CB_MAX_SEQS must be a positive integer")
        })
        .unwrap_or(usize::MAX)
}

fn drain_pending_jobs(
    decoder: &Decoder,
    rx: &Receiver<Job>,
    slots: &mut Vec<Slot>,
    max_seqs: usize,
) -> bool {
    if slots.len() >= max_seqs {
        return false;
    }
    let mut admitted = false;
    while slots.len() < max_seqs {
        match rx.try_recv() {
            Ok(job) => {
                if let Some(slot) = admit(decoder, job) {
                    slots.push(slot);
                    admitted = true;
                }
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => break,
        }
    }
    admitted
}

fn worker_loop(decoder: Arc<Decoder>, decode: DecodeFn, rx: Receiver<Job>) {
    let max_seqs = max_concurrent_seqs();
    let mut slots: Vec<Slot> = Vec::new();
    loop {
        if slots.is_empty() {
            match rx.recv() {
                Ok(job) => {
                    if let Some(slot) = admit(&decoder, job) {
                        slots.push(slot);
                    }
                }
                Err(_) => break,
            }
        }
        let admitted_pending = drain_pending_jobs(&decoder, &rx, &mut slots, max_seqs);
        if slots.is_empty() {
            continue;
        }

        // Prefill-priority: defer one batched decode step when we just
        // admitted queued work and every slot is still actively decoding.
        if admitted_pending && slots.iter().all(|s| s.finish.is_none()) {
            continue;
        }

        let mut step_indices: Vec<usize> = Vec::new();
        for (i, slot) in slots.iter().enumerate() {
            if slot.finish.is_none() && slot.generated_ids.len() < slot.max_tokens {
                step_indices.push(i);
            }
        }

        if step_indices.is_empty() {
            flush_finished(&mut slots);
            continue;
        }

        let mut drop_mask = vec![false; step_indices.len()];
        for (si, &idx) in step_indices.iter().enumerate() {
            let slot = &mut slots[idx];
            let next =
                slot.sampler
                    .sample(&slot.logits, &slot.params.sampling, &slot.generated_ids);
            if Some(next) == slot.eos_id {
                slot.finish = Some(FinishReason::Stop);
                drop_mask[si] = true;
                continue;
            }
            slot.generated_ids.push(next);
            let piece = decode(&[next]);
            if apply_stop_buffer(slot, &piece) {
                drop_mask[si] = true;
            }
        }

        let active: Vec<usize> = step_indices
            .iter()
            .enumerate()
            .filter(|(si, _)| !drop_mask[*si])
            .map(|(_, &idx)| idx)
            .collect();

        if !active.is_empty() {
            let tokens: Vec<usize> = active
                .iter()
                .map(|&idx| *slots[idx].generated_ids.last().unwrap())
                .collect();
            let positions: Vec<usize> = active.iter().map(|&idx| slots[idx].pos).collect();
            let mut cache_refs: Vec<Vec<KvCache>> = active
                .iter()
                .map(|&idx| std::mem::take(&mut slots[idx].caches))
                .collect();
            let logits_batch = decoder.forward_multi_seq(&tokens, &positions, &mut cache_refs);
            for (j, &idx) in active.iter().enumerate() {
                slots[idx].caches = std::mem::take(&mut cache_refs[j]);
                slots[idx].logits = logits_batch[j].clone();
                slots[idx].pos += 1;
                if slots[idx].generated_ids.len() >= slots[idx].max_tokens {
                    slots[idx].finish = Some(FinishReason::Length);
                }
            }
        }

        flush_finished(&mut slots);
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

fn admit(decoder: &Decoder, job: Job) -> Option<Slot> {
    let vocab_size = decoder.config.vocab_size;
    if let Some(&bad) = job.prompt_tokens.iter().find(|&&t| t >= vocab_size) {
        let _ = job.reply.send(Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        }));
        return None;
    }

    let mut caches: Vec<KvCache> = decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();
    let mut pos = 0usize;
    let logits = if job.prompt_tokens.is_empty() {
        let l = decoder.forward_token(0, pos, &mut caches);
        pos += 1;
        l
    } else {
        let mut l = Vec::new();
        for &tok in &job.prompt_tokens {
            l = decoder.forward_token(tok, pos, &mut caches);
            pos += 1;
        }
        l
    };

    Some(Slot {
        caches,
        pos,
        logits,
        sampler: Sampler::new(job.params.seed),
        generated_ids: Vec::with_capacity(job.params.max_tokens),
        visible: String::new(),
        pending: String::new(),
        prompt_tokens: job.prompt_tokens.len(),
        max_tokens: job.params.max_tokens,
        eos_id: job.eos_id,
        params: job.params,
        reply: job.reply,
        finish: None,
    })
}

fn flush_finished(slots: &mut Vec<Slot>) {
    let mut i = 0;
    while i < slots.len() {
        if slots[i].finish.is_some() {
            let mut slot = slots.swap_remove(i);
            let finish = slot.finish.unwrap();
            if !slot.pending.is_empty() {
                slot.visible.push_str(&slot.pending);
                slot.pending.clear();
            }
            let usage = Usage::new(slot.prompt_tokens, slot.generated_ids.len());
            let _ = slot
                .reply
                .send(Ok((finish, slot.generated_ids, slot.visible, usage)));
        } else {
            i += 1;
        }
    }
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

        let batcher = ContinuousBatcher::spawn(Arc::clone(&decoder), identity_decode());
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

        let batcher = ContinuousBatcher::spawn(Arc::clone(&decoder), decode);
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
}
