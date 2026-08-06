//! Shared token-generation loop for both the non-streaming and SSE
//! streaming `/v1/chat/completions` paths: sampling
//! (temperature/top-p/top-k/repetition penalty) with a greedy-argmax
//! path at `temperature<=0.0`,
//! plus stop-sequence handling that's correct even when a stop string
//! spans more than one generated token.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ferrox_core::cache::{KvBlockPool, KvCache, KvPoolExhausted as CacheKvPoolExhausted};
use ferrox_models::sampling::{Sampler, SamplingParams};
use ferrox_models::{Decoder, Engine, PrefixCache, TextTokenizer};

use crate::json_mode::mask_logits_for_json;
use crate::model::ServerTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("prompt encoded to token id {token}, which is outside this model's vocabulary of {vocab_size} (its tokenizer does not match this checkpoint)")]
    TokenOutOfVocab { token: usize, vocab_size: usize },
    #[error("server is at capacity: the shared KV cache block pool has no free blocks for a new request; retry shortly")]
    KvPoolExhausted,
}

/// A shared `KvBlockPool` plus this server's admission-control wait
/// policy: how long a request is willing to retry acquiring its
/// per-layer caches before giving up, when the pool is momentarily
/// exhausted. `queue_wait: Duration::ZERO` (the default) means "try
/// once, reject immediately" -- the original reject-only behavior.
#[derive(Clone)]
pub struct KvPoolConfig {
    pub pool: Arc<Mutex<KvBlockPool>>,
    pub queue_wait: Duration,
}

/// Retries acquiring one `KvCache` per layer from `config.pool` until
/// either all of them succeed or `config.queue_wait` has elapsed since
/// the first attempt. Sleeping between attempts happens on whichever
/// thread calls this -- fine here since generation already runs on
/// tokio's blocking-thread pool (`spawn_blocking`), not an async
/// reactor thread that a `std::thread::sleep` would otherwise stall.
///
/// `max_seq_len` (this request's real worst-case sequence length --
/// prompt length plus `max_tokens`) is passed straight through to
/// `KvCache::with_pool`, so each layer's cache reserves enough blocks
/// for the *whole* request up front rather than growing mid-decode.
/// This isn't just an optimization: `Decoder::forward_token`/
/// `forward_batch` treat `KvCache::push` as infallible, so a pooled
/// cache that under-reserves at construction and then fails to
/// acquire another block later (because some other request took the
/// pool's remaining capacity in the meantime) would panic mid-decode
/// instead of failing this request cleanly at admission time -- caught
/// by a real panic during live testing before this fix.
fn acquire_pooled_caches(
    decoder: &Decoder,
    config: &KvPoolConfig,
    max_seq_len: usize,
) -> Result<Vec<KvCache>, CacheKvPoolExhausted> {
    let deadline = Instant::now() + config.queue_wait;
    loop {
        let attempt: Result<Vec<KvCache>, CacheKvPoolExhausted> = decoder
            .layers
            .iter()
            .map(|_| {
                KvCache::with_pool(
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                    Arc::clone(&config.pool),
                    max_seq_len,
                )
            })
            .collect();
        let now = Instant::now();
        if attempt.is_ok() || now >= deadline {
            return attempt;
        }
        std::thread::sleep(Duration::from_millis(10).min(deadline - now));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// OpenAI-convention token accounting, reported in the response's
/// `usage` field. Counted from the exact token ids the generation loop
/// processed (prompt after BOS insertion, and every generated id), not
/// re-tokenized after the fact -- re-tokenizing decoded text is not
/// guaranteed to round-trip to the same count.
/// Token counts for a completed generation. Rates are optional llama.cpp-
/// style fields filled when the caller timed prefill vs decode.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Prefill throughput (prompt tokens / prefill seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    /// Decode throughput (completion tokens / decode seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_per_second: None,
            predicted_per_second: None,
        }
    }

    pub fn with_timings(mut self, prompt_secs: f64, predicted_secs: f64) -> Self {
        if prompt_secs > 0.0 && self.prompt_tokens > 0 {
            self.prompt_per_second = Some(self.prompt_tokens as f64 / prompt_secs);
        }
        if predicted_secs > 0.0 && self.completion_tokens > 0 {
            self.predicted_per_second = Some(self.completion_tokens as f64 / predicted_secs);
        }
        self
    }
}

#[derive(Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub seed: u64,
    pub stop: Vec<String>,
    /// When true, constrain sampling toward JSON-safe token pieces and
    /// validate the emitted text is a JSON object (best-effort; see
    /// `json_mode` module).
    pub json_object: bool,
}

fn chunked_prefill_tokens() -> Option<usize> {
    std::env::var("FERROX_CHUNKED_PREFILL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
}

fn cpu_kv_offload_enabled() -> bool {
    matches!(
        std::env::var("FERROX_CPU_KV_OFFLOAD").ok().as_deref(),
        Some("1")
    )
}

/// Batched prompt prefill, optionally split into `FERROX_CHUNKED_PREFILL`-sized
/// chunks that append into the same KV caches.
fn forward_prompt_batch(
    decoder: &Decoder,
    tokens: &[usize],
    start_pos: usize,
    caches: &mut [KvCache],
) -> Vec<f32> {
    if let Some(chunk) = chunked_prefill_tokens() {
        let mut pos = start_pos;
        let mut last = Vec::new();
        for part in tokens.chunks(chunk) {
            let rows = decoder.forward_batch(part, pos, caches);
            pos += part.len();
            last = rows.into_iter().last().unwrap_or_default();
        }
        last
    } else {
        let rows = decoder.forward_batch(tokens, start_pos, caches);
        rows.into_iter()
            .last()
            .expect("forward_batch returns one logits row per prompt token")
    }
}

/// Runs the prompt through `decoder`, then generates up to
/// `params.max_tokens` new tokens, calling `emit` with each newly-safe-
/// to-flush chunk of decoded text as it becomes available (see the
/// stop-sequence buffering note below). Returns the reason generation
/// stopped.
///
/// Stop-sequence correctness: a stop string can span more than one
/// generated token (e.g. stop=" END" while the tokenizer emits " "
/// and "END" as separate pieces), so text can't simply be flushed
/// token-by-token as soon as it's decoded -- the tail end of the
/// buffer might still turn into part of a stop match once the next
/// token arrives. This holds back the last `longest_stop_len - 1`
/// bytes of decoded text (respecting UTF-8 char boundaries) until
/// they're confirmed clean, the same buffering approach real
/// inference servers use for this exact reason.
#[allow(clippy::too_many_arguments)] // one clear parameter per concern; a
                                     // bundling struct here would just be GenerationParams's fields plus
                                     // decoder/tokenizer/eos_id/bos_id/prompt/kv_pool/prefix_cache/emit
                                     // re-wrapped for no real benefit at this call depth (two call sites,
                                     // both in this crate).
pub fn generate(
    decoder: &Decoder,
    tokenizer: &ServerTokenizer,
    eos_id: Option<usize>,
    bos_id: Option<usize>,
    prompt: &str,
    params: &GenerationParams,
    kv_pool: Option<&KvPoolConfig>,
    prefix_cache: Option<&Mutex<PrefixCache>>,
    mut emit: impl FnMut(&str),
) -> Result<(FinishReason, Usage), DecodeError> {
    let vocab_size = decoder.config.vocab_size;

    // Metal greedy GPU argmax: fold final_norm+lm_head+argmax into the
    // dense-stack CB and download one token id instead of hidden/vocab.
    // Thread-local so concurrent Arc<Decoder> requests do not race.
    #[cfg(feature = "metal")]
    let _metal_greedy_guard = {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                ferrox_models::set_metal_greedy_argmax(false);
            }
        }
        if params.sampling.temperature <= 0.0 && ferrox_models::metal_greedy_gpu_enabled() {
            ferrox_models::set_metal_greedy_argmax(true);
            Some(Guard)
        } else {
            None
        }
    };

    let mut tokens = tokenizer.encode(prompt);
    if let Some(bos) = bos_id {
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    let prompt_tokens = tokens.len();
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        return Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        });
    }

    // With a shared pool configured, admission control happens here:
    // each layer's cache reserves enough blocks up front for this
    // request's real worst case (prompt length + max_tokens), not just
    // one block -- see `acquire_pooled_caches`'s doc comment for why
    // under-reserving here would let a request panic mid-decode
    // instead of failing cleanly at admission time. If any layer can't
    // get that many blocks, `acquire_pooled_caches` retries (bounded by
    // `config.queue_wait`) before the request is rejected. Each failed
    // attempt's partial `Vec` is dropped immediately (via `collect`),
    // releasing any blocks it did acquire through `KvCache`'s `Drop`
    // impl before the next retry, so a request that ultimately gives up
    // leaves the pool exactly as it found it.
    //
    // Prefix-cache restoration only applies when there's no shared KV
    // block pool: a restored cache is a plain, unpooled clone (see
    // `KvCache`'s `Clone` doc comment), so combining it with pool-based
    // admission control would let a request's real memory usage
    // silently bypass the pool's bounded-memory guarantee. Not
    // supported together yet.
    let max_seq_len = tokens.len() + params.max_tokens;
    let restored = if kv_pool.is_none() {
        prefix_cache.and_then(|pc| {
            let m = pc
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .find_longest_prefix(&tokens);
            (m.matched_len > 0).then_some(m)
        })
    } else {
        None
    };

    let prefill_start = std::time::Instant::now();
    let mut pos;
    let mut logits: Vec<f32>;
    let mut caches: Vec<KvCache>;
    if let Some(m) = restored {
        caches = m
            .kv_caches
            .expect("matched_len > 0 always carries kv_caches");
        let suffix = &tokens[m.matched_len..];
        if suffix.is_empty() {
            if let Some(pl) = m.pending_logits {
                // The whole query was already processed and stored
                // verbatim before (e.g. an exact-repeat prompt with
                // unseeded sampling, so the whole-response cache
                // couldn't serve it) -- zero forward passes needed.
                pos = m.matched_len;
                logits = pl;
            } else {
                // Rare: our query exactly matches a strict PREFIX of a
                // longer stored entry (someone else's conversation
                // continued past this point), so there's no stored
                // "what comes next" for our shorter query. Back the
                // restored cache off by one position and reprocess
                // just that last matched token to get real logits,
                // rather than guessing.
                let back_to = m.matched_len - 1;
                for c in caches.iter_mut() {
                    c.truncate(back_to);
                }
                pos = back_to;
                logits = decoder.forward_token(tokens[back_to], pos, &mut caches);
                pos += 1;
            }
        } else {
            pos = m.matched_len;
            let mut l = Vec::new();
            for &tok in suffix {
                l = decoder.forward_token(tok, pos, &mut caches);
                pos += 1;
            }
            logits = l;
        }
    } else {
        caches = match kv_pool {
            Some(config) => acquire_pooled_caches(decoder, config, max_seq_len)
                .map_err(|_| DecodeError::KvPoolExhausted)?,
            None => decoder
                .layers
                .iter()
                .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
                .collect(),
        };
        // Process the prompt once, capturing the *last* call's logits
        // (which already predict the first generated token) instead of
        // discarding them. A prompt of zero tokens (bos_id unset and an
        // empty-string prompt encoding to nothing) has no real token to
        // seed from, so a synthetic id 0 bootstraps decoding.
        pos = 0;
        logits = if tokens.is_empty() {
            let l = decoder.forward_token(0, pos, &mut caches);
            pos += 1;
            l
        } else {
            // Batched prefill: one pass over the prompt (shared weight
            // traffic on CPU; fewer per-token bookkeeping costs). Last
            // row's logits predict the first generated token — same as
            // the sequential loop this replaces (see unit test below).
            // When `FERROX_CHUNKED_PREFILL` is set, split long prompts
            // into chunks that reuse the same KV caches.
            pos = tokens.len();
            forward_prompt_batch(decoder, &tokens, 0, &mut caches)
        };
    }

    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let decode_start = std::time::Instant::now();
    #[cfg(feature = "metal")]
    let kv_offload = cpu_kv_offload_enabled();
    let decode_token = |id: usize| tokenizer.decode(&[id]);

    // `logits` becomes the prediction for the position after each
    // generated token; every generated token gets exactly one
    // corresponding cache entry via the `step` closure below, matching
    // `prefix_cache`'s `pending_logits` expectations regardless of
    // whether the loop goes on to hit a stop sequence or max_tokens.
    let (finish, generated_ids, final_logits) = sample_until_stop(
        logits,
        pos,
        eos_id,
        params,
        |ids| tokenizer.decode(ids),
        |next, pos| {
            let l = decoder.forward_token(next, pos, &mut caches);
            #[cfg(feature = "metal")]
            if kv_offload {
                decoder.sync_metal_attn_kv_to_host(&mut caches);
            }
            l
        },
        &mut emit,
        if params.json_object {
            Some(&decode_token as &dyn Fn(usize) -> String)
        } else {
            None
        },
    );
    let decode_secs = decode_start.elapsed().as_secs_f64();
    logits = final_logits;
    let usage =
        Usage::new(prompt_tokens, generated_ids.len()).with_timings(prefill_secs, decode_secs);

    // Store the full sequence this request actually processed (prompt
    // plus everything generated) so a future request sharing this
    // prefix -- the common multi-turn-chat case, where each turn's
    // prompt is the previous turn's full prompt+reply plus a little
    // more -- can skip recomputing it. `caches`/`logits` are exactly
    // in the right state for this: `logits` predicts whatever would
    // come after this sequence (the token that triggered an EOS/stop
    // match, if generation stopped that way, or the natural next
    // prediction if it ran to `max_tokens`), and every token in
    // `tokens`/`generated_ids` has exactly one corresponding cache
    // entry -- see the per-token push above. Skipped whenever a KV
    // pool is configured, for the same reason restoration is (see this
    // function's earlier comment).
    //
    // Metal dense-stack decode may leave host KvCache lagging the
    // Metal-resident KV; flush before storing so prefix restore gets
    // complete K/V.
    if kv_pool.is_none() {
        if let Some(pc) = prefix_cache {
            // Greedy Metal argmax returns a 1-element "logits" vec; that is
            // not a full pending distribution and must not be stored for
            // later (possibly non-greedy) prefix restores.
            if logits.len() == vocab_size {
                #[cfg(feature = "metal")]
                decoder.sync_metal_attn_kv_to_host(&mut caches);
                tokens.extend(generated_ids);
                pc.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .store(tokens, caches, logits);
            }
        }
    }

    Ok((finish, usage))
}

/// Shared sampling + stop-sequence-aware emission loop, given already-
/// primed `logits`/`pos` (the prompt has already been processed by the
/// caller) and a `step` closure that advances one position and returns
/// the new logits. Used by both `generate` (the GGUF path, whose own
/// KV-pool/prefix-cache-aware prompt priming stays separate above) and
/// `generate_engine` (any other `Engine`, with simpler priming and no
/// pooling/restoration) so the actual sampling/stop-sequence
/// correctness -- the part most worth not duplicating -- lives in
/// exactly one place. Returns the finish reason, the generated token
/// ids, and the final logits (the prediction for whatever would come
/// next), since `generate`'s prefix-cache storage needs both.
fn sample_until_stop(
    mut logits: Vec<f32>,
    mut pos: usize,
    eos_id: Option<usize>,
    params: &GenerationParams,
    mut decode_one: impl FnMut(&[usize]) -> String,
    mut step: impl FnMut(usize, usize) -> Vec<f32>,
    mut emit: impl FnMut(&str),
    decode_token: Option<&dyn Fn(usize) -> String>,
) -> (FinishReason, Vec<usize>, Vec<f32>) {
    let max_stop_len = params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut sampler = Sampler::new(params.seed);
    let mut generated_ids: Vec<usize> = Vec::with_capacity(params.max_tokens);
    let mut pending = String::new();
    let mut finish = FinishReason::Length;

    for _ in 0..params.max_tokens {
        let next = if params.json_object {
            if let Some(decode_token) = decode_token {
                let mut mask_fn = |scores: &mut [f32]| {
                    mask_logits_for_json(scores, |i| decode_token(i));
                };
                sampler.sample_with_mask(
                    &logits,
                    &params.sampling,
                    &generated_ids,
                    Some(&mut mask_fn),
                )
            } else {
                sampler.sample(&logits, &params.sampling, &generated_ids)
            }
        } else {
            sampler.sample(&logits, &params.sampling, &generated_ids)
        };
        if Some(next) == eos_id {
            finish = FinishReason::Stop;
            break;
        }
        generated_ids.push(next);
        logits = step(next, pos);
        pos += 1;

        pending.push_str(&decode_one(&[next]));

        if let Some(cut) = earliest_stop_match(&pending, &params.stop) {
            emit(&pending[..cut]);
            finish = FinishReason::Stop;
            pending.clear();
            break;
        }

        let hold_back = max_stop_len.saturating_sub(1);
        if pending.len() > hold_back {
            let boundary = floor_char_boundary(&pending, pending.len() - hold_back);
            if boundary > 0 {
                emit(&pending[..boundary]);
                pending.drain(..boundary);
            }
        }
    }

    if !pending.is_empty() {
        emit(&pending);
    }

    (finish, generated_ids, logits)
}

/// The generic counterpart to `generate`, for any `Engine` other than
/// `Decoder` (in practice, Kimi K3's `KimiEngine`). Deliberately
/// simpler: no KV block pool, no `PrefixCache` restoration -- Kimi's
/// KDA state is a fixed-size recurrent matrix that collapses history
/// irreversibly, so it can't support the truncate/restore operations
/// those features need (see `ferrox_models::engine`'s module docs). Every
/// request processes its full prompt from scratch against fresh
/// engine state.
pub fn generate_engine<E: Engine, T: TextTokenizer>(
    engine: &E,
    tokenizer: &T,
    eos_id: Option<usize>,
    bos_id: Option<usize>,
    prompt: &str,
    params: &GenerationParams,
    mut emit: impl FnMut(&str),
) -> Result<(FinishReason, Usage), DecodeError> {
    let vocab_size = engine.vocab_size();
    let mut tokens = tokenizer.encode(prompt);
    if let Some(bos) = bos_id {
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    let prompt_tokens = tokens.len();
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        return Err(DecodeError::TokenOutOfVocab {
            token: bad,
            vocab_size,
        });
    }

    let mut state = engine.new_state();
    let mut pos = 0;
    let logits = if tokens.is_empty() {
        let l = engine.forward_token(0, pos, &mut state);
        pos += 1;
        l
    } else {
        let mut l = Vec::new();
        for &tok in tokens.iter() {
            l = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        l
    };

    let (finish, generated_ids, _final_logits) = sample_until_stop(
        logits,
        pos,
        eos_id,
        params,
        |ids| tokenizer.decode(ids),
        |next, pos| engine.forward_token(next, pos, &mut state),
        &mut emit,
        None,
    );

    Ok((finish, Usage::new(prompt_tokens, generated_ids.len())))
}

/// The earliest byte offset in `text` at which any of `stops` begins,
/// or `None` if none match yet.
pub(crate) fn earliest_stop_match(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// The largest char boundary `<= idx`. `str::floor_char_boundary` is
/// still nightly-only in stable Rust as of this writing; this is the
/// same walk-backward-to-a-boundary logic on stable.
pub(crate) fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::config::test_dense_fixture;

    fn small_decoder() -> Decoder {
        Decoder::new_random_small(test_dense_fixture(), 2, 256)
    }

    fn greedy_params(max_tokens: usize) -> GenerationParams {
        GenerationParams {
            max_tokens,
            sampling: SamplingParams::default(),
            seed: 1,
            stop: Vec::new(),
            json_object: false,
        }
    }

    /// Regression test for a real bug caught by close reading, not by
    /// any earlier test (the earlier tests only checked `generate`
    /// against a test helper that replicated the same buggy pattern,
    /// so they never could have caught it): `generate`'s original
    /// prompt-processing loop pushed every prompt token into the KV
    /// cache once (correct), then its first generation-loop iteration
    /// re-processed the *last* prompt token a second time via another
    /// `forward_token` call at the wrong position (`tokens.len()`
    /// instead of its real position `tokens.len() - 1`) just to obtain
    /// logits -- silently duplicating that token in the cache with a
    /// different RoPE rotation applied, corrupting every subsequent
    /// position's attention. Fixed by capturing the prompt loop's own
    /// last-iteration logits instead of discarding and re-deriving
    /// them. This locks in the fixed pattern (which `generate` uses
    /// internally) against `forward_batch`'s independent ground truth.
    #[test]
    fn prompt_processing_matches_forward_batch_ground_truth_with_no_duplicate_position() {
        let decoder = small_decoder();
        let tokens = vec![1usize, 2, 3, 4];

        let mut fresh_caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let batch_logits = decoder.forward_batch(&tokens, 0, &mut fresh_caches);
        let ground_truth_next_logits = batch_logits.last().unwrap().clone();

        // The exact pattern `generate` now uses: one forward_token call
        // per prompt token, keeping the last call's logits.
        let mut caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            logits = decoder.forward_token(tok, pos, &mut caches);
        }

        assert_eq!(
            caches[0].seq_len, fresh_caches[0].seq_len,
            "must not push any position beyond the real prompt length"
        );
        assert_eq!(
            logits, ground_truth_next_logits,
            "logits predicting the first generated token must match forward_batch's independent computation exactly"
        );
    }

    /// End-to-end version of the same property: `generate`'s full
    /// greedy decode loop (prompt processing + iterative generation)
    /// must produce exactly the token sequence an independent
    /// step-by-step computation (via `forward_batch` for the prompt,
    /// then `forward_token` once per new position, argmax at each
    /// step) would produce. Restricted to ASCII byte values so
    /// `ServerTokenizer::Byte`'s `decode` is lossless in both
    /// directions and the generated text can be compared back to
    /// token ids exactly.
    #[test]
    fn generate_greedy_output_matches_independent_step_by_step_computation() {
        let decoder = small_decoder();
        let prompt_ids = vec![1usize, 2, 3];
        let prompt = String::from_utf8(prompt_ids.iter().map(|&b| b as u8).collect()).unwrap();
        let max_tokens = 8;

        // Independent computation: forward_batch over the prompt, then
        // one forward_token + argmax per new position, decoding each
        // generated id one at a time and concatenating -- exactly
        // `generate`'s own decode granularity (`ServerTokenizer::Byte`
        // is lossy per non-ASCII byte, so decoding token-by-token vs.
        // decoding the whole sequence at once are not equivalent; this
        // must replicate the real call pattern, not just the ids).
        let mut caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut logits = decoder
            .forward_batch(&prompt_ids, 0, &mut caches)
            .pop()
            .unwrap();
        let mut pos = prompt_ids.len();
        let mut expected_text = String::new();
        for _ in 0..max_tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            expected_text.push_str(&ServerTokenizer::Byte.decode(&[next]));
            logits = decoder.forward_token(next, pos, &mut caches);
            pos += 1;
        }

        let mut actual_text = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(max_tokens),
            None,
            None,
            |s| actual_text.push_str(s),
        )
        .unwrap();

        assert_eq!(actual_text, expected_text);
    }

    #[test]
    fn rejects_out_of_vocab_prompt_tokens() {
        // ByteTokenizer emits raw bytes; vocab 32 makes ASCII letters OOV.
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 32);
        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            "hello",
            &greedy_params(4),
            None,
            None,
            |_| {},
        );
        assert!(matches!(result, Err(DecodeError::TokenOutOfVocab { .. })));
    }

    #[test]
    fn greedy_generation_hits_length_without_eos() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let mut chunks = String::new();
        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            |s| chunks.push_str(s),
        )
        .unwrap();
        assert_eq!(finish, FinishReason::Length);
    }

    /// Discovers the real greedy-argmax next-token id after `prompt_ids`
    /// via `forward_batch` (ground truth: its last returned row predicts
    /// the token immediately after the full prompt, exactly what
    /// `generate` computes as its first generation-loop `logits` value)
    /// -- not by decoding it to text and reading bytes back, which is
    /// lossy for `ByteTokenizer`: a standalone byte >= 128 is not valid
    /// UTF-8 on its own, so `String::from_utf8_lossy` replaces it with
    /// the 3-byte U+FFFD replacement character, and reading
    /// `s.bytes().next()` off that recovers 0xEF (239), not the
    /// original token id.
    fn greedy_next_token_after(decoder: &Decoder, prompt_ids: &[usize]) -> usize {
        let mut caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let logits = decoder
            .forward_batch(prompt_ids, 0, &mut caches)
            .pop()
            .unwrap();
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn eos_token_stops_generation_before_max_tokens() {
        let decoder = small_decoder();
        let prompt_ids = vec![1usize, 2];
        let prompt = String::from_utf8(prompt_ids.iter().map(|&b| b as u8).collect()).unwrap();

        // ByteTokenizer::encode is a lossless direct byte->id mapping
        // (only decode is lossy, see greedy_next_token_after's doc
        // comment), so `generate`'s internal prompt replay reaches
        // exactly the same state as this direct computation.
        let eos = greedy_next_token_after(&decoder, &prompt_ids);

        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            Some(eos),
            None,
            &prompt,
            &greedy_params(50),
            None,
            None,
            |_| {},
        )
        .unwrap();
        assert_eq!(
            finish,
            FinishReason::Stop,
            "generation must stop as soon as the greedy-chosen token matches eos_id, not run to max_tokens"
        );
    }

    #[test]
    fn a_stop_sequence_that_never_matches_does_not_drop_any_generated_content() {
        // The hold-back buffering (needed so a stop sequence spanning
        // more than one token is never partially flushed) must not
        // silently swallow output when no stop sequence ever matches:
        // the final emitted text must be byte-for-byte identical to an
        // otherwise-identical run with no stop sequences configured at
        // all, since the buffering is purely about *when* text is
        // flushed, never *whether* it is.
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8]).unwrap();

        let mut baseline = String::new();
        let (baseline_finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(20),
            None,
            None,
            |s| baseline.push_str(s),
        )
        .unwrap();

        let mut with_unmatchable_stop = String::new();
        let (stop_finish, _usage2) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &GenerationParams {
                max_tokens: 20,
                sampling: SamplingParams::default(),
                seed: 1,
                stop: vec!["ZZ_NEVER_MATCHES_ZZ".to_string()],
                json_object: false,
            },
            None,
            None,
            |s| with_unmatchable_stop.push_str(s),
        )
        .unwrap();

        assert_eq!(baseline_finish, FinishReason::Length);
        assert_eq!(stop_finish, FinishReason::Length);
        assert_eq!(with_unmatchable_stop, baseline);
    }

    #[test]
    fn a_stop_sequence_that_does_match_truncates_output_before_it() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8]).unwrap();

        // Discover what greedy decode actually produces, then use a
        // substring of it (starting after the first character, so at
        // least one character of real output precedes the match) as a
        // stop sequence guaranteed to match.
        let mut baseline = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(20),
            None,
            None,
            |s| baseline.push_str(s),
        )
        .unwrap();
        let Some((cut, _)) = baseline.char_indices().nth(1) else {
            // Degenerate case for this decoder/seed: fewer than 2 chars
            // generated: nothing meaningful to truncate, skip.
            return;
        };
        let stop_str = baseline[cut..].to_string();
        if stop_str.is_empty() {
            return;
        }

        let mut truncated = String::new();
        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &GenerationParams {
                max_tokens: 20,
                sampling: SamplingParams::default(),
                seed: 1,
                stop: vec![stop_str],
                json_object: false,
            },
            None,
            None,
            |s| truncated.push_str(s),
        )
        .unwrap();

        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(truncated, baseline[..cut]);
    }

    #[test]
    fn prefix_cache_reuses_a_shared_prefix_and_produces_the_same_output_as_a_fresh_run() {
        let decoder = small_decoder();
        let prompt1 = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));

        let mut out1 = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt1,
            &greedy_params(5),
            None,
            Some(&pc),
            |s| out1.push_str(s),
        )
        .unwrap();
        assert_eq!(pc.lock().unwrap().stats().misses, 1);

        // prompt2's tokens (raw bytes, via ByteTokenizer) start with
        // prompt1's exact bytes -- the common multi-turn-chat shape.
        let prompt2 = String::from_utf8(vec![1u8, 2, 3, 9, 9]).unwrap();

        let mut out2_with_cache = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt2,
            &greedy_params(5),
            None,
            Some(&pc),
            |s| out2_with_cache.push_str(s),
        )
        .unwrap();
        let stats = pc.lock().unwrap().stats();
        assert_eq!(stats.hits, 1, "prompt2 must hit the stored prompt1 entry");
        assert_eq!(stats.total_positions_reused, 3);

        let mut out2_fresh = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt2,
            &greedy_params(5),
            None,
            None,
            |s| out2_fresh.push_str(s),
        )
        .unwrap();

        assert_eq!(
            out2_with_cache, out2_fresh,
            "restoring from the prefix cache must produce identical output to processing the whole prompt from scratch"
        );
    }

    #[test]
    fn prefix_cache_exact_repeat_skips_prompt_processing_via_pending_logits() {
        let decoder = small_decoder();
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));

        let mut out1 = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            None,
            Some(&pc),
            |s| out1.push_str(s),
        )
        .unwrap();

        // The exact same prompt again: the stored entry's `tokens` (the
        // full prompt+completion from the first call) starts with this
        // exact prompt, so this only matches the prompt-length prefix
        // of a longer stored entry, not an exact full-entry match --
        // covering the "no pending_logits available" fallback path,
        // not the zero-forward-pass shortcut. Still must produce
        // identical output to a from-scratch run.
        let mut out2_with_cache = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            None,
            Some(&pc),
            |s| out2_with_cache.push_str(s),
        )
        .unwrap();

        let mut out2_fresh = String::new();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            None,
            None,
            |s| out2_fresh.push_str(s),
        )
        .unwrap();

        assert_eq!(out2_with_cache, out2_fresh);
    }

    #[test]
    fn prefix_cache_is_not_consulted_when_a_kv_pool_is_configured() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2, 3]).unwrap();
        let pc = Mutex::new(PrefixCache::new(4));
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool, Duration::ZERO);

        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            Some(&pc),
            |_| {},
        )
        .unwrap();
        generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            Some(&pc),
            |_| {},
        )
        .unwrap();

        let stats = pc.lock().unwrap().stats();
        assert_eq!(
            stats.hits + stats.misses,
            0,
            "prefix cache must never be consulted while a KV pool is configured"
        );
    }

    fn pool_config(pool: Arc<Mutex<KvBlockPool>>, queue_wait: Duration) -> KvPoolConfig {
        KvPoolConfig { pool, queue_wait }
    }

    #[test]
    fn generate_succeeds_with_a_pool_that_has_enough_blocks() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let mut out = String::new();
        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            |s| out.push_str(s),
        )
        .unwrap();
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "every acquired block must be released once the request finishes"
        );
    }

    /// Regression test for a real bug caught by live testing (not by
    /// any unit test): with a small `block_size`, a request whose
    /// prompt + max_tokens exceeds one block used to reserve only one
    /// block per layer at admission time, then panic deep inside
    /// `Decoder::forward_token` once decode outgrew that block and the
    /// pool had nothing left to grow into (`KvCache::push` returning
    /// `Err` where `forward_token` assumes it can't fail). Fixed by
    /// having `acquire_pooled_caches` reserve blocks for the whole
    /// worst-case sequence length up front. This test must not panic --
    /// it must either succeed cleanly or fail at admission with
    /// `KvPoolExhausted`, never partway through decode.
    #[test]
    fn generate_reserves_enough_blocks_up_front_for_a_sequence_spanning_multiple_blocks() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap(); // 2 tokens via ByteTokenizer
        let max_tokens = 10;
        let block_size = 2;
        // prompt (2) + max_tokens (10) = 12 positions -> 6 blocks/layer * 2 layers = 12 blocks.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 12)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(max_tokens),
            Some(&config),
            None,
            |_| {},
        )
        .unwrap();
        assert_eq!(finish, FinishReason::Length);
        assert_eq!(pool.lock().unwrap().free_blocks(), 12);
    }

    #[test]
    fn generate_fails_at_admission_not_mid_decode_when_the_pool_cannot_cover_the_worst_case() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let max_tokens = 10;
        let block_size = 2;
        // One block short of the 12 the worst case (see the test
        // above) actually needs.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 11)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(max_tokens),
            Some(&config),
            None,
            |_| {},
        );
        assert!(matches!(result, Err(DecodeError::KvPoolExhausted)));
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            11,
            "a rejected request must leave the pool exactly as it found it"
        );
    }

    #[test]
    fn generate_rejects_the_request_without_leaking_blocks_when_the_pool_is_too_small() {
        let decoder = small_decoder(); // 2 layers -> needs 2 blocks, one per layer's cache
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 1)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            |_| {},
        );
        assert!(matches!(result, Err(DecodeError::KvPoolExhausted)));
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            1,
            "a rejected request must leave the pool exactly as it found it"
        );
    }

    #[test]
    fn generate_releases_blocks_so_back_to_back_requests_do_not_starve_the_pool() {
        let decoder = small_decoder(); // 2 layers
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        // Just enough for one request's caches at a time -- a second,
        // concurrent request would be rejected, but a *sequential*
        // second request must succeed once the first has returned its
        // blocks.
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));
        let config = pool_config(pool.clone(), Duration::ZERO);

        for _ in 0..3 {
            let (finish, _usage) = generate(
                &decoder,
                &ServerTokenizer::Byte,
                None,
                None,
                &prompt,
                &greedy_params(5),
                Some(&config),
                None,
                |_| {},
            )
            .unwrap();
            assert_eq!(finish, FinishReason::Length);
        }
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);
    }

    #[test]
    fn generate_with_zero_queue_wait_rejects_immediately() {
        let decoder = small_decoder(); // 2 layers -> needs 2 blocks
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 1))); // too small for 2 layers
        let config = pool_config(pool, Duration::ZERO);

        let started = Instant::now();
        let result = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            |_| {},
        );
        assert!(matches!(result, Err(DecodeError::KvPoolExhausted)));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "queue_wait=0 must reject on the first attempt, not retry: took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn generate_with_a_queue_wait_succeeds_once_another_holder_releases_its_blocks() {
        let decoder = small_decoder(); // 2 layers, needs 2 blocks
        let prompt = String::from_utf8(vec![1u8, 2]).unwrap();
        let pool = Arc::new(Mutex::new(KvBlockPool::new(64, 2)));

        // Hold both blocks on another thread for a short while, then
        // release them -- simulating another in-flight request that's
        // about to finish.
        let holder_pool = pool.clone();
        let holder = std::thread::spawn(move || {
            let mut held = KvCache::with_pool(1, 1, holder_pool.clone(), 0).unwrap();
            held.push(&[0.0], &[0.0]).unwrap(); // crosses into needing the second block
            std::thread::sleep(Duration::from_millis(80));
            drop(held); // returns both blocks to the pool
        });
        // Give the holder a moment to actually acquire before we try.
        std::thread::sleep(Duration::from_millis(15));

        let config = pool_config(pool.clone(), Duration::from_millis(500));
        let (finish, _usage) = generate(
            &decoder,
            &ServerTokenizer::Byte,
            None,
            None,
            &prompt,
            &greedy_params(5),
            Some(&config),
            None,
            |_| {},
        )
        .unwrap();
        assert_eq!(
            finish,
            FinishReason::Length,
            "a sufficiently long queue_wait must let the request succeed once the holder releases"
        );
        holder.join().unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);
    }

    #[test]
    fn earliest_stop_match_finds_the_leftmost_match_across_multiple_stops() {
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string(), "hello".to_string()]),
            Some(0)
        );
        assert_eq!(
            earliest_stop_match("hello world", &["nope".to_string()]),
            None
        );
    }
}
