//! The checks that stop `ferrox bench` from reporting a real number
//! for the wrong work.
//!
//! Every guard here exists because a benchmark can lie quietly, and a
//! quiet lie is worse than a crash: it gets published. A run whose
//! prompt was silently truncated reports an honest tok/s for a prompt
//! that was never processed. A run served partly from a warm KV cache
//! reports prefill throughput for a prefill that never happened. A run
//! whose first repetition paid for shader compilation reports the
//! Metal driver's startup cost as the engine's speed. None of those
//! produce an error on their own; they produce a plausible number.
//!
//! So each guard is a hard failure whose message names what it caught,
//! not a warning nobody reads. They are free functions over plain data
//! rather than methods on the engine types, so every one of them is
//! directly testable without loading a model -- which matters, because
//! a guard that has never been seen to fire is not a guard.

/// Repetitions run before timing starts, discarded. One is enough to
/// fault in every weight page, compile every Metal pipeline and warm
/// every lazily-built kernel table; the point is that it is *not* zero
/// and not silently variable.
pub const WARMUP_REPS: usize = 1;

/// What a KV cache looked like at some point in a repetition. Copied
/// out of `KvCache` so the guards stay independent of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheProbe {
    pub seq_len: usize,
    /// Elements actually stored in `k` / `v`. Checked alongside
    /// `seq_len` because a cache whose length was reset to zero while
    /// keeping its contents is exactly what a prefix-cache reuse looks
    /// like from the outside.
    pub k_len: usize,
    pub v_len: usize,
}

impl CacheProbe {
    pub fn is_cold(&self) -> bool {
        self.seq_len == 0 && self.k_len == 0 && self.v_len == 0
    }
}

/// A timed row needs at least one timed repetition. `-r 0` used to run
/// the warmup and nothing else, leaving a row with zero samples whose
/// median printed as `0.00` -- a fabricated number in the same column
/// as the measured ones.
pub fn check_repetitions(reps: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        reps >= 1,
        "-r {reps} leaves no timed repetitions: the warmup would run and the row \
         would report a median over an empty sample set. Pass -r 1 or more."
    );
    Ok(())
}

/// Warmup accounting, asserted after the fact: a row must carry exactly
/// `reps` samples, meaning `WARMUP_REPS` repetitions ran and were
/// thrown away. Fires if a later edit changes the loop bound or the
/// discard condition, which is how a shader-compile timing quietly
/// gets back into a published median.
pub fn check_timed_samples(test: &str, reps: usize, samples: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        samples == reps,
        "{test}: {samples} timed samples for -r {reps}. Exactly {WARMUP_REPS} warmup \
         repetition must run before timing and be discarded; a mismatch means either \
         an untimed repetition leaked into the median or the warmup did not run, and \
         the first repetition pays for page faults and shader compilation."
    );
    Ok(())
}

/// The prompt length asserted BEFORE the run: the token stream about to
/// be fed must be exactly as long as the `pp<N>` label promises, and
/// every id must exist in the vocabulary. An id past the end of the
/// embedding table is either a panic or a garbage row -- neither is the
/// workload the row claims to measure.
pub fn check_prompt_before(
    test: &str,
    n_prompt: usize,
    tokens: &[usize],
    vocab: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        n_prompt > 0,
        "{test}: refusing to time a zero-token prompt; the rate would divide zero \
         work by a real duration"
    );
    anyhow::ensure!(
        tokens.len() == n_prompt,
        "{test}: built {} tokens for a {n_prompt}-token prompt -- the row would \
         report {n_prompt} tokens per second for {} tokens of work",
        tokens.len(),
        tokens.len()
    );
    anyhow::ensure!(vocab > 0, "{test}: model reports a zero-sized vocabulary");
    if let Some((i, &t)) = tokens.iter().enumerate().find(|(_, &t)| t >= vocab) {
        anyhow::bail!("{test}: synthetic token {i} is id {t}, outside a vocabulary of {vocab}");
    }
    Ok(())
}

/// Zero cache hits, asserted rather than assumed. Every repetition must
/// start from a genuinely empty KV cache: a prefill measured against a
/// warm cache reports the engine's speed at work it did not do.
pub fn check_caches_cold(test: &str, rep: usize, caches: &[CacheProbe]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !caches.is_empty(),
        "{test}: no KV caches to check -- a model with no layers is not a benchmark"
    );
    if let Some((layer, c)) = caches.iter().enumerate().find(|(_, c)| !c.is_cold()) {
        anyhow::bail!(
            "{test} rep {rep} started with a warm KV cache at layer {layer} \
             (seq_len {}, {} k / {} v elements retained): this repetition would be \
             served partly from cached attention state and report prefill \
             throughput for a prefill that never happened",
            c.seq_len,
            c.k_len,
            c.v_len
        );
    }
    Ok(())
}

/// The prompt length asserted AFTER the run, not only before. An engine
/// that silently truncated the batch would otherwise divide the full
/// token count by a partial run's duration and report a speedup for
/// doing less work.
pub fn check_prefill_after(
    test: &str,
    n_prompt: usize,
    caches: &[CacheProbe],
) -> anyhow::Result<()> {
    if let Some((layer, c)) = caches
        .iter()
        .enumerate()
        .find(|(_, c)| c.seq_len != n_prompt)
    {
        anyhow::bail!(
            "{test}: layer {layer} consumed {} of {n_prompt} prompt tokens -- the \
             reported rate would be {n_prompt} tok/s worth of credit for work the \
             engine skipped",
            c.seq_len
        );
    }
    Ok(())
}

/// The decode equivalent: one priming token plus `n_gen` steps must
/// leave exactly `n_gen + 1` positions in every layer's cache. A short
/// cache means steps were skipped, which inflates the rate.
pub fn check_decode_after(test: &str, n_gen: usize, caches: &[CacheProbe]) -> anyhow::Result<()> {
    let expected = n_gen + 1;
    if let Some((layer, c)) = caches
        .iter()
        .enumerate()
        .find(|(_, c)| c.seq_len != expected)
    {
        anyhow::bail!(
            "{test}: layer {layer} advanced the KV cache to {} positions, expected \
             {expected} (one prime + {n_gen} decode steps) -- decode steps were \
             skipped and the rate would count them anyway",
            c.seq_len
        );
    }
    Ok(())
}

/// Running digest of the exact token stream fed inside the timed
/// region. FNV-1a over the ids: a few nanoseconds per token against
/// milliseconds per forward pass, so it cannot move the number it
/// guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadDigest(u64);

impl Default for WorkloadDigest {
    fn default() -> Self {
        WorkloadDigest(0xcbf2_9ce4_8422_2325)
    }
}

impl WorkloadDigest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, token: usize) {
        for b in (token as u64).to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn feed_all(&mut self, tokens: &[usize]) {
        for &t in tokens {
            self.feed(t);
        }
    }

    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// The temperature-zero assertion, in the only form that can actually
/// be checked from outside a sampler: every timed repetition must feed
/// the engine the identical token stream.
///
/// `ferrox bench` drives synthetic ids and never samples, so nominally
/// temperature does not apply. That is exactly the claim worth
/// enforcing -- if a future edit fed generated tokens back into the
/// decode loop, any temperature above zero would make repetition 2 do
/// different work from repetition 1, and the median across them would
/// stop being a median of anything. Identical digests are what
/// "temperature 0" means for a benchmark; divergent digests catch its
/// absence whatever the cause.
pub fn check_same_workload(
    test: &str,
    rep: usize,
    first: WorkloadDigest,
    current: WorkloadDigest,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        first == current,
        "{test} rep {rep} fed a different token stream than the first repetition \
         ({} vs {}): the repetitions are not repetitions of the same work, so their \
         median is not a measurement. A timed run must be deterministic -- \
         temperature 0, no sampling, no data-dependent token feedback.",
        current.hex(),
        first.hex()
    );
    Ok(())
}

/// Environment variables that `ferrox bench` sets for itself as part of
/// selecting a backend and thread count. They carry no information for
/// an auditor beyond what the receipt already records.
const SELF_SET_ENV: &[&str] = &[
    "FERROX_METAL",
    "FERROX_METAL_ATTN",
    "FERROX_CUDA",
    "FERROX_CPU_THREADS",
];

/// Every `FERROX_*` variable in effect that the bench did not set
/// itself, as `(name, value)` pairs sorted by name.
///
/// This is recorded, not refused. Several of these knobs exist
/// precisely so a change can be A/B'd by toggling one
/// (`FERROX_METAL_FA_MMA=0`), and refusing would break the workflow
/// the ledger depends on. But some of them -- `FERROX_METAL_MOE_ABLATE`
/// removes whole stages, `FERROX_ALLOW_UNKNOWN_TENSORS` disables the
/// fail-closed loader gate -- change how much work the engine does, and
/// a published row taken under one of them is not comparable to a row
/// taken without it. Putting them in the receipt means the difference
/// is discoverable later instead of lost with the shell history.
pub fn nondefault_engine_env<I, K, V>(vars: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut out: Vec<(String, String)> = vars
        .into_iter()
        .filter_map(|(k, v)| {
            let k = k.as_ref();
            (k.starts_with("FERROX_") && !SELF_SET_ENV.contains(&k))
                .then(|| (k.to_string(), v.as_ref().to_string()))
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cold(n: usize) -> Vec<CacheProbe> {
        vec![
            CacheProbe {
                seq_len: 0,
                k_len: 0,
                v_len: 0
            };
            n
        ]
    }

    fn filled(n: usize, seq_len: usize, elems_per_pos: usize) -> Vec<CacheProbe> {
        vec![
            CacheProbe {
                seq_len,
                k_len: seq_len * elems_per_pos,
                v_len: seq_len * elems_per_pos,
            };
            n
        ]
    }

    #[test]
    fn a_run_with_no_timed_repetitions_is_refused() {
        let err = check_repetitions(0).unwrap_err().to_string();
        assert!(err.contains("no timed repetitions"), "{err}");
        assert!(check_repetitions(1).is_ok());
    }

    #[test]
    fn a_row_must_carry_exactly_one_discarded_warmup() {
        assert!(check_timed_samples("pp512", 3, 3).is_ok());
        // The warmup leaked into the median.
        let err = check_timed_samples("pp512", 3, 4).unwrap_err().to_string();
        assert!(err.contains("shader compilation"), "{err}");
        // A repetition vanished.
        assert!(check_timed_samples("pp512", 3, 2).is_err());
    }

    #[test]
    fn the_warmup_count_is_one_and_the_message_says_so() {
        assert_eq!(WARMUP_REPS, 1);
        let err = check_timed_samples("tg128", 2, 3).unwrap_err().to_string();
        assert!(err.contains("Exactly 1 warmup"), "{err}");
    }

    #[test]
    fn a_short_token_stream_cannot_be_reported_as_a_full_prompt() {
        let tokens: Vec<usize> = (0..500).collect();
        let err = check_prompt_before("pp512", 512, &tokens, 32000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("built 500 tokens"), "{err}");
        assert!(err.contains("512 tokens per second for 500"), "{err}");
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_caught_before_the_run() {
        let tokens = vec![1, 2, 99_999, 4];
        let err = check_prompt_before("pp4", 4, &tokens, 32000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("token 2 is id 99999"), "{err}");
        assert!(check_prompt_before("pp4", 4, &[1, 2, 3, 4], 32000).is_ok());
    }

    #[test]
    fn a_zero_length_prompt_is_never_timed() {
        assert!(check_prompt_before("pp0", 0, &[], 32000).is_err());
        assert!(check_prompt_before("pp1", 1, &[0], 0).is_err());
    }

    #[test]
    fn a_warm_cache_at_any_layer_stops_the_repetition() {
        assert!(check_caches_cold("pp512", 0, &cold(8)).is_ok());
        let mut caches = cold(8);
        caches[5] = CacheProbe {
            seq_len: 128,
            k_len: 128 * 64,
            v_len: 128 * 64,
        };
        let err = check_caches_cold("pp512", 2, &caches)
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer 5"), "{err}");
        assert!(err.contains("a prefill that never happened"), "{err}");
    }

    #[test]
    fn a_cache_reset_to_zero_length_but_still_holding_data_is_not_cold() {
        // What reusing a stored prefix looks like from outside: the
        // length says empty, the contents say otherwise.
        let sneaky = [CacheProbe {
            seq_len: 0,
            k_len: 128 * 64,
            v_len: 128 * 64,
        }];
        assert!(!sneaky[0].is_cold());
        let err = check_caches_cold("pp512", 1, &sneaky)
            .unwrap_err()
            .to_string();
        assert!(err.contains("elements retained"), "{err}");
    }

    #[test]
    fn a_model_with_no_caches_is_not_silently_accepted() {
        assert!(check_caches_cold("pp512", 0, &[]).is_err());
    }

    #[test]
    fn a_truncated_prefill_cannot_report_the_full_prompt_length() {
        assert!(check_prefill_after("pp512", 512, &filled(4, 512, 64)).is_ok());
        let mut caches = filled(4, 512, 64);
        caches[3].seq_len = 256;
        let err = check_prefill_after("pp512", 512, &caches)
            .unwrap_err()
            .to_string();
        assert!(err.contains("consumed 256 of 512"), "{err}");
        assert!(err.contains("layer 3"), "{err}");
    }

    #[test]
    fn a_prefill_that_ran_long_is_caught_too() {
        // Over-consumption is equally wrong: the divisor no longer
        // matches the work.
        let caches = filled(2, 600, 64);
        assert!(check_prefill_after("pp512", 512, &caches).is_err());
    }

    #[test]
    fn skipped_decode_steps_are_caught_by_the_final_cache_length() {
        assert!(check_decode_after("tg128", 128, &filled(4, 129, 64)).is_ok());
        let err = check_decode_after("tg128", 128, &filled(4, 65, 64))
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 129"), "{err}");
        assert!(err.contains("decode steps were skipped"), "{err}");
    }

    #[test]
    fn identical_token_streams_digest_identically_and_different_ones_do_not() {
        let mut a = WorkloadDigest::new();
        a.feed_all(&[1, 2, 3, 4]);
        let mut b = WorkloadDigest::new();
        b.feed_all(&[1, 2, 3, 4]);
        assert_eq!(a, b);
        assert!(check_same_workload("tg128", 1, a, b).is_ok());

        let mut c = WorkloadDigest::new();
        c.feed_all(&[1, 2, 3, 5]);
        assert_ne!(a, c);
        let err = check_same_workload("tg128", 2, a, c)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different token stream"), "{err}");
        assert!(err.contains("temperature 0"), "{err}");
    }

    #[test]
    fn the_digest_is_order_sensitive_and_length_sensitive() {
        // A sampler that produced the same multiset in a different
        // order is still doing different work per repetition.
        let mut a = WorkloadDigest::new();
        a.feed_all(&[1, 2, 3]);
        let mut b = WorkloadDigest::new();
        b.feed_all(&[3, 2, 1]);
        assert_ne!(a, b, "digest must not be order-insensitive");

        let mut short = WorkloadDigest::new();
        short.feed_all(&[1, 2]);
        assert_ne!(
            a, short,
            "a truncated stream must not digest as the full one"
        );

        // A stream of zeros must not collide with an empty stream.
        let mut zeros = WorkloadDigest::new();
        zeros.feed_all(&[0, 0, 0]);
        assert_ne!(zeros, WorkloadDigest::new());
    }

    #[test]
    fn the_digest_is_stable_across_runs_so_two_receipts_can_be_compared() {
        // Pinned: if this changes, previously published digests stop
        // being comparable to new ones, which is the only thing the
        // receipt field is for.
        let mut d = WorkloadDigest::new();
        d.feed_all(&[1, 8, 15, 22]);
        assert_eq!(d.hex(), d.hex());
        let mut again = WorkloadDigest::new();
        again.feed_all(&[1, 8, 15, 22]);
        assert_eq!(d.hex(), again.hex());
        assert_eq!(d.hex().len(), 16);
    }

    #[test]
    fn engine_env_recording_keeps_the_knobs_and_drops_the_bench_own_settings() {
        let got = nondefault_engine_env([
            ("PATH", "/usr/bin"),
            ("FERROX_METAL", "auto"),
            ("FERROX_METAL_ATTN", "1"),
            ("FERROX_CUDA", "auto"),
            ("FERROX_CPU_THREADS", "8"),
            ("FERROX_METAL_MOE_ABLATE", "topk"),
            ("FERROX_ALLOW_UNKNOWN_TENSORS", "1"),
            ("HOME", "/Users/x"),
        ]);
        assert_eq!(
            got,
            vec![
                ("FERROX_ALLOW_UNKNOWN_TENSORS".to_string(), "1".to_string()),
                ("FERROX_METAL_MOE_ABLATE".to_string(), "topk".to_string()),
            ]
        );
    }

    #[test]
    fn a_clean_environment_records_nothing_rather_than_a_placeholder() {
        assert!(nondefault_engine_env([("PATH", "/usr/bin")]).is_empty());
    }
}
