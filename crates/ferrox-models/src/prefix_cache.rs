//! KV-prefix caching: when a new request's tokens share a leading
//! subsequence with a previously processed request, skip recomputing
//! the KV state for that shared prefix entirely, restoring it from a
//! stored snapshot instead of running `forward_batch` over tokens
//! that were already processed.
//!
//! This is the harder sibling of `ferrox-server::cache::ResponseCache`
//! (which only helps *exact*-repeat requests): prefix caching helps
//! any request that *starts with* something seen before, which is the
//! common case for multi-turn chat (each turn's full prompt is the
//! previous turn's prompt plus a little more) even when no single
//! request repeats exactly.
//!
//! Deliberately scoped: this does a linear scan over a small,
//! LRU-bounded set of stored prefixes to find the longest common
//! prefix, not a trie/radix-tree structure (vLLM's and SGLang's
//! RadixAttention do this properly at production scale). For the
//! small number of concurrent conversations a demo server actually
//! handles, a linear scan is simpler and correctness is easier to
//! verify.

use ferrox_core::cache::KvCache;

/// A stored snapshot: the tokens processed so far, the resulting
/// per-layer KV cache state, and the logits that predict the token
/// immediately after `tokens` (needed so a request that matches this
/// prefix *exactly* -- no new tokens at all -- doesn't need any
/// computation to know what to generate next).
#[derive(Clone)]
struct StoredPrefix {
    tokens: Vec<usize>,
    kv_caches: Vec<KvCache>,
    pending_logits: Vec<f32>,
}

/// LRU-bounded store of `StoredPrefix` snapshots, searched for the
/// longest common prefix with an incoming token sequence.
pub struct PrefixCache {
    entries: Vec<StoredPrefix>,
    max_entries: usize,
    hits_positions_reused: u64,
    hits_count: u64,
    misses_count: u64,
}

/// What was found (or not) for an incoming token sequence.
pub struct PrefixMatch {
    /// How many leading tokens matched a stored prefix (0 if none).
    pub matched_len: usize,
    /// Restored KV cache state covering exactly `matched_len`
    /// positions, ready to continue from. `None` if `matched_len == 0`.
    pub kv_caches: Option<Vec<KvCache>>,
    /// Logits predicting the token at position `matched_len`, valid
    /// only when `matched_len > 0`.
    pub pending_logits: Option<Vec<f32>>,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        PrefixCache {
            entries: Vec::new(),
            max_entries,
            hits_positions_reused: 0,
            hits_count: 0,
            misses_count: 0,
        }
    }

    /// Finds the stored prefix with the longest common leading
    /// subsequence with `tokens`, and returns a ready-to-use clone of
    /// its KV state truncated to exactly that common length (a stored
    /// prefix may itself be longer than the common part, if a later,
    /// different continuation was stored under it -- the KV cache is
    /// truncated to the matching length before being handed back, so
    /// the caller never sees state from a divergent continuation).
    pub fn find_longest_prefix(&mut self, tokens: &[usize]) -> PrefixMatch {
        let mut best: Option<(usize, &StoredPrefix)> = None;
        for entry in &self.entries {
            let common = common_prefix_len(&entry.tokens, tokens);
            if common > 0 && best.map(|(len, _)| common > len).unwrap_or(true) {
                best = Some((common, entry));
            }
        }

        match best {
            Some((matched_len, entry)) => {
                self.hits_count += 1;
                self.hits_positions_reused += matched_len as u64;

                let mut kv_caches = entry.kv_caches.clone();
                for cache in kv_caches.iter_mut() {
                    cache.truncate(matched_len);
                }

                // The stored pending_logits predict the token
                // immediately after entry.tokens' FULL length. They're
                // only valid to hand back if the match covers that
                // entire stored sequence (matched_len ==
                // entry.tokens.len()); a partial match into the middle
                // of a longer stored sequence means the caller is
                // asking about position `matched_len`, not
                // `entry.tokens.len()`, and reusing the stored logits
                // there would silently answer the wrong question.
                let pending_logits = if matched_len == entry.tokens.len() {
                    Some(entry.pending_logits.clone())
                } else {
                    None
                };

                PrefixMatch {
                    matched_len,
                    kv_caches: Some(kv_caches),
                    pending_logits,
                }
            }
            None => {
                self.misses_count += 1;
                PrefixMatch {
                    matched_len: 0,
                    kv_caches: None,
                    pending_logits: None,
                }
            }
        }
    }

    /// Stores a snapshot for `tokens` (all tokens processed so far,
    /// prompt plus any generated continuation) with the given KV cache
    /// state and next-token logits, evicting the least-recently-stored
    /// entry if already at capacity.
    pub fn store(&mut self, tokens: Vec<usize>, kv_caches: Vec<KvCache>, pending_logits: Vec<f32>) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(StoredPrefix {
            tokens,
            kv_caches,
            pending_logits,
        });
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hits: self.hits_count,
            misses: self.misses_count,
            entries: self.entries.len(),
            total_positions_reused: self.hits_positions_reused,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub total_positions_reused: u64,
}

fn common_prefix_len(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cache(seq_len: usize) -> KvCache {
        let mut cache = KvCache::new(1, 1);
        for i in 0..seq_len {
            cache.push(&[i as f32], &[i as f32 * 10.0]).unwrap();
        }
        cache
    }

    #[test]
    fn empty_cache_always_misses() {
        let mut cache = PrefixCache::new(4);
        let m = cache.find_longest_prefix(&[1, 2, 3]);
        assert_eq!(m.matched_len, 0);
        assert!(m.kv_caches.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn exact_prefix_match_returns_full_length_and_pending_logits() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![0.1, 0.2]);

        let m = cache.find_longest_prefix(&[1, 2, 3]);
        assert_eq!(m.matched_len, 3);
        assert!(m.kv_caches.is_some());
        assert_eq!(m.pending_logits, Some(vec![0.1, 0.2]));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn extended_request_matches_the_shared_prefix_length() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3, 4, 5], vec![dummy_cache(5)], vec![9.9]);

        // New request extends the stored one with two more tokens.
        let m = cache.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            m.matched_len, 5,
            "must match the full stored prefix, not just a partial one"
        );
        assert_eq!(m.pending_logits, Some(vec![9.9]));
    }

    #[test]
    fn partial_divergent_match_returns_only_the_common_length_and_no_stale_logits() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3, 4, 5], vec![dummy_cache(5)], vec![9.9]);

        // Diverges after the first 3 tokens.
        let m = cache.find_longest_prefix(&[1, 2, 3, 9, 9]);
        assert_eq!(m.matched_len, 3);
        assert!(
            m.kv_caches.is_some(),
            "a real KV-state saving still exists for the matched prefix"
        );
        assert!(
            m.pending_logits.is_none(),
            "stored pending_logits predicted the token after the FULL stored sequence, not after the partial match point -- must not be reused here"
        );
    }

    #[test]
    fn no_common_prefix_at_all_is_a_clean_miss() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![1.0]);
        let m = cache.find_longest_prefix(&[9, 8, 7]);
        assert_eq!(m.matched_len, 0);
    }

    #[test]
    fn picks_the_longest_match_among_several_stored_entries() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![1, 2, 3, 4], vec![dummy_cache(4)], vec![0.0]);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![0.0]);

        let m = cache.find_longest_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(
            m.matched_len, 4,
            "the longest stored prefix that's actually a prefix of the query must win"
        );
    }

    #[test]
    fn evicts_oldest_entry_when_at_capacity() {
        let mut cache = PrefixCache::new(2);
        cache.store(vec![1, 1], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![2, 2], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![3, 3], vec![dummy_cache(2)], vec![0.0]); // evicts [1,1]

        assert_eq!(
            cache.find_longest_prefix(&[1, 1]).matched_len,
            0,
            "oldest entry must have been evicted"
        );
        assert_eq!(cache.find_longest_prefix(&[2, 2]).matched_len, 2);
        assert_eq!(cache.find_longest_prefix(&[3, 3]).matched_len, 2);
    }

    #[test]
    fn stats_track_positions_reused_not_just_hit_count() {
        let mut cache = PrefixCache::new(4);
        cache.store(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![dummy_cache(8)],
            vec![0.0],
        );
        cache.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(
            cache.stats().total_positions_reused,
            8,
            "should report exactly how many positions were reused, not just that a hit occurred"
        );
    }

    /// The end-to-end property that matters most: using a prefix
    /// cache's restored KV state to continue a real decoder must
    /// produce EXACTLY the same output as processing the full token
    /// sequence from scratch. If this fails, prefix caching is not a
    /// safe optimization -- it would silently change model output
    /// depending on cache state, which is far worse than no caching at
    /// all.
    #[test]
    fn prefix_cached_continuation_matches_from_scratch_decode_exactly() {
        use crate::config::glm_5_2;
        use crate::decoder::Decoder;
        use ferrox_core::cache::KvCache as RealKvCache;

        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        let vocab = 16;

        let shared_prefix = vec![1usize, 2, 3, 4, 5];
        let full_sequence = vec![1usize, 2, 3, 4, 5, 6, 7];

        // "Conversation A": process the shared prefix once, store it.
        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<RealKvCache> = (0..2)
            .map(|_| RealKvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let prefix_logits = decoder_a.forward_batch(&shared_prefix, 0, &mut caches_a);
        let mut prefix_cache = PrefixCache::new(4);
        prefix_cache.store(
            shared_prefix.clone(),
            caches_a,
            prefix_logits.last().unwrap().clone(),
        );

        // "Conversation B": extends the shared prefix. Using the
        // prefix cache, only the new suffix tokens should need
        // computing.
        let decoder_b = Decoder::new_random_small(cfg.clone(), 2, vocab); // same seed => identical weights
        let m = prefix_cache.find_longest_prefix(&full_sequence);
        assert_eq!(m.matched_len, 5);
        let mut restored_caches = m.kv_caches.unwrap();
        let suffix = &full_sequence[m.matched_len..];
        let via_prefix_cache_logits =
            decoder_b.forward_batch(suffix, m.matched_len, &mut restored_caches);

        // Ground truth: process the ENTIRE sequence from scratch on an
        // identically-seeded decoder with a fresh empty cache.
        let decoder_c = Decoder::new_random_small(cfg, 2, vocab);
        let mut fresh_caches: Vec<RealKvCache> = (0..2)
            .map(|_| RealKvCache::new(decoder_c.config.n_kv_heads, decoder_c.config.head_dim))
            .collect();
        let from_scratch_logits = decoder_c.forward_batch(&full_sequence, 0, &mut fresh_caches);

        // The prefix-cache path's logits for the suffix positions must
        // match the from-scratch path's logits for those same
        // positions exactly.
        let from_scratch_suffix = &from_scratch_logits[m.matched_len..];
        assert_eq!(via_prefix_cache_logits.len(), from_scratch_suffix.len());
        for (pos, (a, b)) in via_prefix_cache_logits
            .iter()
            .zip(from_scratch_suffix.iter())
            .enumerate()
        {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-3,
                    "suffix position {pos}, logit {i}: via_prefix_cache={x} from_scratch={y}"
                );
            }
        }

        // And the KV cache state itself must match too, not just the
        // final logits (in case a later request extends even further).
        for (restored, fresh) in restored_caches.iter().zip(fresh_caches.iter()) {
            assert_eq!(restored.seq_len, fresh.seq_len);
            for (a, b) in restored.k.iter().zip(fresh.k.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }
}
