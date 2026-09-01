//! A whole-response cache for `ferrox-server`: exact-repeat requests
//! (same prompt, model, and generation parameters) skip the decode
//! loop entirely and return the previously computed completion.
//!
//! Design inspired by reading Shimmy's `src/cache/response_cache.rs`
//! (an LRU-with-TTL cache keyed by a hash of prompt + model + generation
//! params) -- own eviction bookkeeping here, but the shape of the idea
//! (whole-response caching, not KV-prefix caching, as the simpler thing
//! to do first) is the same one found there.
//!
//! This is deliberately the *simpler* of the two caching strategies
//! this server ships: it helps only when a request is an exact
//! repeat of a recent one (same prompt text, byte for byte). A real
//! KV-prefix cache (reusing the shared prefix of two *different*
//! prompts) is a separate, larger piece of work, tracked separately.

use crate::generate::{FinishReason, Usage};
use ferrox_models::sampling::SamplingParams;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    /// Every sampling parameter that affects output. See
    /// [`SamplingKey`] for why it is a struct built by one function
    /// rather than a handful of fields written out here.
    ///
    /// Only requests with a deterministic outcome -- greedy
    /// (temperature 0) or an explicit seed -- are ever looked up against
    /// this cache at all; see `ChatCompletionRequest::is_cacheable`. A
    /// request without a deterministic outcome must never populate or
    /// read this cache, since a "hit" would silently replay one random
    /// sample forever instead of producing fresh output each call,
    /// defeating the entire point of sampling.
    pub sampling: SamplingKey,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

/// Every field of a resolved [`SamplingParams`], in a form that can be
/// hashed and compared for exact equality.
///
/// One struct rather than six fields inlined into [`CacheKey`], and
/// built by [`sampling_key`] rather than by whoever happens to be
/// constructing a key, because the interesting property is not what is
/// in it -- it is that NOTHING is left out. A sampler setting outside
/// the key means two requests differing only in that setting share one
/// cached answer: the second caller is served output computed under the
/// first caller's parameters, with a 200 and no way to tell.
///
/// `f32`s are stored as bits so they participate in `Eq`/`Hash`. The
/// cache only ever compares these for exact key equality, never
/// arithmetic, so bit-identity is the right notion of "same parameters".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SamplingKey {
    pub temperature_bits: u32,
    pub top_p_bits: u32,
    pub min_p_bits: u32,
    pub top_k: usize,
    pub repetition_penalty_bits: u32,
    pub penalty_last_n: usize,
    pub presence_penalty_bits: u32,
    pub frequency_penalty_bits: u32,
}

/// The cache-key form of a resolved sampling configuration.
///
/// The destructure below is exhaustive ON PURPOSE, and is the whole
/// point of this function: a sampler knob added to `SamplingParams`
/// upstream stops this crate compiling, HERE, until someone decides
/// whether it belongs in the cache key. Reading the fields through `.`
/// accessors instead would let a new knob be added, be honoured by the
/// sampler, and be silently absent from the key -- which is the bug
/// described on [`SamplingKey`], arriving without anyone touching this
/// file.
pub fn sampling_key(params: &SamplingParams) -> SamplingKey {
    let SamplingParams {
        temperature,
        top_p,
        min_p,
        top_k,
        repetition_penalty,
        penalty_last_n,
        presence_penalty,
        frequency_penalty,
    } = params;
    SamplingKey {
        temperature_bits: temperature.to_bits(),
        top_p_bits: top_p.to_bits(),
        min_p_bits: min_p.to_bits(),
        top_k: *top_k,
        repetition_penalty_bits: repetition_penalty.to_bits(),
        penalty_last_n: *penalty_last_n,
        presence_penalty_bits: presence_penalty.to_bits(),
        frequency_penalty_bits: frequency_penalty.to_bits(),
    }
}

impl CacheKey {
    /// A short hex digest of this key, for logging/metrics without
    /// printing a potentially-long prompt verbatim.
    pub fn digest(&self) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// The complete cached outcome of a request: not just the text, but
/// the finish reason and token accounting too, so a cache hit's
/// response is indistinguishable from recomputing it (a hit that
/// reported a hardcoded finish reason or dropped `usage` would leak
/// which responses came from cache).
#[derive(Debug, Clone, PartialEq)]
pub struct CachedCompletion {
    pub content: String,
    pub finish: FinishReason,
    pub usage: Usage,
}

struct Entry {
    completion: CachedCompletion,
    inserted_at: Instant,
}

/// Cache hit/miss counters, exposed for tests and the server's
/// `/cache/stats` endpoint -- observable behavior, not just "it should
/// be faster the second time," which is a flaky thing to assert in a
/// test.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
}

/// A capacity-bounded, TTL-expiring, least-recently-used response
/// cache. Not thread-safe on its own -- `ferrox-server` wraps this in
/// a `tokio::sync::Mutex`, since the server is the only caller and a
/// mutex is simpler and sufficiently fast for a single-process demo
/// server than a lock-free structure would be worth building here.
pub struct ResponseCache {
    entries: HashMap<CacheKey, Entry>,
    /// Most-recently-used key at the back; used to evict the least-
    /// recently-used entry when the cache is full. Kept as a simple
    /// `VecDeque` rather than an intrusive linked list for clarity;
    /// `bump` is O(n) in the number of entries, which is fine for the
    /// small `max_entries` this cache is meant to hold (hundreds, not
    /// millions -- a real production cache at larger scale would want
    /// a proper O(1) LRU structure).
    order: VecDeque<CacheKey>,
    max_entries: usize,
    ttl: Duration,
    hits: u64,
    misses: u64,
}

impl ResponseCache {
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        ResponseCache {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            ttl,
            hits: 0,
            misses: 0,
        }
    }

    /// Looks up `key`. Returns `None` on a miss (not present, or
    /// present but expired -- an expired entry is evicted on lookup
    /// rather than waiting for a background sweep). Updates hit/miss
    /// counters and, on a hit, bumps the key to most-recently-used.
    pub fn get(&mut self, key: &CacheKey) -> Option<CachedCompletion> {
        let is_expired = self
            .entries
            .get(key)
            .map(|e| e.inserted_at.elapsed() > self.ttl)
            .unwrap_or(false);

        if is_expired {
            self.entries.remove(key);
            self.order.retain(|k| k != key);
        }

        match self.entries.get(key) {
            Some(entry) => {
                self.hits += 1;
                self.order.retain(|k| k != key);
                self.order.push_back(key.clone());
                Some(entry.completion.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Inserts or replaces the cached response for `key`, evicting the
    /// least-recently-used entry first if the cache is already at
    /// `max_entries` and `key` isn't already present.
    pub fn put(&mut self, key: CacheKey, completion: CachedCompletion) {
        if !self.entries.contains_key(&key) && self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.retain(|k| k != &key);
        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            Entry {
                completion,
                inserted_at: Instant::now(),
            },
        );
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(prompt: &str) -> CacheKey {
        CacheKey {
            model: "test-model".to_string(),
            prompt: prompt.to_string(),
            max_tokens: 16,
            sampling: sampling_key(&SamplingParams::default()),
            seed: None,
            stop: Vec::new(),
        }
    }

    /// Wraps plain text into a `CachedCompletion` with fixed
    /// finish/usage values, so these LRU/TTL-focused tests can keep
    /// comparing by content strings.
    fn cc(text: &str) -> CachedCompletion {
        CachedCompletion {
            content: text.to_string(),
            finish: FinishReason::Stop,
            usage: Usage::new(3, 5),
        }
    }

    #[test]
    fn miss_then_hit_for_the_same_key() {
        let mut cache = ResponseCache::new(10, Duration::from_secs(60));
        assert_eq!(cache.get(&key("hello")), None);
        cache.put(key("hello"), cc("world"));
        assert_eq!(cache.get(&key("hello")), Some(cc("world")));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn different_keys_do_not_collide() {
        let mut cache = ResponseCache::new(10, Duration::from_secs(60));
        cache.put(key("prompt a"), cc("response a"));
        cache.put(key("prompt b"), cc("response b"));
        assert_eq!(cache.get(&key("prompt a")), Some(cc("response a")));
        assert_eq!(cache.get(&key("prompt b")), Some(cc("response b")));
    }

    #[test]
    fn different_max_tokens_is_a_different_key_even_for_the_same_prompt() {
        let mut cache = ResponseCache::new(10, Duration::from_secs(60));
        let mut k1 = key("same prompt");
        k1.max_tokens = 16;
        let mut k2 = key("same prompt");
        k2.max_tokens = 32;

        cache.put(k1.clone(), cc("short response"));
        assert_eq!(
            cache.get(&k2),
            None,
            "different max_tokens must be a cache miss even with identical prompt text"
        );
        assert_eq!(cache.get(&k1), Some(cc("short response")));
    }

    #[test]
    fn expired_entry_is_a_miss_and_is_evicted() {
        let mut cache = ResponseCache::new(10, Duration::from_millis(10));
        cache.put(key("hello"), cc("world"));
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(
            cache.get(&key("hello")),
            None,
            "entry older than the TTL must be treated as a miss"
        );
        assert_eq!(
            cache.stats().entries,
            0,
            "expired entry must actually be evicted, not just skipped"
        );
    }

    #[test]
    fn evicts_least_recently_used_entry_when_full() {
        let mut cache = ResponseCache::new(2, Duration::from_secs(60));
        cache.put(key("a"), cc("1"));
        cache.put(key("b"), cc("2"));
        // touch "a" so "b" becomes the least-recently-used entry
        assert_eq!(cache.get(&key("a")), Some(cc("1")));
        cache.put(key("c"), cc("3"));

        assert_eq!(
            cache.get(&key("b")),
            None,
            "least-recently-used entry ('b') must have been evicted"
        );
        assert_eq!(
            cache.get(&key("a")),
            Some(cc("1")),
            "recently-touched entry ('a') must survive eviction"
        );
        assert_eq!(
            cache.get(&key("c")),
            Some(cc("3")),
            "newly inserted entry ('c') must be present"
        );
    }

    #[test]
    fn putting_an_existing_key_again_does_not_grow_past_capacity() {
        let mut cache = ResponseCache::new(2, Duration::from_secs(60));
        cache.put(key("a"), cc("1"));
        cache.put(key("b"), cc("2"));
        cache.put(key("a"), cc("1-updated")); // re-insert, should replace, not evict
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.get(&key("a")), Some(cc("1-updated")));
        assert_eq!(
            cache.get(&key("b")),
            Some(cc("2")),
            "unrelated entry must survive a re-insert of another key"
        );
    }

    #[test]
    fn digest_is_stable_for_identical_keys_and_differs_for_different_keys() {
        let a1 = key("hello");
        let a2 = key("hello");
        let b = key("goodbye");
        assert_eq!(a1.digest(), a2.digest());
        assert_ne!(a1.digest(), b.digest());
    }
}
