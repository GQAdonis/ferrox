//! A per-layer KV cache, growable one position at a time during decode.
//! Two growth strategies exist:
//!
//! - `with_pool`: PagedAttention-style block allocation. Many caches
//!   (one per concurrent request, typically) draw fixed-size blocks
//!   from one shared, bounded `KvBlockPool` instead of each
//!   independently pre-committing to a worst-case context length.
//!   Growth happens in fixed block-sized quanta, and a cache's blocks
//!   return to the shared pool when it's dropped, so the pool's free
//!   count is a real, live admission-control signal a caller can check
//!   before accepting a new request. This is the block-*allocation*
//!   half of PagedAttention; it does not (yet) change how attention
//!   reads a cache -- `k`/`v` are still read as one contiguous slice
//!   per sequence (see `Decoder::forward_token`/`forward_batch`), just
//!   backed by capacity that grows in block-sized steps instead of
//!   Rust's default exponential `Vec` growth. Wiring this into
//!   `ferrox-server` as live per-request admission control via
//!   `FERROX_KV_POOL_BLOCKS`/`FERROX_KV_POOL_BLOCK_SIZE`.

use std::sync::{Arc, Mutex};

/// Returned by `KvCache::push` (and `with_pool`) when a pool-backed
/// cache needs another block but its shared `KvBlockPool` has none
/// free. Caches built with `new`/`with_capacity` never return this --
/// their growth is unconditional, matching their pre-paging behavior
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolExhausted;

impl std::fmt::Display for KvPoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KV cache block pool exhausted: no free blocks remain")
    }
}

impl std::error::Error for KvPoolExhausted {}

/// A bounded pool of fixed-size KV-cache blocks (in positions) shared
/// across many `KvCache` instances, typically one pool per server
/// process. Each `KvCache::with_pool` acquires one block up front and
/// one more each time it grows past its currently held capacity;
/// `free_blocks` is therefore a live, accurate admission-control
/// signal -- a caller can check it before accepting a new request
/// rather than discovering exhaustion only after committing memory.
pub struct KvBlockPool {
    block_size: usize,
    total_blocks: usize,
    free_blocks: usize,
}

impl KvBlockPool {
    /// `block_size` positions per block, `total_blocks` blocks in the
    /// whole shared budget (so `block_size * total_blocks` positions
    /// total, across however many caches draw from this pool at once).
    pub fn new(block_size: usize, total_blocks: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        KvBlockPool {
            block_size,
            total_blocks,
            free_blocks: total_blocks,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn free_blocks(&self) -> usize {
        self.free_blocks
    }

    fn try_acquire(&mut self, n: usize) -> bool {
        if n <= self.free_blocks {
            self.free_blocks -= n;
            true
        } else {
            false
        }
    }

    fn release(&mut self, n: usize) {
        self.free_blocks = (self.free_blocks + n).min(self.total_blocks);
    }
}

struct PooledState {
    pool: Arc<Mutex<KvBlockPool>>,
    block_size: usize,
    blocks_held: usize,
}

pub struct KvCache {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub k: Vec<f32>, // [seq_len, n_kv_heads, head_dim], flattened
    pub v: Vec<f32>,
    pub seq_len: usize,
    /// The capacity (in positions) this cache was pre-allocated for,
    /// if any. `None` for caches built with `new` or `with_pool`.
    planned_capacity: Option<usize>,
    /// `Some` for caches built with `with_pool`; tracks the shared
    /// pool and how many blocks this cache currently holds, so its
    /// blocks can be returned on drop.
    pool_state: Option<PooledState>,
}

/// Cloning a pool-backed cache detaches the clone from pool accounting
/// (its `k`/`v`/`seq_len` data is copied normally, but the clone does
/// not hold or later release any blocks itself) -- mirroring how
/// `ferrox-models::prefix_cache` already uses `KvCache::clone` to fork
/// a cached prefix into a new, independent request's cache. Only the
/// original cache's blocks are released, exactly once, when it drops.
impl Clone for KvCache {
    fn clone(&self) -> Self {
        KvCache {
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            k: self.k.clone(),
            v: self.v.clone(),
            seq_len: self.seq_len,
            planned_capacity: self.planned_capacity,
            pool_state: None,
        }
    }
}

impl Drop for KvCache {
    fn drop(&mut self) {
        if let Some(state) = &self.pool_state {
            if let Ok(mut pool) = state.pool.lock() {
                pool.release(state.blocks_held);
            }
        }
    }
}

impl KvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        KvCache {
            n_kv_heads,
            head_dim,
            k: Vec::new(),
            v: Vec::new(),
            seq_len: 0,
            planned_capacity: None,
            pool_state: None,
        }
    }

    /// Pre-allocates storage for up to `max_seq_len` positions, so
    /// `push` never triggers a reallocation-and-copy during decode.
    /// Use this when the maximum context length is known ahead of time
    pub fn with_capacity(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let elems_per_position = n_kv_heads * head_dim;
        KvCache {
            n_kv_heads,
            head_dim,
            k: Vec::with_capacity(max_seq_len * elems_per_position),
            v: Vec::with_capacity(max_seq_len * elems_per_position),
            seq_len: 0,
            planned_capacity: Some(max_seq_len),
            pool_state: None,
        }
    }

    /// Acquires up front however many blocks from `pool` are needed to
    /// cover `max_seq_len` positions (at least one, even if
    /// `max_seq_len` is `0`), so a caller that knows its worst-case
    /// sequence length ahead of time (as `ferrox-server` does: prompt
    /// length + `max_tokens`) never needs to acquire another block
    /// mid-decode. This matters beyond performance: `push` growing past
    /// its currently held capacity can fail if the pool is exhausted by
    /// *other* requests by then, and callers like
    /// `ferrox_models::Decoder::forward_token` treat `push` as
    /// infallible for non-pooled caches -- a pooled cache that
    /// under-reserves at construction and then fails to grow later
    /// would violate that assumption and panic mid-decode. Sizing to
    /// `max_seq_len` up front turns that into an admission-control
    /// decision made once, honestly, before any generation work starts,
    /// exactly mirroring `with_capacity`'s worst-case pre-allocation --
    /// just drawn from a shared pool instead of a private allocation.
    /// Returns `Err(KvPoolExhausted)` without mutating anything if the
    /// pool doesn't have that many blocks free.
    pub fn with_pool(
        n_kv_heads: usize,
        head_dim: usize,
        pool: Arc<Mutex<KvBlockPool>>,
        max_seq_len: usize,
    ) -> Result<Self, KvPoolExhausted> {
        let block_size = pool.lock().unwrap().block_size();
        let blocks_needed = max_seq_len.div_ceil(block_size).max(1);
        if !pool.lock().unwrap().try_acquire(blocks_needed) {
            return Err(KvPoolExhausted);
        }
        let elems_per_position = n_kv_heads * head_dim;
        Ok(KvCache {
            n_kv_heads,
            head_dim,
            k: Vec::with_capacity(blocks_needed * block_size * elems_per_position),
            v: Vec::with_capacity(blocks_needed * block_size * elems_per_position),
            seq_len: 0,
            planned_capacity: None,
            pool_state: Some(PooledState {
                pool,
                block_size,
                blocks_held: blocks_needed,
            }),
        })
    }

    /// Appends one position's key/value vectors (each
    /// `n_kv_heads * head_dim` long) to the cache. For pool-backed
    /// caches, this may need to acquire another block first; if the
    /// shared pool has none free, no data is appended and
    /// `Err(KvPoolExhausted)` is returned. Caches built with `new` or
    /// `with_capacity` always return `Ok`.
    pub fn push(&mut self, k_step: &[f32], v_step: &[f32]) -> Result<(), KvPoolExhausted> {
        assert_eq!(k_step.len(), self.n_kv_heads * self.head_dim);
        assert_eq!(v_step.len(), self.n_kv_heads * self.head_dim);

        let elems_per_position = self.n_kv_heads * self.head_dim;
        if let Some(state) = &mut self.pool_state {
            let capacity_positions = self.k.capacity() / elems_per_position;
            if self.seq_len == capacity_positions {
                if !state.pool.lock().unwrap().try_acquire(1) {
                    return Err(KvPoolExhausted);
                }
                state.blocks_held += 1;
                self.k.reserve_exact(state.block_size * elems_per_position);
                self.v.reserve_exact(state.block_size * elems_per_position);
            }
        }

        self.k.extend_from_slice(k_step);
        self.v.extend_from_slice(v_step);
        self.seq_len += 1;
        Ok(())
    }

    /// Returns this cache's blocks to its shared pool immediately
    /// (rather than waiting for `Drop`) and detaches it from pool
    /// accounting; a no-op for caches that aren't pool-backed, and
    /// idempotent if called more than once.
    pub fn release_to_pool(&mut self) {
        if let Some(state) = self.pool_state.take() {
            if let Ok(mut pool) = state.pool.lock() {
                pool.release(state.blocks_held);
            }
        }
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq_len = 0;
    }

    /// Rolls the cache back to exactly `new_seq_len` positions,
    /// discarding everything after. Used to reject speculatively
    /// decoded draft tokens that turned out wrong: their K/V were
    /// already pushed during batched verification, and rejection means
    /// removing them so the next real decode step continues from the
    /// last *accepted* position, not the last *attempted* one.
    pub fn truncate(&mut self, new_seq_len: usize) {
        assert!(
            new_seq_len <= self.seq_len,
            "truncate target {new_seq_len} must not exceed current seq_len {}",
            self.seq_len
        );
        let elems_per_position = self.n_kv_heads * self.head_dim;
        self.k.truncate(new_seq_len * elems_per_position);
        self.v.truncate(new_seq_len * elems_per_position);
        self.seq_len = new_seq_len;
    }

    /// Bytes currently resident for this cache's K and V buffers
    /// combined (actual allocated capacity, not just used length) --
    /// the number that matters for "does this fit in the context
    /// budget,"
    pub fn allocated_bytes(&self) -> usize {
        (self.k.capacity() + self.v.capacity()) * std::mem::size_of::<f32>()
    }

    /// True if this cache was pre-allocated via `with_capacity` and
    /// has not yet grown past that planned capacity (i.e. `push` has
    /// never had to reallocate). Useful for tests/diagnostics
    /// confirming the pre-allocation path actually avoided reallocs.
    pub fn is_within_planned_capacity(&self) -> bool {
        match self.planned_capacity {
            Some(cap) => {
                self.seq_len <= cap
                    && self.k.capacity() >= self.seq_len * self.n_kv_heads * self.head_dim
            }
            None => false,
        }
    }
}

/// The other half of PagedAttention that `KvBlockPool`/`KvCache::with_pool`
/// deliberately don't implement (see this module's doc comment): real,
/// *shared* physical block storage that many sequences' block tables can
/// address into, instead of each `KvCache` still owning its own private,
/// contiguous `Vec`. `KvBlockPool` only ever bounds a *count* of blocks
/// each cache may grow to; `PagedKvStore` is the actual backing memory,
/// and a sequence's `PagedKvCache` holds a block table (an ordered list
/// of block IDs into this shared store) instead of owning K/V data
/// directly. This is what makes non-contiguous-block reads during
/// attention (`causal_gqa_attention_paged`, in `attention.rs`) possible
/// at all -- `causal_gqa_attention`'s existing contiguous-slice read
/// pattern has no way to express "position 37 lives in block 12, cached
/// out of order relative to block 5."
pub struct PagedKvStore {
    block_size: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k: Vec<f32>, // [total_blocks * block_size, n_kv_heads, head_dim], flattened
    v: Vec<f32>,
    free_block_ids: Vec<usize>,
}

impl PagedKvStore {
    pub fn new(block_size: usize, total_blocks: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        let elems_per_block = block_size * n_kv_heads * head_dim;
        PagedKvStore {
            block_size,
            n_kv_heads,
            head_dim,
            k: vec![0.0; total_blocks * elems_per_block],
            v: vec![0.0; total_blocks * elems_per_block],
            // Pushed in descending order so `pop()` hands out ascending
            // block IDs -- not load-bearing for correctness (any free ID
            // works), just makes manual debugging/inspection saner.
            free_block_ids: (0..total_blocks).rev().collect(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_block_count(&self) -> usize {
        self.free_block_ids.len()
    }

    fn acquire_block(&mut self) -> Option<usize> {
        self.free_block_ids.pop()
    }

    fn release_block(&mut self, id: usize) {
        self.free_block_ids.push(id);
    }

    fn elems_per_block(&self) -> usize {
        self.block_size * self.n_kv_heads * self.head_dim
    }

    /// One position's K (or V) row within block `id` at `offset` (0-based
    /// within the block) -- `[n_kv_heads * head_dim]` long. Used by
    /// `causal_gqa_attention_paged` to read attention inputs directly out
    /// of shared physical storage via a block table, and by
    /// `PagedKvCache::push` to write a new position into it.
    pub fn k_row(&self, id: usize, offset: usize) -> &[f32] {
        let elems_per_position = self.n_kv_heads * self.head_dim;
        let start = id * self.elems_per_block() + offset * elems_per_position;
        &self.k[start..start + elems_per_position]
    }

    pub fn v_row(&self, id: usize, offset: usize) -> &[f32] {
        let elems_per_position = self.n_kv_heads * self.head_dim;
        let start = id * self.elems_per_block() + offset * elems_per_position;
        &self.v[start..start + elems_per_position]
    }

    fn k_row_mut(&mut self, id: usize, offset: usize) -> &mut [f32] {
        let elems_per_position = self.n_kv_heads * self.head_dim;
        let start = id * self.elems_per_block() + offset * elems_per_position;
        &mut self.k[start..start + elems_per_position]
    }

    fn v_row_mut(&mut self, id: usize, offset: usize) -> &mut [f32] {
        let elems_per_position = self.n_kv_heads * self.head_dim;
        let start = id * self.elems_per_block() + offset * elems_per_position;
        &mut self.v[start..start + elems_per_position]
    }
}

/// Returned when a `PagedKvCache` needs another block but its
/// `PagedKvStore` has none free -- the paged-storage analog of
/// `KvPoolExhausted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedStoreExhausted;

impl std::fmt::Display for PagedStoreExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "paged KV store exhausted: no free blocks remain")
    }
}

impl std::error::Error for PagedStoreExhausted {}

/// One sequence's view into a shared `PagedKvStore`: a block table
/// (which physical blocks this sequence's positions live in, in order)
/// plus how many positions have been written so far. Unlike `KvCache`,
/// this holds no K/V data itself -- every read and write goes through
/// the shared store.
#[derive(Debug, Clone, Default)]
pub struct PagedKvCache {
    block_table: Vec<usize>,
    seq_len: usize,
}

impl PagedKvCache {
    pub fn new() -> Self {
        PagedKvCache {
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    /// Appends one position's key/value vectors, acquiring a new block
    /// from `store` first if the current tail block is full (or none
    /// held yet). Mirrors `KvCache::push`'s signature/semantics exactly,
    /// just against shared storage instead of a private buffer.
    pub fn push(
        &mut self,
        store: &mut PagedKvStore,
        k_step: &[f32],
        v_step: &[f32],
    ) -> Result<(), PagedStoreExhausted> {
        let block_size = store.block_size();
        let offset_in_block = self.seq_len % block_size;
        if offset_in_block == 0 {
            let id = store.acquire_block().ok_or(PagedStoreExhausted)?;
            self.block_table.push(id);
        }
        let block_id = *self
            .block_table
            .last()
            .expect("offset_in_block == 0 branch above always pushes one first");
        store
            .k_row_mut(block_id, offset_in_block)
            .copy_from_slice(k_step);
        store
            .v_row_mut(block_id, offset_in_block)
            .copy_from_slice(v_step);
        self.seq_len += 1;
        Ok(())
    }

    /// Releases every block this sequence holds back to `store`. Must be
    /// called explicitly (there's no `Drop` here, since dropping needs a
    /// `&mut PagedKvStore` this type doesn't own a reference to) --
    /// mirrors `KvCache::release_to_pool`, just not automatic.
    pub fn release(&mut self, store: &mut PagedKvStore) {
        for id in self.block_table.drain(..) {
            store.release_block(id);
        }
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_grows_seq_len_and_stores_values() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.seq_len, 1);
        cache
            .push(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0])
            .unwrap();
        assert_eq!(cache.seq_len, 2);
        assert_eq!(cache.k.len(), 2 * 2 * 2);
        assert_eq!(cache.k[4], 9.0);
    }

    #[test]
    #[should_panic]
    fn push_wrong_size_panics() {
        let mut cache = KvCache::new(2, 2);
        let _ = cache.push(&[1.0, 2.0], &[1.0, 2.0]); // too short
    }

    #[test]
    fn clear_resets_state() {
        let mut cache = KvCache::new(1, 1);
        cache.push(&[1.0], &[2.0]).unwrap();
        cache.clear();
        assert_eq!(cache.seq_len, 0);
        assert!(cache.k.is_empty());
    }

    #[test]
    fn truncate_rolls_back_to_exact_length_preserving_earlier_data() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        cache
            .push(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .unwrap();
        cache
            .push(&[9.0, 9.0, 9.0, 9.0], &[90.0, 90.0, 90.0, 90.0])
            .unwrap();
        assert_eq!(cache.seq_len, 3);

        cache.truncate(1);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.v, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn truncate_to_current_length_is_a_no_op() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(1);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k, vec![1.0, 2.0]);
    }

    #[test]
    fn truncate_to_zero_empties_the_cache() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(0);
        assert_eq!(cache.seq_len, 0);
        assert!(cache.k.is_empty());
        assert!(cache.v.is_empty());
    }

    #[test]
    #[should_panic]
    fn truncate_beyond_current_length_panics() {
        let mut cache = KvCache::new(1, 2);
        cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.truncate(5);
    }

    #[test]
    fn push_after_truncate_continues_correctly() {
        let mut cache = KvCache::new(1, 1);
        cache.push(&[1.0], &[10.0]).unwrap();
        cache.push(&[2.0], &[20.0]).unwrap();
        cache.push(&[3.0], &[30.0]).unwrap(); // this one will be "rejected"
        cache.truncate(2);
        cache.push(&[99.0], &[990.0]).unwrap(); // real continuation after rejection
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.k, vec![1.0, 2.0, 99.0]);
        assert_eq!(cache.v, vec![10.0, 20.0, 990.0]);
    }

    #[test]
    fn with_capacity_preallocates_and_never_reallocates_within_plan() {
        let n_kv_heads = 4;
        let head_dim = 8;
        let max_seq_len = 16;
        let mut cache = KvCache::with_capacity(n_kv_heads, head_dim, max_seq_len);

        let expected_elems = max_seq_len * n_kv_heads * head_dim;
        assert!(cache.k.capacity() >= expected_elems);
        assert!(cache.v.capacity() >= expected_elems);

        let step = vec![0.5f32; n_kv_heads * head_dim];
        let k_ptr_before = cache.k.as_ptr();
        for _ in 0..max_seq_len {
            cache.push(&step, &step).unwrap();
        }
        let k_ptr_after = cache.k.as_ptr();
        assert_eq!(
            k_ptr_before, k_ptr_after,
            "pushing exactly up to the planned capacity must not reallocate"
        );
        assert!(cache.is_within_planned_capacity());
    }

    #[test]
    fn allocated_bytes_reflects_preallocated_capacity_not_just_used_length() {
        let cache = KvCache::with_capacity(4, 8, 100);
        // 100 positions * 4 kv_heads * 8 head_dim * 2 (k+v) * 4 bytes/f32
        let expected_min = 100 * 4 * 8 * 2 * 4;
        assert!(
            cache.allocated_bytes() >= expected_min,
            "allocated_bytes={} expected_min={expected_min}",
            cache.allocated_bytes()
        );
        // Nothing has been pushed yet, but the memory is already reserved.
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn grow_as_you_go_cache_reports_not_within_planned_capacity() {
        let mut cache = KvCache::new(2, 2);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert!(
            !cache.is_within_planned_capacity(),
            "a cache built with `new` has no plan to be within"
        );
    }

    #[test]
    fn with_pool_acquires_one_block_and_reports_it_in_free_blocks() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 10)));
        let cache = KvCache::with_pool(2, 2, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 9);
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn with_pool_fails_without_mutating_the_pool_when_exhausted() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 0)));
        let result = KvCache::with_pool(2, 2, pool.clone(), 0);
        assert!(result.is_err());
        assert_eq!(pool.lock().unwrap().free_blocks(), 0);
    }

    #[test]
    fn push_acquires_additional_blocks_as_the_cache_crosses_block_boundaries() {
        let block_size = 2;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 10)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 9);

        // First block holds `block_size` = 2 positions; pushing them
        // must not need a second block.
        cache.push(&[1.0], &[1.0]).unwrap();
        cache.push(&[2.0], &[2.0]).unwrap();
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            9,
            "filling exactly the first block must not acquire a second one"
        );

        // The third position crosses into a second block.
        cache.push(&[3.0], &[3.0]).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 8);
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.k, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn push_returns_pool_exhausted_and_leaves_state_unchanged_when_no_blocks_remain() {
        let block_size = 1;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 1)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 0);

        cache.push(&[1.0], &[1.0]).unwrap(); // fills the one held block

        let before_k = cache.k.clone();
        let result = cache.push(&[2.0], &[2.0]);
        assert_eq!(result, Err(KvPoolExhausted));
        assert_eq!(cache.seq_len, 1, "a failed push must not change seq_len");
        assert_eq!(cache.k, before_k, "a failed push must not append data");
    }

    #[test]
    fn dropping_a_pooled_cache_returns_its_blocks_to_the_pool() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 2)));
        {
            let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
            cache.push(&[1.0], &[1.0]).unwrap(); // fills the first (only held) block
            cache.push(&[2.0], &[2.0]).unwrap(); // crosses into a second block
            assert_eq!(pool.lock().unwrap().free_blocks(), 0);
        }
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "both blocks held by the dropped cache must return to the pool"
        );
    }

    #[test]
    fn release_to_pool_is_explicit_and_idempotent() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 5)));
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 4);

        cache.release_to_pool();
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);

        cache.release_to_pool(); // no-op, must not over-release
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);

        drop(cache); // must not release again either
        assert_eq!(pool.lock().unwrap().free_blocks(), 5);
    }

    #[test]
    fn two_pooled_caches_share_one_bounded_budget() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 1)));
        let cache_a = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        let cache_b = KvCache::with_pool(1, 1, pool.clone(), 0);
        assert!(
            cache_b.is_err(),
            "a second concurrent request must not be admitted when the shared budget is full"
        );

        drop(cache_a);
        let cache_c = KvCache::with_pool(1, 1, pool, 0);
        assert!(
            cache_c.is_ok(),
            "once the first request's cache is dropped, its budget must become available again"
        );
    }

    #[test]
    fn cloning_a_pooled_cache_detaches_the_clone_from_pool_accounting() {
        let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 3)));
        let original = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        assert_eq!(pool.lock().unwrap().free_blocks(), 2);

        let clone = original.clone();
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "cloning must not acquire additional blocks"
        );
        assert_eq!(clone.k, original.k);

        drop(clone);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            2,
            "dropping a detached clone must not release the original's blocks"
        );

        drop(original);
        assert_eq!(
            pool.lock().unwrap().free_blocks(),
            3,
            "dropping the original must release its blocks exactly once"
        );
    }
}
