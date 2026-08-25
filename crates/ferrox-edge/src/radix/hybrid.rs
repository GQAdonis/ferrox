//! The hybrid radix cache: full KV pages plus a recurrent-state
//! snapshot.
//!
//! A hybrid model interleaves attention layers, which keep a KV cache,
//! with linear/gated-delta layers, which keep a fixed-size *recurrent
//! state*. Reusing a prefix means reusing both -- and they are not the
//! same shape of thing. KV is a span: every token of the prefix has
//! pages. The recurrent state is a **point**: one snapshot, valid at
//! exactly one sequence position, because it is the fold of everything
//! before it.
//!
//! So a snapshot attaches to a node's *end* boundary, and reuse
//! truncates to the deepest node that carries one. Matching four more
//! pages of KV past the last snapshot buys nothing: without the
//! recurrent state at that position the request would have to replay
//! those tokens through the linear layers anyway.
//!
//! That also explains why a snapshot never survives a split
//! ([`super::tree::RadixTree::split_at`] keeps it on the suffix): the
//! prefix half now ends somewhere the snapshot was never taken.
//!
//! The cache is pool-agnostic. A `mamba_value` is an opaque slot id
//! from whatever recurrent-state pool the engine runs; this module
//! stores it, hands it back when it evicts it, and never dereferences
//! it.
//!
//! Ported 1:1 from FreeToken's
//! `python/freetoken/kvcache/hybrid_radix_cache.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::tree::{align_down, match_len, NodeId, RadixTree, ROOT};

/// The chunk a linear-attention kernel folds at a time. A snapshot can
/// only be taken on one of these boundaries, so a page must not
/// straddle one.
pub const CHUNK_SIZE: usize = 64;

/// What a hybrid lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMatch {
    pub kv_indices: Vec<u32>,
    /// Always the length at the snapshot's boundary, never the deepest
    /// KV match.
    pub cached_len: usize,
    /// The recurrent-state slot to resume from, if any.
    pub mamba_value: Option<u32>,
    pub node: NodeId,
}

/// What a hybrid insert did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridInsert {
    /// How much of the span was already in the tree -- the caller's
    /// pages for it are duplicates.
    pub matched_len: usize,
    /// True when the node already carried a snapshot (or cannot carry
    /// one). The caller keeps ownership of the slot it offered and must
    /// return it to the pool.
    pub snapshot_exists: bool,
}

/// What an eviction pass released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HybridEvicted {
    pub kv_indices: Vec<u32>,
    /// Recurrent-state slots to return to their pool.
    pub mamba_slots: Vec<u32>,
}

/// A radix prefix cache for hybrid attention/recurrent models.
#[derive(Debug)]
pub struct HybridRadixCache {
    tree: RadixTree,
    full_evictable: usize,
    full_protected: usize,
    /// Counted in **snapshots**, not tokens: the recurrent pool is
    /// allocated in fixed-size slots, so its pressure is a count.
    mamba_evictable: usize,
    mamba_protected: usize,
    clock: i64,
}

impl HybridRadixCache {
    pub fn new(page_size: usize) -> Self {
        assert_eq!(
            CHUNK_SIZE % page_size,
            0,
            "page_size {page_size} must divide the linear-attention chunk of {CHUNK_SIZE}, \
             or a page would straddle a snapshot boundary"
        );
        HybridRadixCache {
            tree: RadixTree::new(page_size),
            full_evictable: 0,
            full_protected: 0,
            mamba_evictable: 0,
            mamba_protected: 0,
            clock: 0,
        }
    }

    pub fn tree(&self) -> &RadixTree {
        &self.tree
    }

    pub fn page_size(&self) -> usize {
        self.tree.page_size()
    }

    pub fn full_evictable_size(&self) -> usize {
        self.full_evictable
    }

    pub fn full_protected_size(&self) -> usize {
        self.full_protected
    }

    /// Snapshots that may be reclaimed.
    pub fn mamba_evictable_size(&self) -> usize {
        self.mamba_evictable
    }

    pub fn mamba_protected_size(&self) -> usize {
        self.mamba_protected
    }

    fn tick(&mut self) -> i64 {
        self.clock += 1;
        self.clock
    }

    fn walk_to(&mut self, input_ids: &[u32]) -> (NodeId, usize) {
        let page_size = self.tree.page_size();
        let tic = self.tick();
        let mut prefix_len = 0usize;
        let mut node = ROOT;
        while prefix_len < input_ids.len() {
            let Some(child) = self.tree.child(node, &input_ids[prefix_len..]) else {
                return (node, prefix_len);
            };
            node = child;
            let matched = align_down(
                match_len(&self.tree.node(node).key, &input_ids[prefix_len..]),
                page_size,
            );
            prefix_len += matched;
            if matched != self.tree.node(node).length() {
                node = self.tree.split_at(node, matched);
                self.tree.node_mut(node).timestamp = tic;
                return (node, prefix_len);
            }
            self.tree.node_mut(node).timestamp = tic;
        }
        (node, prefix_len)
    }

    /// The longest prefix that ends at a node carrying a recurrent
    /// snapshot.
    ///
    /// Truncating to the snapshot rather than to the KV match is the
    /// point: KV past the last snapshot cannot be resumed from.
    pub fn match_prefix(&mut self, input_ids: &[u32]) -> HybridMatch {
        let (node, _) = self.walk_to(input_ids);
        let mut cur = node;
        let mut end_len = self.tree.path_len(node);
        while !self.tree.node(cur).is_root() {
            if let Some(slot) = self.tree.node(cur).mamba_value {
                return HybridMatch {
                    kv_indices: self.tree.path_value(cur),
                    cached_len: end_len,
                    mamba_value: Some(slot),
                    node: cur,
                };
            }
            end_len -= self.tree.node(cur).length();
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
        HybridMatch {
            kv_indices: Vec::new(),
            cached_len: 0,
            mamba_value: None,
            node: ROOT,
        }
    }

    /// Publish `kv_indices` for `input_ids` and donate `mamba_value` at
    /// its end boundary.
    ///
    /// The donation is refused -- `snapshot_exists` -- when the span
    /// resolves to the root (the empty prefix has no state to snapshot)
    /// or when the node already carries one. In both cases the caller
    /// still owns the slot it offered and must return it to the pool;
    /// dropping it on the floor leaks a recurrent slot per request,
    /// which is the failure this return value exists to prevent.
    pub fn insert(
        &mut self,
        input_ids: &[u32],
        kv_indices: &[u32],
        mamba_value: u32,
    ) -> HybridInsert {
        assert_eq!(
            input_ids.len(),
            kv_indices.len(),
            "one page index per token"
        );
        let insert_len = align_down(input_ids.len(), self.tree.page_size());
        let input_ids = &input_ids[..insert_len];
        let kv_indices = &kv_indices[..insert_len];

        let (mut node, prefix_len) = self.walk_to(input_ids);
        if prefix_len != insert_len {
            let tic = self.clock;
            let new_node = self.tree.alloc(tic);
            self.tree.set_key_value(
                new_node,
                input_ids[prefix_len..].to_vec(),
                kv_indices[prefix_len..].to_vec(),
            );
            self.tree.set_parent(new_node, node);
            self.full_evictable += self.tree.node(new_node).length();
            node = new_node;
        }
        if self.tree.node(node).is_root() || self.tree.node(node).mamba_value.is_some() {
            return HybridInsert {
                matched_len: prefix_len,
                snapshot_exists: true,
            };
        }
        self.tree.node_mut(node).mamba_value = Some(mamba_value);
        if self.tree.node(node).mamba_ref_count == 0 {
            self.mamba_evictable += 1;
        }
        HybridInsert {
            matched_len: prefix_len,
            snapshot_exists: false,
        }
    }

    /// Protect the KV path to the root **and** the snapshot at `node`.
    ///
    /// Only at `node`: locking a descendant does not protect an
    /// ancestor's snapshot, because the reader resumed from its own
    /// node's state and never touches the ancestor's.
    pub fn inc_lock(&mut self, node: NodeId) {
        if self.tree.node(node).mamba_value.is_some() {
            if self.tree.node(node).mamba_ref_count == 0 {
                self.mamba_evictable -= 1;
                self.mamba_protected += 1;
            }
            self.tree.node_mut(node).mamba_ref_count += 1;
        }
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            let length = self.tree.node(cur).length();
            if self.tree.node(cur).ref_count == 0 {
                self.full_evictable -= length;
                self.full_protected += length;
            }
            self.tree.node_mut(cur).ref_count += 1;
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    pub fn dec_lock(&mut self, node: NodeId) {
        if self.tree.node(node).mamba_ref_count > 0 {
            self.tree.node_mut(node).mamba_ref_count -= 1;
            if self.tree.node(node).mamba_ref_count == 0
                && self.tree.node(node).mamba_value.is_some()
            {
                self.mamba_evictable += 1;
                self.mamba_protected -= 1;
            }
        }
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            let length = self.tree.node(cur).length();
            let count = self.tree.node(cur).ref_count;
            assert!(count > 0, "unlock without a matching lock");
            self.tree.node_mut(cur).ref_count = count - 1;
            if count - 1 == 0 {
                self.full_evictable += length;
                self.full_protected -= length;
            }
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    /// Free at least `num_tokens` of KV, oldest unlocked leaves first.
    /// A removed node's snapshot goes back with it.
    pub fn evict_full(&mut self, num_tokens: usize) -> HybridEvicted {
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = self
            .tree
            .leaves()
            .into_iter()
            .filter(|id| self.tree.node(*id).ref_count == 0)
            .map(|id| Reverse((self.tree.node(id).timestamp, id.0)))
            .collect();

        let mut out = HybridEvicted::default();
        let mut freed = 0usize;
        while freed < num_tokens {
            let Some(Reverse((_, raw))) = heap.pop() else {
                break;
            };
            let id = NodeId(raw);
            let node = self.tree.node(id);
            if node.ref_count != 0 || !node.is_leaf() || node.is_root() {
                continue;
            }
            let length = node.length();
            freed += length;
            out.kv_indices.extend_from_slice(&node.value);
            self.full_evictable -= length;
            self.release_snapshot(id, &mut out.mamba_slots);
            let parent = self.tree.unlink(id);
            let (parent, cascaded) = self.cascade_snapshotless_leaves(parent, &mut out.kv_indices);
            freed += cascaded;
            if self.tree.node(parent).is_leaf()
                && self.tree.node(parent).ref_count == 0
                && !self.tree.node(parent).is_root()
            {
                heap.push(Reverse((self.tree.node(parent).timestamp, parent.0)));
            }
        }
        out
    }

    /// Free at least `num` **snapshots**.
    ///
    /// An interior (or KV-locked) node just gives up its snapshot and
    /// keeps its pages, so its prefix is still reusable for a request
    /// that can resume from an ancestor. Only a free leaf goes away
    /// entirely.
    pub fn evict_mamba(&mut self, num: usize) -> HybridEvicted {
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = self
            .tree
            .walk()
            .into_iter()
            .filter(|id| {
                self.tree.node(*id).mamba_value.is_some()
                    && self.tree.node(*id).mamba_ref_count == 0
            })
            .map(|id| Reverse((self.tree.node(id).timestamp, id.0)))
            .collect();

        let mut out = HybridEvicted::default();
        let mut freed = 0usize;
        while freed < num {
            let Some(Reverse((_, raw))) = heap.pop() else {
                break;
            };
            let id = NodeId(raw);
            let node = self.tree.node(id);
            if node.mamba_value.is_none() || node.mamba_ref_count != 0 || node.is_root() {
                continue;
            }
            if node.is_leaf() && node.ref_count == 0 {
                let length = node.length();
                out.kv_indices.extend_from_slice(&node.value);
                self.full_evictable -= length;
                self.release_snapshot(id, &mut out.mamba_slots);
                freed += 1;
                let parent = self.tree.unlink(id);
                self.cascade_snapshotless_leaves(parent, &mut out.kv_indices);
            } else {
                self.release_snapshot(id, &mut out.mamba_slots);
                freed += 1;
            }
        }
        out
    }

    fn release_snapshot(&mut self, id: NodeId, out: &mut Vec<u32>) {
        if let Some(slot) = self.tree.node(id).mamba_value {
            out.push(slot);
            self.tree.node_mut(id).mamba_value = None;
            if self.tree.node(id).mamba_ref_count == 0 {
                self.mamba_evictable -= 1;
            }
        }
    }

    /// A leaf with no snapshot serves no resume, so once its subtree is
    /// gone its pages are dead weight.
    fn cascade_snapshotless_leaves(
        &mut self,
        parent: NodeId,
        kv_out: &mut Vec<u32>,
    ) -> (NodeId, usize) {
        let mut parent = parent;
        let mut freed = 0usize;
        while {
            let node = self.tree.node(parent);
            node.mamba_value.is_none() && node.is_leaf() && node.ref_count == 0 && !node.is_root()
        } {
            let length = self.tree.node(parent).length();
            kv_out.extend_from_slice(&self.tree.node(parent).value);
            self.full_evictable -= length;
            freed += length;
            parent = self.tree.unlink(parent);
        }
        (parent, freed)
    }

    pub fn check_integrity(&self) {
        self.tree.check_structure();
        let (mut full_e, mut full_p, mut mamba_e, mut mamba_p) = (0, 0, 0, 0);
        for id in self.tree.walk() {
            let node = self.tree.node(id);
            if node.ref_count == 0 {
                full_e += node.length();
            } else {
                full_p += node.length();
            }
            if node.mamba_value.is_some() {
                if node.mamba_ref_count == 0 {
                    mamba_e += 1;
                } else {
                    mamba_p += 1;
                }
            } else {
                assert_eq!(
                    node.mamba_ref_count, 0,
                    "a node with no snapshot cannot hold a snapshot lock"
                );
            }
        }
        assert_eq!(full_e, self.full_evictable, "full_evictable drifted");
        assert_eq!(full_p, self.full_protected, "full_protected drifted");
        assert_eq!(mamba_e, self.mamba_evictable, "mamba_evictable drifted");
        assert_eq!(mamba_p, self.mamba_protected, "mamba_protected drifted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = 4;

    fn ids(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn pages(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn a_page_must_not_straddle_a_chunk_boundary() {
        for ok in [1usize, 2, 4, 8, 16, 32, 64] {
            let _ = HybridRadixCache::new(ok);
        }
    }

    #[test]
    #[should_panic(expected = "must divide the linear-attention chunk")]
    fn a_page_size_that_does_not_divide_the_chunk_is_rejected() {
        HybridRadixCache::new(48);
    }

    #[test]
    fn reuse_resumes_from_the_deepest_snapshot() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..4), &pages(0, 4), 7);
        cache.insert(&ids(0..8), &pages(0, 8), 9);

        let m = cache.match_prefix(&ids(0..8));
        assert_eq!(m.cached_len, 8);
        assert_eq!(m.mamba_value, Some(9));
        assert_eq!(m.kv_indices, pages(0, 8));
        cache.check_integrity();
    }

    /// KV past the last snapshot is not reusable: without the
    /// recurrent state at that position it would have to be replayed
    /// anyway.
    #[test]
    fn kv_past_the_last_snapshot_is_dropped() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..4), &pages(0, 4), 7);
        // A longer prompt whose snapshot sits at token 12.
        cache.insert(&ids(0..12), &pages(0, 12), 9);
        // Matching eight tokens splits that node; the snapshot stays on
        // the far half, so the near half carries KV and no state.
        let m = cache.match_prefix(&ids(0..8));

        assert_eq!(m.cached_len, 4, "reuse falls back to the last snapshot");
        assert_eq!(m.mamba_value, Some(7));
        assert_eq!(m.kv_indices, pages(0, 4));
        cache.check_integrity();
    }

    /// Matching an interior boundary of a snapshot node yields nothing:
    /// the prefix half ends where no snapshot was ever taken.
    #[test]
    fn a_prefix_of_a_snapshot_node_is_not_reusable() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..8), &pages(0, 8), 9);
        let m = cache.match_prefix(&ids(0..4));
        assert_eq!(m.cached_len, 0);
        assert_eq!(m.mamba_value, None);
        cache.check_integrity();
    }

    /// The donation contract: a refused slot is still the caller's, and
    /// dropping it would leak one recurrent slot per request.
    #[test]
    fn a_duplicate_donation_is_refused_and_stays_the_callers() {
        let mut cache = HybridRadixCache::new(P);
        assert!(!cache.insert(&ids(0..8), &pages(0, 8), 9).snapshot_exists);
        let second = cache.insert(&ids(0..8), &pages(50, 8), 11);
        assert_eq!(second.matched_len, 8);
        assert!(second.snapshot_exists, "slot 11 goes back to the pool");
        assert_eq!(cache.mamba_evictable_size(), 1);
        assert_eq!(cache.match_prefix(&ids(0..8)).mamba_value, Some(9));
        cache.check_integrity();
    }

    /// A sub-page prompt resolves to the root, which cannot hold a
    /// snapshot.
    #[test]
    fn the_root_never_takes_a_snapshot() {
        let mut cache = HybridRadixCache::new(P);
        let res = cache.insert(&ids(0..3), &pages(0, 3), 5);
        assert_eq!(res.matched_len, 0);
        assert!(res.snapshot_exists);
        assert_eq!(cache.mamba_evictable_size(), 0);
        cache.check_integrity();
    }

    /// A node that gave up its snapshot can take a new one.
    #[test]
    fn a_snapshotless_node_can_be_refilled() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..8), &pages(0, 8), 9);
        cache.match_prefix(&ids(0..4)); // split so [0..4) is interior
        let out = cache.evict_mamba(1);
        assert_eq!(out.mamba_slots, vec![9]);

        let res = cache.insert(&ids(0..8), &pages(0, 8), 11);
        assert!(!res.snapshot_exists, "it attaches rather than dedups");
        assert_eq!(cache.match_prefix(&ids(0..8)).mamba_value, Some(11));
        cache.check_integrity();
    }

    /// Snapshot pressure is counted in slots, and taking a slot from an
    /// interior node must not cost any KV.
    #[test]
    fn snapshot_eviction_counts_slots_and_keeps_interior_pages() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..4), &pages(0, 4), 1);
        cache.insert(&ids(0..8), &pages(0, 8), 2);
        cache.insert(&ids(0..12), &pages(0, 12), 3);
        let kv_before = cache.full_evictable_size();

        let out = cache.evict_mamba(2);
        assert_eq!(out.mamba_slots.len(), 2);
        assert!(out.kv_indices.is_empty(), "interior nodes kept their pages");
        assert_eq!(cache.full_evictable_size(), kv_before);
        cache.check_integrity();
    }

    #[test]
    fn locking_pins_the_snapshot_and_the_whole_kv_path() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..4), &pages(0, 4), 1);
        cache.insert(&ids(0..8), &pages(0, 8), 2);
        let deep = cache.match_prefix(&ids(0..8));
        cache.inc_lock(deep.node);

        assert_eq!(cache.full_protected_size(), 8);
        assert_eq!(cache.mamba_protected_size(), 1);
        assert!(cache.evict_full(8).kv_indices.is_empty());

        // The ancestor's snapshot is NOT protected by a descendant's
        // lock: the reader resumed from its own node.
        let out = cache.evict_mamba(2);
        assert_eq!(out.mamba_slots, vec![1]);
        cache.check_integrity();

        cache.dec_lock(deep.node);
        assert_eq!(cache.full_protected_size(), 0);
        assert_eq!(cache.mamba_protected_size(), 0);
        cache.check_integrity();
    }

    #[test]
    fn full_eviction_cascades_through_snapshotless_leaves() {
        let mut cache = HybridRadixCache::new(P);
        cache.insert(&ids(0..8), &pages(0, 8), 9);
        cache.match_prefix(&ids(0..4)); // split; the head has no snapshot

        let out = cache.evict_full(4);
        assert_eq!(out.kv_indices.len(), 8, "both halves, one call");
        assert_eq!(out.mamba_slots, vec![9]);
        assert_eq!(cache.full_evictable_size(), 0);
        cache.check_integrity();
    }

    /// Every recurrent slot handed to the cache comes back exactly
    /// once, whether refused, evicted, or drained.
    #[test]
    fn recurrent_slots_are_conserved() {
        let mut cache = HybridRadixCache::new(P);
        let mut returned: Vec<u32> = Vec::new();
        for round in 0..6u32 {
            let prompt: Vec<u32> = (0..8).map(|i| if i < 4 { i } else { i + round }).collect();
            let res = cache.insert(&prompt, &pages(round * 8, 8), 100 + round);
            if res.snapshot_exists {
                returned.push(100 + round);
            }
            cache.check_integrity();
        }
        let drained = cache.evict_full(usize::MAX);
        returned.extend(drained.mamba_slots);
        returned.sort_unstable();
        returned.dedup();
        assert_eq!(returned, (100..106).collect::<Vec<u32>>());
        assert_eq!(cache.mamba_evictable_size(), 0);
        cache.check_integrity();
    }
}
