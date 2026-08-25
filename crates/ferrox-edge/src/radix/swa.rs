//! The sliding-window radix cache: two currencies over one tree.
//!
//! A window model's layers do not all read the same thing. Full
//! attention layers read every token's KV; window layers read only the
//! last `sliding_window` tokens. So the window pool can be a fraction
//! of the full pool -- and the two run out at different times, which is
//! the whole reason this cache exists.
//!
//! # The tombstone
//!
//! When window memory runs short, a node's *window* KV is freed while
//! its *full* KV survives. The node is marked
//! [`swa_tombstone`](super::tree::Node::swa_tombstone). It still serves
//! a full-KV match, so the pages under it are not wasted -- but a
//! window layer cannot read it, and that is what makes matching subtle:
//!
//! A prompt may only resume from a point where the **last
//! `sliding_window` tokens are all window-live**. A tombstone anywhere
//! inside that trailing run means the window layers would read freed
//! state. So the match walks the path accumulating a *live run*, and
//! commits a reuse boundary only where that run has covered a whole
//! window. A run of exactly `sliding_window` is enough -- the
//! comparison is `>=`, and getting it wrong (`>`) silently collapses
//! every match to zero on any prompt whose live run lands exactly on
//! the window, which is the common case after a trim.
//!
//! # LRU by match, not by write
//!
//! [`SwaRadixCache::match_prefix`] stamps the matched path with
//! *strictly decreasing* timestamps toward the root, so the deepest
//! node is newest. The window evictor pops the minimum, which therefore
//! reclaims near-root nodes of a matched path first: exactly right,
//! because those are the tokens furthest outside the window.
//!
//! Ported 1:1 from FreeToken's
//! `python/freetoken/kvcache/swa_radix_cache.py` (Apache-2.0), which
//! follows SGLang's `SWARadixCache`; see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::tree::{align_down, match_len, NodeId, RadixTree, ROOT};

/// Room for a per-node depth offset under one logical event id, so one
/// `match_prefix` can stamp a whole path without two paths interleaving.
const EVENT_STRIDE: i64 = 1 << 24;

/// What a window-aware lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwaMatch {
    /// Full-pool page indices for the reusable prefix.
    pub kv_indices: Vec<u32>,
    pub cached_len: usize,
    /// The node to lock. May be *shallower* than the deepest matched
    /// node: everything past it failed the window-liveness test.
    pub node: NodeId,
}

/// What an eviction pass released.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SwaEvicted {
    /// Full-pool indices removed from the tree; return these to the
    /// page pool.
    pub kv_indices: Vec<u32>,
    /// Full-pool indices whose *window* slots should be released.
    /// Freeing a window slot is idempotent, so a caller may pass these
    /// through unconditionally.
    pub swa_indices: Vec<u32>,
}

/// What an insert did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwaInsert {
    /// How much of the span was already in the tree.
    pub matched_len: usize,
    /// Pages the caller must return: duplicates of what the tree
    /// already held, plus any stale pages an entry revived over.
    pub freed: Vec<u32>,
}

/// A radix prefix cache for sliding-window models.
#[derive(Debug)]
pub struct SwaRadixCache {
    tree: RadixTree,
    sliding_window: usize,
    full_evictable: usize,
    full_protected: usize,
    swa_evictable: usize,
    swa_protected: usize,
    clock: i64,
    uuid_of_lock: u64,
    revives: u64,
}

impl SwaRadixCache {
    pub fn new(page_size: usize, sliding_window: usize) -> Self {
        assert!(sliding_window > 0, "a window model has a positive window");
        SwaRadixCache {
            tree: RadixTree::new(page_size),
            sliding_window,
            full_evictable: 0,
            full_protected: 0,
            swa_evictable: 0,
            swa_protected: 0,
            clock: 0,
            uuid_of_lock: 0,
            revives: 0,
        }
    }

    pub fn tree(&self) -> &RadixTree {
        &self.tree
    }

    pub fn page_size(&self) -> usize {
        self.tree.page_size()
    }

    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    /// Tokens whose full KV may be reclaimed.
    pub fn full_evictable_size(&self) -> usize {
        self.full_evictable
    }

    pub fn full_protected_size(&self) -> usize {
        self.full_protected
    }

    /// Tokens whose *window* KV is live and unlocked, and so may be
    /// tombstoned.
    pub fn swa_evictable_size(&self) -> usize {
        self.swa_evictable
    }

    pub fn swa_protected_size(&self) -> usize {
        self.swa_protected
    }

    /// How many tombstoned entries have been brought back to life by an
    /// insert. Observability only.
    pub fn revives(&self) -> u64 {
        self.revives
    }

    fn tick(&mut self) -> i64 {
        self.clock += 1;
        self.clock * EVENT_STRIDE
    }

    /// Stamp the matched path with strictly decreasing timestamps
    /// toward the root, so the deepest node is the newest.
    fn stamp_path(&mut self, node: NodeId) {
        self.clock += 1;
        let base = self.clock * EVENT_STRIDE;
        let mut offset = 0i64;
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            self.tree.node_mut(cur).timestamp = base - offset;
            offset += 1;
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    /// The longest prefix of `input_ids` whose trailing `sliding_window`
    /// tokens are all window-live.
    ///
    /// A tombstone-free path is reusable however short -- there is no
    /// freed window state on it to read.
    pub fn match_prefix(&mut self, input_ids: &[u32]) -> SwaMatch {
        let page_size = self.tree.page_size();
        let mut node = ROOT;
        let mut value: Vec<Vec<u32>> = Vec::new();
        // "Infinite" until the first tombstone: a clean path to the
        // root has no freed window state on it at all.
        let mut live_run = usize::MAX;
        let mut best_value_len = 0usize;
        let mut best_node = ROOT;
        let mut pos = 0usize;

        while pos < input_ids.len() {
            let Some(child) = self.tree.child(node, &input_ids[pos..]) else {
                break;
            };
            if self.tree.node(child).swa_tombstone {
                // Commit the boundary at the *parent*, before this
                // tombstone, then start the run over.
                if live_run >= self.sliding_window {
                    best_value_len = value.len();
                    best_node = node;
                }
                live_run = 0;
            }
            let matched = align_down(
                match_len(&self.tree.node(child).key, &input_ids[pos..]),
                page_size,
            );
            if matched == 0 {
                break;
            }
            let partial = matched < self.tree.node(child).length();
            let child = if partial {
                self.tree.split_at(child, matched)
            } else {
                child
            };
            value.push(self.tree.node(child).value.clone());
            if !self.tree.node(child).swa_tombstone {
                live_run = live_run.saturating_add(self.tree.node(child).length());
            }
            node = child;
            pos += matched;
            if partial {
                break;
            }
        }

        if live_run >= self.sliding_window {
            best_value_len = value.len();
            best_node = node;
        }
        self.stamp_path(best_node);
        let kv_indices: Vec<u32> = value[..best_value_len].concat();
        SwaMatch {
            cached_len: kv_indices.len(),
            kv_indices,
            node: best_node,
        }
    }

    fn add_child(
        &mut self,
        parent: NodeId,
        ids: Vec<u32>,
        kv: Vec<u32>,
        tombstone: bool,
    ) -> NodeId {
        let tic = self.tick();
        let child = self.tree.alloc(tic);
        self.tree.set_key_value(child, ids, kv);
        self.tree.set_parent(child, parent);
        self.tree.node_mut(child).swa_tombstone = tombstone;
        let length = self.tree.node(child).length();
        self.full_evictable += length;
        if !tombstone {
            self.swa_evictable += length;
        }
        child
    }

    /// Publish `kv_indices` for `input_ids`, reviving tombstones the
    /// request can pay for.
    ///
    /// Two frontiers make this more than a plain insert:
    ///
    /// - `update_kv_after_len` is the request's *old reused prefix*.
    ///   Below it the tree's pages and the request's pages are the same
    ///   pages, so nothing is freed and nothing is revived.
    /// - `swa_evicted_seqlen` is how far the request itself let its own
    ///   window state slide out during decode. Below it the request's
    ///   window slots are already gone, so a tombstone there stays a
    ///   tombstone; at or above it the request has live window state,
    ///   so a tombstone can be *revived* by adopting the request's
    ///   pages.
    ///
    /// Returns the matched prefix length and every page the caller must
    /// return to the pool.
    pub fn insert(
        &mut self,
        input_ids: &[u32],
        kv_indices: &[u32],
        swa_evicted_seqlen: usize,
        update_kv_after_len: usize,
    ) -> SwaInsert {
        assert_eq!(
            input_ids.len(),
            kv_indices.len(),
            "one page index per token"
        );
        let page_size = self.tree.page_size();
        let insert_len = align_down(input_ids.len(), page_size);
        let input_ids = &input_ids[..insert_len];
        let kv_indices = &kv_indices[..insert_len];

        let mut freed: Vec<u32> = Vec::new();
        let mut node = ROOT;
        let mut total = 0usize;

        while total < insert_len {
            let Some(child) = self.tree.child(node, &input_ids[total..]) else {
                break;
            };
            let matched = align_down(
                match_len(&self.tree.node(child).key, &input_ids[total..]),
                page_size,
            );
            if matched == 0 {
                break;
            }
            let partial = matched < self.tree.node(child).length();
            let child = if partial {
                self.tree.split_at(child, matched)
            } else {
                child
            };
            let seg_kv = &kv_indices[total..total + matched];

            if update_kv_after_len < total + matched {
                if self.tree.node(child).swa_tombstone {
                    assert_eq!(
                        self.tree.node(child).swa_ref_count,
                        0,
                        "a tombstoned node cannot hold a window lock"
                    );
                    if self.tree.node(child).ref_count > 0 {
                        // A full-KV reader is gathering the *current*
                        // pages; swapping them under it would hand it
                        // state it never computed. Keep the tombstone
                        // and give the duplicate back.
                        freed.extend_from_slice(seg_kv);
                    } else if swa_evicted_seqlen <= total {
                        // The request's window state covers all of this
                        // node: revive it whole.
                        freed.extend_from_slice(&self.tree.node(child).value);
                        let key = self.tree.node(child).key.clone();
                        self.tree.set_key_value(child, key, seg_kv.to_vec());
                        let tic = self.tick();
                        let length = self.tree.node(child).length();
                        let node_mut = self.tree.node_mut(child);
                        node_mut.swa_tombstone = false;
                        node_mut.timestamp = tic;
                        self.swa_evictable += length;
                        self.revives += 1;
                    } else if swa_evicted_seqlen < total + matched {
                        // The frontier falls inside this node: split
                        // there, leave the head a tombstone, revive the
                        // live tail.
                        let start = swa_evicted_seqlen - total;
                        self.tree.split_at(child, start);
                        freed.extend_from_slice(&self.tree.node(child).value);
                        freed.extend_from_slice(&seg_kv[..start]);
                        let key = self.tree.node(child).key.clone();
                        self.tree
                            .set_key_value(child, key, seg_kv[start..].to_vec());
                        let tic = self.tick();
                        let length = self.tree.node(child).length();
                        let node_mut = self.tree.node_mut(child);
                        node_mut.swa_tombstone = false;
                        node_mut.timestamp = tic;
                        self.swa_evictable += length;
                        self.revives += 1;
                    } else {
                        // Still wholly outside the request's own window.
                        freed.extend_from_slice(seg_kv);
                    }
                } else {
                    // A live node's pages are canonical; the caller's
                    // are duplicates.
                    freed.extend_from_slice(seg_kv);
                }
            }
            total += matched;
            node = child;
            if partial {
                break;
            }
        }

        if total < insert_len {
            let mut suffix_ids = &input_ids[total..];
            let mut suffix_kv = &kv_indices[total..];
            // How much of the suffix the request has no window state
            // for -- and a clamp that keeps a *leaf* from ever being a
            // tombstone, which both the windowed match and the
            // full-eviction cascade rely on.
            let mut boundary = swa_evicted_seqlen.min(insert_len).saturating_sub(total);
            boundary = boundary.min(suffix_ids.len().saturating_sub(page_size));
            if boundary > 0 {
                node = self.add_child(
                    node,
                    suffix_ids[..boundary].to_vec(),
                    suffix_kv[..boundary].to_vec(),
                    true,
                );
                suffix_ids = &suffix_ids[boundary..];
                suffix_kv = &suffix_kv[boundary..];
            }
            if !suffix_ids.is_empty() {
                self.add_child(node, suffix_ids.to_vec(), suffix_kv.to_vec(), false);
            }
        }

        SwaInsert {
            matched_len: total,
            freed,
        }
    }

    /// Take both locks along the path from `node` to the root.
    ///
    /// The full lock covers the whole path. The window lock covers only
    /// the trailing `sliding_window` tokens of live nodes -- and the
    /// node where that run completes gets a handle, returned here, so
    /// that this reader's [`dec_lock`](Self::dec_lock) releases exactly
    /// its own window and not a deeper reader's.
    ///
    /// `None` means the live path is shorter than a window: it is still
    /// window-pinned as far as it goes, there is simply no boundary to
    /// name.
    pub fn inc_lock(&mut self, node: NodeId) -> Option<u64> {
        let mut boundary_uuid = None;
        let mut window_locked = 0usize;
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            let length = self.tree.node(cur).length();
            if self.tree.node(cur).ref_count == 0 {
                self.full_evictable -= length;
                self.full_protected += length;
            }
            self.tree.node_mut(cur).ref_count += 1;

            if window_locked < self.sliding_window && !self.tree.node(cur).swa_tombstone {
                if self.tree.node(cur).swa_ref_count == 0 {
                    self.swa_evictable -= length;
                    self.swa_protected += length;
                }
                self.tree.node_mut(cur).swa_ref_count += 1;
                window_locked += length;
                if window_locked >= self.sliding_window {
                    if self.tree.node(cur).swa_uuid.is_none() {
                        let uuid = self.tree.next_uuid();
                        self.uuid_of_lock = uuid;
                        self.tree.node_mut(cur).swa_uuid = Some(uuid);
                    }
                    boundary_uuid = self.tree.node(cur).swa_uuid;
                }
            }
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
        boundary_uuid
    }

    /// Release a lock taken by [`inc_lock`](Self::inc_lock).
    ///
    /// Pass back the handle `inc_lock` returned: the window decrement
    /// stops at that boundary (inclusive), so a shallower reader's
    /// window stays pinned. Passing `None` releases the window lock all
    /// the way to the root, which is right when `inc_lock` returned
    /// `None`.
    pub fn dec_lock(&mut self, node: NodeId, boundary_uuid: Option<u64>) {
        let mut releasing_window = true;
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
            if releasing_window
                && !self.tree.node(cur).swa_tombstone
                && self.tree.node(cur).swa_ref_count > 0
            {
                self.tree.node_mut(cur).swa_ref_count -= 1;
                if self.tree.node(cur).swa_ref_count == 0 {
                    self.swa_evictable += length;
                    self.swa_protected -= length;
                }
                if boundary_uuid.is_some() && self.tree.node(cur).swa_uuid == boundary_uuid {
                    releasing_window = false;
                }
            }
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    /// Free at least `num_tokens` of *full* KV, oldest unlocked leaves
    /// first.
    ///
    /// Removing a node also releases its window slots when it had any.
    /// A leaf whose removal exposes a tombstoned ancestor takes that
    /// ancestor too: a tombstone holds full KV nothing can reach any
    /// more once its subtree is gone.
    pub fn evict_full(&mut self, num_tokens: usize) -> SwaEvicted {
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = self
            .tree
            .leaves()
            .into_iter()
            .filter(|id| self.tree.node(*id).ref_count == 0)
            .map(|id| Reverse((self.tree.node(id).timestamp, id.0)))
            .collect();

        let mut out = SwaEvicted::default();
        let mut freed = 0usize;
        while freed < num_tokens {
            let Some(Reverse((_, raw))) = heap.pop() else {
                break;
            };
            let id = NodeId(raw);
            let node = self.tree.node(id);
            // The heap can hold entries a cascade already took, or a
            // node that gained a child; skip rather than assert.
            if node.ref_count != 0 || !node.is_leaf() || node.is_root() {
                continue;
            }
            let length = node.length();
            freed += length;
            out.kv_indices.extend_from_slice(&node.value);
            self.full_evictable -= length;
            if !node.swa_tombstone {
                out.swa_indices.extend_from_slice(&node.value);
                self.swa_evictable -= length;
            }
            let parent = self.tree.unlink(id);
            let (parent, cascaded) = self.cascade_tombstone_leaves(parent, &mut out.kv_indices);
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

    /// Free at least `num_tokens` of *window* KV.
    ///
    /// Unlike the full pass this may take **interior** nodes: their
    /// window state is tombstoned in place and their full KV stays, so
    /// a later prompt still reuses them for its full-attention layers.
    /// Only a node that is already a free leaf is removed outright.
    pub fn evict_swa(&mut self, num_tokens: usize) -> SwaEvicted {
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = self
            .tree
            .walk()
            .into_iter()
            .filter(|id| {
                !self.tree.node(*id).swa_tombstone && self.tree.node(*id).swa_ref_count == 0
            })
            .map(|id| Reverse((self.tree.node(id).timestamp, id.0)))
            .collect();

        let mut out = SwaEvicted::default();
        let mut freed = 0usize;
        while freed < num_tokens {
            let Some(Reverse((_, raw))) = heap.pop() else {
                break;
            };
            let id = NodeId(raw);
            let node = self.tree.node(id);
            if node.swa_tombstone || node.swa_ref_count != 0 || node.is_root() {
                continue;
            }
            let length = node.length();
            if node.is_leaf() && node.ref_count == 0 {
                out.kv_indices.extend_from_slice(&node.value);
                out.swa_indices.extend_from_slice(&node.value);
                self.full_evictable -= length;
                self.swa_evictable -= length;
                freed += length;
                self.tree.node_mut(id).swa_tombstone = true;
                let parent = self.tree.unlink(id);
                self.cascade_tombstone_leaves(parent, &mut out.kv_indices);
            } else {
                out.swa_indices.extend_from_slice(&node.value);
                self.swa_evictable -= length;
                freed += length;
                self.tree.node_mut(id).swa_tombstone = true;
            }
        }
        out
    }

    /// Walk up from `parent` removing tombstoned free leaves, and
    /// return the ancestor that survived.
    ///
    /// The caller **must** use the returned node: continuing with the
    /// one it passed in would re-free a node this loop already took.
    fn cascade_tombstone_leaves(
        &mut self,
        parent: NodeId,
        kv_out: &mut Vec<u32>,
    ) -> (NodeId, usize) {
        let mut parent = parent;
        let mut freed = 0usize;
        while {
            let node = self.tree.node(parent);
            node.swa_tombstone && node.is_leaf() && node.ref_count == 0 && !node.is_root()
        } {
            let length = self.tree.node(parent).length();
            kv_out.extend_from_slice(&self.tree.node(parent).value);
            self.full_evictable -= length;
            freed += length;
            parent = self.tree.unlink(parent);
        }
        (parent, freed)
    }

    /// Give back the window state of everything before `keep_from`,
    /// keeping the full KV.
    ///
    /// Called when a request finishes: the head of a long prompt is
    /// outside any future window, so its window slots are dead weight,
    /// while its full KV is exactly what the next turn of the same
    /// conversation will reuse. Skips locked, already-tombstoned and
    /// leaf nodes -- a tombstoned leaf would be reclaimed whole by the
    /// next full eviction, losing the full KV this exists to keep.
    ///
    /// Idempotent: calling it again trims nothing more.
    pub fn trim_head_swa(&mut self, input_ids: &[u32], keep_from: usize) -> Vec<u32> {
        if keep_from == 0 {
            return Vec::new();
        }
        assert_eq!(
            keep_from % self.tree.page_size(),
            0,
            "keep_from must be page-aligned"
        );
        // Force a node boundary at `keep_from` so the trim can stop
        // exactly there.
        self.match_prefix(&input_ids[..keep_from]);

        let mut freed = Vec::new();
        let mut node = ROOT;
        let mut pos = 0usize;
        while pos < keep_from {
            let Some(child) = self.tree.child(node, &input_ids[pos..]) else {
                break;
            };
            if pos + self.tree.node(child).length() > keep_from {
                break;
            }
            let c = self.tree.node(child);
            if !c.swa_tombstone && c.swa_ref_count == 0 && !c.is_leaf() {
                let length = c.length();
                freed.extend_from_slice(&c.value);
                self.swa_evictable -= length;
                self.tree.node_mut(child).swa_tombstone = true;
            }
            pos += self.tree.node(child).length();
            node = child;
        }
        freed
    }

    /// Both ledgers recomputed from raw node state, plus the two rules
    /// that make the pair coherent.
    pub fn check_integrity(&self) {
        self.tree.check_structure();
        let (mut full_e, mut full_p, mut swa_e, mut swa_p) = (0, 0, 0, 0);
        for id in self.tree.walk() {
            let node = self.tree.node(id);
            assert!(
                node.ref_count >= node.swa_ref_count,
                "the window lock is a suffix of the full lock"
            );
            if node.swa_tombstone {
                assert_eq!(
                    node.swa_ref_count, 0,
                    "a tombstoned node cannot hold a window lock"
                );
            }
            if node.ref_count == 0 {
                full_e += node.length();
            } else {
                full_p += node.length();
            }
            if !node.swa_tombstone {
                if node.swa_ref_count == 0 {
                    swa_e += node.length();
                } else {
                    swa_p += node.length();
                }
            }
        }
        assert_eq!(full_e, self.full_evictable, "full_evictable drifted");
        assert_eq!(full_p, self.full_protected, "full_protected drifted");
        assert_eq!(swa_e, self.swa_evictable, "swa_evictable drifted");
        assert_eq!(swa_p, self.swa_protected, "swa_protected drifted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = 4;
    const W: usize = 8;

    fn ids(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn pages(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    fn seeded(cache: &mut SwaRadixCache, prompt: &[u32], base: u32) {
        cache.insert(prompt, &pages(base, prompt.len()), 0, 0);
    }

    #[test]
    fn a_tombstone_free_prefix_is_reusable_however_short() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..4), 0);
        let m = cache.match_prefix(&ids(0..4));
        assert_eq!(m.cached_len, 4, "one page, shorter than the window");
        assert_eq!(m.kv_indices, pages(0, 4));
        cache.check_integrity();
    }

    /// The rule the whole module turns on: a live run of *exactly* one
    /// window is reusable. Under a strict `>` this collapses to zero,
    /// which is the state a trimmed prompt is normally in.
    #[test]
    fn a_live_run_of_exactly_one_window_is_reusable() {
        let mut cache = SwaRadixCache::new(P, W);
        // head(4) [tombstoned] + live(8) == exactly one window
        seeded(&mut cache, &ids(0..12), 0);
        let trimmed = cache.trim_head_swa(&ids(0..12), 4);
        assert_eq!(trimmed, pages(0, 4), "the head gave up its window slots");

        let m = cache.match_prefix(&ids(0..12));
        assert_eq!(m.cached_len, 12, "the run after the tombstone covers W");
        cache.check_integrity();
    }

    /// ... and a live run shorter than the window is not reusable at
    /// all: a window layer would read freed state.
    #[test]
    fn a_live_run_shorter_than_the_window_is_not_reusable() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..8), 0);
        cache.trim_head_swa(&ids(0..8), 4);
        let m = cache.match_prefix(&ids(0..8));
        assert_eq!(m.cached_len, 0, "only one page is live, the window is two");
        assert_eq!(m.node, ROOT);
        cache.check_integrity();
    }

    /// A trim gives up window slots and *keeps* every full page -- that
    /// is the entire reason it exists.
    #[test]
    fn trimming_the_head_costs_no_full_kv_and_is_idempotent() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..16), 0);
        let before = cache.full_evictable_size();
        let first = cache.trim_head_swa(&ids(0..16), 8);
        assert!(!first.is_empty());
        assert_eq!(cache.full_evictable_size(), before, "no full KV given up");
        assert!(cache.trim_head_swa(&ids(0..16), 8).is_empty());
        cache.check_integrity();
    }

    /// A window eviction on an interior node keeps its full KV, so a
    /// full-attention reuse of that prefix still works.
    #[test]
    fn a_window_eviction_tombstones_in_place_and_frees_no_pages() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..8), 0);
        cache.match_prefix(&ids(0..4)); // split into two nodes
                                        // Stamp the whole path: the root-side half is now the oldest,
                                        // which is the node furthest outside any future window.
        cache.match_prefix(&ids(0..8));
        let full_before = cache.full_evictable_size();

        let out = cache.evict_swa(4);
        assert!(out.kv_indices.is_empty(), "no page left the tree");
        assert_eq!(out.swa_indices.len(), 4);
        assert_eq!(cache.full_evictable_size(), full_before);
        cache.check_integrity();
    }

    #[test]
    fn a_full_eviction_releases_both_currencies() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..4), 0);
        let out = cache.evict_full(4);
        assert_eq!(out.kv_indices, pages(0, 4));
        assert_eq!(out.swa_indices, pages(0, 4), "a live node owned both");
        assert_eq!(cache.full_evictable_size(), 0);
        assert_eq!(cache.swa_evictable_size(), 0);
        cache.check_integrity();
    }

    /// A tombstoned interior node holds full KV that nothing can reach
    /// once its subtree is gone, so the full pass takes it in the same
    /// call.
    #[test]
    fn full_eviction_cascades_through_exposed_tombstones() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..12), 0);
        cache.match_prefix(&ids(0..4));
        cache.match_prefix(&ids(0..8));
        cache.trim_head_swa(&ids(0..12), 8);

        let out = cache.evict_full(4);
        assert_eq!(out.kv_indices.len(), 12, "the leaf plus both tombstones");
        assert_eq!(cache.full_evictable_size(), 0);
        cache.check_integrity();
    }

    #[test]
    fn a_locked_path_survives_both_evictors() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..16), 0);
        let m = cache.match_prefix(&ids(0..16));
        let uuid = cache.inc_lock(m.node);
        assert!(uuid.is_some(), "a 16-token path covers the 8-token window");

        assert!(cache.evict_full(16).kv_indices.is_empty());
        let windowed = cache.evict_swa(16);
        assert!(
            windowed.swa_indices.len() <= 16 - W,
            "the pinned window is never tombstoned"
        );
        cache.check_integrity();

        cache.dec_lock(m.node, uuid);
        assert_eq!(cache.full_protected_size(), 0);
        assert_eq!(cache.swa_protected_size(), 0);
        cache.check_integrity();
    }

    /// Two readers at different depths get different boundaries, and
    /// releasing the deeper one leaves the shallower one's window
    /// pinned.
    #[test]
    fn one_readers_unlock_does_not_release_anothers_window() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..24), 0);
        let shallow = cache.match_prefix(&ids(0..12));
        let deep = cache.match_prefix(&ids(0..24));
        let shallow_uuid = cache.inc_lock(shallow.node);
        let deep_uuid = cache.inc_lock(deep.node);
        assert_ne!(shallow_uuid, deep_uuid);

        cache.dec_lock(deep.node, deep_uuid);
        assert!(
            cache.swa_protected_size() >= W,
            "the shallow reader still holds a whole window"
        );
        cache.check_integrity();
    }

    /// A lock taken before a split still releases exactly its own
    /// window -- the reason `split_at` migrates the boundary handle
    /// root-side.
    #[test]
    fn a_lock_survives_a_split_at_its_own_window_boundary() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..16), 0);
        let m = cache.match_prefix(&ids(0..16));
        let uuid = cache.inc_lock(m.node);
        // Split inside the locked path.
        cache.match_prefix(&ids(0..4));
        cache.dec_lock(m.node, uuid);
        assert_eq!(cache.swa_protected_size(), 0);
        assert_eq!(cache.full_protected_size(), 0);
        cache.check_integrity();
    }

    /// The insert frontier: a node the request still has window state
    /// for comes back to life, and the pages it was holding go back to
    /// the pool.
    #[test]
    fn insert_revives_a_tombstone_the_request_can_pay_for() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..12), 0);
        cache.trim_head_swa(&ids(0..12), 8);
        assert_eq!(cache.revives(), 0);

        let fresh = pages(100, 12);
        let res = cache.insert(&ids(0..12), &fresh, 0, 0);
        assert_eq!(cache.revives(), 1, "the tombstoned head came back");
        // Each revived node adopted the request's pages and handed its
        // stale ones back.
        assert!(res.freed.iter().any(|p| *p < 100), "stale pages returned");
        assert_eq!(res.matched_len, 12);
        cache.check_integrity();

        let m = cache.match_prefix(&ids(0..12));
        assert_eq!(m.cached_len, 12, "and the whole prefix is reusable again");
    }

    /// A tombstone the request has no window state for stays a
    /// tombstone, and the request's pages for it are duplicates.
    #[test]
    fn insert_keeps_a_tombstone_the_request_cannot_pay_for() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..12), 0);
        cache.trim_head_swa(&ids(0..12), 8);

        let fresh = pages(100, 12);
        let res = cache.insert(&ids(0..12), &fresh, 12, 0);
        assert_eq!(cache.revives(), 0);
        assert_eq!(res.freed, fresh, "every page handed back as a duplicate");
        cache.check_integrity();
    }

    /// Pages inside the request's own reused prefix are the tree's own
    /// pages: freeing them would free memory the tree is still using.
    #[test]
    fn insert_frees_nothing_inside_the_reused_prefix() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..8), 0);
        let m = cache.match_prefix(&ids(0..8));

        let mut supplied = m.kv_indices.clone();
        supplied.extend(pages(100, 4));
        let mut prompt = ids(0..8);
        prompt.extend(ids(50..54));

        let res = cache.insert(&prompt, &supplied, 0, 8);
        assert!(res.freed.is_empty(), "nothing to hand back");
        cache.check_integrity();
    }

    /// A leaf must never be a tombstone: the windowed match and the
    /// eviction cascade both depend on it.
    #[test]
    fn the_suffix_clamp_never_creates_a_tombstoned_leaf() {
        for n_pages in 1..=3usize {
            let mut cache = SwaRadixCache::new(P, W);
            let len = n_pages * P;
            let prompt = ids(0..len as u32);
            // Claim the request lost its window state for everything.
            cache.insert(&prompt, &pages(0, len), len, 0);
            for id in cache.tree().walk() {
                let node = cache.tree().node(id);
                assert!(
                    !(node.is_leaf() && node.swa_tombstone),
                    "n_pages={n_pages}: a leaf must carry live window state"
                );
            }
            cache.check_integrity();
        }
    }

    /// Repeated window pressure must converge and conserve: every page
    /// is either in the tree or handed back, never both.
    #[test]
    fn window_pressure_drains_without_leaking_or_double_freeing() {
        let mut cache = SwaRadixCache::new(P, W);
        seeded(&mut cache, &ids(0..32), 0);

        let mut released: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let out = cache.evict_swa(8);
            released.extend(out.kv_indices);
            cache.check_integrity();
        }
        let out = cache.evict_full(32);
        released.extend(out.kv_indices);
        cache.check_integrity();

        released.sort_unstable();
        released.dedup();
        assert_eq!(released, pages(0, 32), "every page exactly once");
        assert_eq!(cache.full_evictable_size(), 0);
        assert_eq!(cache.swa_evictable_size(), 0);
    }
}
