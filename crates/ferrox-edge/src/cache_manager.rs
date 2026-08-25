//! The page ledger: who owns which KV page, and what a request must
//! hand back when it commits.
//!
//! [`crate::radix`] answers "what is already computed". It does not
//! allocate anything, and it deliberately does not know where pages
//! come from. This module is the other half: the free list, the
//! per-request page table, and — the part that is easy to get wrong —
//! the handover when a request publishes its work.
//!
//! # The handover, and the leak that lives in it
//!
//! When a request finishes computing a prefix and offers it to the
//! cache, three spans of its page table mean three different things:
//!
//! ```text
//! [0 .............. locked) already shared; the request never owned it
//! [locked ......... shared) the tree ALREADY had this -- the request's
//!                           pages here are duplicates and must be freed
//! [shared ....... inserted) newly published; the tree owns it now
//! [inserted ......... tail) still private; freed only when finishing
//! ```
//!
//! The middle span is the trap. Two requests computed the same
//! continuation concurrently; one got there first, so the second's
//! pages for it are redundant. [`crate::radix::InsertResult::cached_len`]
//! is documented as *"how much you must free"* rather than "how much I
//! stored" precisely because reading it the other way leaks a page per
//! concurrent duplicate, forever, and the symptom is a pool that fills
//! up under load with no request holding it.
//!
//! And freeing is not enough: the request is still *reading* those
//! positions, so its page table has to be **repointed** at the tree's
//! canonical pages for that span before the duplicates go back on the
//! free list. Free without repointing and the request reads pages that
//! have been handed to someone else.
//!
//! # Why frees are deferred across a batch
//!
//! [`CacheManager::begin_batch`] holds freed pages back until
//! [`CacheManager::end_batch`]. Within one batch several requests
//! commit, and a page freed by the first must not be handed to the
//! third while the second is still reading it. Deferring to the end of
//! the batch is the cheap way to be sure.
//!
//! # Three trees, three commits
//!
//! [`PrefixCache`] says which tree this manager drives, and that is not
//! a formality: a commit settles as many currencies as the tree holds,
//! and the extra currency's rules are where the interesting failures
//! live.
//!
//! ## The window path: tombstone, do not adopt
//!
//! A window request frees its own out-of-window slots as it goes
//! ([`CacheManager::slide_window`] advances
//! [`SequenceState::released_window`], FreeToken's
//! `swa_evicted_seqlen`). Those positions still hold **full** KV -- the
//! full-attention layers read it -- so their pages are published
//! normally. Their **window** slots are gone.
//!
//! So the commit hands that frontier to
//! [`crate::radix::SwaRadixCache::insert`], which marks everything
//! below it a tombstone instead of adopting the request's pages as
//! window-live. Skip that and the tree records a live window node whose
//! slots were handed back to the pool: the node still matches, the next
//! request reuses it, and its window layers gather through a mapping
//! that now reads the reserved sentinel
//! ([`crate::window_pool::NO_SLOT`]). That is slot 0's bytes in every
//! window head, on a path that never errors -- silent corruption, not a
//! crash. The frontier is honoured on **unfinished** commits too:
//! chunked prefill slides between chunks, so a request's frontier is
//! already non-zero long before it finishes.
//!
//! ## The hybrid path: lock before you allocate
//!
//! A hybrid commit *donates* a recurrent-state snapshot to the tree and
//! then takes a fresh slot to replace it. Taking that slot can evict
//! ([`crate::radix::HybridRadixCache::evict_mamba`]), and the LRU
//! candidate set includes the node that was just donated to -- which is
//! a childless leaf holding the request's KV. Evicting it removes the
//! node **and returns its pages to the pool**, under a request that is
//! still decoding into them.
//!
//! [`CacheManager::commit`] therefore locks the committed node before
//! it allocates, never after. The order is the rule; the allocation
//! failing is not.
//!
//! Ported 1:1 from FreeToken's `scheduler/cache.py::CacheManager`
//! (Apache-2.0): `cache_req`, `_cache_req_swa`, `_cache_req_hybrid`,
//! `ensure_swa_slots`, `ensure_mamba_slots` and `_free_req_slots`. See
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::VecDeque;

use crate::anchor::{decode_slide, PingPong, SlideDecision, SlidingRequest, WindowPolicy};
use crate::pool::SWA_RETAIN_GAP;
use crate::radix::{
    align_ceil, align_down, HybridRadixCache, NodeId, RadixCache, RadixTree, SwaRadixCache, ROOT,
};
use crate::window_pool::{WindowPoolExhausted, WindowSlotPool};

/// Which prefix cache a [`CacheManager`] drives.
///
/// The three trees are separate types rather than one parameterized
/// cache ([`crate::radix`] says why: what may be evicted differs at
/// every step), so the manager holds them in an enum and dispatches
/// per commit path. The paths genuinely differ -- a window commit
/// tombstones, a hybrid commit donates -- and folding them into one
/// branchy function is exactly how a two-currency rule gets lost.
#[derive(Debug)]
pub enum PrefixCache {
    /// One currency: full KV pages.
    Plain(RadixCache),
    /// Two: full KV pages and sliding-window KV, evicted independently.
    Swa(SwaRadixCache),
    /// Two: full KV pages and recurrent-state snapshots.
    Hybrid(HybridRadixCache),
}

impl PrefixCache {
    pub fn as_plain(&self) -> Option<&RadixCache> {
        match self {
            PrefixCache::Plain(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_swa(&self) -> Option<&SwaRadixCache> {
        match self {
            PrefixCache::Swa(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_hybrid(&self) -> Option<&HybridRadixCache> {
        match self {
            PrefixCache::Hybrid(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn tree(&self) -> &RadixTree {
        match self {
            PrefixCache::Plain(cache) => cache.tree(),
            PrefixCache::Swa(cache) => cache.tree(),
            PrefixCache::Hybrid(cache) => cache.tree(),
        }
    }

    pub fn page_size(&self) -> usize {
        self.tree().page_size()
    }

    /// Tokens held by nodes no request is reading -- the number
    /// admission may count as available, because a cached prefix nobody
    /// reads is memory rather than occupancy.
    ///
    /// Always the **full**-KV ledger: window slots and recurrent slots
    /// are different pools and do not convert into pages.
    pub fn evictable_tokens(&self) -> usize {
        match self {
            PrefixCache::Plain(cache) => cache.evictable_size(),
            PrefixCache::Swa(cache) => cache.full_evictable_size(),
            PrefixCache::Hybrid(cache) => cache.full_evictable_size(),
        }
    }

    /// Tokens the tree holds at all, read or not.
    pub fn total_tokens(&self) -> usize {
        match self {
            PrefixCache::Plain(cache) => cache.total_size(),
            PrefixCache::Swa(cache) => cache.full_evictable_size() + cache.full_protected_size(),
            PrefixCache::Hybrid(cache) => cache.full_evictable_size() + cache.full_protected_size(),
        }
    }

    pub fn check_integrity(&self) {
        match self {
            PrefixCache::Plain(cache) => cache.check_integrity(),
            PrefixCache::Swa(cache) => cache.check_integrity(),
            PrefixCache::Hybrid(cache) => cache.check_integrity(),
        }
    }
}

/// A hybrid request's recurrent-state slots.
///
/// The names map onto FreeToken's `Req` fields: `live` is
/// `linear_slot_idx`, `ping_pong` is `mamba_ping_pong` (with
/// [`PingPong::next`] as `mamba_next_track_idx`), and `pending_freeze`
/// is `mamba_last_track_seqlen` -- the sequence length of a snapshot
/// the forward has already frozen into the idle ping-pong slot and that
/// has not been handed to the tree yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentSlots {
    /// The slot holding the live recurrent state.
    pub live: u32,
    /// The two snapshot slots the request alternates between. `None`
    /// once they have been handed back -- which is what stops a second
    /// pass from double-freeing them.
    pub ping_pong: Option<PingPong>,
    /// The length a frozen snapshot is valid at, waiting to be donated.
    pub pending_freeze: Option<usize>,
}

impl RecurrentSlots {
    pub fn new(live: u32, ping_pong: PingPong) -> Self {
        RecurrentSlots {
            live,
            ping_pong: Some(ping_pong),
            pending_freeze: None,
        }
    }

    /// The slot the forward just wrote: the one the ping-pong is *not*
    /// pointing at, because a snapshot flips it after writing.
    pub fn frozen_slot(&self) -> Option<u32> {
        self.ping_pong.map(|pp| pp.slots[(pp.next + 1) % 2])
    }

    fn frozen_index(&self) -> usize {
        self.ping_pong.map_or(0, |pp| (pp.next + 1) % 2)
    }
}

/// The pool could not satisfy an allocation even after evicting
/// everything evictable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfMemory {
    /// Not enough KV pages, and nothing left to evict.
    Pages { needed: usize, available: usize },
    /// Not enough window slots.
    Window(WindowPoolExhausted),
}

impl std::fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutOfMemory::Pages { needed, available } => write!(
                f,
                "KV pool exhausted: need {needed} pages, {available} available after eviction"
            ),
            OutOfMemory::Window(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for OutOfMemory {}

/// One in-flight request's claim on the pools.
///
/// `slot` indexes the page table; everything else is what the manager
/// needs to know to hand the request's pages back correctly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceState {
    pub slot: u32,
    /// Positions whose KV is computed and valid.
    pub cached_len: usize,
    /// The prefix-cache node this request holds a lock on, and how long
    /// its path is. `ROOT`/0 before the first match.
    pub locked_node: NodeId,
    pub locked_len: usize,
    /// How far this request's *window* state has been released.
    ///
    /// This is FreeToken's `swa_evicted_seqlen`, and it is not only a
    /// slide bookkeeping number: a window commit hands it to the tree
    /// so that everything below it is tombstoned rather than adopted.
    /// See the module docs.
    pub released_window: usize,
    /// Window models: the handle [`crate::radix::SwaRadixCache::inc_lock`]
    /// returned, so this reader's unlock releases its own window and not
    /// a deeper reader's.
    pub swa_lock: Option<u64>,
    /// Window models: how much of `input_ids` is prompt rather than
    /// generated output, so a finish knows which path to soft-pin for
    /// the next turn. FreeToken reads `max_device_len - output_len`.
    pub prompt_len: usize,
    /// Hybrid models: the request's recurrent-state slots.
    pub recurrent: Option<RecurrentSlots>,
    /// Decode steps taken, for the slide's first-step guard.
    pub decode_step: usize,
}

impl SequenceState {
    pub fn new(slot: u32) -> Self {
        SequenceState {
            slot,
            cached_len: 0,
            locked_node: ROOT,
            locked_len: 0,
            released_window: 0,
            swa_lock: None,
            prompt_len: 0,
            recurrent: None,
            decode_step: 0,
        }
    }
}

/// What a prefix lookup found, in the manager's terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Positions the cache can supply, already locked.
    pub cached_len: usize,
    pub node: NodeId,
    /// Hybrid models: the recurrent-state slot to resume from. `None`
    /// on the other two caches, and on a hybrid miss.
    pub snapshot: Option<u32>,
}

/// What a commit did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitOutcome {
    /// Positions the tree already held, whose duplicate pages went
    /// back. The number that says how much concurrent work was
    /// redundant.
    pub duplicate_len: usize,
    /// Positions the tree owns after this commit.
    pub published_len: usize,
    /// Pages handed back to the pool.
    pub pages_freed: usize,
    /// Hybrid models: the tree took a recurrent snapshot this commit,
    /// so the slot it was offered now belongs to the tree and the
    /// request must not free it.
    pub snapshot_donated: bool,
}

/// The free list, the page table, and the prefix cache they serve.
#[derive(Debug)]
pub struct CacheManager {
    page_size: usize,
    num_pages: usize,
    max_seq_len: usize,
    /// Free page bases, in KV-pool token coordinates: page `p` owns
    /// locations `p .. p + page_size`.
    free_pages: VecDeque<u32>,
    /// Slot-major, `max_seq_len` wide: position -> KV location.
    page_table: Vec<u32>,
    cache: PrefixCache,
    window: Option<WindowSlotPool>,
    /// Hybrid models: free recurrent-state slots.
    state_slots: VecDeque<u32>,
    num_state_slots: usize,
    /// Pages freed since `begin_batch`, held back until `end_batch`.
    deferred: Option<Vec<u32>>,
}

impl CacheManager {
    pub fn new(page_size: usize, num_pages: usize, max_seq_len: usize, num_slots: usize) -> Self {
        CacheManager::with_cache(
            page_size,
            num_pages,
            max_seq_len,
            num_slots,
            PrefixCache::Plain(RadixCache::new(page_size)),
        )
    }

    /// A manager for a sliding-window model.
    ///
    /// The window pool is taken here rather than bolted on afterwards
    /// because this path cannot work without one: every free in a
    /// window commit settles *both* pools, and a manager that silently
    /// had no second pool would leak a window slot per committed page
    /// while reporting nothing wrong.
    pub fn new_swa(
        page_size: usize,
        num_pages: usize,
        max_seq_len: usize,
        num_slots: usize,
        sliding_window: usize,
        window: WindowSlotPool,
    ) -> Self {
        CacheManager::with_cache(
            page_size,
            num_pages,
            max_seq_len,
            num_slots,
            PrefixCache::Swa(SwaRadixCache::new(page_size, sliding_window)),
        )
        .with_window_pool(window)
    }

    /// A manager for a hybrid attention/recurrent model, owning
    /// `num_state_slots` recurrent-state slots.
    ///
    /// The slots are opaque ids: this module hands them out, takes them
    /// back, and never dereferences them -- the same contract
    /// [`crate::radix::HybridRadixCache`] keeps with the tree.
    pub fn new_hybrid(
        page_size: usize,
        num_pages: usize,
        max_seq_len: usize,
        num_slots: usize,
        num_state_slots: usize,
    ) -> Self {
        let mut manager = CacheManager::with_cache(
            page_size,
            num_pages,
            max_seq_len,
            num_slots,
            PrefixCache::Hybrid(HybridRadixCache::new(page_size)),
        );
        manager.state_slots = (0..num_state_slots as u32).collect();
        manager.num_state_slots = num_state_slots;
        manager
    }

    fn with_cache(
        page_size: usize,
        num_pages: usize,
        max_seq_len: usize,
        num_slots: usize,
        cache: PrefixCache,
    ) -> Self {
        assert!(
            page_size > 0 && num_pages > 0,
            "an empty pool serves nothing"
        );
        let stride = align_ceil(max_seq_len, page_size);
        CacheManager {
            page_size,
            num_pages,
            max_seq_len: stride,
            free_pages: (0..num_pages).map(|p| (p * page_size) as u32).collect(),
            page_table: vec![0; num_slots * stride],
            cache,
            window: None,
            state_slots: VecDeque::new(),
            num_state_slots: 0,
            deferred: None,
        }
    }

    /// Attach a window pool, for a sliding-window model.
    pub fn with_window_pool(mut self, pool: WindowSlotPool) -> Self {
        assert_eq!(
            pool.page_size(),
            self.page_size,
            "the window pool and the KV pool must page identically"
        );
        self.window = Some(pool);
        self
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// The plain prefix cache this manager drives.
    ///
    /// Panics on a window or hybrid manager: those trees hold a second
    /// currency per node, so there is no honest `&RadixCache` to hand
    /// back. Use [`prefix_cache`](Self::prefix_cache),
    /// [`swa_cache`](Self::swa_cache) or
    /// [`hybrid_cache`](Self::hybrid_cache) there.
    pub fn cache(&self) -> &RadixCache {
        self.cache
            .as_plain()
            .expect("this manager does not drive the plain radix cache")
    }

    pub fn prefix_cache(&self) -> &PrefixCache {
        &self.cache
    }

    /// The window prefix cache. Panics unless this manager drives one.
    pub fn swa_cache(&self) -> &SwaRadixCache {
        self.cache
            .as_swa()
            .expect("this manager does not drive the window radix cache")
    }

    /// The hybrid prefix cache. Panics unless this manager drives one.
    pub fn hybrid_cache(&self) -> &HybridRadixCache {
        self.cache
            .as_hybrid()
            .expect("this manager does not drive the hybrid radix cache")
    }

    fn plain_mut(&mut self) -> &mut RadixCache {
        match &mut self.cache {
            PrefixCache::Plain(cache) => cache,
            _ => panic!("this manager does not drive the plain radix cache"),
        }
    }

    fn swa_mut(&mut self) -> &mut SwaRadixCache {
        match &mut self.cache {
            PrefixCache::Swa(cache) => cache,
            _ => panic!("this manager does not drive the window radix cache"),
        }
    }

    fn hybrid_mut(&mut self) -> &mut HybridRadixCache {
        match &mut self.cache {
            PrefixCache::Hybrid(cache) => cache,
            _ => panic!("this manager does not drive the hybrid radix cache"),
        }
    }

    pub fn window_pool(&self) -> Option<&WindowSlotPool> {
        self.window.as_ref()
    }

    /// Recurrent-state slots that can still be handed out.
    pub fn state_slots_available(&self) -> usize {
        self.state_slots.len()
    }

    /// Take one recurrent-state slot, evicting a tree snapshot if the
    /// free list is empty.
    ///
    /// **This can evict**, and that is why every caller in this module
    /// locks what it has just published *before* calling it: the
    /// just-donated node is an unlocked childless leaf, i.e. exactly
    /// what [`crate::radix::HybridRadixCache::evict_mamba`] takes, and
    /// taking it hands the caller's own KV pages back to the pool.
    ///
    /// `None` when nothing is free and nothing is evictable: the
    /// request keeps computing, it just cannot freeze another snapshot.
    pub fn alloc_state_slot(&mut self) -> Option<u32> {
        if self.state_slots.is_empty() {
            self.ensure_state_slots(1);
        }
        self.state_slots.pop_front()
    }

    /// Hand a recurrent-state slot back.
    pub fn free_state_slot(&mut self, slot: u32) {
        self.state_slots.push_back(slot);
    }

    /// Positions that could be served without evicting anything the
    /// cache is protecting: free pages plus what is evictable.
    ///
    /// This is the number admission reads, and it counts evictable
    /// cache as available on purpose — a cached prefix nobody is
    /// reading is memory, not occupancy, and refusing a request because
    /// of it would leave the pool idle at capacity.
    pub fn available_tokens(&self) -> usize {
        self.free_pages.len() * self.page_size + self.cache.evictable_tokens()
    }

    /// `(pages in use, pages total)`, for a status line.
    pub fn page_usage(&self) -> (usize, usize) {
        let cached_pages = self.cache.evictable_tokens() / self.page_size;
        (
            self.num_pages - self.free_pages.len() - cached_pages,
            self.num_pages,
        )
    }

    /// The longest cached prefix of `input_ids`, locked for the caller.
    ///
    /// The **last token is deliberately excluded** from the lookup: a
    /// prompt whose whole length matched would have nothing to compute,
    /// and a forward pass needs at least one position to produce
    /// logits from.
    pub fn match_and_lock(&mut self, seq: &mut SequenceState, input_ids: &[u32]) -> PrefixMatch {
        let lookup = &input_ids[..input_ids.len().saturating_sub(1)];
        let had_lock = seq.locked_len > 0 || seq.locked_node != ROOT;
        let old_node = seq.locked_node;
        let old_swa_lock = seq.swa_lock;

        // The request reads the tree's pages for the part it did not
        // compute, so every branch produces the canonical row for the
        // matched span alongside the handle.
        let (matched, canonical) = match &mut self.cache {
            PrefixCache::Plain(cache) => {
                let matched = cache.match_prefix(lookup);
                cache.lock(matched.node);
                if had_lock {
                    cache.unlock(old_node);
                }
                let canonical = cache.matched_indices(matched.node);
                (
                    PrefixMatch {
                        cached_len: matched.cached_len,
                        node: matched.node,
                        snapshot: None,
                    },
                    canonical,
                )
            }
            PrefixCache::Swa(cache) => {
                let matched = cache.match_prefix(lookup);
                seq.swa_lock = cache.inc_lock(matched.node);
                if had_lock {
                    cache.dec_lock(old_node, old_swa_lock);
                }
                (
                    PrefixMatch {
                        cached_len: matched.cached_len,
                        node: matched.node,
                        snapshot: None,
                    },
                    matched.kv_indices,
                )
            }
            PrefixCache::Hybrid(cache) => {
                let matched = cache.match_prefix(lookup);
                cache.inc_lock(matched.node);
                if had_lock {
                    cache.dec_lock(old_node);
                }
                (
                    PrefixMatch {
                        cached_len: matched.cached_len,
                        node: matched.node,
                        snapshot: matched.mamba_value,
                    },
                    matched.kv_indices,
                )
            }
        };

        seq.locked_node = matched.node;
        seq.locked_len = matched.cached_len;
        seq.cached_len = matched.cached_len;
        self.row_mut(seq.slot)[..canonical.len()].copy_from_slice(&canonical);
        matched
    }

    /// Give `seq` pages up to `upto` positions, evicting if it must.
    pub fn allocate(&mut self, seq: &mut SequenceState, upto: usize) -> Result<(), OutOfMemory> {
        assert!(upto <= self.max_seq_len, "a request past the page table");
        let first_page = seq.cached_len.div_ceil(self.page_size);
        let last_page = upto.div_ceil(self.page_size);
        if last_page <= first_page {
            seq.cached_len = seq.cached_len.max(upto);
            return Ok(());
        }
        let needed = last_page - first_page;
        self.ensure_pages(needed)?;

        let mut locations = Vec::with_capacity(needed * self.page_size);
        let page_size = self.page_size;
        let stride = self.max_seq_len;
        let base = seq.slot as usize * stride;
        for step in 0..needed {
            let page = self
                .free_pages
                .pop_front()
                .expect("ensure_pages promised this many");
            let position = (first_page + step) * page_size;
            for offset in 0..page_size {
                let location = page + offset as u32;
                self.page_table[base + position + offset] = location;
                locations.push(location);
            }
        }

        // A window model's tree holds reclaimable window state of its
        // own: tombstoning an LRU node gives its slots back while its
        // full KV stays reusable, so refusing here while the tree still
        // holds unlocked window state would idle the pool at capacity.
        if self.window.is_some() && matches!(self.cache, PrefixCache::Swa(_)) {
            self.ensure_window_slots(locations.len());
        }

        if let Some(window) = self.window.as_mut() {
            if let Err(exhausted) = window.alloc(&locations) {
                // Hand the pages straight back: a half-allocated
                // request is worse than a refused one.
                for chunk in locations.chunks(page_size) {
                    self.free_pages.push_back(chunk[0]);
                }
                return Err(OutOfMemory::Window(exhausted));
            }
        }
        seq.cached_len = upto;
        Ok(())
    }

    /// Publish `input_ids[..seq.cached_len]` and settle up.
    ///
    /// Returns what was redundant. When `finished`, the request's
    /// private tail goes back too; otherwise the request keeps reading
    /// and its row is repointed at the tree's canonical pages for
    /// whatever the tree already had.
    pub fn commit(
        &mut self,
        seq: &mut SequenceState,
        input_ids: &[u32],
        finished: bool,
    ) -> CommitOutcome {
        match self.cache {
            PrefixCache::Plain(_) => self.commit_plain(seq, input_ids, finished),
            PrefixCache::Swa(_) => self.commit_swa(seq, input_ids, finished),
            PrefixCache::Hybrid(_) => self.commit_hybrid(seq, input_ids, finished),
        }
    }

    fn commit_plain(
        &mut self,
        seq: &mut SequenceState,
        input_ids: &[u32],
        finished: bool,
    ) -> CommitOutcome {
        let publish_len = seq.cached_len.min(input_ids.len());
        let row_snapshot: Vec<u32> = self.row(seq.slot)[..publish_len].to_vec();
        let result = self
            .plain_mut()
            .insert_prefix(&input_ids[..publish_len], &row_snapshot);

        let old_locked = seq.locked_len;
        let locked_node = seq.locked_node;
        self.plain_mut().unlock(locked_node);
        let canonical = self.cache().matched_indices(result.node);

        let mut outcome = CommitOutcome {
            duplicate_len: result.cached_len.saturating_sub(old_locked),
            published_len: result.inserted_len,
            pages_freed: 0,
            snapshot_donated: false,
        };

        // The duplicate span: the tree already had it, so these pages
        // are redundant -- but the request is still reading those
        // positions, so repoint before freeing.
        if result.cached_len > old_locked {
            let duplicates: Vec<u32> = row_snapshot[old_locked..result.cached_len].to_vec();
            if !finished {
                let row = self.row_mut(seq.slot);
                row[old_locked..result.cached_len]
                    .copy_from_slice(&canonical[old_locked..result.cached_len]);
            }
            outcome.pages_freed += self.free_locations(&duplicates);
        }

        if finished {
            // Everything past what the tree took is this request's
            // alone. The end is rounded UP to a whole page: a partial
            // page was charged whole, so it is released whole.
            let tail_end = align_ceil(publish_len, self.page_size).min(self.max_seq_len);
            if tail_end > result.inserted_len {
                let tail: Vec<u32> = self.row(seq.slot)[result.inserted_len..tail_end].to_vec();
                outcome.pages_freed += self.free_locations(&tail);
            }
            seq.locked_node = ROOT;
            seq.locked_len = 0;
        } else {
            let node = result.node;
            self.plain_mut().lock(node);
            seq.locked_node = result.node;
            seq.locked_len = result.inserted_len;
        }
        outcome
    }

    /// The hybrid model's commit: publish the KV *and* donate the
    /// recurrent snapshot that makes it resumable.
    ///
    /// A prefix of a hybrid model is only reusable at a position where
    /// a recurrent snapshot exists ([`crate::radix::HybridRadixCache`]
    /// says why), so the KV reaches the tree as the passenger of a
    /// donation, never on its own.
    ///
    /// Two orderings carry the whole path:
    ///
    /// 1. **Lock the committed node before allocating the replacement
    ///    slot.** When the tree takes the frozen snapshot, the request
    ///    needs a fresh slot to freeze into next time -- and that
    ///    allocation can evict. The just-donated node is an unlocked
    ///    childless leaf, i.e. the ideal `evict_mamba` victim, and
    ///    evicting it returns *this request's* KV pages to the pool
    ///    while it is still decoding into them. Lock first and the
    ///    candidate set cannot contain it.
    /// 2. **On finish, insert the pending freeze before the live
    ///    state.** The pending freeze is a strictly shorter prefix.
    ///    Inserting it first advances the dedup free floor to its
    ///    boundary, so `[freeze_len, cached_len)` is already tree-owned
    ///    when the live donate reports what was duplicated. The other
    ///    order makes the live insert publish the request's pages for
    ///    the whole span and *then* has the shorter insert report that
    ///    same span as a duplicate -- so the tree's own pages go back on
    ///    the free list under it.
    fn commit_hybrid(
        &mut self,
        seq: &mut SequenceState,
        input_ids: &[u32],
        finished: bool,
    ) -> CommitOutcome {
        let page_size = self.page_size;
        let publish_len = seq.cached_len.min(input_ids.len());
        let row: Vec<u32> = self.row(seq.slot)[..publish_len].to_vec();
        let old_locked = seq.locked_len;
        let locked_node = seq.locked_node;
        let mut outcome = CommitOutcome::default();

        if !finished {
            // No tracked boundary was crossed this chunk: the request
            // keeps its pages and commits them later.
            let Some(recurrent) = seq.recurrent else {
                return outcome;
            };
            let Some(freeze_len) = recurrent.pending_freeze else {
                return outcome;
            };
            let Some(frozen) = recurrent.frozen_slot() else {
                return outcome;
            };
            if freeze_len == 0 || freeze_len > publish_len {
                return outcome;
            }
            if align_down(freeze_len, page_size) != freeze_len {
                // The insert would align the key down and attach a state
                // encoding `freeze_len` tokens to a SHORTER node, so a
                // later hit would resume from an over-advanced state.
                // Skip it; the next aligned boundary commits instead.
                if let Some(rec) = seq.recurrent.as_mut() {
                    rec.pending_freeze = None;
                }
                return outcome;
            }

            let result =
                self.hybrid_mut()
                    .insert(&input_ids[..freeze_len], &row[..freeze_len], frozen);
            self.hybrid_mut().dec_lock(locked_node);
            let dup_end = result.matched_len.max(old_locked);
            outcome.duplicate_len = dup_end - old_locked;
            outcome.published_len = freeze_len;
            outcome.pages_freed += self.free_locations(&row[old_locked..dup_end]);

            // Lock BEFORE the replacement allocation below. See the
            // method docs: the alloc can evict this very node.
            let matched = self.hybrid_mut().match_prefix(&input_ids[..freeze_len]);
            let repoint_end = dup_end.min(matched.kv_indices.len());
            if repoint_end > old_locked {
                let canonical = matched.kv_indices[old_locked..repoint_end].to_vec();
                self.row_mut(seq.slot)[old_locked..repoint_end].copy_from_slice(&canonical);
            }
            seq.locked_node = matched.node;
            seq.locked_len = matched.cached_len;
            self.hybrid_mut().inc_lock(matched.node);

            if !result.snapshot_exists {
                outcome.snapshot_donated = true;
                let index = recurrent.frozen_index();
                // `None` means the pool is empty and nothing is
                // evictable: the request runs on without a spare slot
                // rather than stealing a slot the tree is serving from.
                match self.alloc_state_slot() {
                    Some(replacement) => {
                        if let Some(rec) = seq.recurrent.as_mut() {
                            if let Some(pp) = rec.ping_pong.as_mut() {
                                pp.slots[index] = replacement;
                            }
                        }
                    }
                    None => {
                        if let Some(rec) = seq.recurrent.as_mut() {
                            rec.ping_pong = None;
                        }
                    }
                }
            }
            if let Some(rec) = seq.recurrent.as_mut() {
                rec.pending_freeze = None;
            }
            return outcome;
        }

        let mut free_upto = old_locked;
        let pending = seq
            .recurrent
            .and_then(|rec| rec.pending_freeze.map(|len| (len, rec.frozen_slot())));
        if let Some((freeze_len, Some(frozen))) = pending {
            if freeze_len > 0
                && freeze_len <= publish_len
                && align_down(freeze_len, page_size) == freeze_len
            {
                let result =
                    self.hybrid_mut()
                        .insert(&input_ids[..freeze_len], &row[..freeze_len], frozen);
                outcome.snapshot_donated |= !result.snapshot_exists;
                // The frozen slot is consumed either way -- taken by the
                // tree or handed back here -- and both ping-pong refs
                // are dropped, so the free below cannot double-free.
                if let Some(pp) = seq.recurrent.and_then(|rec| rec.ping_pong) {
                    for slot in pp.slots {
                        if result.snapshot_exists || slot != frozen {
                            self.free_state_slot(slot);
                        }
                    }
                }
                if let Some(rec) = seq.recurrent.as_mut() {
                    rec.ping_pong = None;
                    rec.pending_freeze = None;
                }
                let dup_end = result.matched_len.max(free_upto);
                outcome.duplicate_len += dup_end - free_upto;
                outcome.pages_freed += self.free_locations(&row[free_upto..dup_end]);
                free_upto = free_upto.max(freeze_len);
            }
        }

        // Donate the live slot: the final full-sequence state, at
        // `publish_len`. Only when that is itself the page-aligned node
        // boundary -- a ragged length would attach an over-advanced
        // state to a shorter node, so nothing is published and the
        // request's pages all go back.
        let insert_len = align_down(publish_len, page_size);
        let live = seq.recurrent.map(|rec| rec.live);
        let mut keep_live = false;
        match live {
            Some(live) if insert_len == publish_len && insert_len > 0 => {
                let result =
                    self.hybrid_mut()
                        .insert(&input_ids[..insert_len], &row[..insert_len], live);
                self.hybrid_mut().dec_lock(locked_node);
                let dup_end = result.matched_len.max(free_upto);
                outcome.duplicate_len += dup_end - free_upto;
                outcome.pages_freed += self.free_locations(&row[free_upto..dup_end]);
                outcome.published_len = insert_len;
                keep_live = !result.snapshot_exists;
                outcome.snapshot_donated |= keep_live;
            }
            _ => {
                self.hybrid_mut().dec_lock(locked_node);
                outcome.pages_freed += self.free_locations(&row[free_upto..publish_len]);
            }
        }
        self.free_request_state_slots(seq, keep_live);
        seq.locked_node = ROOT;
        seq.locked_len = 0;
        outcome
    }

    /// Return a finished request's recurrent slots: both ping-pong
    /// slots, plus the live one unless the tree took it.
    ///
    /// Idempotent by construction -- it takes the slots out of the
    /// request -- which is the defence against the abort-then-finish
    /// double free.
    fn free_request_state_slots(&mut self, seq: &mut SequenceState, keep_live: bool) {
        let Some(recurrent) = seq.recurrent.take() else {
            return;
        };
        if let Some(pp) = recurrent.ping_pong {
            for slot in pp.slots {
                self.free_state_slot(slot);
            }
        }
        if !keep_live {
            self.free_state_slot(recurrent.live);
        }
    }

    /// The window model's commit: publish the full KV, tombstone the
    /// window state the request already gave back, settle both pools.
    ///
    /// Three rules, each of which is a bug when skipped:
    ///
    /// 1. **Insert at the request's own frontier.**
    ///    [`SequenceState::released_window`] is how far this request
    ///    slid its own window out; below it the request holds no window
    ///    slots, so the tree must tombstone those positions rather than
    ///    adopt them as window-live. Adopting them publishes a node
    ///    whose window mapping is the reserved sentinel, and the next
    ///    request to match it gathers slot 0 in every window layer --
    ///    silent corruption on a path that never errors. It holds on
    ///    unfinished commits too: chunked prefill slides *between*
    ///    chunks, so a mid-prompt frontier is already non-zero.
    /// 2. **Free in both pools.** Every page the insert hands back is a
    ///    page whose window slot the request also owns. Freeing one
    ///    without the other leaks the pool that was missed. Freeing a
    ///    revived or out-of-window slot is a no-op, so the pass is
    ///    unconditional.
    /// 3. **Re-stamp the prompt path on finish.** Decode never matches
    ///    the prompt again, so after the unlock the prompt path is the
    ///    stalest thing in the tree and the first
    ///    [`crate::radix::SwaRadixCache::evict_swa`] victim -- exactly
    ///    the path the *next* turn rejoins, because a client that drops
    ///    its reasoning block diverges at the prompt end. So the head's
    ///    window state is trimmed eagerly (its full KV stays, which is
    ///    what the next turn actually reuses) and the retained tail is
    ///    re-matched to make it recent. Still unlocked, so real pressure
    ///    can still reclaim it.
    fn commit_swa(
        &mut self,
        seq: &mut SequenceState,
        input_ids: &[u32],
        finished: bool,
    ) -> CommitOutcome {
        let page_size = self.page_size;
        let publish_len = seq.cached_len.min(input_ids.len());
        let insert_len = align_down(publish_len, page_size);
        let tail_end = align_ceil(publish_len, page_size).min(self.max_seq_len);
        let row: Vec<u32> = self.row(seq.slot)[..tail_end].to_vec();
        let old_locked = seq.locked_len;
        let released = seq.released_window;

        let mut matched_len = 0usize;
        let mut freed: Vec<u32> = Vec::new();
        if insert_len > 0 {
            let result = self.swa_mut().insert(
                &input_ids[..insert_len],
                &row[..insert_len],
                released,
                old_locked,
            );
            matched_len = result.matched_len;
            freed = result.freed;
        }
        // Unlock only once every operation on the old handle is done.
        let (locked_node, swa_lock) = (seq.locked_node, seq.swa_lock);
        self.swa_mut().dec_lock(locked_node, swa_lock);

        let mut outcome = CommitOutcome {
            duplicate_len: matched_len.saturating_sub(old_locked),
            published_len: insert_len,
            pages_freed: 0,
            snapshot_donated: false,
        };
        outcome.pages_freed += self.free_locations(&freed);

        let window = self.swa_cache().sliding_window();
        if finished {
            // The page-unaligned tail was never inserted, and the last
            // partial page was charged -- in *both* pools -- as a whole
            // page, so it is released whole or those slots leak.
            if tail_end > insert_len {
                let tail: Vec<u32> = row[insert_len..tail_end].to_vec();
                outcome.pages_freed += self.free_locations(&tail);
            }
            seq.locked_node = ROOT;
            seq.locked_len = 0;
            seq.swa_lock = None;

            // A request that never recorded a prompt length simply gets
            // no soft pin: the retention is an optimization for the next
            // turn, never a correctness condition for this one.
            let prompt_len = align_down(seq.prompt_len.min(input_ids.len()), page_size);
            if prompt_len > 0 {
                let keep_from = align_down(
                    prompt_len.saturating_sub(window + SWA_RETAIN_GAP),
                    page_size,
                );
                if keep_from > 0 {
                    let trimmed = self
                        .swa_mut()
                        .trim_head_swa(&input_ids[..prompt_len], keep_from);
                    // Window slots only: the full KV is the whole point
                    // of keeping the head.
                    self.free_window_only(&trimmed);
                }
                self.swa_mut().match_prefix(&input_ids[..prompt_len]);
            }
        } else {
            // `inc_lock` is node-granular and the insert above just made
            // this whole extend one node, so locking it would pin the
            // entire chunk's window state for all of decode -- though
            // the request reads only its trailing window from here on.
            // An EXTRA match one window back forces a node boundary
            // there (a match splits), so the lock lands on that window
            // alone and the head stays live, unlocked and reclaimable.
            let keep_from = align_down(
                insert_len.saturating_sub(window + SWA_RETAIN_GAP),
                page_size,
            );
            if keep_from > 0 {
                self.swa_mut().match_prefix(&input_ids[..keep_from]);
            }
            let matched = self.swa_mut().match_prefix(&input_ids[..insert_len]);
            // Repoint at the tree's live slots. Any duplicate the insert
            // reclaimed had its window mapping reset to the sentinel;
            // unlike the full pool -- where the KV survives in place
            // until the page is handed out again -- a stale window
            // mapping makes this request's own next gather read slot 0.
            if matched.cached_len > 0 {
                self.row_mut(seq.slot)[..matched.cached_len].copy_from_slice(&matched.kv_indices);
            }
            seq.swa_lock = self.swa_mut().inc_lock(matched.node);
            seq.locked_node = matched.node;
            seq.locked_len = matched.cached_len;
        }
        outcome
    }

    /// Release a window model's state for positions that have slid out
    /// of every layer's reach.
    ///
    /// The full KV stays: the attention layers still read it, and it is
    /// what the prefix cache will hand to the next turn. Only the
    /// window slots go back.
    ///
    /// `None` when the slide's cadence says not this step. When the
    /// returned decision has `drop_anchor` set, the caller must clear
    /// its anchor.
    pub fn slide_window(
        &mut self,
        seq: &mut SequenceState,
        anchor: Option<usize>,
        policy: &WindowPolicy,
        forward_iter: usize,
    ) -> Option<SlideDecision> {
        self.window.as_ref()?;
        let request = SlidingRequest {
            position: seq.cached_len,
            already_released: seq.released_window,
            locked_prefix: seq.locked_len,
            decode_step: seq.decode_step,
        };
        let decision = decode_slide(&request, anchor, policy, forward_iter)?;
        if !decision.frees_nothing() {
            let span: Vec<u32> = self.row(seq.slot)[decision.free_from..decision.free_to].to_vec();
            if let Some(window) = self.window.as_mut() {
                window.free(&span);
            }
            seq.released_window = decision.free_to;
        }
        Some(decision)
    }

    /// Start deferring frees. See the module docs: within one batch a
    /// page freed by one request must not reach another that is still
    /// reading it.
    pub fn begin_batch(&mut self) {
        debug_assert!(self.deferred.is_none(), "a batch is already open");
        self.deferred = Some(Vec::new());
    }

    /// Release everything deferred since [`begin_batch`](Self::begin_batch).
    pub fn end_batch(&mut self) {
        if let Some(pages) = self.deferred.take() {
            for page in pages {
                self.free_pages.push_back(page);
            }
        }
    }

    /// Evict until `needed` pages are free, or report how far it got.
    fn ensure_pages(&mut self, needed: usize) -> Result<(), OutOfMemory> {
        if self.free_pages.len() >= needed {
            return Ok(());
        }
        let short = needed - self.free_pages.len();
        let tokens = short * self.page_size;
        let evictable = self.cache.evictable_tokens();
        if tokens > evictable {
            return Err(OutOfMemory::Pages {
                needed,
                available: self.free_pages.len() + evictable / self.page_size,
            });
        }
        // Each tree gives up whatever else rode along with the pages:
        // the window slots of a live node, the snapshot of a hybrid one.
        let (evicted, states) = match &mut self.cache {
            PrefixCache::Plain(cache) => (cache.evict(tokens), Vec::new()),
            PrefixCache::Swa(cache) => (cache.evict_full(tokens).kv_indices, Vec::new()),
            PrefixCache::Hybrid(cache) => {
                let out = cache.evict_full(tokens);
                (out.kv_indices, out.mamba_slots)
            }
        };
        self.release_evicted_pages(&evicted);
        for slot in states {
            self.free_state_slot(slot);
        }
        debug_assert!(self.free_pages.len() >= needed);
        Ok(())
    }

    /// Tombstone LRU window state until the window pool can serve
    /// `needed` slots, or until nothing is left to tombstone.
    ///
    /// A tombstoned interior node keeps its full KV, so this trades a
    /// prefix's *window* reusability for slots without giving up any
    /// pages. Only an already-free leaf leaves the tree, and then its
    /// pages come back too.
    fn ensure_window_slots(&mut self, needed: usize) {
        loop {
            let available = match self.window.as_ref() {
                Some(window) => window.available(),
                None => return,
            };
            if available >= needed {
                return;
            }
            let evicted = self.swa_mut().evict_swa(needed - available);
            if evicted.swa_indices.is_empty() && evicted.kv_indices.is_empty() {
                return;
            }
            self.free_window_only(&evicted.swa_indices);
            self.release_evicted_pages(&evicted.kv_indices);
        }
    }

    /// Free recurrent-state slots until `n` are available, by
    /// tombstoning LRU tree snapshots. An interior node just gives up
    /// its snapshot and keeps its pages; a free leaf goes away entirely
    /// and its pages come back with it.
    fn ensure_state_slots(&mut self, n: usize) {
        while self.state_slots.len() < n {
            let short = n - self.state_slots.len();
            let evicted = self.hybrid_mut().evict_mamba(short);
            if evicted.mamba_slots.is_empty() {
                return;
            }
            for slot in evicted.mamba_slots {
                self.free_state_slot(slot);
            }
            self.release_evicted_pages(&evicted.kv_indices);
        }
    }

    /// Return pages the *tree* gave up. They are not on any request's
    /// row, so unlike [`free_locations`](Self::free_locations) they go
    /// straight onto the free list rather than through the batch's
    /// deferred list.
    ///
    /// A node holds whole pages and a page's locations are contiguous,
    /// so every `page_size`-th location is a page base.
    fn release_evicted_pages(&mut self, locations: &[u32]) {
        if locations.is_empty() {
            return;
        }
        if let Some(window) = self.window.as_mut() {
            window.free(locations);
        }
        for chunk in locations.chunks(self.page_size) {
            self.free_pages.push_back(chunk[0]);
        }
    }

    /// Give back only the window slots for `locations`, keeping their
    /// full KV pages. Idempotent over the sentinel.
    fn free_window_only(&mut self, locations: &[u32]) {
        if let Some(window) = self.window.as_mut() {
            window.free(locations);
        }
    }

    /// Hand a span of KV locations back, as whole pages. Returns the
    /// page count.
    fn free_locations(&mut self, locations: &[u32]) -> usize {
        if let Some(window) = self.window.as_mut() {
            window.free(locations);
        }
        let mut pages = 0;
        for chunk in locations.chunks(self.page_size) {
            match self.deferred.as_mut() {
                Some(deferred) => deferred.push(chunk[0]),
                None => self.free_pages.push_back(chunk[0]),
            }
            pages += 1;
        }
        pages
    }

    fn row(&self, slot: u32) -> &[u32] {
        let base = slot as usize * self.max_seq_len;
        &self.page_table[base..base + self.max_seq_len]
    }

    fn row_mut(&mut self, slot: u32) -> &mut [u32] {
        let base = slot as usize * self.max_seq_len;
        &mut self.page_table[base..base + self.max_seq_len]
    }

    /// Every page is on the free list or in the cache, exactly once.
    ///
    /// **Idle only.** A request in flight holds pages that are in
    /// neither place — that is what "in flight" means — so running this
    /// mid-batch reports a false leak.
    pub fn check_integrity(&self) {
        assert!(
            self.deferred.is_none(),
            "check_integrity ran inside an open batch"
        );
        self.cache.check_integrity();
        let cached_pages = self.cache.total_tokens() / self.page_size;
        assert_eq!(
            self.free_pages.len() + cached_pages,
            self.num_pages,
            "pages leaked or double-freed: {} free + {} cached != {} total",
            self.free_pages.len(),
            cached_pages,
            self.num_pages
        );
        let mut seen = vec![false; self.num_pages];
        for page in self.free_pages.iter().copied() {
            let index = page as usize / self.page_size;
            assert!(
                !std::mem::replace(&mut seen[index], true),
                "page {page} is on the free list twice"
            );
        }
        for node in self.cache.tree().walk() {
            for chunk in self.cache.tree().node(node).value.chunks(self.page_size) {
                let index = chunk[0] as usize / self.page_size;
                assert!(
                    !std::mem::replace(&mut seen[index], true),
                    "page {} is both cached and free, or cached twice",
                    chunk[0]
                );
            }
        }
        if let Some(window) = self.window.as_ref() {
            window.check_integrity();
        }
        match &self.cache {
            PrefixCache::Swa(cache) => {
                // The cross-pool invariant, and the one that catches a
                // skipped tombstone: idle, every window slot the pool
                // has handed out belongs to a live tree node, and every
                // live tree node's tokens hold one. A node published as
                // window-live over slots the request had already
                // released shows up here as a shortfall -- which is the
                // corruption the tombstone rule exists to prevent,
                // caught in accounting rather than in a gather.
                if let Some(window) = self.window.as_ref() {
                    let tree_swa = cache.swa_evictable_size() + cache.swa_protected_size();
                    assert_eq!(
                        window.live(),
                        tree_swa,
                        "window slots and live tree window state disagree: \
                         {} live slots vs {tree_swa} live tree tokens",
                        window.live()
                    );
                }
            }
            PrefixCache::Hybrid(cache) => {
                // A bound rather than an equality: a running request
                // legitimately holds slots that are in neither place.
                let tree_slots = cache.mamba_evictable_size() + cache.mamba_protected_size();
                assert!(
                    self.state_slots.len() + tree_slots <= self.num_state_slots,
                    "recurrent slots leaked or double-freed: free({}) + tree({tree_slots}) > \
                     capacity({})",
                    self.state_slots.len(),
                    self.num_state_slots
                );
            }
            PrefixCache::Plain(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4;

    fn manager() -> CacheManager {
        // 16 pages of 4 = 64 positions, 4 request slots.
        CacheManager::new(PAGE, 16, 64, 4)
    }

    fn ids(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    /// Drive one request end to end: match, allocate, commit.
    fn serve(
        manager: &mut CacheManager,
        slot: u32,
        prompt: &[u32],
        finished: bool,
    ) -> CommitOutcome {
        let mut seq = SequenceState::new(slot);
        manager.match_and_lock(&mut seq, prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        manager.commit(&mut seq, prompt, finished)
    }

    #[test]
    fn a_fresh_manager_owns_every_page() {
        let manager = manager();
        assert_eq!(manager.available_tokens(), 64);
        assert_eq!(manager.page_usage(), (0, 16));
        manager.check_integrity();
    }

    #[test]
    fn a_served_request_publishes_its_prefix_and_frees_its_tail() {
        let mut manager = manager();
        let outcome = serve(&mut manager, 0, &ids(0..8), true);
        assert_eq!(outcome.duplicate_len, 0, "nothing was cached before");
        assert_eq!(outcome.published_len, 8);
        manager.check_integrity();

        // The cache now holds it, and a second request reuses it.
        let mut seq = SequenceState::new(1);
        let matched = manager.match_and_lock(&mut seq, &ids(0..8));
        assert_eq!(matched.cached_len, 4, "the last token is never matched");
        manager.cache().check_integrity();
    }

    /// The trap the module exists for: two requests compute the same
    /// continuation, and the loser's pages must go back.
    #[test]
    fn a_concurrent_duplicate_hands_its_pages_back() {
        let mut manager = manager();
        let prompt = ids(0..8);

        let mut first = SequenceState::new(0);
        manager.match_and_lock(&mut first, &prompt);
        manager.allocate(&mut first, prompt.len()).unwrap();

        let mut second = SequenceState::new(1);
        manager.match_and_lock(&mut second, &prompt);
        manager.allocate(&mut second, prompt.len()).unwrap();

        // Both computed the whole prompt; the first publishes it.
        let won = manager.commit(&mut first, &prompt, true);
        assert_eq!(won.duplicate_len, 0);
        assert_eq!(won.published_len, 8);

        // The second's pages for the same span are now redundant.
        let lost = manager.commit(&mut second, &prompt, true);
        assert_eq!(lost.duplicate_len, 8, "the whole span was redundant");
        assert!(lost.pages_freed >= 2);
        manager.check_integrity();
    }

    /// And a request that keeps reading must be repointed at the
    /// tree's pages before its duplicates are handed to anyone else.
    #[test]
    fn an_unfinished_duplicate_is_repointed_before_its_pages_go_back() {
        let mut manager = manager();
        let prompt = ids(0..8);
        serve(&mut manager, 0, &prompt, true);
        let canonical: Vec<u32> = {
            let mut probe = SequenceState::new(3);
            manager.match_and_lock(&mut probe, &ids(0..9));
            manager.cache().matched_indices(probe.locked_node)
        };

        // A second request that computed the same prefix itself.
        let mut seq = SequenceState::new(1);
        manager.allocate(&mut seq, prompt.len()).unwrap();
        let private: Vec<u32> = manager.row(1)[..8].to_vec();
        let outcome = manager.commit(&mut seq, &prompt, false);
        assert_eq!(outcome.duplicate_len, 8);

        let row: Vec<u32> = manager.row(1)[..canonical.len()].to_vec();
        assert_eq!(row, canonical, "the row must read the tree's pages");
        assert_ne!(row[..4], private[..4], "not its own, which just went back");
    }

    /// The whole-lifecycle property: every page is accounted for after
    /// any mix of requests, sharing, and eviction.
    #[test]
    fn pages_are_conserved_across_a_mixed_workload() {
        let mut manager = manager();
        for round in 0..12u32 {
            // Half the prompts share a prefix; half diverge.
            let prompt: Vec<u32> = (0..8)
                .map(|i| if i < 4 { i } else { i + round % 3 })
                .collect();
            serve(&mut manager, round % 4, &prompt, true);
            manager.check_integrity();
        }
        assert_eq!(manager.page_usage().0, 0, "nothing is in flight");
    }

    /// A request that cannot be served must leave the pool exactly as
    /// it was, so the caller can evict and retry.
    #[test]
    fn a_refused_allocation_takes_nothing() {
        let mut manager = CacheManager::new(PAGE, 2, 64, 2);
        let mut seq = SequenceState::new(0);
        let err = manager.allocate(&mut seq, 64).unwrap_err();
        assert!(matches!(err, OutOfMemory::Pages { .. }), "{err}");
        assert_eq!(manager.available_tokens(), 8, "nothing was taken");
        manager.check_integrity();
    }

    /// Evictable cache counts as available: a cached prefix nobody is
    /// reading is memory, not occupancy.
    #[test]
    fn allocation_evicts_rather_than_refusing_while_the_cache_holds_memory() {
        let mut manager = CacheManager::new(PAGE, 4, 64, 2);
        serve(&mut manager, 0, &ids(0..16), true);
        assert_eq!(manager.available_tokens(), 16, "all of it, as cache");
        assert_eq!(manager.page_usage(), (0, 4));

        let mut seq = SequenceState::new(1);
        manager
            .allocate(&mut seq, 16)
            .expect("the cache is evictable");
        assert_eq!(manager.cache().evictable_size(), 0, "it was evicted");
        // Integrity is idle-only, so finish the request first.
        manager.commit(&mut seq, &ids(100..116), true);
        manager.check_integrity();
    }

    /// A page freed by one request in a batch must not reach another
    /// that is still reading it.
    #[test]
    fn frees_inside_a_batch_are_held_until_it_ends() {
        let mut manager = manager();
        let prompt = ids(0..8);
        serve(&mut manager, 0, &prompt, true);
        let free_before = manager.free_pages.len();

        manager.begin_batch();
        let mut seq = SequenceState::new(1);
        manager.allocate(&mut seq, 8).unwrap();
        manager.commit(&mut seq, &prompt, true);
        assert_eq!(
            manager.free_pages.len(),
            free_before - 2,
            "the duplicate's pages are held, not yet reusable"
        );
        manager.end_batch();
        assert_eq!(manager.free_pages.len(), free_before);
        manager.check_integrity();
    }

    /// A window model releases window slots as it slides and keeps
    /// every full-KV page -- that is the point of two pools.
    #[test]
    fn sliding_releases_window_slots_and_keeps_the_full_kv() {
        // The window pool covers the span being allocated in one go.
        // A real prefill chunks against exactly this bound -- see
        // `scheduler::PrefillPass`'s window-bounded chunk -- and
        // slides between chunks; here the whole span is taken at once.
        let mut manager =
            CacheManager::new(PAGE, 16, 64, 2).with_window_pool(WindowSlotPool::new(64, PAGE, 64));
        let policy = WindowPolicy::new(8, PAGE).with_eviction_interval(1);

        let mut seq = SequenceState::new(0);
        manager.allocate(&mut seq, 48).unwrap();
        seq.decode_step = 1;
        let window_before = manager.window_pool().unwrap().available();
        let (pages_used, _) = manager.page_usage();

        let decision = manager
            .slide_window(&mut seq, None, &policy, 1)
            .expect("a slide is due");
        assert!(!decision.frees_nothing(), "{decision:?}");
        assert!(
            manager.window_pool().unwrap().available() > window_before,
            "window slots came back"
        );
        assert_eq!(
            manager.page_usage().0,
            pages_used,
            "not one full-KV page was given up"
        );
        manager.window_pool().unwrap().check_integrity();
    }

    /// An anchor makes the slide hold state the plain slide would have
    /// released -- end to end, through the page table.
    #[test]
    fn an_anchor_holds_window_state_through_the_manager() {
        let policy = WindowPolicy::new(8, PAGE).with_eviction_interval(1);
        let slide = |anchor: Option<usize>| {
            let mut manager = CacheManager::new(PAGE, 16, 64, 2)
                .with_window_pool(WindowSlotPool::new(64, PAGE, 64));
            let mut seq = SequenceState::new(0);
            manager.allocate(&mut seq, 48).unwrap();
            seq.decode_step = 1;
            manager
                .slide_window(&mut seq, anchor, &policy, 1)
                .expect("a slide is due")
                .free_to
        };
        // Inside the drift bound for this geometry: further back and
        // the anchor is dropped instead of held (see `anchor`'s own
        // drop test).
        assert!(
            slide(Some(40)) < slide(None),
            "the anchor must hold state the plain slide released"
        );
    }

    const WINDOW: usize = 8;

    /// A window manager: 64 pages of 4 = 256 positions over 4 request
    /// slots, and a window pool with `usable` allocatable slots.
    fn swa_manager(usable: usize) -> CacheManager {
        CacheManager::new_swa(
            PAGE,
            64,
            256,
            4,
            WINDOW,
            WindowSlotPool::new(256, PAGE, usable + 1),
        )
    }

    fn slide_policy() -> WindowPolicy {
        WindowPolicy::new(WINDOW, PAGE).with_eviction_interval(1)
    }

    /// Every page under a node the tree still calls window-live must map
    /// to a real window slot.
    fn assert_no_live_node_reads_the_sentinel(manager: &CacheManager) {
        let cache = manager.swa_cache();
        let window = manager.window_pool().expect("a window manager has a pool");
        for id in cache.tree().walk() {
            let node = cache.tree().node(id);
            if node.swa_tombstone {
                continue;
            }
            for page in node.value.iter().copied() {
                assert!(
                    window.slot_of(page).is_some(),
                    "page {page} is published as window-live but holds no window slot"
                );
            }
        }
    }

    /// The rule the window path exists for: positions whose window slots
    /// the request already handed back are published as TOMBSTONES, not
    /// adopted as live.
    ///
    /// Committed the naive way -- with a zero frontier, the way the
    /// plain path inserts -- the tree records a live node over slots
    /// that are back in the pool, and the next request to match that
    /// prefix gathers the reserved sentinel in every window layer. This
    /// test fails in that arrangement: it walks the tree and asserts no
    /// live node reads the sentinel.
    #[test]
    fn a_window_commit_tombstones_the_slots_the_request_already_released() {
        let mut manager = swa_manager(96);
        let prompt = ids(0..40);
        let mut seq = SequenceState::new(0);
        seq.prompt_len = prompt.len();
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        seq.decode_step = 1;
        manager
            .slide_window(&mut seq, None, &slide_policy(), 1)
            .expect("a slide is due");
        let released = seq.released_window;
        assert!(released > 0, "the request gave its own slots back");

        manager.commit(&mut seq, &prompt, false);

        assert_no_live_node_reads_the_sentinel(&manager);
        let cache = manager.swa_cache();
        let tombstoned: usize = cache
            .tree()
            .walk()
            .into_iter()
            .filter(|id| cache.tree().node(*id).swa_tombstone)
            .map(|id| cache.tree().node(id).length())
            .sum();
        assert_eq!(
            tombstoned, released,
            "exactly the released span is tombstoned"
        );
    }

    /// A chunk's whole extend becomes one node, and a node-granular lock
    /// would pin all of it for the rest of decode -- though the request
    /// reads only its trailing window from here on.
    ///
    /// Committed the naive way, with a single match, the window lock
    /// covers the whole 60-token chunk. The extra match a window back
    /// forces a node boundary there, so it covers 24.
    #[test]
    fn an_unfinished_window_commit_pins_only_the_trailing_window() {
        let mut manager = swa_manager(96);
        let prompt = ids(0..60);
        let mut seq = SequenceState::new(0);
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        manager.commit(&mut seq, &prompt, false);

        // keep_from = align_down(60 - 8 - 16, 4) = 36.
        assert_eq!(
            manager.swa_cache().swa_protected_size(),
            24,
            "the naive single match pins all 60"
        );
        assert_eq!(
            manager.swa_cache().full_protected_size(),
            60,
            "the FULL lock still covers everything the request reads"
        );
    }

    /// The trim's whole point: the head of a finished prompt gives up
    /// its window slots and keeps every page, because the pages are what
    /// the next turn reuses.
    #[test]
    fn a_finished_window_request_trims_its_prompt_head_and_keeps_the_full_kv() {
        let mut manager = swa_manager(96);
        let prompt = ids(0..40);
        let mut seq = SequenceState::new(0);
        seq.prompt_len = prompt.len();
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        let window_before = manager.window_pool().unwrap().available();

        manager.commit(&mut seq, &prompt, true);

        // keep_from = align_down(40 - 8 - 16, 4) = 16.
        assert_eq!(
            manager.window_pool().unwrap().available(),
            window_before + 16,
            "the head's window slots came back"
        );
        assert_eq!(
            manager.swa_cache().full_evictable_size(),
            40,
            "and not one page did"
        );
        manager.check_integrity();
    }

    /// Decode never matches the prompt again, so a finished request's
    /// prompt path is the stalest thing in the tree -- and it is exactly
    /// the path the next turn rejoins, because a client that drops its
    /// reasoning block diverges at the prompt end.
    ///
    /// Without the closing re-match the retained prompt window is the
    /// oldest live-window node and the first `evict_swa` victim, so the
    /// next turn's cut there misses: this test reads 0 instead of 40.
    #[test]
    fn a_finished_window_request_restamps_the_prompt_it_just_trimmed() {
        let mut manager = swa_manager(96);
        let prompt = ids(0..40);
        let mut seq = SequenceState::new(0);
        seq.prompt_len = prompt.len();
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        // A prefill commit, then a generation that runs on past it: the
        // decode's own nodes are inserted long after the prompt's.
        manager.commit(&mut seq, &prompt, false);
        let generated = ids(0..60);
        manager.allocate(&mut seq, generated.len()).expect("room");
        manager.commit(&mut seq, &generated, true);

        // Window pressure: a second request needs more slots than are
        // free, so the allocation reclaims from the tree.
        let mut other = SequenceState::new(1);
        manager
            .allocate(&mut other, 56)
            .expect("room after reclaim");

        let mut next_turn = ids(0..40);
        next_turn.extend(ids(900..904));
        let mut next = SequenceState::new(2);
        let matched = manager.match_and_lock(&mut next, &next_turn);
        assert_eq!(
            matched.cached_len, 40,
            "the next turn must still rejoin at the prompt end"
        );
    }

    /// A window model's tree holds reclaimable window state, so an
    /// allocation tombstones rather than refusing -- the same reason
    /// page allocation evicts rather than refusing.
    #[test]
    fn a_window_allocation_reclaims_tree_window_state_rather_than_refusing() {
        let mut manager = swa_manager(48);
        let prompt = ids(0..40);
        let mut seq = SequenceState::new(0);
        seq.prompt_len = prompt.len();
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        manager.commit(&mut seq, &prompt, true);
        assert!(
            manager.window_pool().unwrap().available() < 40,
            "the cached prefix holds most of the pool"
        );

        let mut second = SequenceState::new(1);
        manager
            .allocate(&mut second, 40)
            .expect("the tree's window state is reclaimable");
        manager.commit(&mut second, &ids(100..140), true);
        manager.check_integrity();
    }

    /// Both currencies, over a mixed workload: every page and every
    /// window slot is accounted for after any number of turns that
    /// share, slide, trim and evict.
    #[test]
    fn a_window_workload_conserves_pages_and_window_slots() {
        let mut manager = swa_manager(96);
        for round in 0..8u32 {
            // Half the turns share a prefix; half diverge.
            let prompt: Vec<u32> = (0..40)
                .map(|i| if i < 20 { i } else { i + round % 3 })
                .collect();
            let mut seq = SequenceState::new(round % 4);
            seq.prompt_len = prompt.len();
            manager.match_and_lock(&mut seq, &prompt);
            manager.allocate(&mut seq, prompt.len()).expect("room");
            seq.decode_step = 1;
            manager.slide_window(&mut seq, None, &slide_policy(), 1);
            manager.commit(&mut seq, &prompt, true);
            manager.check_integrity();
            assert_no_live_node_reads_the_sentinel(&manager);
        }
        assert_eq!(manager.page_usage().0, 0, "nothing is in flight");
    }

    /// Drive one hybrid request end to end, donating its live state.
    fn serve_hybrid(manager: &mut CacheManager, slot: u32, prompt: &[u32]) -> CommitOutcome {
        let live = manager.alloc_state_slot().expect("a recurrent slot");
        let mut seq = SequenceState::new(slot);
        seq.recurrent = Some(RecurrentSlots {
            live,
            ping_pong: None,
            pending_freeze: None,
        });
        manager.match_and_lock(&mut seq, prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        manager.commit(&mut seq, prompt, true)
    }

    /// Donate a frozen snapshot from an unfinished chunk, the way a
    /// running request does, and hand back the sequence it left locked.
    fn donate_frozen(
        manager: &mut CacheManager,
        slot: u32,
        prompt: &[u32],
        freeze_len: usize,
    ) -> SequenceState {
        let live = manager.alloc_state_slot().expect("a recurrent slot");
        let first = manager.alloc_state_slot().expect("a recurrent slot");
        let second = manager.alloc_state_slot().expect("a recurrent slot");
        let mut seq = SequenceState::new(slot);
        seq.recurrent = Some(RecurrentSlots {
            live,
            // Flipped: the forward has already written `first`.
            ping_pong: Some(PingPong::new(first, second).flipped()),
            pending_freeze: Some(freeze_len),
        });
        manager.match_and_lock(&mut seq, prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");
        manager.commit(&mut seq, prompt, false);
        seq
    }

    /// The hybrid ordering rule: the committed node is locked BEFORE the
    /// replacement slot is allocated.
    ///
    /// The allocation can evict, and the node this commit just donated
    /// to is an unlocked childless leaf -- the ideal `evict_mamba`
    /// victim. Allocate first and the tree takes it back, returning the
    /// request's own KV pages to the pool while it is still decoding
    /// into them: this test's pages land on the free list and the
    /// request ends up holding no node at all.
    #[test]
    fn a_hybrid_commit_locks_its_donated_node_before_allocating_the_replacement() {
        // Seven slots: enough for a request that stays running and for
        // the one under test, and none spare when it commits.
        let mut manager = CacheManager::new_hybrid(PAGE, 16, 64, 4, 7);
        // A request that is still decoding, holding the only other
        // snapshot in the tree -- so the LRU candidate set is empty
        // unless the just-donated node is in it.
        let running = donate_frozen(&mut manager, 0, &ids(200..212), 8);
        assert_eq!(running.locked_len, 8);
        assert_eq!(
            manager.state_slots_available(),
            3,
            "exactly what the request under test is admitted with, and nothing over"
        );

        let prompt = ids(0..12);
        let seq = donate_frozen(&mut manager, 1, &prompt, 8);
        let row: Vec<u32> = manager.row(1)[..8].to_vec();

        assert_eq!(
            seq.locked_len, 8,
            "the request holds the node it donated to"
        );
        assert_eq!(manager.state_slots_available(), 0, "the pool is empty");
        assert!(
            seq.recurrent.unwrap().ping_pong.is_none(),
            "nothing was free or evictable, so it decodes on without a spare slot \
             rather than reclaiming a node someone is serving from"
        );
        let owned: Vec<u32> = manager
            .hybrid_cache()
            .tree()
            .walk()
            .into_iter()
            .flat_map(|id| manager.hybrid_cache().tree().node(id).value.clone())
            .collect();
        for page in row {
            assert!(
                owned.contains(&page),
                "page {page} left the tree under a decoding request"
            );
            assert!(
                !manager.free_pages.contains(&page),
                "page {page} went back on the free list under a decoding request"
            );
        }
    }

    /// A finish donates twice: the pending freeze at its own boundary,
    /// and the live state at the end. The pending one is a strictly
    /// shorter prefix and must go FIRST, so that the live donate's
    /// duplicate span is measured against a floor that already accounts
    /// for it.
    ///
    /// The other order publishes the request's pages for the whole span
    /// and then has the shorter insert report that same span as a
    /// duplicate -- so the tree's own pages go back on the free list.
    /// Both assertions below catch that: the commit reports 8 duplicate
    /// positions where there were none, and `check_integrity` then finds
    /// two more free pages than the pool owns.
    #[test]
    fn a_finished_hybrid_request_donates_its_pending_freeze_before_its_live_state() {
        let mut manager = CacheManager::new_hybrid(PAGE, 16, 64, 4, 6);
        let live = manager.alloc_state_slot().unwrap();
        let frozen = manager.alloc_state_slot().unwrap();
        let idle = manager.alloc_state_slot().unwrap();
        let mut seq = SequenceState::new(0);
        seq.recurrent = Some(RecurrentSlots {
            live,
            ping_pong: Some(PingPong::new(frozen, idle).flipped()),
            pending_freeze: Some(8),
        });
        let prompt = ids(0..16);
        manager.match_and_lock(&mut seq, &prompt);
        manager.allocate(&mut seq, prompt.len()).expect("room");

        let outcome = manager.commit(&mut seq, &prompt, true);
        assert!(outcome.snapshot_donated);
        assert_eq!(outcome.duplicate_len, 0, "nothing was computed twice");
        manager.check_integrity();

        let cache = manager.hybrid_cache();
        let mut snapshots: Vec<u32> = cache
            .tree()
            .walk()
            .into_iter()
            .filter_map(|id| cache.tree().node(id).mamba_value)
            .collect();
        snapshots.sort_unstable();
        let mut expected = vec![live, frozen];
        expected.sort_unstable();
        assert_eq!(snapshots, expected, "both boundaries carry a snapshot");
        assert_eq!(
            manager.state_slots_available(),
            4,
            "only the untaken idle slot came back"
        );
        assert!(seq.recurrent.is_none(), "the request holds no slot now");
    }

    /// The donation contract, through the manager: a refused snapshot is
    /// still the request's, and dropping it leaks one recurrent slot per
    /// request until admission hangs.
    #[test]
    fn a_refused_hybrid_donation_hands_the_slot_back() {
        let mut manager = CacheManager::new_hybrid(PAGE, 16, 64, 4, 6);
        serve_hybrid(&mut manager, 0, &ids(0..8));
        let after_first = manager.state_slots_available();
        assert_eq!(after_first, 5, "the tree took the first request's slot");

        let outcome = serve_hybrid(&mut manager, 1, &ids(0..8));
        assert!(!outcome.snapshot_donated, "the boundary already had one");
        assert_eq!(
            manager.state_slots_available(),
            after_first,
            "the refused slot came back"
        );
        manager.check_integrity();
    }

    /// A recurrent state is valid at exactly one position, so a finish
    /// whose length is not a node boundary donates nothing -- and then
    /// every page it holds, including the partial one it was charged
    /// whole for, goes back.
    #[test]
    fn a_hybrid_finish_at_a_ragged_length_publishes_nothing() {
        let mut manager = CacheManager::new_hybrid(PAGE, 16, 64, 4, 4);
        let outcome = serve_hybrid(&mut manager, 0, &ids(0..14));
        assert_eq!(outcome.published_len, 0, "no node ends at 14");
        assert_eq!(manager.hybrid_cache().full_evictable_size(), 0);
        assert_eq!(
            manager.state_slots_available(),
            4,
            "the undonated live slot came back"
        );
        assert_eq!(manager.page_usage(), (0, 16), "the padding page too");
        manager.check_integrity();
    }

    #[test]
    fn a_window_model_refuses_rather_than_half_allocating() {
        let mut manager = CacheManager::new(PAGE, 16, 64, 2)
            // Only two usable window slots.
            .with_window_pool(WindowSlotPool::new(64, PAGE, 3));
        let mut seq = SequenceState::new(0);
        let err = manager.allocate(&mut seq, 16).unwrap_err();
        assert!(matches!(err, OutOfMemory::Window(_)), "{err}");
        assert_eq!(manager.available_tokens(), 64, "the pages came back");
        manager.check_integrity();
    }
}
