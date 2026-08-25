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
//! Ported 1:1 from FreeToken's `scheduler/cache.py::CacheManager`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::VecDeque;

use crate::anchor::{decode_slide, SlideDecision, SlidingRequest, WindowPolicy};
use crate::radix::{align_ceil, NodeId, RadixCache, ROOT};
use crate::window_pool::{WindowPoolExhausted, WindowSlotPool};

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
    pub released_window: usize,
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
    cache: RadixCache,
    window: Option<WindowSlotPool>,
    /// Pages freed since `begin_batch`, held back until `end_batch`.
    deferred: Option<Vec<u32>>,
}

impl CacheManager {
    pub fn new(page_size: usize, num_pages: usize, max_seq_len: usize, num_slots: usize) -> Self {
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
            cache: RadixCache::new(page_size),
            window: None,
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

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    pub fn window_pool(&self) -> Option<&WindowSlotPool> {
        self.window.as_ref()
    }

    /// Positions that could be served without evicting anything the
    /// cache is protecting: free pages plus what is evictable.
    ///
    /// This is the number admission reads, and it counts evictable
    /// cache as available on purpose — a cached prefix nobody is
    /// reading is memory, not occupancy, and refusing a request because
    /// of it would leave the pool idle at capacity.
    pub fn available_tokens(&self) -> usize {
        self.free_pages.len() * self.page_size + self.cache.evictable_size()
    }

    /// `(pages in use, pages total)`, for a status line.
    pub fn page_usage(&self) -> (usize, usize) {
        let cached_pages = self.cache.evictable_size() / self.page_size;
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
        let matched = self.cache.match_prefix(lookup);
        self.cache.lock(matched.node);
        if seq.locked_len > 0 || seq.locked_node != ROOT {
            self.cache.unlock(seq.locked_node);
        }
        seq.locked_node = matched.node;
        seq.locked_len = matched.cached_len;
        seq.cached_len = matched.cached_len;

        // The request reads the tree's pages for the part it did not
        // compute.
        let canonical = self.cache.matched_indices(matched.node);
        self.row_mut(seq.slot)[..canonical.len()].copy_from_slice(&canonical);
        PrefixMatch {
            cached_len: matched.cached_len,
            node: matched.node,
        }
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
        let publish_len = seq.cached_len.min(input_ids.len());
        let row_snapshot: Vec<u32> = self.row(seq.slot)[..publish_len].to_vec();
        let result = self
            .cache
            .insert_prefix(&input_ids[..publish_len], &row_snapshot);

        let old_locked = seq.locked_len;
        self.cache.unlock(seq.locked_node);

        let mut outcome = CommitOutcome {
            duplicate_len: result.cached_len.saturating_sub(old_locked),
            published_len: result.inserted_len,
            pages_freed: 0,
        };

        // The duplicate span: the tree already had it, so these pages
        // are redundant -- but the request is still reading those
        // positions, so repoint before freeing.
        if result.cached_len > old_locked {
            let duplicates: Vec<u32> = row_snapshot[old_locked..result.cached_len].to_vec();
            if !finished {
                let canonical = self.cache.matched_indices(result.node);
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
            self.cache.lock(result.node);
            seq.locked_node = result.node;
            seq.locked_len = result.inserted_len;
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
        if tokens > self.cache.evictable_size() {
            return Err(OutOfMemory::Pages {
                needed,
                available: self.free_pages.len() + self.cache.evictable_size() / self.page_size,
            });
        }
        let evicted = self.cache.evict(tokens);
        // A node holds whole pages and a page's locations are
        // contiguous, so every `page_size`-th location is a page base.
        for chunk in evicted.chunks(self.page_size) {
            self.free_pages.push_back(chunk[0]);
            if let Some(window) = self.window.as_mut() {
                window.free(chunk);
            }
        }
        debug_assert!(self.free_pages.len() >= needed);
        Ok(())
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
        let cached_pages = self.cache.total_size() / self.page_size;
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
