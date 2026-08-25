//! The window-slot pool: where a sliding-window layer's KV actually
//! lives.
//!
//! A window model's layers do not all need the same amount of memory.
//! Full-attention layers read every token, so their KV is indexed by
//! the token's position in the sequence. Window layers read only the
//! last `sliding_window` tokens, so their KV needs a pool sized for the
//! *window*, not for the history -- often an order of magnitude
//! smaller.
//!
//! That means two different address spaces over the same sequence, and
//! something has to map between them. This is that map: **full position
//! -> window slot**, plus the free list of slots.
//!
//! # Slot 0 is not a slot
//!
//! Slot 0 is permanently reserved to mean "this position has no live
//! window state", so an unmapped position reads as 0 rather than as
//! slot 0's bytes, and freeing a position twice is a no-op instead of
//! handing the same slot to the free list twice. Allocatable slots are
//! `1 .. num_slots`. That single convention removes an entire class of
//! double-free bug from every caller, at the cost of one slot.
//!
//! # The conservation invariant
//!
//! `available + live == num_slots - 1`, checked by
//! [`WindowSlotPool::check_integrity`]. It is an **equality**, not a
//! bound, which is what makes it useful: a `<=` would catch a double
//! free and quietly tolerate a leak, and a leaked window slot is
//! exactly the failure that turns into "the pool is full" an hour into
//! a long-running server, with nothing to point at.
//!
//! Ported 1:1 from the paged-SWA allocator in FreeToken's
//! `kvcache/hybrid_swa_pool.py` (Apache-2.0), which follows SGLang's
//! `SWATokenToKVPoolAllocator`; see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::VecDeque;

/// The pool had fewer free slots than the request needed. Nothing was
/// taken: the check happens before the first slot moves, so a refused
/// allocation leaves the pool exactly as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPoolExhausted {
    pub needed: usize,
    pub available: usize,
}

impl std::fmt::Display for WindowPoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "window pool exhausted: need {} slots, {} available",
            self.needed, self.available
        )
    }
}

impl std::error::Error for WindowPoolExhausted {}

/// The reserved "no live window state" slot.
pub const NO_SLOT: u32 = 0;

/// Maps full sequence positions onto window-pool slots.
#[derive(Debug)]
pub struct WindowSlotPool {
    /// Full position -> window slot, `NO_SLOT` for unmapped.
    mapping: Vec<u32>,
    /// Free slots, taken from the head.
    ///
    /// FIFO rather than LIFO on purpose: a slot that was just freed is
    /// the one most likely to still be resident in cache *and* the one
    /// most likely to be freed again immediately by the same sliding
    /// request, so handing it straight back concentrates churn on a few
    /// slots. Round-robin spreads it.
    free: VecDeque<u32>,
    num_slots: usize,
    page_size: usize,
}

impl WindowSlotPool {
    /// A pool of `num_slots` slots covering `full_tokens` positions.
    ///
    /// The mapping is over-allocated by one page: an allocation
    /// addresses whole pages, so the last one can name a position one
    /// page past the final token, and growing the map is cheaper than
    /// bounds-checking every translate.
    pub fn new(full_tokens: usize, page_size: usize, num_slots: usize) -> Self {
        assert!(page_size > 0, "a page holds at least one token");
        assert!(
            num_slots >= 2,
            "a window pool needs the reserved sentinel plus at least one usable slot"
        );
        WindowSlotPool {
            mapping: vec![NO_SLOT; full_tokens + page_size],
            free: (1..num_slots as u32).collect(),
            num_slots,
            page_size,
        }
    }

    /// Slots that can still be handed out.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Slots that could ever be handed out -- one fewer than the pool
    /// holds, because slot 0 is the sentinel.
    pub fn capacity(&self) -> usize {
        self.num_slots - 1
    }

    /// Positions currently holding window state.
    pub fn live(&self) -> usize {
        self.mapping.iter().filter(|slot| **slot != NO_SLOT).count()
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Give each position in `full_indices` a window slot.
    ///
    /// All or nothing: the capacity check happens before the first slot
    /// moves, so a caller that cannot be served can fall back to
    /// evicting and retrying without first having to undo a partial
    /// allocation.
    pub fn alloc(&mut self, full_indices: &[u32]) -> Result<(), WindowPoolExhausted> {
        if full_indices.is_empty() {
            return Ok(());
        }
        if full_indices.len() > self.free.len() {
            return Err(WindowPoolExhausted {
                needed: full_indices.len(),
                available: self.free.len(),
            });
        }
        for index in full_indices {
            debug_assert_eq!(
                self.mapping[*index as usize], NO_SLOT,
                "position {index} already holds a window slot; allocating over it would leak it"
            );
            let slot = self.free.pop_front().expect("capacity was checked");
            self.mapping[*index as usize] = slot;
        }
        Ok(())
    }

    /// Return the window slots held by `full_indices`.
    ///
    /// Idempotent: a position that holds no slot contributes nothing,
    /// so a caller may pass the same span twice, or a span it is not
    /// sure about, without double-freeing. That is what the sentinel
    /// buys, and several callers depend on it -- an eviction pass and a
    /// request's own window slide can both name the same positions.
    pub fn free(&mut self, full_indices: &[u32]) {
        for index in full_indices {
            let slot = std::mem::replace(&mut self.mapping[*index as usize], NO_SLOT);
            if slot != NO_SLOT {
                self.free.push_back(slot);
            }
        }
    }

    /// The window slot holding `full_index`, if any.
    pub fn slot_of(&self, full_index: u32) -> Option<u32> {
        match self.mapping[full_index as usize] {
            NO_SLOT => None,
            slot => Some(slot),
        }
    }

    /// Translate a full-pool location into a window-pool location.
    ///
    /// A negative input passes straight through as `-1`. Callers carry
    /// "no previous location" as `-1` and would otherwise have to
    /// special-case it at every call site; upstream gets the same
    /// effect from a trailing sentinel row and negative tensor
    /// indexing, which is the same rule written less obviously.
    pub fn translate(&self, full_loc: i64) -> i64 {
        if full_loc < 0 {
            return -1;
        }
        match self.mapping.get(full_loc as usize) {
            Some(&NO_SLOT) | None => 0,
            Some(&slot) => slot as i64,
        }
    }

    /// Resize, dropping everything.
    ///
    /// Every live mapping is discarded because a slot id is a position
    /// in an allocation that is about to stop existing; keeping the map
    /// across a resize would point requests at other requests' bytes.
    /// The map and the free list are rebuilt together for the same
    /// reason -- a half-updated pair is worse than either.
    pub fn rebuild(&mut self, full_tokens: usize, num_slots: usize) {
        assert!(
            num_slots >= 2,
            "a window pool needs the reserved sentinel plus at least one usable slot"
        );
        self.mapping = vec![NO_SLOT; full_tokens + self.page_size];
        self.free = (1..num_slots as u32).collect();
        self.num_slots = num_slots;
    }

    /// Every slot is either free or live, exactly once.
    pub fn check_integrity(&self) {
        let live = self.live();
        assert_eq!(
            self.available() + live,
            self.capacity(),
            "window slots leaked or double-freed: {} free + {} live != {} capacity",
            self.available(),
            live,
            self.capacity()
        );
        let mut seen = vec![false; self.num_slots];
        for slot in self.free.iter().copied() {
            assert!(
                !std::mem::replace(&mut seen[slot as usize], true),
                "slot {slot} is on the free list twice"
            );
        }
        for slot in self.mapping.iter().copied().filter(|s| *s != NO_SLOT) {
            assert!(
                !std::mem::replace(&mut seen[slot as usize], true),
                "slot {slot} is both live and free, or live for two positions"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4;

    fn pool() -> WindowSlotPool {
        // 16 positions, 8 slots -> 7 usable.
        WindowSlotPool::new(16, PAGE, 8)
    }

    #[test]
    fn a_fresh_pool_holds_every_slot_but_the_sentinel() {
        let pool = pool();
        assert_eq!(pool.capacity(), 7);
        assert_eq!(pool.available(), 7);
        assert_eq!(pool.live(), 0);
        assert_eq!(pool.slot_of(0), None);
        pool.check_integrity();
    }

    #[test]
    fn allocated_positions_get_distinct_usable_slots() {
        let mut pool = pool();
        pool.alloc(&[0, 3, 5]).expect("three of seven");
        let slots: Vec<u32> = [0, 3, 5]
            .iter()
            .map(|i| pool.slot_of(*i).unwrap())
            .collect();
        assert_eq!(slots.len(), 3);
        assert!(slots.iter().all(|s| (1..8).contains(s)), "{slots:?}");
        let mut unique = slots.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 3, "no slot serves two positions");
        assert_eq!(pool.available(), 4);
        pool.check_integrity();
    }

    /// An unmapped position reads as the sentinel, not as slot 0's
    /// bytes.
    #[test]
    fn an_unallocated_position_translates_to_the_sentinel() {
        let mut pool = pool();
        pool.alloc(&[2]).unwrap();
        assert_eq!(pool.translate(7), 0);
        assert_eq!(pool.translate(2), pool.slot_of(2).unwrap() as i64);
    }

    /// Callers carry "no previous location" as -1 and must not have to
    /// special-case it.
    #[test]
    fn a_negative_location_passes_through() {
        let pool = pool();
        assert_eq!(pool.translate(-1), -1);
        assert_eq!(pool.translate(-99), -1);
    }

    /// The whole point of the sentinel: an eviction pass and a
    /// request's own slide can name the same positions.
    #[test]
    fn freeing_the_same_span_twice_is_a_no_op() {
        let mut pool = pool();
        pool.alloc(&[1, 2, 3]).unwrap();
        pool.free(&[1, 2, 3]);
        assert_eq!(pool.available(), 7);
        pool.free(&[1, 2, 3]);
        assert_eq!(pool.available(), 7, "the second free took nothing");
        pool.check_integrity();

        // And a span that was never allocated at all.
        pool.free(&[9, 10]);
        assert_eq!(pool.available(), 7);
        pool.check_integrity();
    }

    /// A refused allocation must leave the pool usable: the caller's
    /// next move is to evict and retry, not to undo a half-allocation.
    #[test]
    fn an_oversized_allocation_takes_nothing() {
        let mut pool = pool();
        let err = pool.alloc(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap_err();
        assert_eq!(err.needed, 8);
        assert_eq!(err.available, 7);
        assert_eq!(pool.available(), 7, "nothing was taken");
        assert_eq!(pool.live(), 0);
        pool.check_integrity();

        // Still fully usable afterwards.
        pool.alloc(&[0, 1, 2]).expect("the pool is intact");
        pool.check_integrity();
    }

    #[test]
    fn an_empty_allocation_is_always_fine() {
        let mut pool = pool();
        pool.alloc(&[]).unwrap();
        assert_eq!(pool.available(), 7);
    }

    #[test]
    fn the_pool_can_be_drained_and_refilled_exactly() {
        let mut pool = pool();
        let all: Vec<u32> = (0..7).collect();
        pool.alloc(&all).expect("exactly capacity");
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.live(), 7);
        pool.check_integrity();

        assert!(pool.alloc(&[8]).is_err(), "nothing left");
        pool.free(&all);
        assert_eq!(pool.available(), 7);
        assert_eq!(pool.live(), 0);
        pool.check_integrity();
    }

    /// Slot ids are positions in an allocation that stops existing, so
    /// a resize cannot keep the map.
    #[test]
    fn a_rebuild_drops_every_mapping_and_resets_the_free_list() {
        let mut pool = pool();
        pool.alloc(&[0, 1, 2]).unwrap();
        pool.rebuild(32, 12);
        assert_eq!(pool.capacity(), 11);
        assert_eq!(pool.available(), 11);
        assert_eq!(pool.live(), 0);
        assert_eq!(pool.slot_of(0), None);
        // The new range is addressable.
        pool.alloc(&[31]).expect("the map covers the new size");
        pool.check_integrity();
    }

    /// The invariant is an equality precisely so it catches a leak, not
    /// only a double free.
    #[test]
    #[should_panic(expected = "leaked or double-freed")]
    fn the_invariant_catches_a_leaked_slot() {
        let mut pool = pool();
        pool.alloc(&[0, 1]).unwrap();
        // Simulate a caller that forgot to hand a slot back: drop the
        // mapping without returning the slot to the free list.
        pool.mapping[0] = NO_SLOT;
        pool.check_integrity();
    }

    /// A long-running sliding workload must conserve exactly, however
    /// many times the window moves.
    #[test]
    fn a_sliding_workload_conserves_every_slot() {
        let mut pool = WindowSlotPool::new(4096, PAGE, 64);
        let window = 32u32;
        for cursor in window..2048 {
            pool.alloc(&[cursor]).unwrap_or_else(|e| {
                panic!("cursor {cursor}: {e}");
            });
            // Slide: release everything a window behind.
            pool.free(&[cursor - window]);
            pool.check_integrity();
        }
        // Exactly the live window is held.
        assert_eq!(pool.live(), window as usize);
        assert_eq!(pool.available(), pool.capacity() - window as usize);
    }
}
