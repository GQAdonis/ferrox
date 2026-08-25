//! The recurrent-state slot pool, and the chunk-boundary checkpoint
//! that fills it.
//!
//! A hybrid model's linear/gated-delta layers keep a fixed-size
//! *recurrent state* per sequence rather than a per-token KV span. That
//! state lives in a pool of equal-sized slots, and everything else in
//! this crate refers to it by slot id alone: [`crate::anchor::PingPong`]
//! carries two ids, [`crate::radix::HybridRadixCache`] stores one per
//! node as an opaque `mamba_value`, hands it back in
//! `HybridEvicted::mamba_slots`, and refuses a donation with
//! `HybridInsert::snapshot_exists` when it already has one. This module
//! is the counterparty to all of that: the thing those ids are drawn
//! from and returned to.
//!
//! Without it, every refused donation and every evicted snapshot is a
//! slot nobody owns -- one leaked slot per request, which is exactly the
//! failure `radix/hybrid.rs`'s donation contract exists to prevent, and
//! which surfaces hours later as "the state pool is exhausted" with
//! nothing to point at.
//!
//! Like the state tensors it indexes, nothing here is a tensor. The pool
//! owns **ids**; the bytes those ids address are the engine's, and the
//! two copies this module describes ([`StateSlotPool::restore_copy`],
//! [`track_checkpoint`]) are returned as a from/into pair for the caller
//! to execute on its own stream.
//!
//! # Slot 0 is the padding sink
//!
//! Slot 0 is reserved and never allocatable: allocatable slots are
//! `1 .. num_slots`. It is where a padded (dummy) request in a
//! graph-captured batch writes, so a batch smaller than the captured
//! shape still has somewhere legal to fold state into. That means the
//! padding slot is routinely *held* by the dummy request -- so
//! [`StateSlotPool::free`] ignores it rather than trusting the caller,
//! because a free loop over a padded batch that returned slot 0 to the
//! free list would hand the padding sink to a real request and have two
//! sequences folding into the same state.
//!
//! # One free list, three consumers
//!
//! Live working slots, the two ping-pong track slots, and the snapshots
//! donated to the hybrid radix tree are all drawn from this **single**
//! free list. That is a deliberate refusal to partition: a request that
//! is not caching does not tie up snapshot slots, and a tree full of
//! reusable snapshots gives them back under admission pressure. The
//! sizing of the pool ([`crate::pool::linear_pool_slots`] /
//! [`crate::pool::linear_pool_min_slots`]) already assumes this -- it
//! prices four slots per running request plus a *shared* snapshot cache,
//! not three separate pools.
//!
//! # The conservation invariant
//!
//! `available + allocated == capacity`, checked by
//! [`StateSlotPool::check_integrity`]. It is an **equality**, not a
//! bound, for the same reason the window pool's is: a `<=` would catch a
//! double free and quietly tolerate a leak, and a leaked recurrent slot
//! is the one that cannot be diagnosed after the fact -- the tree that
//! was holding it is long gone.
//!
//! Ported from FreeToken's `kvcache/linear_state_pool.py`
//! (`LinearStatePool`) and the chunk-boundary track in
//! `attention/linear.py` (`_build_track_metadata`) (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use crate::anchor::PingPong;
use crate::radix::hybrid::CHUNK_SIZE;

/// The reserved padding sink. Never allocated, never freed, never
/// carries a sequence's state.
pub const PADDING_SLOT: u32 = 0;

/// The pool had fewer free slots than the request needed. Nothing was
/// taken: the check happens before the first slot moves, so a refused
/// allocation leaves the pool exactly as it was and the caller can
/// evict snapshots and retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatePoolExhausted {
    pub needed: usize,
    pub available: usize,
}

impl std::fmt::Display for StatePoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "recurrent-state pool exhausted: need {} slots, {} available",
            self.needed, self.available
        )
    }
}

impl std::error::Error for StatePoolExhausted {}

/// A whole-sequence state copy the caller must perform: conv +
/// recurrent, every linear layer, `from_slot` -> `into_slot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateCopy {
    pub from_slot: u32,
    pub into_slot: u32,
}

/// A free-list allocator over recurrent-state slots.
#[derive(Debug)]
pub struct StateSlotPool {
    /// Free slot ids.
    ///
    /// LIFO, unlike [`crate::window_pool::WindowSlotPool`]'s FIFO, and
    /// for the opposite reason: a window slot is handed straight back by
    /// the same sliding request, so round-robin is what spreads the
    /// churn. A recurrent slot is fully overwritten before it is ever
    /// read -- zeroed for a fresh sequence, or copy-restored from a
    /// snapshot -- so there is no stale-residency argument, and reusing
    /// the hottest slot is strictly better.
    free: Vec<u32>,
    /// Which slots are currently handed out. Bookkeeping the Python
    /// original does not keep; it is what turns the conservation
    /// invariant below from a count into a check that names the slot.
    allocated: Vec<bool>,
    num_slots: usize,
}

impl StateSlotPool {
    /// A pool of `num_slots` physical slots, of which `num_slots - 1`
    /// are allocatable.
    pub fn new(num_slots: usize) -> Self {
        assert!(
            num_slots >= 2,
            "a state pool needs the padding sink plus at least one usable slot"
        );
        StateSlotPool {
            free: (1..num_slots as u32).collect(),
            allocated: vec![false; num_slots],
            num_slots,
        }
    }

    /// Physical slots, the padding sink included -- the number the state
    /// tensors are shaped for.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Slots that could ever be handed out: one fewer than the pool
    /// holds, because slot 0 is the padding sink.
    pub fn capacity(&self) -> usize {
        self.num_slots - 1
    }

    /// Slots that can still be handed out.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Slots currently held by a request, a ping-pong pair, or the
    /// prefix tree.
    pub fn allocated(&self) -> usize {
        self.allocated.iter().filter(|held| **held).count()
    }

    /// Whether `slot` is currently held by someone.
    pub fn is_allocated(&self, slot: u32) -> bool {
        self.allocated
            .get(slot as usize)
            .copied()
            .unwrap_or_default()
    }

    /// Take `n` slot ids.
    ///
    /// All or nothing: the capacity check happens before the first slot
    /// moves, so a caller that cannot be served can evict snapshots
    /// (`HybridRadixCache::evict_mamba`), free what came back, and retry
    /// without first having to undo a partial allocation.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, StatePoolExhausted> {
        if n > self.free.len() {
            return Err(StatePoolExhausted {
                needed: n,
                available: self.free.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let slot = self.free.pop().expect("capacity was checked");
            self.allocated[slot as usize] = true;
            out.push(slot);
        }
        Ok(out)
    }

    /// Take a single slot -- a request's live working slot.
    pub fn alloc_one(&mut self) -> Result<u32, StatePoolExhausted> {
        Ok(self.alloc(1)?[0])
    }

    /// Take the two track slots a caching request ping-pongs between.
    ///
    /// Both or neither, from the same free list as everything else: a
    /// request that gets one track slot and not the other could snapshot
    /// but never hand the snapshot over, because the destination of the
    /// *next* checkpoint would be the slot still awaiting donation.
    pub fn alloc_ping_pong(&mut self) -> Result<PingPong, StatePoolExhausted> {
        let slots = self.alloc(2)?;
        Ok(PingPong::new(slots[0], slots[1]))
    }

    /// Return slots to the free list.
    ///
    /// The padding sink is ignored rather than rejected: a padded batch
    /// legitimately carries slot 0 as a request's slot, and a free loop
    /// over it must be allowed to say so without handing the sink out to
    /// a real sequence.
    pub fn free(&mut self, slots: &[u32]) {
        for slot in slots.iter().copied() {
            if slot == PADDING_SLOT {
                continue;
            }
            debug_assert!(
                self.is_allocated(slot),
                "slot {slot} was freed while it was not held; freeing it again would put it \
                 on the free list twice and hand one state to two sequences"
            );
            if std::mem::replace(&mut self.allocated[slot as usize], false) {
                self.free.push(slot);
            }
        }
    }

    /// The copy-on-write a prefix hit performs on restore.
    ///
    /// A matched snapshot is **copied** into the request's own live
    /// slot, never resumed from in place: the tree still owns that
    /// snapshot and will hand the same id to the next request that
    /// matches the same prefix, so folding this request's tokens into it
    /// would advance a state every later reader believes is frozen at
    /// the node's boundary.
    ///
    /// The copy itself is the caller's, and must be ordered after the
    /// previous batch's snapshot writes and before this forward reads
    /// the live slot.
    pub fn restore_copy(&self, snapshot: u32, into_live: u32) -> StateCopy {
        assert_ne!(
            snapshot, into_live,
            "restoring a snapshot onto itself writes through the slot the tree still owns"
        );
        assert_ne!(
            into_live, PADDING_SLOT,
            "the padding sink is not a live slot"
        );
        debug_assert!(
            self.is_allocated(snapshot) && self.is_allocated(into_live),
            "restore between slots {snapshot} and {into_live}, one of which is on the free list"
        );
        StateCopy {
            from_slot: snapshot,
            into_slot: into_live,
        }
    }

    /// Hand every non-padding slot back to the free list.
    ///
    /// **Idle-only and destructive by contract.** Every live and donated
    /// state is dropped, so the caller must guarantee that no running
    /// request holds a slot *and* that the prefix tree owning the
    /// donated snapshots is discarded in the same step -- a tree that
    /// survives this still names slots that are now free to hand out,
    /// and the next request to match one of its prefixes would resume
    /// from another sequence's state.
    pub fn reclaim_all(&mut self) {
        self.free = (1..self.num_slots as u32).collect();
        self.allocated = vec![false; self.num_slots];
    }

    /// Resize, dropping everything.
    ///
    /// Same contract as [`Self::reclaim_all`], and for a stronger
    /// reason: a slot id is a position in an allocation that is about to
    /// stop existing, so a snapshot id kept across a rebuild does not
    /// merely name the wrong state, it may name no state at all.
    pub fn rebuild(&mut self, num_slots: usize) {
        assert!(
            num_slots >= 2,
            "a state pool needs the padding sink plus at least one usable slot"
        );
        self.num_slots = num_slots;
        self.reclaim_all();
    }

    /// Every slot is either free or held, exactly once, and slot 0 is
    /// neither.
    pub fn check_integrity(&self) {
        assert_eq!(
            self.available() + self.allocated(),
            self.capacity(),
            "recurrent slots leaked or double-freed: {} free + {} held != {} capacity",
            self.available(),
            self.allocated(),
            self.capacity()
        );
        assert!(
            !self.allocated[PADDING_SLOT as usize],
            "the padding sink was handed out"
        );
        let mut seen = vec![false; self.num_slots];
        for slot in self.free.iter().copied() {
            assert_ne!(slot, PADDING_SLOT, "the padding sink is on the free list");
            assert!(
                !std::mem::replace(&mut seen[slot as usize], true),
                "slot {slot} is on the free list twice"
            );
        }
        for (slot, held) in self.allocated.iter().enumerate() {
            if *held {
                assert!(
                    !std::mem::replace(&mut seen[slot], true),
                    "slot {slot} is both held and free"
                );
            }
        }
    }
}

/// One request's extend, as the chunk-boundary track sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackRequest {
    /// The sequence length this extend starts at -- what the prefix
    /// cache already holds for the request.
    pub cached_len: usize,
    /// Tokens this forward folds for the request.
    pub extend_len: usize,
    /// The request's two donatable track slots. `None` when the request
    /// is not caching (a naive GDN model, or a non-hybrid one), and then
    /// no checkpoint is taken at all.
    pub ping_pong: Option<PingPong>,
}

/// Where a prefill forward freezes a request's recurrent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackCheckpoint {
    /// The track slot to write the frozen state into.
    pub into_slot: u32,
    /// The sequence length the frozen state is valid at -- the key the
    /// hybrid radix tree will attach it under.
    pub at_len: usize,
    /// Which chunk of *this extend* holds the folded state, counted from
    /// the start of the extend. Added to the request's base row (see
    /// [`chunk_row_bases`]) it addresses the kernel's per-chunk state.
    pub chunk: usize,
    /// The ping-pong after the flip, for the caller to store back. The
    /// slot just written is the one this is *not* pointing at.
    pub ping_pong: PingPong,
}

/// Freeze a prefilling request's recurrent state at the deepest chunk
/// boundary strictly inside this extend.
///
/// This is where a hybrid model gets something for the prefix tree to
/// attach. Without it a request produces at most one snapshot, donated
/// at the very end of its prompt, so a follow-up turn that diverges
/// anywhere before that end matches no snapshot at all and replays the
/// whole prompt through the linear layers -- the exact cliff
/// `radix/hybrid.rs` describes, where KV past the last snapshot buys
/// nothing.
///
/// The rules, all three of which have a wrong-looking-right alternative:
///
/// - `chunk = (extend_len - 1) / CHUNK_SIZE`, and nothing is frozen when
///   that is zero. The `- 1` is what makes the boundary **strictly
///   inside** the extend: an extend of exactly one chunk crosses no
///   interior boundary and is skipped.
/// - the boundary is `cached_len + chunk * CHUNK_SIZE`, **never** the
///   extend's end. Only chunk boundaries have a folded state to read at
///   all, and the exact end state is the one sitting in the request's
///   live slot, which is donated when the request finishes -- freezing
///   at the end would spend a track slot to duplicate that donation and
///   leave every interior boundary uncovered.
/// - the destination is the idle half of the ping-pong, which then
///   flips. A chunked prefill calls this once per chunk; writing both
///   into the same slot would overwrite the first checkpoint before the
///   commit that donates it has run, and the tree would end up keyed at
///   one length holding the state of another.
///
/// It is deliberately the same destination-and-flip rule as
/// [`crate::anchor::snapshot_at_anchor`], expressed through the same
/// [`PingPong`], so the chunk-driven and anchor-driven checkpoints
/// cannot drift apart. They never race: this one runs on a prefill
/// extend, that one on a decode step, and a request is doing one or the
/// other in any given forward.
///
/// `at_len` inherits `cached_len`'s page alignment (`CHUNK_SIZE` is a
/// multiple of the page size -- [`crate::radix::HybridRadixCache::new`]
/// refuses any other page size), so an aligned hit stays aligned. The
/// commit path re-checks it before inserting, because a misaligned key
/// would align down and attach a state encoding more tokens to a shorter
/// node.
pub fn track_checkpoint(request: &TrackRequest) -> Option<TrackCheckpoint> {
    let ping_pong = request.ping_pong?;
    // Saturating on 0 rather than underflowing: an empty extend has no
    // boundary, which is what the original's floor division of -1 says.
    let chunk = request.extend_len.checked_sub(1)? / CHUNK_SIZE;
    if chunk < 1 {
        return None;
    }
    Some(TrackCheckpoint {
        into_slot: ping_pong.idle_slot(),
        at_len: request.cached_len + chunk * CHUNK_SIZE,
        chunk,
        ping_pong: ping_pong.flipped(),
    })
}

/// Where each request's per-chunk states start, for a batch whose
/// requests extend by `extend_lens`.
///
/// The exclusive prefix sum of `ceil(extend_len / CHUNK_SIZE)`, one
/// entry per request plus a total, matching the query cumulative-length
/// layout the linear-attention kernel packs its chunk states in. A
/// checkpoint's row is `bases[i] + checkpoint.chunk`.
pub fn chunk_row_bases(extend_lens: &[usize]) -> Vec<usize> {
    let mut bases = Vec::with_capacity(extend_lens.len() + 1);
    let mut total = 0usize;
    bases.push(0);
    for len in extend_lens.iter().copied() {
        total += len.div_ceil(CHUNK_SIZE);
        bases.push(total);
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radix::HybridRadixCache;

    fn pool() -> StateSlotPool {
        // 8 physical slots -> 7 allocatable.
        StateSlotPool::new(8)
    }

    #[test]
    fn a_fresh_pool_holds_every_slot_but_the_padding_sink() {
        let pool = pool();
        assert_eq!(pool.num_slots(), 8);
        assert_eq!(pool.capacity(), 7);
        assert_eq!(pool.available(), 7);
        assert_eq!(pool.allocated(), 0);
        assert!(!pool.is_allocated(PADDING_SLOT));
        pool.check_integrity();
    }

    /// The naive allocator hands out `0 .. num_slots`. That slot is the
    /// padding sink a padded batch folds into, so this test fails if the
    /// free list ever starts at 0 -- a real request would then share its
    /// state with every dummy row in a graph-captured batch.
    #[test]
    fn the_padding_sink_is_never_allocated() {
        let mut pool = pool();
        let all = pool.alloc(7).expect("exactly capacity");
        assert!(!all.contains(&PADDING_SLOT), "{all:?}");
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1..8).collect::<Vec<u32>>());
        assert_eq!(
            pool.alloc(1).unwrap_err().available,
            0,
            "the sink is not a spare slot"
        );
        pool.check_integrity();
    }

    /// And it is not freeable either: a padded batch really does carry
    /// slot 0 as a request's slot, so the free loop over it must be a
    /// no-op rather than a way to make the sink allocatable.
    #[test]
    fn freeing_the_padding_sink_is_a_no_op() {
        let mut pool = pool();
        pool.free(&[PADDING_SLOT, PADDING_SLOT]);
        assert_eq!(pool.available(), 7);
        pool.check_integrity();
        assert!(!pool.alloc(7).unwrap().contains(&PADDING_SLOT));
    }

    #[test]
    fn a_refused_allocation_takes_nothing() {
        let mut pool = pool();
        let err = pool.alloc(9).unwrap_err();
        assert_eq!(err.needed, 9);
        assert_eq!(err.available, 7);
        assert_eq!(pool.available(), 7, "nothing was taken");
        pool.check_integrity();
        // Still fully usable: the caller's next move is evict-and-retry.
        pool.alloc(7).expect("the pool is intact");
        pool.check_integrity();
    }

    #[test]
    fn a_ping_pong_pair_is_both_slots_or_neither() {
        let mut pool = StateSlotPool::new(3);
        let live = pool.alloc_one().expect("one live slot");
        let err = pool.alloc_ping_pong().unwrap_err();
        assert_eq!((err.needed, err.available), (2, 1));
        assert_eq!(pool.available(), 1, "the odd slot was not taken");
        pool.free(&[live]);

        let pair = pool.alloc_ping_pong().expect("two now");
        assert_ne!(pair.slots[0], pair.slots[1]);
        assert_eq!(pair.idle_slot(), pair.slots[0]);
        pool.check_integrity();
    }

    /// Live slots, track slots and tree-donated snapshots come out of
    /// one free list, so a tree that gives its snapshots back funds the
    /// next request's working set.
    #[test]
    fn one_free_list_feeds_all_three_consumers() {
        let mut pool = StateSlotPool::new(5); // 4 usable
        let live = pool.alloc_one().unwrap();
        let track = pool.alloc_ping_pong().unwrap();
        let donated = pool.alloc_one().unwrap();
        assert_eq!(pool.available(), 0, "four usable slots, four in use");
        pool.check_integrity();

        // A snapshot handed back by the tree funds the next admission.
        pool.free(&[donated]);
        let next_live = pool.alloc_one().expect("the donation paid for it");
        assert_eq!(next_live, donated, "the same physical slot, re-purposed");
        pool.free(&[live, next_live]);
        pool.free(&track.slots);
        assert_eq!(pool.available(), 4);
        pool.check_integrity();
    }

    /// The tree still owns a matched snapshot and will hand the same id
    /// to the next request that matches, so a restore copies out of it
    /// rather than resuming in place.
    #[test]
    fn a_restore_copies_out_of_the_snapshot_the_tree_still_owns() {
        let mut pool = pool();
        let snapshot = pool.alloc_one().unwrap();
        let live = pool.alloc_one().unwrap();
        let copy = pool.restore_copy(snapshot, live);
        assert_eq!(copy.from_slot, snapshot);
        assert_eq!(copy.into_slot, live);
        assert!(
            pool.is_allocated(snapshot),
            "the snapshot stays the tree's, ready for the next reader"
        );
    }

    #[test]
    #[should_panic(expected = "writes through the slot the tree still owns")]
    fn a_restore_onto_the_snapshot_itself_is_refused() {
        let mut pool = pool();
        let snapshot = pool.alloc_one().unwrap();
        pool.restore_copy(snapshot, snapshot);
    }

    /// Idle-only and destructive: everything comes back, including what
    /// the tree had, which is why the caller must drop the tree in the
    /// same step.
    #[test]
    fn reclaiming_takes_every_slot_back_including_donated_ones() {
        let mut pool = pool();
        let _live = pool.alloc(3).unwrap();
        pool.reclaim_all();
        assert_eq!(pool.available(), 7);
        assert_eq!(pool.allocated(), 0);
        pool.check_integrity();
        assert!(!pool.alloc(7).unwrap().contains(&PADDING_SLOT));
    }

    #[test]
    fn a_rebuild_resizes_and_drops_everything() {
        let mut pool = pool();
        let _held = pool.alloc(3).unwrap();
        pool.rebuild(16);
        assert_eq!(pool.num_slots(), 16);
        assert_eq!(pool.capacity(), 15);
        assert_eq!(pool.available(), 15);
        assert_eq!(pool.allocated(), 0);
        pool.check_integrity();
    }

    #[test]
    #[should_panic(expected = "padding sink plus at least one usable slot")]
    fn a_pool_with_nothing_but_the_sink_is_refused() {
        StateSlotPool::new(1);
    }

    /// The invariant is an equality precisely so it catches a leak, not
    /// only a double free.
    #[test]
    #[should_panic(expected = "leaked or double-freed")]
    fn the_invariant_catches_a_leaked_slot() {
        let mut pool = pool();
        let held = pool.alloc(2).unwrap();
        // Simulate a caller that dropped a slot on the floor -- an
        // evicted snapshot it never handed back.
        pool.allocated[held[0] as usize] = false;
        pool.check_integrity();
    }

    /// And the free list is checked for duplicates as well as counted,
    /// because a slot handed back twice while another is lost balances
    /// the arithmetic and still serves one state to two sequences.
    #[test]
    #[should_panic(expected = "on the free list twice")]
    fn the_invariant_catches_a_slot_that_is_free_twice() {
        let mut pool = pool();
        let held = pool.alloc(2).unwrap();
        pool.free(&held);
        let duplicate = pool.free[0];
        pool.free[1] = duplicate;
        pool.check_integrity();
    }

    /// Every slot the hybrid cache refuses, evicts or is drained of
    /// comes back to the pool exactly once -- the property that makes
    /// the two modules a matched pair rather than two leak sources.
    #[test]
    fn slots_are_conserved_across_a_hybrid_cache_workload() {
        let mut pool = StateSlotPool::new(64);
        let mut cache = HybridRadixCache::new(4);
        for round in 0..12u32 {
            let prompt: Vec<u32> = (0..8u32)
                .map(|i| if i < 4 { i } else { i + round % 3 })
                .collect();
            let pages: Vec<u32> = (round * 8..round * 8 + 8).collect();

            let live = pool.alloc_one().expect("a live slot");
            let track = pool.alloc_ping_pong().expect("two track slots");
            let donation = track.idle_slot();
            if cache.insert(&prompt, &pages, donation).snapshot_exists {
                // Refused: the slot is still ours.
                pool.free(&[donation]);
                pool.free(&[track.flipped().idle_slot()]);
            } else {
                pool.free(&[track.flipped().idle_slot()]);
            }
            pool.free(&[live]);
            cache.check_integrity();
            pool.check_integrity();
        }
        let drained = cache.evict_full(usize::MAX);
        pool.free(&drained.mamba_slots);
        assert_eq!(pool.available(), pool.capacity(), "every slot came home");
        pool.check_integrity();
    }

    fn track(cached_len: usize, extend_len: usize, ping_pong: PingPong) -> TrackRequest {
        TrackRequest {
            cached_len,
            extend_len,
            ping_pong: Some(ping_pong),
        }
    }

    #[test]
    fn a_multi_chunk_extend_freezes_at_its_deepest_interior_boundary() {
        let request = track(128, 3 * CHUNK_SIZE + 5, PingPong::new(11, 12));
        let checkpoint = track_checkpoint(&request).expect("three chunks crossed");
        assert_eq!(checkpoint.chunk, 3);
        assert_eq!(checkpoint.at_len, 128 + 3 * CHUNK_SIZE);
        assert_eq!(checkpoint.into_slot, 11);
        assert_eq!(checkpoint.ping_pong.idle_slot(), 12);
    }

    /// The naive rule freezes at the extend's end. It must not: only
    /// chunk boundaries have a folded state, and the end state is
    /// already in the live slot, donated at finish. This fails if
    /// `at_len` ever reaches the end of the extend, and if an extend of
    /// exactly one chunk (which crosses no interior boundary) produces a
    /// checkpoint at all.
    #[test]
    fn the_checkpoint_is_strictly_inside_the_extend_never_at_its_end() {
        let pp = PingPong::new(11, 12);
        for extend_len in 1..=(6 * CHUNK_SIZE) {
            for cached_len in [0usize, 64, 512] {
                let request = track(cached_len, extend_len, pp);
                let end = cached_len + extend_len;
                match track_checkpoint(&request) {
                    Some(checkpoint) => {
                        assert!(
                            checkpoint.at_len < end,
                            "extend {extend_len} froze at its end {end}"
                        );
                        assert!(checkpoint.at_len > cached_len, "extend {extend_len}");
                        assert_eq!(
                            checkpoint.at_len % CHUNK_SIZE,
                            cached_len % CHUNK_SIZE,
                            "extend {extend_len} left a chunk boundary"
                        );
                        // Deepest: one more chunk would be at or past the end.
                        assert!(checkpoint.at_len + CHUNK_SIZE >= end);
                    }
                    None => assert!(
                        extend_len <= CHUNK_SIZE,
                        "extend {extend_len} crosses a boundary and was skipped"
                    ),
                }
            }
        }
        // The exact boundary case, spelled out: one whole chunk is
        // skipped, one chunk plus a token freezes at the chunk.
        assert!(track_checkpoint(&track(0, CHUNK_SIZE, pp)).is_none());
        assert_eq!(
            track_checkpoint(&track(0, CHUNK_SIZE + 1, pp))
                .expect("now it crosses")
                .at_len,
            CHUNK_SIZE
        );
    }

    /// The naive rule writes every checkpoint into the same slot. It
    /// must not: a chunked prefill tracks once per chunk, and the second
    /// write would land on the state the commit has not yet donated --
    /// the tree would be keyed at the first boundary holding the second
    /// boundary's state. This fails if the destination does not
    /// alternate.
    #[test]
    fn consecutive_chunks_of_one_prefill_alternate_their_destination() {
        let mut ping_pong = PingPong::new(11, 12);
        let mut cached_len = 0usize;
        let mut destinations = Vec::new();
        let mut boundaries = Vec::new();
        for _ in 0..4 {
            let extend_len = 3 * CHUNK_SIZE;
            let checkpoint = track_checkpoint(&track(cached_len, extend_len, ping_pong))
                .expect("two interior boundaries per chunk");
            destinations.push(checkpoint.into_slot);
            boundaries.push(checkpoint.at_len);
            ping_pong = checkpoint.ping_pong;
            cached_len += extend_len;
        }
        assert_eq!(destinations, vec![11, 12, 11, 12]);
        for pair in destinations.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "a checkpoint overwrote the one before it was donated"
            );
        }
        // And the boundaries advance, so the two live snapshots are
        // genuinely different prefixes.
        assert_eq!(
            boundaries,
            vec![
                2 * CHUNK_SIZE,
                3 * CHUNK_SIZE + 2 * CHUNK_SIZE,
                6 * CHUNK_SIZE + 2 * CHUNK_SIZE,
                9 * CHUNK_SIZE + 2 * CHUNK_SIZE
            ]
        );
    }

    #[test]
    fn a_request_that_is_not_caching_never_tracks() {
        let request = TrackRequest {
            cached_len: 0,
            extend_len: 10 * CHUNK_SIZE,
            ping_pong: None,
        };
        assert!(track_checkpoint(&request).is_none());
    }

    /// An empty extend has no boundary; the original's floor division
    /// says the same with a negative chunk count, which must not
    /// underflow here.
    #[test]
    fn an_empty_extend_is_skipped_rather_than_underflowing() {
        assert!(track_checkpoint(&track(512, 0, PingPong::new(11, 12))).is_none());
    }

    /// The chunk-driven and anchor-driven checkpoints share one
    /// destination rule, so a request that takes one and then the other
    /// still alternates instead of overwriting.
    #[test]
    fn the_chunk_track_and_the_anchor_snapshot_agree_on_the_destination() {
        use crate::anchor::{snapshot_at_anchor, RecurrentState};

        let ping_pong = PingPong::new(11, 12);
        let checkpoint =
            track_checkpoint(&track(0, 2 * CHUNK_SIZE, ping_pong)).expect("one boundary");
        let state = RecurrentState {
            live_slot: 7,
            ping_pong: checkpoint.ping_pong,
            pending_snapshot: None,
            cached_len: 4 * CHUNK_SIZE,
        };
        let anchored = snapshot_at_anchor(Some(4 * CHUNK_SIZE), &state, 64).expect("an anchor");
        assert_ne!(
            anchored.into_slot, checkpoint.into_slot,
            "the anchor snapshot would have overwritten the chunk track's pending freeze"
        );
    }

    #[test]
    fn chunk_row_bases_are_the_exclusive_prefix_sum_of_whole_chunks() {
        let lens = [CHUNK_SIZE, CHUNK_SIZE + 1, 0, 3 * CHUNK_SIZE];
        assert_eq!(chunk_row_bases(&lens), vec![0, 1, 3, 3, 6]);
        assert_eq!(chunk_row_bases(&[]), vec![0]);

        // A checkpoint's row is the request's base plus its chunk.
        let bases = chunk_row_bases(&lens);
        let checkpoint = track_checkpoint(&track(0, lens[3], PingPong::new(1, 2))).unwrap();
        assert_eq!(bases[3] + checkpoint.chunk, 5);
    }
}
