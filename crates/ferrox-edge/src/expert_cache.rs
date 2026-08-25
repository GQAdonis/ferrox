//! The global expert slot cache: which experts are resident on the GPU
//! right now, and what one decode step has to move.
//!
//! # One id space, one pool
//!
//! Experts are addressed by a **flat id**, `layer * num_experts +
//! expert`, and all layers compete for one pool of `cache_size` slots.
//! That is the "global" in global LRU, and it is the point: expert
//! activation is not uniform across layers, so a per-layer cache of
//! `cache_size / num_layers` slots wastes residency on layers whose
//! routing is flat and starves the layers where a few experts take most
//! of the traffic.
//!
//! # What a step produces
//!
//! [`ExpertCache::ensure`] takes the experts a layer routed to and
//! returns, for each, the slot it will be read from -- plus a
//! [`CopyPlan`]: the (slot, host row) pairs the caller must copy before
//! the multiply. Nothing here moves bytes; the plan is the whole
//! output.
//!
//! # Two rules that must not drift
//!
//! - **Victims are chosen by `(usage, slot)` ascending**, and a slot
//!   touched *by this very step* is not a victim. Without the second
//!   rule a step that misses more experts than it hits can evict an
//!   expert it is about to read.
//! - **Duplicate routes collapse.** A batch routing twice to the same
//!   expert counts one miss, issues one copy, and shares one slot.
//!
//! The hybrid entry point adds the [`crate::qstar`] split on top: only
//! the first `fetch` misses get slots, and the rest come back as
//! [`None`], meaning "compute this one on the CPU". Every route is
//! assigned exactly once, to exactly one device.
//!
//! Ported 1:1 from FreeToken's `moe/offload_cache.py` and
//! `moe/offload_kernels.py`, whose LRU is the `lru_ensure` kernel from
//! `flashlib` (both Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use crate::qstar::QStarPolicy;

/// A cached expert's address in the flat id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

/// The copies a step needs before it can multiply.
///
/// `dst_slots[i]` is a cache slot; `src_rows[i]` is a **layer-local**
/// expert row, so it indexes this layer's own host bank rather than a
/// flat table. Keeping it layer-local is what lets each layer's weights
/// live in their own allocation, which in turn is what lets residency
/// (pinned / locked / pageable) differ per layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CopyPlan {
    pub dst_slots: Vec<u32>,
    pub src_rows: Vec<u32>,
}

impl CopyPlan {
    pub fn len(&self) -> usize {
        self.dst_slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dst_slots.is_empty()
    }
}

/// What one `ensure` decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnsurePlan {
    /// Per routed expert, in the caller's order: the slot to read it
    /// from, or `None` for "compute this one on the CPU" (hybrid only).
    pub slots: Vec<Option<u32>>,
    pub copy: CopyPlan,
    /// Distinct experts this step routed to.
    pub active: usize,
    /// Distinct experts that were not resident -- *before* any fetch
    /// cap was applied.
    pub missing: usize,
    /// Distinct experts actually being fetched over the link.
    pub fetched: usize,
}

/// Which miss gets the scarce fetch slots when the split caps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FetchOrder {
    /// Most-recently-active expert first. An expert that keeps being
    /// routed to but keeps losing the cap ranks higher every step until
    /// it wins one, so a hot expert converges into residency instead of
    /// being computed on the CPU forever.
    #[default]
    ByRecency,
    /// Lowest expert id first. Deterministic and cheap; kept because it
    /// is the reference order the two implementations were first
    /// cross-checked against.
    LowestId,
}

/// Running counters, for `/metrics` and for deciding whether the cache
/// is sized right.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    /// `ensure` calls.
    pub calls: u64,
    /// Distinct experts routed to, summed over calls.
    pub active: u64,
    /// Distinct misses, summed over calls, before any cap.
    pub missing: u64,
    /// Misses actually fetched, summed over calls.
    pub fetched: u64,
}

impl ExpertCacheStats {
    /// Fraction of routed experts that were not resident. The number
    /// that says whether the cache is big enough.
    pub fn miss_rate(&self) -> f64 {
        if self.active == 0 {
            return 0.0;
        }
        self.missing as f64 / self.active as f64
    }

    /// Fraction of misses served over the link rather than by the CPU.
    /// The number that says what the `q*` split actually did.
    pub fn fetch_rate(&self) -> f64 {
        if self.missing == 0 {
            return 0.0;
        }
        self.fetched as f64 / self.missing as f64
    }

    pub fn active_per_call(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        self.active as f64 / self.calls as f64
    }
}

/// Marks a slot as unevictable for this step.
const UNEVICTABLE: i64 = i64::MAX;

/// The GPU expert cache's residency map.
#[derive(Debug)]
pub struct ExpertCache {
    num_layers: usize,
    num_experts: usize,
    cache_size: usize,
    /// Flat id -> slot, or -1.
    slot_for_id: Vec<i64>,
    /// Slot -> flat id, or -1 for empty.
    id_of_slot: Vec<i64>,
    /// Slot -> the step that last touched it. The LRU clock.
    usage: Vec<i64>,
    step: i64,
    /// Flat id -> the step this expert was last *routed to*, resident
    /// or not. Distinct from `usage`, which only tracks residency.
    recency: Vec<i64>,
    fetch_order: FetchOrder,
    stats: ExpertCacheStats,
}

impl ExpertCache {
    /// A cold cache of `cache_size` slots shared by every layer.
    ///
    /// `cache_size` must hold at least one whole layer: a prefill
    /// materializes a layer's experts into slots `0..num_experts`, and
    /// a cache that cannot hold one layer cannot serve a prefill at
    /// all.
    pub fn new(num_layers: usize, num_experts: usize, cache_size: usize) -> Self {
        assert!(
            num_layers > 0 && num_experts > 0,
            "a MoE model has layers and experts"
        );
        assert!(
            cache_size >= num_experts,
            "cache_size {cache_size} cannot hold one layer of {num_experts} experts"
        );
        let total_ids = num_layers * num_experts;
        ExpertCache {
            num_layers,
            num_experts,
            cache_size,
            slot_for_id: vec![-1; total_ids],
            id_of_slot: vec![-1; cache_size],
            usage: vec![0; cache_size],
            step: 0,
            recency: vec![-1; total_ids],
            fetch_order: FetchOrder::default(),
            stats: ExpertCacheStats::default(),
        }
    }

    pub fn with_fetch_order(mut self, order: FetchOrder) -> Self {
        self.fetch_order = order;
        self
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = ExpertCacheStats::default();
    }

    /// Total routed experts across the model -- the denominator the
    /// cache's residency rate is quoted against.
    pub fn total_experts(&self) -> usize {
        self.num_layers * self.num_experts
    }

    fn flat(&self, layer: u32, expert: u32) -> usize {
        let layer = layer as usize;
        let expert = expert as usize;
        debug_assert!(layer < self.num_layers && expert < self.num_experts);
        layer * self.num_experts + expert
    }

    /// Which slot holds `(layer, expert)`, if any. For tests and
    /// reporting; the hot path uses [`ensure`](Self::ensure).
    pub fn slot_of(&self, layer: u32, expert: u32) -> Option<u32> {
        let slot = self.slot_for_id[self.flat(layer, expert)];
        (slot >= 0).then_some(slot as u32)
    }

    /// Which expert occupies `slot`, if any.
    pub fn resident_in(&self, slot: u32) -> Option<ExpertId> {
        let id = self.id_of_slot[slot as usize];
        (id >= 0).then(|| ExpertId {
            layer: (id as usize / self.num_experts) as u32,
            expert: (id as usize % self.num_experts) as u32,
        })
    }

    /// Slots currently holding an expert.
    pub fn resident_slots(&self) -> usize {
        self.id_of_slot.iter().filter(|id| **id >= 0).count()
    }

    /// Forget everything. A rebuild is a cold start, so the counters go
    /// too -- carrying them across would skew every rate that is quoted
    /// per call.
    pub fn reset(&mut self) {
        self.slot_for_id.fill(-1);
        self.id_of_slot.fill(-1);
        self.usage.fill(0);
        self.recency.fill(-1);
        self.step = 0;
        self.stats = ExpertCacheStats::default();
    }

    /// Resize the pool, keeping the model geometry.
    ///
    /// Everything resident is dropped: slot ids are positions in an
    /// allocation that no longer exists, so keeping the map would point
    /// at other experts' bytes. Refusing an impossible target *before*
    /// touching anything is deliberate -- the caller can then keep
    /// serving from the cache it already has.
    pub fn rebuild(&mut self, cache_size: usize) -> Result<(), RebuildRejected> {
        if cache_size < self.num_experts {
            return Err(RebuildRejected {
                requested: cache_size,
                minimum: self.num_experts,
            });
        }
        self.cache_size = cache_size;
        self.id_of_slot = vec![-1; cache_size];
        self.usage = vec![0; cache_size];
        self.slot_for_id.fill(-1);
        self.recency.fill(-1);
        self.step = 0;
        self.stats = ExpertCacheStats::default();
        Ok(())
    }

    /// Make every expert this layer routed to resident, evicting by LRU.
    ///
    /// Every route gets a slot: this is the pure-offload path, where
    /// the CPU computes nothing.
    pub fn ensure(&mut self, layer: u32, expert_ids: &[u32]) -> EnsurePlan {
        self.ensure_split(layer, expert_ids, None)
    }

    /// The bandwidth-adaptive path: fetch what the `q*` split says to
    /// fetch, and hand the rest back for the CPU.
    ///
    /// A route that comes back `None` is the caller's to compute from
    /// host RAM. Together with the `Some` routes it covers every
    /// routed expert exactly once -- which is what makes it safe to
    /// simply add the two partial results.
    pub fn ensure_hybrid(
        &mut self,
        layer: u32,
        expert_ids: &[u32],
        policy: &QStarPolicy,
    ) -> EnsurePlan {
        self.ensure_split(layer, expert_ids, Some(policy))
    }

    fn ensure_split(
        &mut self,
        layer: u32,
        expert_ids: &[u32],
        policy: Option<&QStarPolicy>,
    ) -> EnsurePlan {
        self.step += 1;
        let step = self.step;
        let base = self.flat(layer, 0);

        // Distinct routes, first occurrence wins. A batch that routes
        // twice to one expert must not fetch it twice or, worse, evict
        // its own first copy.
        let mut distinct: Vec<u32> = Vec::with_capacity(expert_ids.len());
        for id in expert_ids {
            if !distinct.contains(id) {
                distinct.push(*id);
            }
        }

        let mut missing: Vec<u32> = Vec::new();
        for expert in &distinct {
            let slot = self.slot_for_id[base + *expert as usize];
            if slot >= 0 {
                // A hit refreshes the slot, which also makes it
                // unevictable for the rest of this step.
                self.usage[slot as usize] = step;
            } else {
                missing.push(*expert);
            }
        }

        match (policy, self.fetch_order) {
            // The capped path prefers experts that were routed to most
            // recently, so a hot expert that keeps losing the cap wins
            // one eventually.
            (Some(_), FetchOrder::ByRecency) => {
                missing.sort_by_key(|e| (-self.recency[base + *e as usize], *e));
            }
            _ => missing.sort_unstable(),
        }

        let num_missing = missing.len();
        let num_fetch = match policy {
            Some(policy) => policy.split(num_missing).fetch,
            None => num_missing,
        };

        // Slots this step already committed to -- hits above, and each
        // victim as it is claimed -- are off the table.
        let mut evictable: Vec<i64> = self
            .usage
            .iter()
            .map(|u| if *u == step { UNEVICTABLE } else { *u })
            .collect();
        // Under a cap, a slot holding an expert this step routes to is
        // also off the table even if the route is being sent to the
        // CPU: evicting it would throw away residency the very next
        // step wants.
        if policy.is_some() {
            for (evict, id) in evictable.iter_mut().zip(self.id_of_slot.iter()) {
                if *id < 0 {
                    continue;
                }
                let owner = *id - base as i64;
                if (0..self.num_experts as i64).contains(&owner)
                    && distinct.contains(&(owner as u32))
                {
                    *evict = UNEVICTABLE;
                }
            }
        }

        let mut copy = CopyPlan::default();
        for expert in missing.iter().take(num_fetch) {
            let victim = argmin_slot(&evictable);
            let old = self.id_of_slot[victim];
            if old >= 0 {
                self.slot_for_id[old as usize] = -1;
            }
            let id = base + *expert as usize;
            self.id_of_slot[victim] = id as i64;
            self.slot_for_id[id] = victim as i64;
            self.usage[victim] = step;
            evictable[victim] = UNEVICTABLE;
            copy.dst_slots.push(victim as u32);
            // Layer-local, so it indexes this layer's own host bank.
            copy.src_rows.push(*expert);
        }

        let slots: Vec<Option<u32>> = expert_ids
            .iter()
            .map(|expert| {
                let slot = self.slot_for_id[base + *expert as usize];
                (slot >= 0).then_some(slot as u32)
            })
            .collect();

        if policy.is_some() {
            for expert in &distinct {
                self.recency[base + *expert as usize] = step;
            }
        }

        self.stats.calls += 1;
        self.stats.active += distinct.len() as u64;
        self.stats.missing += num_missing as u64;
        self.stats.fetched += num_fetch as u64;

        EnsurePlan {
            slots,
            copy,
            active: distinct.len(),
            missing: num_missing,
            fetched: num_fetch,
        }
    }

    /// Put a whole layer's experts in slots `0..num_experts`, in expert
    /// order, for a prefill.
    ///
    /// A prefill touches every expert, so streaming them one miss at a
    /// time is pointless -- the layer is copied whole, and **position
    /// equals expert id**, which lets routing ids index the buffer
    /// directly with no slot lookup at all.
    ///
    /// The bookkeeping still has to be exact: any *other* layer's
    /// expert that was living in one of those slots loses its
    /// residency here, or the next decode step would hit on a slot that
    /// now holds someone else's bytes.
    pub fn materialize_layer(&mut self, layer: u32) -> CopyPlan {
        let base = self.flat(layer, 0);
        let owns = |id: i64| id >= base as i64 && id < (base + self.num_experts) as i64;
        // Snapshot before mutating: the checks below all read the
        // pre-existing occupant.
        let previous: Vec<i64> = self.id_of_slot.clone();

        for (slot, old) in previous.iter().enumerate() {
            if owns(*old) {
                // This layer's own stale placements go; they are about
                // to be re-established at position == expert id.
                self.id_of_slot[slot] = -1;
                self.usage[slot] = 0;
            }
        }
        for old in previous.iter().take(self.num_experts) {
            if *old >= 0 && !owns(*old) {
                self.slot_for_id[*old as usize] = -1;
            }
        }

        self.step += 1;
        let mut copy = CopyPlan::default();
        for expert in 0..self.num_experts {
            self.id_of_slot[expert] = (base + expert) as i64;
            self.slot_for_id[base + expert] = expert as i64;
            self.usage[expert] = self.step;
            copy.dst_slots.push(expert as u32);
            copy.src_rows.push(expert as u32);
        }
        copy
    }
}

/// A slot resize the cache refused, having changed nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebuildRejected {
    pub requested: usize,
    pub minimum: usize,
}

impl std::fmt::Display for RebuildRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "expert cache of {} slots cannot hold one layer of {} experts",
            self.requested, self.minimum
        )
    }
}

impl std::error::Error for RebuildRejected {}

/// The lowest-`usage` slot, ties to the lowest slot index.
///
/// The tie-break is not cosmetic: a GPU kernel and a CPU reference have
/// to pick the same victim, or the two halves of a hybrid step disagree
/// about which expert is where.
fn argmin_slot(usage: &[i64]) -> usize {
    let mut best = 0usize;
    let mut best_usage = usage[0];
    for (slot, value) in usage.iter().enumerate().skip(1) {
        if *value < best_usage {
            best = slot;
            best_usage = *value;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> ExpertCache {
        ExpertCache::new(4, 8, 16)
    }

    #[test]
    fn a_cold_miss_is_fetched_and_becomes_resident() {
        let mut cache = cache();
        let plan = cache.ensure(0, &[3, 5]);
        assert_eq!(plan.missing, 2);
        assert_eq!(plan.fetched, 2);
        assert_eq!(plan.copy.src_rows, vec![3, 5], "layer-local rows");
        assert_eq!(plan.slots.len(), 2);
        assert!(plan.slots.iter().all(Option::is_some));

        // The second visit is free.
        let again = cache.ensure(0, &[3, 5]);
        assert_eq!(again.missing, 0);
        assert!(again.copy.is_empty());
        assert_eq!(again.slots, plan.slots);
    }

    /// Layers share one pool, so the same expert *index* in two layers
    /// is two different residents.
    #[test]
    fn layers_share_one_pool_through_a_flat_id_space() {
        let mut cache = cache();
        cache.ensure(0, &[3]);
        let plan = cache.ensure(1, &[3]);
        assert_eq!(plan.missing, 1, "layer 1's expert 3 is a different expert");
        assert_ne!(cache.slot_of(0, 3), cache.slot_of(1, 3));
        assert_eq!(cache.resident_slots(), 2);
    }

    #[test]
    fn duplicate_routes_collapse_into_one_copy_and_one_slot() {
        let mut cache = cache();
        let plan = cache.ensure(0, &[7, 7, 7, 2]);
        assert_eq!(plan.active, 2);
        assert_eq!(plan.copy.len(), 2, "one copy per distinct expert");
        assert_eq!(plan.slots[0], plan.slots[1]);
        assert_eq!(plan.slots[1], plan.slots[2]);
        assert_ne!(plan.slots[0], plan.slots[3]);
    }

    /// The rule that keeps a step from sabotaging itself: an expert
    /// this step just hit cannot be the victim of this step's misses.
    #[test]
    fn a_step_never_evicts_an_expert_it_is_about_to_read() {
        // Two layers competing for exactly one layer's worth of slots.
        let mut cache = ExpertCache::new(2, 4, 4);
        cache.ensure(0, &[0, 1, 2, 3]);
        cache.ensure(1, &[0]);
        let held = cache.slot_of(1, 0).unwrap();

        // A route to a resident expert plus a miss: the miss must take
        // some *other* slot.
        let plan = cache.ensure(1, &[0, 1]);
        assert_eq!(plan.missing, 1);
        assert_eq!(cache.slot_of(1, 0), Some(held), "the hit kept its slot");
        assert_ne!(plan.copy.dst_slots[0], held);
        assert_eq!(plan.slots[0], Some(held));
    }

    #[test]
    fn eviction_takes_the_least_recently_used_slot() {
        let mut cache = ExpertCache::new(2, 4, 4);
        for expert in 0..4u32 {
            cache.ensure(0, &[expert]); // one step each, ascending usage
        }
        cache.ensure(0, &[0]); // refresh 0, making 1 the oldest
        let oldest = cache.slot_of(0, 1).unwrap();

        let plan = cache.ensure(1, &[0]);
        assert_eq!(plan.copy.dst_slots, vec![oldest]);
        assert_eq!(cache.slot_of(0, 1), None, "the oldest went");
        assert!(cache.slot_of(0, 0).is_some(), "the refreshed one stayed");
    }

    /// A large cold batch must give every distinct expert its own slot
    /// -- no two routes may share one.
    #[test]
    fn a_large_cold_batch_assigns_unique_slots() {
        let mut cache = ExpertCache::new(4, 64, 256);
        let ids: Vec<u32> = (0..64).collect();
        let plan = cache.ensure(2, &ids);
        assert_eq!(plan.missing, 64);
        assert_eq!(plan.copy.len(), 64);

        let mut slots: Vec<u32> = plan.slots.iter().map(|s| s.unwrap()).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 64, "no slot serves two experts");
        assert_eq!(
            plan.copy.src_rows, ids,
            "misses are ranked by ascending expert id"
        );
    }

    /// Every routed expert is assigned exactly once, to exactly one
    /// device -- the invariant that makes adding the GPU and CPU
    /// partial results correct.
    #[test]
    fn the_hybrid_split_assigns_every_route_to_exactly_one_device() {
        let mut cache = ExpertCache::new(1, 32, 40);
        let policy = QStarPolicy::from_fraction(0.415);
        let routes: Vec<u32> = (0..8).collect();

        let plan = cache.ensure_hybrid(0, &routes, &policy);
        assert_eq!(plan.missing, 8);
        assert_eq!(plan.fetched, 3, "0.415 * 8 = 3.32 -> 3");
        assert_eq!(plan.copy.len(), 3);
        let on_gpu = plan.slots.iter().filter(|s| s.is_some()).count();
        assert_eq!(on_gpu, 3);
        assert_eq!(plan.slots.iter().filter(|s| s.is_none()).count(), 5);
    }

    /// A hot expert that keeps losing the cap must eventually win one,
    /// or it is computed on the CPU forever while the cache holds cold
    /// experts.
    #[test]
    fn a_recurring_miss_climbs_the_fetch_order() {
        let mut cache = ExpertCache::new(1, 16, 16);
        let policy = QStarPolicy::fixed_cap(1);
        // Expert 9 is routed every step alongside a rotating cast.
        for round in 0..6u32 {
            cache.ensure_hybrid(0, &[9, round], &policy);
        }
        assert!(
            cache.slot_of(0, 9).is_some(),
            "the expert routed every step became resident"
        );
    }

    #[test]
    fn the_fixed_cap_is_the_unbenchmarked_default() {
        let mut cache = ExpertCache::new(1, 32, 40);
        let plan = cache.ensure_hybrid(0, &[0, 1, 2, 3, 4, 5, 6, 7], &QStarPolicy::fixed_cap(1));
        assert_eq!(plan.missing, 8);
        assert_eq!(plan.fetched, 1);
        assert_eq!(plan.copy.len(), 1);
    }

    /// A prefill copies a whole layer and lays it out so that position
    /// equals expert id -- and must not leave another layer thinking it
    /// still owns one of those slots.
    #[test]
    fn materializing_a_layer_reclaims_other_layers_slots_cleanly() {
        let mut cache = ExpertCache::new(2, 4, 4);
        cache.ensure(0, &[0, 1, 2, 3]);
        assert_eq!(cache.resident_slots(), 4);

        let copy = cache.materialize_layer(1);
        assert_eq!(copy.dst_slots, vec![0, 1, 2, 3]);
        assert_eq!(copy.src_rows, vec![0, 1, 2, 3], "position == expert id");
        for expert in 0..4u32 {
            assert_eq!(cache.slot_of(1, expert), Some(expert));
            assert_eq!(
                cache.slot_of(0, expert),
                None,
                "layer 0 must not hit on layer 1's bytes"
            );
        }

        // And a later decode on layer 0 misses and reloads, rather than
        // reading whatever is in the slot.
        let plan = cache.ensure(0, &[3]);
        assert_eq!(plan.missing, 1);
    }

    #[test]
    fn a_second_materialize_of_the_same_layer_is_idempotent() {
        let mut cache = ExpertCache::new(2, 4, 6);
        cache.materialize_layer(1);
        cache.materialize_layer(1);
        for expert in 0..4u32 {
            assert_eq!(cache.slot_of(1, expert), Some(expert));
        }
        assert_eq!(cache.resident_slots(), 4);
    }

    #[test]
    fn stats_answer_whether_the_cache_is_big_enough() {
        let mut cache = ExpertCache::new(1, 2, 2);
        cache.ensure(0, &[0, 1]); // 2 misses
        cache.ensure(0, &[0, 1]); // 2 hits
        let stats = cache.stats();
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.active, 4);
        assert_eq!(stats.missing, 2);
        assert_eq!(stats.miss_rate(), 0.5);
        assert_eq!(stats.fetch_rate(), 1.0);
        assert_eq!(stats.active_per_call(), 2.0);
    }

    #[test]
    fn a_rebuild_that_cannot_hold_a_layer_changes_nothing() {
        let mut cache = ExpertCache::new(2, 8, 16);
        cache.ensure(0, &[1]);
        let err = cache.rebuild(4).unwrap_err();
        assert_eq!(err.minimum, 8);
        assert_eq!(cache.cache_size(), 16, "still serving from the old cache");
        assert!(cache.slot_of(0, 1).is_some());

        cache.rebuild(8).expect("one layer fits");
        assert_eq!(cache.cache_size(), 8);
        assert_eq!(
            cache.resident_slots(),
            0,
            "slot ids no longer mean anything"
        );
    }

    #[test]
    #[should_panic(expected = "cannot hold one layer")]
    fn a_cache_too_small_for_one_layer_is_rejected_at_construction() {
        ExpertCache::new(2, 8, 4);
    }

    /// The residency map must stay coherent under sustained pressure:
    /// no slot claimed by two experts, no expert claiming a slot that
    /// does not point back.
    #[test]
    fn the_residency_map_stays_bijective_under_pressure() {
        let mut cache = ExpertCache::new(4, 16, 20);
        let policy = QStarPolicy::from_fraction(0.5);
        for step in 0..200u32 {
            let layer = step % 4;
            let routes: Vec<u32> = (0..6).map(|i| (step * 7 + i * 3) % 16).collect();
            let plan = if step % 2 == 0 {
                cache.ensure(layer, &routes)
            } else {
                cache.ensure_hybrid(layer, &routes, &policy)
            };
            assert_eq!(plan.slots.len(), routes.len());

            for slot in 0..cache.cache_size() as u32 {
                if let Some(id) = cache.resident_in(slot) {
                    assert_eq!(
                        cache.slot_of(id.layer, id.expert),
                        Some(slot),
                        "slot {slot} and its occupant disagree"
                    );
                }
            }
            assert!(cache.resident_slots() <= cache.cache_size());
        }
    }
}
