//! Elastic memory: how VRAM is split between the expert cache and the
//! KV pools, and how that split is changed while the server is
//! running.
//!
//! # Why MoE gets first claim
//!
//! Both pools compete for the same bytes, but they do not degrade the
//! same way. A KV pool one page short means one fewer concurrent
//! request or a shorter context -- a scheduling limit, felt as a queue.
//! An expert cache one slot short means every step that routes to the
//! missing expert pays a PCIe transfer or a CPU detour -- a *per-token*
//! tax on every request, forever. So [`plan_cache_budget`] fills the
//! expert cache first, up to full residency, and gives the remainder to
//! KV -- with a floor of `kv_reserve_pages`, because a server that
//! cannot hold a context serves nothing at all.
//!
//! # Why a rebuild validates before it frees
//!
//! Re-splitting the pools means dropping allocations and taking new
//! ones. If the new size does not fit, the old pool is already gone and
//! the server is down -- for a request that was never going to work.
//! So every size check happens up front, on arithmetic alone
//! ([`validate_rebuild`]), and a rejection leaves the engine serving
//! exactly what it was serving before.
//!
//! Ported 1:1 from FreeToken's `engine/cache_budget.py` and the pool
//! sizing in `kvcache/hybrid_swa_pool.py` / `kvcache/base.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use crate::radix::align_ceil;

/// How many tokens a window request keeps live past the window itself,
/// so a decode step that slides the window does not immediately need
/// state it just freed.
pub const SWA_RETAIN_GAP: usize = 16;

/// How often the window is slid, in decode steps. Sliding every step
/// would cost a pool operation per token; sliding rarely means the pool
/// must hold that many extra tokens per request. The pool floor below
/// pays for exactly this.
pub const DEFAULT_SWA_EVICTION_INTERVAL: usize = 128;

/// The VRAM a rebuild may spend: a fraction of what was free before the
/// weights loaded, minus the weights, minus whatever the pools need
/// unconditionally.
///
/// The fraction is not padding for its own sake -- it is the room the
/// activations, the workspace, and the captured graphs occupy, none of
/// which is in `weights_bytes`. Spending it produces an allocation
/// failure at the first long prompt rather than at startup.
///
/// Signed on purpose: an over-committed deployment gets a negative
/// budget, which every consumer below then refuses, rather than a
/// wrapped-around enormous one.
pub fn net_cache_budget_bytes(
    memory_ratio: f64,
    baseline_free_bytes: u64,
    weights_bytes: u64,
    fixed_cache_bytes: u64,
) -> i64 {
    (memory_ratio * baseline_free_bytes as f64) as i64
        - weights_bytes as i64
        - fixed_cache_bytes as i64
}

/// What a given split would actually cost.
pub fn required_bytes(
    moe_cache_slots: u64,
    kv_pages: u64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
) -> i64 {
    (moe_cache_slots * bytes_per_expert + kv_pages * bytes_per_page) as i64
}

/// The KV budget at startup, when the weights have just been loaded.
///
/// `init_free - new_free` is what loading actually consumed, measured
/// rather than predicted -- which is why this is not the same
/// expression as [`net_cache_budget_bytes`], and why it is the one used
/// before any pool exists.
pub fn startup_kv_budget(memory_ratio: f64, init_free_bytes: u64, new_free_bytes: u64) -> i64 {
    (memory_ratio * init_free_bytes as f64) as i64
        - (init_free_bytes as i64 - new_free_bytes as i64)
}

/// A pool split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSizes {
    /// Slots in the GPU expert cache.
    pub moe_cache_slots: u64,
    /// Pages in the KV pool.
    pub kv_pages: u64,
    /// Whether the prefill double buffer survived the split. It needs
    /// two layers' worth of slots, and a tight budget can take that
    /// away.
    pub prefill_overlap: bool,
}

/// A split that does not fit, refused before anything was freed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetTooSmall {
    pub needed_bytes: i64,
    pub budget_bytes: i64,
    /// What the arithmetic wanted to allocate.
    pub sizes: PoolSizes,
}

impl std::fmt::Display for BudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requested cache (moe={} slots, kv={} pages) needs {} bytes but the budget is {}; \
             the old cache is kept and still serving",
            self.sizes.moe_cache_slots, self.sizes.kv_pages, self.needed_bytes, self.budget_bytes
        )
    }
}

impl std::error::Error for BudgetTooSmall {}

/// What one expert costs in the GPU cache: the sum of its row across
/// every weight bank.
///
/// Fixed per-model tensors that do **not** scale with the number of
/// cached experts (folded quantization scales, for instance) are
/// deliberately excluded -- counting them here would make each slot
/// look more expensive than it is and under-size the cache.
pub fn expert_bytes_per_slot(bank_row_bytes: &[u64]) -> u64 {
    bank_row_bytes.iter().sum()
}

/// Split `budget_bytes` between the expert cache and the KV pool.
///
/// `max_slots` is a backend ceiling (some fused kernels cannot address
/// more than a fixed number of experts); `total_experts` is full
/// residency, past which more slots buy nothing.
#[allow(clippy::too_many_arguments)]
pub fn plan_cache_budget(
    budget_bytes: i64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
    num_experts: u64,
    total_experts: u64,
    prefill_overlap: bool,
    kv_reserve_pages: u64,
    max_slots: u64,
) -> Result<PoolSizes, BudgetTooSmall> {
    assert!(
        bytes_per_expert > 0 && bytes_per_page > 0,
        "an unpriced pool cannot be sized"
    );
    let hi = total_experts.min(max_slots);
    // The double buffer needs two layers of slots; without room for it
    // the split is planned without it rather than failing.
    let mut overlap = prefill_overlap && hi >= 2 * num_experts;
    let lo = if overlap {
        2 * num_experts
    } else {
        num_experts
    };
    assert!(
        hi >= lo,
        "the expert-cache ceiling of {hi} slots cannot hold the {lo} slots a layer needs"
    );

    // Fill the cache first, but never at the cost of the KV floor.
    let spare = budget_bytes - (kv_reserve_pages * bytes_per_page) as i64;
    let raw = if spare <= 0 {
        0
    } else {
        (spare as u64) / bytes_per_expert
    };
    let moe_cache_slots = raw.min(hi).max(lo);
    overlap = overlap && moe_cache_slots >= 2 * num_experts;

    let remaining = budget_bytes - (moe_cache_slots * bytes_per_expert) as i64;
    let kv_pages = if remaining <= 0 {
        kv_reserve_pages
    } else {
        ((remaining as u64) / bytes_per_page).max(kv_reserve_pages)
    };

    let sizes = PoolSizes {
        moe_cache_slots,
        kv_pages,
        prefill_overlap: overlap,
    };
    let needed = required_bytes(moe_cache_slots, kv_pages, bytes_per_expert, bytes_per_page);
    if needed > budget_bytes || kv_pages <= 1 {
        return Err(BudgetTooSmall {
            needed_bytes: needed,
            budget_bytes,
            sizes,
        });
    }
    Ok(sizes)
}

/// A live re-split request. `None` means "leave this pool alone".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildRequest {
    pub moe_cache_slots: Option<u64>,
    pub kv_pages: Option<u64>,
    pub mamba_slots: Option<u64>,
    pub swa_pages: Option<u64>,
}

impl RebuildRequest {
    pub fn is_empty(&self) -> bool {
        self.moe_cache_slots.is_none()
            && self.kv_pages.is_none()
            && self.mamba_slots.is_none()
            && self.swa_pages.is_none()
    }

    /// Whether this request invalidates the prefix cache.
    ///
    /// Any KV-side resize does: a page index, a recurrent slot id and a
    /// window slot id are all positions in an allocation that is about
    /// to stop existing, and a cached prefix pointing into the old one
    /// would hand back another request's state. A MoE-only resize does
    /// not -- the expert cache holds no per-request state.
    pub fn invalidates_prefix_cache(&self) -> bool {
        self.kv_pages.is_some() || self.mamba_slots.is_some() || self.swa_pages.is_some()
    }
}

/// Why a rebuild was refused. The engine is untouched in every case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildRejected {
    /// The server is not idle. A rebuild frees pools that in-flight
    /// requests hold indices into.
    Busy,
    /// The model has no such pool.
    NoSuchPool(&'static str),
    /// The target is below what the pool must hold to function.
    BelowFloor {
        pool: &'static str,
        requested: u64,
        floor: u64,
    },
    /// The split does not fit in the budget.
    TooLarge(BudgetTooSmall),
}

impl std::fmt::Display for RebuildRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebuildRejected::Busy => {
                write!(f, "the engine is not idle; retry when it drains")
            }
            RebuildRejected::NoSuchPool(pool) => {
                write!(f, "this model has no {pool} pool")
            }
            RebuildRejected::BelowFloor {
                pool,
                requested,
                floor,
            } => write!(
                f,
                "{pool}={requested} is below the working-set floor of {floor}"
            ),
            RebuildRejected::TooLarge(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for RebuildRejected {}

/// What the served model actually has, and what each pool must hold to
/// work at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolFloors {
    /// Minimum expert-cache slots: one whole layer.
    pub moe_slots: u64,
    /// Minimum KV pages.
    pub kv_pages: u64,
    /// `None` when the model has no recurrent-state pool.
    pub mamba_slots: Option<u64>,
    /// `None` when the model has no window pool.
    pub swa_pages: Option<u64>,
}

/// Check a rebuild against the floors and the budget, changing nothing.
///
/// `bytes_per_expert` and `bytes_per_page` price the two resizable
/// pools; `fixed_bytes` is everything the target geometry costs that
/// does not scale with them (the window pool at its target size, the
/// recurrent pool at its target size).
#[allow(clippy::too_many_arguments)]
pub fn validate_rebuild(
    request: &RebuildRequest,
    current: &PoolSizes,
    floors: &PoolFloors,
    idle: bool,
    budget_bytes: i64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
) -> Result<PoolSizes, RebuildRejected> {
    if !idle {
        return Err(RebuildRejected::Busy);
    }
    let below = |pool, requested: u64, floor: u64| RebuildRejected::BelowFloor {
        pool,
        requested,
        floor,
    };
    if let Some(slots) = request.moe_cache_slots {
        if slots < floors.moe_slots {
            return Err(below("moe", slots, floors.moe_slots));
        }
    }
    if let Some(pages) = request.kv_pages {
        if pages < floors.kv_pages.max(2) {
            return Err(below("kv", pages, floors.kv_pages.max(2)));
        }
    }
    if let Some(slots) = request.mamba_slots {
        match floors.mamba_slots {
            None => return Err(RebuildRejected::NoSuchPool("recurrent-state")),
            Some(floor) if slots < floor => return Err(below("mamba", slots, floor)),
            Some(_) => {}
        }
    }
    if let Some(pages) = request.swa_pages {
        match floors.swa_pages {
            None => return Err(RebuildRejected::NoSuchPool("window")),
            Some(floor) if pages < floor => return Err(below("swa", pages, floor)),
            Some(_) => {}
        }
    }

    let target = PoolSizes {
        moe_cache_slots: request.moe_cache_slots.unwrap_or(current.moe_cache_slots),
        kv_pages: request.kv_pages.unwrap_or(current.kv_pages),
        prefill_overlap: current.prefill_overlap,
    };
    let needed = required_bytes(
        target.moe_cache_slots,
        target.kv_pages,
        bytes_per_expert,
        bytes_per_page,
    );
    if needed > budget_bytes {
        return Err(RebuildRejected::TooLarge(BudgetTooSmall {
            needed_bytes: needed,
            budget_bytes,
            sizes: target,
        }));
    }
    Ok(target)
}

/// The window pool a single request needs to survive a decode.
///
/// Every term is a real thing the request holds at once: the window it
/// has locked against eviction (rounded up to whole pages), the window
/// it is currently reading, the tokens it accumulates between two slides
/// of the window, and two pages of slack for the partial pages at each
/// end. `anchor_checkpoints` adds a second retained window, because an
/// anchored request keeps the state around its anchor live as well as
/// the state around its cursor.
pub fn swa_tokens_per_request(
    sliding_window: usize,
    page_size: usize,
    eviction_interval: usize,
    anchor_checkpoints: bool,
) -> usize {
    let locked = align_ceil(sliding_window + SWA_RETAIN_GAP, page_size);
    let mut floor = locked + sliding_window + eviction_interval + 2 * page_size;
    if anchor_checkpoints {
        floor += sliding_window + SWA_RETAIN_GAP + eviction_interval;
    }
    floor
}

/// The window pool floor for `max_running_requests` concurrent
/// requests.
///
/// Below this the pool cannot hold every running request's window at
/// once, and the server deadlocks under concurrency rather than merely
/// running slower -- which is why this is a floor and not a target.
pub fn swa_pool_floor(
    max_running_requests: usize,
    sliding_window: usize,
    page_size: usize,
    eviction_interval: usize,
    anchor_checkpoints: bool,
) -> usize {
    max_running_requests
        * swa_tokens_per_request(
            sliding_window,
            page_size,
            eviction_interval,
            anchor_checkpoints,
        )
}

/// The window pool for a prefix-caching (radix) window model.
///
/// `override_tokens` is an explicit request; otherwise the pool is a
/// fraction of the full pool. Either way the floor wins, and one extra
/// slot is added for the reserved "no live window slot" sentinel that
/// slot 0 always is.
pub fn swa_paged_num_tokens(
    floor: usize,
    full_tokens: usize,
    full_tokens_ratio: f64,
    override_tokens: Option<usize>,
) -> usize {
    let target = match override_tokens {
        Some(tokens) => tokens,
        None => (full_tokens_ratio * full_tokens as f64) as usize,
    };
    floor.max(target) + 1
}

/// The window pool for a model serving *without* prefix reuse.
///
/// With no sharing there is nothing to keep between requests, so the
/// pool is exactly what the running requests need at once: one window
/// plus one forward's worth of new tokens each, rounded to a 32-token
/// granule, with one spare request's worth of headroom.
pub fn naive_swa_num_tokens(
    max_running_requests: usize,
    sliding_window: usize,
    max_forward_len: usize,
) -> usize {
    let width = align_ceil(sliding_window + max_forward_len + 1, 32);
    (max_running_requests + 1) * width
}

/// Slots in the recurrent-state pool.
///
/// Without prefix reuse a request needs one live slot, plus one padding
/// slot the pool reserves. With reuse each request also needs two
/// ping-pong slots (it snapshots into the idle one while computing from
/// the other) and one for the snapshot it has committed but not yet
/// handed over -- hence four per request -- plus a shared cache of
/// committed snapshots sized from `cache_ratio`.
pub fn linear_pool_slots(
    max_running_requests: usize,
    prefix_reuse: bool,
    cache_ratio: f64,
) -> usize {
    if !prefix_reuse {
        return max_running_requests + 1;
    }
    let cached = 4.max((cache_ratio * max_running_requests as f64) as usize);
    4 * max_running_requests + cached + 1
}

/// The floor for the same pool: the live and ping-pong slots, with no
/// cache at all.
pub fn linear_pool_min_slots(max_running_requests: usize, prefix_reuse: bool) -> usize {
    if !prefix_reuse {
        max_running_requests + 1
    } else {
        4 * max_running_requests + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn the_budget_is_free_vram_less_the_weights_and_the_headroom() {
        assert_eq!(
            net_cache_budget_bytes(0.9, 10 * GIB, 4 * GIB, 0),
            (0.9 * (10 * GIB) as f64) as i64 - (4 * GIB) as i64
        );
        // An over-committed deployment gets a negative budget, not a
        // wrapped-around enormous one.
        assert!(net_cache_budget_bytes(0.5, GIB, 4 * GIB, 0) < 0);
    }

    #[test]
    fn the_startup_budget_prices_what_loading_actually_consumed() {
        assert_eq!(startup_kv_budget(0.9, 1000, 400), 900 - 600);
        assert_eq!(startup_kv_budget(1.0, 1000, 1000), 1000);
    }

    /// The expert cache is filled first: it is the pool whose shortfall
    /// is paid per token rather than per request.
    #[test]
    fn the_expert_cache_is_filled_before_kv_gets_the_remainder() {
        let sizes = plan_cache_budget(
            (8 * GIB) as i64,
            16 * MIB, // per expert
            MIB,      // per page
            32,       // experts per layer
            256,      // total experts
            false,
            64,
            u64::MAX,
        )
        .expect("fits");
        assert_eq!(sizes.moe_cache_slots, 256, "full residency");
        let spent = 256 * 16 * MIB;
        assert_eq!(sizes.kv_pages, (8 * GIB - spent) / MIB);
    }

    /// ... but never past full residency: extra slots buy nothing, so
    /// the bytes go to KV.
    #[test]
    fn the_expert_cache_is_capped_at_full_residency() {
        let sizes = plan_cache_budget((64 * GIB) as i64, MIB, MIB, 8, 64, false, 16, u64::MAX)
            .expect("fits");
        assert_eq!(sizes.moe_cache_slots, 64);
        assert!(sizes.kv_pages > 60_000);
    }

    /// A backend ceiling rolls the freed bytes into KV rather than
    /// wasting them.
    #[test]
    fn a_backend_slot_ceiling_gives_its_bytes_to_kv() {
        let capped =
            plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 4096, false, 16, 992).expect("fits");
        assert_eq!(capped.moe_cache_slots, 992);
        let uncapped = plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 4096, false, 16, u64::MAX)
            .expect("fits");
        assert!(uncapped.moe_cache_slots > capped.moe_cache_slots);
        assert!(capped.kv_pages > uncapped.kv_pages);
    }

    /// The KV floor is not negotiable: a server that cannot hold a
    /// context serves nothing, however good its expert residency is.
    #[test]
    fn the_kv_reserve_is_taken_out_before_the_cache_is_sized() {
        let reserve = 1024;
        let sizes = plan_cache_budget(
            (2 * GIB) as i64,
            MIB,
            MIB,
            8,
            4096,
            false,
            reserve,
            u64::MAX,
        )
        .expect("fits");
        assert!(sizes.kv_pages >= reserve);
        assert!(sizes.moe_cache_slots <= 2048 - reserve);
    }

    /// The prefill double buffer costs two layers of slots. A budget
    /// that cannot pay for it loses the buffer rather than the split.
    #[test]
    fn a_tight_budget_drops_the_prefill_double_buffer() {
        let roomy =
            plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 64, true, 16, u64::MAX).unwrap();
        assert!(roomy.prefill_overlap);

        // A ceiling below two layers cannot host the buffer at all.
        let cramped = plan_cache_budget(
            (4 * GIB) as i64,
            MIB,
            MIB,
            8,
            64,
            true,
            16,
            12, // fewer than 2 * 8 slots
        )
        .unwrap();
        assert!(!cramped.prefill_overlap);
    }

    #[test]
    fn a_budget_that_cannot_hold_one_layer_is_refused() {
        let err = plan_cache_budget(MIB as i64, MIB, MIB, 8, 64, false, 16, u64::MAX).unwrap_err();
        assert!(err.needed_bytes > err.budget_bytes);
        assert_eq!(err.sizes.moe_cache_slots, 8, "one layer is the minimum");
    }

    #[test]
    fn expert_slot_cost_is_the_sum_over_banks() {
        assert_eq!(expert_bytes_per_slot(&[512, 256]), 768);
        assert_eq!(expert_bytes_per_slot(&[]), 0);
    }

    #[test]
    fn a_rebuild_is_refused_while_the_engine_is_busy() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            ..PoolFloors::default()
        };
        let request = RebuildRequest {
            moe_cache_slots: Some(128),
            ..RebuildRequest::default()
        };
        assert_eq!(
            validate_rebuild(&request, &current, &floors, false, i64::MAX, MIB, MIB),
            Err(RebuildRejected::Busy)
        );
    }

    #[test]
    fn a_rebuild_below_a_floor_or_over_the_budget_changes_nothing() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            mamba_slots: None,
            swa_pages: Some(32),
        };

        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    moe_cache_slots: Some(4),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                i64::MAX,
                MIB,
                MIB
            ),
            Err(RebuildRejected::BelowFloor { pool: "moe", .. })
        ));

        // A pool this model does not have is named, not silently
        // accepted.
        assert_eq!(
            validate_rebuild(
                &RebuildRequest {
                    mamba_slots: Some(64),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                i64::MAX,
                MIB,
                MIB
            ),
            Err(RebuildRejected::NoSuchPool("recurrent-state"))
        );

        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    kv_pages: Some(1 << 40),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                (2 * GIB) as i64,
                MIB,
                MIB
            ),
            Err(RebuildRejected::TooLarge(_))
        ));
    }

    #[test]
    fn a_rebuild_leaves_untouched_pools_at_their_current_size() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            ..PoolFloors::default()
        };
        let target = validate_rebuild(
            &RebuildRequest {
                moe_cache_slots: Some(128),
                ..RebuildRequest::default()
            },
            &current,
            &floors,
            true,
            (4 * GIB) as i64,
            MIB,
            MIB,
        )
        .expect("fits");
        assert_eq!(target.moe_cache_slots, 128);
        assert_eq!(target.kv_pages, 1024);
    }

    /// Only a KV-side resize invalidates cached prefixes; the expert
    /// cache holds no per-request state.
    #[test]
    fn only_a_kv_side_resize_invalidates_the_prefix_cache() {
        assert!(!RebuildRequest {
            moe_cache_slots: Some(128),
            ..RebuildRequest::default()
        }
        .invalidates_prefix_cache());
        for request in [
            RebuildRequest {
                kv_pages: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                mamba_slots: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                swa_pages: Some(1),
                ..RebuildRequest::default()
            },
        ] {
            assert!(request.invalidates_prefix_cache());
        }
        assert!(RebuildRequest::default().is_empty());
    }

    #[test]
    fn the_window_pool_floor_pays_for_every_window_a_request_holds() {
        let per_req = swa_tokens_per_request(512, 64, DEFAULT_SWA_EVICTION_INTERVAL, false);
        // locked(576 -> 9 pages = 576) + window(512) + interval(128) + 2 pages(128)
        assert_eq!(per_req, 576 + 512 + 128 + 128);
        assert_eq!(
            swa_pool_floor(4, 512, 64, DEFAULT_SWA_EVICTION_INTERVAL, false),
            4 * per_req
        );

        // Anchored requests keep a second window live.
        let anchored = swa_tokens_per_request(512, 64, DEFAULT_SWA_EVICTION_INTERVAL, true);
        assert_eq!(anchored, per_req + 512 + SWA_RETAIN_GAP + 128);
    }

    #[test]
    fn the_window_pool_never_goes_below_its_floor() {
        let floor = 4096;
        // The ratio would ask for far less than the floor.
        assert_eq!(swa_paged_num_tokens(floor, 1024, 0.5, None), floor + 1);
        // A generous full pool wins.
        assert_eq!(swa_paged_num_tokens(floor, 65536, 0.5, None), 32768 + 1);
        // An explicit request is still floored.
        assert_eq!(swa_paged_num_tokens(floor, 65536, 0.5, Some(64)), floor + 1);
        assert_eq!(
            swa_paged_num_tokens(floor, 65536, 0.5, Some(100_000)),
            100_001
        );
    }

    #[test]
    fn a_naive_window_pool_holds_exactly_the_running_requests() {
        // (512 + 256 + 1) rounded up to a 32-token granule is 800.
        assert_eq!(naive_swa_num_tokens(4, 512, 256), 5 * 800);
    }

    #[test]
    fn recurrent_slots_pay_for_ping_pong_only_under_prefix_reuse() {
        assert_eq!(linear_pool_slots(8, false, 0.5), 9);
        assert_eq!(linear_pool_min_slots(8, false), 9);
        // 4 per request + max(4, 0.5*8) cached + 1 padding
        assert_eq!(linear_pool_slots(8, true, 0.5), 32 + 4 + 1);
        assert_eq!(linear_pool_min_slots(8, true), 33);
        // The cache never drops below four entries.
        assert_eq!(linear_pool_slots(2, true, 0.1), 8 + 4 + 1);
    }
}
