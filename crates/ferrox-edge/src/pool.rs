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

use crate::cache_report::{CacheGeometry, Limit, Limits, UnitBytes};
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
/// does *not* scale with them -- the window pool at its target size, the
/// recurrent pool at its target size.
///
/// `fixed_bytes` comes off the budget rather than onto the requirement.
/// That is the same arithmetic either way, but it matches the reference
/// (`net_cache_budget_bytes(.., fixed_cache_size + extra_fixed_bytes)`)
/// and it makes the `budget_bytes` carried in a rejection mean the one
/// useful thing: what was actually left for the two resizable pools.
///
/// Passing `0` for a **window model** prices its window pool at nothing.
/// The fit-check then waves through a full pool that leaves no room for
/// the window, and the rebuild dies of an allocation failure *after* it
/// has freed the old cache -- the one failure this whole path exists to
/// prevent. Price a window model with [`hybrid_swa_kv_cost`] at the
/// rebuild's *target* window and pass both of its terms:
/// [`KvCost::cache_per_page`] as `bytes_per_page`,
/// [`KvCost::fixed_cache_size`] as `fixed_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn validate_rebuild(
    request: &RebuildRequest,
    current: &PoolSizes,
    floors: &PoolFloors,
    idle: bool,
    budget_bytes: i64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
    fixed_bytes: u64,
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
    // Signed, and saturating: an over-committed deployment gets a
    // negative budget that every requirement exceeds, never a
    // wrapped-around enormous one that every requirement fits inside.
    let budget_bytes = budget_bytes.saturating_sub(fixed_bytes as i64);
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

// ---------------------------------------------------------------------
// The windowed KV cost model
// ---------------------------------------------------------------------

/// Bytes of one dense full-pool-to-window index entry: an `int64` slot
/// id per full-pool token. Held for the **full** pool, not the window,
/// because the mapping must answer "where is this position's window
/// state" for every position the full pool can address.
pub const FULL_TO_WINDOW_INDEX_BYTES: u64 = 8;

/// Bytes per element of the DSA index-key slab (`bf16`). The pool
/// allocates the slab at this width; the two must move together, or the
/// cost model prices a slab of a size nobody allocates.
pub const DSA_INDEX_KEY_BYTES: u64 = 2;

/// One paged-KV group's shape: what the pool must store for one token of
/// this group, before any pool family decides how many tokens it holds.
///
/// A *group* is a set of layers sharing a KV layout. A window model has
/// two -- the full-attention group and the sliding-window group -- and a
/// plain model has one. Keeping the cost per group is exactly what lets
/// [`hybrid_swa_kv_cost`] price the window pool differently from the full
/// pool without ever branching on the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvGroupSpec {
    /// Layers in this group. A group with no layers costs nothing.
    pub num_layers: u64,
    /// KV heads *before* the tensor-parallel split; see
    /// [`spec_kv_bytes_per_token`].
    pub num_kv_heads: u64,
    pub head_dim: u64,
    /// Latent-KV (MLA) group: the pool stores **one** slab per token
    /// because V aliases K. Budgeting the usual two for an MLA model
    /// prices every token at double and halves the pool the same VRAM
    /// could have bought.
    pub mla: bool,
    /// DSA index-key slab: one `index_head_dim`-wide row per token per
    /// indexer layer. Zero when the group carries no indexer.
    pub index_head_dim: u64,
    pub num_index_layers: u64,
    /// Whether this is the sliding-window group -- the one whose bytes
    /// live in the separate window pool rather than the full pool.
    pub is_swa: bool,
    /// The window, in tokens. Read only when `is_swa`.
    pub sliding_window: usize,
}

/// One group's KV bytes for one token.
///
/// `(1 or 2 slabs) x head_dim x local KV heads x dtype x layers`, plus
/// the index-key slab when the group carries indexer dims. Pure
/// per-group arithmetic: a pool family composes this over *its own*
/// groups and no family branching happens here.
///
/// `tp_size` splits the KV heads across ranks, and a rank holds only its
/// share. When there are more ranks than heads the heads are
/// *replicated* and each rank holds one -- which is why this is not a
/// plain division: rounding `8 heads / 16 ranks` down to zero would
/// price a rank's KV at nothing and size the pool off a free cache.
///
/// # Panics
///
/// When the heads do not divide evenly across the ranks in either
/// direction. A ragged split is a configuration bug, and silently
/// rounding it either way misprices every pool derived from here.
pub fn spec_kv_bytes_per_token(spec: &KvGroupSpec, dtype_bytes: u64, tp_size: u64) -> u64 {
    let tp_size = tp_size.max(1);
    let local_kv_heads = if tp_size > spec.num_kv_heads {
        assert!(
            spec.num_kv_heads > 0 && tp_size.is_multiple_of(spec.num_kv_heads),
            "{tp_size} ranks cannot replicate {} KV heads evenly",
            spec.num_kv_heads
        );
        1
    } else {
        assert!(
            spec.num_kv_heads.is_multiple_of(tp_size),
            "{} KV heads do not divide across {tp_size} ranks",
            spec.num_kv_heads
        );
        spec.num_kv_heads / tp_size
    };
    let slabs = if spec.mla { 1 } else { 2 };
    let per_token = slabs * spec.head_dim * local_kv_heads * dtype_bytes * spec.num_layers;
    per_token + spec.index_head_dim * spec.num_index_layers * DSA_INDEX_KEY_BYTES
}

/// How a window model sizes its window pool -- which is the whole of
/// what decides whether the window's bytes are a *per-page* cost or a
/// *fixed* one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwaSizing {
    /// Serving without prefix reuse: the pool is concurrency x window
    /// ([`naive_swa_num_tokens`]) and never moves when the full pool
    /// does, so all of it is fixed.
    Naive {
        /// The longest forward the scheduler will run, in tokens.
        max_forward_len: usize,
    },
    /// Prefix reuse with a pinned absolute window -- a startup override,
    /// or the target window a rebuild is being validated against, priced
    /// before it is written anywhere. Also fixed: a pin does not move
    /// when the full pool does.
    Pinned { tokens: usize },
    /// Prefix reuse with no pin: the window is `full_tokens_ratio` x the
    /// full pool, so it grows with every page the full pool gains, and
    /// only the concurrency floor underneath it is fixed.
    Ratio { full_tokens_ratio: f64 },
}

/// The affine price of a KV geometry:
/// `bytes = cache_per_page * pages + fixed_cache_size`.
///
/// Two terms and not one because a page solve divides by the first and
/// subtracts the second, and putting a term in the wrong one is not a
/// rounding error -- it changes how the pool responds to being given
/// more VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvCost {
    /// What one more full-pool page costs, window growth included.
    pub cache_per_page: u64,
    /// What the geometry costs before the first page: the parts that do
    /// not scale with the full pool.
    pub fixed_cache_size: u64,
}

/// Collapse a two-pool window model into the affine `(per page, fixed)`
/// pair a page solve and [`validate_rebuild`] both work in.
///
/// The branching *is* the content:
///
/// * A non-window group rides `cache_per_page` at
///   `per_token * page_size` -- it lives in the full pool and grows with
///   it.
/// * A window group **always** adds [`FULL_TO_WINDOW_INDEX_BYTES`] x
///   `page_size` to `cache_per_page`, whatever the sizing. That is the
///   dense full-to-window index mapping, and it is keyed by full-pool
///   position, so it scales with the **full** pool and not with the
///   window. Charging it to the window instead makes a pinned window
///   look free to grow the full pool against.
/// * Then the window pool itself: [`SwaSizing::Naive`] and
///   [`SwaSizing::Pinned`] are entirely fixed, and [`SwaSizing::Ratio`]
///   **splits** -- `per_token * page_size * ratio` into `cache_per_page`
///   because the window grows with the full pool, and
///   `per_token * swa_pool_floor` into `fixed_cache_size` because the
///   concurrency floor is already there at one page.
///
/// Both halves of that split are load-bearing. Fold the ratio term into
/// `fixed_cache_size` and the per-page price stops carrying the window
/// each new page drags along, so the page solve sizes the split off a
/// price that is not the price. Drop the term altogether and the full
/// pool grows against a window nobody budgeted for, until the first long
/// prompt dies allocating window state.
pub fn hybrid_swa_kv_cost(groups: &[KvGroupSpec], config: &SwaCostConfig) -> KvCost {
    let page_size = config.page_size as u64;
    let mut cost = KvCost::default();
    for spec in groups {
        let per_token = spec_kv_bytes_per_token(spec, config.dtype_bytes, config.tp_size);
        if !spec.is_swa {
            cost.cache_per_page += per_token * page_size;
            continue;
        }
        // The full -> window index mapping, one entry per full-pool
        // token. Full-pool sized on purpose; see above.
        cost.cache_per_page += FULL_TO_WINDOW_INDEX_BYTES * page_size;

        let floor = swa_pool_floor(
            config.max_running_requests,
            spec.sliding_window,
            config.page_size,
            config.eviction_interval,
            config.anchor_checkpoints,
        ) as u64;
        match config.sizing {
            SwaSizing::Naive { max_forward_len } => {
                let tokens = naive_swa_num_tokens(
                    config.max_running_requests,
                    spec.sliding_window,
                    max_forward_len,
                ) as u64;
                cost.fixed_cache_size += per_token * tokens;
            }
            SwaSizing::Pinned { tokens } => {
                // Exactly what `swa_paged_num_tokens` would allocate for
                // this pin: the floor still wins, and the slot-0
                // sentinel is still paid for.
                cost.fixed_cache_size += per_token * (floor.max(tokens as u64) + 1);
            }
            SwaSizing::Ratio { full_tokens_ratio } => {
                cost.cache_per_page += ((per_token * page_size) as f64 * full_tokens_ratio) as u64;
                cost.fixed_cache_size += per_token * floor;
            }
        }
    }
    cost
}

/// Everything outside the group specs that [`hybrid_swa_kv_cost`] prices
/// against: the page granularity, the concurrency the window floor is
/// derived from, the element width, and how the window is sized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwaCostConfig {
    pub page_size: usize,
    pub max_running_requests: usize,
    /// How often the window is slid; see
    /// [`DEFAULT_SWA_EVICTION_INTERVAL`].
    pub eviction_interval: usize,
    /// Whether requests keep a second retained window around an anchor.
    pub anchor_checkpoints: bool,
    /// Bytes per KV element (2 for `bf16`/`fp16`).
    pub dtype_bytes: u64,
    /// Tensor-parallel ranks the KV heads are split across.
    pub tp_size: u64,
    pub sizing: SwaSizing,
}

impl Default for SwaCostConfig {
    fn default() -> Self {
        SwaCostConfig {
            page_size: 64,
            max_running_requests: 1,
            eviction_interval: DEFAULT_SWA_EVICTION_INTERVAL,
            anchor_checkpoints: false,
            dtype_bytes: 2,
            tp_size: 1,
            sizing: SwaSizing::Ratio {
                full_tokens_ratio: 1.0,
            },
        }
    }
}

// ---------------------------------------------------------------------
// The readiness document
// ---------------------------------------------------------------------

/// Per-pool floors **in the unit a client types**, which is not the unit
/// the pool is allocated in.
///
/// [`PoolFloors`] is the other one: it is what [`validate_rebuild`]
/// checks a request against, in allocation units (expert slots, KV
/// *pages*, window *pages*). This one is what a client is *shown* -- KV
/// and window in tokens, the recurrent pool in usable slots -- so that
/// the number a user reads off the range is the number they may type
/// back. Mixing the two hands a client a page count labelled as tokens
/// and it clamps its slider to 1/64th of the real floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheFloors {
    /// One page's worth of tokens: a rebuild rejects a zero-page pool,
    /// so a single page is the smallest thing that exists.
    pub kv_tokens: u64,
    /// One MoE layer's routed experts. Below that some layer's step
    /// cannot be served from cache at all.
    pub moe_experts: u64,
    /// *Usable* recurrent slots -- the physical floor minus the reserved
    /// padding sink.
    pub mamba_slots: u64,
    /// Window-pool tokens, sentinel included.
    pub swa_tokens: u64,
}

/// The window pool of a model that serves it with prefix reuse, as it
/// stands right now.
///
/// `None` on [`CacheReadiness`] covers both "no window at all" and "a
/// window pool sized purely by concurrency". Neither has anything a
/// client may resize: a naive window pool is a function of
/// `max_running_requests` and the window, so advertising a range for it
/// would offer a control that changes nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowReadiness {
    pub sliding_window: usize,
    /// See [`DEFAULT_SWA_EVICTION_INTERVAL`].
    pub eviction_interval: usize,
    pub anchor_checkpoints: bool,
    /// The window pool as allocated, in **physical** tokens -- the
    /// slot-0 sentinel included, exactly as
    /// [`swa_paged_num_tokens`] returned it.
    pub swa_num_tokens: u64,
    /// The current window/full reuse ratio, the one tunable knob of a
    /// ratio-sized window.
    pub full_tokens_ratio: f64,
}

/// What a loaded engine knows about its own caches, in the moment
/// between "the pools are allocated" and "the server is ready".
///
/// This is the *input* side of the readiness document. Everything here
/// has already been measured or read off a config by the caller --
/// ferrox-edge holds no tensors and no device handles, so it cannot
/// measure a pool itself. [`CacheReadiness::cache_geometry`] turns it
/// into the [`CacheGeometry`] every client renders.
///
/// # Why nothing here can fail
///
/// The reference builds this on the readiness path with every read
/// wrapped so that a failure degrades that one field to `0` and never
/// blocks startup: a partial answer beats no answer when the alternative
/// is a server that will not come up. The Rust port keeps the *rule* and
/// moves the fallibility to the caller -- an absent pool and an
/// unmeasurable one arrive here as the same `None`/`0`, every getter
/// below returns a number rather than a `Result`, and every subtraction
/// saturates. A `0` therefore always means "nothing to report", never
/// "genuinely zero bytes", which is the same convention
/// [`crate::cache_report`] renders under.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheReadiness {
    /// Per-unit VRAM costs, measured off the live pools by the caller.
    pub unit_bytes: UnitBytes,
    /// Free VRAM captured after the weights loaded and **before** any
    /// pool was allocated. Stable for the process lifetime, and
    /// unaffected by allocator caching, captured graphs, or later
    /// rebuilds -- which is what makes it usable as an upper bound.
    pub post_weights_free_bytes: u64,
    /// Free VRAM captured before the weights loaded.
    pub baseline_free_bytes: u64,
    pub weights_bytes: u64,
    /// The fraction of the baseline the engine will spend on caches; see
    /// [`net_cache_budget_bytes`].
    pub memory_ratio: f64,
    /// Full-KV pool as allocated.
    pub num_pages: u64,
    pub page_size: u64,
    /// Routed experts per MoE layer, and the number of MoE layers.
    pub num_experts: u64,
    pub num_moe_layers: u64,
    /// Expert-cache slots allocated. `None` when the model has no
    /// offload cache -- which is not the same as a cache of size zero,
    /// and is why this is an `Option`: only the first has no floor.
    pub moe_cache_size: Option<u64>,
    /// The expert cache's eviction policy, so a client can label the
    /// pool without knowing how the server was started.
    pub moe_cache_policy: Option<String>,
    /// The recurrent-state pool as allocated, in **physical** slots --
    /// the reserved padding sink included. `None` when the model has no
    /// such pool.
    pub linear_state_slots: Option<u64>,
    pub max_running_requests: usize,
    /// Whether the recurrent pool serves prefix reuse, which is what
    /// buys it its ping-pong slots.
    pub prefix_reuse: bool,
    /// The resizable window pool, if this model has one.
    pub window: Option<WindowReadiness>,
}

impl CacheReadiness {
    /// The pool-budget baseline: free VRAM after the weights, before any
    /// pool. `0` when it was never captured.
    pub fn free_vram_bytes(&self) -> u64 {
        self.post_weights_free_bytes
    }

    /// Whether the window/full reuse ratio is a knob on this model at
    /// all. It is only meaningful where a *separate* window pool is
    /// sized from it, so a dense model, and a window model served
    /// without prefix reuse, both report `false` -- and a client that
    /// offers the ratio anyway is offering a control whose only effect
    /// is an error from the engine.
    pub fn supports_swa_ratio(&self) -> bool {
        self.window.is_some()
    }

    /// The ratio itself, or `0.0` where it means nothing.
    ///
    /// Offered as a getter rather than a [`CacheGeometry`] field because
    /// the geometry has nowhere to carry it; a client that shows the
    /// knob reads it from here.
    pub fn swa_full_tokens_ratio(&self) -> f64 {
        match &self.window {
            Some(window) => window.full_tokens_ratio,
            None => 0.0,
        }
    }

    /// The whole-cache VRAM ceiling the engine honours: `memory_ratio`
    /// of the pre-weights baseline, minus the weights, with **no** fixed
    /// cost taken out.
    ///
    /// Fixed-cost-free on purpose: this is the ceiling over *every* pool
    /// (KV + MoE + recurrent + window), not the remainder left for the
    /// two resizable ones, so a client can show the real ceiling instead
    /// of reverse-deriving it from the per-pool bounds. A rebuild's own
    /// fit-check is stricter -- it also pays `fixed_bytes` -- so a pool
    /// sized right at this ceiling on a model that carries a window or a
    /// recurrent pool can still be rejected.
    ///
    /// `0` when the baseline was never captured, and clamped at `0` for
    /// an over-committed deployment: a negative ceiling would render as
    /// a wrapped-around enormous one through the unsigned geometry.
    pub fn cache_budget_bytes(&self) -> u64 {
        if self.baseline_free_bytes == 0 {
            return 0;
        }
        net_cache_budget_bytes(
            self.memory_ratio,
            self.baseline_free_bytes,
            self.weights_bytes,
            0,
        )
        .max(0) as u64
    }

    /// The floors a rebuild enforces, in the units a client types.
    ///
    /// Every one is derived live from the pool sizing in this module --
    /// never a baked-in constant, because a constant drifts from the
    /// formula that actually rejects the request and the user is then
    /// told a bound that is not the bound.
    ///
    /// Two of them are **usable** counts and not physical ones:
    ///
    /// * `mamba_slots` is [`linear_pool_min_slots`] minus the reserved
    ///   padding sink, because the sink is the pool's own and a client
    ///   never names it. Reporting the physical count advertises a floor
    ///   one slot above the truth and makes the reported pool size
    ///   disagree with the scheduler's own totals by one.
    /// * `swa_tokens` is [`swa_pool_floor`] **plus** the slot-0
    ///   sentinel, which is the smallest pool
    ///   [`swa_paged_num_tokens`] can return. Reporting the bare floor
    ///   advertises a bound one token under what the pool actually
    ///   allocates.
    ///
    /// A pool the model does not have reports `0`, which every consumer
    /// reads as "no such pool" rather than "a floor of nothing".
    pub fn cache_floors(&self) -> CacheFloors {
        CacheFloors {
            kv_tokens: self.page_size,
            moe_experts: match self.moe_cache_size {
                Some(_) => self.num_experts,
                None => 0,
            },
            mamba_slots: match self.linear_state_slots {
                Some(_) => linear_pool_min_slots(self.max_running_requests, self.prefix_reuse)
                    .saturating_sub(1) as u64,
                None => 0,
            },
            swa_tokens: match &self.window {
                Some(window) => {
                    swa_pool_floor(
                        self.max_running_requests,
                        window.sliding_window,
                        self.page_size as usize,
                        window.eviction_interval,
                        window.anchor_checkpoints,
                    ) as u64
                        + 1
                }
                None => 0,
            },
        }
    }

    /// What a rebuild will accept for each pool: the floor as `min`, and
    /// as `max` the count that pool would reach if it were handed the
    /// whole budget.
    ///
    /// The budget is [`CacheReadiness::cache_budget_bytes`] -- the
    /// ceiling the fit-check actually enforces -- and **not** the raw
    /// post-weights free VRAM, which is larger by the
    /// `(1 - memory_ratio)` graph and activation headroom and would
    /// offer a client room that every rebuild then rejects. Free VRAM is
    /// only the fallback for an engine that reported no budget.
    ///
    /// Because the baseline was captured before any pool existed, each
    /// bound is a plain `budget / unit cost` with no occupancy
    /// correction, and stays put across rebuilds.
    ///
    /// The expert cache is the one pool with a ceiling of its own:
    /// caching more slots than the model *has* routed experts buys
    /// nothing, and on a small MoE model with a large card the budget
    /// alone overshoots the entire model. KV, window and recurrent
    /// capacity all keep paying off, so none of them is capped here.
    ///
    /// A pool whose unit cost or budget is unknown reports `max: 0`,
    /// which [`crate::cache_report::format_range`] renders as an empty
    /// cell -- a client keeps its own bounds rather than clamping to a
    /// fabricated `0..0`.
    pub fn cache_limits(&self) -> Limits {
        let budget = match self.cache_budget_bytes() {
            0 => self.free_vram_bytes(),
            budget => budget,
        };
        let ideal = |unit_cost: u64| -> u64 {
            if unit_cost == 0 || budget == 0 {
                0
            } else {
                budget / unit_cost
            }
        };
        let floors = self.cache_floors();
        let total_experts = self.num_experts * self.num_moe_layers;
        let mut moe_max = ideal(self.unit_bytes.moe_per_expert);
        if total_experts > 0 {
            moe_max = moe_max.min(total_experts);
        }
        Limits {
            moe_experts: Some(Limit {
                min: floors.moe_experts,
                max: moe_max,
            }),
            kv_tokens: Some(Limit {
                min: floors.kv_tokens,
                max: ideal(self.unit_bytes.kv_per_token),
            }),
            mamba_slots: Some(Limit {
                min: floors.mamba_slots,
                max: ideal(self.unit_bytes.mamba_per_slot),
            }),
            swa_tokens: Some(Limit {
                min: floors.swa_tokens,
                max: ideal(self.unit_bytes.swa_per_token),
            }),
        }
    }

    /// The readiness document: the pools as actually allocated, what
    /// each unit costs, and what a rebuild would accept.
    ///
    /// Without this a client only learns the pool sizes from the first
    /// generation's telemetry -- i.e. not until someone has chatted --
    /// so the panel a user sees the moment the server comes up is blank
    /// or, worse, guessed.
    ///
    /// The two pool counts that reserve a slot are reported **usable**,
    /// matching the number a client types back on a rebuild and the
    /// totals the scheduler reports: the recurrent pool as
    /// `physical - 1` (its padding sink is the pool's own) and the
    /// window pool as `physical - 1` (its slot-0 sentinel likewise).
    /// Reporting the physical counts puts every occupancy line one over
    /// its total.
    ///
    /// `swa_page_size` is `1` for a window pool, because a radix window
    /// is token-granular -- it is also the signal
    /// [`crate::cache_report::CachePools::from_geometry`] reads to
    /// decide the model *has* a window pool at all.
    pub fn cache_geometry(&self) -> CacheGeometry {
        CacheGeometry {
            num_experts: self.num_experts,
            num_moe_layers: self.num_moe_layers,
            moe_cache_size: self.moe_cache_size.unwrap_or(0),
            moe_cache_policy: self.moe_cache_policy.clone(),
            num_pages: self.num_pages,
            page_size: self.page_size,
            num_mamba_slots: self
                .linear_state_slots
                .map_or(0, |slots| slots.saturating_sub(1)),
            num_swa_pages: self
                .window
                .as_ref()
                .map_or(0, |window| window.swa_num_tokens.saturating_sub(1)),
            swa_page_size: u64::from(self.window.is_some()),
            cache_budget_bytes: self.cache_budget_bytes(),
            unit_bytes: self.unit_bytes.clone(),
            limits: self.cache_limits(),
        }
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
            validate_rebuild(&request, &current, &floors, false, i64::MAX, MIB, MIB, 0),
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
                MIB,
                0
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
                MIB,
                0
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
                MIB,
                0
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
            0,
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

    // ---- the windowed KV cost model ----

    /// 2 slabs x 64 head_dim x 8 KV heads x 2 bytes x 4 layers.
    const PER_TOKEN: u64 = 2 * 64 * 8 * 2 * 4;
    const PAGE: u64 = 64;
    const RUNNING: usize = 4;

    fn full_group() -> KvGroupSpec {
        KvGroupSpec {
            num_layers: 4,
            num_kv_heads: 8,
            head_dim: 64,
            ..KvGroupSpec::default()
        }
    }

    fn window_group() -> KvGroupSpec {
        KvGroupSpec {
            is_swa: true,
            sliding_window: 512,
            ..full_group()
        }
    }

    fn swa_config(sizing: SwaSizing) -> SwaCostConfig {
        SwaCostConfig {
            page_size: PAGE as usize,
            max_running_requests: RUNNING,
            sizing,
            ..SwaCostConfig::default()
        }
    }

    fn window_floor() -> u64 {
        swa_pool_floor(
            RUNNING,
            512,
            PAGE as usize,
            DEFAULT_SWA_EVICTION_INTERVAL,
            false,
        ) as u64
    }

    #[test]
    fn a_groups_per_token_cost_counts_slabs_heads_layers_and_the_index_slab() {
        assert_eq!(spec_kv_bytes_per_token(&full_group(), 2, 1), PER_TOKEN);

        // An MLA group stores one latent slab, not two: V aliases K.
        // Budgeting two halves the pool the same VRAM could buy.
        let mla = KvGroupSpec {
            mla: true,
            ..full_group()
        };
        assert_eq!(spec_kv_bytes_per_token(&mla, 2, 1), PER_TOKEN / 2);

        // The DSA index-key slab is bf16 and counted over its own layer
        // count, not the group's.
        let dsa = KvGroupSpec {
            index_head_dim: 128,
            num_index_layers: 2,
            ..full_group()
        };
        assert_eq!(spec_kv_bytes_per_token(&dsa, 2, 1), PER_TOKEN + 128 * 2 * 2);

        // A rank holds its share of the heads ...
        assert_eq!(spec_kv_bytes_per_token(&full_group(), 2, 2), PER_TOKEN / 2);
        // ... but when the ranks outnumber the heads the heads are
        // replicated and each rank still holds one whole head. Dividing
        // here would price a rank's KV at nothing.
        let single = KvGroupSpec {
            num_kv_heads: 1,
            ..full_group()
        };
        assert_eq!(spec_kv_bytes_per_token(&single, 2, 4), PER_TOKEN / 8);
    }

    /// A ratio-sized window grows with the full pool, so its per-token
    /// cost belongs in `cache_per_page` and only the concurrency floor
    /// underneath it is fixed. Folding the ratio term into
    /// `fixed_cache_size` -- the naive collapse of a two-pool model into
    /// one pair -- prices page growth as free, and every assertion below
    /// fails.
    #[test]
    fn a_ratio_window_rides_the_page_cost_and_only_its_floor_is_fixed() {
        let cost = hybrid_swa_kv_cost(
            &[full_group(), window_group()],
            &swa_config(SwaSizing::Ratio {
                full_tokens_ratio: 0.5,
            }),
        );
        let ratio_term = PER_TOKEN * PAGE / 2;
        assert_eq!(
            cost.cache_per_page,
            PER_TOKEN * PAGE + FULL_TO_WINDOW_INDEX_BYTES * PAGE + ratio_term
        );
        assert_eq!(cost.fixed_cache_size, PER_TOKEN * window_floor());
        assert_ne!(
            cost.fixed_cache_size,
            PER_TOKEN * window_floor() + ratio_term,
            "the ratio term is a per-page cost, not a fixed one"
        );
    }

    /// A naive pool and a pinned window are both fixed: neither moves
    /// when the full pool gains a page, so charging either per page
    /// would make every extra page look like it drags a window along.
    #[test]
    fn a_naive_or_pinned_window_is_a_fixed_cost() {
        let groups = [full_group(), window_group()];
        let per_page = PER_TOKEN * PAGE + FULL_TO_WINDOW_INDEX_BYTES * PAGE;

        let naive = hybrid_swa_kv_cost(
            &groups,
            &swa_config(SwaSizing::Naive {
                max_forward_len: 256,
            }),
        );
        assert_eq!(naive.cache_per_page, per_page);
        assert_eq!(
            naive.fixed_cache_size,
            PER_TOKEN * naive_swa_num_tokens(RUNNING, 512, 256) as u64
        );

        let pinned =
            hybrid_swa_kv_cost(&groups, &swa_config(SwaSizing::Pinned { tokens: 100_000 }));
        assert_eq!(pinned.cache_per_page, per_page);
        // Exactly what the pool would allocate for that pin: the
        // sentinel is paid for here too.
        assert_eq!(pinned.fixed_cache_size, PER_TOKEN * 100_001);

        // A pin under the concurrency floor still allocates the floor,
        // so pricing it at the pin would under-budget the pool.
        let low = hybrid_swa_kv_cost(&groups, &swa_config(SwaSizing::Pinned { tokens: 8 }));
        assert_eq!(low.fixed_cache_size, PER_TOKEN * (window_floor() + 1));
    }

    /// The dense full-to-window index mapping is keyed by full-pool
    /// position, so it costs 8 bytes per full-pool token under every
    /// sizing mode. Charging it to the window instead (into
    /// `fixed_cache_size`, or scaled by the window) would make a pinned
    /// window look free to grow the full pool against -- this test fails
    /// under both.
    #[test]
    fn the_full_to_window_index_mapping_is_priced_against_the_full_pool() {
        let sizing = SwaSizing::Ratio {
            full_tokens_ratio: 0.0,
        };
        let dense = hybrid_swa_kv_cost(&[full_group()], &swa_config(sizing));
        assert_eq!(dense.cache_per_page, PER_TOKEN * PAGE);
        assert_eq!(dense.fixed_cache_size, 0, "a model with no window pool");

        for sizing in [
            SwaSizing::Naive {
                max_forward_len: 256,
            },
            SwaSizing::Pinned { tokens: 100_000 },
            sizing,
        ] {
            let cost = hybrid_swa_kv_cost(&[full_group(), window_group()], &swa_config(sizing));
            assert_eq!(
                cost.cache_per_page - dense.cache_per_page,
                FULL_TO_WINDOW_INDEX_BYTES * PAGE,
                "the mapping scales with the full pool, whatever the window does"
            );
        }
    }

    /// Pricing a window model's rebuild with the window pool at zero
    /// accepts a full pool that leaves no room for the window, and the
    /// rebuild then dies allocating it -- after it has already freed the
    /// old cache. A `validate_rebuild` that ignores `fixed_bytes` (the
    /// signature this module shipped with) accepts the second call here.
    #[test]
    fn a_rebuild_pays_for_the_window_pool_it_is_not_resizing() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: false,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            ..PoolFloors::default()
        };
        let request = RebuildRequest {
            kv_pages: Some(1024),
            ..RebuildRequest::default()
        };
        let exact = required_bytes(64, 1024, MIB, MIB);

        // With the window priced at nothing the split fits exactly.
        assert!(validate_rebuild(&request, &current, &floors, true, exact, MIB, MIB, 0).is_ok());

        // One byte of window pool is one byte too many.
        let rejected =
            validate_rebuild(&request, &current, &floors, true, exact, MIB, MIB, 1).unwrap_err();
        match rejected {
            RebuildRejected::TooLarge(inner) => {
                assert_eq!(inner.needed_bytes, exact);
                assert_eq!(
                    inner.budget_bytes,
                    exact - 1,
                    "the reported budget is what was left after the fixed pools"
                );
            }
            other => panic!("expected a budget rejection, got {other}"),
        }

        // And the two terms the cost model produces are what a caller
        // hands over: per-page price and fixed price, from one call.
        let cost = hybrid_swa_kv_cost(
            &[full_group(), window_group()],
            &swa_config(SwaSizing::Pinned { tokens: 100_000 }),
        );
        assert!(validate_rebuild(
            &request,
            &current,
            &floors,
            true,
            i64::MAX,
            MIB,
            cost.cache_per_page,
            cost.fixed_cache_size,
        )
        .is_ok());
    }

    // ---- the readiness document ----

    fn readiness() -> CacheReadiness {
        CacheReadiness {
            unit_bytes: UnitBytes {
                moe_per_expert: MIB,
                mamba_per_slot: 1 << 16,
                kv_per_token: 1024,
                swa_per_token: 512,
            },
            post_weights_free_bytes: 8 * GIB,
            baseline_free_bytes: 12 * GIB,
            weights_bytes: 4 * GIB,
            memory_ratio: 0.9,
            num_pages: 512,
            page_size: 64,
            num_experts: 32,
            num_moe_layers: 8,
            moe_cache_size: Some(128),
            moe_cache_policy: Some("lru".to_string()),
            linear_state_slots: Some(33),
            max_running_requests: 8,
            prefix_reuse: true,
            window: Some(WindowReadiness {
                sliding_window: 512,
                eviction_interval: DEFAULT_SWA_EVICTION_INTERVAL,
                anchor_checkpoints: false,
                swa_num_tokens: 20_000,
                full_tokens_ratio: 0.5,
            }),
        }
    }

    /// Both sentinel-bearing pools report what a client may *type*, not
    /// what was allocated: the recurrent pool hides its padding sink,
    /// and the window pool hides its slot-0 sentinel. Passing the
    /// physical counts through -- the obvious port -- fails every
    /// assertion here and puts the scheduler's occupancy line one over
    /// its own total.
    #[test]
    fn the_readiness_document_reports_usable_pool_counts_not_physical_ones() {
        let ready = readiness();
        let geometry = ready.cache_geometry();
        assert_eq!(
            geometry.num_mamba_slots, 32,
            "33 physical slots, one of them the padding sink"
        );
        assert_eq!(
            geometry.num_swa_pages, 19_999,
            "20000 physical window tokens, one of them the sentinel"
        );
        assert_eq!(
            geometry.swa_page_size, 1,
            "a radix window is token-granular"
        );

        let floors = ready.cache_floors();
        assert_eq!(linear_pool_min_slots(8, true), 33);
        assert_eq!(
            floors.mamba_slots, 32,
            "the physical floor minus the padding sink"
        );

        // The window floor goes the other way for the same reason: the
        // sentinel is part of the smallest pool the sizing formula can
        // hand back, so the smallest honest bound includes it.
        let floor = swa_pool_floor(8, 512, 64, DEFAULT_SWA_EVICTION_INTERVAL, false);
        assert_eq!(floors.swa_tokens, floor as u64 + 1);
        assert_eq!(
            swa_paged_num_tokens(floor, 0, 0.0, None),
            floors.swa_tokens as usize
        );

        assert_eq!(floors.kv_tokens, 64, "one page's worth of tokens");
        assert_eq!(floors.moe_experts, 32, "one MoE layer's routed experts");
    }

    /// A pool the model does not have is reported as absent, not as a
    /// pool of size zero with a floor of zero bytes -- a client that
    /// cannot tell the two apart offers a control the engine will only
    /// ever answer with an error.
    #[test]
    fn a_pool_the_model_lacks_has_no_floor_and_no_bounds() {
        let dense = CacheReadiness {
            num_pages: 8,
            page_size: 16,
            ..CacheReadiness::default()
        };
        assert_eq!(
            dense.cache_floors(),
            CacheFloors {
                kv_tokens: 16,
                ..CacheFloors::default()
            }
        );

        let geometry = dense.cache_geometry();
        assert_eq!(geometry.num_mamba_slots, 0);
        assert_eq!(geometry.swa_page_size, 0, "no window pool to advertise");
        let pools = crate::cache_report::CachePools::from_geometry(&geometry);
        assert_eq!(pools.targets(), vec!["kv"]);
        assert!(!dense.supports_swa_ratio());
        assert_eq!(dense.swa_full_tokens_ratio(), 0.0);
        assert_eq!(readiness().swa_full_tokens_ratio(), 0.5);
    }

    /// The advertised ceiling is the whole-cache one -- every pool, no
    /// fixed cost deducted -- so a client can show the real budget
    /// instead of reverse-deriving it from the per-pool bounds.
    #[test]
    fn the_whole_cache_ceiling_takes_no_fixed_cost_out_and_never_goes_negative() {
        let ready = readiness();
        assert_eq!(
            ready.cache_budget_bytes(),
            net_cache_budget_bytes(0.9, 12 * GIB, 4 * GIB, 0) as u64
        );
        assert_eq!(ready.free_vram_bytes(), 8 * GIB);

        // An over-committed deployment advertises no budget rather than
        // a wrapped-around enormous one through the unsigned geometry.
        assert!(net_cache_budget_bytes(0.9, 12 * GIB, 16 * GIB, 0) < 0);
        let over_committed = CacheReadiness {
            weights_bytes: 16 * GIB,
            ..readiness()
        };
        assert_eq!(over_committed.cache_budget_bytes(), 0);

        // A baseline that was never captured is "nothing to report",
        // which is the same 0 -- and the limits then fall back to the
        // free-VRAM baseline.
        let unmeasured = CacheReadiness {
            baseline_free_bytes: 0,
            ..readiness()
        };
        assert_eq!(unmeasured.cache_budget_bytes(), 0);
        assert_eq!(
            unmeasured.cache_limits().kv_tokens.unwrap().max,
            8 * GIB / 1024
        );
    }

    #[test]
    fn rebuild_bounds_are_the_budget_over_the_unit_cost() {
        let ready = readiness();
        let budget = ready.cache_budget_bytes();
        let limits = ready.cache_limits();

        assert_eq!(
            limits.kv_tokens.unwrap(),
            Limit {
                min: 64,
                max: budget / 1024
            }
        );
        assert_eq!(
            limits.mamba_slots.unwrap(),
            Limit {
                min: 32,
                max: budget / (1 << 16)
            }
        );
        assert_eq!(
            limits.swa_tokens.unwrap(),
            Limit {
                min: ready.cache_floors().swa_tokens,
                max: budget / 512
            }
        );

        // The expert cache is the one pool with a ceiling of its own:
        // past full residency more slots buy nothing, even though the
        // budget would pay for them many times over.
        assert!(budget / MIB > 32 * 8);
        assert_eq!(
            limits.moe_experts.unwrap(),
            Limit { min: 32, max: 256 },
            "capped at the model's own routed experts"
        );

        // An unpriced pool advertises no upper bound rather than a
        // fabricated 0..0.
        let unpriced = CacheReadiness {
            unit_bytes: UnitBytes::default(),
            ..readiness()
        };
        assert_eq!(unpriced.cache_limits().kv_tokens.unwrap().max, 0);
        assert_eq!(
            crate::cache_report::format_range(&unpriced.cache_geometry(), "kv"),
            ""
        );
    }

    /// End to end: the document this module produces is the one
    /// `cache_report` was written to render, VRAM column included --
    /// which is what nothing constructing a `CacheGeometry` used to
    /// cost the server.
    #[test]
    fn the_produced_geometry_renders_as_a_priced_status_table() {
        let text = crate::cache_report::format_cache_status(
            &readiness().cache_geometry(),
            "serving",
            "cache: ",
        );
        assert!(
            text.contains("vram"),
            "a priced engine keeps its column: {text}"
        );
        assert!(text.contains("128 slots (lru, 50%)"), "{text}");
        assert!(text.contains("32768 tok (512 x 64)"), "{text}");
        assert!(text.contains("19999 tok (19999 x 1)"), "{text}");
        assert!(text.contains("32 slots"), "{text}");
        assert!(text.contains("rebuild budget"), "{text}");
        assert!(
            text.contains("32..256"),
            "the advertised expert range: {text}"
        );
    }
}
