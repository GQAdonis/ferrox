//! ferrox-edge: the host-side *policy* of edge-native MoE serving.
//!
//! A Rust port of the parts of [FreeToken](https://github.com/FlashML-org/FreeToken)
//! (Apache-2.0) that decide things rather than compute them. FreeToken's
//! thesis, from *Efficient Edge-Native MoE Serving with Bandwidth-Adaptive
//! Execution* (arXiv:2608.16157), is that a personal machine is not a
//! small datacenter GPU but a heterogeneous, elastic pool of resources
//! -- VRAM, host RAM, PCIe, CPU cores -- whose balance differs per
//! machine, and that the serving stack should treat the *link between
//! them* as a scheduling signal rather than a fixed bottleneck.
//!
//! Everything in this crate follows from that. It is deliberately
//! tensor-free: each module takes measured numbers (bandwidths, byte
//! costs, pool sizes, token counts) and returns a decision, so every
//! policy here is testable on any host, with no GPU and no model.
//!
//! | module | decides |
//! |---|---|
//! | [`qstar`] | how many of a step's expert-cache misses to fetch over PCIe vs. run on the CPU |
//! | [`expert_cache`] | which experts stay resident in the GPU expert cache |
//! | [`radix`] | which prefix of a new prompt is already computed, and what may be evicted |
//! | [`anchor`] | which position an agentic turn will come back to, and what to keep for it |
//! | [`window_pool`] | which window-pool slot holds a given position's sliding-window KV |
//! | [`cache_manager`] | who owns which KV page, and what a request hands back when it commits |
//! | [`pool`] | how VRAM is split between the expert cache and the KV pools, and how it is re-split live |
//! | [`scheduler`] | which requests run this step, how much prefill they get, and who is retracted |
//! | [`parser`] | where a model's reasoning ends and its answer begins, and which tool it called |
//! | [`effort`] | which reasoning-effort dialect this checkpoint speaks |
//! | [`detokenize`] | what text is safe to stream after one more token |
//! | [`cache_report`] | what to show a human about all of the above |
//!
//! Where ferrox already had a mechanism the port would have duplicated
//! -- content-addressed KV blocks (`ferrox-core::kv_block`), the SSD
//! expert tier (`ferrox-core::expert_store`), continuous batching
//! (`ferrox-server::batch_scheduler`) -- these modules plug into it
//! instead of shadowing it. Each module's docs say which.

pub mod anchor;
pub mod cache_manager;
pub mod cache_report;
pub mod detokenize;
pub mod effort;
pub mod expert_cache;
pub mod parser;
pub mod placement;
pub mod pool;
pub mod qstar;
pub mod radix;
pub mod scheduler;
pub mod window_pool;

pub use anchor::{
    decode_slide, prefill_slide, resolve_anchor_token, snapshot_at_anchor, AnchorSnapshot,
    AnchorState, PingPong, RecurrentState, SlideDecision, SlidingRequest, WindowPolicy,
};
pub use cache_manager::{CacheManager, CommitOutcome, OutOfMemory, SequenceState};
pub use cache_report::{CacheGeometry, CachePools};
pub use detokenize::{
    find_printable_text, floor_char_boundary, stop_prefix_holdback, DetokenizeManager,
    DetokenizeMsg, Detokenizer,
};
pub use expert_cache::{CopyPlan, EnsurePlan, ExpertCache, ExpertCacheStats, ExpertId};
pub use parser::{
    ReasoningDelta, ReasoningFormat, ReasoningParser, ToolCall, ToolCallEvent, ToolCallFormat,
    ToolCallParser,
};
pub use placement::{auto_cpu_layers, parse_cpu_layers_spec};
pub use pool::{plan_cache_budget, validate_rebuild, PoolSizes, RebuildRequest};
pub use qstar::{
    balanced_fetch, recommend_backend, BandwidthProfile, MoeBackend, QStarPolicy, QStarSplit,
};

pub use window_pool::{WindowPoolExhausted, WindowSlotPool, NO_SLOT};

pub use scheduler::{
    finish_reason, AdmittedChunk, Capacity, DecodeSet, FinishReason, Geometry, NotAdmitted,
    PendingRequest, PrefillPass, SlotTable,
};

pub use radix::{
    HybridMatch, HybridRadixCache, InsertResult, MatchResult, NodeId, RadixCache, SwaMatch,
    SwaRadixCache,
};

pub use effort::{
    broadcast_effort_spellings, derive_think_gears, effective_efforts, probe_effort_profile,
    probe_thinking_profile, quantize_effort, resolve_thinking_mode, sanitize_effort, Effort,
    EffortMapping, EffortProfile, ThinkGears, ThinkingMode, ThinkingProfile, ThinkingState,
};
