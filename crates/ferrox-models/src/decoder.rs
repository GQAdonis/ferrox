//! Generic decoder-only transformer forward pass, assembled from a
//! ModelConfig. Each layer is: RMSNorm -> GQA attention (+RoPE) ->
//! residual -> RMSNorm -> MoE FFN (router + routed experts + shared
//! experts) -> residual. This is the standard decoder block shape
//! shared by the LLaMA/DeepSeek/GLM/Kimi family of open-weight models.
//!
//! Weight loading from a real GGUF checkpoint lives in `loader`
//! (`Decoder::from_gguf`); `Decoder::new_random` builds
//! correctly-shaped, randomly initialized weights so the full pipeline
//! -- embedding lookup, N decoder layers, output head -- can be
//! exercised end to end with real assertions about shapes, finiteness,
//! and determinism, without requiring a multi-hundred-gigabyte
//! checkpoint to be present.

use std::sync::atomic::{AtomicU64, Ordering};

use ferrox_core::attention::{
    apply_rope, apply_rope_interleaved, apply_rope_interleaved_with_freq_factors,
    apply_rope_with_freq_factors, causal_gqa_attention_prefill_shared_kv_windowed,
    causal_gqa_attention_softcap, causal_gqa_attention_windowed_softcap,
};
use ferrox_core::cache::{KvCache, PagedKvCache, PagedStoreExhausted, SharedPagedKv};
use ferrox_core::matmul::{geglu, rms_norm, rms_norm_per_head, softcap_inplace};
use rayon::prelude::*;

/// Whether the CUDA `gqa_decode` kernel should serve the per-token GQA
/// reduction (`FERROX_CUDA_GQA=1`). Off by default and only compiled with
/// `--features cuda`; the host path is byte-identical when unset.
#[cfg(feature = "cuda")]
fn cuda_gqa_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FERROX_CUDA_GQA").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_moe::{
    combine_expert_outputs, route_top_k, run_expert, run_expert_placed, ExpertPlacement,
    ExpertWeights, PlacementPlan,
};

use crate::config::ModelConfig;

pub struct AttnWeights {
    pub q_proj: WeightMatrix, // [n_heads*head_dim, hidden_dim]
    pub k_proj: WeightMatrix, // [n_kv_heads*head_dim, hidden_dim]
    pub v_proj: WeightMatrix, // [n_kv_heads*head_dim, hidden_dim]
    pub o_proj: WeightMatrix, // [hidden_dim, n_heads*head_dim]
    pub norm_weight: Vec<f32>,
    /// OLMoE-style QK-RMSNorm (`attn_q_norm`/`attn_k_norm` GGUF tensors),
    /// applied to the *whole* q_proj/k_proj output (width `n_heads*head_dim`
    /// / `n_kv_heads*head_dim`) before RoPE -- confirmed against
    /// `OlmoeAttention.forward` in `transformers/models/olmoe/modeling_olmoe.py`
    /// (`q_norm(q_proj(x))`, `k_norm(k_proj(x))`, both plain whole-vector
    /// RMSNorm, not per-head). `None` for every model that doesn't ship
    /// these tensors -- absent, not zero/identity-weighted, so existing
    /// presets/fixtures are byte-for-byte unaffected.
    ///
    /// Qwen3 / Gemma3 ship the same tensor names with length `head_dim`
    /// (per-head). Which style is used is selected by
    /// [`ModelConfig::qk_norm_style`] (refined at load from weight length).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// Qwen2/Qwen2-MoE-family QKV attention bias (`attn_{q,k,v}.bias`
    /// GGUF tensors, real `config.qkv_bias`), added elementwise to the
    /// corresponding projection's output before QK-norm/RoPE -- confirmed
    /// against the real `transformers` source
    /// (`Qwen2MoeAttention.__init__`: `q_proj = nn.Linear(..., bias=
    /// config.qkv_bias)`, same for `k_proj`/`v_proj`; `o_proj` has no
    /// bias). Found as a real, previously-unhandled architecture gap:
    /// ferrox's generic GGUF loader silently ignored these real tensors
    /// entirely, producing fluent-but-wrong output on a real downloaded
    /// Qwen1.5-MoE checkpoint (same failure class as OLMoE's missing
    /// QK-norm). `None` for every model that doesn't ship these tensors.
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    /// Gemma 2+/3 post-attention RMSNorm (`blk.N.post_attention_norm.weight`
    /// / llama.cpp `attn_post_norm`). Applied to attention output before
    /// the residual add. `None` for Llama/Qwen/OLMoE.
    pub post_attn_norm: Option<Vec<f32>>,
    /// Gemma 2+/3 post-FFN RMSNorm (`blk.N.post_ffw_norm.weight`).
    pub post_ffn_norm: Option<Vec<f32>>,
}

/// How a layer's routed experts are held. `Resident` is the original
/// always-in-memory form (owned f32 or zero-copy mmap views).
/// `Stored` holds only byte-range layouts; each use acquires the
/// expert's bytes from a bounded, lease-protected
/// `ferrox_core::expert_store::ExpertStore` shared by every layer
/// (one global byte budget), builds temporary `WeightMatrix` views
/// over the leased buffer (`WeightBytes::Shared`, which pins the
/// cache entry for the views' lifetime), and drops them after the
/// expert runs. Dequantized math over identical bytes is identical,
/// so the two backings are bit-equivalent by construction -- pinned
/// by an integration test against the MoE fixture.
pub enum ExpertBacking {
    Resident(Vec<ExpertWeights>),
    Stored {
        store:
            std::sync::Arc<ferrox_core::expert_store::ExpertStore<crate::loader::GgufExpertSource>>,
        layouts: Vec<crate::loader::StoredExpertLayout>,
        layer: u32,
    },
}

impl ExpertBacking {
    pub fn n_experts(&self) -> usize {
        match self {
            ExpertBacking::Resident(v) => v.len(),
            ExpertBacking::Stored { layouts, .. } => layouts.len(),
        }
    }
}

pub struct MoeWeights {
    pub router: WeightMatrix, // [n_experts, hidden_dim]
    pub experts: ExpertBacking,
    pub shared_experts: Vec<ExpertWeights>,
    /// Qwen2-MoE-specific: when present, the shared experts' combined
    /// output is scaled by `sigmoid(shared_expert_gate . x)` before
    /// being added to the routed output, instead of added unconditionally
    /// -- confirmed against the real `transformers` source
    /// (`Qwen2MoeSparseMoeBlock.forward`: `shared_expert_output =
    /// F.sigmoid(self.shared_expert_gate(hidden_states)) *
    /// shared_expert_output`) and llama.cpp's real `qwen2moe.cpp`
    /// (`ffn_gate_inp_shexp` dotted against the hidden state, sigmoid,
    /// multiplied into the shared-expert branch before the final add).
    /// Real on-disk shape is `[hidden_dim]` (a `Linear(hidden_dim, 1,
    /// bias=false)`'s weight, flattened -- ggml's real `create_tensor`
    /// call declares it as `{n_embd}`, not a 2D matrix), so this is a
    /// plain owned vector dotted with the normed hidden state directly,
    /// not a `WeightMatrix`. `None` for every other architecture
    /// (DeepSeek-V3's shared experts, for one real confirmed contrast,
    /// add unconditionally with no gate at all).
    pub shared_expert_gate: Option<Vec<f32>>,
    pub norm_weight: Vec<f32>,
    /// DeepSeek-V3's aux-loss-free expert-selection bias, on disk as
    /// `blk.{N}.exp_probs_b.bias` (llama.cpp's `LLM_TENSOR_FFN_EXP_PROBS_B`
    /// -- note the on-disk name has no `ffn_` prefix, `llama-arch.cpp:416`).
    /// It is added to the *selection* score only: the top-k is taken over
    /// `gating(logit) + bias[expert]`, while each winner's combine weight
    /// comes from the unbiased `gating(logit)`
    /// (`build_moe_ffn`: "leave probs unbiased as it's later used to get
    /// expert weights"). Biasing the weight too would silently skew every
    /// routed contribution away from what the router learned.
    ///
    /// `None` for every checkpoint that does not ship the tensor. When it
    /// *is* present, the GPU MoE fast paths refuse the layer rather than
    /// route without it -- their kernels have no bias input.
    pub exp_probs_bias: Option<Vec<f32>>,
    /// How many times each routed expert (index into `experts`) has been
    /// selected by `route_top_k` across every `forward_token`/
    /// `forward_batch` call so far. Real observed hotness, not a
    /// placeholder -- feeds `placement_plan` below, which is what
    /// `PlacementPlan::from_budget` needs to prioritize actually-hot
    /// experts for GPU residency instead of guessing by index.
    pub activation_counts: Vec<AtomicU64>,
    /// Verified-at-load contiguous expert planes for Metal MoE
    /// (`mul_mm_sg` gather/id). Built in `loader` when every routed expert
    /// is mmap-backed with a simdgroup-GEMM quant (Q4_0 / Q4_K / Q8_0 / …)
    /// and back-to-back gate/up/down slices. Gate/up/down kinds may differ
    /// (Qwen1.5-MoE: Q4_K gate/up + Q8_0 down). `None` for store-backed,
    /// F32, or non-contiguous layouts.
    #[cfg(feature = "metal")]
    pub packed_q4: Option<MoePackedQ4Planes>,
}

/// Load-time validated contiguous expert tensor planes (any `mul_mm_sg` quant).
#[cfg(feature = "metal")]
pub struct MoePackedQ4Planes {
    gate: ferrox_core::weight_matrix::WeightBytes,
    up: ferrox_core::weight_matrix::WeightBytes,
    down: ferrox_core::weight_matrix::WeightBytes,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    n_experts: usize,
    ffn_rows: usize,
    hidden_rows: usize,
    gate_row_bytes: usize,
    down_row_bytes: usize,
    gate_kind: &'static str,
    up_kind: &'static str,
    down_kind: &'static str,
}

#[cfg(feature = "metal")]
impl MoePackedQ4Planes {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gate: ferrox_core::weight_matrix::WeightBytes,
        up: ferrox_core::weight_matrix::WeightBytes,
        down: ferrox_core::weight_matrix::WeightBytes,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        n_experts: usize,
        ffn_rows: usize,
        hidden_rows: usize,
        gate_kind: &'static str,
        up_kind: &'static str,
        down_kind: &'static str,
    ) -> Self {
        Self {
            gate,
            up,
            down,
            gate_stride,
            up_stride,
            down_stride,
            n_experts,
            ffn_rows,
            hidden_rows,
            gate_row_bytes: gate_stride / ffn_rows,
            down_row_bytes: down_stride / hidden_rows,
            gate_kind,
            up_kind,
            down_kind,
        }
    }

    pub fn view(&self) -> ferrox_metal::gpu::MoePackedQ4<'_> {
        ferrox_metal::gpu::MoePackedQ4 {
            gate: self.gate.as_slice(),
            up: self.up.as_slice(),
            down: self.down.as_slice(),
            gate_stride: self.gate_stride,
            up_stride: self.up_stride,
            down_stride: self.down_stride,
            n_experts: self.n_experts,
            ffn_rows: self.ffn_rows,
            hidden_rows: self.hidden_rows,
            gate_row_bytes: self.gate_row_bytes,
            down_row_bytes: self.down_row_bytes,
            gate_kind: self.gate_kind,
            up_kind: self.up_kind,
            down_kind: self.down_kind,
        }
    }
}

impl MoeWeights {
    pub fn n_experts(&self) -> usize {
        self.experts.n_experts()
    }

    /// This routed expert's weight byte footprint, from resident
    /// matrices or the stored layout -- identical numbers either way,
    /// so residency planning is backing-independent.
    pub fn expert_bytes(&self, e: usize) -> usize {
        match &self.experts {
            ExpertBacking::Resident(v) => {
                let ex = &v[e];
                ex.gate.resident_bytes() + ex.up.resident_bytes() + ex.down.resident_bytes()
            }
            ExpertBacking::Stored { layouts, .. } => layouts[e].total_bytes(),
        }
    }

    /// Runs `f` against expert `e`'s weights, materializing them from
    /// the store first when this layer is store-backed. The lease (and
    /// therefore the cache entry's pin) lives exactly as long as `f`'s
    /// borrow.
    pub fn with_expert<R>(&self, e: usize, f: impl FnOnce(&ExpertWeights) -> R) -> R {
        match &self.experts {
            ExpertBacking::Resident(v) => f(&v[e]),
            ExpertBacking::Stored {
                store,
                layouts,
                layer,
            } => {
                let lease = store
                    .acquire(ferrox_core::expert_store::ExpertKey {
                        layer: *layer,
                        expert: e as u32,
                    })
                    .unwrap_or_else(|err| {
                        panic!(
                            "expert store read failed for layer {layer} expert {e}: {err} \
                             (checkpoint file unreadable mid-decode)"
                        )
                    });
                let tmp = layouts[e].materialize(&lease);
                f(&tmp)
            }
        }
    }

    fn record_activations(&self, expert_ids: &[usize]) {
        for &eid in expert_ids {
            if let Some(counter) = self.activation_counts.get(eid) {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A real VRAM-budget-and-hotness-driven placement plan for this
    /// layer's routed experts, built from each expert's actual resident
    /// byte size (`WeightMatrix::resident_bytes()` summed across its
    /// gate/up/down matrices, so it reflects the real quantization
    /// format in use, not an estimate) and the activation counts
    /// observed so far. See `ferrox_moe::PlacementPlan::from_budget`.
    pub fn placement_plan(&self, vram_budget_bytes: u64) -> PlacementPlan {
        let sizes: Vec<usize> = (0..self.n_experts())
            .map(|e| self.expert_bytes(e))
            .collect();
        let counts: Vec<u64> = self
            .activation_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let has_observations = counts.iter().any(|&c| c > 0);
        PlacementPlan::from_budget(
            &sizes,
            has_observations.then_some(counts.as_slice()),
            vram_budget_bytes,
        )
    }
}

pub struct LayerWeights {
    pub attn: AttnWeights,
    pub moe: MoeWeights,
}

/// The per-layer weights the gpt-oss graph carries and the generic GQA
/// layer structs do not.
///
/// Held as a side table on [`Decoder`] rather than as new `Option`
/// fields on [`AttnWeights`]/[`MoeWeights`] for two reasons. The first
/// is mechanical: those two structs have thirty construction sites
/// across seven loaders and every dedicated engine, and none of them
/// will ever set these. The second is the point of the exercise — a
/// checkpoint either has the whole gpt-oss graph or none of it, so
/// `Decoder::gpt_oss.is_some()` is a single, checkable predicate for
/// "this model needs the gpt-oss path", which is what the CPU-only and
/// paged-attention refusals below key off. Scattering five independent
/// `Option`s would make "half the graph is wired" representable, and
/// that state is precisely the silent-wrong-answer bug this work exists
/// to remove.
pub struct GptOssLayer {
    /// `blk.N.attn_sinks.weight`, one learned logit per query head.
    pub attn_sinks: Vec<f32>,
    /// `blk.N.attn_output.bias`, added after the output projection.
    pub o_bias: Vec<f32>,
    /// `blk.N.ffn_gate_inp.bias`, added to the router logits.
    pub router_bias: Vec<f32>,
    /// `blk.N.ffn_{gate,up,down}_exps.bias`, one entry per expert.
    pub expert_bias: Vec<ferrox_moe::ExpertBias>,
}

/// gpt-oss side table: one entry per layer, in layer order.
pub struct GptOssWeights {
    pub layers: Vec<GptOssLayer>,
}

pub struct Decoder {
    pub config: ModelConfig,
    /// `[vocab_size, hidden_dim]`. A `WeightMatrix` rather than an
    /// eagerly-widened f32 `Tensor`, so a quantized `token_embd.weight`
    /// stays quantized on disk/mmap and token lookup dequantizes one
    /// row at a time (`WeightMatrix::dequant_row`) -- a large-vocab
    /// model's embedding table is multi-GB in f32 and only ever read
    /// row-wise.
    pub embedding: WeightMatrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub output_head: WeightMatrix, // [vocab_size, hidden_dim]
    /// Real VRAM budget for GPU-resident routed experts.
    /// `None` (both constructors below
    /// set it) means every expert always runs on CPU -- the exact
    /// behavior this field's absence had before it existed. `Some(bytes)`
    /// makes each forward call build ONE global `ResidencyPlan`
    /// (`Decoder::residency_plan`) across every layer's actual
    /// resident expert sizes and observed activation counts against
    /// this single budget -- the budget is never re-spent per layer --
    /// dispatching device-placed routed experts through
    /// `ferrox_moe::run_expert_placed` (a real CUDA kernel when the
    /// `cuda` feature is compiled in and the expert's quant kind has
    /// one; a correct CPU fallback otherwise, so setting this on a
    /// non-`cuda` build is harmless, just never GPU-accelerated).
    /// Shared experts and a dense layer's sole expert always run on
    /// CPU regardless -- every token activates them, so there's no
    /// routing decision to offload the way routed-expert placement is.
    /// Rebuilding the plan on every forward call is real but not yet
    /// performance-tuned; a real, disclosed limit, not a correctness
    /// gap.
    pub gpu_vram_budget_bytes: Option<u64>,
    /// `Some` only for the gpt-oss family. See [`GptOssWeights`]. When
    /// set, every layer runs the gpt-oss CPU graph (attention sinks,
    /// alternating SWA, biased router + experts, `swiglu_oai`), GPU
    /// offload is refused at load time, and the paged-KV decode path is
    /// refused at call time — neither implements sinks, and answering
    /// with a different distribution is the failure this replaces.
    pub gpt_oss: Option<GptOssWeights>,
    /// Per-layer Metal-resident KV for fused decode/prefill attention
    /// (`FERROX_METAL_ATTN`). Lazily allocated. After
    /// [`ferrox_metal::attn::launch_decode_dense_stack`], Metal KV is
    /// authoritative for the next decode step; host [`KvCache`] may lag
    /// until [`Self::sync_metal_attn_kv_to_host`] or a CPU fallback.
    /// Prefill / prefix restore still upload host → Metal when lengths
    /// diverge for other reasons.
    #[cfg(feature = "metal")]
    pub(crate) metal_attn_kv: std::sync::Mutex<Option<Vec<ferrox_metal::attn::MetalKvBuffers>>>,
    /// Load-time execution plan (family, fused-op caps, SWA/RoPE
    /// policy). Built once; hot path must not re-resolve architecture
    /// strings. See [`crate::execution_plan`].
    pub execution_plan: crate::execution_plan::ExecutionPlan,
    /// Cache key hit → fused caps last used for that geometry (enables
    /// decode/prefill plan reuse without rebuilding residency).
    pub plan_cache: std::sync::Mutex<
        std::collections::HashMap<
            crate::execution_plan::PlanGeometry,
            crate::execution_plan::FusedOpCaps,
        >,
    >,
}

/// Simple deterministic pseudo-random generator so tests are
/// reproducible without pulling in an external `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * 0.1).collect()
    }
}

impl Decoder {
    /// Eagerly resolve every kernel lookup this model's dispatch paths
    /// will make, and record it in
    /// [`ferrox_core::kernel_registry`] before anything runs.
    ///
    /// Call once, at the end of loading, immediately before
    /// [`ferrox_core::kernel_registry::seal`]. Nothing here dispatches
    /// or decides anything: it asks the same predicates the hot path
    /// asks and writes the answers down, so a kernel that is missing
    /// becomes a startup line instead of an unexplained benchmark row.
    ///
    /// Routed experts held in an [`ExpertBacking::Stored`] layer are not
    /// probed -- they exist only as byte ranges until a token routes to
    /// them, and materialising every expert here would defeat the
    /// bounded expert store. Their kinds are the same as the resident
    /// case, and a dispatch-site miss still trips the sealed registry.
    pub fn probe_kernels(&self) {
        use ferrox_core::kernel_registry as reg;

        if !reg::enabled() {
            return;
        }
        self.embedding.probe_kernels("token_embd");
        self.output_head.probe_kernels("output_head");
        for layer in &self.layers {
            layer.attn.q_proj.probe_kernels("attn_q");
            layer.attn.k_proj.probe_kernels("attn_k");
            layer.attn.v_proj.probe_kernels("attn_v");
            layer.attn.o_proj.probe_kernels("attn_o");
            layer.moe.router.probe_kernels("moe_router");
            for e in &layer.moe.shared_experts {
                e.gate.probe_kernels("shexp_gate");
                e.up.probe_kernels("shexp_up");
                e.down.probe_kernels("shexp_down");
            }
            if let ExpertBacking::Resident(experts) = &layer.moe.experts {
                for e in experts {
                    e.gate.probe_kernels("ffn_gate");
                    e.up.probe_kernels("ffn_up");
                    e.down.probe_kernels("ffn_down");
                }
            }
        }
        // The generic decoder has a real batched prefill
        // (`forward_hidden_batch`), so a `pp512` here is one GEMM per
        // projection, not 512 matvecs. Recorded as a hit so that an
        // engine which lacks it stands out as a miss rather than as an
        // absence.
        reg::record_build(
            reg::Lookup::new(
                ferrox_core::weight_matrix::active_backend(),
                reg::op::ENGINE_PREFILL_BATCH,
                None,
            )
            .with_role("generic_decoder"),
            reg::Outcome::Hit,
        );
    }

    /// Builds a decoder with correctly-shaped, randomly initialized
    /// weights for `config`, but overrides `n_layers` and `vocab_size`
    /// with small test-scale numbers so it can actually be allocated and
    /// run inside a CI sandbox. Use this to validate the forward-pass
    /// plumbing only, never to draw conclusions about real model
    /// quality.
    pub fn new_random_small(config: ModelConfig, n_layers: usize, vocab_size: usize) -> Self {
        let mut rng = Lcg::new(42);
        let mut config = config;
        config.n_layers = n_layers;
        config.vocab_size = vocab_size;
        let hidden = config.hidden_dim;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;

        let embedding = WeightMatrix::F32(Tensor::new(
            rng.vec(vocab_size * hidden),
            vec![vocab_size, hidden],
        ));

        let wm = |data: Vec<f32>, shape: Vec<usize>| WeightMatrix::F32(Tensor::new(data, shape));

        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let attn = AttnWeights {
                q_proj: wm(
                    rng.vec(n_heads * head_dim * hidden),
                    vec![n_heads * head_dim, hidden],
                ),
                k_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                v_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                o_proj: wm(
                    rng.vec(hidden * n_heads * head_dim),
                    vec![hidden, n_heads * head_dim],
                ),
                norm_weight: vec![1.0; hidden],
                q_norm: None,
                k_norm: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                post_attn_norm: None,
                post_ffn_norm: None,
            };

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) get a single-expert, no-shared-expert
            // dense-equivalent FFN regardless of this model's global
            // MoE topology, matching the DeepSeek-2/3-family
            // convention found in ik_llama.cpp's source.
            let is_dense_layer = config.layer_is_dense(layer_idx);
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let n_shared = if is_dense_layer {
                0
            } else {
                config.moe.n_shared_experts
            };
            let ffn_dim = config.moe.expert_ffn_dim;
            let make_expert = |rng: &mut Lcg| ExpertWeights {
                gate: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                up: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                down: WeightMatrix::F32(Tensor::new(
                    rng.vec(hidden * ffn_dim),
                    vec![hidden, ffn_dim],
                )),
            };
            let experts: Vec<ExpertWeights> =
                (0..n_experts).map(|_| make_expert(&mut rng)).collect();
            let shared_experts = (0..n_shared).map(|_| make_expert(&mut rng)).collect();
            let activation_counts = (0..experts.len()).map(|_| AtomicU64::new(0)).collect();

            let moe = MoeWeights {
                exp_probs_bias: None,
                router: wm(rng.vec(n_experts * hidden), vec![n_experts, hidden]),
                experts: ExpertBacking::Resident(experts),
                shared_experts,
                shared_expert_gate: None,
                norm_weight: vec![1.0; hidden],
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4: None,
            };

            layers.push(LayerWeights { attn, moe });
        }

        let final_norm = vec![1.0; hidden];
        let output_head = wm(rng.vec(vocab_size * hidden), vec![vocab_size, hidden]);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            crate::capability::DecoderFamily::StandardGqa,
            crate::capability::MemoryKind::KvGqa,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        Decoder {
            config,
            embedding,
            layers,
            final_norm,
            output_head,
            gpu_vram_budget_bytes: None,
            // Synthetic-weights constructor: no checkpoint, no gpt-oss.
            gpt_oss: None,
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Applies RoPE to one head's Q or K slice. Dispatches on both
    /// `rope_layout` (Norm = adjacent-pair / NeoX = split-half -- see
    /// `RopeLayout`) and whether this checkpoint carries a real
    /// `rope_freqs.weight` tensor (Llama 3/3.1/3.2's per-band frequency
    /// correction). Getting the layout wrong for `llama` was the real
    /// root cause of the Llama-3.1-8B early-stop bug: ferrox applied
    /// NeoX pairing to an architecture that needs Norm.
    fn apply_rope_head_theta(&self, slice: &mut [f32], pos: usize, theta: f32) {
        use crate::config::RopeLayout;
        // Partial rotary (llama.cpp `hparams.n_rot` < `n_embd_head_k`,
        // GGUF `<arch>.rope.dimension_count`): Phi-3/Phi-4 rotate only the
        // first 96 of each 128-wide head and pass the remaining 32
        // through untouched. Rotating the whole head instead is not a
        // subtle error — it moves dimensions the model never trained to
        // be position-dependent.
        let slice = match self.config.rope_dim {
            Some(rot) if rot < slice.len() => &mut slice[..rot],
            _ => slice,
        };
        match (self.config.rope_layout, &self.config.rope_freqs) {
            (RopeLayout::Norm, Some(freq_factors)) => {
                apply_rope_interleaved_with_freq_factors(slice, pos, theta, freq_factors)
            }
            (RopeLayout::Norm, None) => apply_rope_interleaved(slice, pos, theta),
            (RopeLayout::Neox, Some(freq_factors)) => {
                apply_rope_with_freq_factors(slice, pos, theta, freq_factors)
            }
            (RopeLayout::Neox, None) => apply_rope(slice, pos, theta),
        }
    }

    fn apply_rope_head_layer(&self, slice: &mut [f32], pos: usize, layer_idx: usize) {
        self.apply_rope_head_theta(slice, pos, self.config.layer_rope_theta(layer_idx))
    }

    /// llama.cpp's RoPE `mscale` (ggml `rope_yarn`), applied where the
    /// QKV biases and QK-norms are: multiplying `cos`/`sin` by a constant
    /// is the same as scaling the vector RoPE rotates, and rotation is
    /// linear, so pre-scaling q and k here is exactly what the kernel
    /// would do post-hoc — without a new uniform on five backends' RoPE
    /// kernels.
    ///
    /// Both q and k are scaled, so attention logits carry `m²`, which is
    /// the whole observable effect (V is untouched, and k enters the
    /// cache scaled exactly as llama.cpp's does).
    #[inline]
    fn apply_rope_attn_factor(&self, q: &mut [f32], k: &mut [f32]) {
        let m = self.config.rope_attn_factor;
        if m == 1.0 {
            return;
        }
        // ggml folds `attn_factor` into cos_theta/sin_theta inside
        // `rope_yarn` (ops.cpp), so it reaches ONLY the rotated channels;
        // `[n_rot, head_dim)` is then copied through untouched by the
        // "fill the remain channels with data from src tensor" loop.
        // Scaling the pass-through tail as well is a different graph, and
        // `ferrox parity` caught it as the one DRIFT verdict in a
        // 17-model sweep: Phi-4-mini rotates 96 of 128 dims with
        // attn_factor 1.1902, so 32 dims per head were scaled that
        // llama.cpp leaves alone.
        let head_dim = self.config.head_dim;
        let rot = self.config.rope_dim.unwrap_or(head_dim).min(head_dim);
        for buf in [q, k] {
            for head in buf.chunks_mut(head_dim) {
                let n = rot.min(head.len());
                for v in head[..n].iter_mut() {
                    *v *= m;
                }
            }
        }
    }

    /// Applies Q/K RMSNorm according to [`ModelConfig::qk_norm_style`].
    fn apply_qk_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        use crate::capability::QkNormStyle;
        match self.config.qk_norm_style {
            QkNormStyle::WholeVector => rms_norm(x, weight, self.config.rms_norm_eps),
            QkNormStyle::PerHead => {
                rms_norm_per_head(x, weight, self.config.head_dim, self.config.rms_norm_eps)
            }
        }
    }

    /// Builds a Metal [`MatvecLaunch`] for a quantized matrix, or `None`
    /// if the storage/kind cannot run on Metal.
    #[cfg(feature = "metal")]
    fn metal_matvec_launch<'a>(m: &'a WeightMatrix) -> Option<ferrox_metal::gpu::MatvecLaunch<'a>> {
        match m {
            WeightMatrix::F32(t) => {
                let rows = t.shape[0];
                let cols = t.shape[1];
                let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                    ferrox_metal::gpu::matvec_launch_meta("F32")?;
                // SAFETY: f32 ↔ little-endian byte view for Metal upload/alias.
                let bytes = unsafe {
                    std::slice::from_raw_parts(t.data.as_ptr() as *const u8, t.data.len() * 4)
                };
                Some(ferrox_metal::gpu::MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights: bytes,
                    rows,
                    row_bytes: cols * 4,
                    rows_per_tg,
                })
            }
            WeightMatrix::Quantized {
                data,
                rows,
                cols: _,
                kind,
            } => {
                let kind_name = match kind {
                    ferrox_core::QuantKind::Q8_0 => "Q8_0",
                    ferrox_core::QuantKind::Q4_0 => "Q4_0",
                    ferrox_core::QuantKind::Q4K => "Q4_K",
                    ferrox_core::QuantKind::Q5K => "Q5_K",
                    ferrox_core::QuantKind::Q6K => "Q6_K",
                    ferrox_core::QuantKind::IQ4XS => "IQ4_XS",
                    _ => return None,
                };
                let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                    ferrox_metal::gpu::matvec_launch_meta(kind_name)?;
                // A zero-row matrix has no rows to stride over, so
                // there is no meaningful row size; `checked_div`
                // says that once instead of splitting it across a
                // guard and a bare division.
                let row_bytes = data.as_slice().len().checked_div(*rows).unwrap_or(0);
                Some(ferrox_metal::gpu::MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights: data.as_slice(),
                    rows: *rows,
                    row_bytes,
                    rows_per_tg,
                })
            }
            _ => None,
        }
    }

    /// True when this layer can use the fused Metal attention block
    /// (Norm or NeoX RoPE, quantized projections; QKV bias + QK-norm
    /// via [`ferrox_metal::attn::AttnExtras`]).
    #[cfg(feature = "metal")]
    fn layer_supports_metal_attn(&self, layer: &LayerWeights) -> bool {
        use crate::config::RopeLayout;
        // gpt-oss: no Metal kernel implements attention sinks, so the
        // fused stacks would compute a *different* attention than the
        // CPU path for the same weights. Keep this family on CPU rather
        // than letting the two backends disagree. See `Decoder::gpt_oss`.
        if self.gpt_oss.is_some() {
            return false;
        }
        if !matches!(self.config.rope_layout, RopeLayout::Norm | RopeLayout::Neox) {
            return false;
        }
        // QKV bias (Qwen2) and QK-norm — per-head (Qwen3/Gemma-3) or
        // whole-vector (OLMoE) — run on Metal via AttnExtras.
        let q_len = self.config.n_heads * self.config.head_dim;
        let k_len = self.config.n_kv_heads * self.config.head_dim;
        let qk_norm_ok = |w: Option<&Vec<f32>>, vec_len: usize| -> bool {
            match w {
                None => true,
                Some(w) if w.len() == self.config.head_dim => true,
                Some(w) if w.len() == vec_len => true,
                _ => false,
            }
        };
        if !qk_norm_ok(layer.attn.q_norm.as_ref(), q_len)
            || !qk_norm_ok(layer.attn.k_norm.as_ref(), k_len)
        {
            return false;
        }
        // Softcaps: final logit softcap is applied on the host after
        // lm_head (Metal-safe). Attention softcap runs on Metal FA-vec /
        // legacy GQA (decode + prefill). attention_scale is compensated
        // by scaling Q on the host/Metal extras path.
        if self.config.head_dim > 256 {
            return false;
        }
        // Partial rotary (`n_rot < head_dim`) and LongRoPE's `mscale`
        // now ride the Metal RoPE kernels as the `rot_dim` / `mscale`
        // uniforms on [`ferrox_metal::attn::MetalRope`], so Phi-3/Phi-4
        // are admitted here. `n_rot` must still be even — ggml's
        // `ggml_rope_impl` asserts it, and an odd width would leave one
        // channel's pairing undefined rather than merely unrotated.
        if self
            .config
            .rope_dim
            .is_some_and(|rot| rot == 0 || rot % 2 != 0 || rot > self.config.head_dim)
        {
            return false;
        }
        Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// Layer features only the fused dense stack implements — the
    /// per-layer Metal launches would silently skip them (wrong output).
    #[cfg(feature = "metal")]
    fn layer_needs_metal_stack(&self, layer: &LayerWeights, layer_idx: usize) -> bool {
        layer.attn.post_attn_norm.is_some()
            || layer.attn.post_ffn_norm.is_some()
            || self.config.layer_sliding_window(layer_idx).is_some()
            || matches!(
                self.config.ffn_activation,
                crate::config::FfnActivation::Gelu
            )
            || self.config.layer_rope_theta(layer_idx) != self.config.rope_theta
    }

    /// Optional QKV bias / QK-norm ops for the Metal attn paths.
    #[cfg(feature = "metal")]
    fn metal_attn_extras<'a>(&self, layer: &'a LayerWeights) -> ferrox_metal::attn::AttnExtras<'a> {
        ferrox_metal::attn::AttnExtras {
            q_bias: layer.attn.q_bias.as_deref(),
            k_bias: layer.attn.k_bias.as_deref(),
            v_bias: layer.attn.v_bias.as_deref(),
            q_norm: layer.attn.q_norm.as_deref(),
            k_norm: layer.attn.k_norm.as_deref(),
            attn_logit_softcap: self.config.attn_logit_softcap,
        }
    }

    /// GPU expert residency only when Metal attention stays on-device
    /// (when the Metal dense+attn path is active). Avoids CPU-attention
    /// ↔ GPU-expert activation ping-pong on Metal MoE.
    #[cfg(feature = "metal")]
    fn expert_residency_plan(&self, use_metal_attn: bool) -> Option<ferrox_moe::ResidencyPlan> {
        if ferrox_core::metal_dense_enabled()
            && ferrox_metal::attn::metal_attn_enabled()
            && !use_metal_attn
        {
            return None;
        }
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    #[cfg(not(feature = "metal"))]
    fn expert_residency_plan(&self, _use_metal_attn: bool) -> Option<ferrox_moe::ResidencyPlan> {
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    /// Map this checkpoint's RoPE onto the Metal kernel uniforms:
    /// pairing convention, rotary width (`n_rot`), and ggml `rope_yarn`'s
    /// `mscale`. The last two are what
    /// [`Decoder::apply_rope_head_theta`] and
    /// [`Decoder::apply_rope_attn_factor`] do on the CPU side, so the
    /// two backends stay one graph.
    #[cfg(feature = "metal")]
    fn metal_rope(&self) -> ferrox_metal::attn::MetalRope {
        use crate::config::RopeLayout;
        let layout = match self.config.rope_layout {
            RopeLayout::Norm => ferrox_metal::attn::MetalRopeLayout::Norm,
            RopeLayout::Neox => ferrox_metal::attn::MetalRopeLayout::Neox,
        };
        ferrox_metal::attn::MetalRope {
            layout,
            rot_dim: self
                .config
                .rope_dim
                .filter(|rot| *rot < self.config.head_dim),
            attn_factor: self.config.rope_attn_factor,
        }
    }

    /// Dense FFN (single expert) with Metal-capable gate/up/down.
    #[cfg(feature = "metal")]
    fn layer_supports_metal_dense_ffn(layer: &LayerWeights) -> bool {
        Self::is_dense_layer(layer)
            && layer.moe.with_expert(0, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            })
    }

    /// Dense layer eligible for the one-CB `mul_mm_sg` prefill stack.
    /// QKV bias / QK-norm are applied on-GPU via [`AttnExtras`] (same as
    /// decode); SWA fit is checked separately.
    #[cfg(feature = "metal")]
    fn metal_prefill_dense_layer_eligible(layer: &LayerWeights) -> bool {
        Self::is_dense_layer(layer)
    }

    #[cfg(feature = "metal")]
    fn metal_prefill_dense_swa_fits(
        &self,
        layer_idx: usize,
        start_pos: usize,
        batch_size: usize,
    ) -> bool {
        match self.config.layer_sliding_window(layer_idx) {
            Some(window) => start_pos + batch_size <= window,
            None => true,
        }
    }

    /// Routed-expert FFN for the fused prefill stack, or `None` when this
    /// layer must keep the host-routed path (`launch_moe_prefill_q4_0`).
    ///
    /// Note: routing happens on the GPU here, so prefill no longer feeds
    /// `record_activations`. Expert hotness for `inspect-plan` comes from
    /// decode, which still routes on the host.
    #[cfg(feature = "metal")]
    fn metal_prefill_moe<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
    ) -> Option<ferrox_metal::gpu::PrefillMoeMetal<'a>> {
        if !ferrox_metal::attn::metal_moe_stack_enabled()
            || !ferrox_metal::attn::metal_moe_resident_enabled()
            || Self::is_dense_layer(layer)
            || !layer.moe.shared_experts.is_empty()
            || !matches!(
                config.ffn_activation,
                crate::config::FfnActivation::Swiglu | crate::config::FfnActivation::SwigluFused
            )
            || !matches!(config.moe.gating, ferrox_moe::GatingFunction::Softmax)
            || config.moe.expert_group_count.is_some()
            // The GPU router kernels take router weights and nothing
            // else: no `exp_probs_b` input, no `expert_weights_scale`
            // uniform. A layer carrying either must stay on the CPU
            // router rather than be routed without them.
            || layer.moe.exp_probs_bias.is_some()
            || config.moe.expert_weights_scale != 1.0
        {
            return None;
        }
        let ferrox_core::weight_matrix::WeightMatrix::F32(router) = &layer.moe.router else {
            return None;
        };
        let packed = Self::moe_packed_q4(&layer.moe)?;
        let moe = ferrox_metal::gpu::PrefillMoeMetal {
            router_w: &router.data,
            top_k: config.moe.n_experts_active,
            renormalize: config.moe.norm_topk_prob,
            packed,
        };
        moe.is_supported().then_some(moe)
    }

    /// FFN half of a fused-prefill-stack layer: dense `mul_mm_sg` launches
    /// or (MoE) the routed-expert description.
    #[cfg(feature = "metal")]
    fn metal_prefill_ffn<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
    ) -> Option<ferrox_metal::attn::PrefillFfnMetal<'a>> {
        if let Some(moe) = Self::metal_prefill_moe(layer, config) {
            return Some(ferrox_metal::attn::PrefillFfnMetal::Moe(moe));
        }
        if !Self::is_dense_layer(layer) {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        Some(ferrox_metal::attn::PrefillFfnMetal::Dense {
            gate: ex.gate.mul_mm_sg_launch()?,
            up: ex.up.mul_mm_sg_launch()?,
            down: ex.down.mul_mm_sg_launch()?,
        })
    }

    /// Length of a consecutive run of Metal prefill-stack layers from
    /// `start`, or `None` when fewer than two layers qualify.
    #[cfg(feature = "metal")]
    fn metal_prefill_dense_stack_run_len(
        &self,
        start: usize,
        start_pos: usize,
        batch_size: usize,
        kv_caches: &[KvCache],
        metal_kvs: Option<&[ferrox_metal::attn::MetalKvBuffers]>,
    ) -> Option<usize> {
        // See `layer_supports_metal_attn`: gpt-oss stays on CPU.
        if self.gpt_oss.is_some() {
            return None;
        }
        let metal_kvs = metal_kvs?;
        let mut run = 0usize;
        for li in start..self.layers.len() {
            let layer = &self.layers[li];
            let cache = &kv_caches[li];
            if !self.metal_prefill_dense_swa_fits(li, start_pos, batch_size) {
                break;
            }
            if metal_kvs[li].seq_len != cache.seq_len || start_pos != cache.seq_len {
                break;
            }
            let ok = layer.attn.q_proj.mul_mm_sg_launch().is_some()
                && layer.attn.k_proj.mul_mm_sg_launch().is_some()
                && layer.attn.v_proj.mul_mm_sg_launch().is_some()
                && layer.attn.o_proj.mul_mm_sg_launch().is_some()
                && Self::metal_prefill_ffn(layer, &self.config).is_some();
            if !ok {
                break;
            }
            run += 1;
        }
        (run >= 2).then_some(run)
    }

    /// Try [`ferrox_metal::attn::launch_prefill_dense_stack`] for
    /// `run_len` layers starting at `start`. Advances host + Metal KV
    /// on success.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn try_metal_prefill_dense_stack(
        &self,
        start: usize,
        run_len: usize,
        hidden_batch: &[f32],
        start_pos: usize,
        batch_size: usize,
        n_heads: usize,
        metal_kvs: &mut [ferrox_metal::attn::MetalKvBuffers],
        kv_caches: &mut [KvCache],
    ) -> Option<Vec<f32>> {
        let gelu = matches!(
            self.config.ffn_activation,
            crate::config::FfnActivation::Gelu
        );
        let mut prefill_layers = Vec::with_capacity(run_len);
        let mut rope_thetas = Vec::with_capacity(run_len);
        for li in start..start + run_len {
            let layer = &self.layers[li];
            let ffn = Self::metal_prefill_ffn(layer, &self.config)?;
            if matches!(ffn, ferrox_metal::attn::PrefillFfnMetal::Dense { .. }) {
                layer.moe.record_activations(&[0]);
            }
            let (q, k, v, o) = (
                layer.attn.q_proj.mul_mm_sg_launch()?,
                layer.attn.k_proj.mul_mm_sg_launch()?,
                layer.attn.v_proj.mul_mm_sg_launch()?,
                layer.attn.o_proj.mul_mm_sg_launch()?,
            );
            prefill_layers.push(ferrox_metal::attn::PrefillDenseLayerMetal {
                attn_norm_w: &layer.attn.norm_weight,
                ffn_norm_w: &layer.moe.norm_weight,
                q,
                k,
                v,
                o,
                ffn,
                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                extras: self.metal_attn_extras(layer),
                layer_idx: li as u32,
            });
            rope_thetas.push(self.config.layer_rope_theta(li));
        }
        let kvs = &mut metal_kvs[start..start + run_len];
        let h_out = ferrox_metal::attn::launch_prefill_dense_stack(
            hidden_batch,
            &prefill_layers,
            kvs,
            n_heads,
            batch_size,
            self.metal_rope(),
            &rope_thetas,
            self.config.rope_freqs.as_deref(),
            start_pos,
            self.config.rms_norm_eps,
            gelu,
            self.config.attn_logit_softcap,
        )
        .ok()?;
        for cache in &mut kv_caches[start..start + run_len] {
            cache
                .advance_len(batch_size)
                .expect("unbounded/planned KvCache growth is infallible");
        }
        Some(h_out)
    }

    /// MoE layer eligible for resident Metal decode (attn+router+experts
    /// without host residual ping-pong). Requires SwiGLU, no shared
    /// experts, Resident expert backing, and Metal router/QKV/O.
    #[cfg(feature = "metal")]
    fn layer_supports_metal_moe_resident(layer: &LayerWeights, config: &ModelConfig) -> bool {
        !Self::is_dense_layer(layer)
            && layer.moe.shared_experts.is_empty()
            && matches!(
                config.ffn_activation,
                crate::config::FfnActivation::Swiglu | crate::config::FfnActivation::SwigluFused
            )
            && matches!(layer.moe.experts, ExpertBacking::Resident(_))
            && Self::metal_matvec_launch(&layer.moe.router).is_some()
            && Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// One Metal CB for all top-k routed experts (weighted sum). Returns
    /// `None` if any expert lacks a Metal launch (caller falls back).
    #[cfg(feature = "metal")]
    fn try_metal_moe_topk(
        layer: &LayerWeights,
        normed2: &[f32],
        decision: &ferrox_moe::RoutingDecision,
    ) -> Option<Vec<f32>> {
        if decision.expert_ids.is_empty() {
            return Some(vec![0f32; normed2.len()]);
        }
        // Build launches while holding each expert briefly; collect owned
        // weight refs via with_expert into temporary MatvecLaunch list.
        let mut launches: Vec<ferrox_metal::gpu::MoeExpertLaunch<'_>> =
            Vec::with_capacity(decision.expert_ids.len());
        // Lifetime: MatvecLaunch borrows WeightMatrix bytes that live in
        // layer.moe for the duration of this call. Collect via a scoped
        // approach — we need all launches alive together.
        // Use indices + rebuild inside a single with_experts loop.
        struct Pending {
            eid: usize,
            weight: f32,
        }
        let pending: Vec<Pending> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| Pending { eid, weight: w })
            .collect();

        // Validate all experts have Metal launches first.
        for p in &pending {
            let ok = layer.moe.with_expert(p.eid, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            });
            if !ok {
                return None;
            }
        }

        // Hold expert refs: Resident experts are in a Vec; with_expert
        // only borrows one at a time. For Resident backing we can get
        // all launches by indexing once.
        match &layer.moe.experts {
            ExpertBacking::Resident(experts) => {
                for p in &pending {
                    let ex = &experts[p.eid];
                    launches.push(ferrox_metal::gpu::MoeExpertLaunch {
                        gate: Self::metal_matvec_launch(&ex.gate)?,
                        up: Self::metal_matvec_launch(&ex.up)?,
                        down: Self::metal_matvec_launch(&ex.down)?,
                        weight: p.weight,
                    });
                }
            }
            ExpertBacking::Stored { .. } => {
                // Streaming experts: fall back (can't hold all refs easily).
                return None;
            }
        }

        match ferrox_metal::gpu::launch_moe_topk_swiglu(normed2, &launches) {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("ferrox: Metal MoE top-k fuse failed, falling back: {e}");
                None
            }
        }
    }

    /// Contiguous Q4_0 expert planes for llama-style `mul_mv_id` MoE.
    #[cfg(feature = "metal")]
    fn moe_packed_q4(moe: &MoeWeights) -> Option<ferrox_metal::gpu::MoePackedQ4<'_>> {
        moe.packed_q4.as_ref().map(MoePackedQ4Planes::view)
    }

    /// Prefill MoE FFN on Metal: host route over T, then one packed-id CB
    /// (`launch_moe_prefill_q4_0`). Shared experts (if any) run as dense
    /// batch FFN on the host/GPU path afterwards — not through `mul_mm_id`.
    /// Returns FFN outs `[T, H]` or `None`.
    #[cfg(feature = "metal")]
    fn try_metal_moe_prefill_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        if batch_size == 0
            || !ferrox_core::metal_dense_enabled()
            || !ferrox_metal::attn::metal_moe_resident_enabled()
            || !matches!(
                config.ffn_activation,
                crate::config::FfnActivation::Swiglu | crate::config::FfnActivation::SwigluFused
            )
            || !matches!(config.moe.gating, ferrox_moe::GatingFunction::Softmax)
            || config.moe.expert_group_count.is_some()
            // The GPU router kernels take router weights and nothing
            // else: no `exp_probs_b` input, no `expert_weights_scale`
            // uniform. A layer carrying either must stay on the CPU
            // router rather than be routed without them.
            || layer.moe.exp_probs_bias.is_some()
            || config.moe.expert_weights_scale != 1.0
        {
            return None;
        }
        let ExpertBacking::Resident(_) = &layer.moe.experts else {
            return None;
        };
        let packed = Self::moe_packed_q4(&layer.moe)?;
        let top_k = config.moe.n_experts_active;
        if top_k == 0 || top_k > 8 || packed.hidden_rows != hidden_dim {
            return None;
        }
        let n_experts = layer.moe.n_experts().max(1);
        let mut ids = Vec::with_capacity(batch_size * top_k);
        let mut route = Vec::with_capacity(batch_size * top_k);
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = route_top_k(logits, top_k, config.moe.gating, config.moe.norm_topk_prob);
            layer.moe.record_activations(&decision.expert_ids);
            if decision.expert_ids.len() != top_k {
                return None;
            }
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                ids.push(eid as i32);
                route.push(w);
            }
        }
        let mut out = match ferrox_metal::gpu::launch_moe_prefill_q4_0(
            normed2_batch,
            batch_size,
            &packed,
            &ids,
            &route,
            top_k,
        ) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("ferrox: Metal MoE prefill failed, CPU fallback: {e}");
                return None;
            }
        };
        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut out,
        );
        Some(out)
    }

    /// Shared expert as dense batch FFN (llama qwen2moe: not through
    /// `mul_mat_id`). Optional sigmoid gate scales per token.
    fn accumulate_shared_experts_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        acc: &mut [f32],
    ) {
        for shex in &layer.moe.shared_experts {
            // Prefer one Metal FFN CB (gate∥up→SiLU→down) over three
            // `apply_batch` round-trips — Qwen shexp is 4× routed width.
            #[cfg(feature = "metal")]
            let down = if ferrox_core::metal_dense_enabled() && batch_size >= 4 {
                match (
                    shex.gate.mul_mm_sg_launch(),
                    shex.up.mul_mm_sg_launch(),
                    shex.down.mul_mm_sg_launch(),
                ) {
                    (Some(g), Some(u), Some(d)) => {
                        ferrox_metal::gpu::launch_dense_ffn_swiglu_batch(
                            &g,
                            &u,
                            &d,
                            normed2_batch,
                            batch_size,
                            false,
                        )
                        .ok()
                    }
                    _ => None,
                }
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let down: Option<Vec<f32>> = None;
            // Without `metal` the binding above is a literal `None`; the
            // fallback is the only arm and clippy flags the unwrap.
            #[cfg_attr(not(feature = "metal"), allow(clippy::unnecessary_literal_unwrap))]
            let down = down.unwrap_or_else(|| {
                let ffn_acts = shex.gate.quantize_batch_acts(normed2_batch, batch_size);
                let gate =
                    shex.gate
                        .apply_batch_with_acts(normed2_batch, batch_size, ffn_acts.as_ref());
                let up =
                    shex.up
                        .apply_batch_with_acts(normed2_batch, batch_size, ffn_acts.as_ref());
                let activated = ferrox_core::matmul::swiglu(&gate, &up);
                shex.down.apply_batch(&activated, batch_size)
            });
            if let Some(gate_w) = &layer.moe.shared_expert_gate {
                for b in 0..batch_size {
                    let x = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    let logit: f32 = gate_w.iter().zip(x.iter()).map(|(g, v)| g * v).sum();
                    let scale = 1.0 / (1.0 + (-logit).exp());
                    let out = &down[b * hidden_dim..(b + 1) * hidden_dim];
                    let row = &mut acc[b * hidden_dim..(b + 1) * hidden_dim];
                    for (a, &o) in row.iter_mut().zip(out.iter()) {
                        *a += scale * o;
                    }
                }
            } else {
                for (a, &o) in acc.iter_mut().zip(down.iter()) {
                    *a += o;
                }
            }
        }
    }

    /// Phase-2 of resident MoE decode: experts on GPU `x2`, add into GPU `h`.
    #[cfg(feature = "metal")]
    fn try_metal_moe_experts_resident(
        layer: &LayerWeights,
        decision: &ferrox_moe::RoutingDecision,
    ) -> Option<()> {
        if decision.expert_ids.is_empty() {
            return Some(());
        }
        let pending: Vec<(usize, f32)> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| (eid, w))
            .collect();
        for &(eid, _) in &pending {
            let ok = layer.moe.with_expert(eid, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            });
            if !ok {
                return None;
            }
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let mut launches = Vec::with_capacity(pending.len());
        for &(eid, weight) in &pending {
            let ex = &experts[eid];
            launches.push(ferrox_metal::gpu::MoeExpertLaunch {
                gate: Self::metal_matvec_launch(&ex.gate)?,
                up: Self::metal_matvec_launch(&ex.up)?,
                down: Self::metal_matvec_launch(&ex.down)?,
                weight,
            });
        }
        match ferrox_metal::attn::launch_moe_decode_experts(&launches) {
            Ok(()) => Some(()),
            Err(e) => {
                eprintln!("ferrox: Metal MoE experts failed, falling back: {e}");
                None
            }
        }
    }

    /// Append host [`KvCache`] positions that Metal already holds but host
    /// skipped (dense-stack fast path). No-op when `cache.seq_len` is caught up.
    #[cfg(feature = "metal")]
    fn catch_up_host_kv_from_metal(mkv: &ferrox_metal::attn::MetalKvBuffers, cache: &mut KvCache) {
        if cache.seq_len >= mkv.seq_len {
            return;
        }
        let start = cache.seq_len;
        let n = mkv.seq_len - start;
        let (k, v) = mkv.tokens_host(start, n);
        let per = cache.n_kv_heads * cache.head_dim;
        for i in 0..n {
            let off = i * per;
            cache
                .push(&k[off..off + per], &v[off..off + per])
                .expect("unbounded/planned KvCache growth is infallible");
        }
    }

    /// Pull every layer's Metal-ahead suffix into `kv_caches` (prefix-cache
    /// store, continuous-batch / CPU readers). Safe no-op without Metal KV.
    #[cfg(feature = "metal")]
    pub fn sync_metal_attn_kv_to_host(&self, kv_caches: &mut [KvCache]) {
        assert_eq!(kv_caches.len(), self.layers.len());
        let Ok(guard) = self.metal_attn_kv.lock() else {
            return;
        };
        let Some(metal_kvs) = guard.as_ref() else {
            return;
        };
        if metal_kvs.len() != kv_caches.len() {
            return;
        }
        for (mkv, cache) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
            Self::catch_up_host_kv_from_metal(mkv, cache);
        }
    }

    /// GQA decode reduction for one token. Uses the CUDA `gqa_decode`
    /// kernel when built with `--features cuda` and `FERROX_CUDA_GQA=1`
    /// (falling back to the host path on any launch error), else the
    /// portable [`causal_gqa_attention`]. With residency enabled the
    /// K/V append stays in [`ferrox_cuda::attn::CudaKvBuffers`] so only
    /// Q crosses the bus per call (plus a prefix refresh on append).
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(
        &self,
        layer: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        {
            if cuda_gqa_enabled() {
                match ferrox_cuda::attn::launch_gqa_decode_resident(
                    layer, q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                ) {
                    Ok(out) => return out,
                    Err(e) => {
                        eprintln!(
                            "ferrox: CUDA GQA resident decode failed, trying full upload: {e}"
                        );
                    }
                }
                match ferrox_cuda::attn::launch_gqa_decode(
                    q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                ) {
                    Ok(out) => return out,
                    Err(e) => {
                        eprintln!("ferrox: CUDA GQA decode failed, host fallback: {e}");
                    }
                }
            }
        }
        let _ = layer;
        causal_gqa_attention_softcap(
            q,
            k,
            v,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            self.config.attn_logit_softcap,
        )
    }

    /// Runs one decode step for `token_id` at position `pos`, updating
    /// `kv_caches` (one per layer) in place, and returns the logits over
    /// the (test-scale) vocabulary.
    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        // Clear stale dense-stack activation TLS. MoE scratch buffers are
        // reused across tokens (re-seeded); cleared after lm_head below.
        #[cfg(feature = "metal")]
        ferrox_metal::gpu::clear_resident_activation();

        assert_eq!(kv_caches.len(), self.layers.len());
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;

        #[cfg(feature = "metal")]
        let metal_embd_kind = {
            let metal_path = ferrox_core::metal_dense_enabled()
                && ferrox_metal::attn::metal_attn_enabled()
                && self
                    .layers
                    .iter()
                    .all(|l| self.layer_supports_metal_attn(l))
                && self.layers.iter().all(Self::layer_supports_metal_dense_ffn);
            // Gemma scales the embedding row (`embedding_scale`) — the GPU
            // gather has no scale op, so dequant + scale on the host.
            if metal_path && self.config.embedding_scale.is_none() {
                Self::metal_matvec_launch(&self.embedding)
                    .and_then(|l| ferrox_metal::embd::EmbdKind::from_fn_name(l.fn_name))
            } else {
                None
            }
        };
        // `metal_embd_kind` is only `Some` when `embedding_scale` is
        // `None` (the GPU gather has no scale op), so the empty vector
        // this leaves behind is one the scale would not have touched.
        #[cfg(feature = "metal")]
        let mut hidden = if metal_embd_kind.is_some() {
            Vec::new()
        } else {
            self.embed_token(token_id)
        };
        #[cfg(not(feature = "metal"))]
        let mut hidden = self.embed_token(token_id);
        #[cfg(feature = "cuda")]
        if cuda_gqa_enabled() {
            // Fixed capacity so ensure_layer_kv does not recreate (and
            // wipe) mid-sequence as pos grows.
            const CUDA_KV_CAP: usize = 4096;
            if let Err(e) = ferrox_cuda::attn::ensure_layer_kv(
                self.layers.len(),
                self.config.n_kv_heads,
                self.config.head_dim,
                CUDA_KV_CAP,
            ) {
                eprintln!("ferrox: CUDA KV residency init failed: {e}");
            }
            if pos == 0 {
                ferrox_cuda::attn::clear_layer_kv();
            }
        }

        #[cfg(feature = "metal")]
        let use_metal_attn = ferrox_core::metal_dense_enabled()
            && ferrox_metal::attn::metal_attn_enabled()
            && self
                .layers
                .iter()
                .all(|l| self.layer_supports_metal_attn(l));

        #[cfg(not(feature = "metal"))]
        let use_metal_attn = false;

        let residency = self.expert_residency_plan(use_metal_attn);

        #[cfg(feature = "metal")]
        let mut metal_kv_guard: Option<
            std::sync::MutexGuard<'_, Option<Vec<ferrox_metal::attn::MetalKvBuffers>>>,
        > = if use_metal_attn {
            Some(self.metal_attn_kv.lock().unwrap())
        } else {
            None
        };

        #[cfg(feature = "metal")]
        if let Some(guard) = metal_kv_guard.as_mut() {
            let need = self.layers.len();
            let cap = kv_caches
                .iter()
                .map(|c| c.seq_len.max(pos + 1).saturating_add(256))
                .max()
                .unwrap_or(512)
                .max(512)
                .max(pos + 1);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    if v.len() != need || v.iter().any(|m| m.capacity() < pos + 1) {
                        // Growing / reshaping: preserve Metal-ahead tokens on host first.
                        if v.len() == need {
                            for (m, c) in v.iter().zip(kv_caches.iter_mut()) {
                                Self::catch_up_host_kv_from_metal(m, c);
                            }
                        }
                        true
                    } else if v.iter().all(|m| m.seq_len == pos) {
                        // Metal already holds tokens [0, pos). Host may lag
                        // after dense-stack decode — do not re-upload from host.
                        false
                    } else {
                        // Stale Metal (new request / prefix restore): rebuild from host.
                        true
                    }
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match ferrox_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    // Sync from host after CPU prefill / prefix restore / capacity grow.
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
                        if c.seq_len > 0 && m.upload_from_host(&c.k, &c.v, c.seq_len).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        **guard = Some(bufs);
                    } else {
                        **guard = None;
                    }
                } else {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let mut metal_stack_done = false;
        #[cfg(feature = "metal")]
        let mut final_norm_done_in_stack = false;
        // OLMoE: all MoE layers in one CB (llama graph style).
        #[cfg(feature = "metal")]
        if use_metal_attn
            && ferrox_metal::attn::metal_moe_resident_enabled()
            && matches!(self.config.moe.gating, ferrox_moe::GatingFunction::Softmax)
            && self
                .layers
                .iter()
                .all(|l| Self::layer_supports_metal_moe_resident(l, &self.config))
            && !self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                if let Some(metal_kvs) = guard.as_mut() {
                    if metal_kvs.iter().all(|m| m.seq_len == pos) {
                        let mut moe_layers = Vec::with_capacity(self.layers.len());
                        let mut ok = true;
                        for layer in &self.layers {
                            let ExpertBacking::Resident(_) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let Some(packed) = Self::moe_packed_q4(&layer.moe) else {
                                ok = false;
                                break;
                            };
                            let (Some(q), Some(k), Some(v), Some(o), Some(r)) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&layer.moe.router),
                            ) else {
                                ok = false;
                                break;
                            };
                            moe_layers.push(ferrox_metal::attn::MoeLayerMetal {
                                attn_norm_w: &layer.attn.norm_weight,
                                ffn_norm_w: &layer.moe.norm_weight,
                                q,
                                k,
                                v,
                                o,
                                router: r,
                                packed,
                                extras: self.metal_attn_extras(layer),
                            });
                        }
                        if ok {
                            // Greedy / FERROX_METAL_LOGITS: fold lm_head(+argmax)
                            // like dense stack — download 1×u32 or vocab, skip host.
                            let greedy_gpu = ferrox_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            let out_launch =
                                if greedy_gpu || ferrox_metal::attn::metal_logits_enabled() {
                                    lm_head_gpu_launch
                                } else {
                                    None
                                };
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales embd on host; GPU gather has no scale.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(ferrox_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            if embd_gather.is_none() && hidden.is_empty() {
                                hidden = self.embedding.dequant_row(token_id);
                                if let Some(scale) = self.config.embedding_scale {
                                    for v in hidden.iter_mut() {
                                        *v *= scale;
                                    }
                                }
                            }
                            let seed = if embd_gather.is_some() {
                                ferrox_metal::attn::moe_decode_ensure(hidden_dim)
                            } else {
                                ferrox_metal::attn::moe_decode_seed(&hidden)
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match seed.and_then(|_| {
                                ferrox_metal::attn::launch_moe_decode_stack(
                                    hidden_ref,
                                    &moe_layers,
                                    metal_kvs,
                                    self.config.moe.n_experts_active,
                                    self.config.moe.norm_topk_prob,
                                    n_heads,
                                    self.metal_rope(),
                                    self.config.rope_theta,
                                    self.config.rope_freqs.as_deref(),
                                    pos,
                                    self.config.rms_norm_eps,
                                    Some(&self.final_norm),
                                    out_launch.as_ref(),
                                    greedy_gpu && out_launch.is_some(),
                                    true,
                                    embd_gather.as_ref(),
                                )
                            }) {
                                Ok((out, per_layer_ids)) => {
                                    for (layer, ids) in self.layers.iter().zip(per_layer_ids.iter())
                                    {
                                        if !ids.is_empty() {
                                            layer.moe.record_activations(ids);
                                        }
                                    }
                                    if out_launch.is_some() {
                                        #[cfg(feature = "metal")]
                                        ferrox_metal::gpu::clear_resident_activation();
                                        return out;
                                    }
                                    hidden = out;
                                    final_norm_done_in_stack = true;
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "ferrox: Metal MoE stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        if !metal_stack_done
            && use_metal_attn
            && self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                let mut clear_metal_after_stack = false;
                if let Some(metal_kvs) = guard.as_mut() {
                    let seq_ok = metal_kvs.iter().all(|m| m.seq_len == pos);
                    if seq_ok {
                        // Build launches only for resident dense experts (Llama path).
                        let mut dense_layers = Vec::with_capacity(self.layers.len());
                        let mut ok = true;
                        for (li, layer) in self.layers.iter().enumerate() {
                            let ExpertBacking::Resident(experts) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let ex = &experts[0];
                            let (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&ex.gate),
                                Self::metal_matvec_launch(&ex.up),
                                Self::metal_matvec_launch(&ex.down),
                            ) else {
                                ok = false;
                                break;
                            };
                            dense_layers.push(ferrox_metal::attn::DenseLayerMetal {
                                attn_norm_w: &layer.attn.norm_weight,
                                ffn_norm_w: &layer.moe.norm_weight,
                                q,
                                k,
                                v,
                                o,
                                gate: g,
                                up: u,
                                down: d,
                                extras: self.metal_attn_extras(layer),
                                rope_theta: {
                                    let t = self.config.layer_rope_theta(li);
                                    (t != self.config.rope_theta).then_some(t)
                                },
                                window: self.config.layer_sliding_window(li),
                                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                            });
                        }
                        if ok {
                            // Prefer greedy GPU argmax-in-stack (1×u32 download)
                            // when generate marked this thread for temperature<=0.
                            // Else opt-in FERROX_METAL_LOGITS downloads full vocab
                            // (often slower). Default: host lm_head after hidden.
                            let greedy_gpu = ferrox_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            // Prefer greedy GPU argmax-in-stack (1×u32 download)
                            // when generate marked this thread for temperature<=0.
                            // Else opt-in FERROX_METAL_LOGITS downloads full vocab
                            // (often slower). Default: host lm_head after hidden.
                            let out_launch =
                                if greedy_gpu || ferrox_metal::attn::metal_logits_enabled() {
                                    lm_head_gpu_launch
                                } else {
                                    None
                                };
                            // Pass final_norm_w when: (1) lm_head runs in stack (out_launch),
                            // OR (2) lm_head will route to GPU after stack (lm_head_gpu_launch
                            // but no out_launch) so we can skip download→reupload via TLS.
                            let final_norm_w =
                                if out_launch.is_some() || lm_head_gpu_launch.is_some() {
                                    Some(self.final_norm.as_slice())
                                } else {
                                    None
                                };
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales the embedding row on the host
                            // (`hidden` already carries sqrt(hidden_dim));
                            // the GPU gather has no scale op — skip it.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(ferrox_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match ferrox_metal::attn::launch_decode_dense_stack(
                                hidden_ref,
                                &dense_layers,
                                metal_kvs,
                                n_heads,
                                self.metal_rope(),
                                self.config.rope_theta,
                                self.config.rope_freqs.as_deref(),
                                pos,
                                self.config.rms_norm_eps,
                                final_norm_w,
                                out_launch.as_ref(),
                                greedy_gpu && out_launch.is_some(),
                                embd_gather.as_ref(),
                                matches!(
                                    self.config.ffn_activation,
                                    crate::config::FfnActivation::Gelu
                                ),
                            ) {
                                Ok(out) => {
                                    // Metal KV advanced in-place. Skip host
                                    // last_token_host+push — host may lag until
                                    // sync_metal_attn_kv_to_host / CPU fallback.
                                    // Dense stack has no MoE routing; skip
                                    // per-layer hotness atomics on the hot path.
                                    if out_launch.is_some() {
                                        // Stack returned logits or [argmax id] —
                                        // skip host final_norm/lm_head. Clear TLS.
                                        #[cfg(feature = "metal")]
                                        ferrox_metal::gpu::clear_resident_activation();
                                        return out;
                                    }
                                    // Stack downloaded hidden (possibly normalized if
                                    // final_norm_w was Some). Track whether host should
                                    // skip final_norm.
                                    final_norm_done_in_stack = final_norm_w.is_some();
                                    hidden = out;
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "ferrox: Metal dense stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                    // Preserve any prior Metal-ahead tokens on host
                                    // before dropping the device buffers.
                                    for (m, c) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
                                        Self::catch_up_host_kv_from_metal(m, c);
                                    }
                                    clear_metal_after_stack = true;
                                }
                            }
                        }
                    }
                }
                if clear_metal_after_stack {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let run_cpu_layers = !metal_stack_done;
        #[cfg(not(feature = "metal"))]
        let run_cpu_layers = true;

        // When true, residual lives in Metal MoE scratch — host `hidden` is stale.
        #[cfg(feature = "metal")]
        let mut metal_moe_resident = false;

        if run_cpu_layers {
            for (l, (layer, cache)) in self.layers.iter().zip(kv_caches.iter_mut()).enumerate() {
                // --- attention block ---
                #[cfg(feature = "metal")]
                if metal_moe_resident
                    && (!Self::layer_supports_metal_moe_resident(layer, &self.config)
                        || self.layer_needs_metal_stack(layer, l))
                {
                    if let Some(h) = ferrox_metal::attn::moe_decode_take_hidden() {
                        hidden = h;
                    }
                    metal_moe_resident = false;
                }

                #[cfg(feature = "metal")]
                let normed = if metal_moe_resident {
                    // Residual is on-device; host rms_norm would use stale hidden.
                    Vec::new()
                } else {
                    rms_norm(&hidden, &layer.attn.norm_weight, self.config.rms_norm_eps)
                };
                #[cfg(not(feature = "metal"))]
                let normed = rms_norm(&hidden, &layer.attn.norm_weight, self.config.rms_norm_eps);

                #[cfg(feature = "metal")]
                {
                    let mut did_metal_attn = false;
                    let mut did_metal_dense = false;
                    let mut did_metal_moe = false;
                    let mut clear_metal_kv = false;
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // Metal-authoritative: host may lag after dense-stack skip.
                            // Stack-only features (SWA / sandwich norms / GeGLU /
                            // per-layer theta) are NOT encoded by the per-layer
                            // launches — those layers must go to CPU here.
                            if metal_kvs[l].seq_len == pos
                                && !self.layer_needs_metal_stack(layer, l)
                            {
                                if let (Some(q_l), Some(k_l), Some(v_l), Some(o_l)) = (
                                    Self::metal_matvec_launch(&layer.attn.q_proj),
                                    Self::metal_matvec_launch(&layer.attn.k_proj),
                                    Self::metal_matvec_launch(&layer.attn.v_proj),
                                    Self::metal_matvec_launch(&layer.attn.o_proj),
                                ) {
                                    // Full dense layer on one CB when FFN is Metal-capable.
                                    if Self::layer_supports_metal_dense_ffn(layer) {
                                        let dense_ok = layer.moe.with_expert(0, |ex| {
                                        let (Some(g_l), Some(u_l), Some(d_l)) = (
                                            Self::metal_matvec_launch(&ex.gate),
                                            Self::metal_matvec_launch(&ex.up),
                                            Self::metal_matvec_launch(&ex.down),
                                        ) else {
                                            return false;
                                        };
                                        match ferrox_metal::attn::launch_decode_dense_layer(
                                            &hidden,
                                            &layer.attn.norm_weight,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            &layer.moe.norm_weight,
                                            &g_l,
                                            &u_l,
                                            &d_l,
                                            n_heads,
                                            self.metal_rope(),
                                            self.config.rope_theta,
                                            self.config.rope_freqs.as_deref(),
                                            pos,
                                            self.config.rms_norm_eps,
                                            &self.metal_attn_extras(layer),
                                        ) {
                                            Ok(new_h) => {
                                                // Catch up any dense-stack lag + this token.
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                layer.moe.record_activations(&[0]);
                                                hidden = new_h;
                                                true
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "ferrox: Metal dense layer failed, CPU fallback: {e}"
                                                );
                                                false
                                            }
                                        }
                                    });
                                        if dense_ok {
                                            did_metal_dense = true;
                                            did_metal_attn = true;
                                        } else if metal_kvs[l].seq_len != cache.seq_len {
                                            // Dense path may have advanced Metal KV before failing.
                                            Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                                            clear_metal_kv = true;
                                        }
                                    }

                                    // Resident MoE: attn+router on GPU, host top-k only,
                                    // then batched experts — no hidden download/upload.
                                    if !did_metal_dense
                                        && !clear_metal_kv
                                        && ferrox_metal::attn::metal_moe_resident_enabled()
                                        && Self::layer_supports_metal_moe_resident(
                                            layer,
                                            &self.config,
                                        )
                                    {
                                        if let Some(router_l) =
                                            Self::metal_matvec_launch(&layer.moe.router)
                                        {
                                            let seed_ok = if metal_moe_resident {
                                                true
                                            } else {
                                                match ferrox_metal::attn::moe_decode_seed(&hidden) {
                                                    Ok(()) => {
                                                        metal_moe_resident = true;
                                                        true
                                                    }
                                                    Err(e) => {
                                                        eprintln!(
                                                            "ferrox: Metal MoE seed failed: {e}"
                                                        );
                                                        false
                                                    }
                                                }
                                            };
                                            if seed_ok {
                                                // Prefer one-CB fused path (GPU top-k + packed experts).
                                                // See
                                                // `layer_supports_metal_moe_resident`:
                                                // the fused decode kernel
                                                // routes on the GPU and
                                                // has no `exp_probs_b` /
                                                // `expert_weights_scale`
                                                // input either.
                                                let fused_ok = matches!(
                                                    self.config.moe.gating,
                                                    ferrox_moe::GatingFunction::Softmax
                                                ) && layer
                                                    .moe
                                                    .exp_probs_bias
                                                    .is_none()
                                                    && self.config.moe.expert_weights_scale == 1.0
                                                    && match &layer.moe.experts {
                                                        ExpertBacking::Resident(_) => {
                                                            if let Some(packed) =
                                                                Self::moe_packed_q4(&layer.moe)
                                                            {
                                                                match ferrox_metal::attn::launch_moe_decode_layer_fused(
                                                                &layer.attn.norm_weight,
                                                                &q_l,
                                                                &k_l,
                                                                &v_l,
                                                                &o_l,
                                                                &mut metal_kvs[l],
                                                                &layer.moe.norm_weight,
                                                                &router_l,
                                                                &packed,
                                                                self.config.moe.n_experts_active,
                                                                self.config.moe.norm_topk_prob,
                                                                n_heads,
                                                                self.metal_rope(),
                                                                self.config.rope_theta,
                                                                self.config.rope_freqs.as_deref(),
                                                                pos,
                                                                self.config.rms_norm_eps,
                                                                &self.metal_attn_extras(layer),
                                                            ) {
                                                                Ok(ids) => {
                                                                    layer.moe.record_activations(&ids);
                                                                    did_metal_moe = true;
                                                                    did_metal_attn = true;
                                                                    true
                                                                }
                                                                Err(e) => {
                                                                    eprintln!(
                                                                        "ferrox: Metal MoE fused layer failed: {e}"
                                                                    );
                                                                    false
                                                                }
                                                            }
                                                            } else {
                                                                false
                                                            }
                                                        }
                                                        _ => false,
                                                    };

                                                if !fused_ok {
                                                    match ferrox_metal::attn::launch_moe_decode_pre(
                                                        &layer.attn.norm_weight,
                                                        &q_l,
                                                        &k_l,
                                                        &v_l,
                                                        &o_l,
                                                        &mut metal_kvs[l],
                                                        &layer.moe.norm_weight,
                                                        &router_l,
                                                        n_heads,
                                                        self.metal_rope(),
                                                        self.config.rope_theta,
                                                        self.config.rope_freqs.as_deref(),
                                                        pos,
                                                        self.config.rms_norm_eps,
                                                        &self.metal_attn_extras(layer),
                                                    ) {
                                                        Ok(logits) => {
                                                            let decision = route_top_k(
                                                                &logits,
                                                                self.config.moe.n_experts_active,
                                                                self.config.moe.gating,
                                                                self.config.moe.norm_topk_prob,
                                                            );
                                                            layer.moe.record_activations(
                                                                &decision.expert_ids,
                                                            );
                                                            if let Some(()) = Self::try_metal_moe_experts_resident(
                                                            layer,
                                                            &decision,
                                                        ) {
                                                            did_metal_moe = true;
                                                            did_metal_attn = true;
                                                        } else if let Some(h) =
                                                            ferrox_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                            metal_moe_resident = false;
                                                            // KV already advanced; finish FFN on host.
                                                            let normed2 = rms_norm(
                                                                &hidden,
                                                                &layer.moe.norm_weight,
                                                                self.config.rms_norm_eps,
                                                            );
                                                            let ffn_out = Self::combine_ffn_outputs_for_position(
                                                                layer,
                                                                &normed2,
                                                                &logits,
                                                                &self.config,
                                                                hidden_dim,
                                                                residency.as_ref().map(|p| p.layer_plan(l)),
                                                            );
                                                            for (h, f) in
                                                                hidden.iter_mut().zip(ffn_out.iter())
                                                            {
                                                                *h += f;
                                                            }
                                                            did_metal_attn = true;
                                                            did_metal_moe = true; // skip second FFN
                                                        }
                                                        }
                                                        Err(e) => {
                                                            eprintln!(
                                                            "ferrox: Metal MoE pre failed, fallback: {e}"
                                                        );
                                                            if let Some(h) =
                                                            ferrox_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                        }
                                                            metal_moe_resident = false;
                                                            if metal_kvs[l].seq_len != cache.seq_len
                                                            {
                                                                Self::catch_up_host_kv_from_metal(
                                                                    &metal_kvs[l],
                                                                    cache,
                                                                );
                                                                clear_metal_kv = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !did_metal_dense && !did_metal_moe && !clear_metal_kv {
                                        match ferrox_metal::attn::launch_decode_attn_block(
                                            &normed,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            n_heads,
                                            self.metal_rope(),
                                            self.config.rope_theta,
                                            self.config.rope_freqs.as_deref(),
                                            pos,
                                            &self.metal_attn_extras(layer),
                                            self.config.rms_norm_eps,
                                        ) {
                                            Ok(projected) => {
                                                // Keep Metal KV authoritative — skip per-layer
                                                // host catch-up (dense-stack style). Host is
                                                // flushed on CPU fallback / prefix sync.
                                                for (h, p) in
                                                    hidden.iter_mut().zip(projected.iter())
                                                {
                                                    *h += p;
                                                }
                                                did_metal_attn = true;
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                "ferrox: Metal attn block failed, CPU fallback: {e}"
                                            );
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                clear_metal_kv = true;
                                            }
                                        }
                                    }
                                }
                            } else if metal_kvs[l].seq_len > cache.seq_len {
                                // Leaving Metal path: host must see full KV for CPU attn.
                                Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                            }
                        }
                        if clear_metal_kv {
                            **guard = None;
                        }
                    }
                    if did_metal_attn {
                        if !did_metal_dense && !did_metal_moe {
                            let normed2 =
                                rms_norm(&hidden, &layer.moe.norm_weight, self.config.rms_norm_eps);
                            let ffn_out = Self::run_ffn_block(
                                layer,
                                &normed2,
                                &self.config,
                                hidden_dim,
                                residency.as_ref().map(|p| p.layer_plan(l)),
                            );
                            for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
                                *h += f;
                            }
                        }
                        continue;
                    }
                }

                let (mut q, mut k, mut v) = {
                    #[cfg(any(feature = "cuda", feature = "metal"))]
                    {
                        if let Some(mut outs) = ferrox_core::WeightMatrix::apply_gpu_multi(
                            &[&layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj],
                            &normed,
                        ) {
                            let v = outs.pop().unwrap();
                            let k = outs.pop().unwrap();
                            let q = outs.pop().unwrap();
                            (q, k, v)
                        } else {
                            ferrox_core::weight_matrix::WeightMatrix::apply_three(
                                &layer.attn.q_proj,
                                &layer.attn.k_proj,
                                &layer.attn.v_proj,
                                &normed,
                            )
                        }
                    }
                    #[cfg(not(any(feature = "cuda", feature = "metal")))]
                    {
                        ferrox_core::weight_matrix::WeightMatrix::apply_three(
                            &layer.attn.q_proj,
                            &layer.attn.k_proj,
                            &layer.attn.v_proj,
                            &normed,
                        )
                    }
                };

                if let Some(bias) = &layer.attn.q_bias {
                    for (x, b) in q.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
                if let Some(bias) = &layer.attn.k_bias {
                    for (x, b) in k.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
                if let Some(bias) = &layer.attn.v_bias {
                    for (x, b) in v.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }

                if let Some(q_norm) = &layer.attn.q_norm {
                    q = self.apply_qk_norm(&q, q_norm);
                }
                if let Some(k_norm) = &layer.attn.k_norm {
                    k = self.apply_qk_norm(&k, k_norm);
                }
                self.apply_rope_attn_factor(&mut q, &mut k);

                for h in 0..n_heads {
                    self.apply_rope_head_layer(&mut q[h * head_dim..(h + 1) * head_dim], pos, l);
                }
                for h in 0..n_kv_heads {
                    self.apply_rope_head_layer(&mut k[h * head_dim..(h + 1) * head_dim], pos, l);
                }
                // When an architecture overrides the score scale (llama.cpp
                // Gemma scales Q then calls build_attn with 1.0), compensate
                // for the kernel's built-in 1/sqrt(head_dim) so the net
                // score scale equals `attention_scale`.
                if let Some(scale) = self.config.attention_scale {
                    let compensate = scale * (head_dim as f32).sqrt();
                    for v in q.iter_mut() {
                        *v *= compensate;
                    }
                }

                cache
                    .push(&k, &v)
                    .expect("unbounded/planned KvCache growth is infallible");

                let oai = self.gpt_oss.as_ref().map(|g| &g.layers[l]);
                let attn_out = match (oai, self.config.layer_sliding_window(l)) {
                    (Some(oai), window) => ferrox_core::causal_gqa_attention_sinks(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                        window,
                        &oai.attn_sinks,
                    ),
                    (None, Some(window)) => causal_gqa_attention_windowed_softcap(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                        window,
                        self.config.attn_logit_softcap,
                    ),
                    (None, None) => self.gqa_attention(
                        l,
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                    ),
                };
                let mut projected = layer.attn.o_proj.apply(&attn_out);
                if let Some(oai) = oai {
                    for (x, b) in projected.iter_mut().zip(oai.o_bias.iter()) {
                        *x += b;
                    }
                }
                if let Some(post) = &layer.attn.post_attn_norm {
                    projected = rms_norm(&projected, post, self.config.rms_norm_eps);
                }

                for (h, p) in hidden.iter_mut().zip(projected.iter()) {
                    *h += p;
                }

                // --- MoE FFN block ---
                let normed2 = rms_norm(&hidden, &layer.moe.norm_weight, self.config.rms_norm_eps);
                let mut ffn_out = match oai {
                    Some(oai) => Self::gpt_oss_ffn(layer, oai, &normed2, &self.config, hidden_dim),
                    None => Self::run_ffn_block(
                        layer,
                        &normed2,
                        &self.config,
                        hidden_dim,
                        residency.as_ref().map(|p| p.layer_plan(l)),
                    ),
                };
                if let Some(post) = &layer.attn.post_ffn_norm {
                    ffn_out = rms_norm(&ffn_out, post, self.config.rms_norm_eps);
                }
                for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
                    *h += f;
                }
            }
        } // run_cpu_layers

        #[cfg(feature = "metal")]
        if metal_moe_resident {
            if let Some(h) = ferrox_metal::attn::moe_decode_take_hidden() {
                hidden = h;
            }
        }

        // If Metal stack already ran final_norm, hidden is normalized; else
        // normalize here.
        #[cfg(feature = "metal")]
        let final_normed = if final_norm_done_in_stack {
            hidden.clone()
        } else {
            rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)
        };
        #[cfg(not(feature = "metal"))]
        let final_normed = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);

        let logits = self.logits_from_normed(&final_normed);
        // Clear dense-stack activation TLS after lm_head (may have consumed it).
        // Keep MoE scratch buffers alive across tokens — `moe_decode_seed`
        // overwrites `h` each token; clearing here forced full realloc.
        #[cfg(feature = "metal")]
        ferrox_metal::gpu::clear_resident_activation();
        logits
    }

    /// Same computation as `forward_token`, but each layer's K/V cache
    /// is a `PagedKvCache` (block-table-indexed into a per-layer
    /// `PagedKvStore`) instead of a `KvCache`'s contiguous buffer --
    /// exercises the paged attention kernel in a real decode loop
    /// instead of only in isolation. `kv_caches`/`stores` are parallel
    /// per-layer arrays, mirroring `forward_token`'s `kv_caches: &mut
    /// [KvCache]`. Must produce bit-identical output to `forward_token`
    /// given stores sized so no layer ever exhausts its blocks --
    /// pinned by
    /// `forward_token_paged_matches_forward_token_bit_identical` and,
    /// per attention arm, by
    /// `every_paged_attention_arm_is_bit_identical_to_its_contiguous_twin`.
    ///
    /// This used to refuse gpt-oss outright, because the paged kernel
    /// had no attention-sink term and no sliding-window arm and would
    /// have answered differently from the contiguous path without
    /// saying so. It now mirrors all three arms of that dispatch, so
    /// the refusal is gone rather than merely relaxed.
    pub fn forward_token_paged(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.layers.len());
        assert_eq!(stores.layer_count(), self.layers.len());
        // All layers advance or none do. Pushing per layer with `?` and
        // failing at layer 3 of 4 leaves layers 0..2 holding a position
        // the rest do not, and nothing downstream reports it: the next
        // step simply attends over a shorter history in the tail
        // layers. Reserving one position everywhere first turns that
        // into a clean refusal.
        //
        // The guards span the check AND the push for the same reason
        // the prefill path holds them: otherwise another request takes
        // the blocks in between.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, 1) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            // Reserve by taking the blocks now, so the per-layer pushes
            // below cannot fail. `PagedKvCache::reserve` grows the block
            // table without advancing `seq_len`, leaving each push a
            // pure write into a block this sequence already owns.
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, 1)
                    .expect("checked against free_block_count under this same guard");
            }
        }
        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;

        let mut hidden = self.embed_token(token_id);
        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for (l, (layer, cache)) in self.layers.iter().zip(kv_caches.iter_mut()).enumerate() {
            // --- attention block ---
            let normed = rms_norm(&hidden, &layer.attn.norm_weight, self.config.rms_norm_eps);

            let (mut q, mut k, mut v) = {
                #[cfg(any(feature = "cuda", feature = "metal"))]
                {
                    if let Some(mut outs) = ferrox_core::WeightMatrix::apply_gpu_multi(
                        &[&layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj],
                        &normed,
                    ) {
                        let v = outs.pop().unwrap();
                        let k = outs.pop().unwrap();
                        let q = outs.pop().unwrap();
                        (q, k, v)
                    } else {
                        ferrox_core::weight_matrix::WeightMatrix::apply_three(
                            &layer.attn.q_proj,
                            &layer.attn.k_proj,
                            &layer.attn.v_proj,
                            &normed,
                        )
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    (
                        layer.attn.q_proj.apply(&normed),
                        layer.attn.k_proj.apply(&normed),
                        layer.attn.v_proj.apply(&normed),
                    )
                }
            };

            if let Some(bias) = &layer.attn.q_bias {
                for (x, b) in q.iter_mut().zip(bias.iter()) {
                    *x += b;
                }
            }
            if let Some(bias) = &layer.attn.k_bias {
                for (x, b) in k.iter_mut().zip(bias.iter()) {
                    *x += b;
                }
            }
            if let Some(bias) = &layer.attn.v_bias {
                for (x, b) in v.iter_mut().zip(bias.iter()) {
                    *x += b;
                }
            }

            if let Some(q_norm) = &layer.attn.q_norm {
                q = self.apply_qk_norm(&q, q_norm);
            }
            if let Some(k_norm) = &layer.attn.k_norm {
                k = self.apply_qk_norm(&k, k_norm);
            }
            self.apply_rope_attn_factor(&mut q, &mut k);

            for h in 0..n_heads {
                self.apply_rope_head_layer(&mut q[h * head_dim..(h + 1) * head_dim], pos, l);
            }
            for h in 0..n_kv_heads {
                self.apply_rope_head_layer(&mut k[h * head_dim..(h + 1) * head_dim], pos, l);
            }

            // Write guard for the push alone; see `SharedPagedKv`.
            // Holding it across attention below would serialise the
            // expensive half and give back a global lock.
            {
                let mut store = stores.write(l);
                cache
                    .push(&mut store, &k, &v)
                    .expect("reserved for every layer before the stack ran");
            }
            let store = stores.read(l);

            // Mirrors the contiguous dispatch above arm for arm. It has
            // to: the premise of paged KV is that it changes where rows
            // live and nothing else, so any arm the paged path did not
            // reproduce would be a model that answers differently
            // depending on whether a KV pool happened to be configured.
            // `causal_gqa_attention_paged_sinks` covers all three, and
            // is bit-identical to its contiguous twin by construction.
            let oai = self.gpt_oss.as_ref().map(|g| &g.layers[l]);
            let window = self.config.layer_sliding_window(l);
            let attn_out = ferrox_core::causal_gqa_attention_paged_sinks(
                &q,
                &store,
                cache.block_table(),
                n_heads,
                n_kv_heads,
                head_dim,
                cache.seq_len(),
                window,
                oai.map(|o| o.attn_sinks.as_slice()),
                // The contiguous sink arm passes no softcap, so neither
                // does this one; the other two carry the config's.
                if oai.is_some() {
                    None
                } else {
                    self.config.attn_logit_softcap
                },
            );
            let projected = layer.attn.o_proj.apply(&attn_out);

            for (h, p) in hidden.iter_mut().zip(projected.iter()) {
                *h += p;
            }

            // --- MoE FFN block ---
            let normed2 = rms_norm(&hidden, &layer.moe.norm_weight, self.config.rms_norm_eps);
            let ffn_out = Self::run_ffn_block(
                layer,
                &normed2,
                &self.config,
                hidden_dim,
                residency.as_ref().map(|p| p.layer_plan(l)),
            );
            for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h += f;
            }
        }

        let final_normed = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);
        Ok(self.logits_from_normed(&final_normed))
    }

    /// The shared expert store's live counters, when this model runs
    /// with store-backed (streamed) routed experts -- `None` for fully
    /// resident models. Every store-backed layer shares one store, so
    /// the first one found speaks for the whole model.
    pub fn expert_store_stats(&self) -> Option<ferrox_core::expert_store::ExpertStoreStats> {
        self.layers.iter().find_map(|l| match &l.moe.experts {
            ExpertBacking::Stored { store, .. } => Some(store.stats()),
            ExpertBacking::Resident(_) => None,
        })
    }

    /// Builds one global device-residency plan across ALL layers'
    /// routed experts against the single configured VRAM budget --
    /// every `(layer, expert)` candidate competes in one hotness-
    /// ordered pass and the running byte total is shared, so the
    /// budget cannot be re-spent per layer (the accounting bug the
    /// earlier per-layer `placement_plan` calls had: N layers would
    /// plan N x the configured bytes). Dense layers contribute no
    /// candidates (their sole expert always runs on CPU). Rebuilt per
    /// forward call so it tracks observed hotness; not yet
    /// performance-tuned, a disclosed limit.
    fn residency_plan(&self, vram_budget_bytes: u64) -> ferrox_moe::ResidencyPlan {
        let mut sizes_per_layer: Vec<Vec<usize>> = Vec::with_capacity(self.layers.len());
        let mut counts_per_layer: Vec<Vec<u64>> = Vec::with_capacity(self.layers.len());
        let mut any_observed = false;
        for layer in &self.layers {
            if Self::is_dense_layer(layer) {
                sizes_per_layer.push(Vec::new());
                counts_per_layer.push(Vec::new());
                continue;
            }
            sizes_per_layer.push(
                (0..layer.moe.n_experts())
                    .map(|e| layer.moe.expert_bytes(e))
                    .collect(),
            );
            let counts: Vec<u64> = layer
                .moe
                .activation_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
            any_observed |= counts.iter().any(|&c| c > 0);
            counts_per_layer.push(counts);
        }
        PlacementPlan::plan_layers_against_global_budget(
            &sizes_per_layer,
            any_observed.then_some(counts_per_layer.as_slice()),
            vram_budget_bytes,
        )
    }

    /// True if this layer has nothing to route: exactly one expert and
    /// no shared experts, the shape every non-MoE model (and every
    /// DeepSeek-style "leading dense layer") loads as. Top-1 selection
    /// out of one expert always picks it, and its weight is always
    /// exactly 1.0 regardless of gating function (softmax over one
    /// logit is trivially 1.0; sigmoid-then-renormalize divides the
    /// selected score by itself) -- so skipping the router matmul,
    /// `route_top_k`'s sort/exp/renormalize work, and
    /// `combine_expert_outputs`'s Vec-wrapping for this case is not an
    /// approximation, it produces the exact same result.
    fn is_dense_layer(layer: &LayerWeights) -> bool {
        layer.moe.n_experts() == 1 && layer.moe.shared_experts.is_empty()
    }

    /// llama.cpp `mul_mat_id` style: shared Q8 act + flat rayon over
    /// `(slot, row_pair)` for gate∥up (2-row SDOT), then SwiGLU, then
    /// per-slot down. One outer fork-join — no nested `apply_cpu_q8`.
    fn cpu_moe_topk_parallel_slots(
        experts: &[ExpertWeights],
        normed2: &[f32],
        decision: &ferrox_moe::RoutingDecision,
        hidden_dim: usize,
    ) -> Option<Vec<(Vec<f32>, f32)>> {
        use rayon::prelude::*;
        if !ferrox_core::weight_matrix::cpu_int_dot_enabled() || !normed2.len().is_multiple_of(32) {
            return None;
        }
        let n_slots = decision.expert_ids.len();
        if n_slots == 0 {
            return Some(Vec::new());
        }
        for &eid in &decision.expert_ids {
            let ex = experts.get(eid)?;
            if ex.gate.rows() == 0
                || ex.up.rows() != ex.gate.rows()
                || ex.down.rows() != hidden_dim
                || ex.gate.cols() != normed2.len()
                || ex.up.cols() != normed2.len()
                || ex.down.cols() != ex.gate.rows()
            {
                return None;
            }
            if !matches!(
                &ex.gate,
                WeightMatrix::Quantized {
                    kind: ferrox_core::QuantKind::Q4_0 | ferrox_core::QuantKind::Q8_0,
                    ..
                }
            ) || !matches!(
                &ex.up,
                WeightMatrix::Quantized {
                    kind: ferrox_core::QuantKind::Q4_0 | ferrox_core::QuantKind::Q8_0,
                    ..
                }
            ) {
                return None;
            }
        }
        let ffn_rows = experts[decision.expert_ids[0]].gate.rows();
        // Even ffn_rows: par_chunks_mut(2) never crosses a slot boundary.
        if !ffn_rows.is_multiple_of(2) {
            return None;
        }
        let act = ferrox_quant::quantize_activations_q8(normed2);
        let eids = &decision.expert_ids;
        let mut gate = vec![0f32; n_slots * ffn_rows];
        let mut up = vec![0f32; n_slots * ffn_rows];
        gate.par_chunks_mut(2)
            .zip(up.par_chunks_mut(2))
            .enumerate()
            .for_each(|(p, (gc, uc))| {
                let row0 = p * 2;
                let slot = row0 / ffn_rows;
                let r = row0 % ffn_rows;
                let ex = &experts[eids[slot]];
                if let (Some((g0, g1)), Some((u0, u1))) = (
                    ex.gate.dot_pair_cpu_q8(r, &act),
                    ex.up.dot_pair_cpu_q8(r, &act),
                ) {
                    gc[0] = g0;
                    gc[1] = g1;
                    uc[0] = u0;
                    uc[1] = u1;
                } else {
                    gc[0] = ex.gate.dot_row_cpu_q8(r, &act).unwrap_or(0.0);
                    gc[1] = ex.gate.dot_row_cpu_q8(r + 1, &act).unwrap_or(0.0);
                    uc[0] = ex.up.dot_row_cpu_q8(r, &act).unwrap_or(0.0);
                    uc[1] = ex.up.dot_row_cpu_q8(r + 1, &act).unwrap_or(0.0);
                }
            });
        let mut activated = vec![0f32; n_slots * ffn_rows];
        activated.par_iter_mut().enumerate().for_each(|(idx, a)| {
            let g = gate[idx];
            *a = (g / (1.0 + (-g).exp())) * up[idx];
        });
        let mut outs: Vec<(Vec<f32>, f32)> = decision
            .weights
            .iter()
            .map(|&w| (vec![0f32; hidden_dim], w))
            .collect();
        outs.par_iter_mut()
            .enumerate()
            .for_each(|(slot, (out, _))| {
                let ex = &experts[eids[slot]];
                let act_slot = &activated[slot * ffn_rows..(slot + 1) * ffn_rows];
                if act_slot.len().is_multiple_of(32) {
                    let q8 = ferrox_quant::quantize_activations_q8(act_slot);
                    if let Some(d) = ex.down.apply_cpu_q8(&q8) {
                        *out = d;
                        return;
                    }
                }
                *out = ex.down.apply(act_slot);
            });
        Some(outs)
    }

    /// Fallback: serial top-k with shared Q8 act (pre-mul_mat_id path).
    fn cpu_moe_serial_experts(
        layer: &LayerWeights,
        normed2: &[f32],
        decision: &ferrox_moe::RoutingDecision,
        plan: Option<&PlacementPlan>,
    ) -> Vec<(Vec<f32>, f32)> {
        let shared_act = if ferrox_core::weight_matrix::cpu_int_dot_enabled()
            && normed2.len().is_multiple_of(32)
            && plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true)
        {
            Some(ferrox_quant::quantize_activations_q8(normed2))
        } else {
            None
        };
        decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| {
                let placement = plan
                    .map(|p| p.placement_for(eid))
                    .unwrap_or(ExpertPlacement::Cpu);
                let out = layer.moe.with_expert(eid, |ex| {
                    if let Some(ref act) = shared_act {
                        if let (Some(gate), Some(up)) =
                            (ex.gate.apply_cpu_q8(act), ex.up.apply_cpu_q8(act))
                        {
                            let activated = ferrox_core::matmul::swiglu(&gate, &up);
                            return ex.down.apply(&activated);
                        }
                    }
                    run_expert_placed(normed2, ex, placement)
                });
                (out, w)
            })
            .collect()
    }

    /// Runs one position's normalized hidden state through this
    /// layer's MoE FFN block, given already-computed router logits for
    /// that position, returning the combined output to add back into
    /// the residual stream. Shared by `forward_token` (router computed
    /// via a single `apply` call, since there's only one position) and
    /// `forward_batch`'s per-position loop (router computed via one
    /// batched `apply_batch` call up front, sliced per position here --
    /// see `forward_batch`'s doc comment for why that batching matters
    /// and must not be lost by calling this per position instead).
    /// `gpu_vram_budget_bytes`: see `Decoder::gpu_vram_budget_bytes`'s
    /// doc comment -- `None` dispatches every routed expert through
    /// `run_expert_placed` with `ExpertPlacement::Cpu`, which is
    /// exactly `run_expert`'s own behavior, so this is a real
    /// zero-behavior-change default, not just "probably fine."
    /// One token's routing decision for one MoE layer.
    ///
    /// Three shapes, in the order llama.cpp's `build_moe_ffn` decides
    /// them: grouped selection when the checkpoint declares expert
    /// groups; the biased/scaled port when the layer carries
    /// `exp_probs_b` or the model carries a non-unit
    /// `expert_weights_scale`; otherwise the plain top-k this decoder has
    /// always used. The last arm is kept rather than folded into
    /// `route_top_k_biased` so that every checkpoint without those two
    /// features routes through byte-identical code to before.
    ///
    /// `exp_probs_b` together with expert groups is refused at load
    /// (`loader.rs`), so that combination cannot reach here.
    fn route_for_layer(
        layer: &LayerWeights,
        router_logits: &[f32],
        config: &ModelConfig,
    ) -> ferrox_moe::RoutingDecision {
        match (
            config.moe.expert_group_count,
            config.moe.expert_group_used_count,
        ) {
            (Some(n_groups), Some(k_per_group)) if n_groups > 1 && k_per_group > 0 => {
                ferrox_moe::route_top_k_grouped(
                    router_logits,
                    n_groups,
                    k_per_group,
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                )
            }
            _ if layer.moe.exp_probs_bias.is_some() || config.moe.expert_weights_scale != 1.0 => {
                ferrox_moe::route_top_k_biased(
                    router_logits,
                    layer.moe.exp_probs_bias.as_deref(),
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                    config.moe.expert_weights_scale,
                )
            }
            _ => route_top_k(
                router_logits,
                config.moe.n_experts_active,
                config.moe.gating,
                config.moe.norm_topk_prob,
            ),
        }
    }

    fn combine_ffn_outputs_for_position(
        layer: &LayerWeights,
        normed2: &[f32],
        router_logits: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
    ) -> Vec<f32> {
        let decision = Self::route_for_layer(layer, router_logits, config);
        layer.moe.record_activations(&decision.expert_ids);
        // Best-effort warm of the routed experts for this layer into
        // the store cache (SSD streaming overlap). Resident-backed
        // layers skip this entirely.
        if let ExpertBacking::Stored {
            store,
            layer: layer_id,
            ..
        } = &layer.moe.experts
        {
            let keys: Vec<ferrox_core::expert_store::ExpertKey> = decision
                .expert_ids
                .iter()
                .map(|&eid| ferrox_core::expert_store::ExpertKey {
                    layer: *layer_id,
                    expert: eid as u32,
                })
                .collect();
            store.prefetch(&keys);
        }

        // Metal: fuse all top-k experts into one CB (one wait) when every
        // routed expert has Metal matvec launches. Shared experts (rare
        // for OLMoE) still run on the host after.
        #[cfg(feature = "metal")]
        if ferrox_core::metal_dense_enabled()
            && matches!(
                config.ffn_activation,
                crate::config::FfnActivation::Swiglu | crate::config::FfnActivation::SwigluFused
            )
            && layer.moe.shared_experts.is_empty()
        {
            if let Some(fused) = Self::try_metal_moe_topk(layer, normed2, &decision) {
                return fused;
            }
        }

        let routed_outputs: Vec<(Vec<f32>, f32)> = {
            // llama.cpp mul_mat_id: one shared Q8 act + flat (slot,row)
            // parallel over all top-k experts (not serial expert loops each
            // with their own rayon fork-join).
            let all_cpu = plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true);
            if let (true, ExpertBacking::Resident(experts)) = (all_cpu, &layer.moe.experts) {
                if let Some(outs) =
                    Self::cpu_moe_topk_parallel_slots(experts, normed2, &decision, hidden_dim)
                {
                    outs
                } else {
                    Self::cpu_moe_serial_experts(layer, normed2, &decision, plan)
                }
            } else {
                Self::cpu_moe_serial_experts(layer, normed2, &decision, plan)
            }
        };
        // Shared experts fire on every token regardless of routing, so
        // there's no offload decision to make for them the way there
        // is for routed experts -- always CPU, matching `run_expert`.
        let mut shared_outputs: Vec<Vec<f32>> = layer
            .moe
            .shared_experts
            .iter()
            .map(|e| run_expert(normed2, e))
            .collect();
        // Qwen2-MoE-specific: see `MoeWeights::shared_expert_gate`'s doc
        // comment. Scaling here (before `combine_expert_outputs`, which
        // is architecture-agnostic and knows nothing about this gate)
        // keeps the gate a decoder-level detail, not a ferrox-moe API
        // change.
        if let Some(gate) = &layer.moe.shared_expert_gate {
            let gate_logit: f32 = gate.iter().zip(normed2.iter()).map(|(g, x)| g * x).sum();
            let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
            for out in shared_outputs.iter_mut() {
                for x in out.iter_mut() {
                    *x *= gate_value;
                }
            }
        }

        combine_expert_outputs(&routed_outputs, &shared_outputs, hidden_dim)
    }

    /// The dense FFN for a whole batch of positions in three batched
    /// matmuls (gate, up, down) instead of three per position.
    ///
    /// This is the counterpart of what `forward_hidden_batch` already
    /// did for Q/K/V and the router, and it is where a dense model's
    /// prefill time actually goes: `WeightMatrix::apply_batch` reads
    /// each weight row once and dots it against every position, rather
    /// than re-reading the whole FFN for each one.
    ///
    /// `None` for anything that is not a plain dense layer -- MoE
    /// routing is per position by construction, so those keep the
    /// sequential path.
    ///
    /// On a GPU backend the per-position alternative is one *fused*
    /// gate+up+SiLU+down launch (`apply_gpu_dense_ffn_swiglu`), so this
    /// used to be gated off there: three separate batched launches lost
    /// to it while `apply_batch` was still a batched *matvec*.
    ///
    /// That stopped being true once the simdgroup GEMM landed, and the
    /// old gate turned out to be the dominant cost of Metal prefill --
    /// a 512-token prompt ran the FFN one position at a time, 512 x
    /// n_layers fused launches, which a profile put at 90% of prefill
    /// while the GEMM it bypassed accounted for 21%.
    ///
    /// Decode (`batch_size == 1`) still takes the fused per-position
    /// launch, which is the right shape there.
    fn dense_ffn_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        // Match the GPU `mul_mm` threshold: below it the per-call launch
        // overhead outweighs the weight reuse.
        if !Self::is_dense_layer(layer) || batch_size < 4 {
            return None;
        }
        // On a GPU backend this only wins when the weights have a real
        // batched GEMM; otherwise `apply_batch` is a batched matvec and
        // loses to the fused per-position launch.
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            let gpu_dense = ferrox_core::weight_matrix::metal_dense_enabled();
            #[cfg(not(feature = "metal"))]
            let gpu_dense = false;
            #[cfg(feature = "cuda")]
            let gpu_dense = gpu_dense || ferrox_core::weight_matrix::cuda_dense_enabled();
            if gpu_dense {
                let all_gemm = layer.moe.with_expert(0, |ex| {
                    ex.gate.prefers_gpu_batch()
                        && ex.up.prefers_gpu_batch()
                        && ex.down.prefers_gpu_batch()
                });
                if !all_gemm {
                    return None;
                }
            }
        }
        layer.moe.record_activations(&[0]);
        // One command buffer for the whole FFN when every matrix has a
        // simdgroup GEMM: gate and up feed the activation and the down
        // projection without the intermediates ever touching the host.
        // Three separate launches cost three round trips per layer plus
        // four copies of a `batch x ffn_dim` tensor.
        #[cfg(feature = "metal")]
        if ferrox_core::weight_matrix::metal_dense_enabled() {
            let gelu = matches!(config.ffn_activation, crate::config::FfnActivation::Gelu);
            let fused = layer.moe.with_expert(0, |ex| {
                let (g, u, d) = (
                    ex.gate.mul_mm_sg_launch()?,
                    ex.up.mul_mm_sg_launch()?,
                    ex.down.mul_mm_sg_launch()?,
                );
                ferrox_metal::gpu::launch_dense_ffn_swiglu_batch(
                    &g,
                    &u,
                    &d,
                    normed2_batch,
                    batch_size,
                    gelu,
                )
                .ok()
            });
            if let Some(out) = fused {
                return Some(out);
            }
        }
        Some(layer.moe.with_expert(0, |ex| {
            let ffn_acts = ex.gate.quantize_batch_acts(normed2_batch, batch_size);
            let gate = ex
                .gate
                .apply_batch_with_acts(normed2_batch, batch_size, ffn_acts.as_ref());
            let up = ex
                .up
                .apply_batch_with_acts(normed2_batch, batch_size, ffn_acts.as_ref());
            let activated: Vec<f32> = match config.ffn_activation {
                crate::config::FfnActivation::Swiglu
                | crate::config::FfnActivation::SwigluFused => {
                    ferrox_core::matmul::swiglu(&gate, &up)
                }
                crate::config::FfnActivation::Gelu => geglu(&gate, &up),
            };
            ex.down.apply_batch(&activated, batch_size)
        }))
    }

    /// CPU MoE prefill: bucket tokens by expert, then one
    /// `apply_batch` per expert with tokens instead of per-token
    /// `combine_ffn_outputs_for_position`. Shared experts append via
    /// [`Self::accumulate_shared_experts_batch`]. `None` when gates fail
    /// (small batch, dense, Metal preferred, non-SwiGLU, non-resident,
    /// or any GPU-placed expert).
    fn moe_ffn_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        config: &ModelConfig,
        plan: Option<&PlacementPlan>,
    ) -> Option<Vec<f32>> {
        if batch_size < 32 || Self::is_dense_layer(layer) {
            return None;
        }
        // Metal prefill owns MoE when dense Metal is on
        // (`try_metal_moe_prefill_batch`); do not steal the path.
        #[cfg(feature = "metal")]
        if ferrox_core::metal_dense_enabled() {
            return None;
        }
        if !matches!(
            config.ffn_activation,
            crate::config::FfnActivation::Swiglu | crate::config::FfnActivation::SwigluFused
        ) {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let n_experts = experts.len();
        let all_cpu = plan
            .map(|p| (0..n_experts).all(|eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu)))
            .unwrap_or(true);
        if !all_cpu || n_experts == 0 {
            return None;
        }

        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = Self::route_for_layer(layer, logits, config);
            layer.moe.record_activations(&decision.expert_ids);
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                buckets[eid].push((b, w));
            }
        }

        let mut acc = vec![0f32; batch_size * hidden_dim];
        for (eid, toks) in buckets.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let mut gathered = vec![0f32; n * hidden_dim];
            for (i, &(tok, _)) in toks.iter().enumerate() {
                gathered[i * hidden_dim..(i + 1) * hidden_dim]
                    .copy_from_slice(&normed2_batch[tok * hidden_dim..(tok + 1) * hidden_dim]);
            }
            let ex = &experts[eid];
            let ffn_acts = ex.gate.quantize_batch_acts(&gathered, n);
            let gate = ex
                .gate
                .apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
            let up = ex.up.apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
            let activated = ferrox_core::matmul::swiglu(&gate, &up);
            let down = ex.down.apply_batch(&activated, n);
            for (i, &(tok, w)) in toks.iter().enumerate() {
                let out = &down[i * hidden_dim..(i + 1) * hidden_dim];
                let row = &mut acc[tok * hidden_dim..(tok + 1) * hidden_dim];
                for (a, &o) in row.iter_mut().zip(out.iter()) {
                    *a += w * o;
                }
            }
        }

        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut acc,
        );
        Some(acc)
    }

    /// gpt-oss's MoE FFN for one position.
    ///
    /// A separate function rather than another branch inside
    /// `combine_ffn_outputs_for_position` on purpose: that path carries
    /// expert-store prefetch, residency placement, a Metal top-k fusion
    /// and a batched parallel-slot kernel, and every one of them would
    /// need its own gpt-oss variant to stay honest. This is the whole
    /// gpt-oss FFN in one readable block, checked end-to-end against
    /// llama.cpp, and slow — routed experts run serially. It is the
    /// correct-first shape; making it fast is a separate change with its
    /// own A/B, not something to smuggle in under a correctness fix.
    ///
    /// Ported from `llama-graph.cpp::build_moe_ffn` with
    /// `gating_op = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`,
    /// `type_op = LLM_FFN_SWIGLU_OAI_MOE`, `norm_w = false`,
    /// `w_scale = 1`, all four bias tensors present.
    fn gpt_oss_ffn(
        layer: &LayerWeights,
        oai: &GptOssLayer,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut router_logits = layer.moe.router.apply(normed2);
        for (x, b) in router_logits.iter_mut().zip(oai.router_bias.iter()) {
            *x += b;
        }
        // Selection on the raw biased logits, softmax over the winners
        // only -- see `route_top_k_softmax_weight`.
        let decision =
            ferrox_moe::route_top_k_softmax_weight(&router_logits, config.moe.n_experts_active);
        layer.moe.record_activations(&decision.expert_ids);

        let mut out = vec![0f32; hidden_dim];
        for (slot, &eid) in decision.expert_ids.iter().enumerate() {
            let w = decision.weights[slot];
            let expert_out = layer.moe.with_expert(eid, |ex| {
                ferrox_moe::run_expert_oai(
                    normed2,
                    ex,
                    &oai.expert_bias[eid],
                    ferrox_moe::SWIGLU_OAI_ALPHA,
                    ferrox_moe::SWIGLU_OAI_LIMIT,
                )
            });
            for (o, e) in out.iter_mut().zip(expert_out.iter()) {
                *o += w * e;
            }
        }
        out
    }

    /// `forward_token`'s MoE FFN block for one position: the dense
    /// fast path (see `is_dense_layer`) or the full router+combine path
    /// with the router computed inline via a single-position `apply`.
    fn run_ffn_block(
        layer: &LayerWeights,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
    ) -> Vec<f32> {
        if Self::is_dense_layer(layer) {
            layer.moe.record_activations(&[0]);
            return layer.moe.with_expert(0, |ex| match config.ffn_activation {
                crate::config::FfnActivation::Swiglu
                | crate::config::FfnActivation::SwigluFused => run_expert(normed2, ex),
                crate::config::FfnActivation::Gelu => {
                    // Share one Q8 act quant across gate+up when INT_DOT
                    // can serve both (Q8_0 / Q4_0); else two `.apply`s.
                    if ferrox_core::weight_matrix::cpu_int_dot_enabled()
                        && normed2.len().is_multiple_of(32)
                    {
                        let act = ferrox_quant::quantize_activations_q8(normed2);
                        if let (Some(gate), Some(up)) =
                            (ex.gate.apply_cpu_q8(&act), ex.up.apply_cpu_q8(&act))
                        {
                            let activated = geglu(&gate, &up);
                            return ex.down.apply(&activated);
                        }
                    }
                    let gate = ex.gate.apply(normed2);
                    let up = ex.up.apply(normed2);
                    let activated = geglu(&gate, &up);
                    ex.down.apply(&activated)
                }
            });
        }
        let router_logits = layer.moe.router.apply(normed2);
        Self::combine_ffn_outputs_for_position(
            layer,
            normed2,
            &router_logits,
            config,
            hidden_dim,
            plan,
        )
    }

    /// Processes multiple new positions in one call instead of calling
    /// `forward_token` once per position. `tokens[i]` is the token at
    /// absolute position `start_pos + i`; all positions attend
    /// causally (position `i` sees positions `0..=i` of this batch
    /// plus everything already in `kv_caches`, nothing later).
    ///
    /// The attention block's Q/K/V/O projections and the MoE router
    /// are computed as batched matmuls (`WeightMatrix::apply_batch`),
    /// which for quantized weights means each weight row is read from
    /// memory once and dotted against every position in the batch,
    /// not once per position -- see `apply_batch`'s doc comment for
    /// why that's a real memory-bandwidth saving, not just fewer
    /// function calls. The expert FFN stage is *not* batched: which
    /// expert(s) a position routes to is data-dependent per position,
    /// so positions routed to different experts can't share a single
    /// matmul the way the shared Q/K/V/router projections can. RoPE
    /// and attention itself (causal masking, softmax) are also
    /// per-position, since they're cheap relative to the matmuls and
    /// batching them would add complexity for little benefit.
    ///
    /// This is what makes prompt-lookup speculative decoding
    /// (`speculative` module) actually save work rather than just
    /// reshuffle it: verifying `k` draft tokens costs one batched call
    /// here, not `k` calls to `forward_token`.
    ///
    /// Thin wrapper over [`Self::forward_hidden_batch`] + `output_head`.
    pub fn forward_batch(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<Vec<f32>> {
        let hiddens = self.forward_hidden_batch(tokens, start_pos, kv_caches);
        if hiddens.is_empty() {
            return Vec::new();
        }
        let batch_size = hiddens.len();
        let flat: Vec<f32> = hiddens.into_iter().flatten().collect();
        self.logits_from_flat_hidden(flat, batch_size)
    }

    /// [`Self::forward_batch`] that also hands back the final-layer
    /// hidden state for every position instead of dropping it.
    ///
    /// `forward_batch` computes these and throws them away; a
    /// hidden-state-conditioned drafter (EAGLE, MTP, dFlash) needs
    /// exactly the vector for the last *verified* position, so
    /// recomputing it would mean running the target model twice for
    /// something the first pass already had in hand. The extra cost
    /// here is one copy of `[batch x hidden]`, which is why
    /// `forward_batch` keeps its move-only path for the prefill case
    /// that does not want them.
    ///
    /// Returns `(logits_per_position, hidden_per_position)`, both
    /// indexed by position in `tokens`.
    pub fn forward_batch_with_hidden(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let hiddens = self.forward_hidden_batch(tokens, start_pos, kv_caches);
        if hiddens.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let batch_size = hiddens.len();
        let flat: Vec<f32> = hiddens.iter().flatten().copied().collect();
        (self.logits_from_flat_hidden(flat, batch_size), hiddens)
    }

    /// One token's embedding row, scaled if this checkpoint scales it.
    ///
    /// `embedding_scale` is `sqrt(hidden_dim)` on the Gemma family and
    /// `None` everywhere else, so a path that dequantizes the row and
    /// forgets the multiply is wrong on exactly one family and right on
    /// every other -- which is why it survived as a drift for as long as
    /// it did. The lookup and the scale live in one function so a caller
    /// cannot obtain the row without it.
    fn embed_token(&self, token_id: usize) -> Vec<f32> {
        let mut row = self.embedding.dequant_row(token_id);
        if let Some(scale) = self.config.embedding_scale {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        row
    }

    /// [`Self::embed_token`] for a whole batch: `[batch, hidden]`,
    /// flattened row-major.
    fn embed_tokens(&self, tokens: &[usize]) -> Vec<f32> {
        tokens.iter().flat_map(|&t| self.embed_token(t)).collect()
    }

    /// The `output_head` half of a single-position forward: project the
    /// final-normed hidden state and softcap the result if this
    /// checkpoint softcaps it.
    ///
    /// The counterpart to [`Self::logits_from_flat_hidden`] for the
    /// one-row case, and held here for the same reason: Gemma-2 caps its
    /// final logits at 30.0, so a path that projects and returns without
    /// capping produces a different distribution -- not an error, just a
    /// quietly wrong one.
    fn logits_from_normed(&self, final_normed: &[f32]) -> Vec<f32> {
        let mut logits = self.output_head.apply(final_normed);
        if let Some(sc) = self.config.final_logit_softcap {
            softcap_inplace(&mut logits, sc);
        }
        logits
    }

    /// The `output_head` half of [`Self::forward_batch`], split out so
    /// the hidden-state-returning variant cannot drift from it (a
    /// second copy of the softcap would be a silent quality bug).
    fn logits_from_flat_hidden(&self, flat: Vec<f32>, batch_size: usize) -> Vec<Vec<f32>> {
        let vocab_size = self.output_head.rows();
        let mut logits_batch = self.output_head.apply_batch(&flat, batch_size);
        if let Some(sc) = self.config.final_logit_softcap {
            softcap_inplace(&mut logits_batch, sc);
        }
        logits_batch
            .chunks(vocab_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// [`Self::forward_batch`] for the common case where only the final
    /// position's logits are wanted: prefill a prompt, then sample the
    /// next token. Runs `output_head` on **one** row instead of all
    /// `batch_size` of them.
    ///
    /// The KV cache and every hidden state are identical either way —
    /// only the vocabulary projection is skipped, and only for rows
    /// whose logits the caller was going to drop. That projection is not
    /// a rounding error: it is `[batch x hidden] x [hidden x vocab]`,
    /// which for a large-vocabulary model with a small body is a large
    /// share of prefill. `V*H / (V*H + L*P_layer)` comes to 30% on
    /// Gemma-3-1B, 21% on Llama-3.2-1B and SmolLM2, 23% on Gemma-2-2B.
    /// llama.cpp does not do this work at all during `pp512` —
    /// `llama_batch_get_one` leaves `logits` unset, so `inp_out_ids`
    /// selects a single row.
    ///
    /// [`Self::forward_batch`] stays for the callers that genuinely need
    /// every row: speculative verification checks each draft position,
    /// and `/v1/embeddings` pools over all of them.
    pub fn forward_batch_last(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        let hiddens = self.forward_hidden_batch(tokens, start_pos, kv_caches);
        let Some(last) = hiddens.last() else {
            return Vec::new();
        };
        self.logits_from_normed(last)
    }

    /// [`Self::forward_batch_last`] over paged KV: the prefill twin of
    /// [`Self::forward_token_paged`].
    ///
    /// # Why this gathers instead of paging the kernel
    ///
    /// `forward_hidden_batch`'s fast arm hands `cache.k` / `cache.v` to
    /// `causal_gqa_attention_prefill_shared_kv_windowed`, which is Rayon
    /// over `[query-block x head]` against one flat KV buffer. That
    /// blocking is why CPU prefill is not the per-query path, and a
    /// block table cannot be handed to it as a slice.
    ///
    /// The alternative was a second blocked kernel that reads through
    /// the table. This file has just finished paying for what a second
    /// copy of a rule costs: the paged decode path silently lost the
    /// window arm, the sink term, the attention softcap, the embedding
    /// scale and the final logit softcap, one at a time, because it was
    /// a copy. A prefill kernel is a much larger surface to keep in
    /// step than any of those. So the pages are materialised, the ONE
    /// prefill implementation every other path uses runs against them,
    /// and the new rows go back.
    ///
    /// Bit-identity is therefore by construction rather than by
    /// agreement between two kernels: this calls the same function with
    /// the same values. What the tests pin is that the gather and the
    /// scatter are faithful, not that two implementations of attention
    /// happen to match.
    ///
    /// The cost is one KV-sized copy per layer per call, against the
    /// matmuls that dominate prefill. Decode is untouched: it still
    /// reads through the block table and copies nothing, which is where
    /// page sharing pays.
    ///
    /// # Failure is checked before anything is written
    ///
    /// Every layer's blocks are reserved up front, so a store too small
    /// for the batch refuses with `PagedStoreExhausted` having mutated
    /// no layer. A partial append would leave some layers longer than
    /// others, and no caller can recover from that.
    pub fn forward_batch_last_paged(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.layers.len());
        assert_eq!(stores.layer_count(), self.layers.len());
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Reserve every layer up front, under guards spanning the check
        // AND the take. Each layer has its own store, so one having
        // room says nothing about the next -- and under concurrency,
        // checking and then taking as separate steps lets another
        // request slip in between and leave this one half-written.
        //
        // Reserving before the forward rather than after also means a
        // request that cannot fit is refused before it burns a prefill.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, tokens.len()) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, tokens.len())
                    .expect("checked against free_block_count under this same guard");
            }
        }

        // Gather under read guards, one layer at a time: the forward
        // below is the expensive part and holds nothing.
        let mut scratch: Vec<KvCache> = kv_caches
            .iter()
            .enumerate()
            .map(|(l, cache)| cache.to_contiguous(&stores.read(l)))
            .collect();

        let logits = self.forward_batch_last(tokens, start_pos, &mut scratch);

        // Scatter into blocks this sequence already owns. Nothing here
        // can fail, which is the point of reserving above.
        for (l, (cache, gathered)) in kv_caches.iter_mut().zip(&scratch).enumerate() {
            let mut store = stores.write(l);
            let width = store.n_kv_heads() * store.head_dim();
            let base = cache.seq_len() * width;
            cache
                .append_contiguous(
                    &mut store,
                    &gathered.k[base..],
                    &gathered.v[base..],
                    tokens.len(),
                )
                .expect("blocks reserved above are still held by this sequence");
        }
        Ok(logits)
    }

    /// Like [`Self::forward_batch`], but returns final RMS-normed hidden
    /// states (pre-`output_head`) — one `hidden_dim` vector per input
    /// token. Used by `/v1/embeddings` pooling (mean / last).
    pub fn forward_hidden_batch(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<Vec<f32>> {
        assert_eq!(kv_caches.len(), self.layers.len());
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens);

        #[cfg(feature = "metal")]
        let use_metal_attn = ferrox_core::metal_dense_enabled()
            && ferrox_metal::attn::metal_attn_enabled()
            && self
                .layers
                .iter()
                .all(|l| self.layer_supports_metal_attn(l));

        #[cfg(not(feature = "metal"))]
        let use_metal_attn = false;

        let residency = self.expert_residency_plan(use_metal_attn);

        #[cfg(feature = "metal")]
        let mut metal_kv_guard: Option<
            std::sync::MutexGuard<'_, Option<Vec<ferrox_metal::attn::MetalKvBuffers>>>,
        > = if use_metal_attn {
            Some(self.metal_attn_kv.lock().unwrap())
        } else {
            None
        };

        #[cfg(feature = "metal")]
        if let Some(guard) = metal_kv_guard.as_mut() {
            let need = self.layers.len();
            let need_cap = start_pos
                .saturating_add(batch_size)
                .saturating_add(256)
                .max(512);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    v.len() != need
                        || v.iter().any(|m| m.capacity() < need_cap)
                        || v.iter()
                            .zip(kv_caches.iter())
                            .any(|(m, c)| m.seq_len != c.seq_len)
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match ferrox_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, need_cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
                        if c.seq_len > 0 && m.upload_from_host(&c.k, &c.v, c.seq_len).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        **guard = Some(bufs);
                    } else {
                        **guard = None;
                    }
                } else {
                    **guard = None;
                }
            }
        }

        let n_layers = self.layers.len();
        let mut l = 0usize;
        while l < n_layers {
            let layer = &self.layers[l];
            let q_width = n_heads * head_dim;
            let kv_width = n_kv_heads * head_dim;

            // Multi-layer dense prefill: one CB, activations stay on GPU.
            #[cfg(feature = "metal")]
            if use_metal_attn && batch_size >= 4 {
                if let Some(guard) = metal_kv_guard.as_mut() {
                    if let Some(metal_kvs) = guard.as_mut() {
                        if let Some(run_len) = self.metal_prefill_dense_stack_run_len(
                            l,
                            start_pos,
                            batch_size,
                            kv_caches,
                            Some(metal_kvs.as_slice()),
                        ) {
                            if let Some(h_out) = self.try_metal_prefill_dense_stack(
                                l,
                                run_len,
                                &hidden_batch,
                                start_pos,
                                batch_size,
                                n_heads,
                                metal_kvs,
                                kv_caches,
                            ) {
                                hidden_batch = h_out;
                                l += run_len;
                                continue;
                            }
                        }
                    }
                }
            }

            let cache = &mut kv_caches[l];

            // One-CB dense prefill (RMSNorm→QKV GEMM→attn→O→FFN) when every
            // projection has mul_mm_sg and the layer has no QKV bias / QK-norm.
            #[cfg(feature = "metal")]
            if use_metal_attn && batch_size >= 4 && Self::metal_prefill_dense_layer_eligible(layer)
            {
                let swa_fits = self.metal_prefill_dense_swa_fits(l, start_pos, batch_size);
                if swa_fits {
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            if metal_kvs[l].seq_len == cache.seq_len && start_pos == cache.seq_len {
                                layer.moe.record_activations(&[0]);
                                let fused = layer.moe.with_expert(0, |ex| {
                                    let (q, k, v, o) = (
                                        layer.attn.q_proj.mul_mm_sg_launch()?,
                                        layer.attn.k_proj.mul_mm_sg_launch()?,
                                        layer.attn.v_proj.mul_mm_sg_launch()?,
                                        layer.attn.o_proj.mul_mm_sg_launch()?,
                                    );
                                    let ffn = ferrox_metal::attn::PrefillFfnMetal::Dense {
                                        gate: ex.gate.mul_mm_sg_launch()?,
                                        up: ex.up.mul_mm_sg_launch()?,
                                        down: ex.down.mul_mm_sg_launch()?,
                                    };
                                    let gelu = matches!(
                                        self.config.ffn_activation,
                                        crate::config::FfnActivation::Gelu
                                    );
                                    let prefill_layer =
                                        ferrox_metal::attn::PrefillDenseLayerMetal {
                                            attn_norm_w: &layer.attn.norm_weight,
                                            ffn_norm_w: &layer.moe.norm_weight,
                                            q,
                                            k,
                                            v,
                                            o,
                                            ffn,
                                            post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                                            post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                                            extras: self.metal_attn_extras(layer),
                                            layer_idx: l as u32,
                                        };
                                    ferrox_metal::attn::launch_prefill_dense_layer(
                                        &hidden_batch,
                                        &prefill_layer,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope(),
                                        self.config.layer_rope_theta(l),
                                        self.config.rope_freqs.as_deref(),
                                        start_pos,
                                        self.config.rms_norm_eps,
                                        gelu,
                                        self.config.attn_logit_softcap,
                                    )
                                    .ok()
                                });
                                if let Some(h_out) = fused {
                                    cache
                                        .advance_len(batch_size)
                                        .expect("unbounded/planned KvCache growth is infallible");
                                    hidden_batch = h_out;
                                    l += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // --- attention block ---
            let normed_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| rms_norm(h, &layer.attn.norm_weight, self.config.rms_norm_eps))
                .flatten()
                .collect();

            // One shared activation-quant pass for q/k/v (plan 1e): the
            // three projections read the same normed batch, so quantize it
            // once instead of once per projection. A kind mismatch inside
            // the group just re-quantizes locally.
            let qkv_acts = layer
                .attn
                .q_proj
                .quantize_batch_acts(&normed_batch, batch_size);
            let mut q_batch = layer.attn.q_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            let mut k_batch = layer.attn.k_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            let mut v_batch = layer.attn.v_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            drop(qkv_acts);

            if let Some(bias) = &layer.attn.q_bias {
                for row in q_batch.chunks_mut(q_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }
            if let Some(bias) = &layer.attn.k_bias {
                for row in k_batch.chunks_mut(kv_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }
            if let Some(bias) = &layer.attn.v_bias {
                for row in v_batch.chunks_mut(kv_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }

            if let Some(q_norm) = &layer.attn.q_norm {
                for row in q_batch.chunks_mut(q_width) {
                    let normed = self.apply_qk_norm(row, q_norm);
                    row.copy_from_slice(&normed);
                }
            }
            if let Some(k_norm) = &layer.attn.k_norm {
                for row in k_batch.chunks_mut(kv_width) {
                    let normed = self.apply_qk_norm(row, k_norm);
                    row.copy_from_slice(&normed);
                }
            }
            // Host-side `mscale`, applied before either backend ropes.
            // The Metal branch below therefore hands its kernels
            // `attn_factor_applied_by_caller()` — folding it into cos/sin
            // there as well would square it.
            self.apply_rope_attn_factor(&mut q_batch, &mut k_batch);

            #[cfg(feature = "metal")]
            {
                let mut did_metal_prefill = false;
                // The Metal prefill kernel is full-causal: only safe on a
                // SWA layer while every causal position is still inside
                // the window. Longer prompts fall back to CPU attention.
                let swa_fits = match self.config.layer_sliding_window(l) {
                    Some(window) => start_pos + batch_size <= window,
                    None => true,
                };
                // Metal prefill applies attn softcap in FA-vec / legacy GQA.
                if let Some(guard) = metal_kv_guard.as_mut() {
                    if let Some(metal_kvs) = guard.as_mut() {
                        if metal_kvs[l].seq_len == cache.seq_len
                            && start_pos == cache.seq_len
                            && swa_fits
                        {
                            let prefill_res = {
                                let o_launch = Self::metal_matvec_launch(&layer.attn.o_proj);
                                // Prefill O fusion: opt-in. Default off until
                                // fair-chat prompt_per_s proves a win without
                                // decode noise (Host B contention-sensitive).
                                let fuse_o = matches!(
                                    std::env::var("FERROX_METAL_PREFILL_FUSE_O").ok().as_deref(),
                                    Some("1") | Some("true") | Some("on")
                                ) && o_launch.as_ref().is_some_and(|o| {
                                    o.fn_name == "q4_0_matvec"
                                        && o.block_bytes == 18
                                        && layer.attn.post_attn_norm.is_none()
                                });
                                if fuse_o {
                                    let o = o_launch.as_ref().unwrap();
                                    ferrox_metal::attn::launch_prefill_attn_o_residual(
                                        &q_batch,
                                        &k_batch,
                                        &v_batch,
                                        &hidden_batch,
                                        o,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope().attn_factor_applied_by_caller(),
                                        self.config.layer_rope_theta(l),
                                        self.config.rope_freqs.as_deref(),
                                        start_pos,
                                        self.config.attn_logit_softcap,
                                    )
                                    .map(|h_out| {
                                        cache.advance_len(batch_size).expect(
                                            "unbounded/planned KvCache growth is infallible",
                                        );
                                        hidden_batch = h_out;
                                        true
                                    })
                                } else {
                                    ferrox_metal::attn::launch_prefill_attn_block(
                                        &q_batch,
                                        &k_batch,
                                        &v_batch,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope().attn_factor_applied_by_caller(),
                                        self.config.layer_rope_theta(l),
                                        self.config.rope_freqs.as_deref(),
                                        start_pos,
                                        self.config.attn_logit_softcap,
                                        false,
                                    )
                                    .map(
                                        |(attn_out_batch, _, _)| {
                                            cache.advance_len(batch_size).expect(
                                                "unbounded/planned KvCache growth is infallible",
                                            );
                                            let projected_batch = layer
                                                .attn
                                                .o_proj
                                                .apply_batch(&attn_out_batch, batch_size);
                                            let projected_batch =
                                                if let Some(post) = &layer.attn.post_attn_norm {
                                                    projected_batch
                                                        .chunks(hidden_dim)
                                                        .flat_map(|row| {
                                                            rms_norm(
                                                                row,
                                                                post,
                                                                self.config.rms_norm_eps,
                                                            )
                                                        })
                                                        .collect::<Vec<_>>()
                                                } else {
                                                    projected_batch
                                                };
                                            for (h, p) in
                                                hidden_batch.iter_mut().zip(projected_batch.iter())
                                            {
                                                *h += p;
                                            }
                                            true
                                        },
                                    )
                                }
                            };
                            match prefill_res {
                                Ok(true) => {
                                    did_metal_prefill = true;
                                }
                                Ok(false) => {}
                                Err(e) => {
                                    eprintln!(
                                        "ferrox: Metal prefill attn failed, CPU fallback: {e}"
                                    );
                                    **guard = None;
                                }
                            }
                        }
                    }
                }
                if did_metal_prefill {
                    // --- MoE FFN block (batched Metal when packed Q4) ---
                    let normed2_batch: Vec<f32> = hidden_batch
                        .chunks(hidden_dim)
                        .flat_map(|h| rms_norm(h, &layer.moe.norm_weight, self.config.rms_norm_eps))
                        .collect();
                    let dense = Self::is_dense_layer(layer);
                    let router_logits_batch = if dense {
                        Vec::new()
                    } else {
                        layer.moe.router.apply_batch(&normed2_batch, batch_size)
                    };
                    let metal_ffn = if !dense {
                        Self::try_metal_moe_prefill_batch(
                            layer,
                            &normed2_batch,
                            &router_logits_batch,
                            batch_size,
                            hidden_dim,
                            &self.config,
                        )
                    } else {
                        None
                    };
                    if let Some(mut ffn_batch) = metal_ffn {
                        if let Some(post) = &layer.attn.post_ffn_norm {
                            ffn_batch = ffn_batch
                                .chunks(hidden_dim)
                                .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                                .collect();
                        }
                        for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                            *h += f;
                        }
                    } else if let Some(mut ffn_batch) =
                        Self::dense_ffn_batch(layer, &normed2_batch, batch_size, &self.config)
                    {
                        if let Some(post) = &layer.attn.post_ffn_norm {
                            ffn_batch = ffn_batch
                                .chunks(hidden_dim)
                                .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                                .collect();
                        }
                        for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                            *h += f;
                        }
                    } else if let Some(mut ffn_batch) = Self::moe_ffn_batch(
                        layer,
                        &normed2_batch,
                        &router_logits_batch,
                        batch_size,
                        hidden_dim,
                        &self.config,
                        residency.as_ref().map(|p| p.layer_plan(l)),
                    ) {
                        if let Some(post) = &layer.attn.post_ffn_norm {
                            ffn_batch = ffn_batch
                                .chunks(hidden_dim)
                                .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                                .collect();
                        }
                        for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                            *h += f;
                        }
                    } else {
                        let n_experts = layer.moe.n_experts().max(1);
                        for b in 0..batch_size {
                            let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                            let mut ffn_out = if dense {
                                Self::run_ffn_block(
                                    layer,
                                    normed2,
                                    &self.config,
                                    hidden_dim,
                                    residency.as_ref().map(|p| p.layer_plan(l)),
                                )
                            } else {
                                let router_logits =
                                    &router_logits_batch[b * n_experts..(b + 1) * n_experts];
                                Self::combine_ffn_outputs_for_position(
                                    layer,
                                    normed2,
                                    router_logits,
                                    &self.config,
                                    hidden_dim,
                                    residency.as_ref().map(|p| p.layer_plan(l)),
                                )
                            };
                            if let Some(post) = &layer.attn.post_ffn_norm {
                                ffn_out = rms_norm(&ffn_out, post, self.config.rms_norm_eps);
                            }
                            let hidden_row =
                                &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
                            for (h, f) in hidden_row.iter_mut().zip(ffn_out.iter()) {
                                *h += f;
                            }
                        }
                    }
                    l += 1;
                    continue;
                }
            }

            // RoPE per token is independent; parallelize for CPU pp512.
            q_batch
                .par_chunks_mut(q_width)
                .zip(k_batch.par_chunks_mut(kv_width))
                .enumerate()
                .for_each(|(b, (q_row, k_row))| {
                    let pos = start_pos + b;
                    for h in 0..n_heads {
                        self.apply_rope_head_layer(
                            &mut q_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                    for h in 0..n_kv_heads {
                        self.apply_rope_head_layer(
                            &mut k_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                });

            let base_seq_len = cache.seq_len;
            for b in 0..batch_size {
                cache
                    .push(
                        &k_batch[b * kv_width..(b + 1) * kv_width],
                        &v_batch[b * kv_width..(b + 1) * kv_width],
                    )
                    .expect("unbounded/planned KvCache growth is infallible");
            }

            // Prefill attention over the just-written KV prefix. Parallel
            // over query positions — the serial loop was a dominant CPU
            // pp512 bottleneck (each query still attends only its causal
            // prefix; K/V slices are immutable after the pushes above).
            let cache_k = &cache.k;
            let cache_v = &cache.v;
            let softcap = self.config.attn_logit_softcap;
            let window = self.config.layer_sliding_window(l);
            let oai = self.gpt_oss.as_ref().map(|g| &g.layers[l]);
            // gpt-oss takes the per-query path on every layer, windowed
            // or not: the blocked kernel has no sink term. Everything
            // else goes through the blocked kernel, which is Rayon over
            // `[query-block x head]` against one shared KV buffer,
            // windowed or not. SWA layers used to take a per-query
            // `causal_gqa_attention_windowed_softcap` instead, which is
            // `online_attn_accumulate`: two scalar `exp` and a
            // head_dim-wide rescale per KV position, with the head axis
            // serial inside each task. On Gemma-3-1B (22 of 26 layers
            // are SWA) that arm was 19.6% of non-idle CPU `pp512`
            // samples while doing the *same* KV work as this one - at
            // `pp512` the 512-wide window covers the whole prompt.
            let attn_out_batch = if let Some(oai) = oai {
                let mut out = vec![0f32; batch_size * q_width];
                out.par_chunks_mut(q_width)
                    .enumerate()
                    .for_each(|(b, dest)| {
                        let seq_len_b = base_seq_len + b + 1;
                        let cache_elems = seq_len_b * kv_width;
                        let attn_out = ferrox_core::causal_gqa_attention_sinks(
                            &q_batch[b * q_width..(b + 1) * q_width],
                            &cache_k[..cache_elems],
                            &cache_v[..cache_elems],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            seq_len_b,
                            window,
                            &oai.attn_sinks,
                        );
                        dest.copy_from_slice(&attn_out);
                    });
                out
            } else {
                causal_gqa_attention_prefill_shared_kv_windowed(
                    &q_batch,
                    cache_k,
                    cache_v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    batch_size,
                    base_seq_len,
                    softcap,
                    window,
                )
            };

            let mut projected_batch = layer.attn.o_proj.apply_batch(&attn_out_batch, batch_size);
            if let Some(oai) = oai {
                for row in projected_batch.chunks_mut(hidden_dim) {
                    for (x, b) in row.iter_mut().zip(oai.o_bias.iter()) {
                        *x += b;
                    }
                }
            }
            let projected_batch = if let Some(post) = &layer.attn.post_attn_norm {
                projected_batch
                    .chunks(hidden_dim)
                    .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                    .collect::<Vec<_>>()
            } else {
                projected_batch
            };
            for (h, p) in hidden_batch.iter_mut().zip(projected_batch.iter()) {
                *h += p;
            }

            // --- MoE FFN block ---
            let normed2_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| rms_norm(h, &layer.moe.norm_weight, self.config.rms_norm_eps))
                .flatten()
                .collect();
            if let Some(oai) = oai {
                // gpt-oss: one position at a time through the single
                // validated FFN. None of the batched fast paths below
                // knows about router bias, expert bias or swiglu_oai.
                for b in 0..batch_size {
                    let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    let ffn_out = Self::gpt_oss_ffn(layer, oai, normed2, &self.config, hidden_dim);
                    let hidden_row = &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    for (h, f) in hidden_row.iter_mut().zip(ffn_out.iter()) {
                        *h += f;
                    }
                }
                l += 1;
                continue;
            }
            let dense = Self::is_dense_layer(layer);
            // Skip the batched router matmul entirely for a dense
            // layer -- there's nothing to route (see
            // `is_dense_layer`'s doc comment), so computing it here
            // just to ignore it below would waste the one matmul this
            // fast path exists to avoid.
            let router_logits_batch = if dense {
                Vec::new()
            } else {
                layer.moe.router.apply_batch(&normed2_batch, batch_size)
            };
            #[cfg(feature = "metal")]
            let metal_ffn = if !dense {
                Self::try_metal_moe_prefill_batch(
                    layer,
                    &normed2_batch,
                    &router_logits_batch,
                    batch_size,
                    hidden_dim,
                    &self.config,
                )
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let metal_ffn: Option<Vec<f32>> = None;
            if let Some(mut ffn_batch) = metal_ffn {
                if let Some(post) = &layer.attn.post_ffn_norm {
                    ffn_batch = ffn_batch
                        .chunks(hidden_dim)
                        .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                        .collect();
                }
                for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                    *h += f;
                }
            } else if let Some(mut ffn_batch) =
                Self::dense_ffn_batch(layer, &normed2_batch, batch_size, &self.config)
            {
                // Dense FFN, batched. Without this the FFN -- the
                // majority of a dense model's prefill work -- ran one
                // position at a time while Q/K/V and the router were
                // already batched, which is why `pp512` measured about
                // the same as `tg128`.
                if let Some(post) = &layer.attn.post_ffn_norm {
                    ffn_batch = ffn_batch
                        .chunks(hidden_dim)
                        .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                        .collect();
                }
                for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                    *h += f;
                }
            } else if let Some(mut ffn_batch) = Self::moe_ffn_batch(
                layer,
                &normed2_batch,
                &router_logits_batch,
                batch_size,
                hidden_dim,
                &self.config,
                residency.as_ref().map(|p| p.layer_plan(l)),
            ) {
                if let Some(post) = &layer.attn.post_ffn_norm {
                    ffn_batch = ffn_batch
                        .chunks(hidden_dim)
                        .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                        .collect();
                }
                for (h, f) in hidden_batch.iter_mut().zip(ffn_batch.iter()) {
                    *h += f;
                }
            } else {
                let n_experts = layer.moe.n_experts().max(1);
                for b in 0..batch_size {
                    let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    let mut ffn_out = if dense {
                        Self::run_ffn_block(
                            layer,
                            normed2,
                            &self.config,
                            hidden_dim,
                            residency.as_ref().map(|p| p.layer_plan(l)),
                        )
                    } else {
                        let router_logits =
                            &router_logits_batch[b * n_experts..(b + 1) * n_experts];
                        Self::combine_ffn_outputs_for_position(
                            layer,
                            normed2,
                            router_logits,
                            &self.config,
                            hidden_dim,
                            residency.as_ref().map(|p| p.layer_plan(l)),
                        )
                    };
                    if let Some(post) = &layer.attn.post_ffn_norm {
                        ffn_out = rms_norm(&ffn_out, post, self.config.rms_norm_eps);
                    }
                    let hidden_row = &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    for (h, f) in hidden_row.iter_mut().zip(ffn_out.iter()) {
                        *h += f;
                    }
                }
            }
            l += 1;
        }

        hidden_batch
            .chunks(hidden_dim)
            .map(|h| rms_norm(h, &self.final_norm, self.config.rms_norm_eps))
            .collect()
    }

    /// Continuous-batching primitive: one decode step across N
    /// independent *sequences*, each contributing exactly one new
    /// token at its own current position, sharing every layer's
    /// projection/router matmuls the same way `forward_batch` shares
    /// them across positions of a single sequence -- but each
    /// sequence keeps its own `KvCache`, independent `seq_len`, and
    /// independent position, so sequences admitted/evicted at
    /// different times can still share one batched matmul per step
    /// (this is what "continuous" batching means: the batch
    /// membership can change every step, unlike `forward_batch`'s
    /// fixed-size prompt-processing batch). `kv_caches[s][l]` is
    /// sequence `s`'s layer-`l` cache; `tokens[s]`/`positions[s]` is
    /// that sequence's next token and its position within its own
    /// history. Returns one logits vector per sequence, same order as
    /// `tokens`.
    ///
    /// Must produce bit-identical output to calling `forward_token`
    /// once per sequence with that sequence's own cache/position --
    /// batching independent sequences together is a scheduling detail,
    /// not a math change (no sequence's attention ever reads another
    /// sequence's cache).
    pub fn forward_multi_seq(
        &self,
        tokens: &[usize],
        positions: &[usize],
        kv_caches: &mut [Vec<KvCache>],
    ) -> Vec<Vec<f32>> {
        assert_eq!(tokens.len(), positions.len());
        assert_eq!(tokens.len(), kv_caches.len());
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }
        for seq in kv_caches.iter() {
            assert_eq!(seq.len(), self.layers.len());
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens);

        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            let normed_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| rms_norm(h, &layer.attn.norm_weight, self.config.rms_norm_eps))
                .flatten()
                .collect();

            // One shared activation-quant pass for q/k/v (plan 1e): the
            // three projections read the same normed batch, so quantize it
            // once instead of once per projection. A kind mismatch inside
            // the group just re-quantizes locally.
            let qkv_acts = layer
                .attn
                .q_proj
                .quantize_batch_acts(&normed_batch, batch_size);
            let mut q_batch = layer.attn.q_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            let mut k_batch = layer.attn.k_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            let mut v_batch = layer.attn.v_proj.apply_batch_with_acts(
                &normed_batch,
                batch_size,
                qkv_acts.as_ref(),
            );
            drop(qkv_acts);

            let q_width = n_heads * head_dim;
            let kv_width = n_kv_heads * head_dim;

            if let Some(bias) = &layer.attn.q_bias {
                for row in q_batch.chunks_mut(q_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }
            if let Some(bias) = &layer.attn.k_bias {
                for row in k_batch.chunks_mut(kv_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }
            if let Some(bias) = &layer.attn.v_bias {
                for row in v_batch.chunks_mut(kv_width) {
                    for (x, b) in row.iter_mut().zip(bias.iter()) {
                        *x += b;
                    }
                }
            }

            if let Some(q_norm) = &layer.attn.q_norm {
                for row in q_batch.chunks_mut(q_width) {
                    let normed = self.apply_qk_norm(row, q_norm);
                    row.copy_from_slice(&normed);
                }
            }
            if let Some(k_norm) = &layer.attn.k_norm {
                for row in k_batch.chunks_mut(kv_width) {
                    let normed = self.apply_qk_norm(row, k_norm);
                    row.copy_from_slice(&normed);
                }
            }
            self.apply_rope_attn_factor(&mut q_batch, &mut k_batch);

            for b in 0..batch_size {
                let pos = positions[b];
                let q_row = &mut q_batch[b * q_width..(b + 1) * q_width];
                for h in 0..n_heads {
                    self.apply_rope_head_layer(
                        &mut q_row[h * head_dim..(h + 1) * head_dim],
                        pos,
                        l,
                    );
                }
                let k_row = &mut k_batch[b * kv_width..(b + 1) * kv_width];
                for h in 0..n_kv_heads {
                    self.apply_rope_head_layer(
                        &mut k_row[h * head_dim..(h + 1) * head_dim],
                        pos,
                        l,
                    );
                }
            }

            let oai = self.gpt_oss.as_ref().map(|g| &g.layers[l]);
            let mut attn_out_batch = vec![0f32; batch_size * q_width];
            for b in 0..batch_size {
                let cache = &mut kv_caches[b][l];
                cache
                    .push(
                        &k_batch[b * kv_width..(b + 1) * kv_width],
                        &v_batch[b * kv_width..(b + 1) * kv_width],
                    )
                    .expect("unbounded/planned KvCache growth is infallible");
                if let Some(oai) = oai {
                    let attn_out = ferrox_core::causal_gqa_attention_sinks(
                        &q_batch[b * q_width..(b + 1) * q_width],
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                        self.config.layer_sliding_window(l),
                        &oai.attn_sinks,
                    );
                    attn_out_batch[b * q_width..(b + 1) * q_width].copy_from_slice(&attn_out);
                    continue;
                }
                let attn_out = match self.config.layer_sliding_window(l) {
                    Some(window) => causal_gqa_attention_windowed_softcap(
                        &q_batch[b * q_width..(b + 1) * q_width],
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                        window,
                        self.config.attn_logit_softcap,
                    ),
                    None => causal_gqa_attention_softcap(
                        &q_batch[b * q_width..(b + 1) * q_width],
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.seq_len,
                        self.config.attn_logit_softcap,
                    ),
                };
                attn_out_batch[b * q_width..(b + 1) * q_width].copy_from_slice(&attn_out);
            }

            let mut projected_batch = layer.attn.o_proj.apply_batch(&attn_out_batch, batch_size);
            if let Some(oai) = oai {
                for row in projected_batch.chunks_mut(hidden_dim) {
                    for (x, b) in row.iter_mut().zip(oai.o_bias.iter()) {
                        *x += b;
                    }
                }
            }
            let projected_batch = if let Some(post) = &layer.attn.post_attn_norm {
                projected_batch
                    .chunks(hidden_dim)
                    .flat_map(|row| rms_norm(row, post, self.config.rms_norm_eps))
                    .collect::<Vec<_>>()
            } else {
                projected_batch
            };
            for (h, p) in hidden_batch.iter_mut().zip(projected_batch.iter()) {
                *h += p;
            }

            // --- MoE FFN block ---
            let normed2_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| rms_norm(h, &layer.moe.norm_weight, self.config.rms_norm_eps))
                .flatten()
                .collect();
            let dense = Self::is_dense_layer(layer);
            let router_logits_batch = if dense || oai.is_some() {
                Vec::new()
            } else {
                layer.moe.router.apply_batch(&normed2_batch, batch_size)
            };
            let n_experts = layer.moe.n_experts().max(1);

            for b in 0..batch_size {
                let normed2 = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                let mut ffn_out = if let Some(oai) = oai {
                    Self::gpt_oss_ffn(layer, oai, normed2, &self.config, hidden_dim)
                } else if dense {
                    Self::run_ffn_block(
                        layer,
                        normed2,
                        &self.config,
                        hidden_dim,
                        residency.as_ref().map(|p| p.layer_plan(l)),
                    )
                } else {
                    let router_logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
                    Self::combine_ffn_outputs_for_position(
                        layer,
                        normed2,
                        router_logits,
                        &self.config,
                        hidden_dim,
                        residency.as_ref().map(|p| p.layer_plan(l)),
                    )
                };
                if let Some(post) = &layer.attn.post_ffn_norm {
                    ffn_out = rms_norm(&ffn_out, post, self.config.rms_norm_eps);
                }
                let hidden_row = &mut hidden_batch[b * hidden_dim..(b + 1) * hidden_dim];
                for (h, f) in hidden_row.iter_mut().zip(ffn_out.iter()) {
                    *h += f;
                }
            }
        }

        let final_normed_batch: Vec<f32> = hidden_batch
            .par_chunks(hidden_dim)
            .map(|h| rms_norm(h, &self.final_norm, self.config.rms_norm_eps))
            .flatten()
            .collect();
        self.logits_from_flat_hidden(final_normed_batch, batch_size)
    }
}

#[cfg(test)]
mod partial_rotary_tests {
    use super::*;

    /// Phi-3/Phi-4 rotate `rope.dimension_count` of each head and pass
    /// the rest through. The tail staying bit-identical is the whole
    /// property: rotating it would make dimensions position-dependent
    /// that the model never trained that way.
    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = crate::config::RopeLayout::Neox;
        cfg.rope_freqs = None;
        cfg.rope_dim = Some(4);
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut head: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let before = head.clone();
        decoder.apply_rope_head_theta(&mut head, 3, 10000.0);

        assert_eq!(
            &head[4..],
            &before[4..],
            "dims at or past rope_dim must not rotate"
        );
        assert!(
            head[..4] != before[..4],
            "dims below rope_dim must rotate at a non-zero position"
        );
    }

    /// `attn_factor` is a magnitude scale folded into cos/sin inside
    /// ggml's `rope_yarn`, so it can only ever touch the rotated
    /// channels. The pass-through tail must come out bit-identical —
    /// scaling it is a different graph, and it was one, until
    /// `ferrox parity` reported Phi-4-mini as the single DRIFT in a
    /// 17-model sweep against llama.cpp.
    #[test]
    fn attn_factor_scales_only_the_rotated_channels() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.rope_dim = Some(4);
        cfg.rope_attn_factor = 2.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        // Two heads, so a per-head slice bug cannot hide behind a single
        // head that happens to span the whole buffer.
        let mut q: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
        let mut k: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
        let before = q.clone();
        decoder.apply_rope_attn_factor(&mut q, &mut k);

        for h in 0..2 {
            let base = h * 8;
            for i in 0..4 {
                assert_eq!(
                    q[base + i],
                    before[base + i] * 2.0,
                    "rotated channel {i} of head {h} must be scaled"
                );
            }
            for i in 4..8 {
                assert_eq!(
                    q[base + i],
                    before[base + i],
                    "pass-through channel {i} of head {h} must be untouched"
                );
            }
        }
        assert_eq!(q, k, "q and k take the same magnitude scale");
    }

    /// With no partial rotary the whole head is rotated, so the whole
    /// head takes the scale — the narrow case must not become the rule.
    #[test]
    fn attn_factor_scales_the_whole_head_without_partial_rotary() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.rope_dim = None;
        cfg.rope_attn_factor = 3.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut q: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let mut k = q.clone();
        let before = q.clone();
        decoder.apply_rope_attn_factor(&mut q, &mut k);
        for i in 0..8 {
            assert_eq!(q[i], before[i] * 3.0);
        }
    }

    /// The same call with no `rope_dim` must rotate everything, so the
    /// narrow case cannot silently become the default.
    #[test]
    fn full_rotary_still_rotates_the_whole_head() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = crate::config::RopeLayout::Neox;
        cfg.rope_freqs = None;
        cfg.rope_dim = None;
        let decoder = Decoder::new_random_small(cfg, 1, 32);

        let mut head: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let before = head.clone();
        decoder.apply_rope_head_theta(&mut head, 3, 10000.0);
        assert!(head[4..] != before[4..]);
    }

    /// `mscale` scales q and k and nothing else; `1.0` must be a literal
    /// no-op so every other model pays nothing.
    #[test]
    fn rope_attn_factor_scales_q_and_k_only() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 2.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        let mut q = vec![1.0f32, -2.0, 3.0];
        let mut k = vec![0.5f32, 4.0];
        decoder.apply_rope_attn_factor(&mut q, &mut k);
        assert_eq!(q, vec![2.0, -4.0, 6.0]);
        assert_eq!(k, vec![1.0, 8.0]);

        let mut cfg = crate::config::test_dense_fixture();
        cfg.rope_attn_factor = 1.0;
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        let mut q = vec![1.0f32, -2.0];
        let mut k = vec![3.0f32];
        decoder.apply_rope_attn_factor(&mut q, &mut k);
        assert_eq!(q, vec![1.0, -2.0]);
        assert_eq!(k, vec![3.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::glm_5_2;
    use ferrox_core::cache::PagedKvStore;

    /// Small config used purely to keep the test fast: same
    /// architecture *shape* (GQA ratio, MoE topology) as GLM-5.2, but
    /// with tiny dims so the whole thing runs in milliseconds.
    fn tiny_test_config() -> ModelConfig {
        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        cfg
    }

    #[test]
    fn forward_pass_produces_finite_logits_of_correct_shape() {
        let vocab = 10;
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, vocab);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        let logits = decoder.forward_token(3, 0, &mut caches);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must not contain NaN/Inf"
        );
    }

    /// `gpu_vram_budget_bytes` must be a real zero-behavior-change
    /// default at `None`, and a *real placement plan that places
    /// nothing* (a zero VRAM budget, so `PlacementPlan::from_budget`
    /// fits no expert at all) must produce byte-identical output to
    /// `None` too -- proving the new plumbing (building a plan,
    /// looking up each routed expert's placement, dispatching through
    /// `run_expert_placed`) doesn't change results when nothing is
    /// actually GPU-placed, without needing real CUDA hardware to
    /// check (that hardware-dependent half is
    /// `ferrox-moe`'s/`ferrox-core`'s own `#[ignore]`d tests).
    #[test]
    fn gpu_vram_budget_bytes_with_nothing_placed_matches_the_default() {
        let mut decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let mut caches_default: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let default_logits = decoder.forward_token(3, 0, &mut caches_default);

        decoder.gpu_vram_budget_bytes = Some(0);
        let mut caches_zero_budget: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let zero_budget_logits = decoder.forward_token(3, 0, &mut caches_zero_budget);

        assert_eq!(
            default_logits, zero_budget_logits,
            "a placement plan that places nothing on GPU must match the None default exactly"
        );
    }

    /// Qwen2-MoE's real shared-expert sigmoid gate
    /// (`MoeWeights::shared_expert_gate`): exact math check by mutating
    /// `layer.moe.shared_expert_gate` in place on an already-built
    /// decoder (no need to reconstruct a `LayerWeights`/`MoeWeights`
    /// from scratch) and comparing against a hand-derived expectation:
    /// the *only* thing the gate changes is the shared experts' own
    /// contribution, scaled by `sigmoid(gate . x)` -- so
    /// `gated_shared_output == ungated_shared_output * sigmoid_value`
    /// exactly, computed independently here via `run_expert` on the
    /// same layer's shared expert.
    #[test]
    fn shared_expert_gate_scales_shared_output_by_sigmoid_of_the_gate_logit() {
        let cfg = tiny_test_config();
        let mut decoder = Decoder::new_random_small(cfg, 2, 8);
        let hidden_dim = decoder.config.hidden_dim;
        assert_eq!(
            decoder.layers[1].moe.shared_experts.len(),
            1,
            "test assumes tiny_test_config's real MoE layer has exactly one shared expert"
        );

        let normed2: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let gate_vec: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.13 - 0.5).collect();

        // Independently compute what the shared expert alone produces,
        // and what sigmoid(gate . x) should scale it by -- this is the
        // ground truth the gated code path must reproduce exactly.
        let shared_out_raw = run_expert(&normed2, &decoder.layers[1].moe.shared_experts[0]);
        let gate_logit: f32 = gate_vec
            .iter()
            .zip(normed2.iter())
            .map(|(g, x)| g * x)
            .sum();
        let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
        let expected_gated_shared: Vec<f32> =
            shared_out_raw.iter().map(|x| x * gate_value).collect();

        // Run the real FFN combine path twice (gate absent, then
        // present) and recover each run's shared-only contribution by
        // subtracting the routed contribution, which the gate never
        // touches and is identical between the two runs (same router,
        // same experts, same input).
        let router_logits = decoder.layers[1].moe.router.apply(&normed2);
        let ungated_total = Decoder::combine_ffn_outputs_for_position(
            &decoder.layers[1],
            &normed2,
            &router_logits,
            &decoder.config,
            hidden_dim,
            None,
        );
        decoder.layers[1].moe.shared_expert_gate = Some(gate_vec);
        let gated_total = Decoder::combine_ffn_outputs_for_position(
            &decoder.layers[1],
            &normed2,
            &router_logits,
            &decoder.config,
            hidden_dim,
            None,
        );

        for (i, ((u, g), expected_shared)) in ungated_total
            .iter()
            .zip(gated_total.iter())
            .zip(expected_gated_shared.iter())
            .enumerate()
        {
            let routed_contribution = u - shared_out_raw[i];
            let gated_shared_recovered = g - routed_contribution;
            assert!(
                (gated_shared_recovered - expected_shared).abs() < 1e-4,
                "index {i}: recovered gated shared output {gated_shared_recovered} != expected {expected_shared} (sigmoid({gate_logit})={gate_value})"
            );
        }
    }

    #[test]
    fn kv_cache_grows_by_one_position_per_layer_per_step() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 3, 5);
        let mut caches: Vec<KvCache> = (0..3)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        decoder.forward_token(0, 0, &mut caches);
        decoder.forward_token(1, 1, &mut caches);
        decoder.forward_token(2, 2, &mut caches);

        for cache in &caches {
            assert_eq!(cache.seq_len, 3);
        }
    }

    #[test]
    fn same_token_same_position_is_deterministic() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        let out_a = decoder.forward_token(4, 0, &mut caches_a);
        let out_b = decoder.forward_token(4, 0, &mut caches_b);
        assert_eq!(out_a, out_b, "identical input state must yield identical output (no hidden randomness in the forward pass)");
    }

    #[test]
    fn multi_step_decode_stays_finite_across_positions() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        for pos in 0..16 {
            let logits = decoder.forward_token(pos % 8, pos, &mut caches);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "position {pos}: logits must stay finite across an extended decode run"
            );
        }
    }

    /// `forward_token_paged` must produce bit-identical output to
    /// `forward_token` across a multi-step decode (each layer's paged
    /// store sized generously so no layer ever exhausts its blocks) --
    /// the block-table indirection is a storage-layout detail, not a
    /// math change.
    #[test]
    fn forward_token_paged_matches_forward_token_bit_identical() {
        paged_matches_contiguous(tiny_test_config());
    }

    /// Every arm of the attention dispatch, not just the plain one.
    ///
    /// The paged path used to implement only full causal attention, and
    /// `forward_token_paged` asserted rather than run gpt-oss, because a
    /// missing sink term would have changed the distribution silently.
    /// Now that it mirrors all three arms, each one has to be held to
    /// the same bar the plain arm always was: BIT-identical, not close.
    ///
    /// A sliding window and a softcap are both driven from the config
    /// here, so a future edit that wires one arm and forgets another
    /// fails on the arm it forgot rather than on a model nobody tests.
    #[test]
    fn every_paged_attention_arm_is_bit_identical_to_its_contiguous_twin() {
        let windowed = || {
            let mut cfg = tiny_test_config();
            // Smaller than the decode length below, so the window really
            // drops positions rather than degenerating to full causal.
            cfg.sliding_window = Some(2);
            cfg.swa_pattern = None;
            cfg
        };
        let softcapped = || {
            let mut cfg = tiny_test_config();
            // Small enough that `sc * tanh(s / sc)` actually compresses.
            // A realistic 30.0 is numerically indistinguishable from no
            // cap at these tiny weights, so a test using it would pass
            // whether or not the arm was wired -- checked by breaking
            // the arm on purpose and watching it still pass.
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        let both = || {
            let mut cfg = windowed();
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        // Alternating window/full layers: the per-layer arm choice has
        // to be honoured per layer, not decided once for the model.
        let alternating = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_pattern = Some(2);
            cfg
        };

        for cfg in [windowed(), softcapped(), both(), alternating()] {
            paged_matches_contiguous(cfg);
        }
    }

    /// The two rules that live OUTSIDE the layer loop, which the arm
    /// test above cannot reach.
    ///
    /// The paged path had drifted from the contiguous one at both ends
    /// of the stack, and neither drift was visible to any existing test
    /// because `tiny_test_config` sets neither field:
    ///
    /// - it called `embedding.dequant_row` directly instead of scaling
    ///   the row by `embedding_scale`, so every Gemma token entered the
    ///   stack `sqrt(hidden_dim)` times too small;
    /// - it returned `output_head.apply(..)` raw instead of applying
    ///   `final_logit_softcap`, so Gemma-2's 30.0 cap never ran.
    ///
    /// Both produce a plausible distribution rather than an error, which
    /// is the whole reason to pin them: a wrong answer that still looks
    /// like an answer is what a parity test is for. Values here are
    /// chosen so each one actually bites -- a scale of 1.0 or a cap far
    /// above the logit range would let this pass either way.
    #[test]
    fn the_paged_path_scales_embeddings_and_softcaps_logits_like_the_contiguous_one() {
        let scaled = || {
            let mut cfg = tiny_test_config();
            cfg.embedding_scale = Some(7.5);
            cfg
        };
        let capped = || {
            let mut cfg = tiny_test_config();
            // Small enough that `sc * tanh(x / sc)` really compresses at
            // this model's logit magnitudes, on the same reasoning as
            // the attention softcap above.
            cfg.final_logit_softcap = Some(0.05);
            cfg
        };
        let both = || {
            let mut cfg = scaled();
            cfg.final_logit_softcap = Some(0.05);
            cfg
        };

        for cfg in [scaled(), capped(), both()] {
            paged_matches_contiguous(cfg);
        }
    }

    /// Paged prefill must agree with contiguous prefill, and must leave
    /// the KV in a state a paged DECODE can continue from.
    ///
    /// The second half is the one worth having. `forward_batch_last`
    /// returns only the last row's logits, so a gather/scatter that
    /// mangled the KV -- wrote the rows in the wrong order, dropped the
    /// part-full tail block, mis-sized a copy -- could still return the
    /// right logits for THIS call and only surface on the next token.
    /// Decoding four more tokens after the prefill is what makes the
    /// stored KV observable, so both paths are compared over the whole
    /// continuation rather than at the seam.
    ///
    /// A block size of 2 against a 5-token prompt is deliberate: it
    /// leaves the tail block part-full, which is the case
    /// `blocks_needed_for` exists for and the one a `n / block_size`
    /// reservation would get wrong.
    fn paged_prefill_matches_contiguous(config: ModelConfig) {
        let n_layers = 2;
        let decoder = Decoder::new_random_small(config, n_layers, 10);
        let prompt = [3usize, 1, 4, 1, 5];
        let continuation = [9usize, 2, 6, 5];

        let mut caches: Vec<KvCache> = (0..n_layers)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut plain = vec![decoder.forward_batch_last(&prompt, 0, &mut caches)];
        for (i, &tok) in continuation.iter().enumerate() {
            plain.push(decoder.forward_token(tok, prompt.len() + i, &mut caches));
        }

        let mut paged_caches: Vec<PagedKvCache> =
            (0..n_layers).map(|_| PagedKvCache::new()).collect();
        let stores = SharedPagedKv::from_stores(
            (0..n_layers)
                .map(|_| {
                    PagedKvStore::new(
                        /* block_size = */ 2,
                        /* total_blocks = */ 16,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );
        let mut paged = vec![decoder
            .forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores)
            .expect("store sized generously, must not exhaust")];
        for (i, &tok) in continuation.iter().enumerate() {
            paged.push(
                decoder
                    .forward_token_paged(tok, prompt.len() + i, &mut paged_caches, &stores)
                    .expect("store sized generously, must not exhaust"),
            );
        }

        assert_eq!(
            paged_caches[0].seq_len(),
            prompt.len() + continuation.len(),
            "paged prefill must advance seq_len by exactly the batch size"
        );
        assert_eq!(plain.len(), paged.len());
        for (step, (a, b)) in plain.iter().zip(paged.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "step {step}: logit count");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "step {step}: paged prefill + decode must be bit-identical to contiguous"
                );
            }
        }
    }

    /// Every arm again, this time through the prefill entry point. The
    /// gather is shared, but the kernel the gathered buffer reaches is
    /// the BLOCKED prefill one rather than the per-query decode one, so
    /// arm coverage here is not implied by the decode tests above.
    #[test]
    fn paged_prefill_is_bit_identical_across_every_arm() {
        let windowed = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_pattern = None;
            cfg
        };
        let scaled_and_capped = || {
            let mut cfg = tiny_test_config();
            cfg.embedding_scale = Some(7.5);
            cfg.final_logit_softcap = Some(0.05);
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        let alternating = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_pattern = Some(2);
            cfg
        };

        for cfg in [
            tiny_test_config(),
            windowed(),
            scaled_and_capped(),
            alternating(),
        ] {
            paged_prefill_matches_contiguous(cfg);
        }
    }

    /// A prefill the stores cannot hold refuses having written NOTHING
    /// -- checked on the case that actually needs the up-front loop.
    ///
    /// Each layer owns its own store, so layer 0 having room says
    /// nothing about layer 1. `append_contiguous` already refuses
    /// rather than half-writing a single layer, so a test whose layers
    /// are sized alike passes with the cross-layer reservation deleted
    /// -- it would be asserting a property it never exercises. Here
    /// layer 0 has room for the whole prompt and layer 1 does not, so
    /// without the up-front check layer 0 is written, layer 1 refuses,
    /// and the sequence ends up with its layers at DIFFERENT lengths.
    /// No caller can recover from that, and nothing downstream would
    /// report it: the next decode step simply attends over a shorter
    /// history in one layer than the others.
    ///
    /// Verified by deleting the reservation loop and watching this fail
    /// on `layer 1 must be untouched`.
    /// Three requests sharing one set of per-layer stores must get
    /// exactly what they would get alone.
    ///
    /// This is the property the RwLock exists for, and it cannot be
    /// asserted single-threaded. Every request writes only blocks it
    /// owns, so sharing changes where rows live and nothing else --
    /// bit-identical, not close. A store that let one request's rows
    /// land in another's blocks shows up here and nowhere else.
    #[test]
    fn concurrent_decodes_against_one_shared_store_match_running_them_alone() {
        use std::sync::Arc;

        let decoder = Arc::new(Decoder::new_random_small(tiny_test_config(), 2, 10));
        let prompts: [&[usize]; 3] = [&[3, 1, 4], &[1, 5, 9], &[2, 6, 5]];
        let continuation = [7usize, 8, 3];

        // Each request run alone, against its own store, is the answer
        // sharing must not change.
        let solo: Vec<Vec<Vec<f32>>> = prompts
            .iter()
            .map(|prompt| {
                let stores = SharedPagedKv::new(
                    2,
                    4,
                    32,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                );
                let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
                run_one(&decoder, prompt, &continuation, &mut caches, &stores)
            })
            .collect();

        // The same three, concurrently, sharing ONE set of per-layer
        // stores. Every request writes only blocks it owns, so the
        // answers must be identical -- not close, identical. A store
        // that let one request's rows land in another's blocks would
        // show up here and nowhere else.
        let shared = Arc::new(SharedPagedKv::new(
            2,
            4,
            96,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        ));
        let together: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = prompts
                .iter()
                .map(|prompt| {
                    let decoder = Arc::clone(&decoder);
                    let shared = Arc::clone(&shared);
                    scope.spawn(move || {
                        let mut caches: Vec<PagedKvCache> =
                            (0..2).map(|_| PagedKvCache::new()).collect();
                        run_one(&decoder, prompt, &continuation, &mut caches, &shared)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (r, (alone, concurrent)) in solo.iter().zip(together.iter()).enumerate() {
            assert_eq!(alone.len(), concurrent.len(), "request {r}: step count");
            for (step, (a, b)) in alone.iter().zip(concurrent.iter()).enumerate() {
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "request {r} step {step}: sharing a store changed the answer"
                    );
                }
            }
        }
    }

    /// Prefill then decode, returning every step's logits.
    fn run_one(
        decoder: &Decoder,
        prompt: &[usize],
        continuation: &[usize],
        caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Vec<Vec<f32>> {
        let mut out = vec![decoder
            .forward_batch_last_paged(prompt, 0, caches, stores)
            .expect("sized generously")];
        for (i, &tok) in continuation.iter().enumerate() {
            out.push(
                decoder
                    .forward_token_paged(tok, prompt.len() + i, caches, stores)
                    .expect("sized generously"),
            );
        }
        out
    }

    /// A decode step the stores cannot hold advances NO layer.
    ///
    /// This was a real defect until the reservation moved into
    /// `forward_token_paged`: it pushed per layer with `?`, so a store
    /// exhausting at layer 1 of 2 left layer 0 holding a position layer
    /// 1 did not. Nothing downstream reports that -- the next step just
    /// attends over a shorter history in the tail layers -- and the
    /// prefill path had the guard while decode never did.
    ///
    /// Layer 0 is given room and layer 1 none, so the bug is reachable:
    /// with the reservation removed, layer 0 advances and layer 1
    /// refuses.
    #[test]
    fn a_decode_step_the_stores_cannot_hold_advances_no_layer() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
        // Block size 1 so "one more position" always needs a block.
        // Layer 0 gets two, layer 1 exactly one: the prompt fills layer
        // 1 completely, so the decode step below cannot fit there.
        let stores = SharedPagedKv::from_stores(
            [2usize, 1]
                .into_iter()
                .map(|blocks| {
                    PagedKvStore::new(
                        1,
                        blocks,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );

        decoder
            .forward_batch_last_paged(&[1usize], 0, &mut caches, &stores)
            .expect("one position fits in both layers");
        assert_eq!(caches[0].seq_len(), 1);
        assert_eq!(caches[1].seq_len(), 1);

        let result = decoder.forward_token_paged(2, 1, &mut caches, &stores);
        assert!(result.is_err(), "layer 1 has no block left");
        assert_eq!(
            caches[0].seq_len(),
            1,
            "layer 0 must not advance past a layer that could not"
        );
        assert_eq!(caches[1].seq_len(), 1);
    }

    #[test]
    fn a_prefill_the_stores_cannot_hold_refuses_before_writing_any_layer() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let prompt = [1usize, 2, 3, 4, 5, 6];
        let mut paged_caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
        // Layer 0 fits the prompt with room to spare; layer 1's two
        // blocks of 2 hold 4 positions against a prompt of 6.
        let stores = SharedPagedKv::from_stores(
            [8usize, 2]
                .into_iter()
                .map(|blocks| {
                    PagedKvStore::new(
                        2,
                        blocks,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );

        let result = decoder.forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores);
        assert!(result.is_err(), "layer 1's store cannot hold the prompt");
        for (i, cache) in paged_caches.iter().enumerate() {
            assert_eq!(cache.seq_len(), 0, "layer {i} must be untouched");
            assert!(cache.block_table().is_empty(), "layer {i} holds no block");
        }
        for (i, expected) in [8usize, 2].into_iter().enumerate() {
            assert_eq!(stores.free_blocks(i), expected, "layer {i} leaked no block");
        }
    }

    /// Chunked prefill: two calls appending into the same sequence must
    /// equal one call over the concatenation.
    ///
    /// This is the case the part-full tail block breaks if
    /// `to_contiguous` or the reservation is wrong, and it is how the
    /// serving path actually prefills long prompts.
    #[test]
    fn two_paged_prefill_chunks_equal_one_call_over_the_whole_prompt() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let prompt = [3usize, 1, 4, 1, 5, 9, 2];
        let split = 3;

        let run = |chunks: &[&[usize]]| {
            let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
            let stores = SharedPagedKv::from_stores(
                (0..2)
                    .map(|_| {
                        PagedKvStore::new(2, 16, decoder.config.n_kv_heads, decoder.config.head_dim)
                    })
                    .collect(),
            );
            let mut pos = 0;
            let mut last = Vec::new();
            for chunk in chunks {
                last = decoder
                    .forward_batch_last_paged(chunk, pos, &mut caches, &stores)
                    .expect("sized generously");
                pos += chunk.len();
            }
            last
        };

        let whole = run(&[&prompt]);
        let chunked = run(&[&prompt[..split], &prompt[split..]]);
        assert_eq!(whole.len(), chunked.len());
        for (x, y) in whole.iter().zip(chunked.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "a chunked prefill must equal one call over the same tokens"
            );
        }
    }

    fn paged_matches_contiguous(config: ModelConfig) {
        let n_layers = 2;
        let decoder = Decoder::new_random_small(config, n_layers, 10);

        let mut caches: Vec<KvCache> = (0..n_layers)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let steps = [3usize, 5, 7, 2, 9, 1];
        let mut plain_logits = Vec::new();
        for (pos, &tok) in steps.iter().enumerate() {
            plain_logits.push(decoder.forward_token(tok, pos, &mut caches));
        }

        let block_size = 2;
        let mut paged_caches: Vec<PagedKvCache> =
            (0..n_layers).map(|_| PagedKvCache::new()).collect();
        let stores = SharedPagedKv::from_stores(
            (0..n_layers)
                .map(|_| {
                    PagedKvStore::new(
                        block_size,
                        /* total_blocks = */ 16,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );
        let mut paged_logits = Vec::new();
        for (pos, &tok) in steps.iter().enumerate() {
            paged_logits.push(
                decoder
                    .forward_token_paged(tok, pos, &mut paged_caches, &stores)
                    .expect("store sized generously, must not exhaust"),
            );
        }

        assert_eq!(plain_logits.len(), paged_logits.len());
        for (a, b) in plain_logits.iter().zip(paged_logits.iter()) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "paged decode must be bit-identical to contiguous decode"
                );
            }
        }
    }

    /// The single most important correctness property of
    /// `forward_batch`: batching positions together for shared matmuls
    /// must produce EXACTLY the same result as processing them one at
    /// a time with `forward_token`, since causal masking guarantees
    /// position `i` only ever sees positions `<= i`. If this test
    /// fails, `forward_batch` is not a safe drop-in replacement for
    /// sequential decode, which would make speculative decoding built
    /// on top of it produce silently wrong output.
    #[test]
    fn forward_batch_matches_sequential_forward_token_exactly() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        // A second decoder built with the same seed produces identical
        // weights (Decoder::new_random_small is deterministic), so
        // this is a fair like-for-like comparison against a fresh
        // cache rather than reusing decoder_a's now-mutated cache.
        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(seq_logits.len(), batch_logits.len());
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-3,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    /// `forward_batch_last` exists to skip the vocabulary projection for
    /// every position but the last, so the one thing that must hold is
    /// that the row it *does* produce is the same row `forward_batch`
    /// would have produced. It must also leave the KV cache in the same
    /// state -- prefill's whole purpose -- which is checked by decoding
    /// one more token from each cache and comparing.
    #[test]
    fn forward_batch_last_matches_the_final_row_of_forward_batch() {
        let cfg = tiny_test_config();
        let vocab = 16;
        let tokens = vec![1usize, 4, 7, 2, 9];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let all_rows = decoder_a.forward_batch(&tokens, 0, &mut caches_a);

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let last = decoder_b.forward_batch_last(&tokens, 0, &mut caches_b);

        let expected = all_rows.last().expect("one row per prompt token");
        assert_eq!(last.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(last.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "logit {i}: forward_batch={a} forward_batch_last={b}"
            );
        }

        // Same KV state: the next token's logits must agree too.
        let next_a = decoder_a.forward_token(3, tokens.len(), &mut caches_a);
        let next_b = decoder_b.forward_token(3, tokens.len(), &mut caches_b);
        for (i, (a, b)) in next_a.iter().zip(next_b.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "post-prefill decode logit {i}: {a} vs {b}"
            );
        }

        // Empty prompt is the degenerate case both paths must survive.
        let mut caches_c: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        assert!(decoder_b
            .forward_batch_last(&[], 0, &mut caches_c)
            .is_empty());
    }

    /// `forward_multi_seq`'s core correctness property: batching N
    /// independent sequences (different token histories, different
    /// current positions, different KV caches) together must produce
    /// EXACTLY the same per-sequence output as running each sequence
    /// through `forward_token` alone, one step at a time. This is what
    /// makes continuous batching safe -- no sequence's attention may
    /// ever be perturbed by another sequence sharing its batched
    /// matmul step.
    #[test]
    fn forward_multi_seq_matches_independent_forward_token_per_sequence() {
        let cfg = tiny_test_config();
        let vocab = 8;
        // 3 independent sequences, deliberately different lengths/
        // histories/current tokens, so no two sequences are at the
        // same position when batched together.
        let seq_histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut independent_logits: Vec<Vec<f32>> = Vec::new();
        for history in seq_histories.iter() {
            let mut caches: Vec<KvCache> = (0..2)
                .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
                .collect();
            let mut logits = Vec::new();
            for (pos, &tok) in history.iter().enumerate() {
                logits = decoder_a.forward_token(tok, pos, &mut caches);
            }
            independent_logits.push(logits);
        }

        // Same seed -> identical weights, fresh caches for a fair
        // comparison (mirrors forward_batch_matches_sequential_forward_token_exactly).
        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut per_seq_caches: Vec<Vec<KvCache>> = seq_histories
            .iter()
            .map(|_| {
                (0..2)
                    .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
                    .collect()
            })
            .collect();

        // Feed every sequence's prefix (all but its last token)
        // through forward_multi_seq one shared step at a time, then
        // do a final batched step for the last token of every
        // sequence so all three arrive at their final position in
        // the same batched call -- exercising genuinely different
        // per-sequence positions/histories within one batch, not just
        // parallel identical-length sequences.
        let max_len = seq_histories.iter().map(|h| h.len()).max().unwrap();
        let mut batched_logits: Vec<Vec<f32>> = vec![Vec::new(); seq_histories.len()];
        for step in 0..max_len {
            let mut tokens = Vec::new();
            let mut positions = Vec::new();
            let mut active: Vec<usize> = Vec::new();
            for (s, history) in seq_histories.iter().enumerate() {
                if step < history.len() {
                    tokens.push(history[step]);
                    positions.push(step);
                    active.push(s);
                }
            }
            if tokens.is_empty() {
                continue;
            }
            let mut active_caches: Vec<Vec<KvCache>> = active
                .iter()
                .map(|&s| std::mem::take(&mut per_seq_caches[s]))
                .collect();
            let step_logits = decoder_b.forward_multi_seq(&tokens, &positions, &mut active_caches);
            for ((&s, caches), logits) in active.iter().zip(active_caches).zip(step_logits) {
                per_seq_caches[s] = caches;
                batched_logits[s] = logits;
            }
        }

        assert_eq!(batched_logits.len(), independent_logits.len());
        for (s, (seq_logits, batch_logits)) in independent_logits
            .iter()
            .zip(batched_logits.iter())
            .enumerate()
        {
            assert_eq!(seq_logits.len(), batch_logits.len());
            for (i, (a, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "sequence {s}, logit {i}: independent={a} batched={b}"
                );
            }
        }
    }

    /// OLMoE-style QK-norm (`attn_q_norm`/`attn_k_norm`, see `AttnWeights`'
    /// doc comment): with both set, `forward_batch` must still match
    /// sequential `forward_token` calls exactly -- the same consistency
    /// property `forward_batch_matches_sequential_forward_token_exactly`
    /// checks for the no-QK-norm path, now exercising the norm-applied
    /// per-row slicing (`q_batch.chunks_mut(q_width)`,
    /// `k_batch.chunks_mut(kv_width)`) instead of trusting it by
    /// inspection.
    #[test]
    fn forward_batch_matches_forward_token_with_qk_norm_present() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;

        let mut decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        for layer in &mut decoder_a.layers {
            layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
            layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
        }
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let mut decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        for layer in &mut decoder_b.layers {
            layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
            layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
        }
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-3,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    /// QK-norm being present must actually change the output -- otherwise
    /// the `Some(...)` branches in `forward_token`/`forward_batch` could
    /// silently be dead code and this feature would ship unverified. Must
    /// decode at least 2 positions: at position 0 with a fresh cache,
    /// causal softmax has exactly one candidate (the token attending to
    /// itself) and always evaluates to weight 1.0 regardless of the Q*K
    /// dot product -- so the attention output there is Q/K-invariant by
    /// construction, and a single-position version of this test would
    /// pass even with `q_norm`/`k_norm` silently never applied.
    #[test]
    fn qk_norm_present_changes_output_versus_absent() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        let tokens = [3usize, 5];

        let without_norm = Decoder::new_random_small(cfg.clone(), 1, vocab);
        let mut with_norm = Decoder::new_random_small(cfg, 1, vocab);
        for layer in &mut with_norm.layers {
            layer.attn.q_norm = Some(vec![2.0; q_width]);
            layer.attn.k_norm = Some(vec![2.0; kv_width]);
        }

        let mut caches_a: Vec<KvCache> = (0..1)
            .map(|_| KvCache::new(without_norm.config.n_kv_heads, without_norm.config.head_dim))
            .collect();
        let mut caches_b: Vec<KvCache> = (0..1)
            .map(|_| KvCache::new(with_norm.config.n_kv_heads, with_norm.config.head_dim))
            .collect();

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            out_a = without_norm.forward_token(t, pos, &mut caches_a);
            out_b = with_norm.forward_token(t, pos, &mut caches_b);
        }

        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "QK-norm weights changed nothing -- forward_token likely isn't applying q_norm/k_norm"
        );
    }

    /// Qwen2/Qwen2-MoE-family QKV attention bias (`AttnWeights::q_bias`/
    /// `k_bias`/`v_bias`): a real, previously-unhandled gap found by
    /// running ferrox's generic GGUF loader against a real downloaded
    /// Qwen1.5-MoE-A2.7B-Chat checkpoint, which produced fluent-but-wrong
    /// output because these real `attn_{q,k,v}.bias` tensors were
    /// silently never added anywhere. Same two real properties checked
    /// as the QK-norm tests above: (1) `forward_batch` must match
    /// sequential `forward_token` exactly with bias present (batched
    /// per-row broadcast must be correct, not just the single-token
    /// path), and (2) bias must actually change the output at position
    /// 0 or later (not silently dead code) -- checked at position 1
    /// specifically, since position 0's causal softmax has exactly one
    /// candidate and is Q/K-invariant regardless of any additive bias
    /// shifting Q/K, for the same reason the QK-norm test above needs
    /// >=2 positions.
    #[test]
    fn forward_batch_matches_forward_token_with_qkv_bias_present() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;

        let mut decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        for layer in &mut decoder_a.layers {
            layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
            layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
            layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
        }
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let mut decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        for layer in &mut decoder_b.layers {
            layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
            layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
            layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
        }
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-3,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    #[test]
    fn qkv_bias_present_changes_output_versus_absent() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        let tokens = [3usize, 5];

        let without_bias = Decoder::new_random_small(cfg.clone(), 1, vocab);
        let mut with_bias = Decoder::new_random_small(cfg, 1, vocab);
        for layer in &mut with_bias.layers {
            layer.attn.q_bias = Some(vec![0.5; q_width]);
            layer.attn.k_bias = Some(vec![0.5; kv_width]);
            layer.attn.v_bias = Some(vec![0.5; kv_width]);
        }

        let mut caches_a: Vec<KvCache> = (0..1)
            .map(|_| KvCache::new(without_bias.config.n_kv_heads, without_bias.config.head_dim))
            .collect();
        let mut caches_b: Vec<KvCache> = (0..1)
            .map(|_| KvCache::new(with_bias.config.n_kv_heads, with_bias.config.head_dim))
            .collect();

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            out_a = without_bias.forward_token(t, pos, &mut caches_a);
            out_b = with_bias.forward_token(t, pos, &mut caches_b);
        }

        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "QKV bias changed nothing -- forward_token likely isn't applying q_bias/k_bias/v_bias"
        );
    }

    #[test]
    fn forward_batch_and_forward_token_leave_kv_caches_in_the_same_state() {
        let cfg = tiny_test_config();
        let tokens = [2usize, 4, 6];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, 8);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        for (pos, &t) in tokens.iter().enumerate() {
            decoder_a.forward_token(t, pos, &mut caches_a);
        }

        let decoder_b = Decoder::new_random_small(cfg, 2, 8);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        for (ca, cb) in caches_a.iter().zip(caches_b.iter()) {
            assert_eq!(ca.seq_len, cb.seq_len);
            assert_eq!(ca.k.len(), cb.k.len());
            for (a, b) in ca.k.iter().zip(cb.k.iter()) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }

    /// Same architecture shape as `tiny_test_config` but genuinely
    /// dense (one expert, no shared experts) -- the shape every non-MoE
    /// model, and every DeepSeek-style leading dense layer, loads as.
    /// Exercises `Decoder::is_dense_layer`'s fast path.
    fn tiny_dense_test_config() -> ModelConfig {
        let mut cfg = tiny_test_config();
        cfg.moe.n_experts = 1;
        cfg.moe.n_experts_active = 1;
        cfg.moe.n_shared_experts = 0;
        cfg
    }

    #[test]
    fn dense_layer_forward_pass_produces_finite_logits_of_correct_shape() {
        let vocab = 10;
        let decoder = Decoder::new_random_small(tiny_dense_test_config(), 2, vocab);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        let logits = decoder.forward_token(3, 0, &mut caches);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must not contain NaN/Inf"
        );
    }

    #[test]
    fn dense_layer_forward_batch_matches_sequential_forward_token_exactly() {
        let cfg = tiny_dense_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-3,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    #[test]
    fn dense_layer_fast_path_still_records_expert_zero_activations() {
        // The dense fast path bypasses `route_top_k` entirely, but
        // must still record an activation for expert 0 every step --
        // `MoeWeights::placement_plan` and hotness-based GPU placement
        // depend on this being real for every model shape, not just
        // genuinely-MoE ones.
        let decoder = Decoder::new_random_small(tiny_dense_test_config(), 1, 8);
        let mut caches: Vec<KvCache> = (0..1)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        decoder.forward_token(0, 0, &mut caches);
        decoder.forward_token(1, 1, &mut caches);
        decoder.forward_token(2, 2, &mut caches);

        let count =
            decoder.layers[0].moe.activation_counts[0].load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(count, 3);
    }

    #[test]
    fn forward_batch_with_empty_tokens_returns_empty() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let out = decoder.forward_batch(&[], 0, &mut caches);
        assert!(out.is_empty());
    }

    #[test]
    fn forward_batch_continues_correctly_after_prior_forward_token_calls() {
        // Realistic usage pattern: some tokens processed one at a time
        // (e.g. the first generated token), then a batch verifying
        // several draft tokens at once, continuing from the same
        // cache. The batch's positions must be numbered starting from
        // wherever the cache left off, not from zero.
        let cfg = tiny_test_config();

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, 8);
        let mut caches_a: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_a.config.n_kv_heads, decoder_a.config.head_dim))
            .collect();
        decoder_a.forward_token(1, 0, &mut caches_a);
        decoder_a.forward_token(3, 1, &mut caches_a);
        let seq_next = decoder_a.forward_token(5, 2, &mut caches_a);

        let decoder_b = Decoder::new_random_small(cfg, 2, 8);
        let mut caches_b: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder_b.config.n_kv_heads, decoder_b.config.head_dim))
            .collect();
        decoder_b.forward_token(1, 0, &mut caches_b);
        let batch_next = decoder_b.forward_batch(&[3, 5], 1, &mut caches_b);

        for (s, b) in seq_next.iter().zip(batch_next[1].iter()) {
            assert!((s - b).abs() < 1e-3, "sequential={s} batched={b}");
        }
    }

    /// `PlacementPlan::from_budget` is
    /// real and tested in isolation, but only meaningful once it's fed
    /// genuinely observed per-expert activation counts rather than
    /// zeros. This proves the full loop: run real forward passes,
    /// confirm `MoeWeights::activation_counts` actually reflects what
    /// `route_top_k` selected, and confirm `placement_plan` prioritizes
    /// the expert that was genuinely hottest -- not just that the
    /// budget/size arithmetic works on synthetic inputs.
    #[test]
    fn placement_plan_reflects_real_observed_expert_activations() {
        let cfg = tiny_test_config(); // 6 experts, top-2 active/token
        let decoder = Decoder::new_random_small(cfg, 2, 16);
        let mut caches: Vec<KvCache> = (0..2)
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();

        let n_calls = 20;
        for pos in 0..n_calls {
            decoder.forward_token(pos % 16, pos, &mut caches);
        }

        let layer0 = &decoder.layers[0].moe;
        let counts: Vec<u64> = layer0
            .activation_counts
            .iter()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .collect();
        let total: u64 = counts.iter().sum();
        assert_eq!(
            total,
            (n_calls as u64) * (decoder.config.moe.n_experts_active as u64),
            "total recorded activations must equal calls * experts_active_per_call"
        );

        // Ties are realistic at this small a sample size; break them the
        // same way `PlacementPlan::from_budget` does (lowest index
        // wins), so this assertion can't spuriously fail on a tie that
        // `from_budget` resolves differently than a naive `max_by_key`
        // (which returns the *last* max element) would.
        let hottest_count = *counts.iter().max().unwrap();
        let hottest_idx = counts.iter().position(|&c| c == hottest_count).unwrap();
        assert!(hottest_count > 0);

        // A per-expert resident size big enough for exactly one expert.
        let per_expert_bytes = layer0.expert_bytes(0);
        let plan = layer0.placement_plan(per_expert_bytes as u64);

        assert_eq!(
            plan.placement_for(hottest_idx),
            ferrox_moe::ExpertPlacement::GpuDevice(0),
            "the genuinely hottest expert (index {hottest_idx}, {hottest_count} activations) \
             must be the one the plan places on GPU when only one expert fits the budget"
        );
    }
}
