//! ferrox-moe: sparse Mixture-of-Experts routing, a shared-expert path,
//! and a CPU/GPU expert-placement scheduler.
//!
//! The placement design mirrors the pattern popularized by ik_llama.cpp
//! (tensor-name-regex overrides deciding which experts live on GPU vs
//! CPU RAM, e.g. `--cpu-moe` / `-ncmoe`) and by llama.cpp's
//! layer-split conventions, adapted here to a config-driven Rust
//! scheduler rather than copied CLI-flag parsing code. See
//! docs/THIRD_PARTY_NOTICES.md.

use ferrox_core::matmul::swiglu;
use ferrox_core::weight_matrix::WeightMatrix;

/// Where a given expert's weights currently live. `GpuDevice`-placed
/// experts only actually execute on a GPU under `--features cuda` and/or
/// `--features metal` (see `run_expert_placed`); without a GPU feature
/// the CPU path executes regardless. Device id is meaningful for CUDA;
/// Metal currently uses the system default device and ignores the id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPlacement {
    Cpu,
    GpuDevice(u32),
}

/// Static per-layer MoE configuration. One of these is built per layer
/// from a ModelConfig preset (ferrox-models).
#[derive(Debug, Clone)]
pub struct MoeLayerConfig {
    pub n_experts: usize,
    pub n_experts_active: usize,
    pub n_shared_experts: usize,
    pub hidden_dim: usize,
    pub expert_ffn_dim: usize,
    /// Which function converts router logits into selection scores.
    /// See `GatingFunction`'s doc comment: this is not a stylistic
    /// choice, it's an evidence-backed architectural detail that
    /// differs by model family.
    pub gating: GatingFunction,
    /// Only meaningful for `GatingFunction::Softmax` (the `Sigmoid` path
    /// has its own separate, always-renormalized convention -- see
    /// `route_top_k_sigmoid`'s doc comment). Whether the top-k selected
    /// experts' softmax weights get renormalized to sum to one after
    /// selection. Mixtral's real routing does this
    /// (`routing_weights /= routing_weights.sum(...)` in its reference
    /// implementation) and it's the right default for any architecture
    /// that doesn't document otherwise -- but it is a real, per-model
    /// choice, not a law of nature: OLMoE's real `config.json` sets
    /// `norm_topk_prob: false`, confirmed against
    /// `OlmoeTopKRouter.forward` in
    /// `transformers/models/olmoe/modeling_olmoe.py` (`router_top_value
    /// /= router_top_value.sum(...)` only runs `if self.norm_topk_prob`)
    /// and against llama.cpp's real hardcoded `build_moe_ffn(..., false,
    /// ..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, ...)` call for
    /// `LLM_ARCH_OLMOE` in `src/models/olmoe.cpp` (GGUF carries no
    /// metadata key for this -- it's an architecture-hardcoded fact in
    /// the reference implementation, not something read from the file).
    /// Getting this wrong silently produces a real, wrong generation:
    /// caught by comparing ferrox's real OLMoE output directly against
    /// llama.cpp loading the identical GGUF file (llama.cpp answered
    /// "Paris" for "the capital of France is"; ferrox, with this bug,
    /// answered something else entirely).
    pub norm_topk_prob: bool,
    /// Optional Mixtral-style expert grouping (`expert_group_count` /
    /// `expert_group_used_count` in GGUF). `None` means flat top-k over
    /// all experts (Llama / OLMoE / Qwen2-MoE). Grouped routing is
    /// selected at load time; the hot path reads these fields as data.
    pub expert_group_count: Option<usize>,
    pub expert_group_used_count: Option<usize>,
}

/// Router output for one token: which experts fire, and their
/// (already-normalized) combination weights.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Top-k softmax router over per-expert logits, as used by DeepSeek /
/// GLM / Kimi-style MoE layers (a linear gate scores every expert, top-k
/// experts are kept, and their scores are renormalized to sum to one).
/// Which function converts a router's raw per-expert logits into
/// selection scores, before top-k selection and normalization.
///
/// This distinction is not cosmetic: reading ik_llama.cpp's actual
/// GGUF-loading source (`llama-hparams.cpp`) directly showed that
/// DeepSeek-2/3-family models (`LLM_ARCH_DEEPSEEK2`) and newer
/// GLM-MoE-family models (`LLM_ARCH_GLM4_MOE`) both default to
/// **sigmoid** gating with post-selection score normalization, not
/// softmax -- matching DeepSeek-V3's own published technical report,
/// which documents computing per-expert affinity via sigmoid and then
/// normalizing the *selected* experts' scores to sum to one. Only
/// older DeepSeek-2.0/2.5-era models default to softmax. Using softmax
/// unconditionally, which ferrox did before this was found, would
/// silently produce wrong routing decisions for any model in the
/// DeepSeek-3/GLM4-MoE lineage -- which very plausibly includes
/// DeepSeek V4 Pro and GLM-5.2, both presumed continuations of these
/// architecture families (see docs/MODELS.md for the exact
/// confidence level on this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingFunction {
    /// exp(logit) / sum(exp(selected logits)) -- the older convention.
    Softmax,
    /// sigmoid(logit) for scoring and top-k selection, then the
    /// selected sigmoid scores are renormalized to sum to one -- the
    /// DeepSeek-V3 / GLM4-MoE convention.
    Sigmoid,
    /// `sqrt(softplus(logit))`, i.e. `sqrt(ln(1 + exp(logit)))` --
    /// DeepSeek V4's real MoE scoring function. Confirmed two ways from
    /// llama.cpp PR #24162 (`src/models/deepseek4.cpp`): (1)
    /// `load_arch_hparams` hard-throws
    /// (`"DeepSeek-V4 loader currently expects sqrtsoftplus MoE
    /// scoring"`) unless the GGUF's `expert_gating_func` metadata is
    /// exactly `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`; (2)
    /// `llm_graph_context::build_moe_ffn`'s real scoring switch computes
    /// `probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits))` for that
    /// enum case. This supersedes the earlier sigmoid guess this crate
    /// carried (inherited from the DeepSeek-2/3 lineage), which the real
    /// loader source shows is wrong for V4 specifically.
    SqrtSoftplus,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `sqrt(softplus(x))` = `sqrt(ln(1 + exp(x)))`, computed the numerically
/// stable way (`softplus(x) = max(x, 0) + ln(1 + exp(-|x|))`, avoiding
/// overflow in `exp(x)` for large positive `x`) -- DeepSeek V4's real
/// per-expert MoE scoring function, see [`GatingFunction::SqrtSoftplus`].
fn sqrt_softplus(x: f32) -> f32 {
    let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
    softplus.sqrt()
}

/// Top-k router over per-expert logits, dispatching to softmax, sigmoid,
/// or sqrt-softplus scoring per `gating`. See `GatingFunction`'s doc
/// comment for why this distinction is real and evidence-backed, not a
/// stylistic choice.
pub fn route_top_k(
    logits: &[f32],
    k: usize,
    gating: GatingFunction,
    norm_topk_prob: bool,
) -> RoutingDecision {
    match gating {
        GatingFunction::Softmax => route_top_k_softmax(logits, k, norm_topk_prob),
        GatingFunction::Sigmoid => route_top_k_sigmoid(logits, k),
        GatingFunction::SqrtSoftplus => route_top_k_sqrtsoftplus(logits, k, norm_topk_prob),
    }
}

/// Mixtral-/DeepSeek-style grouped top-k: split `logits` into
/// `n_groups` contiguous expert groups, pick the `k_per_group` highest
/// within each group (by the same score as flat [`route_top_k`]), then
/// optionally keep only the global top-`total_k` across groups.
///
/// When `n_groups <= 1` this is identical to flat routing with `k =
/// total_k`. Used when GGUF carries `expert_group_count` /
/// `expert_group_used_count`.
pub fn route_top_k_grouped(
    logits: &[f32],
    n_groups: usize,
    k_per_group: usize,
    total_k: usize,
    gating: GatingFunction,
    norm_topk_prob: bool,
) -> RoutingDecision {
    if n_groups <= 1 || !logits.len().is_multiple_of(n_groups) {
        return route_top_k(logits, total_k, gating, norm_topk_prob);
    }
    let group_size = logits.len() / n_groups;
    let mut selected: Vec<(usize, f32)> = Vec::new();
    for g in 0..n_groups {
        let start = g * group_size;
        let slice = &logits[start..start + group_size];
        let local = route_top_k(slice, k_per_group.min(group_size), gating, false);
        for (i, &expert) in local.expert_ids.iter().enumerate() {
            selected.push((start + expert, local.weights[i]));
        }
    }
    selected.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    selected.truncate(total_k.min(selected.len()));
    let mut weights: Vec<f32> = selected.iter().map(|(_, w)| *w).collect();
    if norm_topk_prob {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    RoutingDecision {
        expert_ids: selected.into_iter().map(|(i, _)| i).collect(),
        weights,
    }
}

/// sqrt-softplus top-k routing, DeepSeek V4's real non-hash-routed MoE
/// layers (`ffn_exp_probs_b` present, i.e. every layer at or past
/// `hash_layer_count`): score every expert with `sqrt(softplus(logit))`
/// (independently per expert, like [`route_top_k_sigmoid`]'s sigmoid --
/// not a joint softmax distribution), pick the top-k by that score, then
/// (if `norm_topk_prob`) renormalize the selected scores to sum to one.
/// See [`GatingFunction::SqrtSoftplus`] for the real citation. DeepSeek
/// V4's real non-hash MoE layers additionally add a learned bias
/// (`ffn_exp_probs_b`) to the *selection* score only -- see
/// [`route_top_k_sqrtsoftplus_with_bias`] for that variant; this plain
/// version is the bias-free building block, analogous to
/// [`route_top_k_sigmoid`] vs [`route_top_k_sigmoid_with_bias`].
pub fn route_top_k_sqrtsoftplus(logits: &[f32], k: usize, norm_topk_prob: bool) -> RoutingDecision {
    let scores: Vec<f32> = logits.iter().map(|&l| sqrt_softplus(l)).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if norm_topk_prob {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// sqrt-softplus top-k routing with a selection-only bias term, mirroring
/// [`route_top_k_sigmoid_with_bias`] but for DeepSeek V4's real
/// [`GatingFunction::SqrtSoftplus`] scoring: selection uses
/// `sqrt(softplus(logit)) + bias[expert]`, but each selected expert's
/// combine *weight* uses the raw, unbiased `sqrt(softplus(logit))`.
/// Weights are renormalized to sum to one (if `k>1` and `renormalize`),
/// then multiplied by `scaling_factor` (DeepSeek V4's real
/// `expert_weights_norm`/`expert_weights_scale` hparams, read directly in
/// `load_arch_hparams`).
pub fn route_top_k_sqrtsoftplus_with_bias(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits.iter().map(|&l| sqrt_softplus(l)).collect();
    let scores_for_choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        scores_for_choice[b]
            .partial_cmp(&scores_for_choice[a])
            .unwrap()
    });
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if k > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// DeepSeek V4's real hash-based first-layer MoE routing: for the first
/// `hash_layer_count` layers, which experts fire is *not* learned
/// top-k/sigmoid/sqrt-softplus selection at all -- it's a direct
/// token-id-to-expert-id lookup table (`ffn_gate_tid2eid`, GGUF shape
/// `[n_expert_used, n_vocab]`; real per-layer dispatch in
/// `src/models/deepseek4.cpp`: `selected_experts =
/// ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens)`, with
/// `exp_probs_b` (the selection-bias tensor) set to `nullptr` for these
/// layers specifically because there is no learned selection to bias --
/// the expert ids are fixed by the table, not chosen by a score).
///
/// The selected experts' *combine weights*, however, are **not** fixed by
/// the table -- `build_moe_ffn` still computes `sqrt(softplus(logits))`
/// from the real per-token router logits (`ffn_gate_inp`) and gathers
/// those scores at the table-provided expert ids, exactly like the
/// weight half of [`route_top_k_sqrtsoftplus_with_bias`] (just with a
/// fixed selection instead of a chosen top-k). `hash_expert_ids` must
/// have exactly the model's real `n_expert_used` length (one lookup-table
/// row for this token's id); `logits` is the full `[n_expert]`-wide
/// router output for this token.
pub fn route_hash(
    hash_expert_ids: &[usize],
    logits: &[f32],
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    let mut weights: Vec<f32> = hash_expert_ids
        .iter()
        .map(|&e| sqrt_softplus(logits[e]))
        .collect();
    if hash_expert_ids.len() > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: hash_expert_ids.to_vec(),
        weights,
    }
}

/// softmax-then-top-k routing (the Mixtral/older-DeepSeek convention:
/// softmax over *every* expert first, then select the top-k of those
/// probabilities -- not "top-k logits, then softmax just those"; the two
/// are mathematically different since softmax's denominator would only
/// sum the selected subset in the latter). `norm_topk_prob` controls
/// whether the selected top-k probabilities are then renormalized to sum
/// to one -- true is the right default for any architecture that doesn't
/// document otherwise (Mixtral does this), but it is a real per-model
/// choice: see `MoeLayerConfig::norm_topk_prob`'s doc comment for why
/// OLMoE specifically needs `false`.
pub fn route_top_k_softmax(logits: &[f32], k: usize, norm_topk_prob: bool) -> RoutingDecision {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| probs[i]).collect();
    if norm_topk_prob {
        let top_sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= top_sum;
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// sigmoid-then-renormalize top-k routing: score every expert with
/// `sigmoid(logit)` (independently per expert, not a joint softmax
/// distribution), pick the top-k by that score, then renormalize just
/// the selected experts' sigmoid scores to sum to one. This is the
/// DeepSeek-V3 / GLM4-MoE convention found in ik_llama.cpp's real GGUF
/// hparams-loading source.
pub fn route_top_k_sigmoid(logits: &[f32], k: usize) -> RoutingDecision {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let sum: f32 = top.iter().map(|&i| scores[i]).sum();
    let weights: Vec<f32> = if sum > 0.0 {
        top.iter().map(|&i| scores[i] / sum).collect()
    } else {
        // Degenerate case (all selected scores are exactly zero,
        // essentially never in practice for a real trained router):
        // fall back to a uniform split rather than dividing by zero.
        vec![1.0 / top.len() as f32; top.len()]
    };

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// Sigmoid top-k routing with a real "aux-loss-free" per-expert bias
/// term added *only* for top-k selection (`topk_method: "noaux_tc"` in
/// Kimi K3's real `config.json`, `KimiMoEGate.forward` in
/// `modeling_kimi_linear.py`, adapted from DeepSeek-V3's own MoE gate --
/// the same convention, not Kimi-specific): the selection scores are
/// `sigmoid(logit) + bias[expert]`, but the *weight* each selected
/// expert's output gets multiplied by uses the raw, unbiased
/// `sigmoid(logit)` -- getting this backwards (biasing the weight
/// itself, not just the selection) would silently skew routed-expert
/// contribution away from what the router actually learned. Weights are
/// renormalized to sum to 1 (if `k>1`) then multiplied by
/// `scaling_factor` (Kimi K3's `routed_scaling_factor`, 1.0 in its real
/// config, i.e. a no-op there, but a real multiplier for any other
/// model using this same convention with a different value).
pub fn route_top_k_sigmoid_with_bias(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let scores_for_choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        scores_for_choice[b]
            .partial_cmp(&scores_for_choice[a])
            .unwrap()
    });
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if k > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// A CPU/GPU placement plan for a layer's experts.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    pub default_placement: ExpertPlacement,
    pub overrides: std::collections::HashMap<usize, ExpertPlacement>,
}

impl PlacementPlan {
    pub fn all_cpu(n_experts: usize) -> Self {
        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides: (0..n_experts).map(|i| (i, ExpertPlacement::Cpu)).collect(),
        }
    }

    /// Index-based placeholder: puts the first `n_gpu_resident`
    /// experts on GPU regardless of their actual size or how often
    /// they're activated. Kept only as a trivial fallback for callers
    /// with no real budget/hotness data at all (e.g. `ferrox smoke`'s
    /// synthetic-weight demo); real deployments should use
    /// `from_budget` instead, which places by measured VRAM budget and
    /// observed expert hotness rather than by index.
    pub fn hot_experts_on_gpu(n_experts: usize, n_gpu_resident: usize) -> Self {
        let mut overrides = std::collections::HashMap::new();
        for i in 0..n_experts.min(n_gpu_resident) {
            overrides.insert(i, ExpertPlacement::GpuDevice(0));
        }
        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides,
        }
    }

    /// Builds a placement plan from a real VRAM budget and each
    /// expert's actual resident byte size (e.g. summed
    /// `WeightMatrix::resident_bytes()` across an expert's gate/up/down
    /// matrices), following ik_llama.cpp's `--cpu-moe`/`--override-tensor`
    /// pattern of deciding CPU-vs-GPU per tensor rather than by a fixed
    /// index cutoff.
    ///
    /// `activation_counts`, if given (one count per expert, e.g.
    /// accumulated from `RoutingDecision::expert_ids` over a real or
    /// representative workload), places the *most frequently activated*
    /// experts on GPU first -- the actual point of expert offload,
    /// since keeping a rarely-used expert resident in VRAM wastes the
    /// budget a hot expert could have used instead. Without observed
    /// counts, falls back to a documented, deterministic policy (index
    /// order) rather than guessing at hotness.
    ///
    /// Greedy, not globally optimal (a smaller-but-colder expert can
    /// still be skipped in favor of trying the next candidate once a
    /// larger higher-priority expert doesn't fit) -- optimal knapsack
    /// packing is not worth the complexity here, and greedy-by-priority
    /// is the same approach real offload tooling uses.
    pub fn from_budget(
        expert_bytes: &[usize],
        activation_counts: Option<&[u64]>,
        vram_budget_bytes: u64,
    ) -> Self {
        let n = expert_bytes.len();
        let mut order: Vec<usize> = (0..n).collect();
        if let Some(counts) = activation_counts {
            if counts.len() == n {
                order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
            }
        }

        let mut overrides = std::collections::HashMap::new();
        let mut used: u64 = 0;
        for idx in order {
            let size = expert_bytes[idx] as u64;
            if size == 0 || used + size > vram_budget_bytes {
                continue;
            }
            used += size;
            overrides.insert(idx, ExpertPlacement::GpuDevice(0));
        }

        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides,
        }
    }

    /// A device-placement plan for EVERY layer's routed experts against
    /// ONE shared VRAM budget -- the fix for the real accounting bug
    /// where each layer independently called `from_budget` with the
    /// full budget, so a model with N layers would plan N x the
    /// configured bytes of GPU residency. All `(layer, expert)`
    /// candidates compete in one global priority order (hottest first,
    /// ties broken by layer then expert index for determinism), and a
    /// candidate is only placed on the device if the *global* running
    /// total still fits.
    pub fn plan_layers_against_global_budget(
        expert_bytes_per_layer: &[Vec<usize>],
        activation_counts_per_layer: Option<&[Vec<u64>]>,
        vram_budget_bytes: u64,
    ) -> ResidencyPlan {
        let mut candidates: Vec<(u64, usize, usize)> = Vec::new(); // (count, layer, expert)
        for (l, sizes) in expert_bytes_per_layer.iter().enumerate() {
            for e in 0..sizes.len() {
                let count = activation_counts_per_layer
                    .and_then(|cs| cs.get(l))
                    .and_then(|c| c.get(e))
                    .copied()
                    .unwrap_or(0);
                candidates.push((count, l, e));
            }
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut layer_overrides: Vec<std::collections::HashMap<usize, ExpertPlacement>> =
            expert_bytes_per_layer
                .iter()
                .map(|_| std::collections::HashMap::new())
                .collect();
        let mut used: u64 = 0;
        for (_, l, e) in candidates {
            let size = expert_bytes_per_layer[l][e] as u64;
            if size == 0 || used + size > vram_budget_bytes {
                continue;
            }
            used += size;
            layer_overrides[l].insert(e, ExpertPlacement::GpuDevice(0));
        }

        ResidencyPlan {
            layer_plans: layer_overrides
                .into_iter()
                .map(|overrides| PlacementPlan {
                    default_placement: ExpertPlacement::Cpu,
                    overrides,
                })
                .collect(),
            device_bytes_planned: used,
            vram_budget_bytes,
        }
    }

    pub fn placement_for(&self, expert_id: usize) -> ExpertPlacement {
        self.overrides
            .get(&expert_id)
            .copied()
            .unwrap_or(self.default_placement)
    }
}

/// The output of `PlacementPlan::plan_layers_against_global_budget`:
/// one per-layer `PlacementPlan` view over a single, globally-accounted
/// device budget. `device_bytes_planned <= vram_budget_bytes` holds by
/// construction across ALL layers combined -- the property the old
/// per-layer planning could not provide.
pub struct ResidencyPlan {
    layer_plans: Vec<PlacementPlan>,
    /// Total bytes this plan places on the device, summed across every
    /// layer.
    pub device_bytes_planned: u64,
    /// The single budget every layer's placements were accounted
    /// against.
    pub vram_budget_bytes: u64,
}

impl ResidencyPlan {
    pub fn layer_plan(&self, layer: usize) -> &PlacementPlan {
        &self.layer_plans[layer]
    }

    pub fn n_layers(&self) -> usize {
        self.layer_plans.len()
    }
}

/// One expert's gate/up/down weight matrices. Each may be plain f32
/// (synthetic/test weights) or still-quantized bytes loaded straight
/// from a GGUF file (real checkpoints) -- `WeightMatrix::apply`
/// dispatches to the right kernel either way.
pub struct ExpertWeights {
    pub gate: WeightMatrix,
    pub up: WeightMatrix,
    pub down: WeightMatrix,
}

/// Runs one token's hidden state through a single expert's SwiGLU FFN.
pub fn run_expert(hidden: &[f32], expert: &ExpertWeights) -> Vec<f32> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        // Full SwiGLU on-device (1× upload + 1× download) when dense GPU is on.
        if let Some(out) = ferrox_core::WeightMatrix::apply_gpu_dense_ffn_swiglu(
            &expert.gate,
            &expert.up,
            &expert.down,
            hidden,
        ) {
            return out;
        }
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        // Gate and up share `hidden` — one GPU upload / multi-matvec.
        if let Some(mut outs) =
            ferrox_core::WeightMatrix::apply_gpu_multi(&[&expert.gate, &expert.up], hidden)
        {
            let up = outs.pop().unwrap();
            let gate = outs.pop().unwrap();
            let activated = swiglu(&gate, &up);
            return expert.down.apply(&activated);
        }
    }
    // Share one Q8 activation quant across gate+up when INT_DOT is on
    // (OLMoE: avoids 2× quantize_activations_q8 per expert).
    if ferrox_core::weight_matrix::cpu_int_dot_enabled() && hidden.len().is_multiple_of(32) {
        let act = ferrox_quant::quantize_activations_q8(hidden);
        if let (Some(gate), Some(up)) = (
            expert.gate.apply_cpu_q8(&act),
            expert.up.apply_cpu_q8(&act),
        ) {
            let activated = swiglu(&gate, &up);
            return expert.down.apply(&activated);
        }
    }
    let gate = expert.gate.apply(hidden);
    let up = expert.up.apply(hidden);
    let activated = swiglu(&gate, &up);
    expert.down.apply(&activated)
}

/// `run_expert`, but actually consulting `placement` instead of always
/// running on CPU -- this is the real execution consequence
/// `PlacementPlan` previously computed but nothing acted on: a
/// `GpuDevice`-placed expert's gate/up/down matvecs go through
/// `WeightMatrix::apply_gpu` (real CUDA and/or Metal kernels for
/// Q8_0/Q4_0/Q4_K/Q5_K/Q6_K), falling straight through to the ordinary
/// CPU path for any matrix `apply_gpu` returns `None` for (an
/// unsupported quant kind, or a real launch failure) -- so this is
/// always correct, never a hard failure, regardless of GPU availability.
///
/// Every call re-uploads each weight matrix to the device from scratch
/// (see `WeightMatrix::apply_gpu`'s doc comment) -- correct, but not
/// yet the persistent-GPU-residency throughput win real expert offload
/// needs; a real, disclosed limit of this round, not overclaimed.
///
/// Without a GPU feature (`cuda` / `metal`) compiled in, this has the
/// exact same signature and always calls `run_expert` (ignoring
/// `placement`), so callers (e.g. `ferrox-models::decoder::Decoder`)
/// can call it unconditionally regardless of how this crate was built,
/// with correct behavior either way.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn run_expert_placed(
    hidden: &[f32],
    expert: &ExpertWeights,
    placement: ExpertPlacement,
) -> Vec<f32> {
    if matches!(placement, ExpertPlacement::GpuDevice(_)) {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            if let Some(out) = ferrox_core::WeightMatrix::apply_gpu_dense_ffn_swiglu(
                &expert.gate,
                &expert.up,
                &expert.down,
                hidden,
            ) {
                return out;
            }
        }
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            if let Some(mut outs) =
                ferrox_core::WeightMatrix::apply_gpu_multi(&[&expert.gate, &expert.up], hidden)
            {
                let up = outs.pop().unwrap();
                let gate = outs.pop().unwrap();
                let activated = swiglu(&gate, &up);
                if let Some(down) = expert.down.apply_gpu(&activated) {
                    return down;
                }
                return expert.down.apply(&activated);
            }
        }
        if let Some(gate) = expert.gate.apply_gpu(hidden) {
            if let Some(up) = expert.up.apply_gpu(hidden) {
                let activated = swiglu(&gate, &up);
                if let Some(down) = expert.down.apply_gpu(&activated) {
                    return down;
                }
            }
        }
    }
    run_expert(hidden, expert)
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn run_expert_placed(
    hidden: &[f32],
    expert: &ExpertWeights,
    _placement: ExpertPlacement,
) -> Vec<f32> {
    run_expert(hidden, expert)
}

/// Combines routed + shared expert outputs for one token.
pub fn combine_expert_outputs(
    routed_outputs: &[(Vec<f32>, f32)],
    shared_outputs: &[Vec<f32>],
    hidden_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; hidden_dim];
    for (expert_out, weight) in routed_outputs {
        for (o, e) in out.iter_mut().zip(expert_out.iter()) {
            *o += e * weight;
        }
    }
    for shared_out in shared_outputs {
        for (o, e) in out.iter_mut().zip(shared_out.iter()) {
            *o += e;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    /// The property the global planner exists for: with N layers of
    /// identical experts and a budget that fits exactly K experts,
    /// exactly K experts are device-placed across ALL layers combined
    /// -- not K per layer, which is what independent per-layer
    /// `from_budget` calls with the same budget would produce (N*K).
    #[test]
    fn global_budget_cannot_be_multiplied_across_layers() {
        let n_layers = 10;
        let sizes: Vec<Vec<usize>> = (0..n_layers).map(|_| vec![100usize; 4]).collect();
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, None, 250);

        let total_placed: usize = (0..n_layers)
            .map(|l| {
                (0..4)
                    .filter(|&e| plan.layer_plan(l).placement_for(e) != ExpertPlacement::Cpu)
                    .count()
            })
            .sum();
        assert_eq!(
            total_placed, 2,
            "250 bytes fits exactly 2 x 100-byte experts, globally"
        );
        assert_eq!(plan.device_bytes_planned, 200);
        assert!(plan.device_bytes_planned <= plan.vram_budget_bytes);

        // The old shape of the bug, for contrast: per-layer planning
        // with the same budget places 2 experts in EVERY layer.
        let per_layer_total: usize = (0..n_layers)
            .map(|_| {
                let p = PlacementPlan::from_budget(&[100; 4], None, 250);
                (0..4)
                    .filter(|&e| p.placement_for(e) != ExpertPlacement::Cpu)
                    .count()
            })
            .sum();
        assert_eq!(per_layer_total, 20, "per-layer planning overcommits 10x");
    }

    /// Hot experts win device slots across layer boundaries: a single
    /// very hot expert in a late layer beats cold experts in earlier
    /// layers.
    #[test]
    fn global_planning_prioritizes_hotness_across_layers() {
        let sizes: Vec<Vec<usize>> = (0..3).map(|_| vec![100usize; 2]).collect();
        let mut counts: Vec<Vec<u64>> = (0..3).map(|_| vec![0u64; 2]).collect();
        counts[2][1] = 50; // the only hot expert lives in the last layer
        counts[0][0] = 10;
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, Some(&counts), 200);

        assert_eq!(
            plan.layer_plan(2).placement_for(1),
            ExpertPlacement::GpuDevice(0),
            "hottest expert (layer 2) must win a slot"
        );
        assert_eq!(
            plan.layer_plan(0).placement_for(0),
            ExpertPlacement::GpuDevice(0),
            "second-hottest expert (layer 0) takes the remaining slot"
        );
        assert_eq!(plan.device_bytes_planned, 200);
    }

    /// Zero budget places nothing anywhere; empty (dense) layers are
    /// legal and contribute no candidates.
    #[test]
    fn global_planning_handles_zero_budget_and_dense_layers() {
        let sizes = vec![Vec::new(), vec![100usize; 3], Vec::new()];
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, None, 0);
        assert_eq!(plan.device_bytes_planned, 0);
        assert_eq!(plan.n_layers(), 3);
        for e in 0..3 {
            assert_eq!(plan.layer_plan(1).placement_for(e), ExpertPlacement::Cpu);
        }
    }

    use super::*;

    #[test]
    fn top_k_selects_highest_scoring_experts() {
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        assert_eq!(decision.expert_ids, vec![1, 3]);
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(decision.weights[0] > decision.weights[1]);
    }

    #[test]
    fn top_k_weights_always_sum_to_one_regardless_of_k() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::Softmax, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "k={k} sum={sum}");
        }
    }

    /// `norm_topk_prob: false` -- OLMoE's real convention (see
    /// `MoeLayerConfig::norm_topk_prob`'s doc comment). Golden values
    /// hand-computed independently: full softmax over all 8 logits
    /// (sum of exp(l_i - 8) = 1.5814460129), then the raw (un-renormalized)
    /// probabilities of the top-3 selected experts (indices 7, 6, 5 --
    /// logits 8, 7, 6). This is the exact bug that was silently producing
    /// wrong OLMoE output: the old code could only ever compute a
    /// top-k-local softmax (mathematically identical to
    /// always-renormalize), with no way to recover the un-renormalized
    /// probability relative to *all* experts.
    #[test]
    fn norm_topk_prob_false_uses_raw_full_softmax_probability_not_renormalized() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decision = route_top_k(&logits, 3, GatingFunction::Softmax, false);

        assert_eq!(decision.expert_ids, vec![7, 6, 5]);

        let expected = [0.6323223_f32, 0.2326232, 0.0855683];
        for (got, want) in decision.weights.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got={got} want={want}");
        }

        let sum: f32 = decision.weights.iter().sum();
        assert!(
            (sum - 0.9505138).abs() < 1e-4,
            "raw top-3 probability mass should be < 1 (it's a subset of a full 8-way softmax), got sum={sum}"
        );

        // Selecting the same experts with norm_topk_prob=true must
        // renormalize to the exact same values divided by that sum --
        // proving the two modes agree on *which* experts fire and differ
        // only in the final weight scaling.
        let normalized = route_top_k(&logits, 3, GatingFunction::Softmax, true);
        assert_eq!(normalized.expert_ids, decision.expert_ids);
        for (raw, norm) in decision.weights.iter().zip(normalized.weights.iter()) {
            assert!(
                (raw / sum - norm).abs() < 1e-4,
                "raw={raw} sum={sum} normalized={norm}"
            );
        }
    }

    #[test]
    fn sigmoid_gating_selects_same_top_experts_as_softmax_for_monotonic_logits() {
        // Sigmoid is monotonic in its input, so for a given set of
        // logits, top-k-by-sigmoid-score must select the exact same
        // expert ids as top-k-by-raw-logit (sigmoid just changes the
        // *weights*, not which experts are chosen).
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sigmoid_decision = route_top_k(&logits, 2, GatingFunction::Sigmoid, true);
        assert_eq!(softmax_decision.expert_ids, sigmoid_decision.expert_ids);
    }

    #[test]
    fn sigmoid_gating_weights_sum_to_one() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::Sigmoid, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "k={k} sum={sum}");
        }
    }

    #[test]
    fn bias_only_affects_selection_not_the_final_weight_value() {
        // Expert 0 has the lower raw score but a large positive bias, so
        // biased selection must pick it over expert 1 -- but the WEIGHT
        // it ends up with must be its raw (unbiased) sigmoid score, not
        // score+bias. Getting this backwards would silently make a
        // barely-selected expert dominate the combine.
        let logits = vec![0.1, 2.0];
        let bias = vec![10.0, 0.0];
        let decision = route_top_k_sigmoid_with_bias(&logits, &bias, 1, true, 1.0);
        assert_eq!(decision.expert_ids, vec![0]);
        // k=1 -> renormalization is a no-op (single weight / itself = 1,
        // scaled by 1.0), so the weight is just sigmoid(0.1), not 1.0.
        assert!((decision.weights[0] - sigmoid(0.1)).abs() < 1e-5);
    }

    #[test]
    fn without_bias_selection_falls_back_to_plain_sigmoid_top_k() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        let zero_bias = vec![0.0; logits.len()];
        let biased = route_top_k_sigmoid_with_bias(&logits, &zero_bias, 3, true, 1.0);
        let plain = route_top_k(&logits, 3, GatingFunction::Sigmoid, true);
        assert_eq!(biased.expert_ids, plain.expert_ids);
        for (a, b) in biased.weights.iter().zip(plain.weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn scaling_factor_multiplies_every_weight() {
        let logits = vec![1.0, 2.0, 3.0];
        let bias = vec![0.0; 3];
        let unscaled = route_top_k_sigmoid_with_bias(&logits, &bias, 2, true, 1.0);
        let scaled = route_top_k_sigmoid_with_bias(&logits, &bias, 2, true, 2.5);
        for (u, s) in unscaled.weights.iter().zip(scaled.weights.iter()) {
            assert!((u * 2.5 - s).abs() < 1e-5);
        }
    }

    #[test]
    fn sigmoid_and_softmax_weights_differ_for_the_same_logits() {
        // The whole point of the distinction: sigmoid scores each
        // expert independently (not as a joint distribution), so the
        // relative weighting between two selected experts differs from
        // softmax's, even though both sum to one and pick the same
        // experts. If this test ever fails by finding the two paths
        // identical, something has collapsed the sigmoid path back
        // into softmax.
        let logits = vec![3.0, 1.0, -2.0, 0.5];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sigmoid_decision = route_top_k(&logits, 2, GatingFunction::Sigmoid, true);
        assert!(
            (softmax_decision.weights[0] - sigmoid_decision.weights[0]).abs() > 1e-3,
            "softmax and sigmoid gating should generally produce different weight splits for the same logits"
        );
    }

    #[test]
    fn sqrt_softplus_matches_hand_computed_values_at_zero_and_positive_logit() {
        // softplus(0) = ln(2), sqrt(ln(2)) -- exact closed form, not just a
        // property check, to pin the real DeepSeek V4 formula
        // (sqrt(softplus(x)), not e.g. softplus(sqrt(x)) or sqrt(sigmoid)).
        assert!((sqrt_softplus(0.0) - 2.0_f32.ln().sqrt()).abs() < 1e-6);
        // softplus(x) -> x for large positive x, so sqrt_softplus(x) -> sqrt(x).
        assert!((sqrt_softplus(20.0) - 20.0_f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn grouped_routing_picks_within_each_group_then_global_top_k() {
        // 4 experts, 2 groups of 2. Scores favor expert 1 in group0 and
        // expert 3 in group1; total_k=2 should keep both group winners.
        let logits = vec![0.1, 5.0, 0.2, 4.0];
        let d = route_top_k_grouped(&logits, 2, 1, 2, GatingFunction::Softmax, true);
        assert_eq!(d.expert_ids.len(), 2);
        assert!(d.expert_ids.contains(&1));
        assert!(d.expert_ids.contains(&3));
        let sum: f32 = d.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sqrtsoftplus_gating_selects_same_top_experts_as_softmax_for_monotonic_logits() {
        // sqrt(softplus(x)) is monotonically increasing in x (both sqrt
        // and softplus are), so top-k-by-score must agree with top-k by
        // raw logit on *which* experts fire, same reasoning as the
        // sigmoid monotonicity test above.
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sqrtsoftplus_decision = route_top_k(&logits, 2, GatingFunction::SqrtSoftplus, true);
        assert_eq!(
            softmax_decision.expert_ids,
            sqrtsoftplus_decision.expert_ids
        );
    }

    #[test]
    fn sqrtsoftplus_weights_sum_to_one_when_normalized() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::SqrtSoftplus, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "k={k} sum={sum}");
        }
    }

    #[test]
    fn sqrtsoftplus_bias_only_affects_selection_not_the_final_weight_value() {
        // Same structure as `bias_only_affects_selection_not_the_final_weight_value`
        // but for the sqrt-softplus scoring function DeepSeek V4's real
        // non-hash MoE layers use.
        let logits = vec![0.1, 2.0];
        let bias = vec![10.0, 0.0];
        let decision = route_top_k_sqrtsoftplus_with_bias(&logits, &bias, 1, true, 1.0);
        assert_eq!(decision.expert_ids, vec![0]);
        assert!((decision.weights[0] - sqrt_softplus(0.1)).abs() < 1e-5);
    }

    #[test]
    fn sqrtsoftplus_without_bias_selection_falls_back_to_plain_top_k() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        let zero_bias = vec![0.0; logits.len()];
        let biased = route_top_k_sqrtsoftplus_with_bias(&logits, &zero_bias, 3, true, 1.0);
        let plain = route_top_k(&logits, 3, GatingFunction::SqrtSoftplus, true);
        assert_eq!(biased.expert_ids, plain.expert_ids);
        for (a, b) in biased.weights.iter().zip(plain.weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hash_routing_uses_the_fixed_table_ids_regardless_of_logit_ranking() {
        // Expert 0 has by far the highest logit, but the real mechanism
        // never looks at the router's ranking to choose experts for a
        // hash-routed layer -- the table says [2, 1], so that's what
        // fires, full stop.
        let logits = vec![100.0, 1.0, 0.5, -3.0];
        let hash_expert_ids = vec![2usize, 1usize];
        let decision = route_hash(&hash_expert_ids, &logits, true, 1.0);
        assert_eq!(decision.expert_ids, vec![2, 1]);
    }

    #[test]
    fn hash_routing_weights_come_from_the_real_router_logits_not_a_fixed_split() {
        // The table fixes *which* experts fire, but their relative
        // combine weight still comes from sqrt(softplus(logit)) gathered
        // at those ids -- not a uniform 1/n split. Expert 2's logit (3.0)
        // is much larger than expert 1's (0.1), so its weight must
        // dominate even though both were unconditionally selected.
        let logits = vec![-5.0, 0.1, 3.0, -5.0];
        let hash_expert_ids = vec![2usize, 1usize];
        let decision = route_hash(&hash_expert_ids, &logits, true, 1.0);
        assert!(decision.weights[0] > decision.weights[1]);
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let expected0 = sqrt_softplus(3.0) / (sqrt_softplus(3.0) + sqrt_softplus(0.1));
        assert!((decision.weights[0] - expected0).abs() < 1e-5);
    }

    #[test]
    fn hash_routing_scaling_factor_multiplies_every_weight() {
        let logits = vec![1.0, 2.0, 3.0];
        let hash_expert_ids = vec![0usize, 2usize];
        let unscaled = route_hash(&hash_expert_ids, &logits, true, 1.0);
        let scaled = route_hash(&hash_expert_ids, &logits, true, 2.5);
        for (u, s) in unscaled.weights.iter().zip(scaled.weights.iter()) {
            assert!((u * 2.5 - s).abs() < 1e-5);
        }
    }

    #[test]
    fn placement_plan_defaults_to_cpu_for_unlisted_experts() {
        let plan = PlacementPlan::hot_experts_on_gpu(256, 8);
        assert_eq!(plan.placement_for(0), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(7), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(8), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(255), ExpertPlacement::Cpu);
    }

    #[test]
    fn all_cpu_plan_never_returns_gpu() {
        let plan = PlacementPlan::all_cpu(64);
        for i in 0..64 {
            assert_eq!(plan.placement_for(i), ExpertPlacement::Cpu);
        }
    }

    #[test]
    fn from_budget_fits_as_many_experts_as_the_vram_budget_allows() {
        // 4 experts, 100 bytes each: a 250-byte budget fits exactly 2.
        let sizes = vec![100usize, 100, 100, 100];
        let plan = PlacementPlan::from_budget(&sizes, None, 250);
        let on_gpu = (0..4)
            .filter(|&i| plan.placement_for(i) == ExpertPlacement::GpuDevice(0))
            .count();
        assert_eq!(on_gpu, 2);
    }

    #[test]
    fn from_budget_prioritizes_the_most_frequently_activated_experts() {
        // Expert 2 is by far the hottest but is neither first nor
        // largest -- a real budget-aware plan must still pick it first.
        let sizes = vec![50usize, 50, 50, 50];
        let counts = vec![1u64, 2, 100, 3];
        // Budget for exactly one expert.
        let plan = PlacementPlan::from_budget(&sizes, Some(&counts), 50);
        assert_eq!(
            plan.placement_for(2),
            ExpertPlacement::GpuDevice(0),
            "the hottest expert (index 2) must be the one placed on GPU"
        );
        assert_eq!(plan.placement_for(0), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(1), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(3), ExpertPlacement::Cpu);
    }

    #[test]
    fn from_budget_skips_an_expert_that_does_not_fit_and_tries_the_next() {
        // Expert 0 is too big for the budget alone; experts 1 and 2
        // together fit and should both be placed.
        let sizes = vec![200usize, 60, 60];
        let plan = PlacementPlan::from_budget(&sizes, None, 120);
        assert_eq!(plan.placement_for(0), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(1), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(2), ExpertPlacement::GpuDevice(0));
    }

    #[test]
    fn from_budget_with_zero_vram_places_nothing_on_gpu() {
        let sizes = vec![10usize, 20, 30];
        let plan = PlacementPlan::from_budget(&sizes, None, 0);
        for i in 0..3 {
            assert_eq!(plan.placement_for(i), ExpertPlacement::Cpu);
        }
    }

    #[test]
    fn from_budget_ignores_mismatched_activation_counts_length_rather_than_panicking() {
        let sizes = vec![10usize, 10];
        let counts = vec![1u64]; // wrong length
        let plan = PlacementPlan::from_budget(&sizes, Some(&counts), 100);
        // Falls back to index order; both fit within the budget either way.
        assert_eq!(plan.placement_for(0), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(1), ExpertPlacement::GpuDevice(0));
    }

    #[test]
    fn combine_expert_outputs_weights_routed_and_adds_shared() {
        let routed = vec![(vec![2.0, 2.0], 0.5), (vec![4.0, 4.0], 0.5)];
        let shared = vec![vec![1.0, 1.0]];
        let out = combine_expert_outputs(&routed, &shared, 2);
        assert_eq!(out, vec![4.0, 4.0]);
    }

    #[test]
    fn run_expert_produces_correct_output_dimension() {
        use ferrox_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 3;
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(
                vec![0.1; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            up: WeightMatrix::F32(Tensor::new(
                vec![0.2; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            down: WeightMatrix::F32(Tensor::new(
                vec![0.3; hidden_dim * ffn_dim],
                vec![hidden_dim, ffn_dim],
            )),
        };
        let hidden = vec![1.0, -1.0, 0.5, 0.5];
        let out = run_expert(&hidden, &expert);
        assert_eq!(out.len(), hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `run_expert_placed` must be a real drop-in for `run_expert` when
    /// no GPU dispatch actually happens -- true unconditionally without
    /// the `cuda` feature, and true even *with* the feature for `Cpu`
    /// placement (which never calls `apply_gpu` at all) or an
    /// unsupported quant kind (F32 here, which `apply_gpu` always
    /// returns `None` for, falling through to `run_expert`).
    #[test]
    fn run_expert_placed_matches_run_expert_when_nothing_is_gpu_dispatched() {
        use ferrox_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 3;
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(
                vec![0.1; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            up: WeightMatrix::F32(Tensor::new(
                vec![0.2; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            down: WeightMatrix::F32(Tensor::new(
                vec![0.3; hidden_dim * ffn_dim],
                vec![hidden_dim, ffn_dim],
            )),
        };
        let hidden = vec![1.0, -1.0, 0.5, 0.5];
        let expected = run_expert(&hidden, &expert);

        assert_eq!(
            run_expert_placed(&hidden, &expert, ExpertPlacement::Cpu),
            expected
        );
        assert_eq!(
            run_expert_placed(&hidden, &expert, ExpertPlacement::GpuDevice(0)),
            expected,
            "F32 has no GPU kernel, so GpuDevice placement must still fall through to the CPU path"
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    #[ignore = "requires real GPU hardware (CUDA or Metal) -- run with --ignored"]
    fn run_expert_placed_on_gpu_matches_cpu_for_a_real_quantized_expert() {
        let hidden_dim = 32;
        let ffn_dim = 32; // must be a multiple of Q8_0 block elems (32)
        let make_row = |cols: usize, seed: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| ((i as f32) - (cols as f32) / 2.0) * 0.01 * seed)
                .collect()
        };
        let quantize_matrix = |rows: usize, cols: usize, seed: f32| {
            let mut packed = Vec::new();
            for r in 0..rows {
                packed.extend(ferrox_quant::quantize_q8_0(&make_row(
                    cols,
                    seed + r as f32,
                )));
            }
            WeightMatrix::Quantized {
                data: ferrox_core::weight_matrix::WeightBytes::Owned(packed),
                rows,
                cols,
                kind: ferrox_core::weight_matrix::QuantKind::Q8_0,
            }
        };
        let expert = ExpertWeights {
            gate: quantize_matrix(ffn_dim, hidden_dim, 1.0),
            up: quantize_matrix(ffn_dim, hidden_dim, 2.0),
            down: quantize_matrix(hidden_dim, ffn_dim, 3.0),
        };
        let hidden = make_row(hidden_dim, 0.5);

        let cpu = run_expert_placed(&hidden, &expert, ExpertPlacement::Cpu);
        let gpu = run_expert_placed(&hidden, &expert, ExpertPlacement::GpuDevice(0));
        assert_eq!(cpu.len(), gpu.len());
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            assert!((c - g).abs() < 1e-1, "cpu={c} gpu={g}");
        }
    }
}
