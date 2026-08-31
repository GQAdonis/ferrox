//! Explicit architecture capability registry for the generic GGUF path.
//!
//! Mirrors the pinned llama.cpp `llm_arch` / `LLM_ARCH_NAMES` inventory
//! (`.scratch/llama.cpp/src/llama-arch.{h,cpp}`) with Ferrox-side
//! classification into decoder families, memory kinds, and scope.
//! Unknown strings and detected-but-unimplemented features fail closed
//! (`LoadError`) instead of silently defaulting into fluent-but-wrong
//! logits.
//!
//! Architecture names are registry keys only. Hot-path kernels never
//! branch on them; load-time resolution produces an [`ArchProfile`]
//! whose fields the decoder reads as plain data.

use crate::config::RopeLayout;

/// How far this architecture is in Ferrox's delivery scope (plan:
/// text-generation parity; encoder/multimodal/diffusion/audio deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchScope {
    /// Autoregressive / encoder-decoder text generation — in scope.
    TextGeneration,
    /// Encoder / embedding / pooling models — deferred.
    DeferredEncoderEmbedding,
    /// Vision / multimodal projector paths — deferred.
    DeferredMultimodal,
    /// Diffusion / masked-LM samplers — deferred.
    DeferredDiffusion,
    /// Audio tokenizers / codecs — deferred.
    DeferredAudio,
    /// Enum present in llama.cpp but not a real serve target here.
    EnumOnly,
}

/// Shared execution family (maps many GGUF strings onto one engine path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderFamily {
    /// Standard GQA (+ optional MoE) with whole-vector optional QK-norm.
    StandardGqa,
    /// Qwen3-style: explicit head_dim + per-head Q/K RMSNorm before RoPE.
    Qwen3Family,
    /// Gemma-family: embedding scale, post-norms, softcap, SWA pattern, GeGLU.
    GemmaFamily,
    /// Phi-family: fused QKV and/or fused gate+up SwiGLU.
    PhiFamily,
    /// DeepSeek-2 / Mistral4 MLA (not generic GQA).
    Mla,
    /// Attn + SSM / delta-net hybrids.
    Hybrid,
    /// Pure recurrent (Mamba / RWKV) — no KV cache.
    Recurrent,
    /// T5-style encoder-decoder.
    EncoderDecoder,
    /// Dedicated Ferrox stacks (GLM DSA, DeepSeek V4, Kimi).
    Dedicated,
    /// In-repo synthetic fixtures.
    TestFixture,
}

/// Memory / KV backend selected once at load (llama.cpp `create_memory`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    KvGqa,
    KvIswa,
    KvMla,
    KvDsa,
    KvDsv4,
    Recurrent,
    Hybrid,
    None,
}

/// How `attn_q_norm` / `attn_k_norm` weights are applied (when present).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QkNormStyle {
    /// OLMoE: one RMSNorm over the full Q/K projection width.
    #[default]
    WholeVector,
    /// Qwen3 / Gemma3: RMSNorm per head with weight length `head_dim`.
    PerHead,
}

/// Architectures on the shared generic-GQA path that somebody has
/// actually PROVEN, and the evidence for each.
///
/// The generic path is a guess: it assumes an architecture is plain GQA
/// because nothing said otherwise. That guess has already been wrong
/// five times. `gpt2`, `mpt`, `refact`, `bloom` and `jais` all sat here
/// computing ALiBi or learned absolute position embeddings as though
/// they were NEOX RoPE, and every downstream guard missed them: two
/// hardcode their ALiBi slope with no GGUF key, one leaves no unread
/// tensor, and the RoPE pin excluded their group by construction.
///
/// So membership here is not "we think this works", it is "there is a
/// benchmark row, a pinned logit comparison against llama.cpp, or a
/// fixture". Everything else on the generic path is UNAUDITED and says
/// so at load time rather than running and hoping.
///
/// Adding a name here without evidence defeats the entire point.
pub const AUDITED_GENERIC_GQA: &[&str] = &[
    // Bench rows in benchmarks/suite.json, measured against llama.cpp
    // on the same host and file.
    "llama",    // TinyLlama, Mistral, Mixtral, SmolLM2, Llama-3.x all tag llama
    "qwen2",    // Qwen2.5-0.5B
    "qwen2moe", // Qwen1.5-MoE-A2.7B
    "qwen3",    // Qwen3-0.6B
    "olmoe",    // OLMoE-1B-7B
    "gemma2",   // Gemma-2-2B
    "gemma3",   // Gemma-3-1B
    "phi3",     // Phi-4-mini tags phi3
    // Pinned against real libllama logits in tests/.
    "gpt-oss", "dots1",
];

/// Is this architecture's use of the shared generic path backed by
/// evidence?
pub fn is_audited_generic(arch: &str) -> bool {
    AUDITED_GENERIC_GQA.contains(&arch)
}

/// How the generic `Decoder` / `ModelConfig::from_gguf` path treats a
/// GGUF architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchPath {
    /// Standard GQA (+ optional MoE) decoder; RoPE layout is known.
    GenericGqa { rope: RopeLayout },
    /// In-repo test fixtures (`ferroxtest*`) — not a real model family.
    TestFixture { rope: RopeLayout },
    /// Real architecture, but must not be loaded through the generic
    /// GQA decoder (wrong attention / residual math).
    DedicatedOnly { reason: &'static str },
    /// In the llama.cpp inventory but out of Ferrox scope for now.
    Deferred { reason: &'static str },
}

/// Load-time resolved profile for one GGUF `general.architecture` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchProfile {
    pub gguf_name: &'static str,
    pub scope: ArchScope,
    pub family: DecoderFamily,
    pub memory: MemoryKind,
    pub rope: RopeLayout,
    pub path: ArchPath,
    /// Default QK-norm style when norm tensors are present; loader may
    /// refine from tensor length.
    pub qk_norm: QkNormStyle,
}

fn prof(
    name: &'static str,
    scope: ArchScope,
    fam: DecoderFamily,
    mem: MemoryKind,
    rope: RopeLayout,
    path: ArchPath,
    qk: QkNormStyle,
) -> ArchProfile {
    ArchProfile {
        gguf_name: name,
        scope,
        family: fam,
        memory: mem,
        rope,
        path,
        qk_norm: qk,
    }
}

fn gqa_norm(name: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::StandardGqa,
        MemoryKind::KvGqa,
        RopeLayout::Norm,
        ArchPath::GenericGqa {
            rope: RopeLayout::Norm,
        },
        QkNormStyle::WholeVector,
    )
}

fn gqa_neox(name: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::StandardGqa,
        MemoryKind::KvGqa,
        RopeLayout::Neox,
        ArchPath::GenericGqa {
            rope: RopeLayout::Neox,
        },
        QkNormStyle::WholeVector,
    )
}

fn dedicated(name: &'static str, reason: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::Dedicated,
        MemoryKind::KvGqa,
        RopeLayout::Norm,
        ArchPath::DedicatedOnly { reason },
        QkNormStyle::WholeVector,
    )
}

fn deferred_scope(name: &'static str, scope: ArchScope, reason: &'static str) -> ArchProfile {
    prof(
        name,
        scope,
        DecoderFamily::StandardGqa,
        MemoryKind::None,
        RopeLayout::Neox,
        ArchPath::Deferred { reason },
        QkNormStyle::WholeVector,
    )
}

/// Full inventory keyed by GGUF `general.architecture` string.
/// Kept in sync with `.scratch/llama.cpp/src/llama-arch.cpp` `LLM_ARCH_NAMES`.
pub fn architecture_catalog() -> &'static [ArchProfile] {
    use std::sync::OnceLock;
    use ArchScope::*;
    use DecoderFamily::*;
    use MemoryKind::*;
    use QkNormStyle::*;
    use RopeLayout::*;

    static CAT: OnceLock<Vec<ArchProfile>> = OnceLock::new();
    CAT.get_or_init(|| {
        let mut v = Vec::with_capacity(160);
        // --- Verified / standard GQA (Norm RoPE) ---
        for n in [
            "llama",
            "deci",
            "baichuan",
            "internlm2",
            "xverse",
            "olmo",
            "arctic",
            "deepseek",
            "chatglm",
            "granite",
            "granitemoe",
            "granite-moe",
            "mistral3",
            "maincoder",
            "arcee",
            "ernie4_5",
            "ernie4_5-moe",
            "bailingmoe",
            "nanbeige",
            "plm",
        ] {
            v.push(gqa_norm(n));
        }
        for n in [
            "olmoe", "qwen2", "qwen2moe", "mistral",
            "mixtral", "olmo2", "bitnet",
            "grok", "dbrx", "exaone4", "yi",
            // llama-model.cpp `llama_model_rope_type`: LLM_ARCH_OPENAI_MOE
            // falls in the `return LLAMA_ROPE_TYPE_NEOX` group, and a live
            // load of a gpt-oss GGUF prints `rope type = 2` (= NEOX).
            // ferrox had it on the interleaved (NORM) list, which rotates
            // the wrong pairs of every Q/K head.
            "gpt-oss",
            // Same audit, run over every arch at once against
            // `llama_model_rope_type`'s NEOX group
            // (llama-model.cpp:2613-2683). These 24 were on ferrox's
            // interleaved (NORM) list and reach the generic GQA decoder,
            // so every one of them rotated the wrong pairs of every Q/K
            // head and answered fluently and wrongly. Pinned by
            // `rope_layout_matches_llama_cpp` below; dots1 additionally
            // checked end-to-end against llama.cpp's own logits in
            // `tests/moe_routing_bias.rs`.
            "afmoe",
            "apertus",
            "bailingmoe2",
            "dots1",
            "exaone",
            "exaone-moe",
            "grovemoe",
            "hunyuan-dense",
            "hunyuan-moe",
            "laguna",
            "mellum",
            "mimo2",
            "minicpm3",
            "openelm",
            "plamo3",
            "seed_oss",
            "smallthinker",
            "step35",
            "talkie",
        ] {
            v.push(gqa_neox(n));
        }
        // --- No RoPE at all: refused, not rotated ------------------
        //
        // `llama_model_rope_type` opens with a `LLAMA_ROPE_TYPE_NONE`
        // group, and these five sat on ferrox's NEOX list instead. The
        // generic decoder rotates every Q/K head of every layer, so each
        // of them loaded, ran at full speed, and answered fluently from
        // positions the checkpoint never encodes that way, the same
        // silent failure the 24-arch RoPE audit found, one level worse,
        // because here the right answer is *no rotation*.
        //
        // Worse still for a metadata gate: `bloom` and `refact` hardcode
        // `f_max_alibi_bias = 8.0f` in `load_arch_hparams` and carry no
        // key at all, so `unsupported_feature_keys` could never have seen
        // them. Only the registry can. `tests/rope_layout.rs`'s
        // `LLAMA_NO_ROPE` pins the group so a later edit cannot quietly
        // put one back on a rotating path.
        for (n, reason) in [
            (
                "smollm3",
                "a NoPE layer pattern: llama.cpp hardcodes \
                 `hparams.n_no_rope_layer_step = 4` (src/models/smollm3.cpp:5) and \
                 skips RoPE where `(il + 1) % 4 == 0` (:69), so 9 of a 36-layer \
                 SmolLM3-3B's layers get NO rotation at all. There is NO GGUF key \
                 for it, so no metadata gate could see it: the tensor set matches \
                 the generic llama set exactly and the file loads clean. The \
                 generic decoder rotates every layer, which is a different model. \
                 Same shape as the ALiBi group below, and found the same way",
            ),
            (
                "gpt2",
                "learned absolute position embeddings (`position_embd.weight`, \
                 src/models/gpt2.cpp:19,74) and no RoPE; the generic decoder has no \
                 slot for them and rotates instead",
            ),
            (
                "mpt",
                "ALiBi attention bias (src/models/mpt.cpp:6), plus an optional \
                 learned `position_embd` and an optional QKV clamp; the generic \
                 decoder implements none of the three and applies RoPE instead",
            ),
            (
                "refact",
                "ALiBi attention bias, hardcoded `f_max_alibi_bias = 8.0f` with no \
                 GGUF key to detect it (src/models/refact.cpp:12); the generic \
                 decoder applies RoPE instead",
            ),
            (
                "bloom",
                "ALiBi attention bias, hardcoded `f_max_alibi_bias = 8.0f` with no \
                 GGUF key (src/models/bloom.cpp:18), plus a `token_embd_norm` the \
                 generic decoder never applies; RoPE is applied instead",
            ),
            (
                "jais",
                "ALiBi attention bias (src/models/jais.cpp:5); the generic decoder \
                 applies RoPE instead",
            ),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                StandardGqa,
                KvGqa,
                // No layout is right here. `Norm` is the struct's least
                // surprising filler and nothing reads it: the load
                // refuses in `ModelConfig::from_gguf` before any graph
                // asks. `rope_layout_matches_llama_cpp` skips
                // non-generic paths for exactly this reason.
                Norm,
                ArchPath::DedicatedOnly { reason },
                WholeVector,
            ));
        }
        // --- Required bias tensors the generic decoder has no slot for
        //
        // Found by transcribing every `create_tensor(tn(..., "bias"), ...)`
        // llama.cpp's per-architecture loaders create with flag `0`
        // (REQUIRED, as opposed to `TENSOR_NOT_REQUIRED`). Required means
        // every real checkpoint of that architecture carries it, so this
        // is not a "some files might" gate.
        //
        // `AttnWeights` carries exactly three of them -- `attn_q.bias`,
        // `attn_k.bias`, `attn_v.bias` -- and `GptOssWeights` carries
        // gpt-oss's `attn_output.bias` and `ffn_gate_inp.bias`. Nothing
        // else has anywhere to go:
        //
        // - `attn_output.bias`, `ffn_{up,down,gate}.bias` and the
        //   `output.bias` on the LM head are read by no loader path, so
        //   they are simply dropped: the projection runs unbiased.
        // - `attn_norm.bias` / `ffn_norm.bias` / `output_norm.bias` are
        //   the marker of a real LayerNorm. The generic decoder only has
        //   `rms_norm(x, w, eps)` -- no mean subtraction and no bias --
        //   so it computes a different normalisation at every layer.
        // - `attn_qkv.bias` is the *fused* spelling.
        //   `load_qkv_projections` splits a fused `attn_qkv.weight` but
        //   looks for the bias only under the split `attn_q.bias` names,
        //   finds nothing, and runs unbiased.
        //
        // Every one of these loads clean and answers fluently, which is
        // why they are refused here rather than left to a tensor gate.
        // Pinned by `tests/attn_bias.rs`.
        for (n, rope, reason) in [
            (
                "codeshell",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_down.bias`, `ffn_up.bias` \
                 (src/models/codeshell.cpp:36,42,45), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:24,31,39) \
                 -- the generic decoder is RMSNorm-only and drops all six",
            ),
            (
                "jais2",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_up.bias`, `ffn_down.bias` \
                 (src/models/jais2.cpp:41,48,50), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:20,30,44). \
                 Only its Q/K/V biases (:38-40) would have been applied",
            ),
            (
                "starcoder",
                Norm,
                "required bias tensors with no slot in the generic decoder: the \
                 *fused* `attn_qkv.bias` (src/models/starcoder.cpp:40), which \
                 `load_qkv_projections` never looks for because it reads bias only \
                 under the split `attn_q.bias` names; `attn_output.bias`, \
                 `ffn_down.bias`, `ffn_up.bias` (:43,49,52); and the LayerNorm \
                 biases `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` \
                 (:24,37,46). It also adds a learned `position_embd` to the \
                 embeddings (:75) that the generic decoder has no slot for",
            ),
            (
                "starcoder2",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_down.bias`, `ffn_up.bias` \
                 (src/models/starcoder2.cpp:41,50,51), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:23,35,44)",
            ),
            (
                "phimoe",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias` and an `output.bias` on the LM head \
                 (src/models/phimoe.cpp:33,23), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:21,29,36). \
                 `phi3` stays generic: it requires none of them",
            ),
            (
                "nemotron",
                Neox,
                "required LayerNorm biases `output_norm.bias`, `attn_norm.bias`, \
                 `ffn_norm.bias` (src/models/nemotron.cpp:19,26,35). llama.cpp \
                 normalises with `build_norm(..., LLM_NORM, ...)` and a bias; the \
                 generic decoder applies RMSNorm with weight only, which is a \
                 different function of the same tensors at every layer",
            ),
            (
                "orion",
                Neox,
                "required LayerNorm biases `output_norm.bias`, `attn_norm.bias`, \
                 `ffn_norm.bias` (src/models/orion.cpp:18,25,31); the generic \
                 decoder is RMSNorm-only and drops all three",
            ),
            (
                "stablelm",
                Neox,
                "required LayerNorm biases `output_norm.bias` and `attn_norm.bias` \
                 (src/models/stablelm.cpp:20,28); the generic decoder is \
                 RMSNorm-only and drops both",
            ),
            (
                "qwen",
                Neox,
                "a required *fused* `attn_qkv.bias` (src/models/qwen.cpp:28). \
                 `load_qkv_projections` splits the fused `attn_qkv.weight` but reads \
                 bias only under the split `attn_q.bias` / `attn_k.bias` / \
                 `attn_v.bias` names, so Qwen-1's QKV bias is silently dropped and \
                 every Q, K and V projection runs unbiased. Qwen-2 and later store \
                 the split spelling and stay generic",
            ),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                StandardGqa,
                KvGqa,
                // Unlike the no-RoPE group above, the layout here is
                // real and `rope_layout_matches_llama_cpp` still checks
                // it: refusing for a bias is not a licence to forget
                // what these rotate as.
                rope,
                ArchPath::DedicatedOnly { reason },
                WholeVector,
            ));
        }
        v.push(prof(
            "qwen3",
            TextGeneration,
            Qwen3Family,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "qwen3moe",
            TextGeneration,
            Qwen3Family,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "gemma",
            TextGeneration,
            GemmaFamily,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "gemma2",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "gemma3",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        // Gemma-4 text GGUFs (E2B): per-layer embeddings, shared-KV
        // layers, and split SWA/full head dims — dedicated
        // [`crate::gemma4_engine::Gemma4Engine`] (not GenericGqa).
        for n in ["gemma4", "gemma4-assistant"] {
            v.push(prof(
                n,
                TextGeneration,
                GemmaFamily,
                KvIswa,
                Neox,
                ArchPath::DedicatedOnly {
                    reason: "use load_gemma4_engine_from_path / ServedEngine::Gemma4",
                },
                PerHead,
            ));
        }
        // Refused, not implemented: the generic decoder computes
        // `x + attn(norm(x))` then `y + ffn(norm(y))`, and every arch
        // here computes something else that no tensor and (for MiniCPM)
        // no metadata key makes visible. See
        // `unsupported_scaling_keys` for the metadata-visible half of
        // the same class.
        const PARALLEL_RESIDUAL: &str =
            "parallel attention+FFN residual -- llama.cpp feeds both branches the *same* \
             normed input and sums `inpL + attn_out + ffn_out` once; the generic decoder \
             computes the sequential form, which is a different graph";
        for (n, rope, fam) in [
            // src/models/cohere2.cpp:120-134, cohere2moe.cpp:222-266,
            // command-r.cpp:106-119. All three also carry a
            // `logit_scale` the generic decoder does not apply.
            ("command-r", Norm, StandardGqa),
            ("cohere2", Norm, StandardGqa),
            ("cohere2moe", Norm, StandardGqa),
            // src/models/falcon.cpp:121-135 (and an `attn_norm_2` the
            // generic decoder has no slot for).
            ("falcon", Neox, StandardGqa),
            // src/models/gptneox.cpp:147-195 -- parallel or sequential
            // per `use_par_res`, and the generic decoder implements
            // neither branch of that choice.
            ("gptneox", Neox, StandardGqa),
            // src/models/phi2.cpp:116-117, plamo.cpp:97-112.
            ("phi2", Neox, PhiFamily),
            ("plamo", Neox, StandardGqa),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                fam,
                KvGqa,
                rope,
                ArchPath::DedicatedOnly {
                    reason: PARALLEL_RESIDUAL,
                },
                WholeVector,
            ));
        }
        // MiniCPM is the case `unsupported_scaling_keys` cannot catch:
        // `src/models/minicpm.cpp:4-14` *hardcodes* an embedding
        // multiplier of 12.0, a residual multiplier of
        // `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`,
        // and only then lets the GGUF override them. An older MiniCPM
        // export carrying none of the three keys is still scaled by all
        // three, so a key-presence gate sees nothing and the generic
        // decoder computes an unscaled graph.
        v.push(prof(
            "minicpm",
            TextGeneration,
            StandardGqa,
            KvGqa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "unconditional embedding/residual/logit multipliers that llama.cpp \
                         applies even when the GGUF omits every key; not applied by the \
                         generic decoder",
            },
            WholeVector,
        ));
        v.push(prof(
            "phi3",
            TextGeneration,
            PhiFamily,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            WholeVector,
        ));
        // Phi-4 GGUFs share the phi3 fused-QKV / fused gate+up graph
        // (PhiFamily). Many community checkpoints still tag `phi3`; admit
        // `phi4` the same way so either string can load. Receipts / head-dim
        // FA-vec coverage remain P6 evidence work — not a speed claim.
        v.push(prof(
            "phi4",
            TextGeneration,
            PhiFamily,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            WholeVector,
        ));
        // Llama 4: MoE + interleaved / non-generic attention graph — not
        // safe to admit as GenericGqa (was wrongly listed with plain llama).
        v.push(prof(
            "llama4",
            TextGeneration,
            Dedicated,
            KvGqa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "llama4 MoE + non-GQA attn — see llama4_engine.rs tensor list",
            },
            WholeVector,
        ));
        // MiniMax M2/M3: 256-expert sigmoid MoE + MTP — not generic GQA.
        for n in ["minimax-m2", "minimax-m3"] {
            v.push(prof(
                n,
                TextGeneration,
                Dedicated,
                KvGqa,
                // llama-model.cpp puts both in the NEOX group. Inert
                // today (minimax_engine.rs carries its own RoPE and
                // never reads this field), but the table advertises
                // itself as a mirror of llama.cpp's, so it says NEOX.
                Neox,
                ArchPath::DedicatedOnly {
                    reason: "MiniMax 256-expert sigmoid MoE + MTP — see minimax_engine.rs",
                },
                WholeVector,
            ));
        }
        v.push(prof(
            "deepseek2",
            TextGeneration,
            Mla,
            KvMla,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "DeepSeek-2 MLA needs the MLA engine, not generic GQA",
            },
            WholeVector,
        ));
        v.push(prof(
            "deepseek32",
            TextGeneration,
            Mla,
            KvDsa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "DeepSeek-3.2 DSA/MLA needs the dedicated sparse/MLA stack",
            },
            WholeVector,
        ));
        v.push(prof(
            "mistral4",
            TextGeneration,
            Mla,
            KvMla,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "mistral4 reuses DeepSeek-2 MLA loader/graph in llama.cpp",
            },
            WholeVector,
        ));
        v.push(dedicated(
            "glm-dsa",
            "use ferrox_models::glm52_decoder / glm52_gguf_loader (DSA), not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "glm4",
            "use ferrox_models::glm52_decoder / glm52_gguf_loader, not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "glm4moe",
            "use ferrox_models::glm52_decoder / glm52_gguf_loader, not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "deepseek4",
            "DeepSeek V4 needs CSA/HCA + mHC assembly; generic GQA Decoder is not valid",
        ));
        v.push(dedicated(
            "kimi-linear",
            "use ferrox_models::kimi_decoder / kimi_loader, not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "kimi_k3",
            "use ferrox_models::kimi_decoder / kimi_loader, not the generic GQA Decoder",
        ));
        for (n, rope) in [
            ("jamba", Neox),
            ("falcon-h1", Neox),
            ("plamo2", Neox),
            ("granitehybrid", Norm),
            ("granite-hybrid", Norm),
            ("lfm2", Neox),
            ("lfm2moe", Neox),
            ("nemotron_h", Neox),
            ("nemotron_h_moe", Neox),
            ("qwen3next", Neox),
            ("qwen35", Neox),
            ("qwen35moe", Neox),
        ] {
            let qk = if n.starts_with("qwen3") {
                PerHead
            } else {
                WholeVector
            };
            v.push(prof(
                n,
                TextGeneration,
                DecoderFamily::Hybrid,
                MemoryKind::Hybrid,
                rope,
                ArchPath::DedicatedOnly {
                    reason: "hybrid attn+SSM/delta-net engine not yet on the serve path",
                },
                qk,
            ));
        }
        for n in ["mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"] {
            v.push(prof(
                n,
                TextGeneration,
                DecoderFamily::Recurrent,
                MemoryKind::Recurrent,
                Neox,
                ArchPath::DedicatedOnly {
                    reason: "recurrent engine not yet on the serve path",
                },
                WholeVector,
            ));
        }
        v.push(prof(
            "t5",
            TextGeneration,
            EncoderDecoder,
            None,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "T5 encoder-decoder engine not yet on the serve path",
            },
            WholeVector,
        ));
        for (n, scope, reason) in [
            (
                "t5encoder",
                DeferredEncoderEmbedding,
                "encoder-only; deferred from text-generation parity",
            ),
            ("bert", DeferredEncoderEmbedding, "encoder/embedding; deferred"),
            (
                "modern-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "nomic-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "nomic-bert-moe",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "neo-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "jina-bert-v2",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "jina-bert-v3",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "eurobert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "llama-embed",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            (
                "gemma-embedding",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            (
                "pangu-embedded",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            ("yi-vl", DeferredMultimodal, "Yi vision-language; deferred"),
            ("qwen2vl", DeferredMultimodal, "vision-language; deferred"),
            ("qwen3vl", DeferredMultimodal, "vision-language; deferred"),
            ("qwen3vlmoe", DeferredMultimodal, "vision-language; deferred"),
            ("cogvlm", DeferredMultimodal, "vision-language; deferred"),
            ("chameleon", DeferredMultimodal, "multimodal; deferred"),
            ("hunyuan_vl", DeferredMultimodal, "vision-language; deferred"),
            ("paddleocr", DeferredMultimodal, "OCR multimodal; deferred"),
            ("hy_v3", DeferredMultimodal, "multimodal; deferred"),
            ("deepseek2-ocr", DeferredMultimodal, "OCR multimodal; deferred"),
            ("dream", DeferredDiffusion, "diffusion LM; deferred"),
            ("llada", DeferredDiffusion, "diffusion LM; deferred"),
            ("llada-moe", DeferredDiffusion, "diffusion LM; deferred"),
            ("rnd1", DeferredDiffusion, "diffusion LM; deferred"),
            (
                "wavtokenizer-dec",
                DeferredAudio,
                "audio tokenizer; deferred",
            ),
            (
                "eagle3",
                EnumOnly,
                "speculative draft head; not a standalone decoder target",
            ),
            (
                "dflash",
                EnumOnly,
                "speculative draft head; not a standalone decoder target",
            ),
            ("clip", EnumOnly, "quantize dummy only"),
            ("gptj", EnumOnly, "enum-only in llama.cpp factory gap"),
            ("(unknown)", EnumOnly, "llama.cpp unknown sentinel"),
        ] {
            v.push(deferred_scope(n, scope, reason));
        }
        v.push(prof(
            "gemma3n",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "gemma3n AltUp/Laurel tensors not implemented in the generic decoder",
            },
            PerHead,
        ));
        for n in ["ferroxtest", "ferroxtestmoe", "ferroxtestmixed"] {
            v.push(prof(
                n,
                TextGeneration,
                TestFixture,
                KvGqa,
                Neox,
                ArchPath::TestFixture { rope: Neox },
                WholeVector,
            ));
        }
        v
    })
    .as_slice()
}

/// Resolve a GGUF `general.architecture` value to its profile.
pub fn resolve_profile(arch: &str) -> Option<&'static ArchProfile> {
    architecture_catalog().iter().find(|p| p.gguf_name == arch)
}

/// Resolve a GGUF `general.architecture` value. `None` means the string
/// is not in the registry — callers must fail closed rather than guess.
pub fn resolve_architecture(arch: &str) -> Option<ArchPath> {
    resolve_profile(arch).map(|p| p.path)
}

/// llama.cpp's hardcoded alternating sliding-window layout for one
/// architecture: the period, *and* which end of each period is the
/// full-attention layer.
///
/// `llama_hparams::set_swa_pattern` (`src/llama-hparams.cpp:8-22`) has
/// two phases, and they are not interchangeable:
///
/// - `dense_first = false`: `is_swa[il] = il % p < (p - 1)` — the
///   **last** layer of every period is full attention.
/// - `dense_first = true`:  `is_swa[il] = il % p != 0` — the **first**
///   layer of every period is full attention.
///
/// For a 32-layer period-4 model the two disagree on 16 of the 32
/// layers. Storing only the period would therefore not be a partial
/// transcription, it would be a wrong one for the four architectures
/// llama.cpp passes `dense_first = true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaPattern {
    /// llama.cpp's `swa_period` seed literal.
    pub period: usize,
    /// llama.cpp's `dense_first` argument to `set_swa_pattern`.
    pub dense_first: bool,
}

/// Every architecture for which llama.cpp seeds a sliding-window period
/// *before* letting `{arch}.attention.sliding_window_pattern` override
/// it, transcribed from `src/models/*.cpp`.
///
/// The period is not in the file for these families — llama.cpp
/// hardcodes it per architecture and only lets the metadata key override
/// it (`ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
/// swa_period, false)` after seeding `swa_period` with the literal
/// below). A missing key therefore does **not** mean "every layer is
/// windowed", which is what ferrox assumed: `layer_sliding_window`
/// returns the window for all layers when `swa_pattern` is `None`, so a
/// gpt-oss or cohere2 checkpoint ran its full-attention layers through a
/// 128-token window and answered from a truncated history.
///
/// Two llama.cpp spellings are deliberately absent, because neither is
/// a per-arch *period*:
///
/// - `set_swa_pattern(0)` (`deepseek4.cpp:68`, `dflash.cpp:54`) makes
///   **every** layer sliding, which is what ferrox already does for a
///   declared window with no pattern.
/// - `set_swa_pattern(1)` (`phi3.cpp:23`) makes **no** layer sliding,
///   and phi3 zeroes `n_swa` and sets `swa_type = NONE` on the same
///   branch, so there is no window left to place.
///
/// Architectures that only ever read a per-layer *array*
/// (`get_key_or_arr(..., hparams.is_swa_impl, n_layer)`: `gemma4`,
/// `gemma4-assistant`, `step35`, `mimo2`, `dflash`) seed no scalar and
/// so have no default to pin.
///
/// Pinned by `tests/swa_pattern.rs`.
pub fn default_swa_layout(arch: &str) -> Option<SwaPattern> {
    let last_dense = |period| {
        Some(SwaPattern {
            period,
            dense_first: false,
        })
    };
    let dense_first = |period| {
        Some(SwaPattern {
            period,
            dense_first: true,
        })
    };
    match arch {
        // src/models/openai-moe.cpp:9
        "gpt-oss" => last_dense(2),
        // src/models/gemma2.cpp:6
        "gemma2" => last_dense(2),
        // src/models/gemma3.cpp:7
        "gemma3" => last_dense(6),
        // src/models/gemma3n.cpp:4 says 5, NOT 6. This was transcribed
        // as 6 alongside gemma3 and is simply wrong. Inert only because
        // `gemma3n` refuses for other reasons today.
        "gemma3n" => last_dense(5),
        // src/models/gemma-embedding.cpp:5. Deferred (embedding scope),
        // so latent rather than live.
        "gemma-embedding" => last_dense(6),
        // src/models/cohere2.cpp:5, exaone4.cpp:7, olmo2.cpp:9
        "cohere2" | "exaone4" | "olmo2" => last_dense(4),
        // Added after an audit found this table covered 6 architectures
        // where llama.cpp hardcodes a period for 17. A MISSING entry is
        // not neutral: with no period, every layer gets windowed, so a
        // model whose full-attention layers should see the whole context
        // sees only a window instead. That is a different model, and it
        // fails silently.
        //
        // src/models/mellum.cpp:11
        "mellum" => last_dense(4),
        // src/models/exaone-moe.cpp:6. SWA is unconditional there
        // with n_swa = 128, so without this every layer ran with a
        // 128-token history.
        "exaone-moe" => last_dense(4),
        // src/models/afmoe.cpp:17, plamo3.cpp:9. Both refuse for other
        // reasons today, so these are latent rather than live, and
        // pinned here so they stay right if that changes.
        "afmoe" => last_dense(4),
        "plamo3" => last_dense(8),
        // src/models/llama4.cpp:19 ("pattern: 3 chunked - 1 full").
        // `llama4` is `DedicatedOnly` today, so latent.
        "llama4" => last_dense(4),
        // --- dense_first = true -----------------------------------
        //
        // These four put the full-attention layer at `il % p == 0`, not
        // at `il % p == p - 1`. `ModelConfig::layer_sliding_window`
        // implements only the `dense_first = false` phase, so
        // `default_swa_pattern` below refuses to hand them a bare
        // period rather than place their full layers one index off.
        //
        // src/models/smallthinker.cpp:9-11. LIVE: `smallthinker` is on
        // the generic GQA path.
        "smallthinker" => dense_first(4),
        // src/models/laguna.cpp:39-41 (its own comment: "XS.2: FULL at
        // il%4==0"). LIVE: `laguna` is on the generic GQA path.
        "laguna" => dense_first(4),
        // src/models/cohere2moe.cpp:31-33. `DedicatedOnly` today
        // (parallel attention+FFN residual), so latent.
        "cohere2moe" => dense_first(4),
        // src/models/modern-bert.cpp:8-10. Deferred (encoder scope), so
        // latent.
        "modern-bert" => dense_first(3),
        _ => None,
    }
}

/// The alternating sliding-window period an architecture uses when its
/// GGUF carries `{arch}.attention.sliding_window` but *not*
/// `{arch}.attention.sliding_window_pattern`.
///
/// This is the half of [`default_swa_layout`] that ferrox's decoder can
/// currently act on. `ModelConfig::layer_sliding_window` hardcodes
/// llama.cpp's `dense_first = false` phase (full attention where
/// `(il + 1) % period == 0`), so an architecture llama.cpp loads with
/// `dense_first = true` gets `None` here on purpose:
///
/// - handing the caller the bare period would put every full-attention
///   layer one index off, which for a 32-layer period-4 `smallthinker`
///   is **16** layers attending over the wrong span;
/// - `None` leaves ferrox's existing behaviour, every layer windowed,
///   which is 8 layers wrong for the same model.
///
/// Both are wrong and neither is acceptable; the smaller of the two is
/// what this returns until `ModelConfig` can carry `dense_first` and
/// the loader can refuse what it still cannot express. See
/// `tests/swa_pattern.rs::dense_first_architectures_are_not_handed_a_period_the_decoder_would_misplace`.
pub fn default_swa_pattern(arch: &str) -> Option<usize> {
    match default_swa_layout(arch) {
        Some(p) if !p.dense_first => Some(p.period),
        _ => None,
    }
}

/// True when this architecture's SWA layers use the model's own RoPE
/// base rather than llama.cpp's `rope_freq_base_train_swa` default of
/// `10000`.
///
/// `llama_hparams` defaults that field to `10000.0f`
/// (`src/llama-hparams.h:127`) and the Gemma-3 lineage relies on the
/// default; the architectures listed here instead open with
/// `hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;`
/// before letting `rope.freq_base_swa` override it. ferrox applied the
/// Gemma default to everything, which rotates a gpt-oss SWA layer at
/// theta 10000 instead of its real 150000.
pub fn swa_rope_base_follows_model(arch: &str) -> bool {
    matches!(
        arch,
        "afmoe"
            | "cohere2"
            | "cohere2moe"
            | "dflash"
            | "exaone-moe"
            | "exaone4"
            | "gemma2"
            | "laguna"
            | "llama4"
            | "mellum"
            | "olmo2"
            | "gpt-oss"
            | "smallthinker"
    )
}

/// Metadata keys that, when present with a nonzero value, require math
/// ferrox's generic decoder does not implement *unless* the architecture
/// profile opts into those features (Gemma family).
pub fn unsupported_feature_keys(arch: &str) -> Vec<(String, &'static str)> {
    let profile = resolve_profile(arch);
    // Gemma family implements softcap + SWA pattern; others still refuse.
    if matches!(profile.map(|p| p.family), Some(DecoderFamily::GemmaFamily)) {
        return Vec::new();
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    vec![
        (
            key("attention.logit_softcapping"),
            "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ),
        (
            key("final_logit_softcapping"),
            "final logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ),
        (
            key("attention.sliding_window_pattern"),
            "alternating sliding-window pattern (Gemma 2+); not implemented in the generic decoder",
        ),
    ]
}

/// Scalar multipliers a checkpoint can declare in **metadata** that the
/// generic decoder does not apply, with the value that means "no-op".
///
/// These are the blind spot left by
/// [`crate::loader::assert_every_tensor_consumed`]: that gate catches a
/// missing *tensor*, but Granite / MiniCPM / Command-R style multipliers
/// are hparams, not weights, so a checkpoint carrying them loads
/// cleanly, runs at full speed, and computes a graph scaled differently
/// from the one the checkpoint was trained as. Nothing says so.
///
/// llama.cpp key names (`llama-arch.cpp`):
/// `%s.logit_scale` (`LLM_KV_LOGIT_SCALE`), `%s.residual_scale`,
/// `%s.embedding_scale`, `%s.attention.scale`. Granite reads all four
/// (`src/models/granite.cpp::load_arch_hparams`); MiniCPM and
/// Command-R/Cohere2 read the subset they use.
///
/// **This is a refusal, not an implementation.** `residual_scale` in
/// particular multiplies the attention and FFN branch outputs before
/// every residual add, which in ferrox means every CPU decode/prefill/
/// multi-seq path *and* the fused Metal kernels that fold the residual
/// in — landing it half-way would be exactly the silent divergence this
/// list exists to stop. Until the math is there, a checkpoint that
/// declares one of these is refused by name.
///
/// The no-op value differs by key: the three `*_scale` multipliers are
/// `1.0`, while llama.cpp's `f_attention_scale` uses `0.0` as its
/// "unset, use 1/sqrt(head_dim)" sentinel.
pub fn unsupported_scaling_keys(arch: &str) -> Vec<(String, &'static str, f32)> {
    let profile = resolve_profile(arch);
    // Gemma implements its own embedding scale and attention scale.
    if matches!(profile.map(|p| p.family), Some(DecoderFamily::GemmaFamily)) {
        return Vec::new();
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    vec![
        (
            key("logit_scale"),
            "logit multiplier (Granite / Command-R `logits_scaling`); not applied by the generic decoder",
            1.0,
        ),
        (
            key("residual_scale"),
            "residual multiplier (Granite `residual_multiplier`); not applied by the generic decoder",
            1.0,
        ),
        (
            key("embedding_scale"),
            "embedding multiplier (Granite / MiniCPM `embedding_multiplier`); the generic decoder only scales embeddings for the Gemma family",
            1.0,
        ),
        (
            key("attention.scale"),
            "explicit attention score scale (Granite `attention_multiplier`); the generic decoder always uses 1/sqrt(head_dim)",
            0.0,
        ),
    ]
}

/// Markdown coverage table for docs / CI drift checks.
pub fn coverage_report_markdown() -> String {
    let mut lines = vec![
        "# Architecture coverage manifest".to_string(),
        String::new(),
        "Generated from `ferrox_models::capability::architecture_catalog`.".to_string(),
        "Source of truth for names: pinned llama.cpp `LLM_ARCH_NAMES`.".to_string(),
        String::new(),
        "| GGUF arch | Scope | Family | Memory | Path |".to_string(),
        "|---|---|---|---|---|".to_string(),
    ];
    for p in architecture_catalog() {
        let path = match p.path {
            ArchPath::GenericGqa { .. } => "generic-gqa",
            ArchPath::TestFixture { .. } => "test-fixture",
            ArchPath::DedicatedOnly { .. } => "dedicated",
            ArchPath::Deferred { .. } => "deferred",
        };
        lines.push(format!(
            "| `{}` | {:?} | {:?} | {:?} | {} |",
            p.gguf_name, p.scope, p.family, p.memory, path
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    /// Every audited name must actually be on the generic path.
    ///
    /// A name here that resolves to a dedicated engine, or to nothing,
    /// is a stale entry claiming evidence for a path it does not use.
    #[test]
    fn every_audited_name_is_actually_on_the_generic_path() {
        for name in AUDITED_GENERIC_GQA {
            let profile = resolve_profile(name)
                .unwrap_or_else(|| panic!("audited arch `{name}` is not in the catalog"));
            assert!(
                matches!(profile.path, ArchPath::GenericGqa { .. }),
                "`{name}` is listed as an audited GENERIC-path arch but resolves to {:?}",
                profile.path
            );
        }
    }

    /// The five architectures that were caught computing the wrong
    /// thing must never appear here.
    ///
    /// They are refused outright now, but this pins the intent: the
    /// audited list is evidence of correctness, and these are the
    /// counter-examples that motivated it.
    #[test]
    fn the_architectures_that_were_wrong_are_not_claimed_as_audited() {
        for name in ["gpt2", "mpt", "refact", "bloom", "jais"] {
            assert!(
                !is_audited_generic(name),
                "`{name}` was found computing ALiBi or learned position embeddings as \
                 though it were RoPE; it cannot be on the audited list"
            );
        }
    }

    /// An architecture nobody has checked is not audited, which is the
    /// whole point of the inversion.
    #[test]
    fn an_unchecked_architecture_is_not_audited() {
        assert!(!is_audited_generic("smallthinker"));
        assert!(!is_audited_generic("mellum"));
        assert!(!is_audited_generic("an-arch-that-does-not-exist"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_mainstream_families_resolve() {
        assert_eq!(
            resolve_architecture("llama"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Norm
            })
        );
        assert_eq!(
            resolve_architecture("qwen2moe"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("mistral"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("yi"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("mixtral"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("phi3"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("phi4"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_profile("phi4").map(|p| p.family),
            Some(DecoderFamily::PhiFamily)
        );
        assert_eq!(
            resolve_architecture("gemma3"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        for arch in ["gemma4", "gemma4-assistant"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::DedicatedOnly { .. })
                ),
                "{arch} uses dedicated Gemma4 engine"
            );
            assert_eq!(
                resolve_profile(arch).map(|p| p.family),
                Some(DecoderFamily::GemmaFamily)
            );
        }
        assert!(matches!(
            resolve_architecture("gemma3n"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert_eq!(
            resolve_architecture("deepseek"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Norm
            })
        );
        assert_eq!(
            resolve_profile("qwen3").map(|p| p.qk_norm),
            Some(QkNormStyle::PerHead)
        );
    }

    #[test]
    fn deepseek2_is_dedicated_mla_not_generic() {
        assert!(matches!(
            resolve_architecture("deepseek2"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn unknown_architecture_is_none() {
        assert_eq!(resolve_architecture("totally-unknown-arch"), None);
        // t5 is registered as dedicated encoder-decoder stub
        assert!(matches!(
            resolve_architecture("t5"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn dedicated_paths_are_not_generic() {
        assert!(matches!(
            resolve_architecture("glm-dsa"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert!(matches!(
            resolve_architecture("deepseek4"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        for arch in ["minimax-m2", "minimax-m3"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::DedicatedOnly {
                        reason: "MiniMax 256-expert sigmoid MoE + MTP — see minimax_engine.rs"
                    })
                ),
                "{arch} must fail closed, not silent generic GQA"
            );
        }
        assert!(
            matches!(
                resolve_architecture("llama4"),
                Some(ArchPath::DedicatedOnly {
                    reason: "llama4 MoE + non-GQA attn — see llama4_engine.rs tensor list"
                })
            ),
            "llama4 must fail closed, not silent generic GQA"
        );
        assert!(matches!(
            resolve_architecture("glm4"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert!(matches!(
            resolve_architecture("glm4moe"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn test_fixtures_remain_loadable() {
        for arch in ["ferroxtest", "ferroxtestmoe", "ferroxtestmixed"] {
            assert!(matches!(
                resolve_architecture(arch),
                Some(ArchPath::TestFixture { .. })
            ));
        }
    }

    #[test]
    fn catalog_has_unique_names() {
        let mut seen = std::collections::HashSet::new();
        for p in architecture_catalog() {
            assert!(
                seen.insert(p.gguf_name),
                "duplicate arch name {}",
                p.gguf_name
            );
        }
    }

    #[test]
    fn gemma_family_does_not_fail_closed_on_softcap_keys() {
        assert!(unsupported_feature_keys("gemma3").is_empty());
        assert!(!unsupported_feature_keys("llama").is_empty());
    }

    /// Parallel attention+FFN residual is not a tensor and, for MiniCPM,
    /// not even a metadata key -- llama.cpp hardcodes MiniCPM's three
    /// multipliers. Neither the tensor-consumption gate nor
    /// `unsupported_scaling_keys` can see the difference, so these
    /// architectures must not be admitted to the generic decoder at all.
    #[test]
    fn architectures_with_a_different_residual_topology_are_refused() {
        for arch in [
            "command-r",
            "cohere2",
            "cohere2moe",
            "falcon",
            "gptneox",
            "phi2",
            "plamo",
            "minicpm",
        ] {
            match resolve_architecture(arch) {
                Some(ArchPath::DedicatedOnly { reason }) => {
                    assert!(!reason.is_empty(), "{arch} must say why");
                }
                other => panic!("{arch} must be refused, got {other:?}"),
            }
        }
        // The sequential-residual siblings stay on the generic path --
        // this is a named list, not a family-wide ban.
        //
        // `phimoe`, `starcoder2` and `nemotron` used to be checked here
        // too. They left the generic path for an unrelated reason (the
        // required bias tensors pinned by `tests/attn_bias.rs`), so
        // asserting them generic would now assert the wrong thing; what
        // still has to hold is that neither they nor the archs below are
        // refused for a *residual* reason they do not have.
        for arch in ["phi3", "plamo3", "qwen2", "llama"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::GenericGqa { .. })
                ),
                "{arch} must stay generic"
            );
        }
        for arch in ["phimoe", "starcoder2", "nemotron"] {
            match resolve_architecture(arch) {
                Some(ArchPath::DedicatedOnly { reason }) => assert!(
                    reason.contains("bias"),
                    "{arch} is refused for the wrong reason: {reason}"
                ),
                other => panic!("{arch} must be refused for its biases, got {other:?}"),
            }
        }
    }

    /// Every architecture appears exactly once, so a refusal added next
    /// to an existing entry cannot be shadowed by whichever the lookup
    /// happens to find first.
    #[test]
    fn no_architecture_is_listed_twice() {
        let mut seen = std::collections::HashSet::new();
        for p in architecture_catalog() {
            assert!(seen.insert(p.gguf_name), "{} listed twice", p.gguf_name);
        }
    }
}
