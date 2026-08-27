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
            "starcoder",
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
            "smollm3",
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
            "olmoe", "qwen", "qwen2", "qwen2moe", "stablelm", "mistral",
            "mixtral", "olmo2", "bitnet", "jais2",
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
            "codeshell",
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
            "nemotron",
            "openelm",
            "orion",
            "plamo3",
            "seed_oss",
            "smallthinker",
            "starcoder2",
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
        for (n, fam) in [("phi3", PhiFamily), ("phimoe", PhiFamily)] {
            v.push(prof(
                n,
                TextGeneration,
                fam,
                KvGqa,
                Neox,
                ArchPath::GenericGqa { rope: Neox },
                WholeVector,
            ));
        }
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

/// The alternating sliding-window period an architecture uses when its
/// GGUF carries `{arch}.attention.sliding_window` but *not*
/// `{arch}.attention.sliding_window_pattern`.
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
/// Values transcribed from each arch's `load_arch_hparams`
/// (`src/models/*.cpp`); `None` means "no per-arch default", i.e. the
/// window applies uniformly when one is declared.
pub fn default_swa_pattern(arch: &str) -> Option<usize> {
    match arch {
        // src/models/openai-moe.cpp:10
        "gpt-oss" => Some(2),
        // src/models/gemma2.cpp:8
        "gemma2" => Some(2),
        // src/models/gemma3.cpp:7, gemma3n.cpp:6
        "gemma3" | "gemma3n" => Some(6),
        // src/models/cohere2.cpp:5, exaone4.cpp:7, olmo2.cpp:9
        "cohere2" | "exaone4" | "olmo2" => Some(4),
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
        for arch in ["phi3", "phimoe", "plamo3", "starcoder2", "nemotron"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::GenericGqa { .. })
                ),
                "{arch} must stay generic"
            );
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
