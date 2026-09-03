//! The five ONE-MATCH-ARM architectures, checked against llama.cpp
//! itself.
//!
//! `capability.rs` triaged 46 unaudited architectures into four classes.
//! Seven were ONE MATCH ARM: one small, nameable piece missing -- an
//! activation, a norm slot, a routing flag, an ordering. Five of those
//! seven are closed here. Each needed a different arm, and each fixture
//! is built so that getting THAT arm wrong is a large, obvious
//! divergence rather than a rounding difference:
//!
//! | arch | the arm | what would break |
//! |---|---|---|
//! | `deepseek` | top-k weights are not renormalised | every routed token |
//! | `bailingmoe` | `leading_dense_block_count` is inert | layer 0 fails to load |
//! | `seed_oss` | pre-FFN norm lives in `post_attention_norm` | a whole RMSNorm on the wrong side of a residual |
//! | `maincoder` | QK norm AFTER RoPE | every layer's attention scores |
//! | `hunyuan-moe` | QK norm AFTER RoPE | every layer's attention scores |
//!
//! **Where the numbers come from.** Each `GOLDEN` array below was
//! produced by running **llama.cpp's own graph** for that architecture
//! over the same fixture file, through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama`. Not by re-reading a spec, and not
//! by ferrox checking itself. That is the bar `AUDITED_GENERIC_GQA`
//! sets, and it is why these five moved off the refusal list rather
//! than merely having their verdicts reworded.
//!
//! Every fixture is a 2- or 3-layer synthetic checkpoint from
//! `scripts/make_<arch>_fixture.py`, fixed seed, byte-stable. The
//! scripts' docstrings carry the per-architecture reading of
//! `.scratch/llama.cpp/src/models/*.cpp` that each shape choice comes
//! from.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_seed_oss_fixture.py \
//!     crates/ferrox-models/tests/fixtures/seed_oss_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/seed_oss_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches as caches, graph_fixture_path as fixture,
    load_graph_fixture as load, worst_vs, GRAPH_PROMPT as PROMPT,
};
use ferrox_models::capability::QkNormStyle;
use ferrox_models::{Decoder, ModelConfig};

// --- deepseek (V1) -------------------------------------------------
//
// The arm: `src/models/deepseek.cpp:145-155` passes `norm_w = false` to
// `build_moe_ffn`, and `conversion/deepseek.py`'s `DeepseekModel` never
// writes `{arch}.expert_weights_norm` (only `DeepseekV2Model` does, at
// :354), so no real `deepseek` GGUF carries the key. ferrox has to get
// the answer from `NO_TOPK_RENORMALIZE_ARCHITECTURES`, and the fixture
// has no such key for exactly that reason.

const DEEPSEEK_GOLDEN: [f32; 48] = [
    0.048226774,
    0.2819079,
    -0.19458595,
    0.051145732,
    0.5227392,
    0.23523334,
    0.18219762,
    -0.5215794,
    0.100488976,
    -0.40474808,
    0.028783947,
    0.29588076,
    -0.18639557,
    -0.04201878,
    0.096413225,
    -0.13163471,
    -0.024807326,
    0.3165786,
    0.024499238,
    -0.035591282,
    0.008794859,
    -0.24559931,
    0.21297875,
    -0.2290324,
    -0.40852088,
    -0.17818648,
    -0.29124618,
    0.73868686,
    0.09183925,
    -0.05146555,
    -0.13013509,
    -0.12608205,
    -0.019688487,
    0.016693514,
    0.026398227,
    0.28899318,
    -0.39421725,
    0.036387824,
    0.15263395,
    0.0761507,
    -0.15402263,
    -0.064116284,
    0.15764138,
    -0.19088429,
    0.3309419,
    0.21663469,
    -0.5842512,
    0.23919708,
];

#[test]
fn deepseek_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("deepseek", &DEEPSEEK_GOLDEN);
}

/// What the loader decided, so a regression in a piece the logits alone
/// would not name still names itself.
#[test]
fn the_loader_reads_deepseeks_routing_and_its_leading_dense_layers() {
    let d = load("deepseek");
    // The arm. `false` here is NOT read from the file -- the fixture
    // carries no `expert_weights_norm` -- so this pins the
    // architecture-name fallback, which is the whole mechanism.
    assert!(
        !d.config.moe.norm_topk_prob,
        "deepseek must not renormalise the selected experts' weights"
    );
    // Leading dense IS honoured here, unlike `bailingmoe` below.
    assert!(d.config.layer_is_dense(0));
    assert!(!d.config.layer_is_dense(1));
    assert_eq!(d.config.moe.n_shared_experts, 2);
    assert_eq!(d.config.moe.expert_ffn_dim, 12);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Norm);
}

/// Renormalising the top-k weights is a visible error, not a rounding
/// one.
///
/// Without this the golden comparison would pass just as well with the
/// flag inverted for a fixture whose two selected experts happened to
/// have near-equal weights, and the arm would be untested.
#[test]
fn renormalising_deepseeks_top_k_weights_diverges_from_llama_cpp() {
    let path = fixture("deepseek");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.moe.norm_topk_prob = true;
    let d = Decoder::from_gguf(&path, config).expect("loads");
    let mut kv = caches(&d);
    let worst = worst_vs(&d.forward_batch_last(&PROMPT, 0, &mut kv), &DEEPSEEK_GOLDEN);
    assert!(
        worst > 1e-3,
        "renormalising changed the logits by only {worst}; the fixture cannot see this arm"
    );
}

// --- bailingmoe ----------------------------------------------------
//
// The arm: `src/models/bailingmoe.cpp:5` reads
// `LLM_KV_LEADING_DENSE_BLOCK_COUNT` and then nothing branches on it --
// :39-54 creates the expert and shared-expert tensors unconditionally
// for every layer, and the graph (:119-152) has no dense path. The
// fixture sets the key to 1 and ships no dense FFN on layer 0, so a
// decoder that honours the key dies on a missing tensor.

const BAILINGMOE_GOLDEN: [f32; 48] = [
    0.28660098,
    0.03384116,
    -0.24319153,
    0.089819975,
    -0.3759065,
    0.30608493,
    0.16745271,
    0.23978557,
    -0.04340898,
    0.20000786,
    0.082406804,
    0.014814936,
    -0.19221006,
    0.23547158,
    -0.029625641,
    -0.31056488,
    -0.37496945,
    0.0300856,
    -0.21707577,
    -0.012679767,
    -0.027634194,
    -0.36596966,
    0.040408455,
    0.18680914,
    -0.008012157,
    -0.09278015,
    0.1771502,
    0.5291099,
    -0.022168158,
    -0.08451706,
    -0.060037654,
    -0.12324926,
    0.34276655,
    0.5576784,
    -0.19961314,
    0.023494173,
    0.16179755,
    -0.1449597,
    -0.11124265,
    0.18491973,
    -0.123816594,
    0.031925693,
    -0.6112474,
    0.08633105,
    0.23084326,
    -0.12105487,
    -0.11914092,
    0.23888548,
];

#[test]
fn bailingmoe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("bailingmoe", &BAILINGMOE_GOLDEN);
}

#[test]
fn bailingmoe_ignores_the_leading_dense_key_its_file_carries() {
    let path = fixture("bailingmoe");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    // The key really is in the file -- otherwise this test proves
    // nothing about ignoring it.
    assert_eq!(
        file.metadata_u64("bailingmoe.leading_dense_block_count"),
        Some(1),
        "the fixture must carry the key it is ignoring"
    );
    let d = load("bailingmoe");
    assert!(
        !d.config.layer_is_dense(0),
        "layer 0 must be MoE despite leading_dense_block_count = 1"
    );
    // `expert_weights_norm` IS written by conversion/bailingmoe.py:31,
    // and the fixture sets it false -- the opposite of ferrox's
    // architecture-name default -- so this pins that the file wins.
    assert!(!d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Norm);
}

// --- seed_oss ------------------------------------------------------
//
// The arm: `src/models/seed-oss.cpp:36-37` creates `attn_norm` and
// `attn_post_norm` and no `ffn_norm`, and :113-115 norms `ffn_inp` --
// the post-attention residual -- with `attn_post_norm`. That is
// gpt-oss's slot, and it used to be reachable only through an
// `arch == "gpt-oss"` flag that also gated gpt-oss's attention sinks.

const SEED_OSS_GOLDEN: [f32; 48] = [
    0.4555686,
    0.08081946,
    -0.8877717,
    -0.008781537,
    0.016691634,
    -0.14418256,
    0.1414922,
    0.593925,
    0.28484803,
    -0.011401828,
    0.1034261,
    0.2122338,
    -0.26438826,
    -0.50978255,
    0.041276924,
    0.12567355,
    0.17167503,
    -0.3315112,
    -0.26611036,
    -0.0082461815,
    0.3068763,
    0.11299762,
    -0.4340123,
    -0.24792485,
    0.18786612,
    0.1469672,
    0.0567582,
    -0.2559087,
    0.3925688,
    0.43699247,
    -0.62407005,
    -0.16498223,
    0.049159832,
    0.4050939,
    0.041347966,
    -0.49512297,
    0.23467577,
    0.22438808,
    -0.14997265,
    0.5364686,
    0.4941529,
    -0.17830561,
    -0.08436005,
    0.09402853,
    1.8876046e-05,
    0.8314706,
    0.34555963,
    -0.27246982,
];

#[test]
fn seed_oss_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("seed_oss", &SEED_OSS_GOLDEN);
}

/// The norm slot was widened without widening gpt-oss's extra tensors
/// with it.
///
/// One flag standing for two facts is this repo's dominant bug shape.
/// If `seed_oss` had been admitted by widening `is_gpt_oss`, it would
/// have been handed attention sinks it does not have -- and the
/// attention-sink path also refuses Metal, so the mistake would have
/// been a silent performance cliff as well as wrong math.
#[test]
fn seed_oss_takes_the_norm_slot_without_taking_gpt_osss_other_tensors() {
    let d = load("seed_oss");
    assert!(
        d.gpt_oss.is_none(),
        "seed_oss has no attention sinks, no router bias and no SwiGLU clamp"
    );
    for (il, layer) in d.layers.iter().enumerate() {
        // `post_attention_norm` was consumed as the PRE-FFN norm, so
        // nothing is left in the Gemma post-attention slot.
        assert!(
            layer.attn.post_attn_norm.is_none(),
            "blk.{il}: post_attention_norm must be the pre-FFN norm here"
        );
        assert!(
            layer.moe.norm_weight.iter().any(|w| *w != 0.0),
            "blk.{il}: the pre-FFN norm must have been loaded from somewhere"
        );
    }
    // head_dim comes from `seed_oss.attention.key_length`; n_embd/n_head
    // would be 6.
    assert_eq!(d.config.head_dim, 8);
    assert_eq!(d.config.hidden_dim, 24);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Neox);
}

// --- maincoder and hunyuan-moe: QK norm after RoPE ------------------
//
// The arm: both rotate Q and K and only then norm them
// (`maincoder.cpp:78-95`, `hunyuan-moe.cpp:93-118`), where every
// previously-audited architecture norms first
// (`qwen3moe.cpp:99,108`). Both fixtures carry QK-norm weights centred
// near 1.5 rather than near 1.0, so the two orders are far apart.

const MAINCODER_GOLDEN: [f32; 48] = [
    -0.11138739,
    0.11824765,
    0.13920553,
    -0.30956966,
    0.13018379,
    0.10043175,
    0.3886379,
    -0.19205174,
    0.30031204,
    0.10036778,
    -0.10180198,
    0.09747762,
    -0.28050625,
    -0.04379492,
    -0.2667072,
    0.2032988,
    0.054047327,
    0.11231842,
    -0.32600948,
    -0.05464149,
    -0.13469744,
    0.0149400495,
    -0.20797788,
    -0.123298734,
    0.09001486,
    0.29425055,
    0.012470618,
    -0.61496115,
    0.38163024,
    0.034936484,
    -0.5258511,
    -0.106218845,
    0.012823265,
    0.079234354,
    0.11675485,
    -0.15945944,
    0.23372354,
    -0.053463854,
    0.36375925,
    -0.21172805,
    0.010290567,
    0.08845574,
    -0.13449258,
    -0.42340145,
    0.36418775,
    0.06322843,
    0.3355647,
    0.44385982,
];

const HUNYUAN_MOE_GOLDEN: [f32; 48] = [
    0.35964164,
    -0.03193312,
    0.08860654,
    0.07882613,
    -0.09411318,
    -0.25228012,
    0.10206525,
    0.0019554244,
    0.22584079,
    -0.18803002,
    0.25445765,
    0.24844477,
    -0.18049283,
    -0.004734317,
    0.07832132,
    0.04426696,
    -0.24596754,
    -0.012545568,
    -0.10501391,
    -0.33010307,
    0.11569242,
    0.02127038,
    -0.1883502,
    -0.014134302,
    0.25798446,
    -0.28575167,
    -0.39197293,
    -0.19562551,
    0.11669691,
    -0.18544069,
    0.41355348,
    0.084669836,
    -0.25159535,
    -0.14645867,
    -0.09086223,
    0.23901,
    0.1571095,
    -0.12348597,
    -0.036701947,
    -0.28670818,
    0.08595218,
    -0.08441264,
    -0.04356384,
    0.16307725,
    -0.07740431,
    -0.023180101,
    0.23391853,
    -0.14467773,
];

#[test]
fn maincoder_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("maincoder", &MAINCODER_GOLDEN);
}

#[test]
fn hunyuan_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("hunyuan_moe", &HUNYUAN_MOE_GOLDEN);
}

#[test]
fn both_post_rope_architectures_resolve_per_head_qk_norm_and_the_ordering_flag() {
    for (name, hidden, head_dim) in [("maincoder", 24, 8), ("hunyuan_moe", 24, 8)] {
        let d = load(name);
        assert!(
            d.qk_norm_after_rope,
            "{name} norms Q and K after RoPE and the loader must say so"
        );
        assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead, "{name}");
        assert_eq!(d.config.hidden_dim, hidden, "{name}");
        assert_eq!(d.config.head_dim, head_dim, "{name}");
        for (il, layer) in d.layers.iter().enumerate() {
            assert_eq!(
                layer.attn.q_norm.as_ref().map(Vec::len),
                Some(head_dim),
                "{name} blk.{il}: per-head Q norm"
            );
            assert_eq!(
                layer.attn.k_norm.as_ref().map(Vec::len),
                Some(head_dim),
                "{name} blk.{il}: per-head K norm"
            );
        }
    }
}

/// Norming on the wrong side of RoPE is a large divergence, on both
/// architectures.
///
/// This is the sabotage that makes the two golden comparisons above
/// mean something. Without it, a fixture whose QK-norm weights happened
/// to be near 1.0 would agree with llama.cpp under either order and the
/// arm would be untested.
#[test]
fn norming_before_rope_instead_of_after_diverges_from_llama_cpp() {
    for (name, golden) in [
        ("maincoder", &MAINCODER_GOLDEN),
        ("hunyuan_moe", &HUNYUAN_MOE_GOLDEN),
    ] {
        let mut d = load(name);
        assert!(d.qk_norm_after_rope);
        d.qk_norm_after_rope = false;
        let mut kv = caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: swapping the QK-norm order moved the logits by only {worst}; \
             the fixture cannot see this arm"
        );
    }
}
