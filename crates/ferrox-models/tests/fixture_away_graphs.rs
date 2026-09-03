//! The FIXTURE-AWAY architectures, checked against llama.cpp itself.
//!
//! `capability.rs` triaged every unaudited refusal into four classes.
//! FIXTURE-AWAY is the class that says ferrox ALREADY builds this
//! architecture's graph and what is missing is EVIDENCE. That claim is
//! cheap to make and expensive to be wrong about, so this suite makes it
//! the same way `one_match_arm_graphs.rs` does: a tiny synthetic GGUF
//! whose golden values come from **llama.cpp's own graph** via a real
//! `libllama`, compared on all three forward paths.
//!
//! | arch | RoPE | what its fixture pins beyond the plain graph |
//! |---|---|---|
//! | `internlm2` | NORM | the optional q/k/v projection biases |
//! | `xverse` | NORM | nothing else -- it is llama under another name |
//! | `ernie4_5` | NORM | `head_dim != n_embd / n_head` |
//! | `baichuan` | NORM | 32 layers, because llama.cpp picks ALiBi off the layer count |
//! | `exaone` | NEOX | a tied lm_head, and `head_dim != n_embd / n_head` |
//!
//! **Every row was checked against the C on the six facts this repo has
//! lost at least once**, and every one of the six is asserted below
//! rather than merely read:
//!
//! 1. **RoPE variant.** `llama_model_rope_type` (llama-model.cpp) puts
//!    `internlm2` (:2579), `xverse` (:2581), `baichuan` (:2577) and
//!    `ernie4_5` (:2602) in the NORM group and `exaone` (:2655) in the
//!    NEOX group. `the_rope_variant_each_architecture_uses_is_the_one_llama_cpp_uses`
//!    pins the resolved layout and
//!    `rotating_the_wrong_pairs_diverges_from_llama_cpp` proves each
//!    fixture can see a flip. Getting this wrong is the Llama-3.1-8B
//!    wrong-output bug.
//! 2. **SWA pattern and phase.** None of the five reads
//!    `LLM_KV_ATTENTION_SLIDING_WINDOW` at all: `internlm2.cpp:3-11`,
//!    `xverse.cpp:3-12`, `baichuan.cpp:3-15`, `ernie4-5.cpp:3-21` and
//!    `exaone.cpp:3-10` are the complete `load_arch_hparams` bodies and
//!    none mentions a window, so `hparams.swa_type` stays
//!    `LLAMA_SWA_TYPE_NONE` and there is no phase to get wrong.
//! 3. **`attention_scale`.** All five pass a literal
//!    `1.0f/sqrtf(float(n_embd_head))` to `build_attn`
//!    (internlm2.cpp:92, xverse.cpp:90, baichuan.cpp:103,
//!    ernie4-5.cpp:120, exaone.cpp:93) and none reads
//!    `LLM_KV_ATTENTION_SCALE`, so `ModelConfig::attention_scale` must
//!    stay `None` and let the kernels' own scale stand.
//! 4. **`post_attn_norm`** and **5. `post_ffn_norm`.** None of the five
//!    creates `LLM_TENSOR_ATTN_POST_NORM` or `LLM_TENSOR_FFN_POST_NORM`;
//!    each layer has exactly `attn_norm` and `ffn_norm`, both applied
//!    BEFORE their branch on a sequential residual.
//! 6. **QK-norm and its order.** None of the five creates
//!    `attn_q_norm` / `attn_k_norm`, so there is no ordering question --
//!    unlike `maincoder` and `hunyuan-moe` next door, which norm after
//!    RoPE.
//!
//! None of the five is MoE, so the gating function and the top-k
//! renormalisation flag do not arise; `no_row_here_is_moe` pins that
//! rather than leaving it implied.
//!
//! **Where the numbers come from.** Each `GOLDEN` array was produced by
//! running llama.cpp's own graph for that architecture over the same
//! fixture file, through `scripts/gptoss_reference_logits.cpp` linked
//! against a real `libllama`. Not by re-reading a spec, and not by
//! ferrox checking itself.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_xverse_fixture.py \
//!     crates/ferrox-models/tests/fixtures/xverse_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/xverse_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use ferrox_models::{Decoder, ModelConfig, RopeLayout};

// --- internlm2 -----------------------------------------------------
//
// `src/models/internlm2.cpp` in full: `load_arch_hparams` (:3-11) reads
// only the RMS epsilon, `load_arch_tensors` (:13-35) creates attn_norm,
// split Q/K/V through `create_tensor_qkv`, attn_output, ffn_norm and
// gate/up/down, and the graph (:59-122) is `x + attn(rms(x))` then
// `y + ffn(rms(y))` with `LLM_FFN_SILU, LLM_FFN_PAR` SwiGLU. The
// fixture additionally carries the OPTIONAL q/k/v biases
// `create_tensor_qkv` allows (llama-model.cpp:2897-2899), which real
// InternLM2 exports ship and which ferrox loads generically.

const INTERNLM2_GOLDEN: [f32; 48] = [
    -0.38497463,
    -0.411406,
    -0.23690121,
    -0.079736516,
    -0.086446024,
    0.034069862,
    -0.18947236,
    0.31578812,
    -0.009138606,
    0.23619087,
    0.29375088,
    0.09236469,
    0.46713042,
    -0.38761902,
    0.8017329,
    -0.24398606,
    -0.7430083,
    -0.08430417,
    -0.11138179,
    0.09992017,
    0.2612124,
    -0.4456048,
    0.19681671,
    -0.09016396,
    0.044498637,
    -0.27157593,
    -0.62290406,
    -0.017267626,
    0.33660623,
    -0.19604379,
    0.28323877,
    -0.5600808,
    0.15880197,
    0.097796604,
    -0.31987217,
    -0.17884886,
    -0.11307311,
    -0.035523556,
    -0.37471962,
    -0.2398659,
    0.051130064,
    -0.28694645,
    0.17326327,
    -0.12214558,
    -0.06344788,
    -0.07479449,
    -0.2589051,
    0.2516519,
];

#[test]
fn internlm2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("internlm2", &INTERNLM2_GOLDEN);
}

/// The biases are in the file, are non-zero, and are loaded.
///
/// Without this the golden comparison would pass just as well against a
/// fixture that had none, and the claim that ferrox handles InternLM2's
/// real tensor set would rest on a file that did not have it.
#[test]
fn internlm2_loads_the_optional_qkv_biases_its_file_carries() {
    let d = load_graph_fixture("internlm2");
    for (il, layer) in d.layers.iter().enumerate() {
        for (what, bias) in [
            ("q", &layer.attn.q_bias),
            ("k", &layer.attn.k_bias),
            ("v", &layer.attn.v_bias),
        ] {
            let bias = bias
                .as_ref()
                .unwrap_or_else(|| panic!("blk.{il}: attn_{what}.bias must be loaded"));
            assert!(
                bias.iter().any(|b| *b != 0.0),
                "blk.{il}: attn_{what}.bias is all zeros; it could not fail"
            );
        }
    }
}

// --- xverse --------------------------------------------------------
//
// `src/models/xverse.cpp` is llama under a different name: :3-12 reads
// only the RMS epsilon, :14-35 is the plain tensor set, :59-121 is the
// sequential residual with SiLU SwiGLU. There is deliberately nothing
// extra in this fixture -- the row's whole content is that the graph is
// the generic one, and the evidence for that is the comparison itself.

const XVERSE_GOLDEN: [f32; 48] = [
    -0.06578407,
    0.35991532,
    0.09296507,
    0.0500691,
    0.013582745,
    0.50010747,
    -0.20849259,
    0.053211644,
    0.057681076,
    0.1302229,
    -0.409114,
    0.24452712,
    0.39229798,
    -0.24814555,
    -0.19971883,
    -0.10270727,
    0.36186606,
    0.24045944,
    0.1772546,
    0.38434425,
    0.116239354,
    -0.1490037,
    -0.16259417,
    0.07176717,
    0.27208632,
    -0.1671407,
    0.032589324,
    -0.13619582,
    0.006233629,
    0.29767945,
    0.16670844,
    -0.28568843,
    -0.07653904,
    0.18598174,
    -0.036290076,
    -0.3572907,
    -0.10366354,
    0.472095,
    -0.385141,
    0.06120664,
    0.05274259,
    -0.39883983,
    -0.22407724,
    0.3536656,
    0.004656989,
    -0.14421564,
    0.42171967,
    -0.18553367,
];

#[test]
fn xverse_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("xverse", &XVERSE_GOLDEN);
}

// --- ernie4_5 (dense) ----------------------------------------------
//
// `src/models/ernie4-5.cpp`'s dense branch (:64-68 for the tensors,
// :95-149 for the graph). Its Q projection and `attn_output` are sized
// from `n_embd_head_k * n_head` (:41-42) rather than from n_embd, so
// head_dim may disagree with `n_embd / n_head`, and this fixture makes
// it disagree (8 against 6).
//
// DELIBERATELY ABSENT from the fixture: the OPTIONAL `attn_output.bias`
// at :45. ferrox has no slot for one outside the gpt-oss path and
// refuses a checkpoint carrying it BY NAME through the unread-tensor
// gate, which is the correct behaviour and is why the fixture must not
// have one. `ernie4_5-moe` is a separate row and still refuses.

const ERNIE4_5_GOLDEN: [f32; 48] = [
    0.03883054,
    -0.22302249,
    -0.17906801,
    0.15218028,
    -0.02055931,
    -0.070127405,
    0.1165026,
    -0.034907892,
    -0.12018872,
    -0.048992664,
    0.016463999,
    0.08155338,
    -0.07699711,
    -0.12543437,
    0.28277662,
    0.38459194,
    0.13750306,
    -0.41823462,
    -0.043292582,
    -0.12194319,
    -0.09083214,
    0.071103305,
    0.01813967,
    0.2717319,
    0.031989187,
    -0.024477273,
    0.18046162,
    -0.09042182,
    -0.22470418,
    0.2138165,
    -0.22854619,
    0.008390563,
    -0.12621851,
    -0.4525338,
    0.47459865,
    -0.22544149,
    0.48321664,
    0.0048647877,
    -0.038895264,
    -0.11535422,
    -0.048806466,
    0.031679325,
    0.103515804,
    0.15141746,
    -0.08237043,
    0.10312017,
    -0.1163362,
    -0.32099646,
];

#[test]
fn ernie4_5_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("ernie4_5", &ERNIE4_5_GOLDEN);
}

/// head_dim comes from `attention.key_length`, not from n_embd/n_head.
#[test]
fn ernie4_5_reads_its_head_dim_from_the_file_rather_than_deriving_it() {
    let d = load_graph_fixture("ernie4_5");
    assert_eq!(d.config.hidden_dim, 24);
    assert_eq!(d.config.n_heads, 4);
    // 6 is what dividing would give.
    assert_eq!(d.config.head_dim, 8);
}

// --- baichuan (7B) -------------------------------------------------
//
// One architecture string, two models. `src/models/baichuan.cpp:5-14`
// picks the variant off the LAYER COUNT -- 32 is 7B, 40 is 13B and sets
// `f_max_alibi_bias = 8.0f` -- with its own comment "TODO: become GGUF
// KV parameter". The graph then builds positions only for 7B (:58) and
// applies `ggml_rope_ext` only for 7B (:77-95), so a 13B or an
// unrecognised layer count gets NO rotation at all.
//
// That is why this fixture has 32 layers and not 2: a 2-layer file is
// `LLM_TYPE_UNKNOWN`, falls into the no-RoPE arm at :91, and would be
// evidence about a graph no real checkpoint runs. The 13B stays refused
// by name in `loader.rs` on `block_count == 40`.

const BAICHUAN_GOLDEN: [f32; 48] = [
    0.056164384,
    0.05335981,
    0.034717236,
    -0.016637973,
    -0.1681825,
    -0.19657308,
    -0.093710594,
    0.12201875,
    0.202757,
    0.061601378,
    0.0852424,
    0.017296217,
    -0.13683479,
    0.019249436,
    -0.0067887288,
    0.030473292,
    -0.0056031533,
    0.019746449,
    0.040556442,
    -0.13934176,
    -0.00078091025,
    0.067291364,
    0.04678838,
    -0.04069045,
    -0.07528572,
    0.09840066,
    0.027413199,
    -0.012988582,
    -0.0071283206,
    0.10830741,
    0.046693385,
    0.23345774,
    0.10834331,
    -0.017043814,
    0.04738696,
    -0.09828674,
    -0.018817,
    0.037317395,
    0.10740893,
    -0.086651094,
    0.16450927,
    0.11803734,
    0.07888082,
    0.051779855,
    -0.087531194,
    -0.07406292,
    0.008599423,
    0.0709973,
];

#[test]
fn baichuan_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("baichuan", &BAICHUAN_GOLDEN);
}

/// The fixture is the 7B, and the file says so the only way llama.cpp
/// reads it.
///
/// A regenerated fixture with fewer layers would still load and still
/// compare -- against a reference that had silently stopped rotating.
/// This is the assertion that makes that impossible to miss.
#[test]
fn the_baichuan_fixture_has_the_32_layers_that_select_the_rotating_variant() {
    let d = load_graph_fixture("baichuan");
    assert_eq!(
        d.config.n_layers, 32,
        "baichuan.cpp:5-14 reads the variant off this number; 32 is the 7B, which is the \
         only one that RoPEs"
    );
}

// --- exaone (3.x) --------------------------------------------------
//
// `src/models/exaone.cpp`: :3-10 reads only the RMS epsilon, :12-40 is
// the plain tensor set sized from `n_embd_head_k * n_head`, :65-121 is
// the sequential residual with SiLU SwiGLU. Two things this fixture
// pins that the NORM rows above do not:
//
//   * NEOX RoPE, the opposite of the other four.
//   * A TIED lm_head: `output` is TENSOR_NOT_REQUIRED (:19) and falls
//     back to `token_embd` (:22-24), so the file ships no
//     `output.weight`.
//
// This is EXAONE 3.x. `exaone4` has no pre-attention and no pre-FFN
// norm, and `exaone-moe` skips RoPE entirely on its full-attention
// layers; both are different graphs and both still refuse.

const EXAONE_GOLDEN: [f32; 48] = [
    -0.2249429,
    0.34571093,
    0.028062485,
    -0.3856704,
    0.08937423,
    0.18980339,
    -0.052521326,
    0.014981142,
    0.036303222,
    -0.11778174,
    0.24545757,
    -0.19239902,
    0.14795516,
    0.0996446,
    0.0015762504,
    -0.054319553,
    0.40219936,
    -0.35769135,
    0.22361766,
    0.07902548,
    0.18331085,
    0.21348247,
    0.25453204,
    -0.16258636,
    0.42788702,
    0.2037906,
    -0.018451544,
    0.5553448,
    0.26972133,
    0.13789096,
    -0.30589244,
    0.06253724,
    -0.023459988,
    0.16139778,
    0.26396036,
    0.2515811,
    0.53262925,
    0.1445617,
    -0.4598862,
    0.12606835,
    -0.1691545,
    0.17235437,
    -0.09227791,
    -0.11130126,
    0.046939783,
    -0.2966211,
    -0.31041434,
    -0.3674787,
];

#[test]
fn exaone_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("exaone", &EXAONE_GOLDEN);
}

/// The fixture really is the tied case.
#[test]
fn the_exaone_fixture_ships_no_output_weight_and_ties_the_lm_head() {
    let path = graph_fixture_path("exaone");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    assert!(
        file.find_tensor("output.weight").is_none(),
        "the fixture must be the tied case or it pins nothing about exaone.cpp:22-24"
    );
    // It still produced output above, so the lm_head came from
    // `token_embd.weight`.
    assert!(file.find_tensor("token_embd.weight").is_some());
}

// --- the six facts, asserted rather than assumed --------------------

/// Fact 1: the RoPE variant each row resolves to is the group llama.cpp
/// puts it in.
#[test]
fn the_rope_variant_each_architecture_uses_is_the_one_llama_cpp_uses() {
    for (name, want) in [
        // llama-model.cpp `llama_model_rope_type`, NORM group.
        ("internlm2", RopeLayout::Norm),
        ("xverse", RopeLayout::Norm),
        ("ernie4_5", RopeLayout::Norm),
        ("baichuan", RopeLayout::Norm),
        // ... and the NEOX group.
        ("exaone", RopeLayout::Neox),
    ] {
        assert_eq!(
            load_graph_fixture(name).config.rope_layout,
            want,
            "{name}: rope layout"
        );
    }
}

/// Fact 1, the half that makes fact 1 worth asserting: each fixture can
/// SEE a flipped RoPE variant.
///
/// Rotating the wrong pairs of every Q/K head is the defect that
/// produced the Llama-3.1-8B wrong-output bug, and a fixture whose
/// positions happened not to matter would agree under either variant.
#[test]
fn rotating_the_wrong_pairs_diverges_from_llama_cpp() {
    for (name, golden) in [
        ("internlm2", &INTERNLM2_GOLDEN),
        ("xverse", &XVERSE_GOLDEN),
        ("ernie4_5", &ERNIE4_5_GOLDEN),
        ("baichuan", &BAICHUAN_GOLDEN),
        ("exaone", &EXAONE_GOLDEN),
    ] {
        let path = graph_fixture_path(name);
        let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
        let mut config = ModelConfig::from_gguf(&file).expect("parses");
        config.rope_layout = match config.rope_layout {
            RopeLayout::Norm => RopeLayout::Neox,
            _ => RopeLayout::Norm,
        };
        let d = Decoder::from_gguf(&path, config).expect("loads");
        let mut kv = graph_caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: flipping the RoPE variant moved the output by only {worst}; \
             the fixture cannot see this"
        );
    }
}

/// Facts 2, 3, 4, 5 and 6: no sliding window, no attention-scale
/// override, no post-norms and no QK-norm, on any layer of any of the
/// five.
///
/// Each of these has been lost at least once here by a copied decode
/// path, and each is absent from all five llama.cpp graphs, so the
/// assertion is that ferrox resolved them to absent rather than to
/// something plausible.
#[test]
fn none_of_these_rows_has_a_scale_override_a_post_norm_or_a_qk_norm() {
    for name in ["internlm2", "xverse", "ernie4_5", "baichuan", "exaone"] {
        let d = load_graph_fixture(name);
        // Fact 3: all five pass a literal 1/sqrt(head_dim) to
        // build_attn and read no LLM_KV_ATTENTION_SCALE, so the
        // kernels' own scale must stand.
        assert!(
            d.config.attention_scale.is_none(),
            "{name}: attention_scale must stay unset"
        );
        // Fact 2: none reads LLM_KV_ATTENTION_SLIDING_WINDOW.
        assert!(d.config.sliding_window.is_none(), "{name}: sliding window");
        for (il, layer) in d.layers.iter().enumerate() {
            // Facts 4 and 5.
            assert!(
                layer.attn.post_attn_norm.is_none(),
                "{name} blk.{il}: no LLM_TENSOR_ATTN_POST_NORM upstream"
            );
            assert!(
                layer.attn.post_ffn_norm.is_none(),
                "{name} blk.{il}: no LLM_TENSOR_FFN_POST_NORM upstream"
            );
            // Fact 6.
            assert!(
                layer.attn.q_norm.is_none() && layer.attn.k_norm.is_none(),
                "{name} blk.{il}: no attn_q_norm/attn_k_norm upstream, so no ordering \
                 question either"
            );
        }
    }
}

/// The MoE half of the checklist does not arise, and that is a fact
/// about the files rather than an omission.
///
/// Stated as an assertion because "we did not check the gating function"
/// and "there is no gating function" read identically in a commit
/// message.
#[test]
fn no_row_here_is_moe() {
    for name in ["internlm2", "xverse", "ernie4_5", "baichuan", "exaone"] {
        let d = load_graph_fixture(name);
        assert_eq!(
            d.config.moe.n_experts, 1,
            "{name}: dense, so gating and top-k renormalisation do not arise"
        );
        assert_eq!(d.config.moe.n_shared_experts, 0, "{name}");
    }
}
