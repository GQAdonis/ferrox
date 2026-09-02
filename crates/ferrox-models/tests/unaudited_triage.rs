//! Triage of the `LoadError::UnauditedArchitecture` refusals.
//!
//! ferrox's generic GQA path is opt-in: an architecture that is not in
//! `capability::AUDITED_GENERIC_GQA` refuses rather than running on the
//! guess that it is plain GQA. That closed the "loads and computes
//! something else" class -- `gpt2`, `mpt`, `refact`, `bloom` and `jais`
//! all did exactly that -- but it left **47** architectures refusing with
//! one identical paragraph whose only content is "nobody has checked
//! this".
//!
//! That paragraph is useless for the decision a user actually has, which
//! is whether their model is one fixture away or needs an attention
//! implementation. `docs/plans/llama-cpp-gap-inventory.md` §1.3 shows
//! the 47 split at least three ways, and this suite pins the split for
//! the architectures that have been read on **both** sides:
//!
//! - **fixture-away** -- everything is implemented; evidence is missing.
//! - **one match arm** -- one small, nameable piece: an activation, a
//!   norm slot, a routing flag, an ordering.
//! - **new code** -- a different attention or residual structure.
//! - **UNKNOWN** -- reading did not settle it; the verdict says what
//!   would.
//!
//! **Why this suite exists rather than only the unit tests in
//! `capability.rs`:** the failure mode here is not a compile error, it
//! is a *confident wrong verdict*. This repo has now found four
//! architectures whose refusal named something that was not the real
//! blocker -- `glm4moe` was told it lacked an MLA hyper-parameter it must
//! not have, and `minimax-m2` was blamed on MTP weights no converter can
//! emit. Each assertion below therefore pins a specific claim about a
//! specific llama.cpp line, so that changing the verdict without
//! changing the reading fails.
//!
//! Nothing here loads a checkpoint. Every claim it pins was read in
//! `.scratch/llama.cpp/src/models/*.cpp` against ferrox's generic
//! decoder, and the citation is in the verdict string itself.

use ferrox_models::capability::{
    architecture_catalog, is_audited_generic, unaudited_refusal_detail, unaudited_triage, ArchPath,
    TriageClass, TRIAGE_PENDING,
};

/// Every architecture that reaches the unaudited refusal gets a
/// non-empty detail line -- triaged or not.
///
/// A blank detail would be the old refusal wearing a new field.
#[test]
fn every_unaudited_architecture_renders_a_detail_line() {
    let mut n = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        n += 1;
        let detail = unaudited_refusal_detail(p.gguf_name);
        assert!(
            detail.starts_with("TRIAGE"),
            "`{}` renders {detail:?}",
            p.gguf_name
        );
        assert!(detail.len() > 100, "`{}` renders {detail:?}", p.gguf_name);
    }
    assert_eq!(
        n, 41,
        "the unaudited count moved. It was 47 until the triage itself found `minicpm3` was \
         an MLA model sitting on the generic-GQA row and it was reclassified to \
         DedicatedOnly, and 46 until `deepseek`, `bailingmoe`, `seed_oss`, `maincoder` and \
         `hunyuan-moe` were admitted with libllama-golden fixtures -- five ONE MATCH ARM rows \
         closing is the count going DOWN for the best reason. Either an architecture was \
         audited or reclassified (good -- update the count and the docs) or one was added \
         (check it was triaged)"
    );
}

/// Batch 1: the architectures people actually download, with the class
/// each was placed in and the llama.cpp fact that decides it.
///
/// The class is asserted together with a substring of the blocker on
/// purpose. Asserting the class alone would let somebody flip a verdict
/// and leave the (now-contradictory) reasoning in place, which is
/// exactly how `glm4moe` came to refuse for a reason it did not have.
#[test]
fn batch_one_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // --- fixture-away: implemented, unevidenced ------------------
        //
        // gemma.cpp:16-33 creates only the tensors the generic decoder
        // loads; the three Gemma-specific pieces (sqrt(n_embd)
        // embedding scale, GeGLU, 1/sqrt(head_dim) attention scale) are
        // all implemented for GemmaFamily.
        ("gemma", TriageClass::FixtureAway, "gemma.cpp:16-33"),
        // internlm2.cpp:25-33 is the plain Llama tensor set.
        ("internlm2", TriageClass::FixtureAway, "internlm2.cpp:25-33"),
        // exaone.cpp:29-38 likewise, plus the global rope_freqs.weight
        // loader.rs already loads. NOT exaone4, which is a different
        // graph and a different class below.
        ("exaone", TriageClass::FixtureAway, "exaone.cpp:29-38"),
        // ernie4-5.cpp's dense branch; the only unslotted tensor is an
        // OPTIONAL attn_output.bias, which fails closed by name.
        (
            "ernie4_5",
            TriageClass::FixtureAway,
            "attn_output.bias at :45",
        ),
        // bailingmoe2.cpp: fused QKV that ferrox splits, per-head QK
        // norm BEFORE RoPE, sigmoid/softmax read from metadata,
        // exp_probs_b, shared experts, leading dense -- all implemented.
        (
            "bailingmoe2",
            TriageClass::FixtureAway,
            "bailingmoe2.cpp is plain GQA",
        ),
        // --- one match arm: small and nameable -----------------------
        //
        // `seed_oss`, `deepseek` and `hunyuan-moe` were HERE. All three
        // arms landed, with libllama-golden fixtures
        // (`tests/one_match_arm_graphs.rs`), so they are audited now and
        // carry no verdict at all —
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale verdict outliving its refusal.
        //
        // ernie4-5-moe.cpp:64 -- MoE layers are interleaved by
        // n_moe_layer_step, which `layer_is_dense` does not implement.
        (
            "ernie4_5-moe",
            TriageClass::OneMatchArm,
            "interleave_moe_layer_step",
        ),
        // --- new code: a different graph -----------------------------
        //
        // olmo2.cpp:47,52,92,169 -- no attn_norm and no ffn_norm at all;
        // Q/K/V come off the raw residual.
        ("olmo2", TriageClass::NewCode, "olmo2.cpp:47,52"),
        // exaone4.cpp:60-67,118,159 -- the same post-norm-only topology.
        ("exaone4", TriageClass::NewCode, "exaone4.cpp:60-67"),
        // granite.cpp:7-10,188,241-242,301-302 -- four multipliers the
        // generic decoder does not apply, plus a rope_finetuned gate.
        ("granite", TriageClass::NewCode, "f_residual_scale"),
        ("granitemoe", TriageClass::NewCode, "f_residual_scale"),
        ("granite-moe", TriageClass::NewCode, "f_residual_scale"),
        // --- unknown: say so, and say what would settle it -----------
        //
        // `phi4` is not a llama.cpp architecture at all, so there is no
        // graph to diff against.
        ("phi4", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
    ];

    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The two olmo2-shaped architectures are NOT fixture-away, and this
/// test exists because the inventory said they might be.
///
/// `docs/plans/llama-cpp-gap-inventory.md` §1.3 groups `olmo2`,
/// `seed_oss` and `exaone4` together as "likely a fixture away, if the
/// loader wires the post-norm slots for non-Gemma families". The wiring
/// question has a yes answer -- `loader.rs` reads
/// `blk.N.post_attention_norm.weight` and `blk.N.post_ffw_norm.weight`
/// for every non-gpt-oss architecture, and the decoder applies them in
/// llama.cpp's own places -- but it is the wrong question. `olmo2` and
/// `exaone4` create **no** `attn_norm` and **no** `ffn_norm`
/// (`olmo2.cpp:42-52`, `exaone4.cpp:53-67`) and project Q/K/V straight
/// off the residual (`olmo2.cpp:92`, `exaone4.cpp:118`), while the
/// generic decoder requires both pre-norms and applies them on every
/// layer. And `seed_oss` is a third thing again: it *has* `attn_norm`
/// and uses `attn_post_norm` as its pre-FFN norm.
///
/// So the three do not share a class. Pinning that stops the grouping
/// from being restored from the prose -- and the split is now settled the
/// hard way: `seed_oss` RUNS, checked against llama.cpp's own logits,
/// while `olmo2` and `exaone4` still refuse.
#[test]
fn the_post_norm_group_does_not_share_one_class() {
    let olmo2 = unaudited_triage("olmo2").expect("olmo2 verdict");
    let exaone4 = unaudited_triage("exaone4").expect("exaone4 verdict");

    assert_eq!(olmo2.class, TriageClass::NewCode);
    assert_eq!(exaone4.class, TriageClass::NewCode);
    assert!(
        is_audited_generic("seed_oss") && unaudited_triage("seed_oss").is_none(),
        "seed_oss has attn_norm and olmo2 does not; they cannot share a class"
    );
    for (name, t) in [("olmo2", olmo2), ("exaone4", exaone4)] {
        assert!(
            t.blocker.contains("NO attn_norm") || t.blocker.contains("no attn_norm"),
            "`{name}`'s verdict must name the missing pre-norm, not the post-norms: {}",
            t.blocker
        );
        assert!(
            !t.blocker.contains("fixture away"),
            "`{name}` is not a fixture away"
        );
    }
}

/// `ernie4_5-moe` is softmax-routed, not sigmoid-routed.
///
/// The inventory (§1.3) lists it beside `bailingmoe2` as "sigmoid-routed
/// MoE with `ffn_exp_probs_b` router bias". `bailingmoe2` reads its
/// gating function from metadata (`bailingmoe2.cpp:11`) so it can be
/// either; `ernie4-5-moe.cpp:90` **hardcodes**
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`. Carrying that error into a
/// refusal would send whoever picks the work up looking at the wrong
/// routing path.
#[test]
fn ernie_moe_is_not_described_as_sigmoid_routed() {
    let t = unaudited_triage("ernie4_5-moe").expect("verdict");
    assert!(
        t.blocker.contains("NOT sigmoid-routed"),
        "ernie4_5-moe's verdict must correct the sigmoid claim: {}",
        t.blocker
    );
    assert!(
        t.blocker.contains("SOFTMAX"),
        "and name what it really is: {}",
        t.blocker
    );
}

/// Every triaged architecture really does reach the unaudited refusal.
///
/// A verdict on a row that refuses for a *different*, named reason would
/// be dead text the user never sees -- the same shape as the `glm4moe`
/// bug, where the reason shown and the reason true were two different
/// strings.
#[test]
fn every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited() {
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            matches!(p.path, ArchPath::GenericGqa { .. }),
            "`{}` carries a {:?} verdict but resolves to {:?}, which refuses elsewhere",
            p.gguf_name,
            t.class,
            p.path
        );
        assert!(
            !is_audited_generic(p.gguf_name),
            "`{}` is audited and runs; a triage verdict there is never rendered",
            p.gguf_name
        );
    }
}

/// The pending list is a to-do with a shrinking count, not a parking
/// bay.
///
/// If this number goes UP without the total moving, an architecture lost
/// its verdict.
#[test]
fn the_remaining_work_is_counted() {
    assert_eq!(
        TRIAGE_PENDING.len(),
        0,
        "all 47 unaudited architectures are triaged; a name reappearing here means a new \
         architecture reached the generic path without being read"
    );
    let triaged = architecture_catalog()
        .iter()
        .filter(|p| p.triage.is_some())
        .count();
    assert_eq!(triaged + TRIAGE_PENDING.len(), 41);
}

/// `minicpm3` is refused as an MLA model, not as an unaudited one.
///
/// The triage found the catalog claimed `StandardGqa`/`KvGqa` for a
/// model whose every checkpoint carries `attn_q_a`/`attn_kv_a_mqa` and
/// no `attn_q.weight` (`src/models/minicpm3.cpp:41-46`), so the generic
/// path could never have loaded one. Being told "unaudited" for that is
/// telling the user the wrong thing about their model.
///
/// A message-quality fix rather than a correctness one -- the old
/// failure was already a clean missing-tensor error -- which is why the
/// reason has to name BOTH blockers, the MLA tensor set and the
/// hardcoded MiniCPM multipliers.
#[test]
fn minicpm3_is_refused_as_mla_not_as_unaudited() {
    assert!(
        unaudited_triage("minicpm3").is_none(),
        "minicpm3 left the unaudited generic set"
    );
    match ferrox_models::capability::resolve_architecture("minicpm3") {
        Some(ArchPath::DedicatedOnly { reason }) => {
            assert!(reason.contains("MLA"), "{reason}");
            assert!(
                reason.contains("scale_depth"),
                "the multipliers are the second blocker and must be named: {reason}"
            );
        }
        other => panic!("minicpm3 must be DedicatedOnly, got {other:?}"),
    }
}

/// The untriaged message still works, and still claims no class.
///
/// `TRIAGE_PENDING` is empty now that all 47 are read, so this exercises
/// the branch through a name the catalog does not carry. It is what a
/// NEW architecture added to the generic path would render until
/// somebody reads it, and it must not imply a class: "unaudited" and
/// "untriaged" are different claims, and reading a class into the
/// untriaged message is how a guess becomes a citation.
#[test]
fn the_untriaged_message_claims_no_class() {
    let d = unaudited_refusal_detail("an-architecture-nobody-has-read-yet");
    assert!(d.contains("not done for"), "{d}");
    for label in [
        TriageClass::FixtureAway.label(),
        TriageClass::OneMatchArm.label(),
        TriageClass::NewCode.label(),
    ] {
        assert!(
            !d.contains(label),
            "the untriaged message implies {label}: {d}"
        );
    }
}

/// Batch 2: the next six by download volume, plus the two the
/// activation audit turned up on the way.
///
/// Every one came out `NewCode`, which is itself the finding. Batch 1
/// mixed five fixture-away rows in; past the first dozen the unaudited
/// set is genuinely harder, and the refusal now says so per
/// architecture instead of implying a uniform distance.
#[test]
fn batch_two_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        ("grok", TriageClass::NewCode, "grok.cpp:5-21"),
        ("dbrx", TriageClass::NewCode, "LayerNorm, not RMSNorm"),
        (
            "smallthinker",
            TriageClass::NewCode,
            "router reads a DIFFERENT tensor",
        ),
        ("bitnet", TriageClass::NewCode, "attn_sub_norm"),
        // `minicpm3` was HERE, and the triage that produced this list
        // is what removed it: reading `minicpm3.cpp:5-6,41-46` showed an
        // MLA tensor set on a row the catalog called `StandardGqa`, so
        // it moved to `DedicatedOnly` rather than staying an unaudited
        // generic architecture. Its refusal is now asserted by
        // `minicpm3_is_refused_as_mla_not_as_unaudited` below.
        ("openelm", TriageClass::NewCode, "per-LAYER head counts"),
        ("arcee", TriageClass::NewCode, "UNGATED ReLU-squared"),
        ("plm", TriageClass::NewCode, "UNGATED ReLU-squared"),
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// `smallthinker`'s blocker leads with the routing input, not with the
/// activation.
///
/// It has three separate blockers and they are not equally severe. The
/// ReLU experts are one match arm on their own; the router reading
/// `inpL` instead of the normed FFN input, and the unkeyed NoPE layers,
/// are both "computes something else" and neither leaves a tensor or a
/// metadata key behind. A verdict that named only the activation would
/// read as one match arm and be wrong by two.
#[test]
fn smallthinker_names_the_routing_input_and_the_nope_layers() {
    let t = unaudited_triage("smallthinker").expect("verdict");
    assert_eq!(t.class, TriageClass::NewCode);
    for claim in ["inpL", "n_no_rope_layer_step", "smollm3", "LLM_FFN_RELU"] {
        assert!(
            t.blocker.contains(claim),
            "smallthinker's verdict drops {claim:?}: {}",
            t.blocker
        );
    }
    let routing = t.blocker.find("inpL").expect("inpL");
    let relu = t.blocker.find("LLM_FFN_RELU").expect("relu");
    assert!(
        routing < relu,
        "the severe blocker must lead: {}",
        t.blocker
    );
}

/// `dbrx` is refused for its normalisation, and the verdict says why the
/// existing bias-tensor refusal group does NOT cover it.
///
/// The group keys on required `*_norm.bias` tensors as the marker of a
/// real LayerNorm. `dbrx` creates none of them and is still LayerNorm,
/// because llama.cpp's `LLM_NORM` subtracts the mean with or without a
/// bias. Somebody reading only that group's comment would conclude dbrx
/// is fine.
#[test]
fn dbrx_says_why_the_bias_group_does_not_catch_it() {
    let t = unaudited_triage("dbrx").expect("verdict");
    assert!(
        t.blocker.contains("creates no norm bias tensors"),
        "dbrx's verdict must say the bias marker is absent: {}",
        t.blocker
    );
}

/// Where the refusal a user actually sees is NOT this one, the verdict
/// says so rather than letting the reader assume the triage line is what
/// they got.
///
/// `granite` dies on `capability::unsupported_scaling_keys` and
/// `openelm` on a missing-hparam error for keys its file does carry,
/// both before the unaudited gate. A verdict that stayed silent about
/// that would send someone looking for a message they will never see.
#[test]
fn verdicts_disclose_when_an_earlier_refusal_fires_first() {
    for (arch, marker) in [
        ("granite", "never reaches THIS message"),
        ("openelm", "before the unaudited gate is reached"),
    ] {
        let t = unaudited_triage(arch).expect("verdict");
        assert!(
            t.blocker.contains(marker),
            "`{arch}` must disclose that an earlier refusal fires first: {}",
            t.blocker
        );
    }
}

/// Batch 3: the alias rows and the plain long-tail.
///
/// This is the batch that was expected to be cheap, and half of it was.
/// `xverse`, `baichuan` and `chatglm` really are llama-shaped and say so
/// plainly. `deci` and `olmo` are not, and neither is the alias trio,
/// for a reason that has nothing to do with their graphs.
#[test]
fn batch_three_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        (
            "xverse",
            TriageClass::FixtureAway,
            "llama under a different name",
        ),
        ("baichuan", TriageClass::FixtureAway, "for the 7B"),
        (
            "chatglm",
            TriageClass::FixtureAway,
            "audited phi3 path exactly",
        ),
        ("deci", TriageClass::NewCode, "PER LAYER"),
        ("olmo", TriageClass::NewCode, "NO norm weights at all"),
        ("mistral", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
        ("mixtral", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
        ("yi", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The three alias rows are UNKNOWN, and the verdict names the RoPE
/// hazard rather than calling them llama-shaped.
///
/// Marking them fixture-away would be the cheap answer and the wrong
/// one. `llama` -- the string real Mistral, Mixtral and Yi checkpoints
/// actually ship under -- is in `llama_model_rope_type`'s NORM group,
/// while these three rows are NEOX. A file spelling `mistral` would be
/// rotated on the wrong pairs of every Q/K head, the defect behind the
/// Llama-3.1-8B wrong-logits bug.
#[test]
fn the_alias_rows_name_the_rope_hazard_rather_than_claiming_llama_shape() {
    for arch in ["mistral", "mixtral", "yi"] {
        let t = unaudited_triage(arch).expect("verdict");
        assert_eq!(t.class, TriageClass::Unknown, "`{arch}`");
        for claim in ["NEOX", "NORM group", "LLM_ARCH_NAMES"] {
            assert!(
                t.blocker.contains(claim),
                "`{arch}`'s verdict drops {claim:?}: {}",
                t.blocker
            );
        }
    }
}

/// `baichuan` is one architecture string covering two different models,
/// and the verdict says which one the refusal is about.
#[test]
fn baichuan_says_which_of_its_two_models_the_verdict_covers() {
    let t = unaudited_triage("baichuan").expect("verdict");
    for claim in ["13B is a DIFFERENT model", "f_max_alibi_bias", "32-layer"] {
        assert!(
            t.blocker.contains(claim),
            "baichuan's verdict drops {claim:?}: {}",
            t.blocker
        );
    }
}

/// Batch 4 and batch 5: the remaining long tail.
#[test]
fn batches_four_and_five_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // Batch 4. `maincoder` and `bailingmoe` were here and are now
        // audited; see `tests/one_match_arm_graphs.rs`.
        ("arctic", TriageClass::NewCode, "PARALLEL dense+MoE"),
        (
            "mistral3",
            TriageClass::NewCode,
            "attention temperature tuning",
        ),
        (
            "nanbeige",
            TriageClass::NewCode,
            "RUNS THE SAME PHYSICAL LAYERS MORE THAN ONCE",
        ),
        (
            "mellum",
            TriageClass::NewCode,
            "two per-layer RoPE variants",
        ),
        ("talkie", TriageClass::NewCode, "NO norm weights"),
        ("mimo2", TriageClass::NewCode, "attention sinks"),
        // Batch 5.
        ("plamo3", TriageClass::FixtureAway, "slot for slot"),
        ("afmoe", TriageClass::NewCode, "gated attention"),
        ("apertus", TriageClass::NewCode, "xIELU"),
        (
            "exaone-moe",
            TriageClass::NewCode,
            "GLOBAL layers get no RoPE",
        ),
        ("grovemoe", TriageClass::NewCode, "SECOND bank of experts"),
        (
            "hunyuan-dense",
            TriageClass::OneMatchArm,
            "NTK-alpha RoPE base rescale",
        ),
        ("laguna", TriageClass::NewCode, "second rotary width"),
        ("step35", TriageClass::NewCode, "per-LAYER rotary width"),
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The QK-norm ordering arm was wanted by three architectures; two of
/// them now run on it and the third must NOT still be refused for it.
///
/// The cross-row check was the argument for adding a shared flag rather
/// than special-casing one architecture. Now that
/// `Decoder::qk_norm_after_rope` exists, the same check has to run the
/// other way: `hunyuan-dense` still refuses, and if its verdict still
/// led with an ordering that is implemented it would send whoever picks
/// the work up to a solved problem. That is the `glm4moe` defect exactly
/// -- the reason shown and the reason true being two different strings.
#[test]
fn the_qk_norm_ordering_arm_is_no_longer_anybody_s_leading_blocker() {
    for arch in ["hunyuan-moe", "maincoder"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` got the ordering arm and evidence; it must not still be refused"
        );
    }
    let dense = unaudited_triage("hunyuan-dense").expect("hunyuan-dense still refuses");
    assert!(
        dense.blocker.starts_with("the NTK-alpha RoPE base rescale"),
        "hunyuan-dense must lead with what is actually left: {}",
        dense.blocker
    );
    assert!(
        dense.blocker.contains("that ordering is implemented"),
        "and must say the ordering half landed: {}",
        dense.blocker
    );
}

/// `exaone-moe`'s hardcoded `n_swa = 128` was checked and is NOT a
/// divergence, and the verdict records that.
///
/// A cross-cutting sweep of every `hparams.n_swa =` in llama.cpp turned
/// this up as a candidate for the `deepseek` shape: a per-architecture
/// default with no key to correct it. It is not one, because
/// `exaone-moe.cpp:13` reads the window as a REQUIRED key. A clean
/// result is still a result, and recording it stops the next person
/// re-running the same sweep and re-raising the same false alarm.
#[test]
fn exaone_moe_records_the_swa_sweep_as_clean() {
    let t = unaudited_triage("exaone-moe").expect("verdict");
    assert!(
        t.blocker.contains("CLEAN"),
        "exaone-moe's verdict must record the checked-and-clean axis: {}",
        t.blocker
    );
    assert!(t.blocker.contains("REQUIRED"), "{}", t.blocker);
}

/// Every one of the 47 now carries a verdict, and the four classes are
/// all represented.
///
/// The distribution is the headline: `NewCode` dominates. That is the
/// honest answer to "how far is ferrox from llama.cpp on models", and it
/// is the number this whole item existed to produce.
#[test]
fn every_unaudited_row_is_triaged_and_the_distribution_is_pinned() {
    let mut fixture = 0;
    let mut arm = 0;
    let mut new_code = 0;
    let mut unknown = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        match p.triage.expect("every unaudited row is triaged").class {
            TriageClass::FixtureAway => fixture += 1,
            TriageClass::OneMatchArm => arm += 1,
            TriageClass::NewCode => new_code += 1,
            TriageClass::Unknown => unknown += 1,
        }
    }
    assert_eq!(
        (fixture, arm, new_code, unknown),
        (9, 2, 26, 4),
        "the triage distribution moved; if a verdict changed on evidence that is correct, \
         update this and docs/MODELS.md together"
    );
    assert_eq!(fixture + arm + new_code + unknown, 41);
}
