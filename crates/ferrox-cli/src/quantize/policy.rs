//! Two decisions `ferrox quantize` has to make, and the refusals that
//! follow from them:
//!
//! 1. **Which target may be written at all.** ferrox READS every quant
//!    kind the engine runs and can WRITE two: Q8_0 and Q4_K. A
//!    `quantize` subcommand whose name implies llama.cpp's range while
//!    it can emit two formats is the half-support this repo refuses, so
//!    a target ferrox cannot encode is refused BY NAME, with what it
//!    CAN write spelled out. A target whose llama.cpp MIX ferrox cannot
//!    reproduce -- Q4_K_S and Q4_K_M promote several tensors to Q5_K
//!    and Q6_K -- is refused too, unless `--pure` says to write the
//!    uniform file `llama-quantize --pure` would.
//! 2. **Which tensors get quantized.** llama.cpp keeps a specific set
//!    at source precision, and the set is not obvious: it is not "the
//!    small ones", it is a list of tensors whose values are used as
//!    something other than matrix rows (norms, router gates, position
//!    tables, conv kernels). Getting it wrong produces a file that
//!    loads and is subtly wrong, so the list is transcribed from
//!    `llama.cpp/src/llama-quant.cpp`'s `tensor_allows_quantization`
//!    with its line order intact rather than reinvented.

use ferrox_gguf::GgmlType;

/// The quantization targets `ferrox quantize` can actually encode.
///
/// Two encoders, three names: `Q4_K_S` and `Q4_K_M` are the same Q4_K
/// blocks and differ only in the `general.file_type` they record, which
/// is exactly how `llama-quantize --pure` behaves. See
/// [`Target::mix_promotes_to`] for why `--pure` is not optional for
/// them.
///
/// It is an enum rather than a string so that adding the next one means
/// adding the match arms the compiler asks for -- in [`Target::name`],
/// [`Target::ggml_type`], [`Target::llama_ftype`],
/// [`Target::mix_promotes_to`], [`Target::fallback_note`] and the
/// encoder dispatch -- instead of a name landing in the CLI's help text
/// ahead of a kernel, which is how a format gets "supported" in a table
/// and nowhere else.
// The variants are spelled the way `llama-quantize --type` spells
// them, upper-case and underscored. `Q4KS` would be the camel-case
// name and would be one more place where ferrox's word for a quant and
// the ecosystem's differ.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Q8_0,
    Q4_K_S,
    Q4_K_M,
}

impl Target {
    /// Every target this build can write. The refusal message below is
    /// generated from this, so it cannot fall out of date with the
    /// encoder the way a hand-written "we support: Q8_0" string would.
    pub const ALL: &'static [Target] = &[Target::Q8_0, Target::Q4_K_S, Target::Q4_K_M];

    pub fn name(self) -> &'static str {
        match self {
            Target::Q8_0 => "Q8_0",
            Target::Q4_K_S => "Q4_K_S",
            Target::Q4_K_M => "Q4_K_M",
        }
    }

    pub fn ggml_type(self) -> GgmlType {
        match self {
            Target::Q8_0 => GgmlType::Q8_0,
            Target::Q4_K_S | Target::Q4_K_M => GgmlType::Q4K,
        }
    }

    /// The block format's own name, which stops being the target's name
    /// once one encoder serves two mixes.
    pub fn ggml_type_name(self) -> &'static str {
        match self {
            Target::Q8_0 => "Q8_0",
            Target::Q4_K_S | Target::Q4_K_M => "Q4_K",
        }
    }

    /// llama.cpp's `general.file_type` (`LLAMA_FTYPE_MOSTLY_*`) value,
    /// so a file ferrox writes reports its mix the way every other tool
    /// in the ecosystem reads it.
    pub fn llama_ftype(self) -> u32 {
        match self {
            Target::Q8_0 => 7,    // LLAMA_FTYPE_MOSTLY_Q8_0
            Target::Q4_K_S => 14, // LLAMA_FTYPE_MOSTLY_Q4_K_S
            Target::Q4_K_M => 15, // LLAMA_FTYPE_MOSTLY_Q4_K_M
        }
    }

    /// The types llama.cpp's MIX for this target promotes some tensors
    /// to, over and above [`Target::ggml_type`]. Empty means the mix is
    /// uniform and `--pure` changes nothing.
    ///
    /// This is the difference between "ferrox can encode Q4_K" and
    /// "ferrox can write a Q4_K_M file", and conflating the two would
    /// be exactly the half-support this subcommand exists to refuse:
    /// `llama_tensor_get_type_impl` sends `output.weight` to Q6_K for
    /// any Q4_K target, `attn_v` and `ffn_down` to Q5_K or Q6_K on some
    /// layers, and (for Q4_K_M) `attn_qkv` to Q5_K. A file with those
    /// tensors left at Q4_K is a DIFFERENT file that would still be
    /// called Q4_K_M.
    ///
    /// So a non-empty list means `--pure` is required, and with it the
    /// output matches `llama-quantize --pure --type <name>` instead of
    /// pretending to match the mix.
    pub fn mix_promotes_to(self) -> &'static [&'static str] {
        match self {
            // Not "nothing special happens": `output.weight` DOES reach
            // the output arm and comes back Q8_0, because the promotion
            // there is `else if (new_type != GGML_TYPE_Q8_0) new_type =
            // GGML_TYPE_Q6_K;`.
            Target::Q8_0 => &[],
            Target::Q4_K_S | Target::Q4_K_M => &["Q5_K", "Q6_K"],
        }
    }

    /// What llama.cpp does with a tensor whose row length is not a
    /// multiple of this target's block size -- which is the difference
    /// between "llama.cpp stops here too" and "llama.cpp quietly writes
    /// a different type". Carried per target, because one sentence
    /// covering both was true of only Q8_0.
    pub fn fallback_note(self) -> &'static str {
        match self {
            // `tensor_type_fallback` has no Q8_0 arm: it throws.
            Target::Q8_0 => "llama.cpp has no fallback type for Q8_0 either -- it stops here too.",
            // `tensor_type_fallback`: Q4_K -> Q5_0, and then -> F16 if
            // the row is not a multiple of 32 either.
            Target::Q4_K_S | Target::Q4_K_M => {
                "llama.cpp answers this by changing the tensor's TYPE (Q4_K -> Q5_0, or F16 if the \
                 row is not a multiple of 32 either); ferrox can write neither, so it stops rather \
                 than write a file whose name says Q4_K."
            }
        }
    }

    /// Elements per block. A tensor whose row length is not a multiple
    /// of this cannot be encoded; [`Target::fallback_note`] says what
    /// llama.cpp does about it.
    pub fn block_elems(self) -> usize {
        self.ggml_type().block_layout().1
    }
}

/// Every name `llama-quantize` accepts, lowercased, paired with the
/// ferrox encoder that writes it -- `None` where there is none.
///
/// Transcribed from `tools/quantize/quantize.cpp`'s `QUANT_OPTIONS`,
/// including its aliasing: `q4_k` is listed there as "alias for Q4_K_M"
/// and carries `LLAMA_FTYPE_MOSTLY_Q4_K_M`, so it maps to the same
/// ferrox target and records the same `general.file_type`.
///
/// ONE table, not two. The shape here used to be a list of names plus a
/// separate `Target::ALL` lookup, and the two agreeing was nobody's
/// job; a name spelled differently in the two places refuses a target
/// this build can write, or accepts one it cannot.
const LLAMA_CPP_TARGETS: &[(&str, Option<Target>)] = &[
    ("q1_0", None),
    ("q2_0", None),
    ("q4_0", None),
    ("q4_1", None),
    ("mxfp4_moe", None),
    ("q5_0", None),
    ("q5_1", None),
    ("iq2_xxs", None),
    ("iq2_xs", None),
    ("iq2_s", None),
    ("iq2_m", None),
    ("iq1_s", None),
    ("iq1_m", None),
    ("tq1_0", None),
    ("tq2_0", None),
    ("q2_k", None),
    ("q2_k_s", None),
    ("iq3_xxs", None),
    ("iq3_s", None),
    ("iq3_m", None),
    ("q3_k", None),
    ("iq3_xs", None),
    ("q3_k_s", None),
    ("q3_k_m", None),
    ("q3_k_l", None),
    ("iq4_nl", None),
    ("iq4_xs", None),
    ("q4_k", Some(Target::Q4_K_M)),
    ("q4_k_s", Some(Target::Q4_K_S)),
    ("q4_k_m", Some(Target::Q4_K_M)),
    ("q5_k", None),
    ("q5_k_s", None),
    ("q5_k_m", None),
    ("q6_k", None),
    ("q8_0", Some(Target::Q8_0)),
    ("f16", None),
    ("bf16", None),
    ("f32", None),
    ("copy", None),
];

/// Why a requested target was refused. Three arms, because three very
/// different things go wrong and the user needs to know which: a gap in
/// ferrox, a typo, and a target ferrox can write only in llama.cpp's
/// `--pure` shape.
#[derive(Debug, PartialEq, Eq)]
pub enum TargetRefusal {
    /// A real llama.cpp target that ferrox has no encoder for.
    NotWritableYet(String),
    /// Not a quantization type at all.
    Unknown(String),
    /// ferrox has the block encoder, but llama.cpp's mix of this name
    /// promotes some tensors to types ferrox cannot write.
    MixNeedsPure(Target),
}

impl std::fmt::Display for TargetRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let writable = writable_targets();
        match self {
            TargetRefusal::NotWritableYet(name) => write!(
                f,
                "ferrox cannot WRITE {name} yet. It can read {name} and run it; it has no encoder \
                 for it.\n\
                 `ferrox quantize` writes: {writable}.\n\
                 The remaining K-quant and IQ encoders are each an iterative per-super-block \
                 scale/min fit (and, for the IQ tiers, a lattice search over a codebook). A \
                 min/max approximation of one produces a file that loads and generates measurably \
                 worse text, so ferrox stops here instead of writing it. Use llama.cpp's \
                 `llama-quantize --type {name}` for now; ferrox reads what it produces."
            ),
            TargetRefusal::Unknown(name) => write!(
                f,
                "'{name}' is not a quantization type. `ferrox quantize` writes: {writable}."
            ),
            TargetRefusal::MixNeedsPure(target) => write!(
                f,
                "ferrox can encode {ty} blocks, but it cannot write llama.cpp's {name} MIX: that \
                 mix promotes output.weight, and attn_v / ffn_down / attn_qkv on some layers, to \
                 {promotes}, and ferrox has no encoder for those.\n\
                 Pass --pure to write uniform {ty}, which is what `llama-quantize --pure --type \
                 {lower}` writes. Without --pure the file would be named {name} and not be one.",
                ty = target.ggml_type_name(),
                name = target.name(),
                lower = target.name().to_ascii_lowercase(),
                promotes = target.mix_promotes_to().join(" / "),
            ),
        }
    }
}

/// What `ferrox quantize` can write, in the form a user has to type it.
/// Generated from [`Target::ALL`] and [`Target::mix_promotes_to`], so
/// no message can promise a target the encoder dispatch does not have
/// or omit the `--pure` a target needs.
pub fn writable_targets() -> String {
    Target::ALL
        .iter()
        .map(|t| {
            if t.mix_promotes_to().is_empty() {
                t.name().to_string()
            } else {
                format!("{} (--pure only)", t.name())
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parses a `--type` argument, case-insensitively, the way
/// `llama-quantize` does, and admits it only if this build can actually
/// produce that file.
///
/// `pure` is a parameter rather than a check the caller does afterwards
/// on purpose: every reason a target can be refused lives in this one
/// function, so a second call site cannot admit a target by forgetting
/// one of them.
pub fn parse_target(raw: &str, pure: bool) -> Result<Target, TargetRefusal> {
    let lower = raw.to_ascii_lowercase();
    let Some((_, encoder)) = LLAMA_CPP_TARGETS.iter().find(|(n, _)| *n == lower) else {
        return Err(TargetRefusal::Unknown(raw.to_string()));
    };
    let Some(target) = encoder else {
        return Err(TargetRefusal::NotWritableYet(raw.to_string()));
    };
    if !pure && !target.mix_promotes_to().is_empty() {
        return Err(TargetRefusal::MixNeedsPure(*target));
    }
    Ok(*target)
}

/// What happens to one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition {
    /// Re-encode to the target type.
    Quantize,
    /// Copy the source bytes through unchanged, for the stated reason.
    Copy(&'static str),
}

/// Substring rules that keep a tensor at source precision, transcribed
/// in order from `tensor_allows_quantization` in llama.cpp's
/// `src/llama-quant.cpp`, each with the reason llama.cpp gives.
///
/// A single table, matched by one loop, rather than a chain of
/// hand-written `if`s: a chain is where one condition quietly stops
/// being checked. The reason string is not decoration -- `ferrox
/// quantize` prints it per tensor, so the file's composition is
/// auditable without reading this source.
const KEEP_AT_SOURCE_PRECISION: &[(&str, &str)] = &[
    ("_norm.weight", "norm"),
    ("ffn_gate_inp.weight", "expert gating"),
    (
        "ffn_gate_tid2eid.weight",
        "token-id -> expert-id routing table",
    ),
    ("altup", "tiny"),
    ("laurel", "tiny"),
    ("per_layer_model_proj", "small"),
    ("position_embd.weight", "positional embedding"),
    ("token_types.weight", "token types"),
    ("ssm_conv1d", "conv1d kernel"),
    ("shortconv.conv.weight", "conv kernel"),
    ("indexer.k_proj.weight", "tiny"),
    ("indexer.q_proj.weight", "tiny"),
    ("time_mix_first.weight", "RWKV small 2D"),
    ("time_mix_w0.weight", "RWKV small 2D"),
    ("time_mix_w1.weight", "RWKV small 2D"),
    ("time_mix_w2.weight", "RWKV small 2D"),
    ("time_mix_v0.weight", "RWKV small 2D"),
    ("time_mix_v1.weight", "RWKV small 2D"),
    ("time_mix_v2.weight", "RWKV small 2D"),
    ("time_mix_a0.weight", "RWKV small 2D"),
    ("time_mix_a1.weight", "RWKV small 2D"),
    ("time_mix_a2.weight", "RWKV small 2D"),
    ("time_mix_g1.weight", "RWKV small 2D"),
    ("time_mix_g2.weight", "RWKV small 2D"),
    ("time_mix_decay_w1.weight", "RWKV small 2D"),
    ("time_mix_decay_w2.weight", "RWKV small 2D"),
    ("time_mix_lerp_fused.weight", "RWKV small 2D"),
    ("attn_rel_b.weight", "relative position bias"),
    (".position_embd", "positional embedding"),
    ("sam.pos_embd", "multimodal"),
    ("sam.neck.", "multimodal"),
    ("sam.net_", "multimodal"),
    (".rel_pos", "multimodal"),
    (".patch_embd", "multimodal"),
    (".patch_merger", "multimodal"),
    ("a.rvq.codebook", "audio codebook"),
    ("mm.a.code_embd", "audio codebook"),
];

/// ggml's `ggml_n_dims`: the number of dimensions after trailing 1s are
/// dropped, floored at 1. GGUF stores the declared dims, so a `[4096,
/// 1]` tensor is 1-D to llama.cpp and must be to ferrox too, or the two
/// tools disagree about which tensors are quantized.
fn ggml_n_dims(shape: &[u64]) -> usize {
    for i in (1..shape.len()).rev() {
        if shape[i] > 1 {
            return i + 1;
        }
    }
    1
}

/// What `ferrox quantize` does with one tensor, matching llama.cpp's
/// `MOSTLY_Q8_0` mix.
///
/// Deliberately NOT parameterised on anything llama.cpp's
/// `llama_tensor_get_type` varies per layer, because for every target
/// this build writes it does not vary:
///
/// * For `MOSTLY_Q8_0` the per-layer function is a no-op.
///   `token_embd.weight` and `output.weight` do reach its
///   output/embedding special cases and both come back Q8_0 -- the
///   `else if (new_type != GGML_TYPE_Q8_0) new_type = GGML_TYPE_Q6_K;`
///   arm that bumps other mixes' output head does not fire when the
///   target already IS Q8_0.
/// * For the Q4_K targets it is not reached at all: they are admitted
///   only under `--pure`, and `--pure` is precisely the flag that skips
///   `llama_tensor_get_type_impl`. That is the whole reason
///   [`Target::mix_promotes_to`] makes `--pure` mandatory for them
///   rather than leaving a per-layer table here to be half-filled.
///
/// So this stays a uniform mix, and the tensor keep-list below --
/// llama.cpp's `tensor_allows_quantization`, which `--pure` does NOT
/// skip -- is the only thing that varies per tensor.
pub fn disposition(name: &str, shape: &[u64], dtype: GgmlType, target: Target) -> Disposition {
    if ggml_n_dims(shape) < 2 {
        return Disposition::Copy("1-D");
    }
    if !name.ends_with("weight") {
        return Disposition::Copy("not a weight");
    }
    for (needle, reason) in KEEP_AT_SOURCE_PRECISION {
        if name.contains(needle) {
            return Disposition::Copy(reason);
        }
    }
    if dtype == target.ggml_type() {
        return Disposition::Copy("already the target type");
    }
    Disposition::Quantize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal the whole subcommand is scoped around. A user who
    /// types a quant ferrox has no encoder for must be told so -- not
    /// handed a plausible-looking file.
    #[test]
    fn a_target_with_no_encoder_is_refused_by_name_and_says_what_can_be_written() {
        let err = parse_target("q6_k", false).unwrap_err();
        assert_eq!(err, TargetRefusal::NotWritableYet("q6_k".into()));
        let msg = err.to_string();
        assert!(msg.contains("cannot WRITE q6_k"), "{msg}");
        // What it CAN write, generated from `Target::ALL`.
        assert!(msg.contains("writes: Q8_0"), "{msg}");
        // And it must say ferrox STOPS, not that it did something
        // approximate: a message that leaves the door open is how a
        // user ends up with a file that loads and is worse.
        assert!(msg.contains("stops here instead of writing it"), "{msg}");
    }

    /// The refusal step 2 adds. `q4_k_m` names a MIX, and ferrox has
    /// only the block encoder for it; writing uniform Q4_K under that
    /// name would be a file whose `general.file_type` says Q4_K_M while
    /// its output head is four bits instead of six.
    ///
    /// The message has to name the way forward, because there is one --
    /// this is not a gap the user can do nothing about.
    #[test]
    fn the_q4_k_mixes_are_refused_without_pure_and_the_message_names_pure() {
        for name in ["q4_k", "q4_k_s", "q4_k_m"] {
            let err = parse_target(name, false).unwrap_err();
            let msg = err.to_string();
            assert!(matches!(err, TargetRefusal::MixNeedsPure(_)), "{name}");
            assert!(msg.contains("cannot write llama.cpp's"), "{msg}");
            assert!(msg.contains("Q5_K / Q6_K"), "{msg}");
            assert!(msg.contains("Pass --pure"), "{msg}");
            assert!(parse_target(name, true).is_ok(), "{name} with --pure");
        }
    }

    /// `q4_k` is an alias for `q4_k_m` in `llama-quantize`, and an
    /// alias that resolved to a different `general.file_type` would
    /// make the same command produce two different files depending on
    /// which tool ran it.
    #[test]
    fn q4_k_is_the_same_target_as_q4_k_m_the_way_llama_quantize_aliases_it() {
        assert_eq!(parse_target("q4_k", true).unwrap(), Target::Q4_K_M);
        assert_eq!(parse_target("q4_k_m", true).unwrap(), Target::Q4_K_M);
        assert_eq!(parse_target("q4_k_s", true).unwrap(), Target::Q4_K_S);
        // Same blocks, different declared mix. Both halves matter: the
        // first is why one encoder serves both, the second is why they
        // are two variants and not one.
        assert_eq!(
            Target::Q4_K_S.ggml_type(),
            Target::Q4_K_M.ggml_type(),
            "one encoder serves both"
        );
        assert_ne!(
            Target::Q4_K_S.llama_ftype(),
            Target::Q4_K_M.llama_ftype(),
            "and they must not report the same mix"
        );
    }

    /// Every llama.cpp target ferrox has no encoder for refuses, and
    /// none of them refuses as "unknown" -- a gap and a typo are
    /// different problems and get different messages. Run with
    /// `pure = true` so the only thing under test is the encoder gap.
    #[test]
    fn every_llama_cpp_target_ferrox_cannot_write_refuses_as_a_gap_not_a_typo() {
        for (name, encoder) in LLAMA_CPP_TARGETS {
            let parsed = parse_target(name, true);
            match encoder {
                Some(t) => assert_eq!(parsed.as_ref().ok(), Some(t), "{name} should be writable"),
                None => assert_eq!(
                    parsed.unwrap_err(),
                    TargetRefusal::NotWritableYet((*name).to_string()),
                    "{name}"
                ),
            }
        }
    }

    /// The table and the enum must agree in BOTH directions. One
    /// direction is the interesting one: a `Target` missing from the
    /// table is a target `ferrox quantize --type` can never reach, so
    /// it would sit in the help text as a target nobody can select.
    #[test]
    fn the_target_enum_and_the_llama_cpp_name_table_cover_each_other() {
        for t in Target::ALL {
            assert!(
                LLAMA_CPP_TARGETS
                    .iter()
                    .any(|(n, e)| *n == t.name().to_ascii_lowercase() && *e == Some(*t)),
                "{} is not reachable from the llama-quantize name table",
                t.name()
            );
        }
        for (name, encoder) in LLAMA_CPP_TARGETS {
            if let Some(t) = encoder {
                assert!(
                    Target::ALL.contains(t),
                    "{name} maps to a target that is not in Target::ALL"
                );
            }
        }
    }

    /// The one-line summary a user sees in every refusal and in
    /// `--help` is derived, not restated. If it were restated, adding
    /// Q4_K without editing it would have advertised Q8_0 only, and
    /// nothing would have gone red.
    #[test]
    fn the_writable_summary_names_every_target_and_flags_the_ones_needing_pure() {
        let s = writable_targets();
        for t in Target::ALL {
            assert!(s.contains(t.name()), "{s} is missing {}", t.name());
        }
        assert!(s.contains("Q4_K_M (--pure only)"), "{s}");
        assert!(!s.contains("Q8_0 (--pure only)"), "{s}");
    }

    #[test]
    fn a_name_that_is_not_a_quant_at_all_says_so() {
        let err = parse_target("q4_k_ultra", true).unwrap_err();
        assert_eq!(err, TargetRefusal::Unknown("q4_k_ultra".into()));
        assert!(err.to_string().contains("is not a quantization type"));
    }

    #[test]
    fn target_names_are_case_insensitive_like_llama_quantize() {
        assert_eq!(parse_target("q8_0", false).unwrap(), Target::Q8_0);
        assert_eq!(parse_target("Q8_0", false).unwrap(), Target::Q8_0);
        assert_eq!(parse_target("Q4_K_M", true).unwrap(), Target::Q4_K_M);
        assert_eq!(
            parse_target("Q8_o", false).unwrap_err(),
            TargetRefusal::Unknown("Q8_o".into())
        );
    }

    /// `ggml_n_dims` drops trailing 1s. A `[n, 1]` tensor is 1-D to
    /// llama.cpp; treating it as 2-D would quantize a tensor llama.cpp
    /// leaves alone and the two files would differ.
    #[test]
    fn trailing_unit_dimensions_do_not_make_a_tensor_two_dimensional() {
        assert_eq!(ggml_n_dims(&[4096]), 1);
        assert_eq!(ggml_n_dims(&[4096, 1]), 1);
        assert_eq!(ggml_n_dims(&[4096, 1, 1]), 1);
        assert_eq!(ggml_n_dims(&[4096, 11008]), 2);
        assert_eq!(ggml_n_dims(&[4096, 1, 8]), 3);
    }

    #[test]
    fn the_tensors_llama_cpp_keeps_at_source_precision_are_kept() {
        let two_d = [4096u64, 4096];
        let cases: &[(&str, bool)] = &[
            ("blk.0.attn_q.weight", true),
            ("blk.0.ffn_down.weight", true),
            ("token_embd.weight", true),
            ("output.weight", true),
            ("blk.0.attn_norm.weight", false),
            ("output_norm.weight", false),
            ("blk.0.ffn_gate_inp.weight", false),
            ("blk.0.ssm_conv1d.weight", false),
            ("position_embd.weight", false),
            ("token_types.weight", false),
            ("blk.0.altup_proj.weight", false),
            ("blk.0.attn_q.bias", false),
            ("v.patch_embd.weight", false),
        ];
        for (name, want_quantized) in cases {
            let got = disposition(name, &two_d, GgmlType::F16, Target::Q8_0);
            assert_eq!(
                got == Disposition::Quantize,
                *want_quantized,
                "{name} -> {got:?}"
            );
        }
    }

    /// Requantizing is not what this subcommand is for, but a tensor
    /// that is already the target type must not be re-encoded either:
    /// llama.cpp skips it (`quantize = cur_type != new_type`) and
    /// re-encoding it would need a decoder in the middle.
    #[test]
    fn a_tensor_already_in_the_target_type_is_copied_not_re_encoded() {
        assert_eq!(
            disposition(
                "blk.0.attn_q.weight",
                &[4096, 4096],
                GgmlType::Q8_0,
                Target::Q8_0
            ),
            Disposition::Copy("already the target type")
        );
    }
}
