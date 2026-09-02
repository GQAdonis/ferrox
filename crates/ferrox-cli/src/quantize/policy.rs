//! Two decisions `ferrox quantize` has to make, and the refusals that
//! follow from them:
//!
//! 1. **Which target may be written at all.** ferrox READS every quant
//!    kind the engine runs and can WRITE one. A `quantize` subcommand
//!    whose name implies llama.cpp's range while it can only emit Q8_0
//!    is the half-support this repo refuses, so a target ferrox cannot
//!    encode is refused BY NAME, with what it CAN write spelled out.
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
/// One variant today. It is an enum rather than a bool so that adding
/// the second one means adding a match arm the compiler asks for, in
/// [`Target::name`], [`Target::ggml_type`] and the encoder dispatch --
/// instead of a name landing in the CLI's help text ahead of a kernel,
/// which is how a format gets "supported" in a table and nowhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Q8_0,
}

impl Target {
    /// Every target this build can write. The refusal message below is
    /// generated from this, so it cannot fall out of date with the
    /// encoder the way a hand-written "we support: Q8_0" string would.
    pub const ALL: &'static [Target] = &[Target::Q8_0];

    pub fn name(self) -> &'static str {
        match self {
            Target::Q8_0 => "Q8_0",
        }
    }

    pub fn ggml_type(self) -> GgmlType {
        match self {
            Target::Q8_0 => GgmlType::Q8_0,
        }
    }

    /// llama.cpp's `general.file_type` (`LLAMA_FTYPE_MOSTLY_*`) value,
    /// so a file ferrox writes reports its mix the way every other tool
    /// in the ecosystem reads it.
    pub fn llama_ftype(self) -> u32 {
        match self {
            Target::Q8_0 => 7, // LLAMA_FTYPE_MOSTLY_Q8_0
        }
    }

    /// Elements per block. A tensor whose row length is not a multiple
    /// of this cannot be encoded, and for Q8_0 llama.cpp has no
    /// fallback type either -- it throws.
    pub fn block_elems(self) -> usize {
        self.ggml_type().block_layout().1
    }
}

/// Every name `llama-quantize` accepts, lowercased.
///
/// Transcribed from `tools/quantize/quantize.cpp`'s `QUANT_OPTIONS`.
/// Its only job is to separate two refusals that read very differently
/// to a user: "ferrox cannot write that yet" from "that is not a
/// quantization type". Both are refusals; only the first is a gap.
const LLAMA_CPP_TARGET_NAMES: &[&str] = &[
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "mxfp4_moe",
    "q5_0",
    "q5_1",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq2_m",
    "iq1_s",
    "iq1_m",
    "tq1_0",
    "tq2_0",
    "q2_k",
    "q2_k_s",
    "iq3_xxs",
    "iq3_s",
    "iq3_m",
    "q3_k",
    "iq3_xs",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "iq4_nl",
    "iq4_xs",
    "q4_k",
    "q4_k_s",
    "q4_k_m",
    "q5_k",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "bf16",
    "f32",
    "copy",
];

/// Why a requested target was refused. Both arms exist so the message
/// can say which kind of "no" this is.
#[derive(Debug, PartialEq, Eq)]
pub enum TargetRefusal {
    /// A real llama.cpp target that ferrox has no encoder for.
    NotWritableYet(String),
    /// Not a quantization type at all.
    Unknown(String),
}

impl std::fmt::Display for TargetRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let writable = Target::ALL
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(", ");
        match self {
            TargetRefusal::NotWritableYet(name) => write!(
                f,
                "ferrox cannot WRITE {name} yet. It can read {name} and run it; it has no encoder \
                 for it.\n\
                 `ferrox quantize` writes: {writable}.\n\
                 The K-quant and IQ encoders are an iterative per-super-block scale/min fit (and, \
                 for the IQ tiers, a lattice search over a codebook). A min/max approximation of \
                 one produces a file that loads and generates measurably worse text, so ferrox \
                 stops here instead of writing it. Use llama.cpp's `llama-quantize --type {name}` \
                 for now; ferrox reads what it produces."
            ),
            TargetRefusal::Unknown(name) => write!(
                f,
                "'{name}' is not a quantization type. `ferrox quantize` writes: {writable}."
            ),
        }
    }
}

/// Parses a `--type` argument, case-insensitively, the way
/// `llama-quantize` does.
pub fn parse_target(raw: &str) -> Result<Target, TargetRefusal> {
    let lower = raw.to_ascii_lowercase();
    if let Some(t) = Target::ALL
        .iter()
        .find(|t| t.name().eq_ignore_ascii_case(&lower))
    {
        return Ok(*t);
    }
    if LLAMA_CPP_TARGET_NAMES.contains(&lower.as_str()) {
        return Err(TargetRefusal::NotWritableYet(raw.to_string()));
    }
    Err(TargetRefusal::Unknown(raw.to_string()))
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
/// `llama_tensor_get_type` varies per layer: for `MOSTLY_Q8_0` that
/// function is a no-op. `token_embd.weight` and `output.weight` reach
/// its output/embedding special cases and both come back Q8_0 -- the
/// `else if (new_type != GGML_TYPE_Q8_0) new_type = GGML_TYPE_Q6_K;`
/// arm that bumps other mixes' output head does not fire when the
/// target already IS Q8_0. So this mix is uniform, and saying so here
/// is cheaper than a per-layer table that would be all one value.
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
    /// types the quant everyone actually ships must be told ferrox
    /// cannot write it -- not handed a plausible-looking file.
    #[test]
    fn a_k_quant_target_is_refused_by_name_and_says_what_can_be_written() {
        let err = parse_target("q4_k_m").unwrap_err();
        assert_eq!(err, TargetRefusal::NotWritableYet("q4_k_m".into()));
        let msg = err.to_string();
        assert!(msg.contains("cannot WRITE q4_k_m"), "{msg}");
        // What it CAN write, generated from `Target::ALL`.
        assert!(msg.contains("writes: Q8_0"), "{msg}");
        // And it must say ferrox STOPS, not that it did something
        // approximate: a message that leaves the door open is how a
        // user ends up with a file that loads and is worse.
        assert!(msg.contains("stops here instead of writing it"), "{msg}");
    }

    /// Every llama.cpp target except the one ferrox writes refuses, and
    /// none of them refuses as "unknown" -- a gap and a typo are
    /// different problems and get different messages.
    #[test]
    fn every_llama_cpp_target_ferrox_cannot_write_refuses_as_a_gap_not_a_typo() {
        for name in LLAMA_CPP_TARGET_NAMES {
            let parsed = parse_target(name);
            if Target::ALL
                .iter()
                .any(|t| t.name().eq_ignore_ascii_case(name))
            {
                assert!(parsed.is_ok(), "{name} should be writable");
            } else {
                assert_eq!(
                    parsed.unwrap_err(),
                    TargetRefusal::NotWritableYet((*name).to_string()),
                    "{name}"
                );
            }
        }
    }

    /// The two tables must agree: a target this build can write has to
    /// be a name llama.cpp knows, or `ferrox quantize --type X` and
    /// `llama-quantize --type X` mean different things.
    #[test]
    fn every_writable_target_is_a_name_llama_cpp_also_accepts() {
        for t in Target::ALL {
            assert!(
                LLAMA_CPP_TARGET_NAMES.contains(&t.name().to_ascii_lowercase().as_str()),
                "{} is not a llama-quantize target name",
                t.name()
            );
        }
    }

    #[test]
    fn a_name_that_is_not_a_quant_at_all_says_so() {
        let err = parse_target("q4_k_ultra").unwrap_err();
        assert_eq!(err, TargetRefusal::Unknown("q4_k_ultra".into()));
        assert!(err.to_string().contains("is not a quantization type"));
    }

    #[test]
    fn target_names_are_case_insensitive_like_llama_quantize() {
        assert_eq!(parse_target("q8_0").unwrap(), Target::Q8_0);
        assert_eq!(parse_target("Q8_0").unwrap(), Target::Q8_0);
        assert_eq!(
            parse_target("Q8_o").unwrap_err(),
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
