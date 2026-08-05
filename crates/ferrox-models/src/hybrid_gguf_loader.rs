//! Qwen3.5 / Qwen3-Next hybrid GGUF → GDN weight loader skeleton (**P3**).
//!
//! Reads `{arch}.*` hparams for `qwen35` / `qwen35moe` / `qwen3next`,
//! classifies each block as GDN (linear) vs full attention from tensor
//! presence, and can materialize [`crate::gdn::GdnWeights`] for GDN layers.
//!
//! Full [`crate::hybrid_engine::HybridEngine`] assemble / serve is **not**
//! wired: [`try_load`] always returns
//! [`LoadError::UnsupportedFeature`] listing what is still missing.
//! Unit tests prove GDN tensor → [`gdn_forward_token`] without serve.

use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use ferrox_gguf::{GgmlType, TensorSource};

use crate::gdn::{GdnConfig, GdnWeights};
use crate::hybrid_engine::HybridEngine;
use crate::loader::LoadError;

/// Architectures this loader accepts.
pub const HYBRID_GDN_ARCHES: &[&str] = &["qwen35", "qwen35moe", "qwen3next"];

/// Hyperparameters from `{arch}.*` GGUF metadata (fail-closed if required keys absent).
#[derive(Debug, Clone)]
pub struct HybridHparams {
    pub arch: String,
    pub n_layer: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Depthwise conv kernel (`{arch}.ssm.conv_kernel`).
    pub ssm_conv_kernel: usize,
    /// Unused by equal-head [`GdnConfig`] today; retained for inventory.
    pub ssm_inner_size: usize,
    /// GDN head dim (`{arch}.ssm.state_size`).
    pub ssm_state_size: usize,
    /// Number of V heads (`{arch}.ssm.time_step_rank`).
    pub ssm_time_step_rank: usize,
    /// Number of K heads (`{arch}.ssm.group_count`).
    pub ssm_group_count: usize,
    /// Default full-attn interval when `attention.recurrent_layers` absent.
    pub full_attention_interval: usize,
    pub n_expert: usize,
}

/// Per-block attention kind from tensor presence (not metadata alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    /// Gated Delta Net / linear SSM block.
    Gdn,
    /// Standard GQA (`attn_q`).
    FullAttn,
}

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

/// Read hybrid GDN hparams; errors if arch is wrong or required keys missing.
pub fn read_hybrid_hparams(file: &impl TensorSource) -> Result<HybridHparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if !HYBRID_GDN_ARCHES.contains(&arch.as_str()) {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");

    let n_layer = meta_u64(file, &p("block_count"))? as usize;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dim = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_kv_heads = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(n_heads as u64) as usize;
    let head_dim = file
        .metadata_u64(&p("attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| hidden_dim / n_heads.max(1));

    let ssm_conv_kernel = meta_u64(file, &p("ssm.conv_kernel"))? as usize;
    let ssm_inner_size = meta_u64(file, &p("ssm.inner_size"))? as usize;
    let ssm_state_size = meta_u64(file, &p("ssm.state_size"))? as usize;
    let ssm_time_step_rank = meta_u64(file, &p("ssm.time_step_rank"))? as usize;
    let ssm_group_count = meta_u64(file, &p("ssm.group_count"))? as usize;
    let full_attention_interval = file
        .metadata_u64(&p("full_attention_interval"))
        .unwrap_or(4) as usize;
    let n_expert = file.metadata_u64(&p("expert_count")).unwrap_or(0) as usize;
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 10000.0);

    Ok(HybridHparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        ssm_conv_kernel,
        ssm_inner_size,
        ssm_state_size,
        ssm_time_step_rank,
        ssm_group_count,
        full_attention_interval,
        n_expert,
    })
}

/// Detect layer type from tensors:
/// - GDN if `ssm_conv1d.weight` **or** (`attn_qkv.weight` + `ssm_a`) present
/// - full attn if `attn_q.weight` present
pub fn detect_layer_kind(file: &impl TensorSource, layer_idx: usize) -> Result<HybridLayerKind, LoadError> {
    let l = layer_idx;
    let has_conv = file
        .find_tensor(&format!("blk.{l}.ssm_conv1d.weight"))
        .is_some();
    let has_qkv = file
        .find_tensor(&format!("blk.{l}.attn_qkv.weight"))
        .is_some();
    let has_ssm_a = file.find_tensor(&format!("blk.{l}.ssm_a")).is_some();
    let has_attn_q = file
        .find_tensor(&format!("blk.{l}.attn_q.weight"))
        .is_some();

    if has_conv || (has_qkv && has_ssm_a) {
        return Ok(HybridLayerKind::Gdn);
    }
    if has_attn_q {
        return Ok(HybridLayerKind::FullAttn);
    }
    Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
        format!(
            "blk.{l}: neither GDN (ssm_conv1d / attn_qkv+ssm_a) nor full-attn (attn_q) tensors found"
        ),
    )))
}

/// Classify every trunk layer.
pub fn classify_layers(
    file: &impl TensorSource,
    hp: &HybridHparams,
) -> Result<Vec<HybridLayerKind>, LoadError> {
    let mut out = Vec::with_capacity(hp.n_layer);
    for i in 0..hp.n_layer {
        out.push(detect_layer_kind(file, i)?);
    }
    Ok(out)
}

fn find_info<'a>(
    file: &'a impl TensorSource,
    name: &str,
) -> Result<&'a ferrox_gguf::TensorInfo, LoadError> {
    file.find_tensor(name)
        .ok_or_else(|| LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(name.to_string())))
}

fn load_f32_vec(file: &impl TensorSource, name: &str) -> Result<Vec<f32>, LoadError> {
    let info = find_info(file, name)?;
    let raw = file.tensor_bytes(name)?;
    match info.dtype {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        GgmlType::BF16 => ferrox_quant::dequant_bf16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

fn load_f32_vec_first_of(
    file: &impl TensorSource,
    names: &[&str],
) -> Result<(String, Vec<f32>), LoadError> {
    for &name in names {
        if file.find_tensor(name).is_some() {
            return Ok((name.to_string(), load_f32_vec(file, name)?));
        }
    }
    Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
        names.join(" | "),
    )))
}

fn quant_kind_for(dtype: GgmlType) -> Option<QuantKind> {
    match dtype {
        GgmlType::Q8_0 => Some(QuantKind::Q8_0),
        GgmlType::Q4_0 => Some(QuantKind::Q4_0),
        GgmlType::Q4K => Some(QuantKind::Q4K),
        GgmlType::Q5K => Some(QuantKind::Q5K),
        GgmlType::Q6K => Some(QuantKind::Q6K),
        GgmlType::Q2K => Some(QuantKind::Q2K),
        GgmlType::Q3K => Some(QuantKind::Q3K),
        GgmlType::Q4_1 => Some(QuantKind::Q4_1),
        GgmlType::Q5_0 => Some(QuantKind::Q5_0),
        GgmlType::Q5_1 => Some(QuantKind::Q5_1),
        GgmlType::Q8_1 => Some(QuantKind::Q8_1),
        GgmlType::IQ4NL => Some(QuantKind::IQ4NL),
        GgmlType::IQ4XS => Some(QuantKind::IQ4XS),
        _ => None,
    }
}

fn load_weight_matrix(file: &impl TensorSource, name: &str) -> Result<WeightMatrix, LoadError> {
    let info = find_info(file, name)?;
    let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        other => {
            return Err(LoadError::UnsupportedDtype(
                format!("{name} (expected 2D, got shape {other:?})"),
                info.dtype,
            ))
        }
    };
    match info.dtype {
        GgmlType::F32 | GgmlType::BF16 => {
            let data = load_f32_vec(file, name)?;
            Ok(WeightMatrix::F32(Tensor::new(data, shape)))
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, range) = file.tensor_mapped_range(name)?;
                Ok(WeightMatrix::Quantized {
                    data: WeightBytes::Mapped { mmap, range },
                    rows,
                    cols,
                    kind,
                })
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

fn gdn_config_from_hparams(hp: &HybridHparams) -> Result<GdnConfig, LoadError> {
    if hp.ssm_group_count != hp.ssm_time_step_rank {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "GDN with unequal K/V heads (ssm.group_count={} vs ssm.time_step_rank={}); \
                 gdn_forward_token currently requires equal heads",
                hp.ssm_group_count, hp.ssm_time_step_rank
            ),
        ));
    }
    Ok(GdnConfig {
        hidden_dim: hp.hidden_dim,
        num_v_heads: hp.ssm_time_step_rank,
        head_dim: hp.ssm_state_size,
        conv_kernel_size: hp.ssm_conv_kernel,
        rms_norm_eps: hp.rms_norm_eps,
    })
}

/// Load one GDN layer into [`GdnWeights`] (qwen35 split layout).
pub fn load_gdn_layer_weights(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &HybridHparams,
) -> Result<(GdnConfig, GdnWeights), LoadError> {
    let kind = detect_layer_kind(file, layer_idx)?;
    if kind != HybridLayerKind::Gdn {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{layer_idx} is {kind:?}, not GDN — cannot load into GdnWeights"
            ),
        ));
    }
    let cfg = gdn_config_from_hparams(hp)?;
    let l = layer_idx;

    let attn_qkv = load_weight_matrix(file, &format!("blk.{l}.attn_qkv.weight"))?;
    let attn_gate = load_weight_matrix(file, &format!("blk.{l}.attn_gate.weight"))?;
    let ssm_conv1d = load_f32_vec(file, &format!("blk.{l}.ssm_conv1d.weight"))?;
    let (_, ssm_dt) = load_f32_vec_first_of(
        file,
        &[
            &format!("blk.{l}.ssm_dt.bias"),
            &format!("blk.{l}.ssm_dt"),
        ],
    )?;
    let ssm_a = load_f32_vec(file, &format!("blk.{l}.ssm_a"))?;
    let ssm_beta = load_weight_matrix(file, &format!("blk.{l}.ssm_beta.weight"))?;
    let ssm_alpha = load_weight_matrix(file, &format!("blk.{l}.ssm_alpha.weight"))?;
    let ssm_norm = load_f32_vec(file, &format!("blk.{l}.ssm_norm.weight"))?;
    let ssm_out = load_weight_matrix(file, &format!("blk.{l}.ssm_out.weight"))?;

    let qkv_dim = cfg.qkv_dim();
    let expected_conv = qkv_dim * cfg.conv_kernel_size;
    if ssm_conv1d.len() != expected_conv {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l}.ssm_conv1d.weight has {} elements, expected {expected_conv} \
                 (qkv_dim={qkv_dim} × kernel={})",
                ssm_conv1d.len(),
                cfg.conv_kernel_size
            ),
        ));
    }
    if ssm_dt.len() != cfg.num_v_heads || ssm_a.len() != cfg.num_v_heads {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l} ssm_dt/ssm_a length mismatch: dt={}, a={}, num_v_heads={}",
                ssm_dt.len(),
                ssm_a.len(),
                cfg.num_v_heads
            ),
        ));
    }
    if ssm_norm.len() != cfg.head_dim {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l}.ssm_norm.weight has {} elements, expected head_dim={}",
                ssm_norm.len(),
                cfg.head_dim
            ),
        ));
    }

    Ok((
        cfg,
        GdnWeights {
            attn_qkv,
            attn_gate,
            ssm_conv1d,
            ssm_dt,
            ssm_a,
            ssm_beta,
            ssm_alpha,
            ssm_norm,
            ssm_out,
        },
    ))
}

fn serve_gap_message(hp: &HybridHparams, kinds: &[HybridLayerKind]) -> String {
    let n_gdn = kinds
        .iter()
        .filter(|k| **k == HybridLayerKind::Gdn)
        .count();
    let n_full = kinds
        .iter()
        .filter(|k| **k == HybridLayerKind::FullAttn)
        .count();
    let mut missing = vec![
        "HybridEngine layer assemble (GDN + full-attn residuals / post-norm / FFN)".into(),
        "token_embd / output_norm / lm_head serve path".into(),
        "hybrid KV + recurrent state scheduling".into(),
    ];
    if n_full > 0 {
        missing.push(format!(
            "full-attn GQA decode for {n_full} layer(s) (attn_q path not wired into HybridEngine)"
        ));
    }
    if hp.n_expert > 0 {
        missing.push(format!(
            "MoE expert routing (expert_count={})",
            hp.n_expert
        ));
    }
    if hp.ssm_group_count != hp.ssm_time_step_rank {
        missing.push(format!(
            "unequal GDN K/V heads (group_count={} vs time_step_rank={})",
            hp.ssm_group_count, hp.ssm_time_step_rank
        ));
    }
    if hp.arch == "qwen3next" {
        missing.push("qwen3next legacy fused tensors (ssm_ba / ssm_in) if present".into());
    }
    format!(
        "hybrid GGUF hparams OK (n_layer={}, GDN={n_gdn}, full_attn={n_full}); \
         GDN layer weights loadable via load_gdn_layer_weights; serve blocked — missing: {}",
        hp.n_layer,
        missing.join("; ")
    )
}

/// Attempt HybridEngine load for serve — fail-closed with an inventory of gaps.
///
/// Still validates hparams + layer classification so callers get a precise
/// error rather than a silent generic-Decoder path.
pub fn try_load(file: &impl TensorSource) -> Result<HybridEngine, LoadError> {
    let hp = read_hybrid_hparams(file)?;
    let kinds = classify_layers(file, &hp)?;
    // Prove at least one GDN layer's tensors parse when present.
    if let Some(i) = kinds.iter().position(|k| *k == HybridLayerKind::Gdn) {
        if hp.ssm_group_count == hp.ssm_time_step_rank {
            let _ = load_gdn_layer_weights(file, i, &hp)?;
        }
    }
    Err(LoadError::UnsupportedFeature(
        hp.arch.clone(),
        serve_gap_message(&hp, &kinds),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::{gdn_forward_token, GdnState};
    use byteorder::{LittleEndian, WriteBytesExt};
    use ferrox_gguf::GgufFile;
    use std::io::Write;

    struct FixtureTensor {
        name: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.write_f32::<LittleEndian>(*x).unwrap();
        }
        b
    }

    fn f32_tensor(name: &str, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        assert_eq!(
            values.len(),
            shape.iter().product::<u64>() as usize,
            "{name}"
        );
        FixtureTensor {
            name: name.into(),
            shape,
            bytes: f32_bytes(&values),
        }
    }

    fn build_gguf(arch: &str, kv: &[(&str, u64)], fkv: &[(&str, f32)], tensors: &[FixtureTensor]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        let kv_count = 1 + kv.len() + fkv.len();
        buf.write_u64::<LittleEndian>(kv_count as u64).unwrap();

        let write_string = |buf: &mut Vec<u8>, s: &str| {
            buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
        };
        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap();
        write_string(&mut buf, arch);
        for &(k, v) in kv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(10).unwrap();
            buf.write_u64::<LittleEndian>(v).unwrap();
        }
        for &(k, v) in fkv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(6).unwrap();
            buf.write_f32::<LittleEndian>(v).unwrap();
        }

        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            write_string(&mut buf, &t.name);
            buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
            for &d in t.shape.iter().rev() {
                buf.write_u64::<LittleEndian>(d).unwrap();
            }
            buf.write_u32::<LittleEndian>(0).unwrap(); // F32
            offsets.push(offset);
            buf.write_u64::<LittleEndian>(offset).unwrap();
            offset += (t.bytes.len().div_ceil(32) * 32) as u64;
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            while buf.len() < data_start + off as usize {
                buf.push(0);
            }
            buf.extend_from_slice(&t.bytes);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    fn tiny_gdn_fixture() -> (std::path::PathBuf, GgufFile, HybridHparams) {
        const H: usize = 4;
        const N_HEADS: usize = 2;
        const HEAD: usize = 2;
        const CONV: usize = 2;
        const QKV: usize = 3 * N_HEADS * HEAD; // 12
        const V: usize = N_HEADS * HEAD; // 4
        let arch = "qwen35";

        let mut tensors = Vec::new();
        // Fixture shape = WeightMatrix [rows, cols]; build_gguf+loader reverse to GGML ne order.
        tensors.push(f32_tensor(
            "blk.0.attn_qkv.weight",
            vec![QKV as u64, H as u64],
            vec![0.05; QKV * H],
        ));
        tensors.push(f32_tensor(
            "blk.0.attn_gate.weight",
            vec![V as u64, H as u64],
            vec![0.04; V * H],
        ));
        // Flat layout [qkv_dim, kernel] as gdn::causal_conv_step expects.
        tensors.push(f32_tensor(
            "blk.0.ssm_conv1d.weight",
            vec![CONV as u64, QKV as u64],
            {
                let mut c = vec![0.1; QKV * CONV];
                for d in 0..QKV {
                    c[d * CONV + (CONV - 1)] = 1.0;
                }
                c
            },
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_dt.bias",
            vec![N_HEADS as u64],
            vec![0.1, -0.05],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_a",
            vec![N_HEADS as u64],
            vec![-0.5, -0.75],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_beta.weight",
            vec![N_HEADS as u64, H as u64],
            vec![0.1; N_HEADS * H],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_alpha.weight",
            vec![N_HEADS as u64, H as u64],
            vec![0.08; N_HEADS * H],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_norm.weight",
            vec![HEAD as u64],
            vec![1.0; HEAD],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_out.weight",
            vec![H as u64, V as u64],
            vec![0.06; H * V],
        ));

        let kv = [
            ("qwen35.block_count", 1u64),
            ("qwen35.embedding_length", H as u64),
            ("qwen35.feed_forward_length", 8u64),
            ("qwen35.attention.head_count", 2u64),
            ("qwen35.attention.head_count_kv", 2u64),
            ("qwen35.attention.key_length", HEAD as u64),
            ("qwen35.ssm.conv_kernel", CONV as u64),
            ("qwen35.ssm.inner_size", QKV as u64),
            ("qwen35.ssm.state_size", HEAD as u64),
            ("qwen35.ssm.time_step_rank", N_HEADS as u64),
            ("qwen35.ssm.group_count", N_HEADS as u64),
        ];
        let fkv = [
            ("qwen35.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("qwen35.rope.freq_base", 10000.0f32),
        ];
        let bytes = build_gguf(arch, &kv, &fkv, &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_gdn_test_{}_{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).expect("synthetic hybrid GGUF must parse");
        let hp = read_hybrid_hparams(&file).expect("hparams");
        (path, file, hp)
    }

    #[test]
    fn read_hparams_fails_clear_when_ssm_keys_missing() {
        let tensors = [f32_tensor("token_embd.weight", vec![4, 4], vec![0.0; 16])];
        let bytes = build_gguf(
            "qwen35",
            &[
                ("qwen35.block_count", 1),
                ("qwen35.embedding_length", 4),
                ("qwen35.feed_forward_length", 8),
                ("qwen35.attention.head_count", 2),
            ],
            &[],
            &tensors,
        );
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_missing_ssm_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let err = read_hybrid_hparams(&file).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::MissingHparam(k) => assert!(k.contains("ssm.conv_kernel"), "{k}"),
            other => panic!("expected MissingHparam, got {other:?}"),
        }
    }

    #[test]
    fn synthetic_gdn_layer_loads_and_forward_token() {
        let (path, file, hp) = tiny_gdn_fixture();
        assert_eq!(detect_layer_kind(&file, 0).unwrap(), HybridLayerKind::Gdn);
        let (cfg, weights) = load_gdn_layer_weights(&file, 0, &hp).expect("load GDN");
        std::fs::remove_file(&path).ok();

        let mut state = GdnState::new(&cfg);
        let hidden = [0.2f32, -0.1, 0.3, -0.4];
        let out = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        assert_eq!(out.len(), hp.hidden_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn try_load_returns_unsupported_feature_inventory() {
        let (path, file, _) = tiny_gdn_fixture();
        let err = try_load(&file).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(arch, msg) => {
                assert_eq!(arch, "qwen35");
                assert!(msg.contains("HybridEngine"), "{msg}");
                assert!(msg.contains("serve blocked") || msg.contains("missing"), "{msg}");
                assert!(msg.contains("load_gdn_layer_weights"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    #[test]
    fn full_attn_layer_detected_from_attn_q() {
        let tensors = [f32_tensor(
            "blk.0.attn_q.weight",
            vec![4u64, 4u64],
            vec![0.0; 16],
        )];
        let bytes = build_gguf(
            "qwen35",
            &[
                ("qwen35.block_count", 1),
                ("qwen35.embedding_length", 4),
                ("qwen35.feed_forward_length", 8),
                ("qwen35.attention.head_count", 2),
                ("qwen35.ssm.conv_kernel", 2),
                ("qwen35.ssm.inner_size", 12),
                ("qwen35.ssm.state_size", 2),
                ("qwen35.ssm.time_step_rank", 2),
                ("qwen35.ssm.group_count", 2),
            ],
            &[],
            &tensors,
        );
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_full_attn_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            detect_layer_kind(&file, 0).unwrap(),
            HybridLayerKind::FullAttn
        );
        std::fs::remove_file(&path).ok();
    }
}
