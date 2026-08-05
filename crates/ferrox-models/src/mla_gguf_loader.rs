//! DeepSeek-2 / Mistral-4 GGUF → [`crate::engine::MlaEngine`].
//!
//! Tensor names follow llama.cpp `deepseek2` / `mistral4` (same graph):
//! `blk.{i}.attn_q_a|attn_q_b|attn_kv_a_mqa|attn_kv_b|attn_output` plus
//! optional `attn_q_a_norm` / `attn_kv_a_norm`. Dense FFN:
//! `ffn_{gate,up,down}`. MoE after `leading_dense_block_count` is
//! **fail-closed** until wired (clear error, not silent dense fallback).
//!
//! `use_output_gate` is off (classic DeepSeek-2). RoPE uses interleaved
//! Norm layout via [`crate::config::MlaRopeConfig`].

use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use ferrox_gguf::{GgmlType, TensorSource};

use crate::config::{MlaConfig, MlaRopeConfig};
use crate::engine::{MlaEngine, MlaLayerWeights};
use crate::loader::LoadError;
use crate::mla::MlaAttnWeights;

/// Hyperparameters read from `{arch}.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Deepseek2Hparams {
    pub arch: String,
    pub n_layer: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Layers `[0, leading_dense)` use dense SwiGLU; rest require MoE.
    pub leading_dense_block_count: usize,
    pub n_expert: usize,
}

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

/// Read DeepSeek-2 / Mistral-4 hparams from an opened GGUF.
pub fn read_deepseek2_hparams(file: &impl TensorSource) -> Result<Deepseek2Hparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if arch != "deepseek2" && arch != "mistral4" {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");
    let n_layer = meta_u64(file, &p("block_count"))? as usize;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dim = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let q_lora_rank = meta_u64(file, &p("attention.q_lora_rank"))? as usize;
    let kv_lora_rank = meta_u64(file, &p("attention.kv_lora_rank"))? as usize;
    let qk_nope_head_dim = meta_u64(file, &p("attention.qk_nope_head_dim"))? as usize;
    let qk_rope_head_dim = meta_u64(file, &p("attention.qk_rope_head_dim"))? as usize;
    let v_head_dim = meta_u64(file, &p("attention.v_head_dim"))
        .or_else(|_| meta_u64(file, &p("attention.key_length")))
        .unwrap_or(qk_nope_head_dim as u64) as usize;
    let leading_dense = file
        .metadata_u64(&p("leading_dense_block_count"))
        .unwrap_or(n_layer as u64) as usize;
    let n_expert = file.metadata_u64(&p("expert_count")).unwrap_or(0) as usize;
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 10000.0);
    Ok(Deepseek2Hparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dim,
        n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rms_norm_eps,
        rope_theta,
        leading_dense_block_count: leading_dense.min(n_layer),
        n_expert,
    })
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

fn load_f32_vec_optional(file: &impl TensorSource, name: &str) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(file, name)?))
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

fn load_mla_attn(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaAttnWeights, LoadError> {
    let l = layer_idx;
    let q_head_dim = hp.qk_nope_head_dim + hp.qk_rope_head_dim;
    let q_a_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_a.weight"))?;
    let q_b_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_b.weight"))?;
    let kv_a = load_weight_matrix(file, &format!("blk.{l}.attn_kv_a_mqa.weight"))?;
    let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;

    // Prefer combined `attn_kv_b`; else refuse split k_b/v_b until concat lands.
    let kv_b_proj = if file.find_tensor(&format!("blk.{l}.attn_kv_b.weight")).is_some() {
        load_weight_matrix(file, &format!("blk.{l}.attn_kv_b.weight"))?
    } else {
        return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
            format!(
                "blk.{l}.attn_kv_b.weight (split attn_k_b/attn_v_b not wired for MlaEngine yet)"
            ),
        )));
    };

    let q_a_ln = load_f32_vec_optional(file, &format!("blk.{l}.attn_q_a_norm.weight"))?
        .unwrap_or_else(|| vec![1.0; hp.q_lora_rank]);
    let kv_a_ln = load_f32_vec_optional(file, &format!("blk.{l}.attn_kv_a_norm.weight"))?
        .unwrap_or_else(|| vec![1.0; hp.kv_lora_rank]);

    let _ = (q_head_dim,);
    Ok(MlaAttnWeights {
        q_a_proj,
        q_a_layernorm: q_a_ln,
        q_b_proj,
        kv_a_proj_with_mqa: kv_a,
        kv_a_layernorm: kv_a_ln,
        kv_b_proj,
        o_proj,
        g_proj: None,
    })
}

fn load_dense_layer(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaLayerWeights, LoadError> {
    let l = layer_idx;
    Ok(MlaLayerWeights {
        attn_norm: load_f32_vec(file, &format!("blk.{l}.attn_norm.weight"))?,
        attn: load_mla_attn(file, layer_idx, hp)?,
        ffn_norm: load_f32_vec(file, &format!("blk.{l}.ffn_norm.weight"))?,
        ffn_gate: load_weight_matrix(file, &format!("blk.{l}.ffn_gate.weight"))?,
        ffn_up: load_weight_matrix(file, &format!("blk.{l}.ffn_up.weight"))?,
        ffn_down: load_weight_matrix(file, &format!("blk.{l}.ffn_down.weight"))?,
    })
}

/// Load a dense-lead (or fully dense) DeepSeek-2 / Mistral-4 GGUF into [`MlaEngine`].
pub fn load_mla_engine(file: &impl TensorSource) -> Result<MlaEngine, LoadError> {
    let hp = read_deepseek2_hparams(file)?;
    if hp.n_expert > 0 && hp.leading_dense_block_count < hp.n_layer {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "{}: MoE layers after leading_dense_block_count={} not wired in MlaEngine yet \
             (n_layer={}, n_expert={})",
            hp.arch, hp.leading_dense_block_count, hp.n_layer, hp.n_expert
        )));
    }
    let n_load = if hp.n_expert > 0 {
        hp.leading_dense_block_count
    } else {
        hp.n_layer
    };
    if n_load == 0 {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "{}: no dense layers to load",
            hp.arch
        )));
    }

    let embedding = if file.find_tensor("token_embd.weight").is_some() {
        load_weight_matrix(file, "token_embd.weight")?
    } else {
        return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
            "token_embd.weight".into(),
        )));
    };
    let final_norm = load_f32_vec(file, "output_norm.weight")?;
    let output_head = match load_weight_matrix(file, "output.weight") {
        Ok(w) => w,
        Err(_) => load_weight_matrix(file, "token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(n_load);
    for i in 0..n_load {
        layers.push(load_dense_layer(file, i, &hp)?);
    }

    Ok(MlaEngine {
        embedding,
        layers,
        final_norm,
        output_head,
        mla_cfg: MlaConfig {
            num_heads: hp.n_heads,
            q_lora_rank: hp.q_lora_rank,
            kv_lora_rank: hp.kv_lora_rank,
            qk_nope_head_dim: hp.qk_nope_head_dim,
            qk_rope_head_dim: hp.qk_rope_head_dim,
            v_head_dim: hp.v_head_dim,
            use_output_gate: false,
            rope: Some(MlaRopeConfig {
                theta: hp.rope_theta,
            }),
        },
        rms_norm_eps: hp.rms_norm_eps,
        hidden_dim: hp.hidden_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use crate::engine::Engine;
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
        // general.architecture + uint + float kvs
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
            buf.write_u32::<LittleEndian>(10).unwrap(); // UINT64
            buf.write_u64::<LittleEndian>(v).unwrap();
        }
        for &(k, v) in fkv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(6).unwrap(); // FLOAT32
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
            buf.write_u32::<LittleEndian>(0).unwrap();
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

    #[test]
    fn load_synthetic_deepseek2_dense_and_forward() {
        let h = 16usize;
        let n_heads = 2usize;
        let q_lora = 8usize;
        let kv_lora = 4usize;
        let qk_nope = 4usize;
        let qk_rope = 2usize;
        let v_dim = 4usize;
        let ffn = 32usize;
        let vocab = 8usize;
        let q_head = qk_nope + qk_rope;
        let arch = "deepseek2";

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; h * vocab],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "output.weight",
                vec![vocab as u64, h as u64],
                vec![0.02; h * vocab],
            ),
        ];
        for l in 0..2usize {
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_a.weight"),
                vec![q_lora as u64, h as u64],
                vec![0.01; h * q_lora],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_b.weight"),
                vec![(n_heads * q_head) as u64, q_lora as u64],
                vec![0.01; q_lora * n_heads * q_head],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_kv_a_mqa.weight"),
                vec![(kv_lora + qk_rope) as u64, h as u64],
                vec![0.01; h * (kv_lora + qk_rope)],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_kv_b.weight"),
                vec![(n_heads * (qk_nope + v_dim)) as u64, kv_lora as u64],
                vec![0.01; kv_lora * n_heads * (qk_nope + v_dim)],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_output.weight"),
                vec![h as u64, (n_heads * v_dim) as u64],
                vec![0.01; n_heads * v_dim * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                vec![h as u64, ffn as u64],
                vec![0.01; ffn * h],
            ));
        }

        let kv = [
            ("deepseek2.block_count", 2u64),
            ("deepseek2.embedding_length", h as u64),
            ("deepseek2.feed_forward_length", ffn as u64),
            ("deepseek2.attention.head_count", n_heads as u64),
            ("deepseek2.attention.q_lora_rank", q_lora as u64),
            ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
            ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
            ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
            ("deepseek2.attention.v_head_dim", v_dim as u64),
            ("deepseek2.leading_dense_block_count", 2u64),
            ("deepseek2.expert_count", 0u64),
        ];
        let fkv = [
            ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("deepseek2.rope.freq_base", 10000.0f32),
        ];
        let bytes = build_gguf(arch, &kv, &fkv, &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_mla_gguf_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let engine = load_mla_engine(&file).expect("load mla");
        assert_eq!(engine.layers.len(), 2);
        assert_eq!(engine.vocab_size(), vocab);
        let mut state = engine.new_state();
        let logits = engine.forward_token(0, 0, &mut state);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        let _ = std::fs::remove_file(&path);
    }
}
