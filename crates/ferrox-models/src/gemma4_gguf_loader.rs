//! Gemma-4 GGUF → [`crate::gemma4_engine::Gemma4Engine`].
//!
//! Reads `{arch}.*` hparams (SWA pattern array, shared KV, split head
//! dims, per-layer emb) and loads tensors named as in llama.cpp
//! `gemma4.cpp` / `LLM_TENSOR_NAMES`.

use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use ferrox_gguf::{GgmlType, GgufValue, TensorSource};

use crate::gemma4_engine::{
    Gemma4AttnWeights, Gemma4Engine, Gemma4Hparams, Gemma4LayerWeights, Gemma4Weights,
    GEMMA4_ARCHES,
};
use crate::loader::LoadError;

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

fn meta_usize_array(file: &impl TensorSource, key: &str) -> Option<Vec<usize>> {
    match file.metadata(key)? {
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(v.as_u64()? as usize);
            }
            Some(out)
        }
        v => Some(vec![v.as_u64()? as usize]),
    }
}

fn meta_bool_array(file: &impl TensorSource, key: &str, n: usize) -> Result<Vec<bool>, LoadError> {
    match file.metadata(key) {
        Some(GgufValue::Array(items)) if items.len() == n => {
            let mut out = Vec::with_capacity(n);
            for v in items {
                match v {
                    GgufValue::Bool(b) => out.push(*b),
                    GgufValue::U8(u) => out.push(*u != 0),
                    GgufValue::U32(u) => out.push(*u != 0),
                    GgufValue::U64(u) => out.push(*u != 0),
                    GgufValue::I32(i) => out.push(*i != 0),
                    other => {
                        return Err(LoadError::MissingHparam(format!(
                            "{key}: expected bool array element, got {other:?}"
                        )))
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(LoadError::MissingHparam(format!(
            "{key}: expected bool array of length {n}"
        ))),
        None => Err(LoadError::MissingHparam(key.to_string())),
    }
}

/// Parse Gemma-4 hparams; fail-closed on wrong arch or missing required keys.
pub fn read_gemma4_hparams(file: &impl TensorSource) -> Result<Gemma4Hparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if !GEMMA4_ARCHES.contains(&arch.as_str()) {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");
    let n_layer = meta_u64(file, &p("block_count"))? as usize;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dims = meta_usize_array(file, &p("feed_forward_length"))
        .ok_or_else(|| LoadError::MissingHparam(p("feed_forward_length")))?;
    if ffn_dims.len() != 1 && ffn_dims.len() != n_layer {
        return Err(LoadError::MissingHparam(format!(
            "{}: expected 1 or {n_layer} feed_forward_length entries, got {}",
            p("feed_forward_length"),
            ffn_dims.len()
        )));
    }
    let ffn_dims = if ffn_dims.len() == 1 {
        vec![ffn_dims[0]; n_layer]
    } else {
        ffn_dims
    };
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_kv_heads = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(1) as usize;
    let head_dim_full = meta_u64(file, &p("attention.key_length"))? as usize;
    let head_dim_swa = file
        .metadata_u64(&p("attention.key_length_swa"))
        .unwrap_or(head_dim_full as u64) as usize;
    let sliding_window = meta_u64(file, &p("attention.sliding_window"))? as usize;
    let is_swa = meta_bool_array(file, &p("attention.sliding_window_pattern"), n_layer)?;
    let shared_kv = file
        .metadata_u64(&p("attention.shared_kv_layers"))
        .unwrap_or(0) as usize;
    let n_layer_kv_from_start = if shared_kv > 0 && shared_kv < n_layer {
        n_layer - shared_kv
    } else {
        n_layer
    };
    let embd_per_layer = file
        .metadata_u64(&p("embedding_length_per_layer_input"))
        .unwrap_or(0) as usize;
    let final_logit_softcap = file
        .metadata_f32(&p("final_logit_softcapping"))
        .filter(|&v| v > 0.0);
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 1_000_000.0);
    let rope_theta_swa = meta_f32(file, &p("rope.freq_base_swa"), 10_000.0);
    Ok(Gemma4Hparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dims,
        n_heads,
        n_kv_heads,
        head_dim_full,
        head_dim_swa,
        sliding_window,
        is_swa,
        n_layer_kv_from_start,
        embd_per_layer,
        rms_norm_eps,
        rope_theta,
        rope_theta_swa,
        final_logit_softcap,
        attention_scale: 1.0,
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
        GgmlType::F16 => ferrox_quant::dequant_f16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::F16)),
        GgmlType::BF16 => ferrox_quant::dequant_bf16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
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
        GgmlType::IQ2XS => Some(QuantKind::IQ2XS),
        GgmlType::IQ2S => Some(QuantKind::IQ2S),
        GgmlType::IQ3S => Some(QuantKind::IQ3S),
        GgmlType::IQ1M => Some(QuantKind::IQ1M),
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
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
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

fn load_layer(
    file: &impl TensorSource,
    hp: &Gemma4Hparams,
    il: usize,
) -> Result<Gemma4LayerWeights, LoadError> {
    let head_dim = hp.head_dim(il);
    let has_kv = hp.has_kv(il);
    let k_name = format!("blk.{il}.attn_k.weight");
    let kn_name = format!("blk.{il}.attn_k_norm.weight");
    let (k_proj, k_norm) = if has_kv || file.find_tensor(&k_name).is_some() {
        (
            Some(load_weight_matrix(file, &k_name)?),
            Some(load_f32_vec(file, &kn_name)?),
        )
    } else {
        (None, None)
    };
    let v_name = format!("blk.{il}.attn_v.weight");
    let v_proj = if file.find_tensor(&v_name).is_some() {
        Some(load_weight_matrix(file, &v_name)?)
    } else {
        None
    };
    let out_scale = if file
        .find_tensor(&format!("blk.{il}.layer_output_scale.weight"))
        .is_some()
    {
        let v = load_f32_vec(file, &format!("blk.{il}.layer_output_scale.weight"))?;
        Some(v.first().copied().unwrap_or(1.0))
    } else {
        None
    };
    let (per_layer_inp_gate, per_layer_proj, per_layer_post_norm) = if hp.embd_per_layer > 0 {
        (
            Some(load_weight_matrix(
                file,
                &format!("blk.{il}.inp_gate.weight"),
            )?),
            Some(load_weight_matrix(file, &format!("blk.{il}.proj.weight"))?),
            Some(load_f32_vec(file, &format!("blk.{il}.post_norm.weight"))?),
        )
    } else {
        (None, None, None)
    };
    let _ = head_dim;
    Ok(Gemma4LayerWeights {
        attn_norm: load_f32_vec(file, &format!("blk.{il}.attn_norm.weight"))?,
        attn: Gemma4AttnWeights {
            q_proj: load_weight_matrix(file, &format!("blk.{il}.attn_q.weight"))?,
            k_proj,
            v_proj,
            o_proj: load_weight_matrix(file, &format!("blk.{il}.attn_output.weight"))?,
            q_norm: load_f32_vec(file, &format!("blk.{il}.attn_q_norm.weight"))?,
            k_norm,
            post_attn_norm: load_f32_vec(file, &format!("blk.{il}.post_attention_norm.weight"))?,
        },
        ffn_norm: load_f32_vec(file, &format!("blk.{il}.ffn_norm.weight"))?,
        ffn_gate: load_weight_matrix(file, &format!("blk.{il}.ffn_gate.weight"))?,
        ffn_up: load_weight_matrix(file, &format!("blk.{il}.ffn_up.weight"))?,
        ffn_down: load_weight_matrix(file, &format!("blk.{il}.ffn_down.weight"))?,
        ffn_post_norm: load_f32_vec(file, &format!("blk.{il}.post_ffw_norm.weight"))?,
        per_layer_inp_gate,
        per_layer_proj,
        per_layer_post_norm,
        out_scale,
    })
}

/// Load a Gemma-4 GGUF into [`Gemma4Engine`].
pub fn load_gemma4_engine(file: &impl TensorSource) -> Result<Gemma4Engine, LoadError> {
    let hp = read_gemma4_hparams(file)?;
    let token_embd = load_weight_matrix(file, "token_embd.weight")?;
    let output_head = match load_weight_matrix(file, "output.weight") {
        Ok(w) => w,
        Err(_) => load_weight_matrix(file, "token_embd.weight")?,
    };
    let (per_layer_token_embd, per_layer_model_proj, per_layer_proj_norm) = if hp.embd_per_layer > 0
    {
        (
            Some(load_weight_matrix(file, "per_layer_token_embd.weight")?),
            Some(load_weight_matrix(file, "per_layer_model_proj.weight")?),
            Some(load_f32_vec(file, "per_layer_proj_norm.weight")?),
        )
    } else {
        (None, None, None)
    };
    let rope_freqs = if file.find_tensor("rope_freqs.weight").is_some() {
        Some(load_f32_vec(file, "rope_freqs.weight")?)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(hp.n_layer);
    for il in 0..hp.n_layer {
        layers.push(load_layer(file, &hp, il)?);
    }
    let engine = Gemma4Engine {
        weights: Gemma4Weights {
            token_embd,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            output_norm: load_f32_vec(file, "output_norm.weight")?,
            output_head,
            rope_freqs,
        },
        hp,
    };
    // Same contract as the generic loader: resolve every kernel now,
    // then seal. This engine has no batched prefill, which the probe
    // records explicitly -- see `Gemma4Engine::probe_kernels`.
    engine.probe_kernels();
    ferrox_core::kernel_registry::seal_or_error()
        .map_err(|e| LoadError::StrictKernels(e.to_string()))?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::gemma4_engine::Gemma4Hparams;
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
        FixtureTensor {
            name: name.into(),
            shape,
            bytes: f32_bytes(&values),
        }
    }

    fn build_gguf(
        arch: &str,
        kv: &[(&str, u64)],
        fkv: &[(&str, f32)],
        bool_arr: Option<(&str, &[bool])>,
        tensors: &[FixtureTensor],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        let kv_count = 1 + kv.len() + fkv.len() + usize::from(bool_arr.is_some());
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
        if let Some((k, arr)) = bool_arr {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(9).unwrap(); // ARRAY
            buf.write_u32::<LittleEndian>(7).unwrap(); // BOOL
            buf.write_u64::<LittleEndian>(arr.len() as u64).unwrap();
            for &b in arr {
                buf.write_u8(u8::from(b)).unwrap();
            }
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
    fn parse_hparams_routing_from_synthetic_gguf() {
        let n_layer = 5usize;
        let h = 32usize;
        let swa = [true, true, true, true, false];
        let tensors = vec![
            f32_tensor("token_embd.weight", vec![4u64, h as u64], vec![0.01; 4 * h]),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        ];
        // Minimal tensors — only testing hparam parse via read_gemma4_hparams
        // through a tiny file that still opens.
        let kv = [
            ("gemma4.block_count", n_layer as u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", 64u64),
            ("gemma4.attention.head_count", 4u64),
            ("gemma4.attention.head_count_kv", 1u64),
            ("gemma4.attention.key_length", 16u64),
            ("gemma4.attention.key_length_swa", 8u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 2u64),
            ("gemma4.embedding_length_per_layer_input", 4u64),
        ];
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-6f32),
            ("gemma4.rope.freq_base", 1e6f32),
            ("gemma4.rope.freq_base_swa", 1e4f32),
            ("gemma4.final_logit_softcapping", 30.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &swa)),
            &tensors,
        );
        let path =
            std::env::temp_dir().join(format!("ferrox_gemma4_hp_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let hp = read_gemma4_hparams(&file).expect("hparams");
        assert_eq!(hp.n_layer, 5);
        assert_eq!(hp.n_layer_kv_from_start, 3);
        assert_eq!(hp.head_dim_full, 16);
        assert_eq!(hp.head_dim_swa, 8);
        assert_eq!(hp.is_swa, swa.to_vec());
        assert!(hp.has_kv(2));
        assert!(!hp.has_kv(3));
        assert_eq!(hp.kv_reuse_layer(3), 1); // SWA → 3-2? n_kv=3, swa → 3-2=1
        assert_eq!(hp.kv_reuse_layer(4), 2); // full → 3-1=2
        let _ = std::fs::remove_file(&path);
        let _ = Gemma4Hparams { ..hp };
    }

    #[test]
    fn load_tiny_gemma4_and_forward() {
        let n_layer = 2usize;
        let h = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let hd_swa = 8usize;
        let hd_full = 8usize; // same dim so shared-KV reuse is safe in tiny fixture
        let ffn = 32usize;
        let n_pl = 4usize;
        let vocab = 4usize;
        let swa = [true, false];

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; vocab * h],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "per_layer_token_embd.weight",
                vec![vocab as u64, (n_pl * n_layer) as u64],
                vec![0.01; vocab * n_pl * n_layer],
            ),
            f32_tensor(
                "per_layer_model_proj.weight",
                vec![(n_pl * n_layer) as u64, h as u64],
                vec![0.01; n_pl * n_layer * h],
            ),
            f32_tensor(
                "per_layer_proj_norm.weight",
                vec![n_pl as u64],
                vec![1.0; n_pl],
            ),
        ];
        for (il, &is_swa) in swa.iter().enumerate().take(n_layer) {
            let hd = if is_swa { hd_swa } else { hd_full };
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_q.weight"),
                vec![(n_heads * hd) as u64, h as u64],
                vec![0.01; n_heads * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_k.weight"),
                vec![(n_kv * hd) as u64, h as u64],
                vec![0.01; n_kv * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_v.weight"),
                vec![(n_kv * hd) as u64, h as u64],
                vec![0.01; n_kv * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_output.weight"),
                vec![h as u64, (n_heads * hd) as u64],
                vec![0.01; h * n_heads * hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_q_norm.weight"),
                vec![hd as u64],
                vec![1.0; hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_k_norm.weight"),
                vec![hd as u64],
                vec![1.0; hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_attention_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_gate.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; ffn * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_up.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; ffn * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_down.weight"),
                vec![h as u64, ffn as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_ffw_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.inp_gate.weight"),
                vec![n_pl as u64, h as u64],
                vec![0.01; n_pl * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.proj.weight"),
                vec![h as u64, n_pl as u64],
                vec![0.01; h * n_pl],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.layer_output_scale.weight"),
                vec![1u64],
                vec![1.0],
            ));
        }
        let kv = [
            ("gemma4.block_count", n_layer as u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", ffn as u64),
            ("gemma4.attention.head_count", n_heads as u64),
            ("gemma4.attention.head_count_kv", n_kv as u64),
            ("gemma4.attention.key_length", hd_full as u64),
            ("gemma4.attention.key_length_swa", hd_swa as u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 0u64),
            ("gemma4.embedding_length_per_layer_input", n_pl as u64),
        ];
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("gemma4.rope.freq_base", 10000.0f32),
            ("gemma4.rope.freq_base_swa", 10000.0f32),
            ("gemma4.final_logit_softcapping", 30.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &swa)),
            &tensors,
        );
        let path =
            std::env::temp_dir().join(format!("ferrox_gemma4_fwd_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let engine = load_gemma4_engine(&file).expect("load");
        assert_eq!(engine.hp.n_layer, 2);
        assert_eq!(engine.vocab_size(), vocab);
        let mut state = engine.new_state();
        let logits = engine.forward_token(0, 0, &mut state);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        let _ = std::fs::remove_file(&path);
    }
}
