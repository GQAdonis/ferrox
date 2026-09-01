//! BERT GGUF → [`BertEncoder`].
//!
//! Tensor names and requiredness follow llama.cpp
//! `llama_model_bert::load_arch_tensors` (`src/models/bert.cpp`), and
//! hparam keys follow its `load_arch_hparams` plus the shared
//! `LLM_KV_*` set.
//!
//! # `bert.cpp` upstream is five architectures; this is one of them
//!
//! `nomic-bert`, `nomic-bert-moe`, `jina-bert-v2`, `jina-bert-v3` and
//! `neo-bert` all build their graph from the same file, each switching
//! on `model.arch` for RoPE, a gated FFN, expert layers or a second
//! attention norm. [`crate::bert_encoder`] implements only the plain
//! `bert` shape, so every one of those differences is checked for here
//! and **refused by name**: a checkpoint that carries `ffn_gate` or
//! `attn_q_norm` is not silently run without it.
//!
//! The last line of defence is
//! [`crate::loader::assert_every_tensor_consumed`], which the load ends
//! with: for `bge-small-en-v1.5-q8_0.gguf` this graph reads all 197
//! tensors, so any weight a variant adds and this module has no home
//! for stops the load instead of being ignored.

use ferrox_gguf::{ShardedGguf, TensorSource};

use crate::bert_encoder::{BertEncoder, BertHparams, BertLayer};
use crate::loader::{
    assert_every_tensor_consumed, load_f32_vec, load_f32_vec_optional, load_weight_matrix,
    LoadError,
};
use crate::pooling::PoolingType;

/// The architecture string this loader implements.
pub const BERT_ARCH: &str = "bert";

/// llama.cpp's `tokenizer_model == "bert"` defaults, applied when the
/// GGUF carries no explicit id (`llama-vocab.cpp`).
const DEFAULT_CLS_ID: u32 = 101;
const DEFAULT_SEP_ID: u32 = 102;

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn refuse(what: &str) -> LoadError {
    LoadError::UnsupportedFeature(BERT_ARCH.to_string(), what.to_string())
}

/// Refuses if `name` exists, naming the upstream variant that carries it.
fn reject_tensor(file: &ShardedGguf, name: &str, why: &str) -> Result<(), LoadError> {
    if file.find_tensor(name).is_some() {
        return Err(refuse(&format!("checkpoint carries '{name}': {why}")));
    }
    Ok(())
}

/// The whole architecture policy of this loader, in one place so it can
/// be tested without a GGUF: `bert` and nothing else.
pub fn check_arch(arch: &str) -> Result<(), LoadError> {
    if arch == BERT_ARCH {
        Ok(())
    } else {
        Err(LoadError::UnsupportedArchitecture(arch.to_string()))
    }
}

/// Reads and checks `bert.*` hparams. Fails closed on anything the
/// graph in [`crate::bert_encoder`] does not implement.
pub fn read_bert_hparams(file: &impl TensorSource) -> Result<BertHparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    check_arch(&arch)?;
    let p = |suffix: &str| format!("{arch}.{suffix}");

    let n_layer = meta_u64(file, &p("block_count"))? as usize;
    let n_embd = meta_u64(file, &p("embedding_length"))? as usize;
    let n_ff = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_head = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_head_kv = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(n_head as u64) as usize;
    let n_ctx_train = meta_u64(file, &p("context_length"))? as usize;

    // `LLM_KV_ATTENTION_LAYERNORM_EPS` is read with `get_key(..., true)`
    // upstream, i.e. required: there is no sane default for a norm this
    // small (this checkpoint's is 1e-12, a thousand times tighter than
    // any RMSNorm eps in the rest of this codebase).
    let layer_norm_eps = file
        .metadata_f32(&p("attention.layer_norm_epsilon"))
        .ok_or_else(|| LoadError::MissingHparam(p("attention.layer_norm_epsilon")))?;

    // `n_token_types` is required by upstream's own loader, which
    // throws "model needs to define token type count".
    let n_token_types = meta_u64(file, "tokenizer.ggml.token_type_count")? as usize;
    if n_token_types == 0 {
        return Err(refuse("tokenizer.ggml.token_type_count is 0"));
    }

    // An encoder is bidirectional by construction. If a checkpoint ever
    // says otherwise, this graph is the wrong one for it.
    if file.metadata_bool(&p("attention.causal")).unwrap_or(false) {
        return Err(refuse(
            "bert.attention.causal is true, but this graph applies no mask — \
             a causal BERT would need a decoder path",
        ));
    }

    if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(refuse(&format!(
            "head_count {n_head} is not a multiple of head_count_kv {n_head_kv}"
        )));
    }
    if !n_embd.is_multiple_of(n_head) {
        return Err(refuse(&format!(
            "embedding_length {n_embd} is not divisible by head_count {n_head}"
        )));
    }
    if file.metadata_u64(&p("expert_count")).unwrap_or(0) != 0
        || file.metadata_u64(&p("moe_every_n_layers")).unwrap_or(0) != 0
    {
        return Err(refuse(
            "expert layers (nomic-bert-moe's moe_every_n_layers) are not implemented",
        ));
    }

    // Upstream defaults `hparams.pooling_type` to NONE and reads the key
    // as optional, so an absent key means "return every row", not
    // "guess CLS".
    let pooling = PoolingType::from_gguf(file, &arch)
        .map_err(|e| refuse(&e.to_string()))?
        .unwrap_or(PoolingType::None);

    let cls_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .unwrap_or(u64::from(DEFAULT_CLS_ID)) as u32;
    let sep_id = file
        .metadata_u64("tokenizer.ggml.seperator_token_id")
        .unwrap_or(u64::from(DEFAULT_SEP_ID)) as u32;

    Ok(BertHparams {
        arch,
        n_layer,
        n_embd,
        n_ff,
        n_head,
        n_head_kv,
        n_ctx_train,
        n_token_types,
        layer_norm_eps,
        pooling,
        cls_id,
        sep_id,
    })
}

/// Loads a `bert` GGUF into a runnable encoder.
pub fn load_bert_encoder_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<BertEncoder, LoadError> {
    load_bert_encoder(&ShardedGguf::open(path.as_ref())?)
}

/// Same, from an already-open file — so a caller that also needs the
/// tokenizer out of it ([`crate::embedding_model`]) mmaps it once.
pub fn load_bert_encoder(file: &ShardedGguf) -> Result<BertEncoder, LoadError> {
    let hp = read_bert_hparams(file)?;

    let tok_embd = load_weight_matrix(file, "token_embd.weight")?;
    let pos_embd = load_weight_matrix(file, "position_embd.weight")?;
    if pos_embd.rows() != hp.n_ctx_train {
        return Err(refuse(&format!(
            "position_embd.weight has {} rows but {}.context_length says {} — the learned \
             position table and the advertised context disagree",
            pos_embd.rows(),
            hp.arch,
            hp.n_ctx_train
        )));
    }
    if pos_embd.cols() != hp.n_embd || tok_embd.cols() != hp.n_embd {
        return Err(refuse(&format!(
            "embedding tables are {} / {} wide but embedding_length is {}",
            tok_embd.cols(),
            pos_embd.cols(),
            hp.n_embd
        )));
    }

    // Row 0 only: upstream views `type_embd` at offset 0 and adds it,
    // because a single-sequence embedding pass is always "Sentence A".
    // The tensor is `TENSOR_NOT_REQUIRED` there, so its absence is not
    // an error — but the whole table is then unused, and
    // `assert_every_tensor_consumed` would flag any other row anyway.
    let type_embd_row0 = match file.find_tensor("token_types.weight") {
        Some(_) => {
            let table = load_weight_matrix(file, "token_types.weight")?;
            if table.rows() != hp.n_token_types || table.cols() != hp.n_embd {
                return Err(refuse(&format!(
                    "token_types.weight is {}x{}, expected {}x{}",
                    table.rows(),
                    table.cols(),
                    hp.n_token_types,
                    hp.n_embd
                )));
            }
            Some(table.dequant_row(0))
        }
        None => None,
    };

    let tok_norm_w = load_f32_vec(file, "token_embd_norm.weight")?;
    let tok_norm_b = load_f32_vec(file, "token_embd_norm.bias")?;

    let mut layers = Vec::with_capacity(hp.n_layer);
    for l in 0..hp.n_layer {
        let b = format!("blk.{l}");
        reject_tensor(
            file,
            &format!("{b}.attn_qkv.weight"),
            "a fused QKV projection; this graph reads separate attn_q/attn_k/attn_v",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_q_norm.weight"),
            "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_k_norm.weight"),
            "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_norm_2.weight"),
            "jina-bert-v2's second attention norm, not implemented",
        )?;
        reject_tensor(
            file,
            &format!("{b}.ffn_gate.weight"),
            "a gated FFN (nomic-bert / jina-bert-v2 GEGLU); this graph runs a plain GELU MLP",
        )?;
        reject_tensor(
            file,
            &format!("{b}.ffn_up_exps.weight"),
            "MoE expert tensors (nomic-bert-moe), not implemented",
        )?;

        layers.push(BertLayer {
            wq: load_weight_matrix(file, &format!("{b}.attn_q.weight"))?,
            bq: load_f32_vec_optional(file, &format!("{b}.attn_q.bias"))?,
            wk: load_weight_matrix(file, &format!("{b}.attn_k.weight"))?,
            bk: load_f32_vec_optional(file, &format!("{b}.attn_k.bias"))?,
            wv: load_weight_matrix(file, &format!("{b}.attn_v.weight"))?,
            bv: load_f32_vec_optional(file, &format!("{b}.attn_v.bias"))?,
            wo: load_weight_matrix(file, &format!("{b}.attn_output.weight"))?,
            bo: load_f32_vec_optional(file, &format!("{b}.attn_output.bias"))?,
            attn_out_norm_w: load_f32_vec(file, &format!("{b}.attn_output_norm.weight"))?,
            attn_out_norm_b: load_f32_vec(file, &format!("{b}.attn_output_norm.bias"))?,
            ffn_up: load_weight_matrix(file, &format!("{b}.ffn_up.weight"))?,
            ffn_up_b: load_f32_vec_optional(file, &format!("{b}.ffn_up.bias"))?,
            ffn_down: load_weight_matrix(file, &format!("{b}.ffn_down.weight"))?,
            ffn_down_b: load_f32_vec_optional(file, &format!("{b}.ffn_down.bias"))?,
            layer_out_norm_w: load_f32_vec(file, &format!("{b}.layer_output_norm.weight"))?,
            layer_out_norm_b: load_f32_vec(file, &format!("{b}.layer_output_norm.bias"))?,
        });
    }

    let kv_dim = hp.n_head_kv * hp.head_dim();
    for (l, layer) in layers.iter().enumerate() {
        for (name, m, rows) in [
            ("attn_q", &layer.wq, hp.n_embd),
            ("attn_k", &layer.wk, kv_dim),
            ("attn_v", &layer.wv, kv_dim),
            ("attn_output", &layer.wo, hp.n_embd),
            ("ffn_up", &layer.ffn_up, hp.n_ff),
            ("ffn_down", &layer.ffn_down, hp.n_embd),
        ] {
            if m.rows() != rows {
                return Err(refuse(&format!(
                    "blk.{l}.{name}.weight has {} output rows, expected {rows}",
                    m.rows()
                )));
            }
        }
    }

    assert_every_tensor_consumed(file)?;

    Ok(BertEncoder {
        hp,
        tok_embd,
        type_embd_row0,
        pos_embd,
        tok_norm_w,
        tok_norm_b,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `bert`. The other eleven encoder rows in the catalog share
    /// `bert.cpp` upstream, and each of them differs from this graph in
    /// a way that would load clean and embed wrong.
    #[test]
    fn a_non_bert_architecture_is_refused_by_name() {
        for arch in [
            "nomic-bert",
            "nomic-bert-moe",
            "jina-bert-v2",
            "jina-bert-v3",
            "neo-bert",
            "modern-bert",
            "llama",
        ] {
            let err = check_arch(arch).unwrap_err();
            assert!(
                matches!(&err, LoadError::UnsupportedArchitecture(a) if a == arch),
                "{arch} was not refused: {err}"
            );
        }
        assert!(check_arch(BERT_ARCH).is_ok());
    }
}
