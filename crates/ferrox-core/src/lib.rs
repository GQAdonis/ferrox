//! ferrox-core: tensor primitives, quantized matmul, RMSNorm, RoPE, and
//! grouped-query causal attention with a simple KV cache.
//!
//! CPU reference implementation. The op set and naming (RMSNorm, RoPE,
//! GQA, KV cache) follow the now-standard vocabulary popularized by
//! llama.cpp / vLLM / candle-transformers; the actual Rust code below is
//! written independently. See docs/THIRD_PARTY_NOTICES.md for design credit.

pub mod attention;
pub mod cache;
pub mod csa_hca_compress;
pub mod deepseek_v4_attention;
pub mod expert_store;
pub mod matmul;
pub mod tensor;
pub mod turboquant;
pub mod weight_matrix;

pub use attention::{
    apply_rope_back, apply_rope_interleaved, apply_rope_interleaved_back,
    apply_rope_interleaved_with_freq_factors, apply_rope_with_freq_factors, causal_gqa_attention,
    causal_gqa_attention_paged, causal_gqa_attention_prefill, causal_gqa_attention_softcap,
    causal_gqa_attention_windowed, causal_gqa_attention_windowed_softcap, lightning_indexer_topk,
};
pub use cache::{
    KvBlockPool, KvCache, KvPoolExhausted, PagedKvCache, PagedKvStore, PagedStoreExhausted,
};
pub use csa_hca_compress::{channel_gated_pool, compress_block};
pub use deepseek_v4_attention::{csa_attention, hca_attention};
pub use matmul::{
    geglu, gelu, matmul_f32, rms_norm, rms_norm_per_head, silu, situ_and_mul, softcap_inplace,
    swiglu,
};
pub use tensor::Tensor;
#[cfg(feature = "cuda")]
pub use weight_matrix::cuda_dense_enabled;
#[cfg(feature = "metal")]
pub use weight_matrix::metal_dense_enabled;
pub use weight_matrix::{QuantKind, WeightMatrix};
