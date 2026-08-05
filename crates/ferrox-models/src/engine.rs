//! A trait-based abstraction over the two structurally different but
//! text-in/text-out-shaped forward passes this crate has: `Decoder`
//! (GQA+RoPE, used by GLM-5.2/DeepSeek V4 Pro and every real GGUF
//! checkpoint) and Kimi K3's dedicated hybrid KDA/Gated-MLA stack
//! (`kimi_decoder`). This lets `ferrox-server` share one generic
//! generation loop across both engines (see
//! `ferrox-server::generate::generate_engine`) for the actual
//! sampling/stop-sequence logic, rather than hand-duplicating it --
//! while keeping GGUF-only features (the KV block pool, `PrefixCache`)
//! as `Decoder`-specific code layered on top, not forced into this
//! trait. The reason: Kimi's KDA state is
//! a fixed-size recurrent matrix that collapses history irreversibly,
//! so it cannot support the same restore/truncate operations
//! `KvCache` can -- unifying those too would mean either a leaky
//! abstraction or silently pretending Kimi supports something it
//! doesn't.
//!
//! `forward_token`'s `pos` parameter is meaningful for `Decoder` (used
//! directly for RoPE) but not for `KimiEngine`: Kimi's real forward
//! pass (`kimi_forward_token`) derives position purely from its own
//! per-layer state (KDA's recurrent state, MLA's growing K/V buffers)
//! -- its real signature has no `pos` parameter at all. `KimiEngine`
//! ignores the argument; this is a real architectural fact about the
//! model, not an oversight in this trait's design.

use crate::config::{KdaConfig, MlaConfig};
use crate::decoder::Decoder;
use crate::deepseek_v4_decoder::{
    deepseek_v4_forward_token, DeepseekV4DecodeState, DeepseekV4DecoderConfig,
    DeepseekV4DecoderWeights,
};
use crate::glm52_decoder::{
    glm52_forward_token, Glm52DecodeState, Glm52DecoderConfig, Glm52DecoderWeights,
};
use crate::kimi_decoder::{
    kimi_forward_token, KimiDecodeState, KimiDecoderConfig, KimiDecoderWeights,
};
use crate::kimi_tokenizer::KimiTokenizer;
use ferrox_core::cache::KvCache;
use ferrox_core::weight_matrix::WeightMatrix;

/// A decoder that can run one incremental forward step given a token id
/// and position, updating its own per-layer state in place.
pub trait Engine {
    type State;

    /// Builds fresh (empty) per-layer state for a new request.
    fn new_state(&self) -> Self::State;

    fn vocab_size(&self) -> usize;

    fn forward_token(&self, token_id: usize, pos: usize, state: &mut Self::State) -> Vec<f32>;
}

impl Engine for Decoder {
    type State = Vec<KvCache>;

    fn new_state(&self) -> Vec<KvCache> {
        self.layers
            .iter()
            .map(|_| KvCache::new(self.config.n_kv_heads, self.config.head_dim))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn forward_token(&self, token_id: usize, pos: usize, state: &mut Self::State) -> Vec<f32> {
        Decoder::forward_token(self, token_id, pos, state)
    }
}

/// Bundles Kimi K3's weights with the three real config structs
/// `kimi_forward_token` needs, so `Engine::forward_token`'s three-
/// argument shape (`token_id`, `pos`, `state`) can wrap Kimi's real
/// four-config-argument function.
pub struct KimiEngine {
    pub weights: KimiDecoderWeights,
    pub cfg: KimiDecoderConfig,
    pub mla_cfg: MlaConfig,
    pub kda_cfg: KdaConfig,
}

impl Engine for KimiEngine {
    type State = KimiDecodeState;

    fn new_state(&self) -> KimiDecodeState {
        KimiDecodeState::new(&self.weights, &self.kda_cfg)
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token(&self, token_id: usize, _pos: usize, state: &mut Self::State) -> Vec<f32> {
        kimi_forward_token(
            &self.weights,
            &self.cfg,
            &self.mla_cfg,
            &self.kda_cfg,
            token_id,
            state,
        )
    }
}

/// GLM-5.2 dedicated DSA stack behind the same [`Engine`] trait as Kimi.
/// Synthetic / loader-backed weights only — no claim of a full real
/// ~744B serve path. Lets `generate_engine` exercise GLM without
/// forcing DSA into the GQA [`Decoder`].
pub struct Glm52Engine {
    pub weights: Glm52DecoderWeights,
    pub cfg: Glm52DecoderConfig,
}

impl Engine for Glm52Engine {
    type State = Glm52DecodeState;

    fn new_state(&self) -> Glm52DecodeState {
        Glm52DecodeState::new(&self.weights)
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token(&self, token_id: usize, _pos: usize, state: &mut Self::State) -> Vec<f32> {
        glm52_forward_token(&self.weights, &self.cfg, token_id, state)
    }
}

/// Multi-layer MLA stack for DeepSeek-2 / Mistral-4-style GGUF serve.
///
/// Uses [`crate::mla::mla_forward_token`] with asymmetric K/V caches
/// (plain `Vec<f32>`, not [`KvCache`]). Fail-closed at GGUF load until a
/// real DeepSeek-2 weight loader lands — this engine is the serve wire
/// for synthetic / fixture weights and for future loader integration.
pub struct MlaEngine {
    pub embedding: WeightMatrix,
    pub layers: Vec<MlaLayerWeights>,
    pub final_norm: Vec<f32>,
    pub output_head: WeightMatrix,
    pub mla_cfg: MlaConfig,
    pub rms_norm_eps: f32,
    pub hidden_dim: usize,
}

pub struct MlaLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn: crate::mla::MlaAttnWeights,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: WeightMatrix,
    pub ffn_up: WeightMatrix,
    pub ffn_down: WeightMatrix,
}

pub struct MlaDecodeState {
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl MlaEngine {
    pub fn new_state(&self) -> MlaDecodeState {
        MlaDecodeState {
            layers: (0..self.layers.len())
                .map(|_| (Vec::new(), Vec::new()))
                .collect(),
        }
    }
}

impl Engine for MlaEngine {
    type State = MlaDecodeState;

    fn new_state(&self) -> MlaDecodeState {
        MlaEngine::new_state(self)
    }

    fn vocab_size(&self) -> usize {
        self.output_head.rows()
    }

    fn forward_token(&self, token_id: usize, _pos: usize, state: &mut Self::State) -> Vec<f32> {
        use ferrox_core::matmul::{rms_norm, swiglu};
        let mut hidden = self.embedding.dequant_row(token_id);
        for (layer, (k_cache, v_cache)) in self.layers.iter().zip(state.layers.iter_mut()) {
            let normed = rms_norm(&hidden, &layer.attn_norm, self.rms_norm_eps);
            let attn_out = crate::mla::mla_forward_token(
                &layer.attn,
                &self.mla_cfg,
                &normed,
                self.rms_norm_eps,
                k_cache,
                v_cache,
            );
            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }
            let ffn_in = rms_norm(&hidden, &layer.ffn_norm, self.rms_norm_eps);
            let gate = layer.ffn_gate.apply(&ffn_in);
            let up = layer.ffn_up.apply(&ffn_in);
            let activated = swiglu(&gate, &up);
            let down = layer.ffn_down.apply(&activated);
            for (h, d) in hidden.iter_mut().zip(down.iter()) {
                *h += d;
            }
        }
        let final_normed = rms_norm(&hidden, &self.final_norm, self.rms_norm_eps);
        self.output_head.apply(&final_normed)
    }
}

/// DeepSeek V4 synthetic stack behind [`Engine`]. Preset `deepseek_v4_pro`
/// remains a sketch until a real GGUF loader + incremental DSV4 KV land.
pub struct DeepseekV4Engine {
    pub weights: DeepseekV4DecoderWeights,
    pub cfg: DeepseekV4DecoderConfig,
}

impl Engine for DeepseekV4Engine {
    type State = DeepseekV4DecodeState;

    fn new_state(&self) -> DeepseekV4DecodeState {
        DeepseekV4DecodeState::new(self.weights.embedding.shape[1])
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token(&self, token_id: usize, _pos: usize, state: &mut Self::State) -> Vec<f32> {
        deepseek_v4_forward_token(&self.weights, &self.cfg, token_id, state)
    }
}

/// A minimal text<->token-id interface shared by every real tokenizer
/// this crate has, regardless of each one's native id width
/// (`GgufBpeTokenizer`/`GgufSpmTokenizer`/`GgufUnigramTokenizer` use
/// `u32`, `KimiTokenizer` also uses `u32`) -- lets a generic generation
/// loop encode/decode without caring which concrete tokenizer it was
/// given.
pub trait TextTokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
}

impl TextTokenizer for KimiTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        KimiTokenizer::encode(self, text)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        KimiTokenizer::decode(self, &ids32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_dense_fixture;

    /// Locks in the refactor: calling `Decoder` through the generic
    /// `Engine` trait must be bit-identical to calling its own
    /// `forward_token`/`forward_batch` directly -- the whole point of
    /// the trait is that `ferrox-server`'s generic generation loop can
    /// use it as a drop-in replacement with zero numeric difference.
    #[test]
    fn decoder_via_engine_trait_matches_direct_forward_token_calls() {
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 64);
        let tokens = [3usize, 7, 1, 9];

        let mut direct_caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let mut direct_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            direct_logits = decoder.forward_token(tok, pos, &mut direct_caches);
        }

        let mut engine_state = Engine::new_state(&decoder);
        let mut engine_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            engine_logits = Engine::forward_token(&decoder, tok, pos, &mut engine_state);
        }

        assert_eq!(engine_logits, direct_logits);
        assert_eq!(Engine::vocab_size(&decoder), decoder.config.vocab_size);
    }

    /// Same equivalence, but against `forward_batch`'s independent
    /// computation (the ground truth every other test in this
    /// workspace already uses) rather than a second manual loop.
    #[test]
    fn decoder_via_engine_trait_matches_forward_batch_ground_truth() {
        let decoder = Decoder::new_random_small(test_dense_fixture(), 2, 64);
        let tokens = vec![2usize, 5, 8];

        let mut batch_caches: Vec<KvCache> = decoder
            .layers
            .iter()
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let batch_logits = decoder.forward_batch(&tokens, 0, &mut batch_caches);
        let ground_truth = batch_logits.last().unwrap().clone();

        let mut engine_state = Engine::new_state(&decoder);
        let mut engine_logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            engine_logits = Engine::forward_token(&decoder, tok, pos, &mut engine_state);
        }

        assert_eq!(engine_logits, ground_truth);
    }
}
