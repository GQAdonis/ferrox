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
use crate::glm52_decoder::{
    glm52_forward_token, Glm52DecodeState, Glm52DecoderConfig, Glm52DecoderWeights,
};
use crate::kimi_decoder::{
    kimi_forward_token, KimiDecodeState, KimiDecoderConfig, KimiDecoderWeights,
};
use crate::kimi_tokenizer::KimiTokenizer;
use ferrox_core::cache::KvCache;

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
