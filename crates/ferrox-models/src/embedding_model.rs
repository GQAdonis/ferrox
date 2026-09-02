//! One GGUF path in, one embedding vector out.
//!
//! Binds a tokenizer to a [`TextEncoder`] and owns the two steps
//! between them that neither half should own alone: adding the model's
//! own special tokens (`[CLS] … [SEP]`) around the tokenizer's pieces,
//! and pooling the hidden states the way the checkpoint's
//! `pooling_type` says.
//!
//! This is the type `/v1/embeddings` and the CLI both hold. It exists
//! so neither of them has to know that `bert` is an encoder, that
//! WordPiece does not add its own specials, or that CLS pooling means
//! row zero.

use thiserror::Error;

use crate::bert_gguf_loader::{load_bert_encoder, read_bert_hparams, BERT_ARCH};
use crate::encoder::{EncodeError, TextEncoder};
use crate::loader::LoadError;
use crate::pooling::{l2_normalize, pool, PoolingType};
use crate::rank_head::{load_rank_head, RankHead};
use crate::tokenizer::{GgufWordPieceTokenizer, TokenizerLoadError};

/// Encoder architectures upstream builds from `bert.cpp` and the other
/// embedding rows in the capability catalog, with what each one needs
/// that this crate does not have. Used to refuse *by name* instead of
/// with a generic "unsupported".
const NOT_YET: &[(&str, &str)] = &[
    ("nomic-bert", "RoPE on Q/K and a gated FFN"),
    ("nomic-bert-moe", "RoPE, a gated FFN and MoE expert layers"),
    ("jina-bert-v2", "GEGLU and a second attention norm"),
    ("jina-bert-v3", "RoPE and per-projection QK norm"),
    ("neo-bert", "per-projection QK norm"),
    (
        "modern-bert",
        "its own graph (local/global alternating attention)",
    ),
    ("eurobert", "its own graph"),
    ("t5encoder", "the T5 encoder stack"),
    ("llama-embed", "a decoder embedding path, not an encoder"),
    (
        "gemma-embedding",
        "a decoder embedding path, not an encoder",
    ),
    ("pangu-embedded", "a decoder embedding path, not an encoder"),
];

/// True when `general.architecture` names an encoder / embedding model
/// rather than something with an output head.
///
/// This is the question a *server* asks before it decides which loader
/// a checkpoint path goes to: an encoder can never reach the decoder
/// path, so routing it there produces a refusal about a missing tensor
/// instead of "this is an embedding model". The answer comes from the
/// capability registry's own [`crate::capability::ArchScope`] and not
/// from a second list beside [`NOT_YET`], because two lists of the same
/// architectures is the copy this repo has already paid for seven times
/// — a row added to the registry is covered here the moment it lands.
///
/// `true` does not mean ferrox can serve it. It means
/// [`EmbeddingModel::from_gguf_path`] is the loader that will either
/// build it or refuse it *by name*.
pub fn is_embedding_arch(arch: &str) -> bool {
    crate::capability::resolve_profile(arch).is_some_and(|p| {
        matches!(
            p.scope,
            crate::capability::ArchScope::DeferredEncoderEmbedding
        )
    })
}

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerLoadError),
    #[error(transparent)]
    Encode(#[from] EncodeError),
    #[error(
        "architecture {arch:?} is an embedding model ferrox cannot serve yet: it needs {needs}. \
         Only {BERT_ARCH:?} is implemented"
    )]
    NotYetImplemented { arch: String, needs: &'static str },
    #[error(
        "architecture {0:?} is not an embedding model this build knows. \
         Only {BERT_ARCH:?} is implemented"
    )]
    NotAnEmbeddingModel(String),
    #[error(
        "{arch:?} carries tokenizer.ggml.model = {model:?}, but this embedding path only has \
         WordPiece (\"bert\")"
    )]
    UnsupportedTokenizer { arch: String, model: String },
    #[error(
        "the embedding model {name:?} ({arch}) carries no reranker classification head: the \
         checkpoint has no cls / cls.output tensors, so it has no relevance score to \
         report. It can only produce embeddings"
    )]
    NoRankHead { name: String, arch: String },
    #[error(
        "the encoder for {arch:?} has no two-segment (query, document) input form, which a \
         cross-encoder rerank needs. Concatenating the two texts would score fluently and \
         wrongly, so this refuses instead"
    )]
    NoPairInput { arch: String },
}

/// A loaded embedding model: tokenizer + encoder + the checkpoint's own
/// pooling rule.
pub struct EmbeddingModel {
    encoder: Box<dyn TextEncoder + Send + Sync>,
    tokenizer: GgufWordPieceTokenizer,
    /// The reranker classification head, when the checkpoint carries
    /// one. `None` for a plain embedding model, and that is what makes
    /// `/v1/rerank` refuse rather than substitute a cosine similarity.
    rank_head: Option<RankHead>,
    arch: String,
    name: String,
}

impl EmbeddingModel {
    /// Opens `path` and builds whichever embedding stack its
    /// `general.architecture` names, or refuses naming what is missing.
    pub fn from_gguf_path(path: impl AsRef<std::path::Path>) -> Result<Self, EmbedError> {
        let file = ferrox_gguf::ShardedGguf::open(path.as_ref()).map_err(LoadError::from)?;
        let arch = ferrox_gguf::TensorSource::metadata_str(&file, "general.architecture")
            .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
            .to_string();
        if arch != BERT_ARCH {
            return Err(match NOT_YET.iter().find(|(a, _)| *a == arch) {
                Some((_, needs)) => EmbedError::NotYetImplemented { arch, needs },
                None => EmbedError::NotAnEmbeddingModel(arch),
            });
        }
        let tok_model = ferrox_gguf::TensorSource::metadata_str(&file, "tokenizer.ggml.model")
            .unwrap_or_default()
            .to_string();
        if tok_model != "bert" {
            return Err(EmbedError::UnsupportedTokenizer {
                arch,
                model: tok_model,
            });
        }
        let name = ferrox_gguf::TensorSource::metadata_str(&file, "general.name")
            .map(str::to_string)
            .unwrap_or_else(|| arch.clone());
        let tokenizer = GgufWordPieceTokenizer::from_gguf(&file)?;

        // ORDER IS LOAD-BEARING. `load_rank_head` MUST run before
        // `load_bert_encoder`, which ends in
        // `assert_every_tensor_consumed`: `cls.weight`, `cls.output.*`
        // and `cls.norm.weight` are read by nothing else in this crate,
        // so with the two lines swapped every reranker checkpoint dies
        // with an `UnconsumedTensors` refusal listing tensors ferrox
        // does in fact read. `read_bert_hparams` touches metadata only,
        // so asking for the geometry twice costs nothing.
        let hp = read_bert_hparams(&file)?;
        let rank_head = load_rank_head(&file, &hp.arch, hp.n_embd, hp.layer_norm_eps)?;
        let encoder = load_bert_encoder(&file)?;

        Ok(Self {
            encoder: Box::new(encoder),
            tokenizer,
            rank_head,
            arch,
            name,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.arch
    }

    /// The checkpoint's `general.name`, or its architecture when the
    /// file carries none. What `/v1/embeddings` reports as `model`.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn n_embd(&self) -> usize {
        self.encoder.n_embd()
    }

    pub fn n_ctx_train(&self) -> usize {
        self.encoder.n_ctx_train()
    }

    pub fn pooling_type(&self) -> PoolingType {
        self.encoder.pooling_type()
    }

    /// The exact ids the encoder will see for `text`: the tokenizer's
    /// pieces wrapped in the model's own special tokens. Public because
    /// `/v1/embeddings` has to report `usage.prompt_tokens`, and that
    /// number is this length — llama.cpp counts the specials too.
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        self.encoder.wrap_special(&self.tokenizer.encode(text))
    }

    /// Text for `ids`, through this checkpoint's own vocabulary.
    ///
    /// The counterpart to [`Self::token_ids`], so `/v1/detokenize`
    /// answers for an encoder rather than refusing. An embedding
    /// model's whole contract is the vector it returns for a string,
    /// and when that vector is surprising the first question is what
    /// tokens it actually saw. Without this the only way to ask was to
    /// load the checkpoint in a second tool.
    ///
    /// Not `wrap_special`'s inverse: it decodes exactly the ids given,
    /// including specials if the caller passes them, because a caller
    /// checking a tokenization wants to see what it sent.
    pub fn decode_tokens(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// Pooled embedding for `text`. `normalize` applies L2 normalization,
    /// which is what an OpenAI-compatible `/v1/embeddings` response is
    /// expected to carry and what llama.cpp's server does by default;
    /// the raw pooled vector is what the graph produced.
    pub fn embed(&self, text: &str, normalize: bool) -> Result<Vec<f32>, EmbedError> {
        let ids = self.token_ids(text);
        let mut v = self.encoder.embed_tokens(&ids)?;
        if normalize {
            l2_normalize(&mut v);
        }
        Ok(v)
    }

    /// Un-pooled `n_tokens × n_embd` hidden states, for a caller that
    /// wants to pool differently (or not at all).
    pub fn hidden_states(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.encoder.encode_tokens(&self.token_ids(text))?)
    }

    /// The checkpoint's reranker classification head, or `None` for a
    /// plain embedding model. What `/v1/rerank` checks before it
    /// promises a caller a relevance score.
    pub fn rank_head(&self) -> Option<&RankHead> {
        self.rank_head.as_ref()
    }

    /// The exact ids [`Self::rerank_score`] will see for one
    /// `(query, document)` pair: `[CLS] query [SEP] document [SEP]`.
    ///
    /// Separate from the scoring call for the same reason
    /// [`Self::token_ids`] is separate from [`Self::embed`] — a route
    /// has to report `usage.prompt_tokens`, and that number is this
    /// length.
    pub fn rerank_token_ids(&self, query: &str, document: &str) -> Result<Vec<u32>, EmbedError> {
        self.encoder
            .wrap_special_pair(
                &self.tokenizer.encode(query),
                &self.tokenizer.encode(document),
            )
            .ok_or_else(|| EmbedError::NoPairInput {
                arch: self.arch.clone(),
            })
    }

    /// The head's relevance score for a pair sequence built by
    /// [`Self::rerank_token_ids`].
    ///
    /// This is upstream's RANK path in full: encode, take the **CLS**
    /// row, run the classification head, report output 0
    /// (`send_rerank`'s `embd[0]`). The CLS row is taken here regardless
    /// of what `{arch}.pooling_type` says, because the head was trained
    /// on that position — `pooling_type = RANK` is the checkpoint
    /// *declaring* this path, not naming a pooling rule, which is why
    /// [`crate::pooling::pool`] still refuses RANK and must keep
    /// refusing it.
    ///
    /// No L2 normalization and no sigmoid: upstream reports the raw
    /// logit, so a score is comparable only against other scores from
    /// the same head, and this must not quietly squash it into `0..1`.
    pub fn rerank_score(&self, pair_ids: &[u32]) -> Result<f32, EmbedError> {
        let head = self
            .rank_head
            .as_ref()
            .ok_or_else(|| EmbedError::NoRankHead {
                name: self.name.clone(),
                arch: self.arch.clone(),
            })?;
        let hidden = self.encoder.encode_tokens(pair_ids)?;
        let cls = pool(&hidden, self.encoder.n_embd(), PoolingType::Cls)
            .map_err(|e| EmbedError::Encode(EncodeError::Pooling(e)))?;
        Ok(head.score(&cls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every deferred embedding architecture must produce a refusal
    /// that names it and names what it needs — not a generic error.
    #[test]
    fn every_deferred_embedding_arch_is_named_in_its_own_refusal() {
        for (arch, needs) in NOT_YET {
            let err = EmbedError::NotYetImplemented {
                arch: (*arch).to_string(),
                needs,
            };
            let msg = err.to_string();
            assert!(msg.contains(arch), "{msg} does not name {arch}");
            assert!(msg.contains(needs), "{msg} does not say what is missing");
        }
    }

    /// The catalog rows this module claims to cover must actually be
    /// the encoder/embedding rows the capability registry defers, so a
    /// new row added there cannot silently fall through to the generic
    /// "not an embedding model" arm.
    #[test]
    fn the_deferred_list_is_a_subset_of_the_capability_registry() {
        for (arch, _) in NOT_YET {
            assert!(
                crate::capability::resolve_profile(arch).is_some(),
                "{arch} is not in the capability registry"
            );
        }
    }

    /// [`is_embedding_arch`] is what a server routes on, so it has to
    /// name *exactly* the architectures this module can answer for:
    /// `bert`, which loads, plus every row in [`NOT_YET`], which
    /// refuses by name. A registry row scoped
    /// `DeferredEncoderEmbedding` that is in neither would be routed
    /// here and hit the generic `NotAnEmbeddingModel` arm, which says
    /// the opposite of the truth about it.
    #[test]
    fn is_embedding_arch_covers_the_registry_rows_and_nothing_else() {
        let mut registry: Vec<&str> = crate::capability::architecture_catalog()
            .iter()
            .filter(|p| {
                matches!(
                    p.scope,
                    crate::capability::ArchScope::DeferredEncoderEmbedding
                )
            })
            .map(|p| p.gguf_name)
            .collect();
        registry.sort_unstable();
        let mut known: Vec<&str> = NOT_YET
            .iter()
            .map(|(a, _)| *a)
            .chain(std::iter::once(BERT_ARCH))
            .collect();
        known.sort_unstable();
        assert_eq!(
            registry, known,
            "the registry's encoder/embedding rows and this module's own list disagree"
        );
        for arch in &registry {
            assert!(is_embedding_arch(arch), "{arch} is not routed to this path");
        }
        // A decoder must NOT be routed here, or `FERROX_MODEL_PATH`
        // pointing at a llama GGUF would be told it is an embedding
        // model.
        for arch in ["llama", "qwen3", "gemma3", "deepseek2"] {
            assert!(!is_embedding_arch(arch), "{arch} was routed to this path");
        }
    }
}
