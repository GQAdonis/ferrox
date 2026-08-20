//! Compatibility marking for stored KV blocks: whether a block found
//! in a cache may be *used*, as opposed to merely found.
//!
//! A [`BlockHash`](crate::kv_block::BlockHash) says which tokens a block
//! covers. It says nothing about the shape of the tensors stored under
//! it -- how many layers, how many KV heads, what head dimension, what
//! dtype, how many token positions are really there. A cache that
//! survives a restart will eventually be read by a process configured
//! differently from the one that wrote it, and reading a block whose
//! layout does not match is not a miss: it is silently wrong attention
//! state, which produces confident wrong tokens.
//!
//! The one rule this module exists to enforce:
//!
//! > **A signature is derived from the block's own payload, never from
//! > the manager's expectation, and an unmarked block is rejected
//! > rather than trusted.**
//!
//! So [`CacheSignature::from_payload`] measures the tensors; it takes no
//! "expected" shape to fill gaps with. [`UnverifiedBlock::verify`] then
//! makes three separate checks, in order:
//!
//! 1. the block carries a signature at all ([`SignatureError::Unmarked`]
//!    otherwise -- absence is never treated as agreement);
//! 2. the recorded signature matches what the payload actually is
//!    ([`SignatureError::PayloadMismatch`]) -- a stamp may not vouch for
//!    a width the payload does not have;
//! 3. only then, that this shape is the shape the reader wants
//!    ([`SignatureError::Incompatible`]).
//!
//! Step 2 is the one that is easy to skip and expensive to skip: without
//! it, "the stamp says 32 heads" and "there are 32 heads" are different
//! claims that a cache would be treating as one.

use crate::cache::KvCache;

/// Layout version of a stored block payload. A reader accepts only the
/// versions in [`READABLE_FORMAT_VERSIONS`]; anything else is rejected
/// rather than guessed at.
pub const BLOCK_FORMAT_VERSION: u32 = 1;

/// Versions this build can read. Kept explicit (rather than `<=
/// BLOCK_FORMAT_VERSION`) so dropping support for an old layout is a
/// deliberate edit and not an accident of arithmetic.
pub const READABLE_FORMAT_VERSIONS: &[u32] = &[1];

/// Element type of the stored K/V tensors.
///
/// Only `F32` exists today, because [`KvCache`] stores `Vec<f32>`. The
/// enum is here so a future f16/quantized KV tier changes the signature
/// -- and therefore invalidates blocks written by an f32 build -- rather
/// than reinterpreting their bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    F32,
}

impl KvDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            KvDtype::F32 => "f32",
        }
    }
}

/// What a block's payload *is*: the shape any reader must match to use
/// it. Construct it from a payload with [`Self::from_payload`], or as a
/// reader's requirement with [`Self::expected`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSignature {
    pub format_version: u32,
    /// Identifies the weights the KV state was computed under. The one
    /// field no payload can prove about itself -- which is exactly why
    /// it is checked against the reader's expectation in step 3.
    pub model: String,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvDtype,
    /// Token positions actually stored, per layer.
    pub tokens: usize,
}

impl CacheSignature {
    /// Derives a signature by *measuring* `layers`. There is
    /// deliberately no parameter to fill a gap from: every field except
    /// `model` comes from the tensors themselves.
    ///
    /// Fails if the payload cannot describe itself coherently: no
    /// layers at all, layers that disagree with each other, or a layer
    /// whose buffers do not match its own declared shape.
    pub fn from_payload(model: &str, layers: &[KvCache]) -> Result<Self, SignatureError> {
        let first = layers.first().ok_or(SignatureError::EmptyPayload)?;
        let n_kv_heads = first.n_kv_heads;
        let head_dim = first.head_dim;
        if n_kv_heads == 0 || head_dim == 0 {
            return Err(SignatureError::DegenerateLayer {
                layer: 0,
                n_kv_heads,
                head_dim,
            });
        }
        let per_token = n_kv_heads * head_dim;
        let tokens = measure_layer(0, first, per_token)?;

        for (index, layer) in layers.iter().enumerate().skip(1) {
            if layer.n_kv_heads != n_kv_heads || layer.head_dim != head_dim {
                return Err(SignatureError::RaggedPayload {
                    layer: index,
                    field: "layer shape",
                    expected: format!("{n_kv_heads}x{head_dim}"),
                    found: format!("{}x{}", layer.n_kv_heads, layer.head_dim),
                });
            }
            let layer_tokens = measure_layer(index, layer, per_token)?;
            if layer_tokens != tokens {
                return Err(SignatureError::RaggedPayload {
                    layer: index,
                    field: "token count",
                    expected: tokens.to_string(),
                    found: layer_tokens.to_string(),
                });
            }
        }

        Ok(CacheSignature {
            format_version: BLOCK_FORMAT_VERSION,
            model: model.to_string(),
            n_layers: layers.len(),
            n_kv_heads,
            head_dim,
            dtype: KvDtype::F32,
            tokens,
        })
    }

    /// A reader's requirement: the shape this process would compute
    /// itself. Never stamped onto a block -- only compared against one.
    pub fn expected(
        model: &str,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        tokens: usize,
    ) -> Self {
        CacheSignature {
            format_version: BLOCK_FORMAT_VERSION,
            model: model.to_string(),
            n_layers,
            n_kv_heads,
            head_dim,
            dtype: KvDtype::F32,
            tokens,
        }
    }

    /// Field-by-field comparison, naming the first field that differs
    /// so an operator learns *what* changed rather than "cache miss".
    fn compare(
        &self,
        other: &CacheSignature,
        mismatch: fn(&'static str, String, String) -> SignatureError,
    ) -> Result<(), SignatureError> {
        if self.format_version != other.format_version {
            return Err(mismatch(
                "format_version",
                self.format_version.to_string(),
                other.format_version.to_string(),
            ));
        }
        if self.model != other.model {
            return Err(mismatch("model", self.model.clone(), other.model.clone()));
        }
        if self.n_layers != other.n_layers {
            return Err(mismatch(
                "n_layers",
                self.n_layers.to_string(),
                other.n_layers.to_string(),
            ));
        }
        if self.n_kv_heads != other.n_kv_heads {
            return Err(mismatch(
                "n_kv_heads",
                self.n_kv_heads.to_string(),
                other.n_kv_heads.to_string(),
            ));
        }
        if self.head_dim != other.head_dim {
            return Err(mismatch(
                "head_dim",
                self.head_dim.to_string(),
                other.head_dim.to_string(),
            ));
        }
        if self.dtype != other.dtype {
            return Err(mismatch(
                "dtype",
                self.dtype.as_str().to_string(),
                other.dtype.as_str().to_string(),
            ));
        }
        if self.tokens != other.tokens {
            return Err(mismatch(
                "tokens",
                self.tokens.to_string(),
                other.tokens.to_string(),
            ));
        }
        Ok(())
    }
}

/// Measures one layer, rejecting a layer whose buffers disagree with
/// its own declared `seq_len` -- `seq_len` is a claim, `k.len()` is the
/// evidence.
fn measure_layer(index: usize, layer: &KvCache, per_token: usize) -> Result<usize, SignatureError> {
    if !layer.k.len().is_multiple_of(per_token) {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "k length",
            expected: format!("a multiple of {per_token}"),
            found: layer.k.len().to_string(),
        });
    }
    if layer.v.len() != layer.k.len() {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "v length",
            expected: layer.k.len().to_string(),
            found: layer.v.len().to_string(),
        });
    }
    let tokens = layer.k.len() / per_token;
    if layer.seq_len != tokens {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "seq_len",
            expected: tokens.to_string(),
            found: layer.seq_len.to_string(),
        });
    }
    Ok(tokens)
}

/// Why a stored block was refused. Every variant is a refusal to guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignatureError {
    /// The block carries no signature. Not treated as "probably fine":
    /// an unmarked block was written by something whose layout is
    /// unknown, which is precisely the case that must not be trusted.
    Unmarked,
    /// A block with no layers describes nothing and can vouch for
    /// nothing.
    EmptyPayload,
    /// A layer with no heads or zero head dimension.
    DegenerateLayer {
        layer: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    /// The payload does not agree with itself: layers of different
    /// shapes or lengths, or a layer whose buffers contradict its own
    /// `seq_len`.
    RaggedPayload {
        layer: usize,
        field: &'static str,
        expected: String,
        found: String,
    },
    /// The recorded signature claims something the payload is not. The
    /// stamp is wrong (or was written by a build with a different
    /// layout); the payload is the truth.
    PayloadMismatch {
        field: &'static str,
        recorded: String,
        actual: String,
    },
    /// The payload is coherent and honestly stamped, but it is not what
    /// this reader needs -- a different model, a config change, a
    /// different block size.
    Incompatible {
        field: &'static str,
        expected: String,
        found: String,
    },
    /// Written by a build whose payload layout this one cannot read.
    UnsupportedFormat {
        found: u32,
        readable: &'static [u32],
    },
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignatureError::Unmarked => write!(
                f,
                "KV block carries no cache signature; refusing to trust an unmarked block"
            ),
            SignatureError::EmptyPayload => {
                write!(f, "KV block has no layers; nothing to verify")
            }
            SignatureError::DegenerateLayer {
                layer,
                n_kv_heads,
                head_dim,
            } => write!(
                f,
                "KV block layer {layer} is degenerate: {n_kv_heads} kv heads x {head_dim} head dim"
            ),
            SignatureError::RaggedPayload {
                layer,
                field,
                expected,
                found,
            } => write!(
                f,
                "KV block payload is inconsistent at layer {layer}: {field} is {found}, expected {expected}"
            ),
            SignatureError::PayloadMismatch {
                field,
                recorded,
                actual,
            } => write!(
                f,
                "KV block signature vouches for {field}={recorded} but its payload has {field}={actual}"
            ),
            SignatureError::Incompatible {
                field,
                expected,
                found,
            } => write!(
                f,
                "KV block is incompatible: {field} is {found}, this server needs {expected}"
            ),
            SignatureError::UnsupportedFormat { found, readable } => write!(
                f,
                "KV block format version {found} is not readable by this build (readable: {readable:?})"
            ),
        }
    }
}

impl std::error::Error for SignatureError {}

/// A block whose signature has been verified against its own payload
/// and against the reader's expectation. Only way to get one is
/// [`KvBlock::stamp`] (writing) or [`UnverifiedBlock::verify`]
/// (reading), so holding one is itself the proof.
pub struct KvBlock {
    signature: CacheSignature,
    layers: Vec<KvCache>,
}

/// Summarizes rather than dumping tensors: a block's `Debug` is for a
/// log line or a failing assertion, and printing every f32 in a KV
/// block helps nobody.
impl std::fmt::Debug for KvBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvBlock")
            .field("signature", &self.signature)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl std::fmt::Debug for UnverifiedBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnverifiedBlock")
            .field("signature", &self.signature)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl KvBlock {
    /// Stamps a block from its own payload. The caller supplies only
    /// the model identity; every shape field is measured.
    pub fn stamp(model: &str, layers: Vec<KvCache>) -> Result<Self, SignatureError> {
        let signature = CacheSignature::from_payload(model, &layers)?;
        Ok(KvBlock { signature, layers })
    }

    pub fn signature(&self) -> &CacheSignature {
        &self.signature
    }

    pub fn tokens(&self) -> usize {
        self.signature.tokens
    }

    pub fn layers(&self) -> &[KvCache] {
        &self.layers
    }

    pub fn into_layers(self) -> Vec<KvCache> {
        self.layers
    }
}

/// A block as it comes back from an untrusted source: a disk tier, a
/// shared cache, or a build that is not this one. `signature` is an
/// `Option` on purpose -- "no signature recorded" is a state a real
/// stored block can be in, and it must be representable so it can be
/// refused.
pub struct UnverifiedBlock {
    pub signature: Option<CacheSignature>,
    pub layers: Vec<KvCache>,
}

impl UnverifiedBlock {
    pub fn new(signature: Option<CacheSignature>, layers: Vec<KvCache>) -> Self {
        UnverifiedBlock { signature, layers }
    }

    /// Verifies in the order the module doc describes: marked, honest,
    /// then compatible. `expected` is used only in the last step -- it
    /// never contributes a field to the signature being checked.
    pub fn verify(self, expected: &CacheSignature) -> Result<KvBlock, SignatureError> {
        let recorded = self.signature.ok_or(SignatureError::Unmarked)?;
        if !READABLE_FORMAT_VERSIONS.contains(&recorded.format_version) {
            return Err(SignatureError::UnsupportedFormat {
                found: recorded.format_version,
                readable: READABLE_FORMAT_VERSIONS,
            });
        }
        // Measured from the payload. `recorded.model` is carried over
        // because no payload can prove which weights produced it; the
        // model check happens against `expected` below, where a wrong
        // model is caught as an incompatibility.
        let actual = CacheSignature::from_payload(&recorded.model, &self.layers)?;
        recorded.compare(&actual, |field, recorded, actual| {
            SignatureError::PayloadMismatch {
                field,
                recorded,
                actual,
            }
        })?;
        expected.compare(&actual, |field, expected, found| {
            SignatureError::Incompatible {
                field,
                expected,
                found,
            }
        })?;
        Ok(KvBlock {
            signature: actual,
            layers: self.layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(n_kv_heads: usize, head_dim: usize, tokens: usize) -> KvCache {
        let mut cache = KvCache::new(n_kv_heads, head_dim);
        let step = vec![0.5f32; n_kv_heads * head_dim];
        for _ in 0..tokens {
            cache.push(&step, &step).expect("unpooled push cannot fail");
        }
        cache
    }

    fn payload(n_layers: usize, n_kv_heads: usize, head_dim: usize, tokens: usize) -> Vec<KvCache> {
        (0..n_layers)
            .map(|_| layer(n_kv_heads, head_dim, tokens))
            .collect()
    }

    /// The core rule, in its positive form: every shape field comes
    /// from the tensors. `stamp` is given a model name and nothing else.
    #[test]
    fn signature_is_measured_from_the_payload() {
        let block = KvBlock::stamp("model-a", payload(3, 2, 8, 4)).expect("stamp");
        let sig = block.signature();
        assert_eq!(sig.n_layers, 3);
        assert_eq!(sig.n_kv_heads, 2);
        assert_eq!(sig.head_dim, 8);
        assert_eq!(sig.tokens, 4);
        assert_eq!(sig.dtype, KvDtype::F32);
        assert_eq!(sig.format_version, BLOCK_FORMAT_VERSION);
        assert_eq!(block.tokens(), 4);
        assert_eq!(block.layers().len(), 3);
    }

    #[test]
    fn a_stamped_block_round_trips_through_verification() {
        let layers = payload(3, 2, 8, 4);
        let signature = CacheSignature::from_payload("model-a", &layers).expect("signature");
        let expected = CacheSignature::expected("model-a", 3, 2, 8, 4);
        let block = UnverifiedBlock::new(Some(signature), layers)
            .verify(&expected)
            .expect("a block that is what it says it is must verify");
        assert_eq!(block.layers().len(), 3);
        assert_eq!(block.into_layers().len(), 3);
    }

    /// Absence of a signature is not agreement. This is the difference
    /// between a persistent cache and silent corruption after a config
    /// change: an unmarked block came from something whose layout is
    /// unknown by definition.
    #[test]
    fn an_unmarked_block_is_rejected_not_trusted() {
        let expected = CacheSignature::expected("model-a", 3, 2, 8, 4);
        let err = UnverifiedBlock::new(None, payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("an unmarked block must be refused");
        assert_eq!(err, SignatureError::Unmarked);
    }

    /// The rule the plan states as "a signature must never vouch for a
    /// width the payload does not have". Here the stamp is exactly what
    /// the reader expects -- and the payload is not. A cache that
    /// trusted the stamp (or, equivalently, stamped from the manager's
    /// expectation) would hand back tensors of the wrong width and
    /// produce confident wrong tokens.
    #[test]
    fn a_signature_that_overstates_its_payload_is_rejected() {
        let expected = CacheSignature::expected("model-a", 3, 2, 16, 4);
        let mut lying = expected.clone();
        assert_eq!(lying.head_dim, 16);
        let err = UnverifiedBlock::new(Some(lying.clone()), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims head_dim 16 over an 8-wide payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "head_dim",
                recorded: "16".into(),
                actual: "8".into(),
            }
        );

        // Same shape, overstated depth: 8 token positions claimed over
        // a 4-position payload.
        lying.head_dim = 8;
        lying.tokens = 8;
        let expected = CacheSignature::expected("model-a", 3, 2, 8, 8);
        let err = UnverifiedBlock::new(Some(lying.clone()), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims 8 tokens over a 4-token payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "tokens",
                recorded: "8".into(),
                actual: "4".into(),
            }
        );

        // And overstated layer count.
        lying.tokens = 4;
        lying.n_layers = 4;
        let expected = CacheSignature::expected("model-a", 4, 2, 8, 4);
        let err = UnverifiedBlock::new(Some(lying), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims 4 layers over a 3-layer payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "n_layers",
                recorded: "4".into(),
                actual: "3".into(),
            }
        );
    }

    /// A payload that lies to itself: `seq_len` says 4, the buffers
    /// hold 3. `seq_len` is a claim; `k.len()` is the evidence.
    #[test]
    fn a_layer_whose_seq_len_contradicts_its_buffers_is_rejected() {
        let mut layers = payload(2, 2, 8, 4);
        layers[1].seq_len = 7;
        let err = CacheSignature::from_payload("model-a", &layers)
            .expect_err("seq_len must be verified, not believed");
        assert_eq!(
            err,
            SignatureError::RaggedPayload {
                layer: 1,
                field: "seq_len",
                expected: "4".into(),
                found: "7".into(),
            }
        );
    }

    #[test]
    fn a_ragged_payload_is_rejected() {
        let mut layers = payload(3, 2, 8, 4);
        layers[2] = layer(2, 4, 4);
        let err = CacheSignature::from_payload("model-a", &layers).expect_err("shape disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 2,
                field: "layer shape",
                ..
            }
        ));

        let mut layers = payload(3, 2, 8, 4);
        layers[1] = layer(2, 8, 3);
        let err = CacheSignature::from_payload("model-a", &layers).expect_err("depth disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 1,
                field: "token count",
                ..
            }
        ));

        let mut layers = payload(2, 2, 8, 4);
        layers[0].v.truncate(8);
        let err = CacheSignature::from_payload("model-a", &layers).expect_err("k/v disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 0,
                field: "v length",
                ..
            }
        ));
    }

    #[test]
    fn an_empty_payload_is_rejected() {
        assert_eq!(
            CacheSignature::from_payload("model-a", &[]).expect_err("nothing to vouch for"),
            SignatureError::EmptyPayload
        );
    }

    /// An honest block of the wrong shape is a *miss*, reported as an
    /// incompatibility naming the field that changed -- not a
    /// corruption, and not a silent fallback.
    #[test]
    fn an_honest_block_from_a_different_config_is_incompatible() {
        let layers = payload(3, 2, 8, 4);
        let signature = CacheSignature::from_payload("model-a", &layers).expect("signature");
        let err = UnverifiedBlock::new(Some(signature.clone()), layers)
            .verify(&CacheSignature::expected("model-b", 3, 2, 8, 4))
            .expect_err("a different model must not share KV state");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "model",
                expected: "model-b".into(),
                found: "model-a".into(),
            }
        );

        let layers = payload(3, 2, 8, 4);
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected("model-a", 3, 4, 8, 4))
            .expect_err("a different KV head count must not be reused");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "n_kv_heads",
                expected: "4".into(),
                found: "2".into(),
            }
        );
    }

    #[test]
    fn an_unreadable_format_version_is_rejected() {
        let layers = payload(2, 2, 8, 4);
        let mut signature = CacheSignature::from_payload("model-a", &layers).expect("signature");
        signature.format_version = 99;
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected("model-a", 2, 2, 8, 4))
            .expect_err("an unknown layout must not be guessed at");
        assert_eq!(
            err,
            SignatureError::UnsupportedFormat {
                found: 99,
                readable: READABLE_FORMAT_VERSIONS,
            }
        );
    }

    #[test]
    fn errors_name_the_field_that_changed() {
        let text = SignatureError::Incompatible {
            field: "head_dim",
            expected: "128".into(),
            found: "64".into(),
        }
        .to_string();
        assert!(text.contains("head_dim"), "{text}");
        assert!(text.contains("64"), "{text}");
        assert!(text.contains("128"), "{text}");
        assert!(SignatureError::Unmarked.to_string().contains("unmarked"));
    }
}
