//! The reranker classification head: `cls`, `cls.output`, `cls.norm`
//! and `{arch}.classifier.output_labels`.
//!
//! Transcribed from llama.cpp `llm_graph_context::build_pooling`'s
//! `LLAMA_POOLING_TYPE_RANK` arm (`src/llama-graph.cpp`), which is a
//! **classification head and not a pooling rule** — the reason
//! [`crate::pooling::PoolingType::Rank`] still refuses in
//! [`crate::pooling::pool`] and always will. `pool` sees hidden states
//! and a width; it cannot see these matrices, so RANK is not a
//! question it can answer. The rank path is CLS pooling *followed by*
//! this head, and this module is the "followed by".
//!
//! # The graph, for the `bert` shape
//!
//! ```text
//! cur = hidden[CLS]                      (row 0 — see below)
//! if cls:      cur = cls · cur + cls_b
//!              cur = tanh(cur)
//!              if cls_norm: cur = LayerNorm(cur, cls_norm, no bias)
//! if cls_out:  cur = cls_out · cur + cls_out_b
//! score = cur[0]
//! ```
//!
//! Three of upstream's branches are deliberately **not** here, and each
//! is refused rather than approximated, because each belongs to an
//! architecture [`crate::bert_gguf_loader`] already refuses by name:
//!
//! * `modern-bert` pools with MEAN rather than CLS and uses GELU in
//!   place of the `tanh`.
//! * `qwen3` / `qwen3vl` take the **last** token rather than row 0
//!   (`build_inp_cls`'s `last` flag, `llama-graph.cpp:296-299`) and
//!   append a softmax over the outputs.
//! * `jina-reranker-v1-tiny-en` is the checkpoint upstream cites for
//!   the `cls_out`-absent case; it is `jina-bert-v2`, which this crate
//!   does not load.
//!
//! Since only `bert` reaches here, row 0 is the CLS row unconditionally
//! and there is no softmax. If another architecture is ever admitted,
//! this module has to grow its branch — it must not inherit `bert`'s.
//!
//! # Which float is the score
//!
//! `send_rerank` (`tools/server/server-context.cpp`) reads `embd[0]`:
//! the FIRST of `n_cls_out` outputs, whatever the rest are. That is
//! fine for a one-output relevance head and is a silent choice for a
//! many-output classifier, so [`load_rank_head`] refuses a multi-output
//! head that does not name its labels — see [`RankHead::labels`].

use ferrox_core::matmul::layer_norm;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_gguf::{GgufValue, ShardedGguf, TensorSource};

use crate::loader::{load_f32_vec_optional, load_weight_matrix, LoadError};

/// `LLM_TENSOR_CLS` / `CLS_OUT` / `CLS_NORM`, by their GGUF names
/// (`llama-arch.cpp:428-430`).
const CLS_W: &str = "cls.weight";
const CLS_B: &str = "cls.bias";
const CLS_OUT_W: &str = "cls.output.weight";
const CLS_OUT_B: &str = "cls.output.bias";
const CLS_NORM_W: &str = "cls.norm.weight";
const CLS_NORM_B: &str = "cls.norm.bias";

/// A dense projection with an optional bias: `W · x (+ b)`.
struct Dense {
    w: WeightMatrix,
    b: Option<Vec<f32>>,
}

impl Dense {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.w.apply(x);
        if let Some(b) = &self.b {
            for (o, bv) in out.iter_mut().zip(b.iter()) {
                *o += bv;
            }
        }
        out
    }
}

/// The classification head a reranker checkpoint carries on top of the
/// encoder.
pub struct RankHead {
    /// `cls` + `cls.bias`, followed by `tanh`. Optional upstream: some
    /// checkpoints project straight to the output.
    dense: Option<Dense>,
    /// `cls.norm`, applied after the `tanh`. Upstream calls
    /// `build_norm(cur, cls_norm, NULL, LLM_NORM, -1)` — LayerNorm with
    /// a weight and **no bias**, so a `cls.norm.bias` in a checkpoint
    /// would be a tensor this graph does not apply and is refused.
    norm: Option<Vec<f32>>,
    /// `cls.output` + `cls.output.bias`, the projection to `n_cls_out`.
    out: Option<Dense>,
    /// `{arch}.classifier.output_labels`, verbatim. Empty only when the
    /// head has exactly one output, where a name adds nothing and
    /// upstream's own default (`hparams.n_cls_out = 1`) is unambiguous.
    labels: Vec<String>,
    eps: f32,
}

impl RankHead {
    /// How many floats [`Self::apply`] returns.
    pub fn n_cls_out(&self) -> usize {
        self.labels.len().max(1)
    }

    /// `{arch}.classifier.output_labels`, in output order. Empty for a
    /// single-output head that named none.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Every output of the head, for `pooled` — which must already be
    /// the CLS row, not the whole hidden-state matrix.
    pub fn apply(&self, pooled: &[f32]) -> Vec<f32> {
        let mut cur = pooled.to_vec();
        if let Some(dense) = &self.dense {
            cur = dense.apply(&cur);
            for v in cur.iter_mut() {
                *v = v.tanh();
            }
            if let Some(w) = &self.norm {
                // No bias: `build_norm(..., NULL, ...)` upstream.
                let zeros = vec![0.0f32; w.len()];
                cur = layer_norm(&cur, w, &zeros, self.eps);
            }
        }
        if let Some(out) = &self.out {
            cur = out.apply(&cur);
        }
        cur
    }

    /// The single relevance score `/v1/rerank` reports: output 0, which
    /// is what upstream's `send_rerank` sends (`res->score = embd[0]`).
    pub fn score(&self, pooled: &[f32]) -> f32 {
        self.apply(pooled).first().copied().unwrap_or(0.0)
    }
}

fn refuse(what: String) -> LoadError {
    LoadError::UnsupportedFeature("bert".to_string(), what)
}

/// Reads `{arch}.classifier.output_labels`, an array of strings.
///
/// Absent is `Ok(vec![])`, which is only *allowed* for a one-output
/// head — [`load_rank_head`] enforces that. A key of the wrong type is
/// an error rather than a silent empty: a checkpoint that says
/// something about its labels and is not understood must stop the load.
fn read_labels(file: &impl TensorSource, arch: &str) -> Result<Vec<String>, LoadError> {
    let key = format!("{arch}.classifier.output_labels");
    let Some(value) = file.metadata(&key) else {
        return Ok(Vec::new());
    };
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| match v {
                GgufValue::String(s) => Ok(s.clone()),
                other => Err(refuse(format!(
                    "{key} contains a non-string entry {other:?}; it must be an array of \
                     label names"
                ))),
            })
            .collect(),
        other => Err(refuse(format!(
            "{key} is {other:?}, but it must be an array of strings"
        ))),
    }
}

/// Builds the head a `bert` checkpoint carries, or `None` when it
/// carries none (a plain embedding model).
///
/// Refuses, rather than loading something that would score wrongly:
///
/// * a `cls.norm.bias`, which upstream never applies;
/// * labels whose count disagrees with `cls.output.weight`'s row count,
///   because then one of the two is describing a different model;
/// * a multi-output head with no labels, because
///   [`RankHead::score`] would silently pick output 0 out of several
///   that nothing has named.
pub fn load_rank_head(
    file: &ShardedGguf,
    arch: &str,
    n_embd: usize,
    eps: f32,
) -> Result<Option<RankHead>, LoadError> {
    let has_any = [CLS_W, CLS_OUT_W, CLS_NORM_W]
        .iter()
        .any(|n| file.find_tensor(n).is_some());
    let labels = read_labels(file, arch)?;
    if !has_any {
        // Labels without a head is a checkpoint describing a classifier
        // whose weights are not here. Loading it as a plain embedding
        // model would quietly drop what the file says it is.
        if !labels.is_empty() {
            return Err(refuse(format!(
                "{arch}.classifier.output_labels names {} label(s) ({}) but the checkpoint \
                 carries no {CLS_W}, {CLS_OUT_W} or {CLS_NORM_W} — the classification head \
                 the labels describe is not in this file",
                labels.len(),
                labels.join(", "),
            )));
        }
        return Ok(None);
    }

    let dense = match file.find_tensor(CLS_W) {
        Some(_) => {
            let w = load_weight_matrix(file, CLS_W)?;
            if w.cols() != n_embd {
                return Err(refuse(format!(
                    "{CLS_W} takes {} inputs but the encoder is {n_embd} wide",
                    w.cols()
                )));
            }
            Some(Dense {
                b: load_f32_vec_optional(file, CLS_B)?,
                w,
            })
        }
        None => None,
    };

    if file.find_tensor(CLS_NORM_B).is_some() {
        return Err(refuse(format!(
            "checkpoint carries {CLS_NORM_B}, but upstream's head norm is \
             build_norm(cur, cls_norm, NULL, ...) — a weight and no bias. Applying the \
             weight and dropping the bias would score wrongly and silently"
        )));
    }
    let norm = load_f32_vec_optional(file, CLS_NORM_W)?;
    if norm.is_some() && dense.is_none() {
        return Err(refuse(format!(
            "checkpoint carries {CLS_NORM_W} but no {CLS_W}; upstream applies the head norm \
             only inside the `cls` branch, so this norm would never run"
        )));
    }

    let out = match file.find_tensor(CLS_OUT_W) {
        Some(_) => {
            let w = load_weight_matrix(file, CLS_OUT_W)?;
            let want = dense.as_ref().map(|d| d.w.rows()).unwrap_or(n_embd);
            if w.cols() != want {
                return Err(refuse(format!(
                    "{CLS_OUT_W} takes {} inputs but the value reaching it is {want} wide",
                    w.cols()
                )));
            }
            Some(Dense {
                b: load_f32_vec_optional(file, CLS_OUT_B)?,
                w,
            })
        }
        None => None,
    };

    // How many scores this head really produces, from the weights.
    let n_out = match &out {
        Some(d) => d.w.rows(),
        None => match &dense {
            Some(d) => d.w.rows(),
            // Unreachable while `has_any` is true and norm-without-dense
            // is refused, but stated rather than assumed.
            None => n_embd,
        },
    };
    if !labels.is_empty() && labels.len() != n_out {
        return Err(refuse(format!(
            "{arch}.classifier.output_labels names {} label(s) ({}) but the head produces \
             {n_out} output(s) — the metadata and the weights describe different models",
            labels.len(),
            labels.join(", "),
        )));
    }
    if labels.is_empty() && n_out > 1 {
        return Err(refuse(format!(
            "this classification head produces {n_out} outputs and the checkpoint carries no \
             {arch}.classifier.output_labels, so nothing says which one is the relevance \
             score. Refusing rather than reporting output 0 as if it were named"
        )));
    }

    Ok(Some(RankHead {
        dense,
        norm,
        out,
        labels,
        eps,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;

    fn dense(rows: usize, cols: usize, fill: f32, bias: Option<f32>) -> Dense {
        let data: Vec<f32> = (0..rows * cols).map(|i| fill * (i as f32 + 1.0)).collect();
        Dense {
            w: WeightMatrix::F32(Tensor::new(data, vec![rows, cols])),
            b: bias.map(|b| vec![b; rows]),
        }
    }

    /// The head's ORDER is the whole content of it: dense, tanh, norm,
    /// output. Applying the same pieces in another order still returns
    /// a plausible float, which is why this is checked against a
    /// hand-computed value rather than against "it ran".
    #[test]
    fn the_head_applies_dense_then_tanh_then_output() {
        let head = RankHead {
            // 1x2, weights [1, 2], bias 0 -> 1*x0 + 2*x1
            dense: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![1.0, 2.0], vec![1, 2])),
                b: Some(vec![0.5]),
            }),
            norm: None,
            // 1x1, weight [3], bias -1 -> 3*y - 1
            out: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![3.0], vec![1, 1])),
                b: Some(vec![-1.0]),
            }),
            labels: vec![],
            eps: 1e-12,
        };
        // dense: 1*0.25 + 2*0.5 + 0.5 = 1.75; tanh(1.75) = 0.94138...
        // out:   3*0.94138 - 1 = 1.82414...
        let want = 3.0 * 1.75f32.tanh() - 1.0;
        let got = head.score(&[0.25, 0.5]);
        assert!(
            (got - want).abs() < 1e-6,
            "head produced {got}, hand-computed {want}"
        );
        assert_eq!(head.n_cls_out(), 1);
    }

    /// A head with no `cls` is the direct-projection shape upstream
    /// documents for `jina-reranker-v1-tiny-en`: no tanh anywhere.
    #[test]
    fn a_head_with_no_dense_does_not_apply_a_tanh() {
        let head = RankHead {
            dense: None,
            norm: None,
            out: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![2.0, 0.0], vec![1, 2])),
                b: None,
            }),
            labels: vec![],
            eps: 1e-12,
        };
        // 2 * 5.0 = 10.0. A stray tanh would make this 1.0.
        assert!((head.score(&[5.0, 1.0]) - 10.0).abs() < 1e-6);
    }

    /// `n_cls_out` follows the labels, and `score` is output 0 of
    /// however many there are -- upstream's `embd[0]`.
    #[test]
    fn score_is_the_first_output_and_labels_size_the_head() {
        let head = RankHead {
            dense: None,
            norm: None,
            out: Some(dense(3, 2, 1.0, None)),
            labels: vec!["a".into(), "b".into(), "c".into()],
            eps: 1e-12,
        };
        assert_eq!(head.n_cls_out(), 3);
        assert_eq!(head.labels(), ["a", "b", "c"]);
        let all = head.apply(&[1.0, 1.0]);
        assert_eq!(all.len(), 3);
        assert_eq!(head.score(&[1.0, 1.0]), all[0]);
        assert_ne!(all[0], all[1], "the fixture must distinguish the outputs");
    }
}
