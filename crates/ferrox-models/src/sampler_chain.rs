//! llama.cpp's sampler chain, as the shrinking candidate list it
//! actually is.
//!
//! Every sampler in `src/llama-sampler.cpp` takes one
//! `llama_token_data_array` -- a list of `(token id, logit, p)` -- and
//! **removes entries from it**. The next sampler in the chain then sees
//! only the survivors, and any sampler that needs probabilities calls
//! `llama_sampler_softmax_impl` (`:293`), which renormalises **over the
//! survivors only**.
//!
//! That renormalisation is the part a "just zero the logits you don't
//! want" implementation silently gets wrong, and ferrox's did. With
//! `--top-k 40 --top-p 0.95`, llama.cpp's top-p sums probabilities that
//! were divided by the mass of the top 40; ferrox summed probabilities
//! divided by the mass of the **whole vocabulary**, which is larger, so
//! its running sum crossed 0.95 later and it kept MORE tokens than
//! llama.cpp for the same two flags. Same idea as the temperature-order
//! bug: not a reordering of independent steps, a different candidate
//! set.
//!
//! So this module models the candidate list rather than a keep-mask,
//! and each filter here is a transcription of the corresponding
//! `_apply` in llama.cpp, cited at its definition.
//!
//! **`min_keep`.** Several of llama.cpp's filters take a `min_keep`
//! floor and guard their cutoff with `i + 1 >= min_keep`. ferrox has no
//! such parameter (llama.cpp's own CLI does not expose one either --
//! `common/arg.cpp` has no `--min-keep`; only the server's JSON body
//! carries it), and `common_params_sampling::min_keep` defaults to `0`
//! (`common/common.h:228`), which makes every one of those guards
//! vacuously true. They are therefore folded out below rather than
//! carried as a field nothing sets.

/// The candidate list a chain of samplers narrows.
///
/// Held **sorted by descending logit** at all times. llama.cpp tracks a
/// `sorted` flag instead and lets each sampler sort lazily, but every
/// filter that cares either sorts first or only reads the maximum, so
/// maintaining the invariant eagerly reaches the same sets with less
/// state to get wrong.
pub(crate) struct Candidates {
    /// Token id of each live candidate.
    ids: Vec<usize>,
    /// Logit of each live candidate, parallel to `ids`.
    logits: Vec<f32>,
    /// Probability of each live candidate, parallel to `ids`. Only
    /// meaningful immediately after [`Self::softmax`]; the filters that
    /// need it call that themselves, exactly as llama.cpp's do.
    probs: Vec<f32>,
}

impl Candidates {
    /// The whole vocabulary as one candidate list, sorted by descending
    /// logit.
    ///
    /// Ties break on the lower token id so that a run is reproducible
    /// given a seed even when a model emits exactly equal logits, which
    /// `sort_unstable_by` on the logit alone does not guarantee.
    pub(crate) fn new(logits: &[f32]) -> Self {
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let ordered: Vec<f32> = ids.iter().map(|&i| logits[i]).collect();
        let probs = vec![0.0f32; ordered.len()];
        Candidates {
            ids,
            logits: ordered,
            probs,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Keep only the first `n` candidates.
    fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.logits.truncate(n);
        self.probs.truncate(n);
    }

    /// `llama_sampler_softmax_impl` (`src/llama-sampler.cpp:293`):
    /// softmax over the **live candidates only**, so the normaliser is
    /// the mass of whatever survived so far.
    fn softmax(&mut self) {
        if self.logits.is_empty() {
            return;
        }
        // The list is sorted, so `data[0]` is the maximum -- the same
        // shortcut llama.cpp takes when `cur_p->sorted`.
        let max = self.logits[0];
        let mut sum = 0.0f32;
        for (p, &l) in self.probs.iter_mut().zip(self.logits.iter()) {
            *p = (l - max).exp();
            sum += *p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            let uniform = 1.0 / self.probs.len() as f32;
            self.probs.fill(uniform);
            return;
        }
        for p in self.probs.iter_mut() {
            *p /= sum;
        }
    }

    /// `llama_sampler_top_k_impl` (`src/llama-sampler.cpp:321`): keep
    /// the `k` highest logits; `k <= 0` disables the filter.
    pub(crate) fn top_k(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        self.truncate(k.min(self.len()));
    }

    /// `llama_sampler_top_p_apply` (`src/llama-sampler.cpp:1360`).
    ///
    /// The cutoff test is `cum_sum >= p`, and the candidate that crosses
    /// it is **included** (`last_idx = i + 1`). `p >= 1.0` disables.
    pub(crate) fn top_p(&mut self, p: f32) {
        if p >= 1.0 {
            return;
        }
        self.softmax();
        let mut cum_sum = 0.0f32;
        let mut last_idx = self.len();
        for i in 0..self.len() {
            cum_sum += self.probs[i];
            if cum_sum >= p {
                last_idx = i + 1;
                break;
            }
        }
        self.truncate(last_idx);
    }

    /// `llama_sampler_min_p_apply` (`src/llama-sampler.cpp:1556`).
    ///
    /// Keeps every candidate whose probability is at least `p` times the
    /// **top** candidate's, which llama.cpp expresses on the logits
    /// rather than the probabilities:
    ///
    /// ```text
    /// min_logit = data[0].logit + logf(p)   // p_i >= p * p_max
    /// ```
    ///
    /// -- the softmax normaliser cancels out of the ratio, so no
    /// `softmax` call is needed and the answer does not depend on which
    /// filters ran before this one.
    ///
    /// It does depend, absolutely, on the **temperature not having run
    /// yet**: temperature scales every logit, so it scales the gap
    /// `logit_i - logit_max` that is being compared against the fixed
    /// `ln(p)`. Running min-p after temperature would make the surviving
    /// set a function of `--temp`, which in llama.cpp it is not
    /// (`common/common.h:259-269` puts `MIN_P` before `TEMPERATURE`).
    ///
    /// `i` starts at 1 because "the first token always matches": the
    /// list is never emptied, even by a `p > 1.0` that no candidate can
    /// satisfy.
    pub(crate) fn min_p(&mut self, p: f32) {
        if p <= 0.0 || self.logits.is_empty() {
            return;
        }
        let min_logit = self.logits[0] + p.ln();
        let mut i = 1;
        while i < self.len() && self.logits[i] >= min_logit {
            i += 1;
        }
        self.truncate(i);
    }

    /// `llama_sampler_temp_impl` (`src/llama-sampler.cpp:265`) for
    /// `temp > 0`: divide every surviving logit by the temperature.
    ///
    /// LAST in llama.cpp's default chain, after every truncation filter
    /// -- see [`Self::min_p`] and `sampling::filtered_distribution` for
    /// why that is a specification and not a convenience.
    pub(crate) fn temperature(&mut self, temp: f32) {
        if temp <= 0.0 {
            return;
        }
        for l in self.logits.iter_mut() {
            *l /= temp;
        }
    }

    /// The surviving candidates as a full-vocabulary distribution:
    /// softmaxed over the survivors, scattered back to token ids, zero
    /// everywhere the chain filtered out.
    pub(crate) fn into_distribution(mut self, vocab: usize) -> Vec<f32> {
        self.softmax();
        let mut out = vec![0.0f32; vocab];
        for (&id, &p) in self.ids.iter().zip(self.probs.iter()) {
            if let Some(slot) = out.get_mut(id) {
                *slot = p;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// min-p's threshold is llama.cpp's, on the logits:
    /// `data[0].logit + ln(p)`.
    ///
    /// Checked against arithmetic done by hand rather than against the
    /// implementation: logits `[4, 3, 2, 1]` with `p = 0.2` give
    /// `ln(0.2) = -1.6094`, so the threshold is `2.3905` and exactly the
    /// candidates at 4 and 3 clear it.
    #[test]
    fn min_p_keeps_candidates_within_ln_p_of_the_top_logit() {
        let mut c = Candidates::new(&[4.0, 3.0, 2.0, 1.0]);
        c.min_p(0.2);
        assert_eq!(c.ids, vec![0, 1], "threshold is 4 + ln(0.2) = 2.3905");

        // Equality is inclusive (`>=` in llama.cpp's loop guard).
        let mut c = Candidates::new(&[0.0, (0.5f32).ln(), -5.0]);
        c.min_p(0.5);
        assert_eq!(c.ids, vec![0, 1], "p_i == p * p_max must be kept");

        // The first token always matches, so the list is never emptied.
        let mut c = Candidates::new(&[1.0, 0.9, 0.8]);
        c.min_p(2.0);
        assert_eq!(c.ids, vec![0]);

        // 0.0 disables.
        let mut c = Candidates::new(&[4.0, 3.0, 2.0, 1.0]);
        c.min_p(0.0);
        assert_eq!(c.len(), 4);
    }

    /// top-p sums probabilities renormalised over the **survivors of the
    /// earlier filters**, not over the whole vocabulary.
    ///
    /// This is the divergence a keep-mask implementation cannot express.
    /// Logits `[3, 2, 1, 0, 0, 0, 0, 0]`: the top-2 mass is
    /// `e^3 + e^2` and within it token 0 already holds `e^3 / (e^3 +
    /// e^2) = 0.731`, so `--top-k 2 --top-p 0.72` keeps ONE token. Over
    /// the full-vocabulary softmax token 0 holds only `0.578`, so the
    /// same two flags would keep two.
    #[test]
    fn top_p_renormalises_over_what_top_k_left() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut c = Candidates::new(&logits);
        c.top_k(2);
        c.top_p(0.72);
        assert_eq!(
            c.ids,
            vec![0],
            "renormalised over the top 2, token 0 already holds 0.731"
        );

        // Sanity on the other half of the claim: without the top-k the
        // full-vocabulary mass of token 0 is below 0.72, so the same
        // top-p keeps a second candidate. If this ever equals the case
        // above, the test above no longer proves renormalisation.
        let mut c = Candidates::new(&logits);
        c.top_p(0.72);
        assert_eq!(c.ids, vec![0, 1]);
    }

    /// The candidate that crosses the top-p threshold is kept, not
    /// dropped (`last_idx = i + 1`).
    #[test]
    fn top_p_includes_the_candidate_that_crosses_the_threshold() {
        // Two candidates at p = 0.5 each; `cum_sum >= 0.5` fires on the
        // first, which is therefore the last one kept.
        let mut c = Candidates::new(&[1.0f32, 1.0]);
        c.top_p(0.5);
        assert_eq!(c.ids, vec![0]);

        let mut c = Candidates::new(&[1.0f32, 1.0]);
        c.top_p(0.6);
        assert_eq!(c.ids, vec![0, 1]);
    }

    /// Equal logits sort by ascending token id, so a run stays
    /// reproducible given a seed. `sort_unstable_by` on the logit alone
    /// leaves ties in an unspecified order.
    #[test]
    fn equal_logits_break_ties_on_the_token_id() {
        let c = Candidates::new(&[1.0f32; 6]);
        assert_eq!(c.ids, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn the_published_distribution_is_normalised_and_zero_outside_the_survivors() {
        let mut c = Candidates::new(&[3.0f32, 2.0, 1.0, 0.0]);
        c.top_k(2);
        c.temperature(0.5);
        let probs = c.into_distribution(4);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
        // temp 0.5 doubles the logit gap: e^2 / (e^2 + 1) = 0.8808.
        assert!((probs[0] - 0.880_797).abs() < 1e-5, "got {}", probs[0]);
    }
}
