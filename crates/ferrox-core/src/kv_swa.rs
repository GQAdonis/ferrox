//! Block-size alignment for sliding-window attention (SWA).
//!
//! The block cache slices a sequence into fixed runs of `block_size`
//! token positions and treats each run as an independently storable,
//! evictable, restorable unit. That works without further thought for
//! full causal attention: block `b` covers positions
//! `[b*B, (b+1)*B)`, restoring blocks `0..m` reproduces positions
//! `[0, m*B)`, and nothing about position `p`'s KV state depends on how
//! the range was cut.
//!
//! **SWA breaks that, and it breaks it silently.** A sliding layer only
//! ever needs -- and, once the KV is capped rather than kept whole,
//! only ever *holds* -- the most recent `window` positions. Dropping
//! stale positions has to happen in whole blocks, because a block is
//! the eviction unit. So the resident set is a whole number of blocks,
//! and the only way a whole number of blocks can be exactly the last
//! `window` positions is:
//!
//! ```text
//! window % block_size == 0
//! ```
//!
//! When it does not divide, the boundary between "still inside the
//! window" and "safe to drop" falls in the middle of a block, and there
//! are only two things an implementation can do with that block: keep
//! it (so the window is silently *wider* than the model's, changing the
//! attention mask) or drop it (so positions the model must attend to
//! are silently *missing*). Neither errors. Both produce confident
//! wrong tokens, which is the same failure class
//! [`kv_signature`](crate::kv_signature) exists to prevent -- so it is
//! prevented the same way: refuse, naming both numbers.
//!
//! > Wording note, for anyone holding `docs/plans/serving-and-tiered-kv.md`
//! > open: the plan states this as "block size must be a multiple of the
//! > sliding-window size". That is the same constraint with the operands
//! > the other way round, and this direction is the one that is
//! > implementable -- forcing `block_size` up to a multiple of a 128-token
//! > gpt-oss window would make every block at least a whole window, which
//! > defeats the point of blocks. vLLM states it as
//! > `sliding_window % block_size == 0`; so does this module.
//!
//! # Why this is live, not hypothetical
//!
//! ferrox runs a real alternating-SWA gpt-oss graph (window 128, every
//! other layer) and Gemma-3 SWA prefill (window 512, every 6th layer
//! full-attention). An alternating model does not weaken the rule: the
//! full-attention layers impose no constraint, and one mis-aligned
//! sliding layer is enough to corrupt the answer. And the disk tier
//! ([`kv_disk`](crate::kv_disk)) makes a mis-aligned block *durable* --
//! it would outlive the process that created it and be handed to the
//! next one. That is why [`BlockLayout`] is carried in the cache
//! signature rather than checked once at startup: a block written under
//! one window is refused by a build expecting another, instead of being
//! read back as if the window had never changed.

/// Why a block layout was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockLayoutError {
    /// A zero-token block is not a unit of anything.
    ZeroBlockSize,
    /// `Some(0)` is not "no window": a zero window would mean a query
    /// attends to nothing, which no model does. A model with no
    /// sliding-window attention must say `None`.
    ZeroWindow,
    /// The eviction boundary would fall inside a block. See the module
    /// note for what the two possible responses to that both corrupt.
    Misaligned {
        window: usize,
        block_size: usize,
        remainder: usize,
    },
}

impl std::fmt::Display for BlockLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockLayoutError::ZeroBlockSize => write!(f, "KV block size must be positive"),
            BlockLayoutError::ZeroWindow => write!(
                f,
                "sliding window must be positive; a model without SWA has no window at all"
            ),
            BlockLayoutError::Misaligned {
                window,
                block_size,
                remainder,
            } => write!(
                f,
                "KV block size {block_size} does not divide the sliding window {window} \
                 ({window} % {block_size} = {remainder}); a block would straddle the \
                 window boundary and be either kept too long or dropped too early"
            ),
        }
    }
}

impl std::error::Error for BlockLayoutError {}

/// How a sequence is cut into cache blocks, and the sliding window (if
/// any) those blocks must line up with.
///
/// Only constructible through [`BlockLayout::new`], so a `BlockLayout`
/// value is itself the proof that the alignment rule holds -- there is
/// no path that produces a mis-aligned one to be checked later and
/// forgotten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    block_size: usize,
    sliding_window: Option<usize>,
}

impl BlockLayout {
    /// Validates and builds a layout. `sliding_window` is `None` for a
    /// full-causal model and `Some(w)` for one whose sliding layers use
    /// window `w` -- including alternating models, where some layers
    /// are full-attention: those layers are unconstrained, so the
    /// model's constraint is the window of the layers that have one.
    pub fn new(block_size: usize, sliding_window: Option<usize>) -> Result<Self, BlockLayoutError> {
        if block_size == 0 {
            return Err(BlockLayoutError::ZeroBlockSize);
        }
        match sliding_window {
            None => Ok(BlockLayout {
                block_size,
                sliding_window: None,
            }),
            Some(0) => Err(BlockLayoutError::ZeroWindow),
            Some(window) => {
                let remainder = window % block_size;
                if remainder != 0 {
                    return Err(BlockLayoutError::Misaligned {
                        window,
                        block_size,
                        remainder,
                    });
                }
                Ok(BlockLayout {
                    block_size,
                    sliding_window: Some(window),
                })
            }
        }
    }

    /// A full-causal model's layout. Cannot fail except on a zero block
    /// size.
    pub fn full_attention(block_size: usize) -> Result<Self, BlockLayoutError> {
        Self::new(block_size, None)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// How many whole blocks a sliding layer keeps resident. `None` for
    /// a full-causal model, which keeps all of them.
    ///
    /// Exact by construction: the layout would not exist if the window
    /// were not a whole number of blocks.
    pub fn blocks_per_window(&self) -> Option<usize> {
        self.sliding_window.map(|w| w / self.block_size)
    }
}

/// The largest block size no greater than `desired` that satisfies the
/// rule for `window`.
///
/// This is what a *configuration* layer should call: an operator asking
/// for 256-token blocks on a 128-token-window gpt-oss should get 128,
/// not an error and not silent corruption. It never rounds *up*, since
/// a block larger than asked for costs more memory per eviction step
/// than the operator budgeted for; and it never returns 0.
///
/// `window == None` returns `desired` unchanged. `desired == 0` is
/// treated as 1, because there is no smaller honest answer.
///
/// The search walks down from `desired`; block sizes are small (tens to
/// low hundreds of tokens), so this is cheaper than factoring the
/// window and is called once per model load.
pub fn aligned_block_size(desired: usize, window: Option<usize>) -> usize {
    let desired = desired.max(1);
    let Some(window) = window.filter(|w| *w > 0) else {
        return desired;
    };
    (1..=desired.min(window))
        .rev()
        .find(|candidate| window.is_multiple_of(*candidate))
        // 1 divides every positive window, so the iterator is never
        // empty -- but expressing the fallback beats an unwrap.
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the module. gpt-oss's real window is 128; a
    /// 48-token block does not divide it, and the two things an
    /// implementation could do with the straddling block are both
    /// wrong-answer bugs, not misses.
    #[test]
    fn a_block_size_that_does_not_divide_the_window_is_refused() {
        let err = BlockLayout::new(48, Some(128)).expect_err("48 does not divide 128");
        assert_eq!(
            err,
            BlockLayoutError::Misaligned {
                window: 128,
                block_size: 48,
                remainder: 32,
            }
        );
        // The operator has to be able to fix it from the message alone.
        let text = err.to_string();
        assert!(text.contains("48"), "{text}");
        assert!(text.contains("128"), "{text}");
    }

    #[test]
    fn a_block_size_that_divides_the_window_is_accepted() {
        let layout = BlockLayout::new(32, Some(128)).expect("32 divides 128");
        assert_eq!(layout.block_size(), 32);
        assert_eq!(layout.sliding_window(), Some(128));
        assert_eq!(layout.blocks_per_window(), Some(4));
    }

    /// A block larger than the whole window is the most obvious form of
    /// the bug and must not slip through as "well, it covers it".
    #[test]
    fn a_block_larger_than_the_window_is_refused() {
        assert!(matches!(
            BlockLayout::new(256, Some(128)),
            Err(BlockLayoutError::Misaligned { .. })
        ));
        // ... unless it is exactly the window, which is aligned.
        assert!(BlockLayout::new(128, Some(128)).is_ok());
    }

    #[test]
    fn a_full_causal_model_constrains_nothing() {
        let layout = BlockLayout::full_attention(48).expect("no window, no constraint");
        assert_eq!(layout.sliding_window(), None);
        assert_eq!(layout.blocks_per_window(), None);
    }

    /// `Some(0)` and `None` are different claims and only one of them
    /// is "this model has no SWA".
    #[test]
    fn a_zero_window_is_not_the_same_as_no_window() {
        assert_eq!(
            BlockLayout::new(16, Some(0)),
            Err(BlockLayoutError::ZeroWindow)
        );
        assert!(BlockLayout::new(16, None).is_ok());
    }

    #[test]
    fn a_zero_block_size_is_refused_with_or_without_a_window() {
        assert_eq!(
            BlockLayout::new(0, None),
            Err(BlockLayoutError::ZeroBlockSize)
        );
        assert_eq!(
            BlockLayout::new(0, Some(128)),
            Err(BlockLayoutError::ZeroBlockSize)
        );
    }

    /// Whatever `aligned_block_size` returns must be constructible --
    /// otherwise the config layer hands the cache a size the cache
    /// rejects, which is a startup crash rather than a fix.
    #[test]
    fn the_aligned_size_is_always_a_size_the_layout_accepts() {
        for window in [1usize, 2, 3, 128, 512, 1024, 4096, 4099] {
            for desired in [1usize, 7, 16, 31, 32, 100, 128, 256, 5000] {
                let size = aligned_block_size(desired, Some(window));
                assert!(size > 0 && size <= desired, "{desired}/{window} -> {size}");
                BlockLayout::new(size, Some(window)).unwrap_or_else(|e| {
                    panic!("aligned_block_size({desired}, {window}) = {size} is not valid: {e}")
                });
            }
        }
    }

    #[test]
    fn the_aligned_size_rounds_down_never_up() {
        // gpt-oss: a 256-token request on a 128 window becomes 128, not
        // 256 and not 384.
        assert_eq!(aligned_block_size(256, Some(128)), 128);
        // Gemma-3: 512-token window, 100 requested -> 64, the largest
        // divisor at or below 100.
        assert_eq!(aligned_block_size(100, Some(512)), 64);
        // Already aligned: untouched.
        assert_eq!(aligned_block_size(64, Some(512)), 64);
        // A prime window leaves only 1 below itself.
        assert_eq!(aligned_block_size(100, Some(4099)), 1);
    }

    #[test]
    fn no_window_leaves_the_desired_size_alone() {
        assert_eq!(aligned_block_size(48, None), 48);
        assert_eq!(aligned_block_size(0, None), 1);
    }
}
