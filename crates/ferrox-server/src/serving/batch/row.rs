//! One in-flight request's state, and the keyed table it lives in.
//!
//! Rows are addressed by [`Uid`], never by batch position. Batch
//! membership changes on almost every tick, and a positional table
//! renumbers its survivors when that happens; a `Uid` captured before a
//! removal still names its own row afterwards, or nothing at all. See
//! the keyed-row-state note in [`super`] for the bug class that avoids.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Sender;

use ferrox_core::cache::KvCache;
use ferrox_models::sampling::Sampler;
use ferrox_models::tokenizer::StopTokens;

use crate::generate::{FinishReason, GenerationParams, PagedLease, Usage};
use crate::stop::StopMatcher;

use super::block_budget::BlockBudget;
use super::config::JobResult;
use super::queue::AbortId;

/// Where one batched row keeps its KV.
///
/// A paged row holds a whole [`PagedLease`], not just its caches: the
/// lease is what returns the row's page groups when the row ends, and
/// every way a row can end -- finished, cancelled, evicted, refused at
/// validation -- goes through dropping it. Splitting the caches from
/// the lease would mean one more path that has to remember to free.
pub(super) enum RowKv {
    Contiguous(Vec<KvCache>),
    Paged(PagedLease),
}

pub(super) struct Job {
    pub(super) prompt_tokens: Vec<usize>,
    pub(super) params: GenerationParams,
    pub(super) stop_tokens: StopTokens,
    pub(super) reply: Sender<JobResult>,
    /// Cancellation handle for this job, from submission onwards.
    pub(super) abort: AbortId,
    /// KV blocks this job will need for its whole lifetime, computed
    /// once by the submitter (which already knows the prompt length and
    /// `max_tokens`) so the worker's admission check is a comparison
    /// rather than arithmetic.
    pub(super) blocks: usize,
}

/// Stable identity for one in-flight request, handed out once at
/// admission and never reused. Unlike a batch index it does not move
/// when another row leaves the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct Uid(u64);

/// The in-flight rows: state keyed by [`Uid`], plus the admission order
/// the batch is built in. Removing a row cannot renumber another one --
/// see the keyed-row-state note in the module docs for what that
/// prevents.
#[derive(Default)]
pub(super) struct Rows {
    pub(super) state: HashMap<Uid, Slot>,
    /// Admission order. Kept explicit so batch composition is
    /// deterministic; `HashMap` iteration order is not.
    pub(super) order: Vec<Uid>,
    pub(super) next_uid: u64,
}

impl Rows {
    pub(super) fn insert(&mut self, slot: Slot) -> Uid {
        let uid = Uid(self.next_uid);
        self.next_uid += 1;
        self.state.insert(uid, slot);
        self.order.push(uid);
        uid
    }

    pub(super) fn len(&self) -> usize {
        self.order.len()
    }

    /// KV blocks the rows in this table are holding right now.
    pub(super) fn blocks_held(&self) -> usize {
        self.state.values().map(|slot| slot.blocks).sum()
    }

    /// Marks any row whose abort id is in `ids` as cancelled, and
    /// reports which ids were consumed.
    ///
    /// Marking, not removing: the row leaves through the same
    /// `flush_finished` path as every other finished row, so its blocks
    /// are released and its caller is replied to exactly once. A second
    /// removal path is a second place to forget one of those.
    pub(super) fn mark_cancelled(&mut self, ids: &HashSet<AbortId>) -> Vec<AbortId> {
        let mut consumed = Vec::new();
        for uid in &self.order {
            if let Some(slot) = self.state.get_mut(uid) {
                if ids.contains(&slot.abort) && slot.finish.is_none() {
                    slot.finish = Some(FinishReason::Cancelled);
                    consumed.push(slot.abort);
                }
            }
        }
        consumed
    }

    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// `None` for a uid that has already left the batch. A stale uid
    /// resolves to nothing -- never to whichever row happens to sit
    /// where it used to.
    pub(super) fn get(&self, uid: Uid) -> Option<&Slot> {
        self.state.get(&uid)
    }

    pub(super) fn get_mut(&mut self, uid: Uid) -> Option<&mut Slot> {
        self.state.get_mut(&uid)
    }

    pub(super) fn remove(&mut self, uid: Uid) -> Option<Slot> {
        self.order.retain(|&u| u != uid);
        self.state.remove(&uid)
    }

    /// Rows that should take a decode step this tick, in admission
    /// order.
    pub(super) fn ready(&self) -> Vec<Uid> {
        self.order
            .iter()
            .copied()
            .filter(|uid| {
                self.state
                    .get(uid)
                    .is_some_and(|s| s.finish.is_none() && s.generated_ids.len() < s.max_tokens)
            })
            .collect()
    }

    /// Replies to and removes every row that has finished, returning
    /// each row's KV blocks to the budget as it goes. Release happens
    /// here, on the one path every finished row takes, so a row cannot
    /// leave the table without giving its capacity back.
    pub(super) fn flush_finished(&mut self, budget: &BlockBudget) {
        let finished: Vec<Uid> = self
            .order
            .iter()
            .copied()
            .filter(|uid| self.state.get(uid).is_some_and(|s| s.finish.is_some()))
            .collect();
        for uid in finished {
            if let Some(mut slot) = self.remove(uid) {
                budget.release(slot.blocks);
                // Publish before the lease drops, so the next request
                // with this prefix can adopt it.
                //
                // The batched path used to ADOPT from the radix tree and
                // never contribute to it, so under continuous batching
                // prefix sharing ran against a tree nothing filled: the
                // first request paid full prefill and so did every one
                // after it. The private generate loop published all
                // along; only the batcher did not, because the prompt
                // ids were dropped at the prefill-to-decode handover.
                if let RowKv::Paged(lease) = &mut slot.kv {
                    let mut seq = std::mem::take(&mut slot.prompt_ids);
                    seq.extend_from_slice(&slot.generated_ids);
                    let bs = lease.block_size();
                    crate::generate::publish_to_radix(lease, &seq, bs);
                }
                reply_finished(slot);
            }
        }
    }
}

pub(super) struct Slot {
    pub(super) kv: RowKv,
    pub(super) pos: usize,
    pub(super) logits: Vec<f32>,
    pub(super) sampler: Sampler,
    pub(super) generated_ids: Vec<usize>,
    /// The prompt this row ran, kept so a finished paged row can
    /// PUBLISH its prefix to the radix tree. Without it the batched
    /// path adopted prefixes and never contributed one.
    pub(super) prompt_ids: Vec<usize>,
    /// Detokenized text already safe to expose (past the stop
    /// hold-back).
    pub(super) visible: String,
    /// Both stop layers for this row.
    pub(super) stops: StopMatcher,
    pub(super) prompt_tokens: usize,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: StopTokens,
    pub(super) params: GenerationParams,
    pub(super) reply: Sender<JobResult>,
    pub(super) finish: Option<FinishReason>,
    pub(super) abort: AbortId,
    /// KV blocks this row reserved at admission, returned when it ends.
    pub(super) blocks: usize,
}

/// Sends one finished row's result to its own waiting caller. Takes the
/// `Slot` by value, so a row's reply channel travels with its state and
/// cannot be paired with another row's output.
pub(super) fn reply_finished(mut slot: Slot) {
    let finish = slot.finish.expect("only a finished row is replied to");
    // Text withheld against a stop that never arrived is ordinary
    // output; dropping it would truncate every answer whose tail looks
    // like the start of a stop string.
    let tail = slot.stops.flush();
    slot.visible.push_str(&tail);
    let usage = Usage::new(slot.prompt_tokens, slot.generated_ids.len());
    let _ = slot
        .reply
        .send(Ok((finish, slot.generated_ids, slot.visible, usage)));
}
