//! What the batched path owes the radix prefix cache.
//!
//! Its own group, because the thing under test is neither admission nor
//! the tick: it is the CONTRACT between a finished row and the tree --
//! publish what you computed, adopt what someone else did, and give the
//! pages back when the pool runs dry. All three have to hold together
//! or the cache is worse than not having one.
//!
//! The bug these were written against: the batched path ADOPTED from
//! the tree and never contributed to it, because the prompt ids were
//! dropped at the prefill-to-decode handover and publishing needs the
//! whole sequence. Prefix sharing under `FERROX_CONTINUOUS_BATCHING=1`
//! therefore ran against a tree nothing filled.

use std::sync::mpsc;
use std::sync::Mutex;

use ferrox_core::cache::{PageGroup, SharedPagedKv};

use crate::policy::radix::RadixCache;

use super::super::row::{Job, RowKv, Rows};
use super::super::worker::accept;
use super::*;

const BLOCK: usize = 4;

/// A paged config over a store of `groups` page groups, with a fresh
/// radix tree the test can inspect.
fn paged_with_radix(
    decoder: &Arc<Decoder>,
    groups: usize,
) -> (PagedKvConfig, Arc<SharedPagedKv>, Arc<Mutex<RadixCache>>) {
    let radix = Arc::new(Mutex::new(RadixCache::new(BLOCK)));
    let store = Arc::new(SharedPagedKv::new(
        decoder.layers.len(),
        BLOCK,
        /* blocks_per_layer = */ groups,
        decoder.config.n_kv_heads,
        decoder.config.head_dim,
    ));
    let config = PagedKvConfig {
        store: Arc::clone(&store),
        queue_wait: std::time::Duration::ZERO,
        radix: Some(Arc::clone(&radix)),
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    (config, store, radix)
}

fn paged_job(prompt: Vec<usize>, max_tokens: usize) -> (Job, mpsc::Receiver<BatcherEvent>) {
    let (tx, rx) = mpsc::channel();
    (
        Job {
            prompt_tokens: prompt,
            params: greedy_params(max_tokens, 7),
            stop_tokens: StopTokens::default(),
            reply: tx,
            abort: AbortId(0),
            blocks: 1,
        },
        rx,
    )
}

/// Runs one request all the way to an admitted, prefilled row.
///
/// Deliberately the REAL path -- `accept` acquires the lease (and so
/// consults the tree), `step_chunk` runs every prompt token, and
/// `into_slot` performs the prefill-to-decode handover that used to
/// drop the prompt ids. A test that built a `Slot` by hand would prove
/// nothing about the handover, which is where the bug lived.
fn admit_prefilled(
    decoder: &Arc<Decoder>,
    config: &PagedKvConfig,
    prompt: Vec<usize>,
    max_tokens: usize,
) -> Option<(Slot, mpsc::Receiver<BatcherEvent>)> {
    let (job, rx) = paged_job(prompt, max_tokens);
    let mut prefill = accept(decoder, job, /* chunk_size = */ 1, Some(config))?;
    while !prefill.state.step_chunk() {}
    Some((prefill.into_slot(), rx))
}

/// Finishes a row through the one path every finished row takes, which
/// is also the only place the batched publish happens.
fn finish_row(slot: Slot) {
    let mut rows = Rows::default();
    let uid = rows.insert(slot);
    rows.get_mut(uid).expect("just inserted").finish = Some(FinishReason::Length);
    rows.flush_finished(&no_budget());
}

/// The page groups the tree holds for `prompt`, one per block.
fn published_groups(radix: &Arc<Mutex<RadixCache>>, prompt: &[usize]) -> (usize, Vec<u32>) {
    let ids: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let mut tree = radix.lock().unwrap();
    let m = tree.match_prefix(&ids);
    if m.cached_len == 0 {
        return (0, Vec::new());
    }
    let per_token = tree.matched_indices(m.node);
    (
        m.cached_len,
        per_token[..m.cached_len]
            .iter()
            .copied()
            .step_by(BLOCK)
            .collect(),
    )
}

/// A finished batched row must PUBLISH its prefix, not just adopt one.
///
/// Asserts the TREE grew, which is the thing that was missing. A test
/// that only checked the second request was fast would pass on a warm
/// page cache.
#[test]
fn a_finished_batched_row_publishes_its_prefix() {
    let decoder = tiny_decoder();
    let (config, _store, radix) = paged_with_radix(&decoder, 64);
    assert_eq!(
        radix.lock().unwrap().total_size(),
        0,
        "the tree starts empty"
    );

    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
    let (slot, _rx) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(slot);

    let (cached, groups) = published_groups(&radix, &prompt);
    assert_eq!(
        cached,
        prompt.len(),
        "a finished paged row must leave its whole prefix in the tree"
    );
    assert_eq!(groups.len(), prompt.len() / BLOCK);
}

/// The point of publishing: the NEXT request adopts those very pages.
///
/// Page IDENTITY, not merely a hit count. The assertion that carries
/// the test is the refcount: while the second row is alive, each group
/// the first published has two holders -- the tree and the second row
/// -- which is only true if the second row is attending over the first
/// row's actual pages rather than fresh copies of them.
#[test]
fn a_second_batched_request_adopts_the_pages_the_first_published() {
    let decoder = tiny_decoder();
    let (config, store, radix) = paged_with_radix(&decoder, 64);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    let (first, _rx1) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(first);

    let (cached, published) = published_groups(&radix, &prompt);
    assert_eq!(cached, prompt.len(), "the first row must have published");
    for &g in &published {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            1,
            "with no row alive, only the tree holds a published page"
        );
    }

    let (second, _rx2) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    let RowKv::Paged(lease) = &second.kv else {
        panic!("a paged config must produce a paged row");
    };
    // Never the whole prompt: prefill has to run over at least one
    // token to produce the logits that predict the next one.
    let adopted = lease.adopted_positions(BLOCK);
    assert!(
        adopted > 0,
        "the second request must adopt the prefix the first published, \
         got {adopted} adopted positions"
    );

    for &g in published.iter().take(adopted / BLOCK) {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            2,
            "an adopted page is held by the tree AND by the row reading it"
        );
    }
    drop(second);
    for &g in &published {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            1,
            "the row's hold goes back on Drop; the tree's survives"
        );
    }
}

/// Publishing must not make the pool shrink monotonically.
///
/// `publish_to_radix` retains a group for every page it hands the tree.
/// Nothing released them, so a long-running server ended up refusing
/// requests that fit while the tree sat on pages no request was
/// reading. The batched path acquires through the same
/// `acquire_paged_caches` as the private one, so the eviction wired
/// into its retry loop has to cover batched rows too.
///
/// Sized so the arithmetic is the assertion: the pool holds 8 groups,
/// each request needs 3 and leaves 2 in the tree, so a tree that is
/// never evicted from starves the fourth request. Twelve run here.
#[test]
fn batched_publishing_does_not_starve_the_page_pool() {
    let decoder = tiny_decoder();
    let (config, store, radix) = paged_with_radix(&decoder, 8);
    let free_at_start = store.free_groups();
    assert_eq!(free_at_start, 8);

    for cycle in 0..12u32 {
        // Distinct at the very first token, so no cycle adopts another
        // one's prefix: every pass must pay for its own pages, which is
        // what puts the pool under pressure.
        let prompt: Vec<usize> = std::iter::once(10 + cycle as usize % 20)
            .chain([1, 2, 3, 4, 5, 6, 7])
            .collect();
        let (slot, _rx) = admit_prefilled(&decoder, &config, prompt, 4).unwrap_or_else(|| {
            panic!(
                "cycle {cycle} was refused: the tree is holding pages it will \
                 not give back, so the pool shrank monotonically \
                 (free = {}, tree = {})",
                store.free_groups(),
                radix.lock().unwrap().total_size()
            )
        });
        finish_row(slot);
    }

    assert!(
        store.free_groups() > 0,
        "every page ended up parked in the tree"
    );
    // The pages the tree keeps are the whole point, so the pool does
    // NOT return to its starting size -- it just must not reach zero.
    assert!(store.free_groups() <= free_at_start);
}
