//! The ring buffer behind `/admin/stats`.
//!
//! Keyed by the `request_id` the server already assigns and already
//! states on the wire, so a UI joins a log row to the message that
//! produced it by equality rather than by the claiming heuristic
//! `docs/plans/ferrox-ui.md` describes one reference product resorting
//! to when concurrent chats stole each other's numbers.
//!
//! `duration_ms` and `decode_ms` stay separate here and all the way out
//! to the wire. `duration_ms` is the whole server-side request: queue
//! wait, prefill and decode. `decode_ms` is the decode loop alone.
//! Dividing completion tokens by the former reports a 50 tok/s model as
//! 5 whenever the prompt is long, and every number computed from that
//! is then wrong in the same direction -- which is exactly why the plan
//! calls conflating them out by name.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use ferrox_api::RecentRequest;

/// Requests remembered. The contract says the last 200.
pub(crate) const RING_CAPACITY: usize = 200;

/// Everything `/admin/stats` reports that is not already an
/// `AppState` counter.
#[derive(Default)]
pub(crate) struct Stats {
    recent: Mutex<VecDeque<RecentRequest>>,
    tokens_prompt_total: AtomicU64,
    tokens_generated_total: AtomicU64,
}

impl Stats {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records one finished request. Called after the response has been
    /// produced, from whichever path knows the real token counts -- the
    /// streaming path records when its generation task ends, not when
    /// the SSE handler returns, because the handler returns before a
    /// single token exists.
    pub(crate) fn record(&self, entry: RecentRequest) {
        self.tokens_prompt_total
            .fetch_add(entry.prompt_tokens as u64, Ordering::Relaxed);
        self.tokens_generated_total
            .fetch_add(entry.completion_tokens as u64, Ordering::Relaxed);
        let mut ring = self.recent.lock().unwrap_or_else(|p| p.into_inner());
        if ring.len() == RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(entry);
    }

    pub(crate) fn recent(&self) -> Vec<RecentRequest> {
        self.recent
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub(crate) fn tokens_prompt_total(&self) -> u64 {
        self.tokens_prompt_total.load(Ordering::Relaxed)
    }

    pub(crate) fn tokens_generated_total(&self) -> u64 {
        self.tokens_generated_total.load(Ordering::Relaxed)
    }
}

/// Builds one ring-buffer entry from what a finished request knows.
///
/// Takes the two durations separately and never derives one from the
/// other; a caller that cannot time the decode loop passes `None`
/// rather than reusing `duration_ms`.
pub(crate) fn entry(
    request_id: &str,
    route: &str,
    status: u16,
    stream: bool,
    duration_ms: u64,
    usage: Option<&ferrox_api::Usage>,
) -> RecentRequest {
    RecentRequest {
        request_id: request_id.to_string(),
        at_ms: crate::tasks::now_ms(),
        route: route.to_string(),
        status,
        prompt_tokens: usage.map(|u| u.prompt_tokens).unwrap_or(0),
        completion_tokens: usage.map(|u| u.completion_tokens).unwrap_or(0),
        ttft_ms: usage.and_then(|u| u.time_to_first_token_ms),
        duration_ms,
        decode_ms: usage.and_then(|u| u.generation_duration_ms),
        stream,
        acceptance_length: usage.and_then(|u| u.acceptance_length),
        draft_accept_rate_per_position: usage
            .and_then(|u| u.draft_accept_rate_per_position.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage() -> ferrox_api::Usage {
        ferrox_api::Usage::new(100, 10)
            .with_timings(1.0, 0.1)
            .with_ttft(0.9)
    }

    #[test]
    fn an_entry_keeps_the_two_durations_apart() {
        let e = entry(
            "chatcmpl-1",
            ferrox_api::routes::V1_CHAT_COMPLETIONS,
            200,
            true,
            1_100,
            Some(&usage()),
        );
        assert_eq!(e.duration_ms, 1_100);
        assert_eq!(e.decode_ms, Some(100.0));
        assert_eq!(e.ttft_ms, Some(900.0));
        assert_eq!(e.prompt_tokens, 100);
        assert_eq!(e.completion_tokens, 10);
    }

    #[test]
    fn an_untimed_request_reports_null_rather_than_reusing_the_total() {
        let e = entry("chatcmpl-2", "/v1/completions", 200, false, 42, None);
        assert_eq!(e.decode_ms, None);
        assert_eq!(e.ttft_ms, None);
        assert_eq!(e.duration_ms, 42);
        assert_eq!(e.prompt_tokens, 0);
    }

    #[test]
    fn speculation_metrics_reach_the_admin_ring() {
        // /admin/stats is where a drafter's behaviour over many
        // requests is visible; per-position accept rates only mean
        // something in aggregate, so they have to survive the hop from
        // the response body into the ring.
        let usage = ferrox_api::Usage::new(100, 12)
            .with_timings(1.0, 0.1)
            .with_speculation(5, 7, 10, vec![0.95, 0.7, 0.4]);
        let e = entry(
            "chatcmpl-3",
            ferrox_api::routes::V1_CHAT_COMPLETIONS,
            200,
            false,
            1_100,
            Some(&usage),
        );
        assert_eq!(e.acceptance_length, Some(2.4));
        assert_eq!(
            e.draft_accept_rate_per_position,
            Some(vec![0.95, 0.7, 0.4]),
            "the per-position curve is the only way suffix decay shows up"
        );
    }

    #[test]
    fn a_non_speculative_request_leaves_the_acceptance_columns_empty() {
        let e = entry(
            "chatcmpl-4",
            ferrox_api::routes::V1_CHAT_COMPLETIONS,
            200,
            false,
            42,
            Some(&usage()),
        );
        assert_eq!(e.acceptance_length, None);
        assert_eq!(e.draft_accept_rate_per_position, None);
    }

    #[test]
    fn token_totals_accumulate_across_requests() {
        let stats = Stats::new();
        for _ in 0..3 {
            stats.record(entry("id", "/r", 200, false, 1, Some(&usage())));
        }
        assert_eq!(stats.tokens_prompt_total(), 300);
        assert_eq!(stats.tokens_generated_total(), 30);
    }

    #[test]
    fn the_ring_keeps_the_newest_entries_and_drops_the_oldest() {
        let stats = Stats::new();
        for i in 0..RING_CAPACITY + 25 {
            stats.record(entry(&format!("id-{i}"), "/r", 200, false, 1, None));
        }
        let recent = stats.recent();
        assert_eq!(recent.len(), RING_CAPACITY);
        assert_eq!(recent[0].request_id, "id-25");
        assert_eq!(
            recent[RING_CAPACITY - 1].request_id,
            format!("id-{}", RING_CAPACITY + 24)
        );
    }

    #[test]
    fn an_error_response_is_recorded_with_its_status() {
        let stats = Stats::new();
        stats.record(entry("id-e", "/v1/chat/completions", 503, false, 3, None));
        assert_eq!(stats.recent()[0].status, 503);
    }
}
