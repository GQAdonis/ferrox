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

use crate::attribution::Attribution;

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

/// Everything a finished request knows about itself, named at the call
/// site.
///
/// A struct rather than a parameter list because the two durations and
/// the two attribution fields are individually easy to swap by accident
/// and impossible to catch by type -- `duration_ms` and a decode time
/// are both `u64`-ish, `via_api_key` and `client` are both
/// `Option<String>`. Naming them at every call site is the cheapest
/// guard there is.
pub(crate) struct Record<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) route: &'a str,
    pub(crate) status: u16,
    pub(crate) stream: bool,
    /// Whole server-side wall time: queue wait, prefill and decode.
    pub(crate) duration_ms: u64,
    /// `None` from a path that cannot time itself. Never a stand-in
    /// built out of `duration_ms`.
    pub(crate) usage: Option<&'a ferrox_api::Usage>,
    pub(crate) attribution: &'a Attribution,
}

/// Builds one ring-buffer entry from what a finished request knows.
///
/// Keeps the two durations separate and never derives one from the
/// other; a caller that cannot time the decode loop passes `usage:
/// None` rather than reusing `duration_ms`.
pub(crate) fn entry(record: Record<'_>) -> RecentRequest {
    let usage = record.usage;
    RecentRequest {
        request_id: record.request_id.to_string(),
        at_ms: crate::tasks::now_ms(),
        route: record.route.to_string(),
        status: record.status,
        prompt_tokens: usage.map(|u| u.prompt_tokens).unwrap_or(0),
        completion_tokens: usage.map(|u| u.completion_tokens).unwrap_or(0),
        ttft_ms: usage.and_then(|u| u.time_to_first_token_ms),
        duration_ms: record.duration_ms,
        decode_ms: usage.and_then(|u| u.generation_duration_ms),
        stream: record.stream,
        via_api_key: record.attribution.via_api_key.clone(),
        client: record.attribution.client.clone(),
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
        let e = entry(Record {
            request_id: "chatcmpl-1",
            route: ferrox_api::routes::V1_CHAT_COMPLETIONS,
            status: 200,
            stream: true,
            duration_ms: 1_100,
            usage: Some(&usage()),
            attribution: &Attribution::default(),
        });
        assert_eq!(e.duration_ms, 1_100);
        assert_eq!(e.decode_ms, Some(100.0));
        assert_eq!(e.ttft_ms, Some(900.0));
        assert_eq!(e.prompt_tokens, 100);
        assert_eq!(e.completion_tokens, 10);
    }

    #[test]
    fn an_untimed_request_reports_null_rather_than_reusing_the_total() {
        let e = entry(Record {
            request_id: "chatcmpl-2",
            route: "/v1/completions",
            status: 200,
            stream: false,
            duration_ms: 42,
            usage: None,
            attribution: &Attribution::default(),
        });
        assert_eq!(e.decode_ms, None);
        assert_eq!(e.ttft_ms, None);
        assert_eq!(e.duration_ms, 42);
        assert_eq!(e.prompt_tokens, 0);
    }

    #[test]
    fn token_totals_accumulate_across_requests() {
        let stats = Stats::new();
        for _ in 0..3 {
            stats.record(entry(Record {
                request_id: "id",
                route: "/r",
                status: 200,
                stream: false,
                duration_ms: 1,
                usage: Some(&usage()),
                attribution: &Attribution::default(),
            }));
        }
        assert_eq!(stats.tokens_prompt_total(), 300);
        assert_eq!(stats.tokens_generated_total(), 30);
    }

    #[test]
    fn the_ring_keeps_the_newest_entries_and_drops_the_oldest() {
        let stats = Stats::new();
        for i in 0..RING_CAPACITY + 25 {
            stats.record(entry(Record {
                request_id: &format!("id-{i}"),
                route: "/r",
                status: 200,
                stream: false,
                duration_ms: 1,
                usage: None,
                attribution: &Attribution::default(),
            }));
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
        stats.record(entry(Record {
            request_id: "id-e",
            route: "/v1/chat/completions",
            status: 503,
            stream: false,
            duration_ms: 3,
            usage: None,
            attribution: &Attribution::default(),
        }));
        assert_eq!(stats.recent()[0].status, 503);
    }

    /// Attribution is carried, not re-derived: the entry states the key
    /// fingerprint and the self-declared label the request arrived
    /// with, and nothing else.
    #[test]
    fn an_entry_carries_the_attribution_it_was_given() {
        let attribution = Attribution {
            via_api_key: Some("key-deadbeef".to_string()),
            client: Some("ferrox-studio".to_string()),
        };
        let e = entry(Record {
            request_id: "id",
            route: "/r",
            status: 200,
            stream: false,
            duration_ms: 1,
            usage: None,
            attribution: &attribution,
        });
        assert_eq!(e.via_api_key.as_deref(), Some("key-deadbeef"));
        assert_eq!(e.client.as_deref(), Some("ferrox-studio"));

        let anonymous = entry(Record {
            request_id: "id",
            route: "/r",
            status: 200,
            stream: false,
            duration_ms: 1,
            usage: None,
            attribution: &Attribution::default(),
        });
        assert_eq!(anonymous.via_api_key, None);
        assert_eq!(anonymous.client, None);
    }
}
