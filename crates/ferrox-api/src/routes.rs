//! Every path `ferrox-server` serves, named once.
//!
//! Only routes that actually exist belong here. A constant for a
//! not-yet-implemented endpoint is worse than no constant at all: it
//! reads as a promise, and a client that imports it gets a 404 with the
//! contract crate's blessing.

/// Liveness + readiness + capability handshake. Never behind auth, so a
/// probe works regardless of `FERROX_API_KEY`.
pub const HEALTH: &str = "/health";

/// Prometheus text-exposition metrics.
pub const METRICS: &str = "/metrics";

/// Response- and prefix-cache counters.
pub const CACHE_STATS: &str = "/cache/stats";

pub const V1_MODELS: &str = "/v1/models";
pub const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
pub const V1_COMPLETIONS: &str = "/v1/completions";
pub const V1_TOKENIZE: &str = "/v1/tokenize";
pub const V1_DETOKENIZE: &str = "/v1/detokenize";
pub const V1_EMBEDDINGS: &str = "/v1/embeddings";

/// Anthropic-compatible messages endpoint.
pub const V1_MESSAGES: &str = "/v1/messages";

/// Explicit cancellation of one in-flight generation, by the
/// `request_id` the server states on the first streamed chunk.
///
/// Under `/v1` rather than at the root the plan sketched it at, for two
/// reasons that both matter: it acts on inference and so belongs behind
/// the same `FERROX_API_KEY` gate as the endpoint that started the
/// work, and `/v1` is where every other inference path already lives,
/// so a client configured with one base URL reaches all of them.
///
/// This is the second tier of cancellation, not the only one -- a
/// client that simply drops the connection is also honoured. It exists
/// because the first tier is unreliable: proxies buffer, and a page
/// unload races the abort it is supposed to send. `keepalive: true`
/// makes this one survive that.
pub const V1_CANCEL: &str = "/v1/cancel";

/// Reconnect into a stream started with `stream_resumable: true`,
/// resuming after the `Last-Event-ID` the client last saw.
///
/// **A template, not a literal** -- see [`ADMIN_TASK_CANCEL`] for why
/// this crate writes placeholders in the OpenAPI style. Build a
/// concrete path with [`v1_stream`].
///
/// Behind the same key as the endpoint that started the work: the
/// replay buffer holds the model's output, so reading it must cost
/// exactly what producing it cost.
pub const V1_STREAM: &str = "/v1/stream/{request_id}";

/// The same replay buffer over plain JSON, for the case SSE cannot
/// survive: a reverse proxy that buffers `text/event-stream` turns a
/// stream into one long silence, and cannot do that to a short response
/// that has already ended. Build a concrete path with
/// [`v1_stream_poll`].
pub const V1_STREAM_POLL: &str = "/v1/stream/{request_id}/poll";

// ---------------------------------------------------------------------
// Control surface.
//
// Everything under `/admin` either changes what the server serves or
// writes to disk, so all of it sits behind the same `FERROX_API_KEY`
// gate as `/v1/*` -- never on the unauthenticated `/health` side.
// ---------------------------------------------------------------------

/// Model inventory: what is on disk, what is loaded, what failed.
pub const ADMIN_MODELS: &str = "/admin/models";

/// Start loading a discovered model by its `id`. Answers `202` with a
/// task id; the load itself runs off the request.
pub const ADMIN_MODELS_LOAD: &str = "/admin/models/load";

/// Drop the active model. Synchronous: unloading is releasing one
/// `Arc`, and requests already decoding keep theirs.
pub const ADMIN_MODELS_UNLOAD: &str = "/admin/models/unload";

/// Fetch a `.gguf` from the Hugging Face Hub into the model directory.
/// Answers `202` with a task id.
pub const ADMIN_DOWNLOAD: &str = "/admin/download";

/// Every long-running job this server knows about, newest first.
pub const ADMIN_TASKS: &str = "/admin/tasks";

/// Request cancellation of one task. **A template, not a literal**: the
/// `{task_id}` placeholder is written in the OpenAPI style rather than
/// any one web framework's, because this crate is imported by clients
/// that have never heard of the server's router. Build a concrete path
/// with [`admin_task_cancel`].
pub const ADMIN_TASK_CANCEL: &str = "/admin/tasks/{task_id}/cancel";

/// Counters, uptime, and the recent-request ring buffer.
pub const ADMIN_STATS: &str = "/admin/stats";

/// The concrete cancel path for one task id.
pub fn admin_task_cancel(task_id: &str) -> String {
    ADMIN_TASK_CANCEL.replace("{task_id}", task_id)
}

/// The concrete resume path for one request id.
pub fn v1_stream(request_id: &str) -> String {
    V1_STREAM.replace("{request_id}", request_id)
}

/// The concrete polling-fallback path for one request id.
pub fn v1_stream_poll(request_id: &str) -> String {
    V1_STREAM_POLL.replace("{request_id}", request_id)
}

/// Every fixed route above, for clients that want to enumerate the
/// surface (and for the round-trip test below).
///
/// [`ADMIN_TASK_CANCEL`], [`V1_STREAM`] and [`V1_STREAM_POLL`] are
/// deliberately absent: they are templates, and a caller iterating this
/// list to probe paths would get a 404 for a literal `{task_id}`.
pub const ALL: &[&str] = &[
    HEALTH,
    METRICS,
    CACHE_STATS,
    V1_MODELS,
    V1_CHAT_COMPLETIONS,
    V1_COMPLETIONS,
    V1_TOKENIZE,
    V1_DETOKENIZE,
    V1_EMBEDDINGS,
    V1_MESSAGES,
    V1_CANCEL,
    ADMIN_MODELS,
    ADMIN_MODELS_LOAD,
    ADMIN_MODELS_UNLOAD,
    ADMIN_DOWNLOAD,
    ADMIN_TASKS,
    ADMIN_STATS,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_route_is_absolute_and_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for route in ALL {
            assert!(route.starts_with('/'), "{route} is not an absolute path");
            assert!(!route.ends_with('/'), "{route} has a trailing slash");
            assert!(seen.insert(*route), "{route} is listed twice");
        }
    }

    #[test]
    fn the_admin_surface_is_namespaced() {
        for route in ALL.iter().filter(|r| r.starts_with("/admin")) {
            assert!(
                route.starts_with("/admin/"),
                "{route} would collide with the /admin prefix itself"
            );
        }
    }

    #[test]
    fn the_cancel_template_is_not_enumerated_as_a_real_path() {
        assert!(!ALL.contains(&ADMIN_TASK_CANCEL));
        assert!(ADMIN_TASK_CANCEL.contains("{task_id}"));
    }

    /// Same rule for the stream templates: a client that probed this
    /// list would ask for a literal `{request_id}` and get a 404 with
    /// the contract crate's blessing.
    #[test]
    fn the_stream_templates_are_not_enumerated_as_real_paths() {
        for template in [V1_STREAM, V1_STREAM_POLL] {
            assert!(!ALL.contains(&template));
            assert!(template.contains("{request_id}"));
        }
        assert_eq!(v1_stream("chatcmpl-7"), "/v1/stream/chatcmpl-7");
        assert_eq!(
            v1_stream_poll("chatcmpl-7"),
            "/v1/stream/chatcmpl-7/poll".to_string()
        );
        assert!(!v1_stream("chatcmpl-7").contains('{'));
    }

    /// The polling fallback must sit under the stream it falls back
    /// from, so one base URL and one key reach both.
    #[test]
    fn the_poll_route_is_nested_under_the_resume_route() {
        assert!(V1_STREAM_POLL.starts_with(V1_STREAM));
        assert!(V1_STREAM.starts_with("/v1/"));
    }

    #[test]
    fn a_cancel_path_substitutes_the_only_placeholder() {
        assert_eq!(
            admin_task_cancel("task-7"),
            "/admin/tasks/task-7/cancel".to_string()
        );
        assert!(!admin_task_cancel("task-7").contains('{'));
    }
}
