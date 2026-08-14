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

/// The embedded UI. Served at both `/` and `/ui` only when the server
/// is started with `--ui-server` / `FERROX_UI=1`.
pub const ROOT: &str = "/";
/// See [`ROOT`].
pub const UI: &str = "/ui";

pub const V1_MODELS: &str = "/v1/models";
pub const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
pub const V1_COMPLETIONS: &str = "/v1/completions";
pub const V1_TOKENIZE: &str = "/v1/tokenize";
pub const V1_DETOKENIZE: &str = "/v1/detokenize";
pub const V1_EMBEDDINGS: &str = "/v1/embeddings";

/// Anthropic-compatible messages endpoint.
pub const V1_MESSAGES: &str = "/v1/messages";

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

/// Every fixed route above, for clients that want to enumerate the
/// surface (and for the round-trip test below).
///
/// [`ADMIN_TASK_CANCEL`] is deliberately absent: it is a template, and
/// a caller iterating this list to probe paths would get a 404 for a
/// literal `{task_id}`.
pub const ALL: &[&str] = &[
    HEALTH,
    METRICS,
    CACHE_STATS,
    ROOT,
    UI,
    V1_MODELS,
    V1_CHAT_COMPLETIONS,
    V1_COMPLETIONS,
    V1_TOKENIZE,
    V1_DETOKENIZE,
    V1_EMBEDDINGS,
    V1_MESSAGES,
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
            assert!(
                !route.ends_with('/') || *route == ROOT,
                "{route} has a trailing slash"
            );
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

    #[test]
    fn a_cancel_path_substitutes_the_only_placeholder() {
        assert_eq!(
            admin_task_cancel("task-7"),
            "/admin/tasks/task-7/cancel".to_string()
        );
        assert!(!admin_task_cancel("task-7").contains('{'));
    }
}
