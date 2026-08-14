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

/// Every route above, for clients that want to enumerate the surface
/// (and for the round-trip test below).
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
}
