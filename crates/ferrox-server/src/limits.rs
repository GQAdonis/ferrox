//! Optional bearer-token auth and a simple rate limiter for
//! ferrox-server. Both are off by default -- unset `FERROX_API_KEY` /
//! `FERROX_RATE_LIMIT_PER_MINUTE` and the server behaves exactly as
//! before -- following the same opt-in convention llama.cpp's server
//! uses for `--api-key`, so existing deployments and tests aren't
//! affected unless explicitly configured.

use axum::{
    body::Body,
    extract::State,
    http::{header::AUTHORIZATION, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct AuthConfig {
    pub api_key: Arc<String>,
}

/// Rejects any request whose `Authorization: Bearer <key>` header
/// doesn't match the configured key. Only ever added to the router
/// when `FERROX_API_KEY` is set (see `main.rs`) -- there is no
/// "disabled" state to check here, keeping the common unauthenticated
/// path free of any per-request branch.
pub async fn require_api_key(
    State(config): State<AuthConfig>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    if provided == Some(config.api_key.as_str()) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "invalid or missing API key"}})),
        )
            .into_response()
    }
}

struct Bucket {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

/// A simple token-bucket rate limiter, shared across all requests
/// (global, not per-client): `capacity` tokens, continuously refilled
/// at `refill_per_sec` tokens/second, one token consumed per request
/// that passes through it. A reasonable minimum for a single-model
/// self-hosted deployment; a real multi-tenant deployment would want
/// per-API-key buckets instead of one global bucket, a possible
/// follow-on.
pub struct RateLimiter {
    inner: Mutex<Bucket>,
}

impl RateLimiter {
    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {
        RateLimiter {
            inner: Mutex::new(Bucket {
                tokens: capacity as f64,
                capacity: capacity as f64,
                refill_per_sec,
                last_refill: Instant::now(),
            }),
        }
    }

    pub fn per_minute(requests_per_minute: u32) -> Self {
        // Bucket capacity equals a full minute's allowance, so a burst
        // after idle time isn't punished, refilled continuously at the
        // equivalent per-second rate.
        Self::new(requests_per_minute, requests_per_minute as f64 / 60.0)
    }

    /// Attempts to consume one token. Returns `false` (and consumes
    /// nothing) if none are available right now.
    pub fn try_acquire(&self) -> bool {
        let mut b = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        let elapsed = now.duration_since(b.last_refill).as_secs_f64();
        b.tokens = (b.tokens + elapsed * b.refill_per_sec).min(b.capacity);
        b.last_refill = now;
        if b.tokens >= 1.0 {
            b.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub async fn rate_limit(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if limiter.try_acquire() {
        next.run(req).await
    } else {
        (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({"error": {"message": "rate limit exceeded"}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_allows_up_to_capacity_then_blocks() {
        let limiter = RateLimiter::new(3, 0.0); // no refill, isolate the capacity check
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(
            !limiter.try_acquire(),
            "a 4th request within the same instant must be rejected"
        );
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        let limiter = RateLimiter::new(1, 1000.0); // fast refill for a quick test
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(
            limiter.try_acquire(),
            "tokens must refill over time, not stay exhausted forever"
        );
    }

    #[test]
    fn per_minute_constructor_sets_a_full_minute_of_burst_capacity() {
        let limiter = RateLimiter::per_minute(60);
        for _ in 0..60 {
            assert!(limiter.try_acquire());
        }
        assert!(!limiter.try_acquire());
    }
}
