//! Serving the embedded studio frontend.
//!
//! The whole UI -- one HTML shell, one stylesheet and a handful of ES
//! modules -- is compiled into the binary by `rust-embed`, so a single
//! `ferrox-server` executable carries it. `debug-embed` is on
//! deliberately: without it a debug build reads `static/` off disk at
//! runtime, which means the tests below would exercise a code path the
//! shipped binary never takes.
//!
//! ## Why there is a fallback at all
//!
//! The frontend routes client-side under `/ui/...` using the History
//! API, so a reload of `/ui/models` asks this server for a path no
//! router knows. The fallback answers those with the shell and lets the
//! frontend sort out which screen that is.
//!
//! Three rules keep that from swallowing anything it shouldn't:
//!
//! 1. **A real asset always wins over the shell.** `/app.js` is looked
//!    up in the embedded set first.
//! 2. **API paths are never answered with HTML.** A typo'd `/v1/...`
//!    stays a JSON 404, in the same shape the rest of the surface uses.
//!    A client that got an HTML shell back with a `200` for a mistyped
//!    endpoint would report it as a parse error, not a 404.
//! 3. **Anything that looks like a file stays a 404.** Handing the HTML
//!    shell back for a missing `/app.css` produces a MIME-type error in
//!    the console instead of the missing-file error that is true.
//!
//! All of this is registered only when the server is started with
//! `--ui-server` / `FERROX_UI=1`; without it the router is exactly what
//! it was before.

use axum::{
    http::{header, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use ferrox_api::routes;
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "static/"]
struct Assets;

/// The SPA shell. Every client-side route resolves to this file.
const INDEX: &str = "index.html";

/// Path prefixes owned by the HTTP API. Nothing under these is ever
/// answered with the shell, whatever the frontend's routing does.
///
/// Kept as prefixes rather than exact paths so an unknown sibling
/// (`/v1/rerank`, `/admin/whatever`) is covered too -- those are the
/// requests most likely to be a client's typo, and the ones an HTML
/// `200` would confuse most. `every_api_route_is_shielded_from_the_spa_fallback`
/// below asserts this list still covers `routes::ALL`.
const API_PREFIXES: &[&str] = &["/v1/", "/admin/", "/health", "/metrics", "/cache/"];

/// Registers the UI shell, its assets and the SPA fallback.
///
/// Generic over the router's state so it composes with whatever
/// `main` has built: none of these handlers reads application state.
pub(crate) fn attach<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .route(routes::ROOT, get(shell))
        .route(routes::UI, get(shell))
        .fallback(fallback)
}

async fn shell() -> Response {
    serve(INDEX).unwrap_or_else(|| {
        // Unreachable with the embedded folder present; stated rather
        // than unwrapped so a packaging mistake is a readable 500 and
        // not a panic in a request handler.
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "the embedded UI is missing from this build",
        )
            .into_response()
    })
}

async fn fallback(method: Method, uri: Uri) -> Response {
    let path = uri.path();
    if method != Method::GET && method != Method::HEAD {
        return not_found(path);
    }
    if let Some(asset) = serve(path.trim_start_matches('/')) {
        return asset;
    }
    if is_api_path(path) || looks_like_a_file(path) {
        return not_found(path);
    }
    shell().await
}

fn serve(relative: &str) -> Option<Response> {
    // `Assets::get` is an exact lookup into a compile-time map, so a
    // `..` segment or a percent-encoded one simply misses -- there is
    // no filesystem to traverse out of.
    let file = Assets::get(relative)?;
    Some(
        (
            [
                (header::CONTENT_TYPE, content_type(relative)),
                // The shell and its modules ship together and change
                // together; revalidating costs one local round trip and
                // removes the "why is my UI stale" class of bug.
                (header::CACHE_CONTROL, "no-cache"),
            ],
            file.data.into_owned(),
        )
            .into_response(),
    )
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("json") | Some("map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn is_api_path(path: &str) -> bool {
    API_PREFIXES.iter().any(|p| path.starts_with(p))
}

/// Whether the last path segment names a file. `/ui/models` does not;
/// `/app.css` and `/favicon.ico` do.
fn looks_like_a_file(path: &str) -> bool {
    path.rsplit('/').next().is_some_and(|seg| seg.contains('.'))
}

fn not_found(path: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": format!("no route for {path}"),
                "type": "not_found",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    /// A router shaped like `main`'s: one representative API route,
    /// plus the UI. State-free on purpose -- what is under test is
    /// routing precedence and asset serving, not any handler.
    fn app() -> Router {
        attach(
            Router::new()
                .route(
                    routes::V1_MODELS,
                    get(|| async { Json(serde_json::json!({"object": "list", "data": []})) }),
                )
                .route(
                    routes::ADMIN_STATS,
                    get(|| async { Json(serde_json::json!({"requests_total": 0})) }),
                ),
        )
    }

    async fn get_path(path: &str) -> (StatusCode, String, String) {
        let response = app()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .map(|v| v.to_str().unwrap().to_string())
            .unwrap_or_default();
        let body = to_bytes(response.into_body(), 4 << 20).await.unwrap();
        (status, content_type, String::from_utf8_lossy(&body).into())
    }

    #[tokio::test]
    async fn the_shell_is_served_at_both_root_and_ui() {
        for path in [routes::ROOT, routes::UI] {
            let (status, content_type, body) = get_path(path).await;
            assert_eq!(status, StatusCode::OK, "{path}");
            assert_eq!(content_type, "text/html; charset=utf-8", "{path}");
            assert!(body.contains("<!DOCTYPE html>"), "{path} served {body:.80}");
            assert!(body.contains("/app.js"), "{path} does not load the app");
        }
    }

    #[tokio::test]
    async fn assets_are_served_with_the_content_type_a_browser_needs() {
        // A stylesheet served as text/plain is ignored, and an ES
        // module served as anything but JavaScript is a hard load
        // error -- these two headers are the whole reason the fallback
        // checks the embedded set before it answers with HTML.
        let (status, content_type, body) = get_path("/app.css").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/css; charset=utf-8");
        assert!(body.contains("--"), "expected CSS custom properties");

        let (status, content_type, body) = get_path("/app.js").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/javascript; charset=utf-8");
        assert!(body.contains("import"), "expected an ES module");
    }

    #[tokio::test]
    async fn a_client_side_route_falls_back_to_the_shell() {
        let (_, _, shell) = get_path(routes::UI).await;
        for path in ["/ui/chat", "/ui/models", "/ui/activity", "/ui/connect"] {
            let (status, content_type, body) = get_path(path).await;
            assert_eq!(status, StatusCode::OK, "{path}");
            assert_eq!(content_type, "text/html; charset=utf-8", "{path}");
            assert_eq!(body, shell, "{path} served something other than the shell");
        }
    }

    #[tokio::test]
    async fn api_routes_still_win_over_the_fallback() {
        let (status, content_type, body) = get_path(routes::V1_MODELS).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            content_type.starts_with("application/json"),
            "{content_type}"
        );
        assert!(body.contains("\"object\":\"list\""), "{body}");

        let (status, _, body) = get_path(routes::ADMIN_STATS).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("requests_total"), "{body}");
    }

    #[tokio::test]
    async fn an_unknown_api_path_stays_a_json_404() {
        for path in ["/v1/rerank", "/admin/nope", "/metrics/extra"] {
            let (status, content_type, body) = get_path(path).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert!(content_type.starts_with("application/json"), "{path}");
            assert!(body.contains("not_found"), "{path} -> {body}");
        }
    }

    #[tokio::test]
    async fn a_missing_asset_is_a_404_rather_than_the_shell() {
        // Answering `200 text/html` here would surface in the browser
        // as a MIME-type error about app.js, which is the wrong bug.
        let (status, _, _) = get_path("/nope.js").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = get_path("/assets/missing.css").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_non_get_request_is_never_answered_with_the_shell() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/ui/chat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn every_api_route_is_shielded_from_the_spa_fallback() {
        for route in routes::ALL {
            if *route == routes::ROOT || *route == routes::UI {
                continue;
            }
            assert!(
                is_api_path(route),
                "{route} is not covered by API_PREFIXES, so a 404 for it \
                 would be answered with the HTML shell"
            );
        }
        assert!(is_api_path(&routes::admin_task_cancel("t1")));
    }

    #[test]
    fn the_shell_and_every_module_it_needs_are_embedded() {
        let embedded: Vec<String> = Assets::iter().map(|f| f.to_string()).collect();
        for required in [
            "index.html",
            "app.css",
            "app.js",
            "api.js",
            "dom.js",
            "md.js",
            "chat.js",
            "models.js",
            "activity.js",
            "connect.js",
        ] {
            assert!(
                embedded.iter().any(|f| f == required),
                "{required} is not embedded; have {embedded:?}"
            );
        }
    }

    #[test]
    fn only_a_named_extension_gets_a_typed_content_type() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type("app.js"), "text/javascript; charset=utf-8");
        assert_eq!(content_type("noextension"), "application/octet-stream");
    }

    #[test]
    fn a_path_without_a_dotted_last_segment_is_a_route_not_a_file() {
        assert!(!looks_like_a_file("/ui/models"));
        assert!(!looks_like_a_file("/ui"));
        assert!(looks_like_a_file("/app.css"));
        // A dot in an earlier segment does not make the target a file.
        assert!(!looks_like_a_file("/v1.5/chat"));
    }
}
