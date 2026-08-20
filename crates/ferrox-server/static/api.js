// The only place in the frontend that talks HTTP.
//
// The UI is just another API client: it calls the same public
// `/v1/chat/completions` an editor would, with the same key, over the
// same streaming path. There is no private UI endpoint anywhere in this
// app, which is what stops the public contract from rotting silently.
//
// Route strings mirror `ferrox_api::routes` one-for-one. Keeping them
// in a single object means a rename shows up here as one diff rather
// than as a scattering of string literals.

export const routes = {
  health: '/health',
  metrics: '/metrics',
  cacheStats: '/cache/stats',
  models: '/v1/models',
  chatCompletions: '/v1/chat/completions',
  cancel: '/v1/cancel',
  adminModels: '/admin/models',
  adminModelsLoad: '/admin/models/load',
  adminModelsUnload: '/admin/models/unload',
  adminDownload: '/admin/download',
  adminTasks: '/admin/tasks',
  adminStats: '/admin/stats',
  adminTaskCancel: (taskId) => `/admin/tasks/${encodeURIComponent(taskId)}/cancel`,
};

const KEY_STORAGE = 'ferrox.studio.apiKey';

/** The API key, when the operator has set FERROX_API_KEY and told us. */
export function apiKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || '';
  } catch {
    return '';
  }
}

export function setApiKey(value) {
  try {
    if (value) localStorage.setItem(KEY_STORAGE, value);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* private browsing: the key simply does not persist */
  }
}

export const baseUrl = () => window.location.origin;

function headers(extra = {}) {
  const h = { ...extra };
  const key = apiKey();
  if (key) h.Authorization = `Bearer ${key}`;
  return h;
}

/**
 * An HTTP failure with the status kept intact.
 *
 * `status` is load-bearing, not decoration: a 404 from an `/admin/*`
 * route means "this build does not have the control surface" and each
 * screen renders a different, honest empty state for it. Collapsing
 * every failure into one message would make that impossible.
 */
export class ApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }

  get isMissingEndpoint() {
    return this.status === 404 || this.status === 405;
  }

  get isAuth() {
    return this.status === 401 || this.status === 403;
  }
}

async function parse(response) {
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!response.ok) {
    const message =
      body?.error?.message || body?.message || text.slice(0, 300) || response.statusText;
    throw new ApiError(response.status, message, body);
  }
  return body;
}

export async function getJson(path, { signal } = {}) {
  let response;
  try {
    response = await fetch(path, { headers: headers(), signal });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError(0, `cannot reach ${baseUrl()}: ${cause.message}`, null);
  }
  return parse(response);
}

export async function postJson(path, payload, { signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: headers({ 'Content-Type': 'application/json' }),
      body: payload === undefined ? '{}' : JSON.stringify(payload),
      signal,
    });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError(0, `cannot reach ${baseUrl()}: ${cause.message}`, null);
  }
  return parse(response);
}

/**
 * Ask the server to stop the generation behind `requestId`.
 *
 * The second tier of cancellation. Aborting the fetch closes the socket
 * and the server now notices that too, but a socket is not a reliable
 * signal: a reverse proxy can hold the connection open on the backend
 * side, and a page unload races the abort it is supposed to send.
 * `keepalive: true` is what makes this one outlive the page, which is
 * exactly the case the socket tier is worst at.
 *
 * Deliberately not awaited by callers and deliberately silent: it is
 * best-effort cleanup, and a 404 here only means the answer finished
 * first. Nothing the user did went wrong, so nothing is shown.
 */
export function cancelGeneration(requestId) {
  if (!requestId) return;
  try {
    fetch(routes.cancel, {
      method: 'POST',
      headers: headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ request_id: requestId }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* the stream is already being torn down; there is nothing to report */
  }
}

// --------------------------------------------------------------------
// Streaming chat
// --------------------------------------------------------------------

/**
 * How long a stream may go without delivering a single byte before the
 * UI says so.
 *
 * Measured against *bytes*, not tokens, which is what makes the number
 * safe to keep low-ish: the server sends an SSE keep-alive comment
 * every 15 s, so a healthy stream resets this timer four times a minute
 * even while a long prompt is still prefilling. A slow model therefore
 * never trips it; a proxy that swallowed the connection does.
 *
 * It reports and never aborts. A stalled stream may still recover, and
 * killing a generation the user is waiting on — after paying its
 * prefill — to show a tidier error would be the worse outcome.
 */
const STALL_MS = 45000;

/**
 * POST a chat completion with `stream: true` and drive the callbacks.
 *
 * Three things the server states that this reads rather than guesses:
 *
 * - `request_id` arrives on the **first** chunk only, so it is captured
 *   once and never re-derived. A UI that instead claimed an id out of a
 *   stats snapshot would mis-attribute the moment two chats overlap.
 * - `usage` arrives on the **final** chunk and carries per-phase
 *   timings. Client wall-clock is never substituted for them.
 * - A stream that ends without `[DONE]` and without a `finish_reason`
 *   was truncated. That is surfaced as an error, not as a finished
 *   message, because the two are indistinguishable on screen otherwise.
 *
 * @returns {Promise<{requestId: string|null, usage: object|null, finishReason: string|null}>}
 */
export async function streamChat(
  request,
  { signal, onToken, onRequestId, onStall, stallMs = STALL_MS } = {},
) {
  const response = await fetch(routes.chatCompletions, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
    body: JSON.stringify({ ...request, stream: true }),
    signal,
  });

  if (!response.ok || !response.body) {
    // An error answer is JSON, not SSE — parse it as such.
    return parse(response);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;
  let requestId = null;
  let usage = null;
  let finishReason = null;

  const handle = (payload) => {
    if (payload === '[DONE]') {
      done = true;
      return;
    }
    let chunk;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return; // a keep-alive comment or a partial frame; not fatal
    }
    if (chunk.request_id && !requestId) {
      requestId = chunk.request_id;
      onRequestId?.(requestId);
    }
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    const text = choice?.delta?.content;
    if (text) onToken?.(text);
  };

  // Reports a stream that has gone quiet, once, and keeps waiting. The
  // timer is armed against every read rather than every token, so the
  // keep-alive comments disarm it on a healthy but slow stream.
  let stalled = false;
  let stallTimer = null;
  const armStallTimer = () => {
    clearTimeout(stallTimer);
    if (!onStall) return;
    stallTimer = setTimeout(() => {
      stalled = true;
      onStall(stallMs);
    }, stallMs);
  };
  const disarmStallTimer = () => {
    clearTimeout(stallTimer);
    stallTimer = null;
  };

  try {
    armStallTimer();
    for (;;) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      armStallTimer();
      if (stalled) {
        // It came back. Saying so matters as much as saying it stopped:
        // a banner left up after recovery is a lie with a long tail.
        stalled = false;
        onStall?.(null);
      }
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line; a `data:` field may be
      // split across reads, so only complete frames are consumed.
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of frame.split('\n')) {
          if (line.startsWith('data:')) handle(line.slice(5).trimStart());
        }
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    disarmStallTimer();
    if (stalled) onStall?.(null);
  }

  if (!done && !finishReason) {
    throw new ApiError(
      0,
      'the stream ended without a finish reason — the response was truncated, not completed',
      null,
    );
  }

  return { requestId, usage, finishReason };
}
