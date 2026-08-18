// Activity: the live request log.
//
// `duration_ms` and `decode_ms` get their own columns and are never
// added, averaged or collapsed into one "latency". `duration_ms` is
// wall time for the whole request — queue wait, prefill and decode —
// while `decode_ms` is the decode loop alone. A screen that showed only
// the first would read a 50 tok/s model as 5 whenever the prompt is
// long, and every throughput number a user quoted from it would be
// wrong in the same direction.
//
// Rows are keyed by `request_id`, which the server states in the first
// SSE chunk of the response that produced them, so a chat message and
// its log line can be joined exactly rather than by a timing heuristic.

import { el, mount as fill, clear, fmtInt, fmtMs, fmtDuration, fmtClock, isNum } from './dom.js';
import { getJson, routes, ApiError } from './api.js';

export const title = 'Activity';

const POLL_MS = 2000;

export function mount(container) {
  let disposed = false;
  let timer = null;
  let unsupported = false;
  let lastError = null;

  const countersPanel = el('div', { class: 'panel' });
  const logPanel = el('div', { class: 'panel' });
  const bannerSlot = el('div');

  container.appendChild(el('div', { class: 'screen' }, [bannerSlot, countersPanel, logPanel]));

  const counter = (label, value, title) =>
    el('div', { class: 'counter', title: title || '' }, [
      el('div', { class: 'counter-label', text: label }),
      el('div', { class: 'counter-value', text: value }),
    ]);

  function paintUnsupported() {
    clear(bannerSlot);
    fill(
      countersPanel,
      el('h2', { class: 'panel-title', text: 'Server counters' }),
      el('p', { class: 'notice notice-warn' }, [
        'Not available in this build. ',
        el('code', { text: routes.adminStats }),
        ' answered 404, so there is no request log to show. ',
        el('code', { text: '/metrics' }),
        ' may still carry Prometheus counters.',
      ]),
    );
    clear(logPanel);
  }

  function paint(stats) {
    clear(bannerSlot);
    fill(
      countersPanel,
      el('h2', { class: 'panel-title', text: 'Server counters' }),
      el('div', { class: 'counters' }, [
        counter('uptime', fmtDuration(stats.uptime_seconds)),
        counter('requests', fmtInt(stats.requests_total)),
        counter('errors', fmtInt(stats.errors_total)),
        counter('cache hits', fmtInt(stats.cache_hits)),
        counter('cache misses', fmtInt(stats.cache_misses)),
        counter('prompt tokens', fmtInt(stats.tokens_prompt_total)),
        counter('generated tokens', fmtInt(stats.tokens_generated_total)),
        counter(
          'generating now',
          isNum(stats.generating_now) ? fmtInt(stats.generating_now) : '—',
          'Streamed generations decoding at this instant — the ones POST /v1/cancel could stop. '
            + 'Work in progress, not a queue depth: nothing waits in front of a decode here.',
        ),
        counter(
          'last request',
          isNum(stats.last_request_age_seconds) ? `${fmtDuration(stats.last_request_age_seconds)} ago` : '—',
          'Recent activity is positive evidence of liveness even when /health is slow to answer.',
        ),
      ]),
    );

    // The ring buffer arrives newest-last; a log reads newest-first.
    const recent = [...(stats.recent || [])].reverse();
    const rows = recent.map((row) =>
      el('tr', {}, [
        el('td', { class: 'mono', text: fmtClock(row.at_ms) }),
        el('td', { class: 'mono', text: row.request_id, title: row.request_id }),
        el('td', { text: row.route }),
        el('td', { class: 'num' }, [
          el('span', { class: row.status >= 400 ? 'err' : '', text: String(row.status) }),
        ]),
        el('td', { text: row.stream ? 'stream' : 'once' }),
        el('td', { class: 'num', text: fmtInt(row.prompt_tokens) }),
        el('td', { class: 'num', text: fmtInt(row.completion_tokens) }),
        el('td', { class: 'num', text: isNum(row.ttft_ms) ? fmtMs(row.ttft_ms) : '—' }),
        el('td', { class: 'num', text: fmtMs(row.duration_ms) }),
        el('td', { class: 'num', text: isNum(row.decode_ms) ? fmtMs(row.decode_ms) : '—' }),
      ]),
    );

    fill(
      logPanel,
      el('h2', { class: 'panel-title', text: `Recent requests (${recent.length})` }),
      recent.length
        ? el('div', { class: 'table-wrap' }, [
            el('table', {}, [
              el('thead', {}, [
                el('tr', {}, [
                  el('th', { text: 'at' }),
                  el('th', { text: 'request_id' }),
                  el('th', { text: 'route' }),
                  el('th', { class: 'num', text: 'status' }),
                  el('th', { text: 'mode' }),
                  el('th', { class: 'num', text: 'prompt' }),
                  el('th', { class: 'num', text: 'gen' }),
                  el('th', { class: 'num', text: 'ttft' }),
                  el('th', { class: 'num', text: 'duration' }),
                  el('th', { class: 'num', text: 'decode' }),
                ]),
              ]),
              el('tbody', {}, rows),
            ]),
          ])
        : el('p', { class: 'muted small', text: 'No requests yet. Send a message on the Chat screen.' }),
      el('p', { class: 'small faint' }, [
        el('strong', { text: 'duration' }),
        ' is the whole request — queue wait, prefill and decode. ',
        el('strong', { text: 'decode' }),
        ' is the decode loop alone; it is “—” when the engine did not time itself or the answer came from cache. Dividing generated tokens by ',
        el('strong', { text: 'duration' }),
        ' is how a fast model gets reported as a slow one, so the two are never combined here.',
      ]),
    );
  }

  async function refresh() {
    if (disposed) return;
    try {
      const stats = await getJson(routes.adminStats);
      if (disposed) return;
      lastError = null;
      paint(stats);
    } catch (error) {
      if (disposed) return;
      if (error instanceof ApiError && error.isMissingEndpoint) {
        unsupported = true;
        paintUnsupported();
        return;
      }
      // A transient failure must not wipe the last good table: keep it
      // and say the numbers are stale.
      if (lastError !== error.message) {
        lastError = error.message;
        fill(bannerSlot, el('p', { class: 'notice notice-error', text: `Stale: ${error.message}` }));
      }
    }
    if (!disposed && !unsupported) timer = setTimeout(refresh, POLL_MS);
  }

  fill(countersPanel, el('h2', { class: 'panel-title', text: 'Server counters' }), el('p', { class: 'muted', text: 'Loading…' }));
  refresh();

  return () => {
    disposed = true;
    clearTimeout(timer);
  };
}
