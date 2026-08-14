// Shell: client-side routing and the header health pill.
//
// Routing is History-API based under `/ui/<screen>`; the server answers
// any such path with this same shell (see ferrox-server's `ui` module),
// so a reload or a bookmark lands on the right screen. Navigation never
// reloads the page, which matters more here than usual: a reload would
// drop an in-flight stream.

import { el, setText, clear } from './dom.js';
import { getJson, ApiError, routes } from './api.js';
import * as chat from './chat.js';
import * as models from './models.js';
import * as activity from './activity.js';
import * as connect from './connect.js';

const SCREENS = { chat, models, activity, connect };
const DEFAULT_SCREEN = 'chat';

const main = document.getElementById('main');
const nav = document.getElementById('nav');
const pill = document.getElementById('health-pill');
const dot = document.getElementById('health-dot');
const pillText = document.getElementById('health-text');
const detail = document.getElementById('health-detail');

/** The last `/health` body, shared with any screen that wants it. */
export const health = { current: null };

let teardown = null;

function screenFor(pathname) {
  const parts = pathname.split('/').filter(Boolean);
  const name = parts[0] === 'ui' ? parts[1] : parts[0];
  return Object.prototype.hasOwnProperty.call(SCREENS, name) ? name : DEFAULT_SCREEN;
}

function render(pathname) {
  const name = screenFor(pathname);
  // A screen's mount() returns its own cleanup — polling intervals and
  // abort controllers live and die with the screen rather than piling
  // up behind the router.
  if (teardown) {
    try { teardown(); } catch { /* a broken teardown must not wedge navigation */ }
    teardown = null;
  }
  clear(main);
  document.title = `Ferrox Studio — ${SCREENS[name].title}`;
  for (const link of nav.querySelectorAll('a')) {
    if (link.dataset.screen === name) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  try {
    teardown = SCREENS[name].mount(main) || null;
  } catch (error) {
    main.appendChild(
      el('p', { class: 'notice notice-error', text: `This screen failed to render: ${error.message}` }),
    );
  }
}

function navigate(pathname, { replace = false } = {}) {
  if (replace) window.history.replaceState({}, '', pathname);
  else window.history.pushState({}, '', pathname);
  render(pathname);
  main.focus({ preventScroll: true });
}

document.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const anchor = event.target.closest?.('a[href^="/"]');
  if (!anchor || anchor.target === '_blank') return;
  event.preventDefault();
  navigate(anchor.getAttribute('href'));
});

window.addEventListener('popstate', () => render(window.location.pathname));

// --------------------------------------------------------------------
// Health
//
// Three states, and the third one is the point: while the server is
// still probing backends it answers `detecting`, and this shows a
// probing pill rather than a verdict. Rendering "CPU only" from a guess
// is pixel-identical to rendering it from a measurement, and the user
// cannot tell which they were shown.
// --------------------------------------------------------------------

function paintHealth(body, error) {
  if (error) {
    dot.dataset.state = 'down';
    setText(pillText, error.status === 503 ? 'unavailable' : 'unreachable');
    return;
  }
  const state = body.state || 'unavailable';
  dot.dataset.state = state;
  const label =
    state === 'ready'
      ? body.model?.id || 'ready'
      : state === 'detecting'
        ? 'detecting backends…'
        : body.reason || 'unavailable';
  setText(pillText, label);
}

function paintHealthDetail() {
  clear(detail);
  const body = health.current;
  if (!body) {
    detail.appendChild(el('p', { class: 'muted', text: 'The server did not answer /health.' }));
    return;
  }
  const head = [`state: ${body.state}`];
  if (body.detail) head.push(body.detail);
  if (body.model) {
    head.push(
      `model: ${body.model.id} (tokenizer ${body.model.tokenizer})${
        body.model.synthetic_weights ? ' — SYNTHETIC random weights, output is noise' : ''
      }`,
    );
  }
  detail.appendChild(el('div', { text: head.join(' · ') }));
  const list = el('ul');
  for (const cap of body.capabilities || []) {
    // The server pairs every flag with a machine reason *and* a human
    // sentence precisely so the UI never re-derives the explanation.
    list.appendChild(
      el('li', { class: cap.available ? 'cap-yes' : 'cap-no', title: cap.detail }, [
        el('strong', { text: cap.id }),
        ` — ${cap.available ? 'available' : cap.reason}: ${cap.detail}`,
      ]),
    );
  }
  detail.appendChild(list);
  detail.appendChild(
    el('p', { class: 'small faint', text: `version ${body.version} · pid ${body.pid}` }),
  );
}

pill.addEventListener('click', () => {
  const open = detail.hidden;
  detail.hidden = !open;
  pill.setAttribute('aria-expanded', String(open));
  if (open) paintHealthDetail();
});

async function pollHealth() {
  try {
    health.current = await getJson(routes.health);
    paintHealth(health.current, null);
  } catch (error) {
    if (error instanceof ApiError && error.body?.state) {
      // 503 still carries a full body; it is an answer, not a failure.
      health.current = error.body;
      paintHealth(error.body, null);
    } else {
      health.current = null;
      paintHealth(null, error);
    }
  }
  if (!detail.hidden) paintHealthDetail();
}

pollHealth();
setInterval(pollHealth, 5000);

// `/` and `/ui` both land here; normalise so the nav highlight and the
// address bar agree from the first paint.
const initial = window.location.pathname;
if (screenFor(initial) === DEFAULT_SCREEN && !initial.startsWith('/ui/')) {
  navigate(`/ui/${DEFAULT_SCREEN}`, { replace: true });
} else {
  render(initial);
}

export { navigate };
