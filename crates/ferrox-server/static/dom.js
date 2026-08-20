// DOM construction and formatting helpers.
//
// There is exactly one way to put text on the screen in this app:
// `document.createTextNode`, via `el()`'s `text` option or `setText()`.
// Nothing here or anywhere else assigns `innerHTML`. Model output,
// error strings from the server and file paths from disk are all
// untrusted; a model can and will emit `<script>` or `<img onerror=…>`,
// and the only durable defence is never to have a parser in the path.

/** Build an element. `attrs.text` sets textContent; `attrs.html` does not exist. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'text') {
      node.appendChild(document.createTextNode(String(value)));
    } else if (key === 'class') {
      node.className = value;
    } else if (key === 'dataset') {
      Object.assign(node.dataset, value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) {
      node.setAttribute(key, '');
    } else {
      node.setAttribute(key, String(value));
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function setText(node, value) {
  clear(node).appendChild(document.createTextNode(String(value)));
  return node;
}

export function mount(container, ...children) {
  clear(container);
  for (const child of children) if (child) container.appendChild(child);
  return container;
}

// --------------------------------------------------------------------
// Formatting
//
// Every one of these answers "unknown" as an em dash rather than as a
// zero. The API is deliberate about `null` meaning "not established"
// (see ferrox-api's admin module); printing `0 B/s` for a rate the
// server refused to estimate would undo that on the last hop.
// --------------------------------------------------------------------

export const UNKNOWN = '—';

export function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

export function fmtInt(v) {
  return isNum(v) ? Math.round(v).toLocaleString() : UNKNOWN;
}

export function fmtNum(v, digits = 1) {
  return isNum(v) ? v.toFixed(digits) : UNKNOWN;
}

export function fmtBytes(v) {
  if (!isNum(v)) return UNKNOWN;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = v;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i === 0 ? 0 : n < 10 ? 2 : 1)} ${units[i]}`;
}

export function fmtRate(bytesPerSecond) {
  return isNum(bytesPerSecond) ? `${fmtBytes(bytesPerSecond)}/s` : UNKNOWN;
}

export function fmtMs(v) {
  if (!isNum(v)) return UNKNOWN;
  if (v < 1000) return `${v.toFixed(v < 10 ? 1 : 0)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

export function fmtDuration(seconds) {
  if (!isNum(seconds)) return UNKNOWN;
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  return `${h}h ${Math.floor((s % 3600) / 60)}m`;
}

export function fmtParams(count) {
  if (!isNum(count)) return UNKNOWN;
  if (count >= 1e12) return `${(count / 1e12).toFixed(2)}T`;
  if (count >= 1e9) return `${(count / 1e9).toFixed(count < 1e10 ? 2 : 1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(0)}M`;
  return fmtInt(count);
}

/** Wall-clock time of a server-supplied epoch-millisecond stamp. */
export function fmtClock(unixMs) {
  if (!isNum(unixMs)) return UNKNOWN;
  const d = new Date(unixMs);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** Copy `text`, reporting success by flipping the button's own label. */
export function copyButton(label, getText) {
  const button = el('button', {
    class: 'small',
    type: 'button',
    text: label,
    onclick: async () => {
      const original = button.textContent;
      try {
        await navigator.clipboard.writeText(getText());
        setText(button, 'copied');
      } catch {
        // Clipboard access can be refused (non-secure origin, denied
        // permission). Selecting the text is a real fallback; silently
        // doing nothing is not.
        setText(button, 'press ⌘C / Ctrl+C');
      }
      setTimeout(() => setText(button, original), 1400);
    },
  });
  return button;
}
