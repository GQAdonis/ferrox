// Chat: the streaming screen.
//
// Conversation state lives in memory and is mirrored into
// `localStorage`. That is a deliberate limit, not an oversight: this
// server has no conversation-persistence API, and inventing a
// client-side "sync" against endpoints that do not exist would be a lie
// the user could not see. The screen says where the transcript lives.
//
// Every number under a response comes from the server's `usage` object.
// The browser holds no stopwatch here, because a client stopwatch
// cannot separate prefill from decode and would report a 50 tok/s model
// as 5 whenever the prompt is long.

import { el, mount as fill, clear, setText, copyButton, fmtMs, fmtNum, fmtInt, isNum } from './dom.js';
import { streamChat, getJson, cancelGeneration, routes, ApiError } from './api.js';
import { renderMarkdown } from './md.js';

export const title = 'Chat';

const CHAT_KEY = 'ferrox.studio.chat.v1';
const SETTINGS_KEY = 'ferrox.studio.sampling.v1';

const DEFAULT_SETTINGS = {
  system: '',
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 512,
};

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : structuredClone(fallback);
  } catch {
    return structuredClone(fallback);
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota or private browsing — the in-memory transcript still works */
  }
}

/** Turns the server's `usage` into one line, omitting anything absent. */
function statLine(usage, requestId) {
  const parts = [];
  if (isNum(usage?.time_to_first_token_ms)) parts.push(`TTFT ${fmtMs(usage.time_to_first_token_ms)}`);
  if (usage) {
    const prefill = [`${fmtInt(usage.prompt_tokens)} tok`];
    if (isNum(usage.prompt_per_second)) prefill.push(`${fmtNum(usage.prompt_per_second)} tok/s`);
    if (isNum(usage.prompt_eval_duration_ms)) prefill.push(fmtMs(usage.prompt_eval_duration_ms));
    parts.push(`prefill ${prefill.join(' · ')}`);

    const decode = [`${fmtInt(usage.completion_tokens)} tok`];
    if (isNum(usage.predicted_per_second)) decode.push(`${fmtNum(usage.predicted_per_second)} tok/s`);
    if (isNum(usage.generation_duration_ms)) decode.push(fmtMs(usage.generation_duration_ms));
    parts.push(`decode ${decode.join(' · ')}`);

    if (isNum(usage.cached_tokens)) parts.push(`cached ${fmtInt(usage.cached_tokens)} tok`);
  }
  if (requestId) parts.push(requestId);
  return parts.join('  ·  ');
}

export function mount(container) {
  let messages = [];
  try {
    const raw = localStorage.getItem(CHAT_KEY);
    if (raw) messages = JSON.parse(raw) || [];
  } catch {
    messages = [];
  }
  const settings = load(SETTINGS_KEY, DEFAULT_SETTINGS);

  let modelId = null;
  let controller = null;
  let disposed = false;
  /** The id of the generation currently on the wire, or null. */
  let liveRequestId = null;

  // Two tiers, because one is not enough. Aborting the fetch closes the
  // socket — which the server now notices — but a proxy can swallow
  // that, and an unloading page may never send it at all. The explicit
  // POST carries `keepalive`, so it survives the unload that killed the
  // stream. Both end at the same server-side flag, so doing both is
  // never worse than doing either.
  function stopGenerating() {
    const id = liveRequestId;
    controller?.abort();
    cancelGeneration(id);
  }

  // The tab closing mid-answer is precisely the case an AbortSignal
  // cannot cover: the page is gone before the abort is delivered.
  const onPageHide = () => {
    if (controller) cancelGeneration(liveRequestId);
  };
  window.addEventListener('pagehide', onPageHide);

  const list = el('div', { class: 'messages', id: 'messages' });
  const banner = el('div');

  const input = el('textarea', {
    rows: 2,
    placeholder: 'Message…  (Enter to send, Shift+Enter for a newline)',
    'aria-label': 'Message',
    onkeydown: (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        send();
      }
    },
    oninput: () => {
      input.style.height = 'auto';
      input.style.height = `${Math.min(input.scrollHeight, 224)}px`;
    },
  });

  const sendButton = el('button', { class: 'primary', type: 'button', text: 'Send', onclick: () => send() });
  const stopButton = el('button', {
    type: 'button',
    text: 'Stop',
    hidden: true,
    onclick: () => stopGenerating(),
  });

  const field = (label, attrs) => {
    const control = el('input', attrs);
    return { control, node: el('label', {}, [label, control]) };
  };

  const system = el('textarea', {
    rows: 2,
    placeholder: 'You are a helpful assistant.',
    'aria-label': 'System prompt',
  });
  system.value = settings.system;
  const temperature = field('temperature', { type: 'number', min: '0', max: '2', step: '0.05', value: settings.temperature });
  const topP = field('top_p', { type: 'number', min: '0', max: '1', step: '0.05', value: settings.topP });
  const maxTokens = field('max_tokens', { type: 'number', min: '1', max: '32768', step: '1', value: settings.maxTokens });

  const persistSettings = () => {
    settings.system = system.value;
    settings.temperature = Number(temperature.control.value);
    settings.topP = Number(topP.control.value);
    settings.maxTokens = Math.max(1, Math.round(Number(maxTokens.control.value) || 1));
    save(SETTINGS_KEY, settings);
  };
  for (const node of [system, temperature.control, topP.control, maxTokens.control]) {
    node.addEventListener('change', persistSettings);
  }

  const modelLabel = el('span', { class: 'small muted', text: 'model: …' });

  const controls = el('details', { class: 'settings panel' }, [
    el('summary', { text: 'Sampling & system prompt' }),
    el('div', { class: 'controls' }, [
      temperature.node,
      topP.node,
      maxTokens.node,
      el('label', { class: 'spacer' }, ['system prompt', system]),
    ]),
    el('div', { class: 'row small faint' }, [
      modelLabel,
      el('span', { class: 'spacer' }),
      el('span', { text: 'Transcript is stored in this browser only.' }),
      el('button', {
        class: 'small',
        type: 'button',
        text: 'New chat',
        onclick: () => {
          stopGenerating();
          messages = [];
          save(CHAT_KEY, messages);
          paint();
        },
      }),
    ]),
  ]);

  container.appendChild(
    el('div', { class: 'screen chat' }, [
      banner,
      controls,
      list,
      el('div', { class: 'composer' }, [input, sendButton, stopButton]),
    ]),
  );

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function bubble(message) {
    const body = el('div', { class: 'md msg-body' });
    body.appendChild(renderMarkdown(message.content || ''));
    const node = el('div', { class: `msg msg-${message.role}${message.error ? ' msg-error' : ''}` }, [
      el('div', { class: 'msg-role' }, [
        message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Assistant',
        message.role === 'assistant' && message.content
          ? el('span', {}, [' ', copyButton('copy', () => message.content)])
          : null,
      ]),
      body,
    ]);
    if (message.stats) node.appendChild(el('div', { class: 'msg-stats', text: message.stats }));
    return node;
  }

  function paint() {
    clear(list);
    if (!messages.length) {
      list.appendChild(
        el('p', { class: 'empty' }, [
          'Nothing here yet. This screen talks to ',
          el('code', { text: '/v1/chat/completions' }),
          ' with ',
          el('code', { text: 'stream: true' }),
          ' — the same endpoint any other client uses.',
        ]),
      );
      return;
    }
    for (const message of messages) list.appendChild(bubble(message));
    list.scrollTop = list.scrollHeight;
  }

  function setBusy(busy) {
    sendButton.disabled = busy;
    stopButton.hidden = !busy;
    input.disabled = busy;
  }

  function notice(text, kind = 'notice-warn') {
    fill(banner, el('p', { class: `notice ${kind}`, text }));
  }

  // ------------------------------------------------------------------
  // Sending
  // ------------------------------------------------------------------

  async function send() {
    const text = input.value.trim();
    if (!text || controller) return;
    persistSettings();

    input.value = '';
    input.style.height = 'auto';
    messages.push({ role: 'user', content: text });

    const reply = { role: 'assistant', content: '', stats: null };
    messages.push(reply);
    paint();

    const replyNode = list.lastElementChild;
    const replyBody = replyNode.querySelector('.msg-body');
    replyBody.classList.add('cursor');

    // Re-render markdown at most once a frame: the transcript is
    // rebuilt from source text on every token, and doing that
    // synchronously per token makes a fast decode loop stutter.
    let scheduled = false;
    const repaintBody = () => {
      if (scheduled || disposed) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        clear(replyBody).appendChild(renderMarkdown(reply.content));
        list.scrollTop = list.scrollHeight;
      });
    };

    const wire = [];
    if (settings.system.trim()) wire.push({ role: 'system', content: settings.system.trim() });
    for (const message of messages) {
      if (message === reply || message.error) continue;
      wire.push({ role: message.role, content: message.content });
    }

    controller = new AbortController();
    setBusy(true);
    let requestId = null;
    try {
      const result = await streamChat(
        {
          model: modelId || 'ferrox',
          messages: wire,
          temperature: settings.temperature,
          top_p: settings.topP,
          max_tokens: settings.maxTokens,
        },
        {
          signal: controller.signal,
          onRequestId: (id) => {
            requestId = id;
            // Named on the first chunk, which is what makes an explicit
            // cancel possible at all — there is nothing to cancel by
            // before the server has said what this generation is called.
            liveRequestId = id;
          },
          onToken: (token) => {
            reply.content += token;
            repaintBody();
          },
          // A stream that has gone quiet is not the same as a slow
          // model — the server sends a keep-alive comment every 15 s,
          // so silence on the wire means the connection, not the
          // decode. Said out loud rather than left as a spinner that
          // never resolves; `null` means it recovered.
          onStall: (ms) => {
            if (disposed) return;
            if (ms === null) clear(banner);
            else
              notice(
                `No data for ${Math.round(ms / 1000)}s. The generation may still be running — `
                  + 'a proxy between you and the server may be buffering text/event-stream. '
                  + 'Stop cancels it on the server, not just here.',
              );
          },
        },
      );
      const line = statLine(result?.usage, result?.requestId || requestId);
      // The server won the race: it noticed the cancel and closed the
      // stream cleanly, so this arrives as a finished response rather
      // than as an AbortError. Saying so is the difference between a
      // short answer and a truncated one, which look identical.
      reply.stats =
        result?.finishReason === 'cancelled' ? `stopped  ·  ${line}` : line;
    } catch (error) {
      if (error?.name === 'AbortError') {
        // A stopped generation is not a failure: the tokens that did
        // arrive are kept, and the line says why there are no timings.
        reply.stats = `stopped by you${requestId ? `  ·  ${requestId}` : ''}`;
        if (!reply.content) reply.content = '_(stopped before any token arrived)_';
      } else if (error instanceof ApiError && error.isAuth) {
        reply.error = true;
        reply.content = `${error.message}\n\nThis server requires an API key. Set it on the Connect screen.`;
      } else {
        reply.error = true;
        reply.content = String(error?.message || error);
      }
    } finally {
      controller = null;
      liveRequestId = null;
      if (!disposed) {
        setBusy(false);
        save(CHAT_KEY, messages);
        paint();
        input.focus();
      }
    }
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------

  paint();
  input.focus();

  getJson(routes.models)
    .then((body) => {
      if (disposed) return;
      modelId = body?.data?.[0]?.id || null;
      setText(modelLabel, `model: ${modelId || 'none loaded'}`);
      if (!modelId) {
        notice('No model is loaded, so a message would fail. Load one on the Models screen.');
      } else if (body.data[0].ferrox_synthetic_weights) {
        notice(
          `"${modelId}" is running on synthetic random weights — the output is noise, not a bad model.`,
        );
      }
    })
    .catch((error) => {
      if (disposed) return;
      setText(modelLabel, 'model: unknown');
      notice(`Could not read ${routes.models}: ${error.message}`, 'notice-error');
    });

  return () => {
    disposed = true;
    // Navigating away from Chat is a cancel too: the tokens have
    // nowhere left to land, and letting the backend keep decoding for
    // a screen nobody is looking at is the exact waste this feature
    // exists to end.
    stopGenerating();
    window.removeEventListener('pagehide', onPageHide);
  };
}
