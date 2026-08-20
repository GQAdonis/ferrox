// Connect: the snippets that point something else at this server.
//
// Everything here is filled from what is live right now — the origin
// this page was served from and the model id `/v1/models` reports —
// rather than from a placeholder the reader has to remember to edit.
// A snippet with `YOUR_MODEL_HERE` in it is a snippet that gets pasted
// with `YOUR_MODEL_HERE` still in it.

import { el, mount as fill, clear, copyButton, setText } from './dom.js';
import { getJson, routes, baseUrl, apiKey, setApiKey } from './api.js';

export const title = 'Connect';

export function mount(container) {
  let disposed = false;
  let modelId = null;

  const base = baseUrl();
  const snippets = el('div', { class: 'panel' });
  const status = el('span', { class: 'small muted', text: 'reading /v1/models…' });

  const keyInput = el('input', {
    type: 'password',
    placeholder: 'unset — this server may not require one',
    value: apiKey(),
    size: 32,
    'aria-label': 'API key',
  });

  const keyPanel = el('div', { class: 'panel' }, [
    el('h2', { class: 'panel-title', text: 'Connection' }),
    el('div', { class: 'row' }, [
      el('label', {}, ['base URL', el('input', { type: 'text', readonly: true, value: base, size: 28, onfocus: (e) => e.target.select() })]),
      el('label', {}, ['API key (FERROX_API_KEY)', keyInput]),
      el('div', {}, [
        el('button', {
          type: 'button',
          text: 'Save key',
          onclick: () => {
            setApiKey(keyInput.value.trim());
            paint();
          },
        }),
      ]),
      el('span', { class: 'spacer' }),
      status,
    ]),
    el('p', { class: 'small faint', text:
      'The key is stored in this browser\'s localStorage and sent as an Authorization header, exactly as any other client would. '
      + 'Leave it empty when the server was started without FERROX_API_KEY. It is never put in a URL.' }),
  ]);

  container.appendChild(el('div', { class: 'screen' }, [keyPanel, snippets]));

  function snippet(heading, note, code) {
    return el('div', { class: 'snippet' }, [
      el('div', { class: 'snippet-head' }, [
        el('h3', { text: heading }),
        el('span', { class: 'spacer' }),
        copyButton('copy', () => code),
      ]),
      note ? el('p', { class: 'small faint', text: note }) : null,
      el('div', { class: 'codeblock' }, [el('pre', {}, [el('code', { text: code })])]),
    ]);
  }

  function paint() {
    const key = apiKey();
    const model = modelId || 'ferrox';
    const authHeaderCurl = key ? ` \\\n  -H "Authorization: Bearer ${key}"` : '';
    const pyKey = key ? `"${key}"` : 'os.environ.get("FERROX_API_KEY", "not-needed")';

    const curl = `curl ${base}${routes.chatCompletions} \\
  -H "Content-Type: application/json"${authHeaderCurl} \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 64,
    "stream": true
  }'`;

    const python = `import os
from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key=${pyKey},
)

stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)`;

    const env = `# Any OpenAI-compatible tool (editors, agents, SDKs)
OPENAI_BASE_URL=${base}/v1
OPENAI_API_KEY=${key || 'not-needed'}
# The model id this server is serving right now:
#   ${model}`;

    const health = `curl -s ${base}${routes.health} | jq
curl -s ${base}${routes.models} | jq
curl -s ${base}${routes.adminStats} | jq '.recent[-1]'`;

    fill(
      snippets,
      el('h2', { class: 'panel-title', text: 'Snippets' }),
      snippet('curl — streaming chat completion', 'The same endpoint this Studio\'s Chat screen uses.', curl),
      snippet('Python — the official OpenAI SDK', 'Point base_url at /v1; nothing else changes.', python),
      snippet('Environment — editors and agents', 'See docs/AGENTS_COOKBOOK.md for per-tool configuration.', env),
      snippet('Probes', 'Liveness, what is loaded, and the most recent request.', health),
    );
  }

  paint();

  getJson(routes.models)
    .then((body) => {
      if (disposed) return;
      modelId = body?.data?.[0]?.id || null;
      setText(status, modelId ? `serving: ${modelId}` : 'no model loaded — snippets use the placeholder id "ferrox"');
      paint();
    })
    .catch((error) => {
      if (disposed) return;
      setText(status, `could not read ${routes.models}: ${error.message}`);
    });

  return () => {
    disposed = true;
    clear(container);
  };
}
