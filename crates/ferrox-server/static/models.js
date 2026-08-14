// Models: the inventory, the swap, and the download job.
//
// Two rules this screen exists to respect.
//
// **A rate is shown only when the server calls the task `stable`.** The
// backend runs a rolling-window estimator that refuses to divide until
// it has enough samples, and sends `null` for rate and ETA until then.
// Recomputing either from `bytes_done` deltas on this side would put
// back exactly the "123 GB/s" first-tick flash the estimator exists to
// prevent, so nothing here ever divides.
//
// **A missing control surface is a state, not a crash.** `/admin/*` is
// only present in builds that have it; a 404 renders as a plain
// explanation rather than as a broken table.

import {
  el, mount as fill, clear, fmtBytes, fmtInt, fmtParams, fmtRate, fmtDuration, isNum,
} from './dom.js';
import { getJson, postJson, routes, ApiError } from './api.js';

export const title = 'Models';

/** Poll fast while something is moving, slowly when nothing is. */
const BUSY_POLL_MS = 1000;
const IDLE_POLL_MS = 5000;

export function mount(container) {
  let disposed = false;
  let timer = null;
  let inventory = null;
  let tasks = [];
  let unsupported = null;
  let banner = null;

  const inventoryPanel = el('div', { class: 'panel' });
  const tasksPanel = el('div', { class: 'panel' });
  const bannerSlot = el('div');

  const repoInput = el('input', { type: 'text', placeholder: 'unsloth/Llama-3.2-3B-Instruct-GGUF', required: true, size: 34 });
  const fileInput = el('input', { type: 'text', placeholder: '*Q4_K_M.gguf', value: '*Q4_K_M.gguf', required: true, size: 20 });
  const downloadButton = el('button', { class: 'primary', type: 'submit', text: 'Download' });

  const downloadForm = el('form', {
    class: 'row',
    onsubmit: async (event) => {
      event.preventDefault();
      downloadButton.disabled = true;
      try {
        // The server resolves a `*` glob against the repo's file list
        // and refuses anything that is not a plain `.gguf` child of the
        // model directory, so no validation is duplicated here.
        await postJson(routes.adminDownload, { repo: repoInput.value.trim(), file: fileInput.value.trim() });
        setBanner(`Download queued for ${repoInput.value.trim()}.`, 'notice');
        await refresh();
      } catch (error) {
        setBanner(`Download refused: ${error.message}`, 'notice-error');
      } finally {
        if (!disposed) downloadButton.disabled = false;
      }
    },
  }, [
    el('label', {}, ['Hugging Face repo', repoInput]),
    el('label', {}, ['file (name or glob)', fileInput]),
    el('div', {}, [downloadButton]),
  ]);

  const downloadPanel = el('div', { class: 'panel' }, [
    el('h2', { class: 'panel-title', text: 'Download a checkpoint' }),
    downloadForm,
    el('p', { class: 'small faint', text: 'POST /admin/download starts a task; progress appears below.' }),
  ]);

  container.appendChild(
    el('div', { class: 'screen' }, [bannerSlot, inventoryPanel, downloadPanel, tasksPanel]),
  );

  function setBanner(text, kind) {
    banner = text ? { text, kind } : null;
    if (banner) fill(bannerSlot, el('p', { class: `notice ${banner.kind}`, text: banner.text }));
    else clear(bannerSlot);
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------

  async function act(label, run) {
    try {
      await run();
      setBanner(null);
    } catch (error) {
      setBanner(`${label} failed: ${error.message}`, 'notice-error');
    }
    await refresh();
  }

  const loadModel = (id) => act(`Loading ${id}`, () => postJson(routes.adminModelsLoad, { id }));
  const unloadModel = () => act('Unload', () => postJson(routes.adminModelsUnload));
  const cancelTask = (taskId) => act('Cancel', () => postJson(routes.adminTaskCancel(taskId)));

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function notAvailable(node, what) {
    fill(
      node,
      el('h2', { class: 'panel-title', text: what }),
      el('p', { class: 'notice notice-warn' }, [
        'Not available in this build. This server answered ',
        el('code', { text: '404' }),
        ' for the /admin control surface, so model inventory, loading and downloads cannot be driven from here. Chat and Activity are unaffected.',
      ]),
    );
  }

  function stateBadge(entry, activeId) {
    const state = entry.id === activeId ? 'loaded' : entry.state;
    const classes = { loaded: 'badge-loaded', loading: 'badge-loading', error: 'badge-error' };
    return el('span', {
      class: `badge ${classes[state] || ''}`,
      text: state,
      title: entry.error || '',
    });
  }

  function paintInventory() {
    if (unsupported) return notAvailable(inventoryPanel, 'Model inventory');
    if (!inventory) {
      return fill(inventoryPanel, el('h2', { class: 'panel-title', text: 'Model inventory' }), el('p', { class: 'muted', text: 'Loading…' }));
    }

    const active = inventory.active;
    const head = el('div', { class: 'row small' }, [
      el('span', { class: 'muted', text: `scanned: ${inventory.model_dir || 'no model directory configured'}` }),
      el('span', { class: 'spacer' }),
      el('span', { text: active ? `active: ${active}` : 'nothing loaded' }),
      active ? el('button', { class: 'small', type: 'button', text: 'Unload', onclick: unloadModel }) : null,
      el('button', { class: 'small', type: 'button', text: 'Refresh', onclick: () => refresh() }),
    ]);

    if (!inventory.models.length) {
      return fill(
        inventoryPanel,
        el('h2', { class: 'panel-title', text: 'Model inventory' }),
        head,
        el('p', { class: 'muted', text: inventory.model_dir
          ? 'No .gguf checkpoints in the scanned directory. Download one below.'
          : 'No model directory is configured — set FERROX_MODEL_PATH or FERROX_MODEL_DIR.' }),
      );
    }

    const rows = inventory.models.map((entry) => {
      const loaded = entry.id === active;
      const busy = entry.state === 'loading' || inventory.models.some((m) => m.state === 'loading');
      return el('tr', {}, [
        el('td', {}, [el('span', { class: 'mono', text: entry.id, title: entry.path })]),
        el('td', { text: entry.quant || '—' }),
        el('td', { text: entry.arch || '—' }),
        el('td', { class: 'num', text: isNum(entry.context_length) ? fmtInt(entry.context_length) : '—' }),
        el('td', { class: 'num', text: fmtParams(entry.param_count) }),
        el('td', { class: 'num', text: fmtBytes(entry.size_bytes) }),
        // `resident_bytes` is null for anything the server cannot
        // measure; that is reported as unknown rather than as the file
        // size, which would be a guess dressed as a measurement.
        el('td', { class: 'num', text: isNum(entry.resident_bytes) ? fmtBytes(entry.resident_bytes) : '—' }),
        el('td', {}, [stateBadge(entry, active)]),
        el('td', {}, [
          loaded
            ? el('button', { class: 'small', type: 'button', text: 'Unload', onclick: unloadModel })
            : el('button', {
                class: 'small',
                type: 'button',
                text: 'Load',
                disabled: busy,
                title: busy ? 'a load is already in progress' : '',
                onclick: () => loadModel(entry.id),
              }),
        ]),
      ]);
    });

    fill(
      inventoryPanel,
      el('h2', { class: 'panel-title', text: 'Model inventory' }),
      head,
      el('div', { class: 'table-wrap' }, [
        el('table', {}, [
          el('thead', {}, [
            el('tr', {}, [
              el('th', { text: 'id' }),
              el('th', { text: 'quant' }),
              el('th', { text: 'arch' }),
              el('th', { class: 'num', text: 'context' }),
              el('th', { class: 'num', text: 'params' }),
              el('th', { class: 'num', text: 'on disk' }),
              el('th', { class: 'num', text: 'resident' }),
              el('th', { text: 'state' }),
              el('th', { text: '' }),
            ]),
          ]),
          el('tbody', {}, rows),
        ]),
      ]),
    );
  }

  function paintTask(task) {
    const p = task.progress || {};
    const fraction = isNum(p.fraction) ? p.fraction : null;
    const bar = el('div', { class: `bar${fraction === null ? ' indeterminate' : ''}` }, [
      el('span', { style: fraction === null ? '' : `width:${(fraction * 100).toFixed(1)}%` }),
    ]);

    const facts = [`${p.bytes_done ? fmtBytes(p.bytes_done) : '0 B'}${isNum(p.bytes_total) ? ` / ${fmtBytes(p.bytes_total)}` : ''}`];
    if (p.state === 'stable') {
      // Only here. `warming` means the server declined to estimate, and
      // the honest render of that is the word, not a number.
      facts.push(fmtRate(p.rate_bytes_per_s));
      if (isNum(p.eta_seconds)) facts.push(`ETA ${fmtDuration(p.eta_seconds)}`);
    } else if (task.status === 'running') {
      facts.push('measuring rate…');
    }
    if (task.error) facts.push(task.error);

    const terminal = ['done', 'error', 'cancelled'].includes(task.status);
    return el('div', { class: 'task' }, [
      el('div', { class: 'row small' }, [
        el('strong', { text: task.label }),
        el('span', { class: `badge${task.status === 'error' ? ' badge-error' : task.status === 'done' ? ' badge-loaded' : ''}`, text: task.status }),
        el('span', { class: 'spacer' }),
        terminal ? null : el('button', { class: 'small danger', type: 'button', text: 'Cancel', onclick: () => cancelTask(task.task_id) }),
      ]),
      terminal ? null : bar,
      el('div', { class: 'small faint mono', text: facts.join('  ·  ') }),
    ]);
  }

  function paintTasks() {
    if (unsupported) return notAvailable(tasksPanel, 'Tasks');
    fill(
      tasksPanel,
      el('h2', { class: 'panel-title', text: 'Tasks' }),
      tasks.length
        ? el('div', {}, tasks.map(paintTask))
        : el('p', { class: 'muted small', text: 'No downloads or loads have run yet.' }),
    );
  }

  // ------------------------------------------------------------------
  // Polling
  // ------------------------------------------------------------------

  async function refresh() {
    if (disposed) return;
    try {
      const [models, taskList] = await Promise.all([
        getJson(routes.adminModels),
        getJson(routes.adminTasks),
      ]);
      if (disposed) return;
      inventory = models;
      tasks = taskList.tasks || [];
      unsupported = null;
    } catch (error) {
      if (disposed) return;
      if (error instanceof ApiError && error.isMissingEndpoint) {
        unsupported = error;
      } else {
        setBanner(`Could not read the control surface: ${error.message}`, 'notice-error');
      }
    }
    paintInventory();
    paintTasks();
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    if (disposed || unsupported) return;
    const busy =
      tasks.some((t) => t.status === 'queued' || t.status === 'running') ||
      (inventory?.models || []).some((m) => m.state === 'loading');
    timer = setTimeout(refresh, busy ? BUSY_POLL_MS : IDLE_POLL_MS);
  }

  paintInventory();
  paintTasks();
  refresh();

  return () => {
    disposed = true;
    clearTimeout(timer);
  };
}
