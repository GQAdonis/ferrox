import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import {
  AssistantRuntimeProvider,
  type AssistantRuntime,
} from "@assistant-ui/react";
import * as Popover from "@radix-ui/react-popover";
import { Link, useOutletContext } from "react-router";
import { Check, ChevronDown, Loader2, SlidersHorizontal, SquarePen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, Input, Textarea } from "@/components/ui/field";
import { Notice } from "@/components/ui/feedback";
import { Badge } from "@/components/ui/badge";
import {
  ApiError,
  getJson,
  postJson,
  routes,
  type Inventory,
} from "@/lib/api";
import type { HealthState } from "@/lib/use-health";
import { fmtBytes } from "@/lib/format";
import { useLatest } from "@/lib/use-latest";
import { cn } from "@/lib/utils";
import { Thread } from "@/screens/chat/thread";
import {
  DEFAULT_SAMPLING,
  useFerroxRuntime,
  type Sampling,
} from "@/screens/chat/runtime";
import { clearTranscript, usePersistedThread } from "@/screens/chat/persistence";

const SETTINGS_KEY = "ferrox.studio.sampling.v1";

function loadSampling(): Sampling {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw
      ? { ...DEFAULT_SAMPLING, ...JSON.parse(raw) }
      : { ...DEFAULT_SAMPLING };
  } catch {
    return { ...DEFAULT_SAMPLING };
  }
}

function saveSampling(value: Sampling) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(value));
  } catch {
    /* quota or private browsing — the in-memory settings still apply */
  }
}

type Loaded = {
  modelId: string | null;
  synthetic: boolean;
  error: string | null;
};

/**
 * What `/v1/models` says is serving right now, kept current.
 *
 * Re-read whenever `/health` reports a different model, which the shell
 * already polls every five seconds — so no second poller is added, and
 * a model loaded from the Models screen (or by anything else talking to
 * this server) un-gates the composer on its own. Reading it once at
 * mount was survivable when the id was only a label; now that an empty
 * one blocks Send, a stale "no model loaded" would strand the user on a
 * server that is ready.
 */
function useServingModel(healthModelId: string | null): [Loaded, () => void] {
  const [state, setState] = useState<Loaded>({
    modelId: null,
    synthetic: false,
    error: null,
  });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getJson<{ data?: { id: string; ferrox_synthetic_weights?: boolean }[] }>(
      routes.models,
    )
      .then((body) => {
        if (cancelled) return;
        const first = body?.data?.[0];
        setState({
          modelId: first?.id ?? null,
          synthetic: !!first?.ferrox_synthetic_weights,
          error: null,
        });
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setState({ modelId: null, synthetic: false, error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, [nonce, healthModelId]);

  return [state, useCallback(() => setNonce((n) => n + 1), [])];
}

/**
 * Switch the served model without leaving the conversation.
 *
 * A swap is a real server-side action (`POST /admin/models/load`), not a
 * per-request parameter — this server serves one checkpoint at a time.
 * The menu says so rather than implying the next message could pick a
 * different model on its own.
 */
function ModelSwitcher({
  active,
  onSwitched,
}: {
  active: string | null;
  onSwitched: () => void;
}) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    getJson<Inventory>(routes.adminModels)
      .then(setInventory)
      .catch((e) => {
        if (e instanceof ApiError && e.isMissingEndpoint) setUnsupported(true);
      });
  }, []);

  const swap = async (id: string) => {
    setBusy(id);
    setError(null);
    try {
      await postJson(routes.adminModelsLoad, { id });
      onSwitched();
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Popover.Root onOpenChange={(open) => open && refresh()}>
      <Popover.Trigger asChild>
        <Button variant="default" size="sm" className="max-w-[16rem]">
          <span className="truncate font-mono text-[0.6875rem]">
            {active ?? "no model loaded"}
          </span>
          <ChevronDown className="text-faint" />
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(24rem,calc(100vw-1.5rem))] rounded-card border border-line bg-raised p-1.5 shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          {unsupported ? (
            <p className="p-2 text-xs text-faint">
              This build has no <code className="font-mono">/admin</code>{" "}
              control surface, so the model cannot be swapped from here.
            </p>
          ) : !inventory ? (
            <p className="p-2 text-xs text-faint">Reading the inventory…</p>
          ) : !inventory.models.length ? (
            <p className="p-2 text-xs text-faint">
              No checkpoints found.{" "}
              <Link to="/ui/models" className="text-accent underline">
                Download one
              </Link>
              .
            </p>
          ) : (
            <ul className="max-h-72 space-y-0.5 overflow-y-auto">
              {inventory.models.map((entry) => {
                const isActive = entry.id === inventory.active;
                return (
                  <li key={entry.id}>
                    <button
                      type="button"
                      disabled={isActive || !!busy}
                      onClick={() => swap(entry.id)}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors",
                        isActive
                          ? "bg-accent-soft text-accent"
                          : "hover:bg-inset disabled:opacity-50",
                      )}
                    >
                      {busy === entry.id ? (
                        <Loader2 className="size-3.5 shrink-0 animate-spin" />
                      ) : isActive ? (
                        <Check className="size-3.5 shrink-0" />
                      ) : (
                        <span className="size-3.5 shrink-0" />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-xs">
                          {entry.id}
                        </span>
                        <span className="block truncate text-[0.6875rem] text-faint">
                          {[entry.quant, entry.arch, fmtBytes(entry.size_bytes)]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {error ? (
            <p className="mt-1 rounded-lg bg-err-soft px-2 py-1.5 text-[0.6875rem] text-err">
              {error}
            </p>
          ) : null}
          <p className="mt-1 border-t border-line px-2 pt-1.5 text-[0.6875rem] text-faint">
            Loading a checkpoint swaps it for every client of this server. A
            request already in flight finishes on the weights it started on.
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function SamplingPanel({
  value,
  onChange,
}: {
  value: Sampling;
  onChange: (next: Sampling) => void;
}) {
  const set = <K extends keyof Sampling>(key: K, next: Sampling[K]) =>
    onChange({ ...value, [key]: next });

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button variant="default" size="sm">
          <SlidersHorizontal />
          <span className="hidden sm:inline">Sampling</span>
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(24rem,calc(100vw-1.5rem))] space-y-3 rounded-card border border-line bg-raised p-3 shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          <div className="grid grid-cols-3 gap-2">
            <Field label="temperature">
              <Input
                type="number"
                min="0"
                max="2"
                step="0.05"
                value={value.temperature}
                onChange={(e) => set("temperature", Number(e.target.value))}
              />
            </Field>
            <Field label="top_p">
              <Input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={value.topP}
                onChange={(e) => set("topP", Number(e.target.value))}
              />
            </Field>
            <Field label="max_tokens">
              <Input
                type="number"
                min="1"
                max="32768"
                step="1"
                value={value.maxTokens}
                onChange={(e) =>
                  set(
                    "maxTokens",
                    Math.max(1, Math.round(Number(e.target.value) || 1)),
                  )
                }
              />
            </Field>
          </div>
          <Field
            label="system prompt"
            hint="Sent as the first message of every request, not stored on the server."
          >
            <Textarea
              rows={3}
              placeholder="You are a helpful assistant."
              value={value.system}
              onChange={(e) => set("system", e.target.value)}
            />
          </Field>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function ChatInner({
  sampling,
  setSampling,
  serving,
  refreshServing,
  stall,
}: {
  sampling: Sampling;
  setSampling: (next: Sampling) => void;
  serving: Loaded;
  refreshServing: () => void;
  stall: string | null;
}) {
  const runtime = useFerroxRuntimeContext();
  usePersistedThread(runtime);

  const disabledReason = serving.modelId
    ? null
    : serving.error
      ? `Could not read ${routes.models}: ${serving.error}`
      : "No model is loaded — load one on the Models screen before sending.";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 flex-wrap items-center gap-2 border-b border-line bg-raised/70 px-4 py-2.5 backdrop-blur">
        <h1 className="text-sm font-semibold tracking-tight">Chat</h1>
        {serving.synthetic ? (
          <Badge tone="err">synthetic weights</Badge>
        ) : null}
        <span className="flex-1" />
        <ModelSwitcher active={serving.modelId} onSwitched={refreshServing} />
        <SamplingPanel value={sampling} onChange={setSampling} />
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            runtime.thread.cancelRun();
            runtime.thread.reset();
            clearTranscript();
          }}
        >
          <SquarePen />
          <span className="hidden sm:inline">New chat</span>
        </Button>
      </header>

      {stall || serving.synthetic || disabledReason ? (
        <div className="shrink-0 space-y-2 border-b border-line bg-raised/40 px-4 py-2.5">
          {disabledReason ? (
            <Notice tone="warn">
              {disabledReason}{" "}
              <Link to="/ui/models" className="underline underline-offset-2">
                Open Models
              </Link>
            </Notice>
          ) : null}
          {serving.synthetic ? (
            <Notice tone="err">
              <code className="font-mono">{serving.modelId}</code> is running on
              synthetic random weights — the output is noise, not a bad model.
            </Notice>
          ) : null}
          {stall ? <Notice tone="warn">{stall}</Notice> : null}
        </div>
      ) : null}

      <div className="min-h-0 flex-1">
        <Thread
          disabledReason={disabledReason}
          footer={
            <p className="text-center text-[0.6875rem] text-faint">
              Transcript is stored in this browser only — this server has no
              conversation API.
            </p>
          }
        />
      </div>
    </div>
  );
}

// `useFerroxRuntime` must run above `AssistantRuntimeProvider`, and
// `usePersistedThread` needs the same object; passing it down through a
// tiny context keeps both without prop-drilling the runtime into every
// child that happens to sit under the provider.
const RuntimeContext = createContext<AssistantRuntime | null>(null);
function useFerroxRuntimeContext(): AssistantRuntime {
  const runtime = useContext(RuntimeContext);
  if (!runtime) throw new Error("no runtime in scope");
  return runtime;
}

export function ChatScreen() {
  const health = useOutletContext<HealthState>();
  const [sampling, setSamplingState] = useState<Sampling>(loadSampling);
  const [serving, refreshServing] = useServingModel(
    health?.health?.model?.id ?? null,
  );
  const [stall, setStall] = useState<string | null>(null);

  const samplingRef = useLatest(sampling);
  const servingRef = useLatest(serving);

  const setSampling = useCallback((next: Sampling) => {
    setSamplingState(next);
    saveSampling(next);
  }, []);

  const runtime = useFerroxRuntime({
    modelId: () => servingRef.current.modelId,
    sampling: () => samplingRef.current,
    // A stream that has gone quiet is not the same as a slow model — the
    // server sends a keep-alive comment every 15 s, so silence on the
    // wire means the connection, not the decode. Said out loud rather
    // than left as a spinner that never resolves; `null` means it
    // recovered, and the banner comes down.
    onStall: (ms) =>
      setStall(
        ms === null
          ? null
          : `No data for ${Math.round(ms / 1000)}s. The generation may still be running — ` +
              "a proxy between you and the server may be buffering text/event-stream. " +
              "Stop cancels it on the server, not just here.",
      ),
  });

  return (
    <RuntimeContext.Provider value={runtime}>
      <AssistantRuntimeProvider runtime={runtime}>
        <ChatInner
          sampling={sampling}
          setSampling={setSampling}
          serving={serving}
          refreshServing={refreshServing}
          stall={stall}
        />
      </AssistantRuntimeProvider>
    </RuntimeContext.Provider>
  );
}
