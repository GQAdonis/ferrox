import { useCallback, useEffect, useRef, useState } from "react";
import {
  ExportedMessageRepository,
  type AssistantRuntime,
} from "@assistant-ui/react";
import { ApiError } from "@/lib/api";
import {
  createConversation,
  deleteConversation,
  getConversation,
  hasWork,
  listConversations,
  pendingAppend,
  storedIds,
  toBranchable,
  updateConversation,
  type ConversationSummary,
  type ExportedRepository,
} from "@/lib/conversations";
import { useLatest } from "@/lib/use-latest";

// Where the transcript lives.
//
// Server-side when the server has `/v1/conversations`, which it now
// does: the tree is stored there, keyed by message id and parent id, so
// it survives a reload, a different browser and a cleared profile, and
// edit/regenerate branches survive with it.
//
// `localStorage` is still here as the fallback for a server that
// answers 404 on that route -- an older build, or this app pointed at
// something else that speaks the OpenAI API. The screen says which of
// the two is in use, because "your chats are saved" and "your chats are
// saved in this browser" are different promises and only one of them
// survives a laptop.
//
// The one thing neither mode does is guess. When the store cannot be
// reached, nothing is silently dropped and nothing is silently kept
// somewhere else: the mode is reported and the reason with it.

const LOCAL_KEY = "ferrox.studio.thread.v2";

/** Anything unparseable is dropped: a corrupt blob must not wedge Chat. */
function readLocal(): { messages?: unknown[] } | null {
  try {
    const raw = localStorage.getItem(LOCAL_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeLocal(exported: { messages: unknown[] }) {
  try {
    if (!exported.messages.length) clearLocalTranscript();
    else localStorage.setItem(LOCAL_KEY, JSON.stringify(exported));
  } catch {
    /* quota or private browsing — the in-memory thread still works */
  }
}

export function clearLocalTranscript() {
  try {
    localStorage.removeItem(LOCAL_KEY);
  } catch {
    /* private browsing: nothing was persisted to begin with */
  }
}

/** How long the transcript sits still before it is written. A decode
 * loop notifies once per token; writing on each would put a request
 * between every pair of them. */
const DEBOUNCE_MS = 500;

export type TranscriptMode = "checking" | "server" | "local";

export type Transcript = {
  mode: TranscriptMode;
  /** Why the transcript is browser-local, when it is. */
  reason: string | null;
  /** A write is in flight. */
  saving: boolean;
  /** The last write or read that failed, as a sentence. */
  error: string | null;
  current: { id: string; title: string | null } | null;
  summaries: ConversationSummary[];
  refresh: () => void;
  open: (id: string) => void;
  newChat: () => void;
  remove: (id: string) => void;
};

type SyncState = {
  mode: TranscriptMode;
  conversationId: string | null;
  /** Ids the server is holding. Only ever grown from what the server
   * echoed back, never from what was sent -- a request that failed
   * halfway must not leave this claiming the messages landed. */
  stored: Set<string>;
  storedHead: string | null;
  busy: boolean;
  dirty: boolean;
  /** Writes are held off while the thread is being replaced from the
   * outside (a load, a reset), so an import is not immediately written
   * back as if the user had typed it. */
  suspended: boolean;
  migrating: boolean;
};

function sentence(cause: unknown): string {
  if (cause instanceof ApiError && cause.isAuth)
    return `${cause.message} — set the API key on the Connect screen.`;
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * Keep the thread and the server's conversation store in step.
 *
 * The loop is one-directional by design: assistant-ui owns the
 * transcript in the tab, and every change is pushed to the server as an
 * append. Nothing is ever pulled back into a live thread, because two
 * writers on one transcript is how a message gets lost, and this server
 * serves one person's Chat screen.
 */
export function useTranscript(
  runtime: AssistantRuntime,
  deps: { model: () => string | null },
): Transcript {
  const [mode, setMode] = useState<TranscriptMode>("checking");
  const [reason, setReason] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<{
    id: string;
    title: string | null;
  } | null>(null);
  const [summaries, setSummaries] = useState<ConversationSummary[]>([]);

  const depsRef = useLatest(deps);
  const sync = useRef<SyncState>({
    mode: "checking",
    conversationId: null,
    stored: new Set(),
    storedHead: null,
    busy: false,
    dirty: false,
    suspended: true,
    migrating: false,
  });

  const setBoth = useCallback((next: TranscriptMode) => {
    sync.current.mode = next;
    setMode(next);
  }, []);

  const refresh = useCallback(() => {
    if (sync.current.mode !== "server") return;
    listConversations()
      .then(setSummaries)
      .catch(() => {
        // A failed listing is not worth an error banner: the store is
        // still writable, and the next refresh will say otherwise if it
        // is not.
      });
  }, []);

  const flush = useCallback(async () => {
    const s = sync.current;
    if (s.suspended) return;

    if (s.mode === "local") {
      writeLocal(runtime.thread.export() as { messages: unknown[] });
      return;
    }
    if (s.mode !== "server") return;
    if (s.busy) {
      s.dirty = true;
      return;
    }

    // assistant-ui's exported shape is wider than the four fields the
    // store reads; the cast names that rather than restating the
    // library's type here, where it would drift.
    const exported = runtime.thread.export() as unknown as ExportedRepository;
    const pending = pendingAppend(exported, s.stored);
    if (!hasWork(pending, s.storedHead)) return;

    s.busy = true;
    setSaving(true);
    try {
      const head = pending.headId ? { head_id: pending.headId } : {};
      const conversation = s.conversationId
        ? await updateConversation(s.conversationId, {
            append: pending.messages,
            ...head,
          })
        : await createConversation({
            messages: pending.messages,
            ...head,
            // Recorded, not enforced. It says which checkpoint was
            // loaded when the conversation started, which is a fact
            // worth keeping when the answer is later read back.
            ...(depsRef.current.model() ? { model: depsRef.current.model()! } : {}),
          });
      s.conversationId = conversation.id;
      s.stored = storedIds(conversation);
      s.storedHead = conversation.head_id;
      setCurrent({ id: conversation.id, title: conversation.title });
      setError(null);
      if (s.migrating) {
        // The browser copy is dropped only once the server has echoed
        // the messages back. Clearing it on the way out would lose the
        // transcript if the write failed.
        clearLocalTranscript();
        s.migrating = false;
      }
      refresh();
    } catch (cause) {
      // `stored` is deliberately untouched, so the same nodes are
      // offered again on the next change rather than being counted as
      // saved.
      setError(sentence(cause));
    } finally {
      s.busy = false;
      setSaving(false);
      if (s.dirty) {
        s.dirty = false;
        void flush();
      }
    }
  }, [runtime, depsRef, refresh]);

  const open = useCallback(
    async (id: string) => {
      const s = sync.current;
      s.suspended = true;
      setError(null);
      try {
        const conversation = await getConversation(id);
        runtime.thread.cancelRun();
        const { items, headId } = toBranchable(conversation);
        runtime.thread.import(
          ExportedMessageRepository.fromBranchableArray(items, { headId }),
        );
        s.conversationId = conversation.id;
        s.stored = storedIds(conversation);
        s.storedHead = conversation.head_id;
        setCurrent({ id: conversation.id, title: conversation.title });
      } catch (cause) {
        setError(sentence(cause));
      } finally {
        s.suspended = false;
      }
    },
    [runtime],
  );

  const newChat = useCallback(() => {
    const s = sync.current;
    s.suspended = true;
    runtime.thread.cancelRun();
    runtime.thread.reset();
    s.conversationId = null;
    s.stored = new Set();
    s.storedHead = null;
    s.migrating = false;
    setCurrent(null);
    setError(null);
    if (s.mode === "local") clearLocalTranscript();
    s.suspended = false;
  }, [runtime]);

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        if (sync.current.conversationId === id) newChat();
        refresh();
      } catch (cause) {
        setError(sentence(cause));
      }
    },
    [newChat, refresh],
  );

  // Probe once: does this server keep conversations at all?
  useEffect(() => {
    let cancelled = false;
    const s = sync.current;

    listConversations()
      .then(async (list) => {
        if (cancelled) return;
        setBoth("server");
        setSummaries(list);
        const local = readLocal();
        if (local?.messages?.length) {
          // A transcript from before this server had a store. It is
          // imported into the thread and then written through the
          // normal sync path, so there is one code path that creates a
          // conversation rather than two.
          try {
            runtime.thread.import(local as never);
            s.migrating = true;
          } catch {
            clearLocalTranscript();
          }
        } else if (list.length) {
          await open(list[0].id);
        }
      })
      .catch((cause) => {
        if (cancelled) return;
        setBoth("local");
        setReason(
          cause instanceof ApiError && cause.isMissingEndpoint
            ? "This server has no conversation API, so the transcript is kept in this browser only."
            : `The conversation store could not be reached (${sentence(cause)}), so the transcript is kept in this browser only.`,
        );
        const local = readLocal();
        if (local?.messages?.length) {
          try {
            runtime.thread.import(local as never);
          } catch {
            clearLocalTranscript();
          }
        }
      })
      .finally(() => {
        if (!cancelled) s.suspended = false;
      });

    return () => {
      cancelled = true;
    };
    // Runs once for the life of the runtime: a re-probe would replace a
    // thread the user is in the middle of.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtime]);

  // Coalesced writes. The thread notifies once per decoded token, and a
  // request per token would stutter the stream it is recording.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = runtime.thread.subscribe(() => {
      clearTimeout(timer);
      timer = setTimeout(() => void flush(), DEBOUNCE_MS);
    });
    return () => {
      clearTimeout(timer);
      unsubscribe();
    };
  }, [runtime, flush]);

  return {
    mode,
    reason,
    saving,
    error,
    current,
    summaries,
    refresh,
    open: (id) => void open(id),
    newChat,
    remove: (id) => void remove(id),
  };
}
