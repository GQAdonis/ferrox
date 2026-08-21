import { useEffect } from "react";
import type { AssistantRuntime } from "@assistant-ui/react";

// Where the transcript lives.
//
// A deliberate limit, not an oversight: this server has no conversation
// API, so there is nothing to sync against. Inventing a client-side
// "sync" against endpoints that do not exist would be a lie the user
// could not see, so the transcript stays in this browser and the screen
// says so on its face.
//
// What is stored is assistant-ui's own exported repository — messages
// plus their parent ids — so edit/regenerate branches survive a reload
// rather than collapsing to a flat list.

const KEY = "ferrox.studio.thread.v2";

/** Anything unparseable is dropped: a corrupt blob must not wedge Chat. */
function read(): unknown {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function clearTranscript() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* private browsing: nothing was persisted to begin with */
  }
}

export function usePersistedThread(runtime: AssistantRuntime) {
  useEffect(() => {
    const saved = read() as { messages?: unknown[] } | null;
    if (saved?.messages?.length) {
      try {
        runtime.thread.import(saved as never);
      } catch {
        // A repository written by an older shape is not worth a crash.
        clearTranscript();
      }
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = runtime.thread.subscribe(() => {
      // Coalesced: a decode loop notifies once per token, and writing
      // the whole transcript to localStorage that often would stutter
      // the stream it is trying to record.
      clearTimeout(timer);
      timer = setTimeout(() => {
        try {
          const exported = runtime.thread.export();
          if (!exported.messages.length) clearTranscript();
          else localStorage.setItem(KEY, JSON.stringify(exported));
        } catch {
          /* quota or private browsing — the in-memory thread still works */
        }
      }, 400);
    });

    return () => {
      clearTimeout(timer);
      unsubscribe();
    };
  }, [runtime]);
}
