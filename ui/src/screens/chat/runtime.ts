// The bridge between assistant-ui's thread runtime and ferrox's public
// `/v1/chat/completions`.
//
// assistant-ui owns the transcript, the composer, autoscroll, branching
// and the abort signal. This file owns exactly one thing: turning a run
// into an SSE request and turning the server's answer back into message
// parts. Nothing here measures time — every number the UI prints comes
// from the server's own `usage` block, carried on the message as
// `metadata.custom`.

import { useState } from "react";
import {
  useLocalRuntime,
  type ChatModelAdapter,
  type ChatModelRunResult,
  type ThreadMessage,
} from "@assistant-ui/react";
import {
  ApiError,
  cancelGeneration,
  streamChat,
  type ChatMessage,
  type StreamResult,
  type Usage,
} from "@/lib/api";
import { fmtInt, fmtMs, fmtNum, isNum } from "@/lib/format";
import { useLatest } from "@/lib/use-latest";

export type Sampling = {
  system: string;
  temperature: number;
  topP: number;
  maxTokens: number;
};

export const DEFAULT_SAMPLING: Sampling = {
  system: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 512,
};

/**
 * What the UI prints under an answer.
 *
 * `line` is the server's `usage`, formatted. `outcome` says how the
 * generation ended when that is not simply "it finished" — a short
 * answer and a truncated one look identical otherwise.
 */
export type AnswerStats = {
  line: string;
  requestId: string | null;
  outcome: "ok" | "stopped-by-you" | "stopped-by-server" | "error";
  usage: Usage | null;
};

/** Turns the server's `usage` into one line, omitting anything absent. */
export function statLine(
  usage: Usage | null | undefined,
  requestId: string | null,
): string {
  const parts: string[] = [];
  if (isNum(usage?.time_to_first_token_ms))
    parts.push(`TTFT ${fmtMs(usage.time_to_first_token_ms)}`);
  if (usage) {
    const prefill = [`${fmtInt(usage.prompt_tokens)} tok`];
    if (isNum(usage.prompt_per_second))
      prefill.push(`${fmtNum(usage.prompt_per_second)} tok/s`);
    if (isNum(usage.prompt_eval_duration_ms))
      prefill.push(fmtMs(usage.prompt_eval_duration_ms));
    parts.push(`prefill ${prefill.join(" · ")}`);

    const decode = [`${fmtInt(usage.completion_tokens)} tok`];
    if (isNum(usage.predicted_per_second))
      decode.push(`${fmtNum(usage.predicted_per_second)} tok/s`);
    if (isNum(usage.generation_duration_ms))
      decode.push(fmtMs(usage.generation_duration_ms));
    parts.push(`decode ${decode.join(" · ")}`);

    if (isNum(usage.cached_tokens))
      parts.push(`cached ${fmtInt(usage.cached_tokens)} tok`);
  }
  if (requestId) parts.push(requestId);
  return parts.join("  ·  ");
}

/** Flattens assistant-ui's parts back into the wire format. */
function toWire(messages: readonly ThreadMessage[]): ChatMessage[] {
  const wire: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    // A message that failed carries no answer worth replaying.
    if (message.status?.type === "incomplete" && message.status.reason === "error")
      continue;
    const text = message.content
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("");
    if (!text) continue;
    wire.push({ role: message.role, content: text });
  }
  return wire;
}

/** A callback stream turned into something `for await` can drain. */
function pump<T>() {
  const queue: T[] = [];
  let wake: (() => void) | null = null;
  let ended = false;
  return {
    push(value: T) {
      queue.push(value);
      wake?.();
      wake = null;
    },
    end() {
      ended = true;
      wake?.();
      wake = null;
    },
    async *drain(): AsyncGenerator<T, void> {
      for (;;) {
        while (queue.length) yield queue.shift()!;
        if (ended) return;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
    },
  };
}

export type ChatDeps = {
  /** The model id `/v1/models` reports right now. */
  modelId: () => string | null;
  sampling: () => Sampling;
  /** `ms` when the stream has gone quiet, `null` when it came back. */
  onStall: (ms: number | null) => void;
};

function makeAdapter(deps: ChatDeps): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const sampling = deps.sampling();
      const wire: ChatMessage[] = [];
      if (sampling.system.trim())
        wire.push({ role: "system", content: sampling.system.trim() });
      wire.push(...toWire(messages));

      const tokens = pump<string>();
      let requestId: string | null = null;
      let result: StreamResult | null = null;
      let failure: unknown = null;

      // Both cancellation tiers, because one is not enough. assistant-ui
      // aborts the fetch, which closes the socket — and the server now
      // notices that. But a proxy can hold the backend connection open,
      // so the explicit `POST /v1/cancel` goes out too. Both end at the
      // same server-side flag, so doing both is never worse than either.
      const onAbort = () => cancelGeneration(requestId);
      abortSignal.addEventListener("abort", onAbort, { once: true });

      const task = streamChat(
        {
          model: deps.modelId() || "ferrox",
          messages: wire,
          temperature: sampling.temperature,
          top_p: sampling.topP,
          max_tokens: sampling.maxTokens,
        },
        {
          signal: abortSignal,
          onRequestId: (id) => {
            // Named on the first chunk, which is what makes an explicit
            // cancel possible at all — there is nothing to cancel by
            // before the server has said what this generation is called.
            requestId = id;
            live.add(id);
          },
          onToken: (token) => tokens.push(token),
          onStall: deps.onStall,
        },
      )
        .then((r) => {
          result = r;
        })
        .catch((error) => {
          failure = error;
        })
        .finally(() => {
          if (requestId) live.delete(requestId);
          tokens.end();
        });

      let text = "";
      try {
        for await (const token of tokens.drain()) {
          text += token;
          yield { content: [{ type: "text", text }] };
        }
        await task;
      } finally {
        abortSignal.removeEventListener("abort", onAbort);
      }

      const finished = result as StreamResult | null;
      const error = failure;

      if (!error) {
        const id = finished?.requestId || requestId;
        // The server won the race: it noticed the cancel and closed the
        // stream cleanly, so this arrives as a finished response rather
        // than as an AbortError. Saying so is the difference between a
        // short answer and a truncated one, which look identical.
        const cancelled = finished?.finishReason === "cancelled";
        const stats: AnswerStats = {
          line: statLine(finished?.usage, id),
          requestId: id,
          outcome: cancelled ? "stopped-by-server" : "ok",
          usage: finished?.usage ?? null,
        };
        yield {
          content: [{ type: "text", text }],
          status: cancelled
            ? { type: "incomplete", reason: "cancelled" }
            : { type: "complete", reason: "stop" },
          metadata: { custom: { stats } },
        } satisfies ChatModelRunResult;
        return;
      }

      if (error instanceof Error && error.name === "AbortError") {
        // A stopped generation is not a failure: the tokens that did
        // arrive are kept, and the line says why there are no timings.
        yield {
          content: [
            {
              type: "text",
              text: text || "_(stopped before any token arrived)_",
            },
          ],
          status: { type: "incomplete", reason: "cancelled" },
          metadata: {
            custom: {
              stats: {
                line: "",
                requestId,
                outcome: "stopped-by-you",
                usage: null,
              } satisfies AnswerStats,
            },
          },
        } satisfies ChatModelRunResult;
        return;
      }

      const message =
        error instanceof ApiError && error.isAuth
          ? `${error.message}\n\nThis server requires an API key. Set it on the Connect screen.`
          : error instanceof Error
            ? error.message
            : String(error);
      throw new Error(message);
    },
  };
}

/**
 * Ids of generations that are on the wire right now.
 *
 * The tab closing mid-answer is precisely the case an AbortSignal
 * cannot cover: the page is gone before the abort is delivered. A
 * `keepalive` POST is the one request that survives it.
 */
const live = new Set<string>();

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    for (const id of live) cancelGeneration(id);
  });
}

export function useFerroxRuntime(deps: ChatDeps) {
  // The adapter is built once and reads its inputs through a latest-value
  // box, so a sampling change mid-conversation applies to the next send
  // without tearing down the runtime (which would drop the transcript).
  const ref = useLatest(deps);

  // Read through the box, never captured: each arrow is called at send
  // time, so the adapter always sees the settings as they are then.
  // eslint-disable-next-line react-hooks/refs -- see lib/use-latest.ts
  const [adapter] = useState(() =>
    makeAdapter({
      modelId: () => ref.current.modelId(),
      sampling: () => ref.current.sampling(),
      onStall: (ms) => ref.current.onStall(ms),
    }),
  );

  return useLocalRuntime(adapter);
}
