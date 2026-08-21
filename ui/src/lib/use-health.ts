import { useEffect, useState } from "react";
import { ApiError, getJson, routes, type Health } from "@/lib/api";

const POLL_MS = 5000;

export type HealthState = {
  /** The last `/health` body, whether it came back 200 or 503. */
  health: Health | null;
  /** Set only when the server did not answer with a health body at all. */
  error: ApiError | null;
};

/**
 * Poll `/health`.
 *
 * A 503 is not a failure here: it still carries a full body, and the
 * body is the answer ("unavailable", with a reason). Only a request
 * that produced no body at all lands in `error`.
 */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({
    health: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const poll = async () => {
      try {
        const body = await getJson<Health>(routes.health, {
          signal: controller.signal,
        });
        if (!cancelled) setState({ health: body, error: null });
      } catch (error) {
        if (cancelled || (error as Error)?.name === "AbortError") return;
        if (error instanceof ApiError && (error.body as Health)?.state) {
          setState({ health: error.body as Health, error: null });
        } else {
          setState({ health: null, error: error as ApiError });
        }
      }
    };

    void poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  return state;
}
