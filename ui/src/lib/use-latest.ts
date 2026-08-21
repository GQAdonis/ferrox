import { useEffect, useRef } from "react";

/**
 * A box holding the newest `value`, readable from a long-lived closure.
 *
 * The chat adapter is built once and outlives every render; it still has
 * to see the sampling settings and model id as they are at *send* time,
 * not as they were when it was created. Rebuilding the adapter on each
 * change would tear down the assistant-ui runtime and take the
 * transcript with it.
 *
 * The write happens in an effect rather than during render, because a
 * render can be thrown away or replayed and a value written during one
 * would outlive a screen that never appeared. Every reader is an event
 * handler or a network callback, all of which run after commit, so the
 * effect timing costs nothing.
 */
export function useLatest<T>(value: T): { readonly current: T } {
  const box = useRef(value);
  useEffect(() => {
    box.current = value;
  }, [value]);
  return box;
}
