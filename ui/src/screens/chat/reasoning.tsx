import { useState } from "react";
import type { ReasoningMessagePartComponent } from "@assistant-ui/react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * A reasoning model's thinking, shown above the answer it led to.
 *
 * assistant-ui renders NOTHING for a reasoning part unless a component
 * is supplied for it, which is how this went unnoticed: the server
 * streamed `reasoning_content` correctly, the client dropped it, and an
 * R1 distill therefore spent most of its wall-clock producing tokens
 * that never reached the screen. The transcript looked frozen and the
 * answer arrived whole, which reads as "streaming is broken" rather
 * than "thinking is hidden".
 *
 * Collapsed by default, because it is working-out and not the answer.
 * It is deliberately NOT markdown: thinking is where a model emits
 * half-open code fences and unbalanced brackets, and a renderer that
 * reflows them makes the text harder to read, not easier.
 */
export const ReasoningPart: ReasoningMessagePartComponent = ({ text }) => {
  const [open, setOpen] = useState(false);
  const trimmed = text.trim();
  // A part that exists but has not been written to yet would otherwise
  // render an empty, clickable box under every answer.
  if (!trimmed) return null;
  return (
    <div className="mb-2 rounded-lg border border-line bg-inset">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-xs text-muted transition-colors hover:text-fg"
      >
        <ChevronRight
          aria-hidden
          className={cn(
            "size-3.5 shrink-0 transition-transform",
            open && "rotate-90",
          )}
        />
        <span>Thinking</span>
        <span className="text-muted/70">
          {trimmed.length.toLocaleString()} chars
        </span>
      </button>
      {open && (
        <div className="max-h-80 overflow-y-auto whitespace-pre-wrap border-t border-line px-2.5 py-2 text-xs leading-relaxed text-muted">
          {trimmed}
        </div>
      )}
    </div>
  );
};
