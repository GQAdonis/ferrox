import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button, type ButtonProps } from "@/components/ui/button";

type State = "idle" | "copied" | "manual";

/**
 * Copy `getText()`, reporting success on the button itself.
 *
 * Clipboard access can be refused (non-secure origin, denied
 * permission). Telling the user to press the shortcut is a real
 * fallback; silently doing nothing is not.
 */
export function CopyButton({
  getText,
  label = "Copy",
  showLabel = false,
  variant = "ghost",
  size = "sm",
  className,
  ...props
}: Omit<ButtonProps, "onClick" | "children"> & {
  getText: () => string;
  label?: string;
  showLabel?: boolean;
}) {
  const [state, setState] = useState<State>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  const onClick = useCallback(async () => {
    let next: State = "copied";
    try {
      await navigator.clipboard.writeText(getText());
    } catch {
      next = "manual";
    }
    setState(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), 1600);
  }, [getText]);

  const text =
    state === "copied"
      ? "Copied"
      : state === "manual"
        ? "Press ⌘C / Ctrl+C"
        : label;

  return (
    <Button
      type="button"
      variant={variant}
      size={showLabel ? size : size === "sm" ? "iconSm" : "icon"}
      onClick={onClick}
      aria-label={label}
      title={label}
      className={className}
      {...props}
    >
      {state === "copied" ? <Check className="text-ok" /> : <Copy />}
      {showLabel || state !== "idle" ? <span>{text}</span> : null}
    </Button>
  );
}
