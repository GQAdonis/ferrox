import type * as React from "react";
import { cn } from "@/lib/utils";

/** The header every screen shares: title, one sentence, right-hand actions. */
export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-end justify-between gap-x-4 gap-y-2",
        className,
      )}
    >
      <div className="min-w-0 space-y-0.5">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {description ? (
          <p className="text-xs text-faint">{description}</p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}

/** A scrolling screen body with the standard gutters and max width. */
export function Page({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div className="h-full overflow-y-auto">
      <div
        className={cn(
          "mx-auto flex w-full max-w-6xl flex-col gap-5 p-4 md:p-6",
          className,
        )}
        {...props}
      />
    </div>
  );
}
