import type * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { AlertTriangle, CircleAlert, Info } from "lucide-react";
import { cn } from "@/lib/utils";

const noticeVariants = cva(
  "flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm",
  {
    variants: {
      tone: {
        info: "border-line bg-inset text-muted",
        warn: "border-warn/35 bg-warn-soft text-warn",
        err: "border-err/35 bg-err-soft text-err",
      },
    },
    defaultVariants: { tone: "info" },
  },
);

const noticeIcon = { info: Info, warn: AlertTriangle, err: CircleAlert };

export function Notice({
  className,
  tone = "info",
  children,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof noticeVariants>) {
  const Icon = noticeIcon[tone ?? "info"];
  return (
    <div
      role={tone === "err" ? "alert" : "status"}
      className={cn(noticeVariants({ tone }), className)}
      {...props}
    >
      <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

/** A grey block that holds the layout while real content is on the wire. */
export function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      aria-hidden
      className={cn("animate-pulse rounded-md bg-inset", className)}
      {...props}
    />
  );
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: React.ReactNode;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 px-6 py-12 text-center",
        className,
      )}
    >
      <span className="grid size-11 place-items-center rounded-xl border border-line bg-inset text-faint">
        <Icon className="size-5" />
      </span>
      <div className="space-y-1">
        <p className="text-sm font-medium text-fg">{title}</p>
        {children ? (
          <div className="mx-auto max-w-md text-xs text-faint">{children}</div>
        ) : null}
      </div>
      {action}
    </div>
  );
}
