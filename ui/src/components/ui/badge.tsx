import type * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.6875rem] font-medium leading-4 whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "border-line bg-inset text-muted",
        accent: "border-accent/35 bg-accent-soft text-accent",
        ok: "border-ok/35 bg-ok-soft text-ok",
        warn: "border-warn/35 bg-warn-soft text-warn",
        err: "border-err/35 bg-err-soft text-err",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type BadgeProps = React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants>;

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
