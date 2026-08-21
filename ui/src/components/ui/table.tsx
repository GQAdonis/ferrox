import type * as React from "react";
import { cn } from "@/lib/utils";

/** Wide tables scroll inside their own box; the page never scrolls sideways. */
export function TableScroll({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cn("w-full overflow-x-auto overscroll-x-contain", className)}
      {...props}
    />
  );
}

export function Table({ className, ...props }: React.ComponentProps<"table">) {
  return (
    <table
      className={cn("w-full border-collapse text-sm", className)}
      {...props}
    />
  );
}

export function Th({
  className,
  numeric,
  ...props
}: React.ComponentProps<"th"> & { numeric?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "sticky top-0 z-10 border-b border-line bg-raised px-3 py-2 text-left text-[0.6875rem] font-semibold tracking-wide text-faint uppercase",
        numeric && "text-right tabular-nums",
        className,
      )}
      {...props}
    />
  );
}

export function Td({
  className,
  numeric,
  mono,
  ...props
}: React.ComponentProps<"td"> & { numeric?: boolean; mono?: boolean }) {
  return (
    <td
      className={cn(
        "border-b border-line/70 px-3 py-2 align-middle",
        numeric && "text-right tabular-nums",
        mono && "font-mono text-[0.8125rem]",
        className,
      )}
      {...props}
    />
  );
}

export function Tr({ className, ...props }: React.ComponentProps<"tr">) {
  return (
    <tr
      className={cn("transition-colors hover:bg-inset/60", className)}
      {...props}
    />
  );
}
