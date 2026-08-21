import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** The shadcn/ui class helper: conditional classes, last-one-wins merge. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
