import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border border-white/12 bg-white/[0.06] px-3 py-1 text-xs font-semibold uppercase tracking-wide text-primary/75",
        className
      )}
      {...props}
    />
  );
}
