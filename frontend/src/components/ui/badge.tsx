import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn("inline-flex items-center rounded-full bg-white/[0.08] px-3 py-1 text-[12px] font-bold text-primary/85", className)}
      {...props}
    />
  );
}
