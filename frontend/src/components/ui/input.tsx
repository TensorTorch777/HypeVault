import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-12 min-h-[44px] w-full rounded-button border border-primary/12 bg-card px-4 text-sm text-primary shadow-sm transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25 placeholder:text-primary/35",
        className
      )}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";
