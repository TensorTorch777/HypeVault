import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-12 min-h-[44px] w-full rounded-xl border border-white/12 bg-[#0E0F1E]/70 px-4 text-[15px] text-primary transition placeholder:text-primary/35 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25",
        className
      )}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";
