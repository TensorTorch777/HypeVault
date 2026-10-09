import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-12 min-h-[44px] w-full rounded-button border border-white/15 bg-[#2a1840] px-4 text-sm text-[#FFEDF6] shadow-sm transition focus:border-[#FF7A1A] focus:outline-none focus:ring-2 focus:ring-[#FF7A1A]/30 placeholder:text-white/35",
        className
      )}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";
