import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const variants = cva(
  "inline-flex min-h-[44px] items-center justify-center rounded-button px-5 text-sm font-semibold transition-all duration-150 hover:-translate-y-px hover:brightness-110 active:translate-y-0 active:scale-[0.98] active:brightness-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-accent text-white",
        outline:
          "border border-white/15 bg-card text-primary hover:border-white/25 hover:bg-white/[0.04]",
        ghost: "text-primary hover:bg-white/[0.06] active:bg-white/10",
      },
    },
    defaultVariants: { variant: "primary" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof variants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, ...props }, ref) => (
    <button ref={ref} className={cn(variants({ variant }), className)} {...props} />
  )
);
Button.displayName = "Button";
