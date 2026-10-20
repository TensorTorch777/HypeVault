import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const variants = cva(
  "inline-flex min-h-[2.75rem] cursor-pointer items-center justify-center gap-2 rounded-full px-5 text-[0.875rem] font-bold transition-[transform,background-color,color,border-color] duration-200 ease-out hover:-translate-y-px active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-45",
  {
    variants: {
      variant: {
        primary: "bg-accent text-ink hover:bg-[#FFD0E1]",
        outline: "border border-primary/25 bg-background/30 text-primary hover:border-accent hover:text-accent",
        ghost: "text-primary hover:bg-white/[0.06] active:bg-white/10",
        sand: "bg-neon-yellow text-ink hover:bg-[#FFE6BD]",
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
