import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-card border border-white/10 bg-card text-primary shadow-[0_8px_36px_rgba(0,0,0,0.5)] transition-[transform,border-color,box-shadow] duration-200 hover:-translate-y-0.5 hover:border-white/20 hover:shadow-[0_16px_48px_rgba(0,0,0,0.55)]",
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("border-b border-white/8 p-5", className)} {...props} />;
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}
