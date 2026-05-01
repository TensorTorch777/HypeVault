"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/utils";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: "primary" | "ghost" | "outline";
  strength?: number;
};

export function MagneticButton({
  children,
  className,
  variant = "primary",
  strength = 28,
  ...rest
}: Props) {
  const ref = useRef<HTMLButtonElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 180, damping: 18, mass: 0.2 });
  const sy = useSpring(my, { stiffness: 180, damping: 18, mass: 0.2 });
  const rotate = useTransform(sx, (v) => v / 4);

  const onMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    mx.set(((e.clientX - cx) / r.width) * strength);
    my.set(((e.clientY - cy) / r.height) * strength);
  };

  const onLeave = () => {
    mx.set(0);
    my.set(0);
  };

  const variants = {
    primary:
      "bg-[#FF3B00] text-white shadow-[0_10px_30px_-8px_rgba(255,59,0,0.55)] hover:shadow-[0_18px_50px_-10px_rgba(255,59,0,0.75)]",
    ghost:
      "bg-white/[0.04] text-white border border-white/10 hover:bg-white/[0.08] hover:border-white/20",
    outline:
      "bg-transparent text-white border border-white/20 hover:bg-white/[0.04] hover:border-white/40",
  }[variant];

  return (
    <motion.button
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ x: sx, y: sy, rotate }}
      className={cn(
        "relative inline-flex min-h-[52px] items-center justify-center gap-2 overflow-hidden rounded-full px-8 text-sm font-bold tracking-wide transition-[background,border] duration-200",
        variants,
        className
      )}
      {...(rest as object)}
    >
      <span className="relative z-[1]">{children}</span>
    </motion.button>
  );
}
