"use client";

import { motion, type HTMLMotionProps, type Variants } from "framer-motion";
import { type ReactNode } from "react";

const EASE_OUT = [0.22, 1, 0.36, 1] as const;
const EASE_IN_OUT = [0.65, 0, 0.35, 1] as const;

/* ─── Reusable variants ───────────────────────────────────── */

export const fadeUp: Variants = {
  hidden:  { opacity: 0, y: 28, filter: "blur(6px)" },
  visible: { opacity: 1, y: 0,  filter: "blur(0px)", transition: { duration: 0.95, ease: EASE_OUT } },
};

export const fadeIn: Variants = {
  hidden:  { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 1.0, ease: EASE_IN_OUT } },
};

export const scaleIn: Variants = {
  hidden:  { opacity: 0, scale: 0.94 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.85, ease: EASE_OUT } },
};

export const slideInLeft: Variants = {
  hidden:  { opacity: 0, x: -36 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.8, ease: EASE_OUT } },
};

export const slideInRight: Variants = {
  hidden:  { opacity: 0, x: 36 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.8, ease: EASE_OUT } },
};

/* Container variant — children stagger in sequence */
export const stagger = (gap = 0.08): Variants => ({
  hidden:  {},
  visible: { transition: { staggerChildren: gap, delayChildren: 0.05 } },
});

/* ─── <Reveal> component ──────────────────────────────────── */

type RevealVariant = "fadeUp" | "fadeIn" | "scaleIn" | "slideLeft" | "slideRight";

const VARIANTS: Record<RevealVariant, Variants> = {
  fadeUp,
  fadeIn,
  scaleIn,
  slideLeft:  slideInLeft,
  slideRight: slideInRight,
};

export function Reveal({
  children,
  variant = "fadeUp",
  delay = 0,
  className,
  once = true,
  margin = "-60px",
  ...rest
}: {
  children: ReactNode;
  variant?: RevealVariant;
  delay?: number;
  className?: string;
  once?: boolean;
  margin?: string;
} & Omit<HTMLMotionProps<"div">, "children" | "variants" | "initial" | "whileInView" | "viewport">) {
  const v = VARIANTS[variant];
  return (
    <motion.div
      variants={v}
      initial="hidden"
      whileInView="visible"
      viewport={{ once, margin: margin as `${number}px` }}
      transition={{ delay, ease: EASE_OUT }}
      className={className}
      {...rest}
    >
      {children}
    </motion.div>
  );
}

/* ─── <Stagger> container + item ──────────────────────────── */

export function Stagger({
  children,
  gap = 0.08,
  className,
  once = true,
  margin = "-60px",
}: {
  children: ReactNode;
  gap?: number;
  className?: string;
  once?: boolean;
  margin?: string;
}) {
  return (
    <motion.div
      variants={stagger(gap)}
      initial="hidden"
      whileInView="visible"
      viewport={{ once, margin: margin as `${number}px` }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({
  children,
  variant = "fadeUp",
  className,
}: {
  children: ReactNode;
  variant?: RevealVariant;
  className?: string;
}) {
  return (
    <motion.div variants={VARIANTS[variant]} className={className}>
      {children}
    </motion.div>
  );
}

/* ─── <Float> — gentle continuous floating ────────────────── */

export function Float({
  children,
  amount = 8,
  duration = 5,
  className,
}: {
  children: ReactNode;
  amount?: number;
  duration?: number;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      animate={{ y: [0, -amount, 0, amount, 0] }}
      transition={{ duration, ease: "easeInOut", repeat: Infinity }}
    >
      {children}
    </motion.div>
  );
}

/* ─── <Breathe> — gentle continuous scale/opacity pulse ──── */

export function Breathe({
  children,
  scale = 1.02,
  duration = 3.5,
  className,
}: {
  children: ReactNode;
  scale?: number;
  duration?: number;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      animate={{ scale: [1, scale, 1] }}
      transition={{ duration, ease: "easeInOut", repeat: Infinity }}
    >
      {children}
    </motion.div>
  );
}
