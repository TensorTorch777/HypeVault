"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/**
 * Letter-by-letter spring bounce — each character drops in with overshoot.
 */
export function BouncyText({
  text,
  className,
  delay = 0,
  stagger = 0.04,
  bounce = 0.55,
}: {
  text: string;
  className?: string;
  delay?: number;
  stagger?: number;
  bounce?: number;
}) {
  const letters = [...text];

  return (
    <span aria-label={text} className={cn("inline-flex flex-nowrap items-baseline", className)}>
      {letters.map((ch, i) => (
        <motion.span
          key={`${ch}-${i}`}
          aria-hidden
          initial={{ y: -50, opacity: 0, rotate: -8 }}
          animate={{ y: 0, opacity: 1, rotate: 0 }}
          transition={{
            type: "spring",
            stiffness: 380,
            damping: 12,
            mass: 0.7,
            bounce,
            delay: delay + i * stagger,
          }}
          className="inline-block shrink-0"
          style={{ whiteSpace: ch === " " ? "pre" : undefined }}
        >
          {ch}
        </motion.span>
      ))}
    </span>
  );
}

/**
 * Hover the wrapper → text wobbles like jelly.
 * Each letter rotates back and forth slightly with stagger.
 */
export function WobbleText({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const [hovered, setHovered] = useState(false);
  const letters = [...text];

  return (
    <span
      aria-label={text}
      className={cn("inline-flex flex-nowrap items-baseline", className)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {letters.map((ch, i) => (
        <motion.span
          key={`${ch}-${i}`}
          aria-hidden
          animate={
            hovered
              ? { rotate: [0, -10, 10, -7, 7, -3, 0], y: [0, -3, 0, -2, 0] }
              : { rotate: 0, y: 0 }
          }
          transition={{
            duration: 0.7,
            delay: i * 0.03,
            ease: "easeInOut",
          }}
          className="inline-block shrink-0"
          style={{ whiteSpace: ch === " " ? "pre" : undefined }}
        >
          {ch}
        </motion.span>
      ))}
    </span>
  );
}

/**
 * Continuously wobbles forever — for cute idle accents.
 * Use sparingly so the page doesn't feel chaotic.
 */
export function GobblyText({
  text,
  className,
  amount = 3,
  duration = 2.4,
}: {
  text: string;
  className?: string;
  amount?: number;
  duration?: number;
}) {
  const letters = [...text];
  // Hydration safety — only animate after mount
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <span aria-label={text} className={cn("inline-flex flex-nowrap items-baseline", className)}>
      {letters.map((ch, i) => (
        <motion.span
          key={`${ch}-${i}`}
          aria-hidden
          animate={
            mounted
              ? { y: [0, -amount, 0, amount, 0], rotate: [0, -2, 0, 2, 0] }
              : undefined
          }
          transition={{
            duration,
            ease: "easeInOut",
            repeat: Infinity,
            delay: i * 0.12,
          }}
          className="inline-block shrink-0"
          style={{ whiteSpace: ch === " " ? "pre" : undefined }}
        >
          {ch}
        </motion.span>
      ))}
    </span>
  );
}
