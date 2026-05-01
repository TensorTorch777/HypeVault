"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

/**
 * Word-by-word rise reveal. Use for headlines.
 */
export function SplitWords({
  text,
  className,
  delay = 0,
  stagger = 0.06,
  as: Tag = "span",
}: {
  text: string;
  className?: string;
  delay?: number;
  stagger?: number;
  as?: "span" | "h1" | "h2" | "h3" | "p";
}) {
  const words = text.split(" ");
  // framer-motion v11: motion.create() replaces motion()
  const MotionTag = (motion as unknown as { create: (t: string) => typeof motion.span }).create(Tag);

  return (
    <MotionTag
      aria-label={text}
      initial="hidden"
      animate="shown"
      className={cn("inline-block", className)}
    >
      {words.map((w, i) => (
        <span key={i} className="inline-block overflow-hidden pb-[0.1em] pr-[0.24em]">
          <motion.span
            variants={{
              hidden: { y: "110%", opacity: 0 },
              shown:  { y: "0%",   opacity: 1 },
            }}
            transition={{
              delay: delay + i * stagger,
              duration: 0.8,
              ease: [0.22, 1, 0.36, 1],
            }}
            className="inline-block"
          >
            {w}
          </motion.span>
        </span>
      ))}
    </MotionTag>
  );
}
