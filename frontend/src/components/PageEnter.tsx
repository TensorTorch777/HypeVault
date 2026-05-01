"use client";

import { motion } from "framer-motion";

/**
 * Wraps the entire page contents.
 * Soft fade + gentle rise + blur, easing out cinematically.
 */
export function PageEnter({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
      animate={{ opacity: 1, y: 0,  filter: "blur(0px)" }}
      transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}
