"use client";

import { motion } from "framer-motion";
import Link from "next/link";

import { cn } from "@/lib/utils";

type Props = {
  text: string;
  /** degrees */
  rotate?: number;
  className?: string;
  href?: string;
  delay?: number;
};

/**
 * Postage / ticket style card: dashed perforation, paper fill, heavy grain (campaign collage).
 */
export function StampTicket({ text, rotate = 0, className, href, delay = 0 }: Props) {
  const body = (
    <div
      className={cn(
        "hv-stamp-grain relative overflow-hidden rounded-[3px] border-[3px] border-dashed border-black/30 bg-[#d8d4cc] px-3.5 py-3 shadow-[0_20px_50px_rgba(0,0,0,0.45)]",
        "ring-1 ring-black/10 transition-transform duration-300",
        href && "hover:-translate-y-0.5 hover:shadow-[0_26px_60px_rgba(0,0,0,0.5)]"
      )}
    >
      <p className="relative z-10 text-[10px] font-extrabold uppercase leading-[1.35] tracking-[0.06em] text-[#1c1c1c] md:text-[11px]">
        {text}
      </p>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.92, rotate: rotate - 4 }}
      animate={{ opacity: 1, scale: 1, rotate }}
      transition={{ delay, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className={cn("w-[min(220px,42vw)] select-none", className)}
    >
      {href ? (
        <Link href={href} className="pointer-events-auto block">
          {body}
        </Link>
      ) : (
        body
      )}
    </motion.div>
  );
}
