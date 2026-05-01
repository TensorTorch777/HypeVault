"use client";

import { motion, useMotionValue, useTransform, PanInfo } from "framer-motion";
import Link from "next/link";
import { useState } from "react";

export type InsightCard = {
  id: string;
  eyebrow: string;
  title: string;
  href: string;
};

const DRAG_THRESHOLD = 80;

export function SwipeInsightRail({ items }: { items: InsightCard[] }) {
  const [i, setI] = useState(0);
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-6, 6]);

  const onDragEnd = (_: unknown, info: PanInfo) => {
    const o = info.offset.x;
    if (o < -DRAG_THRESHOLD && i < items.length - 1) {
      setI((v) => v + 1);
    } else if (o > DRAG_THRESHOLD && i > 0) {
      setI((v) => v - 1);
    }
    x.set(0);
  };

  const card = items[i];
  if (!card) return null;

  return (
    <div className="relative mx-auto max-w-lg">
      <motion.div
        key={card.id}
        style={{ x, rotate }}
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.65}
        onDragEnd={onDragEnd}
        whileTap={{ cursor: "grabbing" }}
        className="relative cursor-grab touch-pan-y rounded-[20px] border border-white/12 bg-zinc-900/60 p-8 shadow-[0_28px_90px_rgba(0,0,0,0.55)] backdrop-blur-xl"
      >
        <p className="text-[10px] font-bold uppercase tracking-[0.32em] text-primary/40">{card.eyebrow}</p>
        <p className="mt-5 font-[family-name:var(--font-display)] text-2xl font-extrabold leading-snug tracking-tight text-primary md:text-3xl">
          {card.title}
        </p>
        <div className="mt-8 flex items-center justify-between gap-4">
          <Link href={card.href} className="text-sm font-bold text-accent hover:underline">
            Explore →
          </Link>
          <span className="text-[10px] font-bold uppercase tracking-[0.22em] text-primary/35">Swipe</span>
        </div>
      </motion.div>

      <div className="mt-5 flex justify-center gap-2">
        {items.map((item, idx) => (
          <button
            key={item.id}
            type="button"
            aria-label={`Card ${idx + 1}`}
            onClick={() => setI(idx)}
            className={`h-2 rounded-full transition-all ${idx === i ? "w-8 bg-accent" : "w-2 bg-white/15"}`}
          />
        ))}
      </div>
    </div>
  );
}
