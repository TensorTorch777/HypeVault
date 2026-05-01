"use client";

import { Globe, ShieldCheck, Zap } from "lucide-react";
import { motion } from "framer-motion";

import { AnimatedCounter } from "@/components/campaign/AnimatedCounter";
import { Stagger, StaggerItem } from "@/components/motion/Reveal";

const cards = [
  {
    key: "checks",
    icon: ShieldCheck,
    tone: "#FF1FA4",
    bg:   "rgba(255, 31, 164, 0.14)",
    value: 12840,
    suffix: "+",
    prefix: "" as string,
    label: "Checks queued",
    description: "Images screened for authenticity signals.",
  },
  {
    key: "markets",
    icon: Globe,
    tone: "#00E1FF",
    bg:   "rgba(0, 225, 255, 0.14)",
    value: 3,
    suffix: "",
    prefix: "",
    label: "Marketplaces",
    description: "StockX · Chrono24 · eBay — one comparison surface.",
  },
  {
    key: "latency",
    icon: Zap,
    tone: "#FFD24A",
    bg:   "rgba(255, 210, 74, 0.14)",
    value: 5,
    suffix: "s",
    prefix: "<",
    label: "Latency target",
    description: "Inference + pricing snapshot, cached when possible.",
  },
] as const;

export function HomeVaultStats() {
  return (
    <Stagger gap={0.1} className="grid auto-rows-fr gap-5 md:grid-cols-3 md:gap-6">
      {cards.map((c) => (
        <StaggerItem key={c.key}>
          <motion.div
            className="hv-card h-full p-8"
            whileHover={{ y: -6 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            <motion.div
              className="flex h-12 w-12 items-center justify-center rounded-2xl"
              style={{ background: c.bg, boxShadow: `0 0 32px ${c.tone}33` }}
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 3.6, ease: "easeInOut", repeat: Infinity }}
            >
              <c.icon className="h-6 w-6" strokeWidth={1.75} style={{ color: c.tone }} aria-hidden />
            </motion.div>
            <p
              className="mt-8 font-[family-name:var(--font-display)] text-[44px] font-extrabold italic leading-none tracking-tight text-[#FFEDF6] tabular-nums md:text-[52px]"
              style={{ textShadow: `0 0 24px ${c.tone}55` }}
            >
              {c.key === "latency" ? (
                <AnimatedCounter value={c.value} prefix={c.prefix} suffix={c.suffix} />
              ) : (
                <AnimatedCounter value={c.value} suffix={c.suffix} />
              )}
            </p>
            <p className="mt-3 text-[11px] font-bold uppercase tracking-[0.18em]" style={{ color: c.tone }}>
              {c.label}
            </p>
            <p className="mt-2 text-[15px] leading-[1.5] text-[#FFEDF6]/65">
              {c.description}
            </p>
          </motion.div>
        </StaggerItem>
      ))}
    </Stagger>
  );
}
