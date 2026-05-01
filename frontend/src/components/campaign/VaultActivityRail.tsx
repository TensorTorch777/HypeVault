"use client";

import { motion } from "framer-motion";
import {
  Activity,
  CheckCircle2,
  Clock,
  Radio,
  ScanLine,
  ShieldCheck,
  TrendingUp,
  Zap,
} from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type Tone = "paper" | "accent" | "electric" | "mint" | "violet" | "ink";

export type VaultFeedItem = {
  id: string;
  tone: Tone;
  icon: "live" | "zap" | "scan" | "shield" | "trend" | "check" | "clock" | "activity";
  label: string;
  title: string;
  meta: string;
  live?: boolean;
};

const ICONS: Record<VaultFeedItem["icon"], ReactNode> = {
  live: <Radio className="h-5 w-5" strokeWidth={2.25} />,
  zap: <Zap className="h-5 w-5" strokeWidth={2.25} />,
  scan: <ScanLine className="h-5 w-5" strokeWidth={2.25} />,
  shield: <ShieldCheck className="h-5 w-5" strokeWidth={2.25} />,
  trend: <TrendingUp className="h-5 w-5" strokeWidth={2.25} />,
  check: <CheckCircle2 className="h-5 w-5" strokeWidth={2.25} />,
  clock: <Clock className="h-5 w-5" strokeWidth={2.25} />,
  activity: <Activity className="h-5 w-5" strokeWidth={2.25} />,
};

const shell: Record<Tone, string> = {
  paper: "hv-glass-dark text-white",
  accent: "hv-glass-dark--accent text-white",
  electric: "hv-glass-dark--electric text-white",
  mint: "hv-glass-dark--mint text-white",
  violet: "hv-glass-dark--violet text-white",
  ink: "hv-glass-ink text-white",
};

function GlassVaultFeedCard({ item, className }: { item: VaultFeedItem; className?: string }) {
  const isInk = item.tone === "ink";

  return (
    <motion.div
      initial={{ opacity: 1, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        "relative w-full p-4 transition-all duration-300 hover:-translate-y-0.5 md:p-5",
        isInk
          ? "hover:shadow-[0_28px_72px_rgba(0,0,0,0.5)]"
          : "hover:shadow-[0_24px_64px_rgba(0,0,0,0.35)]",
        shell[item.tone],
        className
      )}
    >
      <div className="flex gap-3.5 md:gap-4">
        <div
          className={cn(
            "flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border backdrop-blur-md md:h-12 md:w-12",
            "border-white/18 bg-white/[0.08] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]",
            item.tone === "accent" && "border-orange-400/40 bg-orange-500/15 text-orange-200",
            item.tone === "electric" && "border-blue-400/40 bg-blue-500/15 text-blue-200",
            item.tone === "mint" && "border-emerald-400/40 bg-emerald-500/12 text-emerald-200",
            item.tone === "violet" && "border-violet-400/40 bg-violet-500/14 text-violet-200",
            isInk && "border-white/22 bg-white/[0.1]"
          )}
        >
          {ICONS[item.icon]}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-white/55 md:tracking-[0.2em]">
              {item.label}
            </p>
            {item.live ? (
              <span className="relative flex h-2 w-2 shrink-0 translate-y-1">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/50" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
              </span>
            ) : (
              <span className="shrink-0 text-[10px] font-semibold tabular-nums text-white/45">{item.meta}</span>
            )}
          </div>
          <p className="mt-2 font-[family-name:var(--font-display)] text-[0.95rem] font-extrabold leading-snug tracking-tight text-white md:text-base">
            {item.title}
          </p>
          {item.live ? <p className="mt-1.5 text-[11px] font-medium text-white/50">{item.meta}</p> : null}
        </div>
      </div>
    </motion.div>
  );
}

export const VAULT_FEED_LEFT: VaultFeedItem[] = [
  {
    id: "l1",
    tone: "accent",
    icon: "live",
    label: "Live · Pipeline",
    title: "Jordan 4 Black Cat — 6/7 authenticity signals green",
    meta: "Just now",
    live: true,
  },
  {
    id: "l2",
    tone: "ink",
    icon: "zap",
    label: "Price pulse",
    title: "StockX ask synced · spread vs Chrono24 −3.8%",
    meta: "1m ago",
  },
  {
    id: "l3",
    tone: "electric",
    icon: "scan",
    label: "Your queue",
    title: "Upload slot #12 · screening ETA under 5 min",
    meta: "2m ago",
  },
  {
    id: "l4",
    tone: "ink",
    icon: "activity",
    label: "Inference",
    title: "Triton batch cleared · 847 frames scored this hour",
    meta: "3m ago",
  },
  {
    id: "l5",
    tone: "violet",
    icon: "clock",
    label: "SLA watch",
    title: "P95 verify latency 4.1s · inside target band",
    meta: "5m ago",
  },
];

export const VAULT_FEED_RIGHT: VaultFeedItem[] = [
  {
    id: "r1",
    tone: "ink",
    icon: "trend",
    label: "Markets",
    title: "eBay + Chrono24 comparables pinned to this listing",
    meta: "Now",
  },
  {
    id: "r2",
    tone: "accent",
    icon: "shield",
    label: "Trust layer",
    title: "Seller history cross-checked across 3 marketplaces",
    meta: "4m ago",
  },
  {
    id: "r3",
    tone: "electric",
    icon: "check",
    label: "Verified drop",
    title: "Rolex Sub ref. matched · box & papers queued for OCR",
    meta: "6m ago",
  },
  {
    id: "r4",
    tone: "mint",
    icon: "scan",
    label: "OCR lane",
    title: "Serial region crop locked · glare score acceptable",
    meta: "7m ago",
  },
  {
    id: "r5",
    tone: "ink",
    icon: "zap",
    label: "Cache hit",
    title: "Chrono24 price snapshot served from edge · 12ms",
    meta: "8m ago",
  },
];

type RailProps = {
  children: ReactNode;
  className?: string;
};

/** Glass status cards flanking the scratch panel — full-width rails with ambient color. */
export function VaultActivityRail({ children, className }: RailProps) {
  const mobileFeed = [...VAULT_FEED_LEFT, ...VAULT_FEED_RIGHT];

  return (
    <div className={cn("relative", className)}>
      <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden rounded-[2rem] border border-white/[0.08] bg-gradient-to-br from-white/[0.05] via-zinc-900/35 to-zinc-950/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-md">
        <div
          className="absolute -left-[10%] top-1/4 h-[min(420px,55vh)] w-[min(420px,45vw)] rounded-full bg-gradient-to-br from-orange-500/22 via-amber-400/12 to-transparent blur-3xl"
          aria-hidden
        />
        <div
          className="absolute -right-[8%] bottom-[12%] h-[min(380px,50vh)] w-[min(380px,42vw)] rounded-full bg-gradient-to-tl from-[#1a5fff]/22 via-violet-400/16 to-transparent blur-3xl"
          aria-hidden
        />
        <div
          className="absolute left-1/3 -top-24 h-52 w-52 rounded-full bg-emerald-400/10 blur-3xl"
          aria-hidden
        />
        <div
          className="absolute left-1/2 top-1/2 h-[min(360px,40vh)] w-[min(520px,70vw)] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black/50 blur-[80px]"
          aria-hidden
        />
      </div>

      <div className="relative px-3 pb-8 pt-6 sm:px-5 md:px-7 md:pb-10 md:pt-8">
        {/* One {children} — responsive grid swaps rails vs sheet without remounting scratch canvas */}
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(280px,1fr)_minmax(300px,560px)_minmax(280px,1fr)] xl:gap-8 xl:items-start">
          <div className="order-1 col-span-full min-w-0 xl:order-none xl:col-span-1 xl:col-start-2 xl:row-start-1 xl:sticky xl:top-24">
            {children}
          </div>

          <div className="order-2 hidden w-full flex-col justify-between gap-4 xl:order-none xl:col-start-1 xl:row-start-1 xl:flex 2xl:gap-5">
            {VAULT_FEED_LEFT.map((item) => (
              <GlassVaultFeedCard key={item.id} item={item} />
            ))}
          </div>

          <div className="order-3 hidden w-full flex-col justify-between gap-4 xl:order-none xl:col-start-3 xl:row-start-1 xl:flex 2xl:gap-5">
            {VAULT_FEED_RIGHT.map((item) => (
              <GlassVaultFeedCard key={item.id} item={item} />
            ))}
          </div>

          {/* &lt; xl: fill width — horizontal strip on small screens, grid from sm */}
          <div className="order-2 col-span-full flex gap-4 overflow-x-auto pb-1 pl-0.5 [-ms-overflow-style:none] [scrollbar-width:none] sm:hidden [&::-webkit-scrollbar]:hidden xl:hidden">
            {mobileFeed.map((item) => (
              <GlassVaultFeedCard key={item.id} item={item} className="w-[min(320px,88vw)] shrink-0" />
            ))}
          </div>
          <div className="order-2 col-span-full hidden grid-cols-1 gap-4 sm:grid sm:grid-cols-2 xl:hidden">
            {mobileFeed.map((item) => (
              <GlassVaultFeedCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
