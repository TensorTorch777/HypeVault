"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

import { cn } from "@/lib/utils";

export type AskVariant = "white" | "blue" | "dark";

export type AskItem = {
  id: string;
  category: string;
  question: string;
  href: string;
  variant: AskVariant;
};

const shell: Record<AskVariant, string> = {
  white: "hv-glass-ask-frost text-zinc-950",
  blue: "hv-glass-ask-blue text-white",
  dark: "hv-glass-ask-smoke text-white",
};

function AskCard({ item, index }: { item: AskItem; index: number }) {
  const isLight = item.variant === "white";

  return (
    <motion.div
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ delay: index * 0.06, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <Link
        href={item.href}
        className={cn(
          "group flex h-full min-h-[200px] flex-col rounded-xl p-6 transition-all duration-300 md:min-h-[220px] md:p-7",
          "hover:-translate-y-1 hover:border-white/30 hover:shadow-[0_28px_80px_rgba(0,0,0,0.5)]",
          item.variant === "blue" && "hover:shadow-[0_28px_80px_rgba(26,95,255,0.25)]",
          shell[item.variant]
        )}
      >
        <div className="flex items-start justify-between gap-3">
          <span
            className={cn(
              "inline-flex rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-[0.2em] backdrop-blur-sm",
              isLight
                ? "border-zinc-900/12 bg-zinc-900/[0.08] text-zinc-800"
                : "border-white/18 bg-white/[0.1] text-white/92"
            )}
          >
            {item.category}
          </span>
          <span
            className={cn(
              "flex h-11 w-11 shrink-0 items-center justify-center rounded-full border backdrop-blur-md transition duration-300",
              isLight
                ? "border-zinc-900/18 bg-white/30 text-zinc-900 group-hover:border-zinc-900/28 group-hover:bg-white/45"
                : "border-white/22 bg-white/[0.08] text-white group-hover:border-white/35 group-hover:bg-white/[0.14]"
            )}
          >
            <ArrowUpRight className="h-5 w-5" strokeWidth={2.2} />
          </span>
        </div>
        <p
          className={cn(
            "mt-5 flex-1 font-[family-name:var(--font-display)] text-xl font-extrabold leading-snug tracking-tight md:text-2xl",
            isLight ? "text-zinc-950" : "text-white"
          )}
        >
          {item.question}
        </p>
      </Link>
    </motion.div>
  );
}

const DEFAULT_ASKS: AskItem[] = [
  {
    id: "a1",
    category: "Sneakers",
    question: "Why can’t buyers verify a drop before they wire trust to a stranger’s camera roll?",
    href: "/product/Sneakers",
    variant: "white",
  },
  {
    id: "a2",
    category: "Market",
    question: "Why is comparing prices across platforms still a maze of tabs and guesswork?",
    href: "/product/Luxury%20watches",
    variant: "blue",
  },
  {
    id: "a3",
    category: "Trust",
    question: "Why shouldn’t authenticity be machine-checked before a listing ever goes live?",
    href: "/register",
    variant: "dark",
  },
  {
    id: "a4",
    category: "Watches",
    question: "Why do five-figure watches ship with the same proof bar as a t-shirt listing?",
    href: "/product/Luxury%20watches",
    variant: "white",
  },
  {
    id: "a5",
    category: "Sellers",
    question: "Why is proving you’re legitimate harder than listing the product itself?",
    href: "/seller/upload",
    variant: "blue",
  },
  {
    id: "a6",
    category: "Buyers",
    question: "Why can’t one dashboard show delivery, rating, and lowest ask in one breath?",
    href: "/product/Sneakers",
    variant: "dark",
  },
];

export function ProblemAskGrid({
  title = "Top asks",
  subtitle = "The market spoke. Here’s what we’re solving.",
  items = DEFAULT_ASKS,
}: {
  title?: string;
  subtitle?: string;
  items?: AskItem[];
}) {
  return (
    <section className="border-t border-white/[0.06] bg-[#0a0a0a] hv-section-y text-white">
      <div className="mx-auto max-w-6xl px-4">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="max-w-3xl"
        >
          <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-white/55">{title}</p>
          <h2 className="mt-0 font-[family-name:var(--font-display)] text-3xl font-extrabold leading-[1.05] tracking-tight md:text-5xl">
            {subtitle}
          </h2>
        </motion.div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 lg:gap-5">
          {items.map((item, i) => (
            <AskCard key={item.id} item={item} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
