"use client";

import { motion } from "framer-motion";
import Link from "next/link";

import { cn } from "@/lib/utils";

type Tone = "paper" | "accent" | "electric";

type Taunt = {
  id: string;
  eyebrow: string;
  line: string;
  href: string;
  position: string;
  z: number;
  rotate: number;
  tone: Tone;
};

/** Glass taunts — kept sparse so the hero headline + search stay the focus. */
const TAUNTS: Taunt[] = [
  {
    id: "t1",
    eyebrow: "ORDER UPDATE",
    line: "Still trusting one blurry photo?",
    href: "/register",
    position: "left-[1%] top-[6%] md:left-[2%]",
    z: 14,
    rotate: -5,
    tone: "accent",
  },
  {
    id: "t3",
    eyebrow: "RIDER NEARBY",
    line: "Authenticity 2 min away — in our dreams.",
    href: "/register",
    position: "right-[1%] top-[5%] md:right-[3%]",
    z: 18,
    rotate: 4,
    tone: "electric",
  },
  {
    id: "t5",
    eyebrow: "HOT DEAL",
    line: "Fake drop energy · real bank damage",
    href: "/product/Sneakers",
    position: "left-[0%] top-[28%] hidden sm:block md:left-[2%]",
    z: 20,
    rotate: -6,
    tone: "accent",
  },
  {
    id: "t6",
    eyebrow: "STATUS",
    line: "Box pics loading… trust still buffering",
    href: "/seller/upload",
    position: "right-[0%] top-[26%] hidden sm:block md:right-[2%]",
    z: 12,
    rotate: 5,
    tone: "paper",
  },
  {
    id: "t9",
    eyebrow: "FLASH",
    line: "Price matched. Peace of mind sold out.",
    href: "/product/Sneakers",
    position: "left-[1%] top-[52%] hidden md:block md:left-[3%]",
    z: 16,
    rotate: 6,
    tone: "accent",
  },
  {
    id: "t10",
    eyebrow: "LIVE",
    line: "Your size just ‘verified’ somewhere else 👀",
    href: "/product/Sneakers",
    position: "right-[1%] top-[50%] hidden md:block md:right-[4%]",
    z: 10,
    rotate: -5,
    tone: "paper",
  },
  {
    id: "t15",
    eyebrow: "TRACKING",
    line: "Your parcel of trust is… rerouted again.",
    href: "/register",
    position: "left-[2%] bottom-[12%] hidden lg:block md:left-[4%]",
    z: 22,
    rotate: 3,
    tone: "accent",
  },
  {
    id: "t16",
    eyebrow: "SAVED",
    line: "Wishlist full. Proof folder empty.",
    href: "/seller/upload",
    position: "right-[2%] bottom-[10%] hidden lg:block md:right-[5%]",
    z: 17,
    rotate: -6,
    tone: "paper",
  },
];

const glassShell: Record<Tone, string> = {
  paper: "hv-glass-dark text-white",
  accent: "hv-glass-dark--accent text-white",
  electric: "hv-glass-dark--electric text-white",
};

type Props = {
  className?: string;
};

export function HeroTauntCollage({ className }: Props) {
  return (
    <div
      className={cn("pointer-events-none select-none", className)}
      aria-label="Campaign style notifications"
    >
      <p className="pointer-events-none absolute left-3 top-20 z-[1] max-w-[10rem] text-[7px] font-bold uppercase leading-relaxed tracking-[0.28em] text-white/25 md:left-5 md:top-24 md:text-[8px]">
        Taunt feed · not real orders
      </p>
      <div className="absolute inset-0 overflow-hidden">
        {TAUNTS.map((t, i) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 1, scale: 0.98, rotate: t.rotate }}
            animate={{ opacity: 1, scale: 1, rotate: t.rotate }}
            transition={{ delay: 0.04 + i * 0.025, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            className={cn(
              "pointer-events-auto absolute w-[min(280px,88vw)] sm:w-[min(310px,52vw)] md:w-[min(340px,40vw)] lg:w-[min(380px,34vw)]",
              t.position
            )}
            style={{ zIndex: t.z }}
          >
            <Link
              href={t.href}
              className="block focus-visible:rounded-2xl focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/80"
            >
              <div
                className={cn(
                  "p-4 transition-all duration-300 md:p-5",
                  "hover:-translate-y-1 hover:shadow-[0_24px_60px_rgba(0,0,0,0.55)] hover:brightness-[1.05]",
                  glassShell[t.tone]
                )}
              >
                <div className="flex items-start gap-3 md:gap-3.5">
                  <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-gradient-to-br from-white/90 to-white/30 shadow-[0_0_14px_rgba(255,255,255,0.4)] ring-1 ring-white/35 md:mt-1.5 md:h-3 md:w-3" />
                  <div className="min-w-0">
                    <p className="text-[9px] font-bold uppercase leading-tight tracking-[0.18em] text-white/55 md:text-[10px]">
                      {t.eyebrow}
                    </p>
                    <p className="mt-2 font-[family-name:var(--font-display)] text-[13px] font-extrabold leading-snug tracking-tight text-white/95 sm:text-sm md:mt-2.5 md:text-[15px] lg:text-base">
                      {t.line}
                    </p>
                  </div>
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
