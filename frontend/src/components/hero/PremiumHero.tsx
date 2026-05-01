"use client";

import Link from "next/link";
import { useMemo } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";

import { BouncyText, WobbleText } from "@/components/motion/BouncyText";
import { fetchMe } from "@/lib/api";

/**
 * GTA VI / Vice City hero — sunset gradient, neon italic display type.
 */
export function PremiumHero() {
  const meQuery = useQuery({
    queryKey: ["auth-me-hero"],
    queryFn: fetchMe,
    retry: false,
  });

  const greetingName = useMemo(() => {
    const email = meQuery.data?.email;
    if (!email) return null;
    const left = email.split("@")[0] ?? "";
    return left
      .split(/[._-]+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }, [meQuery.data?.email]);

  return (
    <section className="relative isolate overflow-hidden pt-32 md:pt-40">
      {/* Sunset orb */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-[12vh] -z-[1] h-[60vmin] w-[60vmin] -translate-x-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(circle, #FFD24A 0%, #FF7A1A 30%, #FF1FA4 70%, transparent 75%)",
          filter: "blur(40px)",
          opacity: 0.55,
        }}
      />
      {/* Palm silhouette band */}
      <PalmBand />

      <div className="relative z-[2] mx-auto max-w-[1080px] px-5 text-center">
        {/* Top greeting — defaults to chip, becomes big GTA-style greeting when logged in */}
        {greetingName ? (
          <motion.p
            initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto max-w-[980px] text-balance font-[family-name:var(--font-display)] text-[30px] font-extrabold italic leading-none tracking-[-0.03em] md:text-[56px]"
            style={{
              background: "linear-gradient(95deg, #FFD24A 8%, #FF7A1A 35%, #FF1FA4 65%, #9B5BFF 92%)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
              filter: "drop-shadow(0 0 16px rgba(255,31,164,0.35))",
            }}
          >
            <WobbleText text={`WELCOME, ${greetingName.toUpperCase()}.`} />
          </motion.p>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto w-fit"
          >
            <motion.div
              className="inline-flex items-center gap-2 rounded-full border border-[rgba(255,31,164,0.4)] bg-[rgba(255,31,164,0.08)] px-4 py-1.5 backdrop-blur"
              animate={{ boxShadow: ["0 0 0 0 rgba(255,31,164,0)", "0 0 30px 0 rgba(255,31,164,0.45)", "0 0 0 0 rgba(255,31,164,0)"] }}
              transition={{ duration: 3.2, ease: "easeInOut", repeat: Infinity }}
            >
              <motion.span
                className="h-1.5 w-1.5 rounded-full bg-[#FF1FA4]"
                animate={{ scale: [1, 1.6, 1], opacity: [1, 0.55, 1] }}
                transition={{ duration: 1.6, ease: "easeInOut", repeat: Infinity }}
              />
              <span className="text-[11px] font-bold uppercase tracking-[0.32em] text-[#FF1FA4]">
                Welcome to the vault
              </span>
            </motion.div>
          </motion.div>
        )}

        {/* Headline — Line 1: smooth blur-rise. Line 2: bouncy letter-by-letter. */}
        <h1 className="mt-7 font-[family-name:var(--font-display)] hv-display-xl">
          <motion.span
            initial={{ opacity: 0, y: 30, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.15, duration: 0.95, ease: [0.22, 1, 0.36, 1] }}
            className="hv-neon-title block"
          >
            <WobbleText text="REAL LUXURY." />
          </motion.span>
          <span className="hv-hero-accent-text mt-1 block">
            <BouncyText text="VERIFIED." delay={0.7} stagger={0.07} bounce={0.65} />
          </span>
        </h1>

        {/* Sub */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18, duration: 0.6 }}
          className="mx-auto mt-7 max-w-2xl text-[18px] leading-[1.45] text-[#FFEDF6]/75 md:text-[22px]"
        >
          A vision-transformer inspects every listing before it&apos;s live.
          Then we show you the truth across StockX, Chrono24 &amp; eBay.
        </motion.p>

        {/* CTAs — magnetic hover */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="mt-10 flex flex-wrap items-center justify-center gap-4"
        >
          {!meQuery.data ? (
            <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}>
              <Link href="/register" className="hv-btn-pill hv-btn-primary">
                Enter the vault
              </Link>
            </motion.div>
          ) : null}
          <motion.div whileHover={{ x: 4 }} transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}>
            <Link href="/#how" className="hv-chevron">
              See the method
            </Link>
          </motion.div>
        </motion.div>
      </div>

      {/* Hero device card */}
      <motion.div
        initial={{ opacity: 0, y: 32, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ delay: 0.4, duration: 0.95, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-[2] mx-auto mt-16 max-w-[1080px] px-5 md:mt-20"
      >
        <HeroDeviceMock />
      </motion.div>
    </section>
  );
}

function PalmBand() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 1600 240"
      preserveAspectRatio="xMidYMax slice"
      className="pointer-events-none absolute inset-x-0 top-[42vh] -z-[1] h-[35vh] w-full opacity-80"
    >
      {/* horizon line */}
      <line x1="0" y1="180" x2="1600" y2="180" stroke="#FF1FA4" strokeWidth="1" opacity="0.4" />
      {/* palm 1 */}
      <Palm x={140} scale={1.2} />
      <Palm x={1380} scale={1.4} />
      <Palm x={420} scale={0.8} />
    </svg>
  );
}

function Palm({ x, scale = 1 }: { x: number; scale?: number }) {
  return (
    <g transform={`translate(${x},180) scale(${scale})`} fill="#0B0118">
      {/* trunk */}
      <path d="M-3 0 C -2 -40, 1 -80, 0 -120 L 5 -120 C 6 -80, 4 -40, 3 0 Z" />
      {/* fronds */}
      <path d="M0 -120 C -40 -130, -70 -120, -90 -100 C -65 -115, -30 -125, 0 -120 Z" />
      <path d="M0 -120 C 40 -130, 70 -120, 90 -100 C 65 -115, 30 -125, 0 -120 Z" />
      <path d="M0 -120 C -25 -150, -20 -175, 0 -190 C 5 -160, 5 -135, 0 -120 Z" />
      <path d="M0 -120 C 25 -150, 20 -175, 0 -190 C -5 -160, -5 -135, 0 -120 Z" />
      <path d="M0 -120 C -55 -135, -85 -150, -100 -150 C -55 -150, -25 -130, 0 -120 Z" />
      <path d="M0 -120 C 55 -135, 85 -150, 100 -150 C 55 -150, 25 -130, 0 -120 Z" />
    </g>
  );
}

function HeroDeviceMock() {
  return (
    <div className="hv-neon-frame relative mx-auto aspect-[16/9] w-full overflow-hidden rounded-[28px] bg-[#0B0118]">
      {/* window chrome */}
      <div className="absolute inset-x-0 top-0 z-[3] flex h-9 items-center gap-2 border-b border-[rgba(255,31,164,0.18)] bg-[#0B0118]/70 px-4 backdrop-blur">
        <span className="h-2.5 w-2.5 rounded-full bg-[#FF1FA4]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#FFD24A]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#00E1FF]" />
        <span className="ml-4 text-[12px] font-bold uppercase tracking-[0.18em] text-[#FFEDF6]/55">
          hypevault.app/verify
        </span>
      </div>

      <div className="absolute inset-x-0 top-9 bottom-0 grid grid-cols-1 gap-6 p-6 md:grid-cols-[5fr,4fr] md:gap-10 md:p-10">
        {/* Product card */}
        <div
          className="relative overflow-hidden rounded-2xl border border-[rgba(255,31,164,0.25)]"
          style={{
            background:
              "radial-gradient(120% 90% at 50% 0%, #2A0440 0%, #160329 70%, #0a0118 100%)",
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/products/watch-royal-oak.jpg"
            alt="Audemars Piguet Royal Oak"
            className="absolute inset-0 h-full w-full object-cover opacity-90"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "linear-gradient(180deg, rgba(11,1,24,0.15) 0%, rgba(11,1,24,0.55) 60%, rgba(11,1,24,0.92) 100%)",
            }}
          />
          {/* Verified badge */}
          <div
            className="absolute left-5 top-5 inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-white shadow-lg"
            style={{
              background: "linear-gradient(95deg, #38EFA1, #00E1FF)",
              color: "#0B0118",
            }}
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="none">
              <path d="M5 10.5L8.5 14L15 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-[12px] font-bold uppercase tracking-wide">AUTHENTIC</span>
          </div>
          {/* Confidence */}
          <div className="absolute right-5 top-5 rounded-full border border-[rgba(255,237,246,0.18)] bg-black/50 px-3 py-1.5 text-[11px] font-bold tracking-wider text-white backdrop-blur">
            99.4% CONF
          </div>
          {/* Price strip */}
          <div className="absolute inset-x-5 bottom-5 flex items-end justify-between">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#FF1FA4]">
                Audemars Piguet
              </p>
              <p className="font-[family-name:var(--font-display)] text-[20px] font-extrabold italic text-white md:text-[24px]">
                Royal Oak 15500ST
              </p>
            </div>
            <div className="text-right">
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#00E1FF]">
                Lowest ask
              </p>
              <p className="font-[family-name:var(--font-display)] text-[24px] font-extrabold italic tabular-nums text-white md:text-[30px]">
                $33,900
              </p>
            </div>
          </div>
        </div>

        {/* Comparison rows */}
        <div className="grid grid-rows-3 gap-3">
          {[
            { brand: "StockX",   bg: "#38EFA1", code: "SX",  price: "$33,900", note: "5–7 days · 4.8★" },
            { brand: "Chrono24", bg: "#00E1FF", code: "C24", price: "$34,200", note: "7–10 days · 4.9★" },
            { brand: "eBay",     bg: "#FFD24A", code: "eB",  price: "$33,750", note: "3–5 days · 4.6★" },
          ].map((r) => (
            <div
              key={r.brand}
              className="flex items-center justify-between rounded-xl border border-[rgba(255,237,246,0.08)] bg-[#160329]/80 px-4 py-3 backdrop-blur"
            >
              <div className="flex items-center gap-3">
                <span
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-[10px] font-extrabold italic"
                  style={{ background: r.bg, color: "#0B0118" }}
                >
                  {r.code}
                </span>
                <div>
                  <p className="text-[14px] font-bold italic text-[#FFEDF6]">{r.brand}</p>
                  <p className="text-[11px] text-[#FFEDF6]/55">{r.note}</p>
                </div>
              </div>
              <p className="font-[family-name:var(--font-display)] text-[16px] font-extrabold italic tabular-nums text-[#FFEDF6]">
                {r.price}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
