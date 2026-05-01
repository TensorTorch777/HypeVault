"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

/**
 * Editorial "How it works" — alternating image/copy rows with a HypeVault-style
 * proprietary feel: a vertical orange timeline rail connects the three steps.
 */
export function HowItWorks() {
  return (
    <section
      id="how"
      className="relative bg-[#0B0118] py-28 md:py-36"
    >
      <div className="mx-auto max-w-[1080px] px-5">
        <p className="hv-eyebrow">The HypeVault method</p>
        <motion.h2
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-40px" }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="mt-3 max-w-3xl font-[family-name:var(--font-display)] hv-display-lg text-[#FFEDF6]"
        >
          A NEW STANDARD OF TRUST, BUILT AROUND A TRANSFORMER THAT{" "}
          <span className="hv-gradient-text">SEES GEOMETRY</span>.
        </motion.h2>
      </div>

      {/* Step 1 — visual LEFT */}
      <Step
        index={1}
        flip={false}
        eyebrow="Inspection"
        title="Every photo is read by DINOv2-Giant."
        body="A 1.1-billion-parameter vision transformer reads each listing's structural geometry — lace threading, sole curvature, dial symmetry — long before a buyer sees it."
        cta={{ label: "Read the research", href: "/#how" }}
      >
        <ScanMockup />
      </Step>

      {/* Step 2 — visual RIGHT */}
      <Step
        index={2}
        flip
        eyebrow="Live market"
        title="Three platforms. One transparent surface."
        body="We pull real-time prices, delivery windows and seller ratings from StockX, Chrono24 and eBay. No tab-hopping, no hidden fees — just the truth."
        cta={{ label: "See the comparison", href: "/product/Sneakers" }}
      >
        <PriceChartMockup />
      </Step>

      {/* Step 3 — visual LEFT */}
      <Step
        index={3}
        flip={false}
        last
        eyebrow="The verdict"
        title="Authentic. Confident. Or it doesn't ship."
        body="A green AUTHENTIC badge means the model reached ≥95% confidence. Anything lower is flagged for human review — never quietly listed."
        cta={{ label: "Read the policy", href: "/" }}
      >
        <VerdictMockup />
      </Step>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────
   Editorial alternating row — proper grid this time
   ───────────────────────────────────────────────────────────── */

function Step({
  index,
  flip,
  last = false,
  eyebrow,
  title,
  body,
  cta,
  children,
}: {
  index: number;
  flip: boolean;
  last?: boolean;
  eyebrow: string;
  title: string;
  body: string;
  cta: { label: string; href: string };
  children: React.ReactNode;
}) {
  const stepLabel = index.toString().padStart(2, "0");

  return (
    <div className="relative">
      <div className="mx-auto mt-24 grid max-w-[1080px] grid-cols-1 items-center gap-10 px-5 md:mt-32 md:grid-cols-12 md:gap-16">
        {/* Visual column */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className={`md:col-span-7 ${flip ? "md:order-2" : "md:order-1"}`}
        >
          {children}
        </motion.div>

        {/* Copy column */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ delay: 0.1, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className={`md:col-span-5 ${flip ? "md:order-1" : "md:order-2"}`}
        >
          {/* Step badge — proprietary HV stamp */}
          <div className="inline-flex items-center gap-3">
            <motion.span
              className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-[#FF1FA4] font-[family-name:var(--font-display)] text-[13px] font-extrabold italic text-[#FF1FA4]"
              style={{ boxShadow: "0 0 24px rgba(255,31,164,0.45)" }}
              animate={{ boxShadow: [
                "0 0 18px rgba(255,31,164,0.4)",
                "0 0 36px rgba(255,31,164,0.7)",
                "0 0 18px rgba(255,31,164,0.4)",
              ] }}
              transition={{ duration: 3.4, ease: "easeInOut", repeat: Infinity }}
            >
              {stepLabel}
            </motion.span>
            <span className="text-[12px] font-bold uppercase tracking-[0.22em] text-[#00E1FF]" style={{ textShadow: "0 0 14px rgba(0,225,255,0.45)" }}>
              {eyebrow}
            </span>
          </div>

          <h3
            className="mt-6 font-[family-name:var(--font-display)] font-extrabold italic text-[#FFEDF6]"
            style={{
              fontSize: "clamp(1.75rem, 3.4vw, 2.5rem)",
              lineHeight: 1.08,
              letterSpacing: "-0.025em",
            }}
          >
            {title}
          </h3>
          <p className="mt-5 text-[17px] leading-[1.5] text-[#FFEDF6]/65">
            {body}
          </p>
          <Link href={cta.href} className="hv-chevron mt-6 inline-flex">
            {cta.label}
          </Link>
        </motion.div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   MOCKUPS
   ───────────────────────────────────────────────────────────── */

function ScanMockup() {
  return (
    <div
      className="relative aspect-[4/3] w-full overflow-hidden rounded-3xl border border-black/[0.06]"
      style={{
        boxShadow:
          "0 30px 80px -30px rgba(0,0,0,0.18), 0 10px 30px -10px rgba(0,0,0,0.06)",
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/products/feed-richard-mille.jpg"
        alt="Watch under inspection"
        className="absolute inset-0 h-full w-full object-cover"
      />

      {/* Scan grid */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,59,0,0.55) 1px, transparent 1px), linear-gradient(90deg, rgba(255,59,0,0.55) 1px, transparent 1px)",
          backgroundSize: "36px 36px",
        }}
      />

      {/* Scan line */}
      <ScanLine />

      {/* Targets */}
      <Target x="22%" y="24%" label="Bezel" />
      <Target x="68%" y="44%" label="Crown" />
      <Target x="40%" y="68%" label="Tourbillon" />

      {/* HUD top-left */}
      <div className="absolute left-5 top-5 inline-flex items-center gap-2 rounded-full bg-black/75 px-3 py-1.5 backdrop-blur-md">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#FF3B00]" />
        <span className="text-[11px] font-semibold tracking-wide text-white">
          DINOv2-G · 184ms
        </span>
      </div>

      {/* HUD bottom-right */}
      <div className="absolute right-5 bottom-5 rounded-xl bg-white/95 px-3 py-2 backdrop-blur-md shadow-lg">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-[#86868B]">
          Confidence
        </p>
        <p className="font-[family-name:var(--font-display)] text-[18px] font-bold tabular-nums text-[#1D1D1F]">
          99.4%
        </p>
      </div>
    </div>
  );
}

function ScanLine() {
  return (
    <motion.div
      aria-hidden
      initial={{ y: "-10%" }}
      whileInView={{ y: "110%" }}
      viewport={{ once: false }}
      transition={{ duration: 3, ease: "linear", repeat: Infinity }}
      className="absolute inset-x-0 h-[2px]"
      style={{
        background:
          "linear-gradient(90deg, transparent, #FF3B00 50%, transparent)",
        boxShadow: "0 0 16px 4px rgba(255,59,0,0.35)",
      }}
    />
  );
}

function Target({ x, y, label }: { x: string; y: string; label: string }) {
  return (
    <div
      className="absolute h-12 w-12 -translate-x-1/2 -translate-y-1/2"
      style={{ left: x, top: y }}
    >
      <span className="absolute inset-0 animate-ping rounded-full border-2 border-[#FF3B00] opacity-60" />
      <span className="absolute inset-0 rounded-full border-2 border-[#FF3B00] shadow-[0_0_0_8px_rgba(255,59,0,0.12)]" />
      <span className="absolute -bottom-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-[#FF3B00] px-2 py-0.5 text-[10px] font-semibold text-white">
        {label}
      </span>
    </div>
  );
}

function PriceChartMockup() {
  // Build a smooth animated price curve as SVG
  const data = [42, 38, 45, 41, 50, 47, 53, 49, 56, 52, 60, 58];
  const max = Math.max(...data);
  const min = Math.min(...data);

  const platforms = [
    { name: "StockX",   price: "$33,900", color: "#00C851", bestValue: true,  delta: "+0%" },
    { name: "Chrono24", price: "$34,200", color: "#0088FF", bestValue: false, delta: "+0.9%" },
    { name: "eBay",     price: "$33,750", color: "#FFD600", bestValue: false, lowest: true, delta: "−0.4%" },
  ];

  // Build chart path
  const chartW = 360;
  const chartH = 80;
  const stepX = chartW / (data.length - 1);
  const points = data.map((v, i) => {
    const x = i * stepX;
    const y = chartH - ((v - min) / (max - min)) * chartH;
    return [x, y] as const;
  });
  const pathD = points
    .map(([x, y], i) => (i === 0 ? `M${x},${y}` : `L${x},${y}`))
    .join(" ");
  const fillD = pathD + ` L${chartW},${chartH} L0,${chartH} Z`;

  return (
    <div
      className="relative aspect-[4/3] w-full overflow-hidden rounded-3xl border border-black/[0.06] bg-white p-7 md:p-8"
      style={{
        boxShadow:
          "0 30px 80px -30px rgba(0,0,0,0.18), 0 10px 30px -10px rgba(0,0,0,0.06)",
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-[#86868B]">
            Royal Oak 15500ST · 30-day
          </p>
          <p className="font-[family-name:var(--font-display)] text-[26px] font-bold text-[#1D1D1F] tabular-nums">
            $33,900
          </p>
          <p className="text-[12px] font-medium text-[#00A652]">
            ↓ 4.2% vs last month
          </p>
        </div>
        <div className="flex gap-1 rounded-full bg-[#F5F5F7] p-0.5 text-[11px] font-semibold">
          {["7D", "30D", "1Y"].map((r, i) => (
            <span
              key={r}
              className={`rounded-full px-3 py-1 ${
                i === 1 ? "bg-white text-[#1D1D1F] shadow-sm" : "text-[#86868B]"
              }`}
            >
              {r}
            </span>
          ))}
        </div>
      </div>

      {/* Chart */}
      <svg
        viewBox={`0 0 ${chartW} ${chartH}`}
        className="mt-5 w-full"
        preserveAspectRatio="none"
        style={{ height: 80 }}
      >
        <defs>
          <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"  stopColor="#FF3B00" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#FF3B00" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={fillD} fill="url(#priceGrad)" />
        <path d={pathD} stroke="#FF3B00" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        {/* Last point dot */}
        <circle
          cx={points[points.length - 1][0]}
          cy={points[points.length - 1][1]}
          r="4"
          fill="#FF3B00"
        />
        <circle
          cx={points[points.length - 1][0]}
          cy={points[points.length - 1][1]}
          r="9"
          fill="#FF3B00"
          fillOpacity="0.18"
        />
      </svg>

      {/* Platform list */}
      <div className="mt-5 space-y-2">
        {platforms.map((p) => (
          <div
            key={p.name}
            className="flex items-center justify-between rounded-xl border border-black/[0.05] bg-[#FBFBFD] px-3 py-2.5"
          >
            <div className="flex items-center gap-2.5">
              <span className="h-7 w-7 rounded-md" style={{ backgroundColor: p.color }} />
              <span className="text-[13px] font-semibold text-[#1D1D1F]">{p.name}</span>
              {p.lowest ? (
                <span className="rounded-full bg-[#00A652]/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-[#00A652]">
                  Lowest
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-medium text-[#86868B] tabular-nums">
                {p.delta}
              </span>
              <span className="font-[family-name:var(--font-display)] text-[14px] font-bold tabular-nums text-[#1D1D1F]">
                {p.price}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function VerdictMockup() {
  return (
    <div
      className="relative aspect-[4/3] w-full overflow-hidden rounded-3xl border border-black/[0.06]"
      style={{
        background: "radial-gradient(120% 90% at 70% 0%, #DEFBE9 0%, #FBFBFD 75%)",
        boxShadow:
          "0 30px 80px -30px rgba(0,0,0,0.18), 0 10px 30px -10px rgba(0,0,0,0.06)",
      }}
    >
      {/* Pulse rings */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="absolute h-56 w-56 animate-ping rounded-full border border-[#00A652]/20" />
        <div
          className="absolute h-72 w-72 animate-ping rounded-full border border-[#00A652]/10"
          style={{ animationDelay: "1s" }}
        />
      </div>

      {/* Verdict card */}
      <div className="absolute left-1/2 top-1/2 w-[80%] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-[0_20px_50px_-10px_rgba(0,0,0,0.18)]">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#00A652]">
            <svg className="h-6 w-6 text-white" viewBox="0 0 20 20" fill="none">
              <path
                d="M5 10.5L8.5 14L15 7"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[#00A652]">
              Verdict
            </p>
            <p className="font-[family-name:var(--font-display)] text-[22px] font-bold text-[#1D1D1F]">
              AUTHENTIC
            </p>
          </div>
        </div>

        {/* Confidence meter */}
        <div className="mt-6">
          <div className="flex items-baseline justify-between">
            <p className="text-[11px] font-semibold text-[#86868B]">Model confidence</p>
            <p className="font-[family-name:var(--font-display)] text-[20px] font-bold tabular-nums text-[#1D1D1F]">
              99.4<span className="text-[14px] text-[#86868B]">%</span>
            </p>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#F5F5F7]">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#FF3B00] via-[#FFAA00] to-[#00A652]"
              style={{ width: "99.4%" }}
            />
          </div>
          <div className="mt-2 flex justify-between text-[9px] font-medium text-[#86868B]">
            <span>FAKE</span>
            <span>REVIEW</span>
            <span>AUTHENTIC ›</span>
          </div>
        </div>

        {/* Listing meta */}
        <div className="mt-5 flex items-center justify-between border-t border-black/[0.06] pt-4">
          <div>
            <p className="text-[11px] font-medium text-[#86868B]">Audemars Piguet</p>
            <p className="text-[14px] font-semibold text-[#1D1D1F]">Royal Oak 15500ST</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] font-medium text-[#86868B]">Status</p>
            <p className="text-[13px] font-semibold text-[#00A652]">Live now ›</p>
          </div>
        </div>
      </div>
    </div>
  );
}
