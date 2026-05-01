"use client";

import Link from "next/link";
import { motion } from "framer-motion";

/**
 * Apple's signature 2-column promotional bento.
 * Each tile: massive headline → sub → two chips → product visual at the bottom.
 */
export function AppleBentoPromo() {
  return (
    <section className="bg-[#FBFBFD] py-4">
      <div className="mx-auto grid max-w-[1080px] gap-2 px-2 md:grid-cols-2 md:gap-3 md:px-3">
        {/* SNEAKERS — light tile */}
        <BentoTile
          tone="light"
          eyebrow="HypeVault for Sneakers"
          title="Heat, only real."
          subtitle="Designer drops, runway pieces — every photo screened by AI."
          ctaPrimary={{ label: "Shop sneakers",  href: "/product/Sneakers" }}
          ctaSecondary={{ label: "How AI sees ›", href: "/#how" }}
        >
          <SneakerVisual />
        </BentoTile>

        {/* WATCHES — dark tile */}
        <BentoTile
          tone="dark"
          eyebrow="HypeVault for Watches"
          title="Quiet luxury, loud trust."
          subtitle="From Royal Oak to RM. Compare grail prices in one tap."
          ctaPrimary={{ label: "Shop watches", href: "/product/Luxury%20watches" }}
          ctaSecondary={{ label: "See pricing ›", href: "/product/Luxury%20watches" }}
        >
          <WatchVisual />
        </BentoTile>
      </div>
    </section>
  );
}

function BentoTile({
  tone,
  eyebrow,
  title,
  subtitle,
  ctaPrimary,
  ctaSecondary,
  children,
}: {
  tone: "light" | "dark";
  eyebrow: string;
  title: string;
  subtitle: string;
  ctaPrimary: { label: string; href: string };
  ctaSecondary: { label: string; href: string };
  children: React.ReactNode;
}) {
  const isDark = tone === "dark";

  return (
    <motion.article
      initial={{ opacity: 0, y: 30, filter: "blur(8px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      whileHover={{ y: -6 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.95, ease: [0.22, 1, 0.36, 1] }}
      className={`group relative isolate flex aspect-[5/6] flex-col overflow-hidden rounded-[28px] md:aspect-[4/5] ${
        isDark ? "bg-[#160329] text-white" : "bg-[#FFE9F4] text-[#0B0118]"
      }`}
    >
      <div className="relative z-[2] px-8 pt-10 text-center md:px-14 md:pt-16">
        <p
          className={`text-[14px] font-medium ${
            isDark ? "text-white/55" : "text-[#86868B]"
          }`}
        >
          {eyebrow}
        </p>
        <h3
          className="mt-2 font-[family-name:var(--font-display)] font-bold leading-[1.05]"
          style={{
            fontSize: "clamp(2rem, 4.6vw, 3.25rem)",
            letterSpacing: "-0.03em",
          }}
        >
          {title}
        </h3>
        <p
          className={`mx-auto mt-5 max-w-md text-[16px] leading-[1.4] md:text-[18px] ${
            isDark ? "text-white/70" : "text-[#1D1D1F]/70"
          }`}
        >
          {subtitle}
        </p>

        <div className="mt-7 flex flex-wrap items-center justify-center gap-x-6 gap-y-3">
          <Link
            href={ctaPrimary.href}
            className="hv-btn-pill hv-btn-blue !min-h-[36px] !text-[14px] !px-5"
          >
            {ctaPrimary.label}
          </Link>
          <Link
            href={ctaSecondary.href}
            className={`text-[15px] ${
              isDark
                ? "text-[#2997FF] hover:text-[#5cb3ff]"
                : "text-[#0066CC] hover:text-[#1d80f0]"
            }`}
          >
            {ctaSecondary.label}
          </Link>
        </div>
      </div>

      <div className="relative z-[1] mt-auto h-1/2 w-full">{children}</div>
    </motion.article>
  );
}

/* ─── Real Louis Vuitton sneaker shot ──────────────────────── */
function SneakerVisual() {
  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/products/sneaker-louis-vuitton.jpg"
        alt="Louis Vuitton trainer"
        className="absolute inset-x-0 bottom-0 h-full w-full object-cover object-center"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(245,245,247,0.75) 0%, rgba(245,245,247,0) 25%, rgba(245,245,247,0) 100%)",
        }}
      />
    </div>
  );
}

/* ─── Real Patek Nautilus shot ───────────────────────────── */
function WatchVisual() {
  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/products/watch-patek-nautilus.jpg"
        alt="Patek Philippe Nautilus"
        className="absolute inset-x-0 bottom-0 h-full w-full object-cover object-center"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(15,15,15,0.75) 0%, rgba(15,15,15,0) 25%, rgba(15,15,15,0.4) 100%)",
        }}
      />
    </div>
  );
}
