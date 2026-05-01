"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

import { PremiumHero } from "@/components/hero/PremiumHero";
import { AppleBentoPromo } from "@/components/home/AppleBentoPromo";
import { HowItWorks } from "@/components/home/HowItWorks";
import { PhilosophyBento } from "@/components/home/PhilosophyBento";
import { ClosingCTA } from "@/components/home/ClosingCTA";
import { HomeVaultStats } from "@/components/campaign/HomeVaultStats";
import { SearchBar } from "@/components/SearchBar";
import { Reveal, Stagger, StaggerItem } from "@/components/motion/Reveal";

const PhysicsBrandsCanvas = dynamic(
  () => import("@/components/physics/PhysicsBrandsCanvas").then((m) => m.PhysicsBrandsCanvas),
  { ssr: false, loading: () => <div className="h-[520px] w-full bg-[#0B0118]" /> }
);

const FEED_SNEAKER_IMAGES = [
  "/products/feed-dior.jpg",
  "/products/feed-balenciaga.jpg",
  "/products/feed-gucci.jpg",
  "/products/feed-lv-2.jpg",
  "/products/feed-dior-2.jpg",
];

const FEED_WATCH_IMAGES = [
  "/products/feed-richard-mille.jpg",
  "/products/feed-ap-2.jpg",
  "/products/feed-patek-2.jpg",
  "/products/watch-royal-oak.jpg",
];

const GENERIC_RECENT_ITEMS = [
  { name: "Nike Dunk Low Panda", brand: "Nike", category: "sneaker", status: "live", verdict: "AUTHENTIC" },
  { name: "Adidas Yeezy Boost 350 V2", brand: "Adidas", category: "sneaker", status: "live", verdict: "AUTHENTIC" },
  { name: "Air Jordan 1 Retro High OG", brand: "Jordan", category: "sneaker", status: "live", verdict: "AUTHENTIC" },
  { name: "New Balance 9060", brand: "New Balance", category: "sneaker", status: "live", verdict: "AUTHENTIC" },
  { name: "Rolex Submariner Date", brand: "Rolex", category: "watch", status: "live", verdict: "AUTHENTIC" },
  { name: "Omega Speedmaster Moonwatch", brand: "Omega", category: "watch", status: "live", verdict: "AUTHENTIC" },
  { name: "Audemars Piguet Royal Oak", brand: "Audemars Piguet", category: "watch", status: "live", verdict: "AUTHENTIC" },
  { name: "Patek Philippe Nautilus", brand: "Patek Philippe", category: "watch", status: "live", verdict: "AUTHENTIC" },
] as const;

export default function HomePage() {
  return (
    <>
      {/* 1. Hero */}
      <PremiumHero />

      {/* 1b. Physics playground */}
      <section className="relative border-y border-[rgba(255,31,164,0.18)] bg-[#0B0118]">
        <div className="mx-auto flex max-w-[1080px] items-end justify-between gap-6 px-5 pt-10 pb-2">
          <Reveal variant="slideLeft">
            <p className="hv-eyebrow">Inventory · live</p>
            <h2
              className="mt-2 font-[family-name:var(--font-display)] font-extrabold italic text-[#FFEDF6]"
              style={{ fontSize: "clamp(1.75rem, 3.4vw, 2.5rem)", letterSpacing: "-0.025em", lineHeight: 1.05 }}
            >
              THE LABELS THAT MATTER,{" "}
              <span className="hv-gradient-text">ALL IN ONE VAULT</span>.
            </h2>
          </Reveal>
          <Reveal variant="slideRight" delay={0.15}>
            <p className="hidden max-w-[28ch] pb-2 text-right text-[12px] uppercase tracking-[0.18em] text-[#FFEDF6]/55 md:block">
              Drag the pills. Throw them around.<br/>
              They have weight, they collide.
            </p>
          </Reveal>
        </div>
        <Reveal variant="fadeIn" delay={0.25}>
          <PhysicsBrandsCanvas height={520} />
        </Reveal>
      </section>

      {/* 2. Apple bento */}
      <AppleBentoPromo />

      {/* 3. Search band */}
      <section className="bg-[#0B0118] py-16 md:py-20">
        <div className="mx-auto max-w-2xl px-5 text-center">
          <Reveal>
            <p className="hv-eyebrow">Find an item</p>
            <h2 className="mx-auto mt-3 max-w-md font-[family-name:var(--font-display)] hv-display-md text-[#FFEDF6]">
              ANY SNEAKER. ANY GRAIL.
            </h2>
          </Reveal>
          <Reveal delay={0.15} className="mt-8">
            <SearchBar variant="dark" />
          </Reveal>
        </div>
      </section>

      {/* 4. How it works */}
      <HowItWorks />

      {/* 5. Stats */}
      <section className="bg-[#160329] py-24 md:py-32">
        <div className="mx-auto max-w-[1080px] px-5">
          <Reveal className="mb-14 max-w-xl">
            <p className="hv-eyebrow">By the numbers</p>
            <h2 className="mt-3 font-[family-name:var(--font-display)] hv-display-lg text-[#FFEDF6]">
              SCALE YOU CAN <span className="hv-gradient-text">MEASURE</span>.
            </h2>
          </Reveal>
          <HomeVaultStats />
        </div>
      </section>

      {/* 6. Philosophy bento */}
      <PhilosophyBento />

      {/* 7. Recent listings */}
      <section id="recent" className="bg-[#0B0118] py-24 md:py-32">
        <div className="mx-auto max-w-[1080px] px-5">
          <div className="mb-14 flex items-end justify-between gap-4">
            <Reveal variant="slideLeft">
              <p className="hv-eyebrow">Recently verified</p>
              <h2 className="mt-3 font-[family-name:var(--font-display)] hv-display-lg text-[#FFEDF6]">
                FRESH IN THE VAULT.
              </h2>
            </Reveal>
            <Reveal variant="slideRight" delay={0.1}>
              <Link href="/seller/upload" className="hv-chevron hidden md:inline-flex">
                List your item
              </Link>
            </Reveal>
          </div>

          <Stagger gap={0.07} className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {GENERIC_RECENT_ITEMS.map((l, idx) => {
              const fallback =
                l.category === "watch"
                  ? FEED_WATCH_IMAGES[idx % FEED_WATCH_IMAGES.length]
                  : FEED_SNEAKER_IMAGES[idx % FEED_SNEAKER_IMAGES.length];
              return (
                <StaggerItem key={`${l.category}-${l.name}`}>
                  <Link href={`/product/${encodeURIComponent(l.name)}`} prefetch className="group block h-full">
                    <motion.div
                      whileHover={{ y: -8, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } }}
                      className="hv-card h-full overflow-hidden"
                    >
                      <div className="relative aspect-square overflow-hidden bg-gradient-to-br from-[#1B0533] to-[#0B0118]">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <motion.img
                          src={fallback}
                          alt={l.name}
                          className="h-full w-full object-cover"
                          whileHover={{ scale: 1.06 }}
                          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                        />
                        {l.verdict === "AUTHENTIC" ? (
                          <span
                            className="absolute left-3 top-3 inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase text-[#0B0118] shadow"
                            style={{ background: "linear-gradient(95deg, #38EFA1, #00E1FF)" }}
                          >
                            ✓ Authentic
                          </span>
                        ) : null}
                      </div>
                      <div className="p-5">
                        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#FF1FA4]">
                          {l.brand}
                        </p>
                        <p className="mt-1 line-clamp-2 text-[15px] font-bold italic text-[#FFEDF6]">
                          {l.name}
                        </p>
                        <div className="mt-4 flex items-center justify-between border-t border-[rgba(255,31,164,0.14)] pt-3">
                          <span className="text-[11px] font-bold uppercase tracking-wide text-[#00E1FF]">
                            {l.status}
                          </span>
                          <motion.span
                            className="inline-block text-[#FFEDF6]/55 group-hover:text-[#00E1FF]"
                            whileHover={{ x: 3, y: -3 }}
                            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                          >
                            <ArrowUpRight className="h-4 w-4" />
                          </motion.span>
                        </div>
                      </div>
                    </motion.div>
                  </Link>
                </StaggerItem>
              );
            })}
          </Stagger>

        </div>
      </section>

      {/* 8. Closing CTA */}
      <ClosingCTA />
    </>
  );
}
