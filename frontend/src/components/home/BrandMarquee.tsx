"use client";

import { LUXURY_SNEAKER_BRANDS, LUXURY_WATCH_BRANDS } from "@/lib/catalog";

const ROW_1 = [...LUXURY_SNEAKER_BRANDS, ...LUXURY_WATCH_BRANDS.slice(0, 3)];
const ROW_2 = [...LUXURY_WATCH_BRANDS.slice(3), ...LUXURY_SNEAKER_BRANDS.slice(0, 3).reverse()];

function Row({ items, reverse = false, speed = "animate-hv-marquee" }: { items: string[]; reverse?: boolean; speed?: string }) {
  const doubled = [...items, ...items];
  return (
    <div className="relative overflow-hidden hv-marquee-mask">
      <div
        className={`flex w-max ${speed} gap-x-14`}
        style={reverse ? { animationDirection: "reverse" } : undefined}
      >
        {doubled.map((b, i) => (
          <span
            key={`${b}-${i}`}
            className="flex shrink-0 items-center gap-5 whitespace-nowrap font-[family-name:var(--font-display)] text-[clamp(1.6rem,3vw,3rem)] font-semibold tracking-tight text-[#1D1D1F]/25 transition-colors duration-300 hover:text-[#1D1D1F]"
          >
            {b}
            <span aria-hidden className="inline-block h-1 w-1 rounded-full bg-[#C7C7CC]" />
          </span>
        ))}
      </div>
    </div>
  );
}

export function BrandMarquee() {
  return (
    <section className="border-y border-black/[0.06] bg-white py-10">
      <p className="hv-eyebrow mb-6 text-center !text-[12px]">
        Inventory from the labels you actually want
      </p>
      <div className="flex flex-col gap-3">
        <Row items={ROW_1} speed="animate-hv-marquee-slow" />
        <Row items={ROW_2} speed="animate-hv-marquee" reverse />
      </div>
    </section>
  );
}
