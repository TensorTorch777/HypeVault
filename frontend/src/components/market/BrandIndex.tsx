"use client";

import Image from "next/image";
import Link from "next/link";
import { useQueries, useQuery } from "@tanstack/react-query";

import { Skeleton } from "@/components/ui/skeleton";
import { formatPrice, searchMarket } from "@/lib/market";
import { HOUSE_SKIES, STUDIO_WATCHES } from "@/lib/studioWatches";
import { cn } from "@/lib/utils";

export type HouseCard = {
  brand: string;
  piece: string;
  city: string;
  since: string;
  render: string | null;
  backdrop: string | null;
  tint: string;
};

/**
 * One wide card per manufacture, stacking like a deck as you scroll (each sticks a little
 * lower than the last). Median price and listing count are read live from the market API.
 */
export function BrandIndex({ houses }: { houses: HouseCard[] }) {
  const results = useQueries({
    queries: houses.map((h) => ({
      queryKey: ["market-brand", h.brand],
      queryFn: () => searchMarket(h.brand, h.brand),
      staleTime: 10 * 60 * 1000,
    })),
  });

  return (
    <ol className="relative mx-auto max-w-[66.25rem] space-y-6">
      {houses.map((h, i) => {
        const r = results[i];
        const stats = r.data?.sources.chrono24_dataset.stats ?? null;
        const flip = i % 2 === 1;
        return (
          <li key={h.brand} className="sticky" style={{ top: `${(88 + i * 18) / 16}rem` }}>
            <article
              className="relative grid min-h-[21.25rem] overflow-hidden rounded-[1.375rem] border border-white/10 shadow-[0_40px_80px_-40px_rgba(0,0,0,0.9)] md:grid-cols-2"
              style={{ background: h.tint }}
            >
              {h.backdrop ? <Image src={h.backdrop} alt="" fill sizes="(min-width: 1921px) 1590px, 1060px" className="object-cover opacity-70" /> : null}
              <div className={cn("absolute inset-0", flip ? "bg-gradient-to-l" : "bg-gradient-to-r", "from-[#14112A]/85 via-[#14112A]/40 to-transparent")} />
              <div className={cn("relative z-10 flex flex-col justify-center p-7 md:p-10", flip && "md:order-2")}>
                <p className="hv-eyebrow">
                  {h.city} · since {h.since}
                </p>
                <h3 className="mt-1 font-display text-[clamp(2.4rem,4.6vw,3.6rem)] font-extrabold uppercase leading-[0.9] text-white">{h.brand}</h3>
                <div className="mt-4 min-h-[3rem] max-w-sm text-[0.9375rem] leading-6 text-white/85">
                  {r.isLoading ? (
                    <Skeleton className="h-12 w-full bg-white/10" />
                  ) : stats ? (
                    <p>
                      Median asking <span className="font-bold text-white">{formatPrice(stats.median, "USD", true)}</span> across{" "}
                      {stats.count.toLocaleString("en-US")} listings, from {formatPrice(stats.min, "USD", true)} to{" "}
                      {formatPrice(stats.max, "USD", true)}.
                    </p>
                  ) : (
                    <p>{r.isError ? "Market data could not be loaded." : "No dataset listings yet."}</p>
                  )}
                </div>
                <Link href={`/market?brand=${encodeURIComponent(h.brand)}`} className="hv-btn-pill hv-btn-primary mt-6 self-start">
                  Explore the market <span aria-hidden="true">→</span>
                </Link>
              </div>
              <div className={cn("relative min-h-[16.25rem]", flip && "md:order-1")}>
                {h.render ? (
                  <Image src={h.render} alt={`${h.brand} — ${h.piece}, studio render`} fill sizes="(min-width: 1921px) 795px, 530px" className="hv-fade-y object-contain p-4 drop-shadow-[0_30px_40px_rgba(10,11,23,0.6)]" />
                ) : null}
                <p className="hv-mono absolute bottom-3 right-4 text-[0.5625rem] text-white/55">{h.piece} · studio model</p>
              </div>
            </article>
          </li>
        );
      })}
    </ol>
  );
}

type Manifest = { watches?: Record<string, { hero?: boolean }> };

/** BrandIndex for client pages: reads the render manifest in the browser. */
export function BrandIndexAuto() {
  const manifest = useQuery({
    queryKey: ["render-manifest"],
    queryFn: async (): Promise<Manifest> => {
      const res = await fetch("/renders/manifest.json");
      return res.ok ? res.json() : {};
    },
    staleTime: Infinity,
  });
  const houses: HouseCard[] = STUDIO_WATCHES.map((s, i) => ({
    brand: s.brand,
    piece: s.piece,
    city: s.city,
    since: s.since,
    render: manifest.data?.watches?.[s.id]?.hero ? `/renders/${s.id}/hero.webp` : null,
    backdrop: null,
    tint: HOUSE_SKIES[i],
  }));
  return <BrandIndex houses={houses} />;
}
