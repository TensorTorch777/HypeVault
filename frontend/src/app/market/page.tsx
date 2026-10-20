"use client";

import Image from "next/image";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";

import { BrandIndexAuto } from "@/components/market/BrandIndex";
import { MarketPanel } from "@/components/market/MarketPanel";
import { type MarketSort, searchMarket } from "@/lib/market";
import { STUDIO_WATCHES } from "@/lib/studioWatches";
import { cn } from "@/lib/utils";

const SORTS: MarketSort[] = ["near_median", "lowest", "highest"];

function MarketView() {
  const router = useRouter();
  const params = useSearchParams();
  const brand = params.get("brand");
  const q = params.get("q") ?? "";
  const sortParam = params.get("sort") as MarketSort | null;
  const sort: MarketSort = sortParam && SORTS.includes(sortParam) ? sortParam : "near_median";
  const [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [q]);

  const house = STUDIO_WATCHES.find((w) => w.brand === brand) ?? null;
  const effectiveQuery = q.trim().length >= 2 ? q.trim() : brand ?? "";
  const query = useQuery({
    queryKey: ["market-search", effectiveQuery, brand, sort],
    queryFn: () => searchMarket(effectiveQuery, brand, { sort, limit: 24 }),
    enabled: effectiveQuery.length >= 2,
    placeholderData: (prev) => prev,
  });

  const go = (next: { q?: string; brand?: string | null; sort?: MarketSort }) => {
    const sp = new URLSearchParams();
    const nq = next.q ?? q;
    const nb = next.brand === undefined ? brand : next.brand;
    const ns = next.sort ?? sort;
    if (nq.trim()) sp.set("q", nq.trim());
    if (nb) sp.set("brand", nb);
    if (ns !== "near_median") sp.set("sort", ns);
    router.push(`/market${sp.toString() ? `?${sp}` : ""}`, { scroll: false });
  };

  const scene = house?.backdrop ?? "bay_dusk";
  const title = q.trim() ? `“${q.trim()}”` : house ? house.brand : "The market";

  return (
    <div className="pb-24">
      {/* Cinematic header with search */}
      <header className="relative isolate overflow-hidden">
        <div className="hv-sky absolute inset-0 -z-20" aria-hidden="true" />
        <Image key={scene} src={`/renders/backdrops/${scene}_0.webp`} alt="" fill priority sizes="100vw" className="hv-kenburns -z-20 object-cover" />
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-background/80 via-background/30 to-background" aria-hidden="true" />
        <div className="absolute inset-0 -z-10 bg-gradient-to-r from-background/85 via-background/25 to-transparent" aria-hidden="true" />
        {house ? (
          <div key={house.id} className="absolute bottom-[-10%] right-[2%] -z-10 hidden aspect-square w-[min(34vw,27.5rem)] animate-in fade-in slide-in-from-bottom-6 duration-700 md:block" aria-hidden="true">
            <Image src={`/renders/${house.id}/hero.webp`} alt="" fill sizes="34vw" className="hv-fade-y object-contain drop-shadow-[0_30px_50px_rgba(10,11,23,0.7)]" />
          </div>
        ) : null}

        <div className="mx-auto max-w-[73.75rem] px-4 pb-10 pt-28 sm:px-6 md:pt-36">
          <p className="hv-eyebrow">{house ? `${house.city} · since ${house.since}` : "Market intelligence"}</p>
          <h1 className="mt-1 max-w-3xl font-display text-[clamp(3rem,7vw,5.6rem)] font-extrabold uppercase leading-[0.86] text-white">{title}</h1>
          <p className="mt-4 max-w-xl text-[0.9375rem] leading-7 text-white/80">
            Asking prices from a dated Chrono24 dataset and the live eBay Browse API. Every figure names its source and date; nothing is
            estimated where data is missing.
          </p>

          <form
            role="search"
            className="mt-8 flex max-w-2xl gap-2 rounded-full border border-white/15 bg-background/60 p-1.5 backdrop-blur-xl focus-within:border-accent/60"
            onSubmit={(e) => {
              e.preventDefault();
              go({ q: draft });
            }}
          >
            <label htmlFor="market-q" className="sr-only">
              Search by model or reference
            </label>
            <Search className="ml-3 h-5 w-5 shrink-0 self-center text-primary/50" aria-hidden="true" />
            <input
              id="market-q"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Nautilus 5711, Royal Oak 15500, Lange 1…"
              autoComplete="off"
              className="min-w-0 flex-1 bg-transparent px-2 text-[0.9375rem] text-primary placeholder:text-primary/40 focus:outline-none"
            />
            <button type="submit" className="hv-btn-pill hv-btn-primary">
              Search
            </button>
          </form>
        </div>
      </header>

      <div className="mx-auto max-w-[73.75rem] px-4 sm:px-6">
        {/* House tabs */}
        <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-2 sm:mx-0 sm:flex-wrap sm:px-0" role="group" aria-label="Filter by manufacture">
          {[null, ...STUDIO_WATCHES].map((w) => {
            const b = w?.brand ?? null;
            const active = b === (brand ?? null);
            return (
              <button
                key={b ?? "all"}
                type="button"
                aria-pressed={active}
                onClick={() => go({ brand: b, q: "" })}
                className={cn(
                  "flex min-h-[3rem] shrink-0 cursor-pointer items-center gap-2 rounded-full border py-1 pl-1 pr-4 text-[0.8125rem] font-bold transition-colors",
                  active ? "border-accent bg-accent text-ink" : "border-white/12 bg-card text-primary/80 hover:border-white/35 hover:text-primary"
                )}
              >
                <span className={cn("relative h-10 w-10 overflow-hidden rounded-full", active ? "bg-ink/10" : "hv-sky")}>
                  {w ? <Image src={`/renders/${w.id}/flat.webp`} alt="" fill sizes="40px" className="scale-[1.9] object-contain" /> : null}
                </span>
                {b ?? "All houses"}
              </button>
            );
          })}
        </div>

        <section className="mt-10" aria-live="polite">
          {effectiveQuery.length >= 2 ? (
            <MarketPanel data={query.data} loading={query.isLoading} error={query.isError} sort={sort} onSort={(s) => go({ sort: s })} />
          ) : (
            <>
              <h2 className="font-display text-[2.2rem] font-extrabold uppercase leading-none text-primary">The five houses</h2>
              <p className="mt-2 text-[0.875rem] text-primary/60">Median asking price and range in the Chrono24 dataset. Pick a house to see its market.</p>
              <div className="mt-8">
                <BrandIndexAuto />
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export default function MarketPage() {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <MarketView />
    </Suspense>
  );
}
