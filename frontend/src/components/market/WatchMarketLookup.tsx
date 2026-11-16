"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Loader2 } from "lucide-react";

import { MarketPanel } from "@/components/market/MarketPanel";
import { EbayPhotoMatches, FoundBanner } from "@/components/market/PhotoIdentify";
import { type IdentifyCandidate, type MarketSort, identifyWatch, searchMarket } from "@/lib/market";
import { cn } from "@/lib/utils";

function marketHref(c: IdentifyCandidate): string {
  const sp = new URLSearchParams({ q: c.query });
  if (c.brand) sp.set("brand", c.brand);
  if (c.short_query && c.short_query !== c.query) sp.set("alt", c.short_query);
  return `/market?${sp}`;
}

/**
 * After a photo has been screened: recognise the watch in it, search the top guess on eBay and the
 * Chrono24 dataset straight away, and show the market (where it was found, median price, listings).
 * The other guesses stay one click away in case the top one is wrong.
 */
export function WatchMarketLookup({ photo, compact = false, className }: { photo: File; compact?: boolean; className?: string }) {
  const photoKey = `${photo.name}:${photo.size}:${photo.lastModified}`;
  const identify = useQuery({
    queryKey: ["identify-photo", photoKey],
    queryFn: () => identifyWatch(photo, ""),
    staleTime: Infinity,
    retry: false,
  });
  const [chosen, setChosen] = useState<IdentifyCandidate | null>(null);
  const [sort, setSort] = useState<MarketSort>("near_median");
  useEffect(() => setChosen(null), [photoKey]);

  const candidates = identify.data?.candidates ?? [];
  const active = chosen ?? candidates[0] ?? null;
  const market = useQuery({
    queryKey: ["market-search", active?.query, active?.brand, sort, active?.short_query],
    queryFn: () => searchMarket(active!.query, active!.brand, { sort, limit: 12, fallback: active!.short_query }),
    enabled: Boolean(active),
    placeholderData: (prev) => prev,
  });

  const status = identify.data?.status;
  const message =
    status === "not_a_watch"
      ? `The photo looks like ${identify.data?.looks_like ?? "something other than a watch"}, so there is no market to show.`
      : status === "unsupported_brand"
        ? `It looks like a ${identify.data?.other_brand ?? "different"} watch. The market covers A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille and Vacheron Constantin only.`
        : status === "not_recognised"
          ? "We couldn't recognise this watch from the photo. Search the market by name instead."
          : null;

  return (
    <section className={cn("rounded-[1.375rem] border border-white/10 bg-card/60 p-5 md:p-7", className)} aria-labelledby="lookup-title" aria-live="polite">
      <p className="hv-eyebrow">Market lookup · from this photo</p>
      <h3 id="lookup-title" className="mt-1 font-display text-[2rem] font-extrabold uppercase leading-none text-primary">
        What this watch sells for
      </h3>

      {identify.isLoading ? (
        <p className="mt-4 flex items-center gap-2 text-[0.875rem] text-primary/70" role="status">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Recognising the watch in your photo…
        </p>
      ) : identify.isError ? (
        <p className="mt-4 text-[0.875rem] font-semibold text-danger" role="alert">
          The watch could not be recognised right now. You can still{" "}
          <Link href="/market" className="underline underline-offset-4">
            search the market by name
          </Link>
          .
        </p>
      ) : message || !active ? (
        <p className="mt-4 text-[0.875rem] leading-6 text-primary/75">
          {message ?? "No name could be found for this photo."}{" "}
          <Link href="/market" className="font-bold underline underline-offset-4 hover:text-accent">
            Open the market
          </Link>
        </p>
      ) : (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
            <p className="text-[0.9375rem] text-primary/75">
              {status === "unsure" ? "Best guess" : "Recognised as"}{" "}
              <span className="font-display text-[1.35rem] font-extrabold text-primary">{active.label}</span>
              {active.confidence != null ? <span className="ml-2 text-[0.75rem] font-bold text-primary/50">{Math.round(active.confidence * 100)}%</span> : null}
            </p>
            <Link href={marketHref(active)} className="inline-flex items-center gap-1 text-[0.75rem] font-bold text-primary/70 underline-offset-4 hover:text-accent hover:underline">
              Open in Market <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
            </Link>
          </div>
          {active.evidence.length ? <p className="mt-1 text-[0.75rem] text-primary/55">{active.evidence.slice(0, 2).join(" · ")}</p> : null}
          {candidates.length > 1 ? (
            <div className="mt-3 flex flex-wrap items-center gap-2" role="group" aria-label="Other possible names">
              <span className="text-[0.75rem] text-primary/55">Not right? Try:</span>
              {candidates
                .filter((c) => c.label !== active.label)
                .map((c) => (
                  <button
                    key={c.label}
                    type="button"
                    onClick={() => setChosen(c)}
                    className="min-h-[2.25rem] cursor-pointer rounded-full border border-white/15 px-3 text-[0.75rem] font-bold text-primary/75 transition-colors hover:border-accent/60 hover:text-primary"
                  >
                    {c.label}
                  </button>
                ))}
            </div>
          ) : null}

          <EbayPhotoMatches matches={identify.data?.ebay_photo_matches} className="mt-5" />

          <div className="mt-6">
            {market.data && !market.isLoading ? <FoundBanner data={market.data} /> : null}
            <MarketPanel data={market.data} loading={market.isLoading} error={market.isError} sort={sort} onSort={setSort} compact={compact} />
          </div>
        </>
      )}
    </section>
  );
}
