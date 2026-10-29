"use client";

import { ArrowUpRight, MapPin } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Skeleton } from "@/components/ui/skeleton";
import { type Chrono24Listing, fetchListingChrono24, formatDate, formatPrice } from "@/lib/market";

function Chrono24Card({ item }: { item: Chrono24Listing }) {
  const meta = [item.reference ? `Ref. ${item.reference}` : null, item.year, item.condition].filter(
    (m): m is string => Boolean(m) && !/^unknown$/i.test(String(m).trim())
  );
  return (
    <a
      href={item.listing_url}
      target="_blank"
      rel="noopener noreferrer nofollow"
      className="group flex flex-col overflow-hidden rounded-[1.125rem] border border-white/10 bg-white/[0.03] transition-colors hover:border-white/25"
    >
      <div className="relative aspect-square w-full bg-white/[0.04]">
        {item.image_url ? (
          // Chrono24 serves the photos; next/image would need its host whitelisted.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={item.image_url} alt={item.title} loading="lazy" referrerPolicy="no-referrer" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-primary/40">No photo</div>
        )}
        {item.image_urls.length > 1 ? (
          <span className="absolute bottom-2 right-2 rounded-full bg-black/60 px-2 py-0.5 text-[0.625rem] font-semibold text-white">
            {item.image_urls.length} photos
          </span>
        ) : null}
      </div>
      <div className="flex flex-1 flex-col p-4">
        <div className="flex items-start justify-between gap-3">
          <p className="font-display text-[1.5rem] font-extrabold leading-none text-primary hv-num">
            {item.price ? formatPrice(item.price, item.currency ?? "USD") : "Price on request"}
          </p>
          <ArrowUpRight className="h-4 w-4 shrink-0 text-primary/40 transition-colors group-hover:text-accent" aria-hidden="true" />
        </div>
        <p className="mt-3 line-clamp-2 min-h-[2.5rem] text-[0.8125rem] font-semibold leading-5 text-primary/90">{item.title}</p>
        {meta.length ? (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {meta.slice(0, 3).map((m) => (
              <span key={m} className="rounded-full bg-white/[0.07] px-2.5 py-0.5 text-[0.6875rem] font-semibold text-primary/70">
                {m}
              </span>
            ))}
          </div>
        ) : null}
        <div className="mt-auto flex items-center justify-between gap-2 pt-4 text-[0.6875rem] text-primary/50">
          <span className="flex min-w-0 items-center gap-1 truncate">
            {item.location ? (
              <>
                <MapPin className="h-3 w-3 shrink-0" aria-hidden="true" />
                {item.location}
              </>
            ) : null}
          </span>
          <span className="shrink-0">{item.seller_name ?? item.seller_type ?? ""}</span>
        </div>
      </div>
    </a>
  );
}

/** Chrono24 listings for a screened (live) HypeVault listing. */
export function Chrono24Results({ listingId, limit = 12, title = "On Chrono24 now" }: { listingId: string; limit?: number; title?: string }) {
  const q = useQuery({
    queryKey: ["chrono24", listingId, limit],
    queryFn: () => fetchListingChrono24(listingId, limit),
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });

  return (
    <section aria-labelledby={`c24-${listingId}`} className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="hv-eyebrow">Chrono24 {q.data?.provider === "dataset" ? "· snapshot" : "· live"}</p>
          <h3 id={`c24-${listingId}`} className="mt-2 font-display text-[1.6rem] font-extrabold uppercase leading-none text-white">
            {title}
          </h3>
          {q.data ? (
            <p className="mt-2 text-xs text-primary/55">
              {q.data.count} match{q.data.count === 1 ? "" : "es"} for “{q.data.query}”
              {q.data.price_range
                ? ` · ${formatPrice(q.data.price_range.min, q.data.price_range.currency ?? "USD", true)}–${formatPrice(q.data.price_range.max, q.data.price_range.currency ?? "USD", true)}`
                : ""}
              {` · ${formatDate(q.data.fetched_at)}`}
            </p>
          ) : null}
        </div>
      </div>

      {q.isLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4" aria-busy="true">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="aspect-[3/4] rounded-[1.125rem]" />
          ))}
          <p className="col-span-full text-xs text-primary/55" role="status">Searching Chrono24 for this watch…</p>
        </div>
      ) : q.isError ? (
        <p className="text-sm text-danger" role="alert">Chrono24 search is unavailable right now.</p>
      ) : q.data && q.data.items.length ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4">
          {q.data.items.map((item) => (
            <Chrono24Card key={item.listing_id} item={item} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-primary/60">No Chrono24 listings matched this watch.</p>
      )}

      {q.data?.note ? <p className="text-[0.6875rem] leading-5 text-primary/45">{q.data.note} Matches say nothing about whether this watch is genuine.</p> : null}
    </section>
  );
}
