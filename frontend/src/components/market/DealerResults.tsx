"use client";

import { ArrowUpRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Skeleton } from "@/components/ui/skeleton";
import { type DealerItem, type DealerSection, fetchListingDealers, formatDate, formatPrice } from "@/lib/market";

function boxPapers(item: DealerItem): string | null {
  if (item.box && item.papers) return "Box & papers";
  if (item.box) return "Box only";
  if (item.papers) return "Papers only";
  if (item.box === false && item.papers === false) return "Watch only";
  return null;
}

function DealerCard({ item }: { item: DealerItem }) {
  const meta = [item.reference ? `Ref. ${item.reference}` : null, item.year, item.condition, boxPapers(item)].filter(
    (m): m is string => Boolean(m)
  );
  return (
    <a
      href={item.item_url ?? "#"}
      target="_blank"
      rel="noopener noreferrer nofollow"
      className="group flex flex-col overflow-hidden rounded-[1.125rem] border border-white/10 bg-white/[0.03] transition-colors hover:border-white/25"
    >
      <div className="aspect-square w-full bg-white/[0.04]">
        {item.image_url ? (
          // Dealer CDNs serve the photos; next/image would need each host whitelisted.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={item.image_url} alt={item.title} loading="lazy" referrerPolicy="no-referrer" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-primary/40">No photo</div>
        )}
      </div>
      <div className="flex flex-1 flex-col p-4">
        <div className="flex items-start justify-between gap-3">
          <p className="font-display text-[1.375rem] font-extrabold leading-none text-primary hv-num">{formatPrice(item.price, item.currency)}</p>
          <ArrowUpRight className="h-4 w-4 shrink-0 text-primary/40 transition-colors group-hover:text-accent" aria-hidden="true" />
        </div>
        <p className="mt-3 line-clamp-2 min-h-[2.5rem] text-[0.8125rem] font-semibold leading-5 text-primary/90">{item.title}</p>
        {meta.length ? (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {meta.slice(0, 4).map((m) => (
              <span key={m} className="rounded-full bg-white/[0.07] px-2.5 py-0.5 text-[0.6875rem] font-semibold text-primary/70">
                {m}
              </span>
            ))}
          </div>
        ) : null}
        <p className="mt-auto pt-4 text-[0.6875rem] text-primary/45">Seen {formatDate(item.observed_on)}</p>
      </div>
    </a>
  );
}

function Section({ section }: { section: DealerSection }) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-sm font-bold text-white">{section.label}</h4>
        <p className="text-[0.6875rem] text-primary/50">
          {section.items.length
            ? `${section.count} match${section.count === 1 ? "" : "es"}${section.broadened ? " in this collection" : ""}`
            : "No current listings"}
        </p>
      </div>
      {section.items.length ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4">
          {section.items.map((item) => (
            <DealerCard key={`${item.source}-${item.item_url}`} item={item} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Listings for this watch from our crawled dealer sources (The 1916 Company, Chronext, and others). */
export function DealerResults({ listingId, limit = 8, title = "From dealers" }: { listingId: string; limit?: number; title?: string }) {
  const q = useQuery({
    queryKey: ["dealers", listingId, limit],
    queryFn: () => fetchListingDealers(listingId, limit),
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });

  const sourceNames = q.data?.sections.map((s) => s.label).join(" · ") || "Dealers";

  return (
    <section aria-labelledby={`dealers-${listingId}`} className="space-y-6">
      <div>
        <p className="hv-eyebrow">{sourceNames}</p>
        <h3 id={`dealers-${listingId}`} className="mt-2 font-display text-[1.6rem] font-extrabold uppercase leading-none text-white">
          {title}
        </h3>
        {q.data ? <p className="mt-2 text-xs text-primary/55">Matches for “{q.data.query}”, last {q.data.max_age_days} days</p> : null}
      </div>

      {q.isLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4" aria-busy="true">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="aspect-[3/4] rounded-[1.125rem]" />
          ))}
        </div>
      ) : q.isError ? (
        <p className="text-sm text-danger" role="alert">Dealer listings are unavailable right now.</p>
      ) : q.data && q.data.total ? (
        <div className="space-y-8">
          {q.data.sections
            .filter((s) => s.items.length)
            .map((s) => (
              <Section key={s.source} section={s} />
            ))}
        </div>
      ) : (
        <p className="text-sm text-primary/60">No dealer listings matched this watch in the latest crawl.</p>
      )}

      {q.data?.note ? <p className="text-[0.6875rem] leading-5 text-primary/45">{q.data.note}</p> : null}
    </section>
  );
}
