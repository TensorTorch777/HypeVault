"use client";

import { ArrowUpRight } from "lucide-react";

import { type DealerGroup, type DealerItem, type DealersSection, formatDate, formatPrice } from "@/lib/market";
import { cn } from "@/lib/utils";

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

function Group({ group, compact }: { group: DealerGroup; compact: boolean }) {
  const items = compact ? group.items.slice(0, 4) : group.items;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-sm font-bold text-white">{group.label}</h4>
        <p className="text-[0.6875rem] text-primary/50">
          {group.count} match{group.count === 1 ? "" : "es"}
          {group.stats && group.count > 1 ? ` · median ${formatPrice(group.stats.median, group.currency)}` : ""}
        </p>
      </div>
      <div className={cn("grid grid-cols-2 gap-4", !compact && "md:grid-cols-3 xl:grid-cols-4")}>
        {items.map((item) => (
          <DealerCard key={`${item.source}-${item.item_url}`} item={item} />
        ))}
      </div>
    </div>
  );
}

/**
 * Dealer listings from our crawler (The 1916 Company, Chronext, ...). MarketPanel renders this below the eBay
 * and dataset listings from the same market search, so it always comes after eBay and uses the same query.
 */
export function DealerResults({ section, compact = false }: { section: DealersSection; compact?: boolean }) {
  const withMatches = section.dealers.filter((g) => g.items.length);
  return (
    <section aria-labelledby="dealers-title" className="space-y-6">
      <div>
        {withMatches.length ? <p className="hv-eyebrow">{withMatches.map((g) => g.label).join(" · ")}</p> : null}
        <h3 id="dealers-title" className="mt-2 font-display text-[2rem] font-extrabold uppercase leading-none text-primary">
          From dealers
        </h3>
        <p className="mt-2 text-xs text-primary/55">
          Dealer asking prices from our latest crawl, last {section.max_age_days} days. Prices in each dealer&rsquo;s currency.
        </p>
      </div>

      {section.status === "error" ? (
        <p className="text-sm text-danger" role="alert">
          Dealer listings are unavailable right now.
        </p>
      ) : withMatches.length ? (
        <div className="space-y-8">
          {withMatches.map((g) => (
            <Group key={g.source} group={g} compact={compact} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-primary/60">
          No dealer listings matched this watch in the latest crawl{section.searched.length ? ` (searched ${section.searched.join(", ")})` : ""}.
        </p>
      )}

      <p className="text-[0.6875rem] leading-5 text-primary/45">{section.note}</p>
    </section>
  );
}
