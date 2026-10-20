"use client";

import { useState } from "react";
import { ArrowUpRight, Database, Radio } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import {
  type HistogramBin,
  type MarketItem,
  type MarketSearch,
  type MarketSort,
  type PriceStats,
  formatDate,
  formatPrice,
} from "@/lib/market";
import { cn } from "@/lib/utils";

const SOURCE_LABEL: Record<MarketItem["source"], string> = {
  ebay: "eBay · live",
  chrono24_dataset: "Chrono24 · dataset",
};

const SORTS: { id: MarketSort; label: string }[] = [
  { id: "near_median", label: "Typical" },
  { id: "lowest", label: "Lowest" },
  { id: "highest", label: "Highest" },
];

/** Min–median–max on a log scale, so a single outlier does not flatten the bar. */
export function PriceRange({ stats, className }: { stats: PriceStats; className?: string }) {
  const lo = Math.log(Math.max(stats.min, 1));
  const hi = Math.log(Math.max(stats.max, stats.min + 1));
  const at = ((Math.log(Math.max(stats.median, 1)) - lo) / (hi - lo || 1)) * 100;
  return (
    <div className={className}>
      <div className="relative h-[0.1875rem] rounded-full bg-gradient-to-r from-neon-violet via-salmon to-neon-yellow">
        <span className="absolute -top-[0.3125rem] h-[0.8125rem] w-[0.8125rem] -translate-x-1/2 rounded-full border-2 border-background bg-white" style={{ left: `${at}%` }} />
      </div>
      <div className="mt-2 flex justify-between text-[0.6875rem] text-primary/55">
        <span>{formatPrice(stats.min, "USD", true)}</span>
        <span>{formatPrice(stats.max, "USD", true)}</span>
      </div>
    </div>
  );
}

/** Log-binned distribution of asking prices with the median marked. Hover a bar for its range. */
function Distribution({ bins, stats }: { bins: HistogramBin[]; stats: PriceStats }) {
  const [hover, setHover] = useState<number | null>(null);
  const peak = Math.max(...bins.map((b) => b.count), 1);
  const lo = Math.log(bins[0].from);
  const hi = Math.log(bins[bins.length - 1].to);
  const medianAt = ((Math.log(stats.median) - lo) / (hi - lo || 1)) * 100;
  const active = hover === null ? null : bins[hover];
  return (
    <div>
      <div className="flex items-end justify-between gap-4">
        <p className="hv-eyebrow">Where the asking prices sit</p>
        <p className="min-h-[1.125rem] text-right text-[0.75rem] text-primary/70" aria-live="polite">
          {active ? (
            <>
              <span className="font-bold text-primary">{active.count.toLocaleString("en-US")}</span> listings between {formatPrice(active.from, "USD", true)} and{" "}
              {formatPrice(active.to, "USD", true)}
            </>
          ) : (
            "Log scale · hover a bar"
          )}
        </p>
      </div>
      <div className="relative mt-5 h-40" onMouseLeave={() => setHover(null)}>
        <div className="absolute inset-0 flex items-end gap-[0.1875rem]">
          {bins.map((b, i) => (
            <button
              key={i}
              type="button"
              aria-label={`${b.count} listings between ${formatPrice(b.from)} and ${formatPrice(b.to)}`}
              onMouseEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              className="group relative h-full flex-1 cursor-default"
            >
              <span
                className={cn(
                  "absolute inset-x-0 bottom-0 rounded-t-[0.1875rem] bg-gradient-to-t from-neon-violet/70 via-salmon/80 to-neon-yellow/90 transition-opacity duration-200",
                  hover !== null && hover !== i ? "opacity-35" : "opacity-100"
                )}
                style={{ height: `${b.count ? Math.max(3, (b.count / peak) * 100) : 0}%` }}
              />
            </button>
          ))}
        </div>
        <div className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${medianAt}%` }}>
          <span className="absolute -bottom-7 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-full bg-white px-2 py-0.5 text-[0.625rem] font-bold text-ink">
            Median {formatPrice(stats.median, "USD", true)}
          </span>
        </div>
      </div>
      <div className="mt-1 flex justify-between border-t border-white/10 pt-2 text-[0.6875rem] text-primary/55">
        <span>{formatPrice(stats.min, "USD", true)}</span>
        <span>{formatPrice(stats.max, "USD", true)}</span>
      </div>
    </div>
  );
}

function ListingCard({ item }: { item: MarketItem }) {
  // The dataset truncates "1996 (Approximation)" and uses "Unknown" for blanks; tidy both.
  const year = item.year?.match(/^(\d{4})\s*\(approx/i) ? `c. ${item.year.slice(0, 4)}` : item.year;
  const meta = [item.reference ? `Ref. ${item.reference}` : null, year, item.condition].filter(
    (m): m is string => Boolean(m) && !/^unknown$/i.test(String(m).trim())
  );
  const body = (
    <>
      <div className="flex items-start justify-between gap-3">
        <p className="font-display text-[2rem] font-extrabold leading-none text-primary hv-num">{formatPrice(item.price, item.currency)}</p>
        {item.item_url ? <ArrowUpRight className="h-4 w-4 shrink-0 text-primary/40 transition-colors group-hover:text-accent" aria-hidden="true" /> : null}
      </div>
      <p className="mt-3 line-clamp-2 min-h-[2.6rem] text-[0.875rem] font-semibold leading-5 text-primary/90">{item.title}</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {meta.slice(0, 3).map((m) => (
          <span key={m} className="rounded-full bg-white/[0.07] px-2.5 py-0.5 text-[0.6875rem] font-semibold text-primary/70">
            {m}
          </span>
        ))}
      </div>
      <p className="mt-4 text-[0.6875rem] text-primary/45">
        {SOURCE_LABEL[item.source]} · seen {formatDate(item.observed_on)}
      </p>
    </>
  );
  const cls = "group block rounded-[1.125rem] border border-white/10 bg-card p-5 transition-[border-color,transform] duration-300 hover:-translate-y-0.5 hover:border-accent/40";
  return item.item_url ? (
    <a href={item.item_url} target="_blank" rel="noreferrer" className={cls}>
      {body}
    </a>
  ) : (
    <div className={cls}>{body}</div>
  );
}

export function MarketPanel({
  data,
  loading,
  error,
  sort,
  onSort,
}: {
  data: MarketSearch | undefined;
  loading: boolean;
  error: boolean;
  sort?: MarketSort;
  onSort?: (s: MarketSort) => void;
}) {
  if (loading) {
    return (
      <div className="grid gap-4">
        <Skeleton className="h-72 rounded-[1.375rem]" />
        <Skeleton className="h-56 rounded-[1.375rem]" />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div role="alert" className="hv-hud p-7">
        <p className="font-display text-[1.6rem] font-extrabold uppercase text-danger">Market data could not be loaded</p>
        <p className="mt-2 text-[0.875rem] leading-6 text-primary/70">
          Nothing is estimated in the meantime. No prices are shown rather than stale or estimated ones.
        </p>
      </div>
    );
  }

  const ebay = data.sources.ebay;
  const c24 = data.sources.chrono24_dataset;
  const items = [...ebay.items, ...c24.items];
  const stats = c24.stats ?? ebay.stats;
  const ebayStatus =
    ebay.status === "ok"
      ? { cls: "bg-success/15 text-success", text: `${ebay.stats?.count ?? 0} live` }
      : ebay.status === "not_configured"
        ? { cls: "bg-white/10 text-primary/60", text: "Not connected" }
        : { cls: "bg-review/15 text-review", text: "Unavailable" };

  return (
    <div className="space-y-5">
      {/* Price board */}
      <section className="relative overflow-hidden rounded-[1.375rem] border border-white/10 bg-card p-6 md:p-8" aria-label="Price summary">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-salmon/10 blur-3xl" aria-hidden="true" />
        {stats ? (
          <div className="relative grid gap-10 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)]">
            <div>
              <p className="hv-eyebrow">Median asking price</p>
              <p className="mt-2 font-display text-[clamp(3.6rem,8vw,6rem)] font-black leading-[0.85] text-white hv-num">{formatPrice(stats.median)}</p>
              <dl className="mt-6 grid grid-cols-3 gap-3 text-[0.75rem]">
                {[
                  ["Listings", stats.count.toLocaleString("en-US")],
                  ["Lowest", formatPrice(stats.min, "USD", true)],
                  ["Highest", formatPrice(stats.max, "USD", true)],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-primary/50">{k}</dt>
                    <dd className="mt-0.5 font-display text-[1.4rem] font-extrabold text-primary hv-num">{v}</dd>
                  </div>
                ))}
              </dl>
              <div className="mt-6 flex flex-wrap gap-2 text-[0.6875rem] font-bold">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-neon-yellow/15 px-3 py-1 text-neon-yellow">
                  <Database className="h-3.5 w-3.5" aria-hidden="true" /> Chrono24 · Snapshot {formatDate(c24.snapshot_date)}
                </span>
                <span className={cn("inline-flex items-center gap-1.5 rounded-full px-3 py-1", ebayStatus.cls)}>
                  <Radio className="h-3.5 w-3.5" aria-hidden="true" /> eBay · {ebayStatus.text}
                </span>
              </div>
            </div>
            {c24.histogram && c24.histogram.length > 1 && c24.stats ? (
              <Distribution bins={c24.histogram} stats={c24.stats} />
            ) : (
              <PriceRange stats={stats} className="self-end" />
            )}
          </div>
        ) : (
          <div className="relative">
            <p className="font-display text-[1.8rem] font-extrabold uppercase text-primary">No dataset listing matches</p>
            <p className="mt-2 max-w-lg text-[0.875rem] text-primary/65">Try a model name or reference number. No prices are shown rather than stale or estimated ones.</p>
          </div>
        )}
        <p className="relative mt-6 border-t border-white/10 pt-4 text-[0.75rem] leading-5 text-primary/50">
          Asking prices, not sale prices. The Chrono24 figures are a dated dataset of public listings, a reference level rather than today&rsquo;s price.
          {ebay.status === "not_configured" ? " Live eBay prices appear once an eBay developer key is connected. Nothing is estimated in the meantime." : ""}
        </p>
      </section>

      {/* Listings */}
      <section aria-labelledby="listings-title">
        <div className="mt-10 flex flex-wrap items-end justify-between gap-4">
          <h3 id="listings-title" className="font-display text-[2rem] font-extrabold uppercase leading-none text-primary">
            On the market
          </h3>
          {onSort ? (
            <div className="flex gap-2" role="tablist" aria-label="Sort listings">
              {SORTS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  role="tab"
                  aria-selected={sort === s.id}
                  onClick={() => onSort(s.id)}
                  className={cn(
                    "min-h-[2.25rem] cursor-pointer rounded-full border px-4 text-[0.8125rem] font-bold transition-colors",
                    sort === s.id ? "border-accent bg-accent text-ink" : "border-white/15 text-primary/70 hover:border-white/40 hover:text-primary"
                  )}
                >
                  {s.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        {items.length ? (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((it, i) => (
              <ListingCard key={`${it.source}-${i}-${it.title}`} item={it} />
            ))}
          </div>
        ) : (
          <p className="mt-5 rounded-[1.125rem] border border-dashed border-white/15 p-8 text-center text-[0.875rem] text-primary/60">No listings match this search yet.</p>
        )}
      </section>
    </div>
  );
}
