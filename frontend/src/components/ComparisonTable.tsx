"use client";

import { motion } from "framer-motion";
import { RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { PriceCard, type PlatformRow } from "@/components/PriceCard";
import type { ComparisonPayload } from "@/lib/api";
import { buildMockComparisonPayload, isComparisonEmpty } from "@/lib/mockComparison";
import { cn } from "@/lib/utils";

type SortKey = "price" | "delivery" | "rating";

function pickBest(rows: Array<Record<string, unknown>> | undefined) {
  if (!rows || rows.length === 0) return null;
  let best = rows[0];
  let bestPrice = Number.MAX_VALUE;
  for (const r of rows) {
    const p = r.lowest_ask;
    if (typeof p === "number" && p < bestPrice) {
      bestPrice = p;
      best = r;
    }
  }
  return best;
}

function deliveryDays(s: string | null | undefined): number | null {
  if (!s) return null;
  const m = s.match(/(\d+)\s*[-–]\s*(\d+)/);
  if (!m) return null;
  try {
    const a = Number(m[1]);
    const b = Number(m[2]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return (a + b) / 2;
  } catch {
    return null;
  }
}

function parseRating(r: string | null | undefined): number {
  if (!r || r === "N/A") return 0.84;
  if (r.includes("%")) {
    const mPct = r.match(/(\d+(\.\d+)?)/);
    if (!mPct) return 0.84;
    const pct = Number(mPct[1]);
    if (!Number.isFinite(pct)) return 0.84;
    return Math.max(0, Math.min(1, pct / 100));
  }
  const m = r.match(/(\d+(\.\d+)?)/);
  if (!m) return 0.84;
  const v = Number(m[1]);
  return Number.isFinite(v) ? Math.min(5, v) / 5 : 0.84;
}

function bestValueWinner(rows: PlatformRow[]) {
  let best: PlatformRow | null = null;
  let bestScore = -1;
  for (const r of rows) {
    if (r.price == null) continue;
    const days = deliveryDays(r.delivery);
    const invP = 1 / Math.max(r.price, 1);
    const invD = days ? 1 / Math.max(days, 1) : 0.2;
    const rating = parseRating(r.rating);
    const score = invP * 0.5 + invD * 0.3 + rating * 0.2;
    if (score > bestScore) {
      bestScore = score;
      best = r;
    }
  }
  return best?.platform ?? null;
}

function ComparisonRowSkeleton() {
  return (
    <div className="overflow-hidden rounded-2xl border border-white/[0.08]">
      <div className="hv-comparison-skeleton flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-1 items-center gap-4">
          <div className="h-10 w-10 shrink-0 rounded-lg bg-white/[0.06]" />
          <div className="flex-1 space-y-3">
            <div className="h-4 w-28 rounded bg-white/[0.06]" />
            <div className="h-8 w-40 rounded bg-white/[0.06]" />
            <div className="h-3 w-full max-w-xs rounded bg-white/[0.06]" />
          </div>
        </div>
        <div className="h-12 w-full rounded-xl bg-white/[0.06] sm:w-36" />
      </div>
    </div>
  );
}

export function ComparisonTable({
  data,
  isLoading,
  isError,
  listingId,
  queryKey,
  mockSeed,
}: {
  data?: ComparisonPayload;
  isLoading: boolean;
  isError: boolean;
  listingId?: string;
  queryKey: unknown[];
  mockSeed: string;
}) {
  const qc = useQueryClient();
  const [sort, setSort] = useState<SortKey>("price");

  const usingMock = useMemo(() => Boolean(data && isComparisonEmpty(data)), [data]);

  const effectiveData = useMemo(() => {
    if (!data) return undefined;
    if (isComparisonEmpty(data)) return buildMockComparisonPayload(mockSeed);
    return data;
  }, [data, mockSeed]);

  const rows = useMemo(() => {
    if (!effectiveData) return [];
    const sx = pickBest(effectiveData.stockx);
    const ch = pickBest(effectiveData.chrono24);
    const eb = pickBest(effectiveData.ebay);

    const base: PlatformRow[] = [
      {
        platform: "StockX",
        logo: "SX",
        price: typeof sx?.lowest_ask === "number" ? sx.lowest_ask : null,
        delivery: typeof sx?.estimated_delivery === "string" ? sx.estimated_delivery : null,
        rating: typeof sx?.seller_rating === "string" ? sx.seller_rating : "4.2",
        url: "https://stockx.com",
      },
      {
        platform: "Chrono24",
        logo: "C24",
        price: typeof ch?.lowest_ask === "number" ? ch.lowest_ask : null,
        delivery: typeof ch?.estimated_delivery === "string" ? ch.estimated_delivery : null,
        rating: typeof ch?.seller_rating === "string" ? ch.seller_rating : "4.2",
        url: "https://www.chrono24.in",
      },
      {
        platform: "eBay",
        logo: "eB",
        price: typeof eb?.lowest_ask === "number" ? eb.lowest_ask : null,
        delivery: typeof eb?.estimated_delivery === "string" ? eb.estimated_delivery : null,
        rating: typeof eb?.seller_rating === "string" ? eb.seller_rating : "4.2",
        url: "https://www.ebay.com",
      },
    ];

    const prices = base.map((b) => b.price).filter((x): x is number => x != null);
    const minPrice = prices.length ? Math.min(...prices) : null;

    const deliveries = base.map((b) => ({ b, d: deliveryDays(b.delivery) })).filter((x) => x.d != null) as Array<{
      b: PlatformRow;
      d: number;
    }>;
    const minDays = deliveries.length ? Math.min(...deliveries.map((x) => x.d)) : null;

    const winner = bestValueWinner(base);

    return base
      .map((b) => ({
        ...b,
        highlightPrice: minPrice != null && b.price != null && b.price === minPrice,
        highlightDelivery:
          minDays != null && deliveryDays(b.delivery) != null && deliveryDays(b.delivery) === minDays,
        bestValue: winner === b.platform,
      }))
      .sort((a, b) => {
        if (sort === "price") return (a.price ?? 1e9) - (b.price ?? 1e9);
        if (sort === "delivery") {
          return (deliveryDays(a.delivery) ?? 1e9) - (deliveryDays(b.delivery) ?? 1e9);
        }
        return parseRating(b.rating) - parseRating(a.rating);
      });
  }, [effectiveData, sort]);

  const freshness = useMemo(() => {
    if (!effectiveData?.scraped_at) return { label: "Unknown", tone: "text-primary/55", age: "—" };
    const ts = new Date(effectiveData.scraped_at).getTime();
    if (!Number.isFinite(ts)) return { label: "Unknown", tone: "text-primary/55", age: "—" };
    const ageMin = Math.max(0, Math.floor((Date.now() - ts) / 60000));
    const cacheRemaining = effectiveData.cache_remaining_sec ?? null;
    if (cacheRemaining != null && cacheRemaining <= 300) {
      return { label: "Refreshing soon", tone: "text-[#f59e0b]", age: `${ageMin}m ago` };
    }
    if (ageMin <= 10) return { label: "Fresh", tone: "text-[#22c55e]", age: `${ageMin}m ago` };
    if (ageMin <= 30) return { label: "Recent", tone: "text-[#86efac]", age: `${ageMin}m ago` };
    return { label: "Stale", tone: "text-[#f59e0b]", age: `${ageMin}m ago` };
  }, [effectiveData?.cache_remaining_sec, effectiveData?.scraped_at]);

  if (isLoading) {
    return (
      <div className="grid gap-4">
        <ComparisonRowSkeleton />
        <ComparisonRowSkeleton />
        <ComparisonRowSkeleton />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="rounded-2xl border border-danger/25 bg-danger/10 p-6 text-primary">
        <p className="font-extrabold text-danger">Price comparison unavailable</p>
        <p className="mt-2 text-sm text-primary/65">
          External market data could not be loaded. Retry, or check back after a few moments.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-[#888888]">Market snapshot</p>
          <p className="text-sm text-primary/65">
            Last updated:{" "}
            <span className="font-semibold text-primary">
              {effectiveData?.scraped_at ? new Date(effectiveData.scraped_at).toLocaleString() : "—"}
            </span>
            <span className={cn("ml-2 text-xs font-semibold", freshness.tone)}>
              · {freshness.label} ({freshness.age})
            </span>
            {usingMock ? (
              <span className="ml-2 text-xs font-medium text-[#888888]">· illustrative data</span>
            ) : null}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {(["price", "delivery", "rating"] as SortKey[]).map((k) => (
            <Button
              key={k}
              variant={sort === k ? "primary" : "outline"}
              className="min-h-[44px] capitalize"
              onClick={() => setSort(k)}
            >
              {k === "price" ? "Sort: price" : k === "delivery" ? "Sort: delivery" : "Sort: rating"}
            </Button>
          ))}
          <Button
            variant="ghost"
            className="min-h-[44px]"
            onClick={() => qc.invalidateQueries({ queryKey })}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      <div
        className={cn(
          "overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0a0a0a]",
          listingId ? "" : ""
        )}
      >
        {rows.map((r, idx) => (
          <motion.div
            key={r.platform}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.08, duration: 0.35, ease: "easeOut" }}
          >
            <PriceCard row={r} />
          </motion.div>
        ))}
      </div>
    </div>
  );
}
