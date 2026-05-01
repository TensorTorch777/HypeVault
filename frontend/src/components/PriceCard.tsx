"use client";

import { Clock, ExternalLink } from "lucide-react";

import { cn } from "@/lib/utils";

export type PlatformRow = {
  platform: "StockX" | "Chrono24" | "eBay";
  logo: string;
  price: number | null;
  delivery: string | null;
  rating: string | null;
  url?: string;
  highlightPrice?: boolean;
  highlightDelivery?: boolean;
  bestValue?: boolean;
};

const platformStyles = {
  StockX: {
    badge: "bg-[#00C851] text-black",
    buy: "bg-[#00C851] text-black hover:brightness-110",
  },
  Chrono24: {
    badge: "bg-[#007AFF] text-white",
    buy: "bg-[#007AFF] text-white hover:brightness-110",
  },
  eBay: {
    badge: "bg-[#FFCC00] text-black",
    buy: "bg-[#FFCC00] text-black hover:brightness-110",
  },
} as const;

function parseRating(r: string | null | undefined): number {
  if (r == null || r === "" || r === "N/A") return 4.2;
  const m = String(r).match(/(\d+(\.\d+)?)/);
  if (!m) return 4.2;
  const v = Number(m[1]);
  return Number.isFinite(v) ? Math.min(5, Math.max(0, v)) : 4.2;
}

function StarRow({ value }: { value: number }) {
  const full = Math.min(5, Math.floor(value + 1e-6));
  const empty = Math.max(0, 5 - full);
  return (
    <span className="text-[#fbbf24] tabular-nums" aria-hidden>
      {"★".repeat(full)}
      <span className="text-white/25">{"☆".repeat(empty)}</span>
    </span>
  );
}

export function PriceCard({ row }: { row: PlatformRow }) {
  const styles = platformStyles[row.platform];
  const ratingVal = parseRating(row.rating);
  const priceStr = row.price != null ? `$${row.price.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : null;

  return (
    <div
      className={cn(
        "group relative border-b border-white/[0.06] transition-colors duration-200 hover:bg-white/[0.04]",
        row.bestValue && "border-l-4 border-l-[#FF3B00] pl-3 sm:pl-4"
      )}
    >
      {row.bestValue ? (
        <span className="absolute right-3 top-3 z-[1] rounded bg-[#FF3B00]/20 px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-[0.1em] text-[#FF3B00]">
          Best value
        </span>
      ) : null}

      <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6 sm:py-5 sm:pr-4">
        <div className="flex min-w-0 flex-1 items-start gap-4">
          <div
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-xs font-black",
              styles.badge
            )}
            aria-hidden
          >
            {row.logo}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-base font-extrabold text-white">{row.platform}</p>
              {row.highlightPrice ? (
                <span className="rounded bg-[#00C851]/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-[#4ade80]">
                  Lowest
                </span>
              ) : null}
            </div>
            <p
              className="mt-2 text-[28px] font-bold leading-none tracking-tight text-white tabular-nums"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {priceStr ?? "—"}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-[#a1a1aa]">
              <span className="inline-flex items-center gap-1.5">
                <Clock className="h-4 w-4 shrink-0 text-white/45" aria-hidden />
                <span className="text-white/80">{row.delivery ?? "—"}</span>
              </span>
              <span className="inline-flex items-center gap-2">
                <StarRow value={ratingVal} />
                <span className="text-xs tabular-nums text-[#888888]">{ratingVal.toFixed(1)}</span>
              </span>
            </div>
          </div>
        </div>

        <button
          type="button"
          className={cn(
            "inline-flex min-h-[48px] shrink-0 items-center justify-center gap-2 rounded-xl px-6 text-sm font-bold transition-all duration-150",
            "hover:scale-[1.02] active:scale-[0.98] active:translate-y-0",
            "hover:-translate-y-px",
            styles.buy
          )}
          onClick={() => {
            if (row.url) window.open(row.url, "_blank", "noopener,noreferrer");
          }}
        >
          Buy now
          <ExternalLink className="h-4 w-4 opacity-90" />
        </button>
      </div>
    </div>
  );
}
