import type { ComparisonPayload } from "@/lib/api";

/** Deterministic mock market rows when scrapers return nothing — avoids broken “—” UI. */
export function buildMockComparisonPayload(seed: string): ComparisonPayload {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const base = 420 + (h % 120);

  return {
    scraped_at: new Date().toISOString(),
    cache_ttl_sec: 30 * 60,
    cache_remaining_sec: 30 * 60,
    stockx: [
      {
        lowest_ask: base + 35,
        estimated_delivery: "4–6 days",
        seller_rating: "4.7",
      },
    ],
    chrono24: [
      {
        lowest_ask: base + 78,
        estimated_delivery: "5–8 days",
        seller_rating: "4.9",
      },
    ],
    ebay: [
      {
        lowest_ask: base - 12,
        estimated_delivery: "3–5 days",
        seller_rating: "4.2",
      },
    ],
  };
}

export function isComparisonEmpty(data: ComparisonPayload | undefined): boolean {
  if (!data) return true;
  const sx = data.stockx?.length ?? 0;
  const ch = data.chrono24?.length ?? 0;
  const eb = data.ebay?.length ?? 0;
  return sx === 0 && ch === 0 && eb === 0;
}
