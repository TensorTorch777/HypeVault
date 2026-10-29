import { api } from "@/lib/api";

export const SUPPORTED_BRANDS = [
  "A. Lange & Söhne",
  "Audemars Piguet",
  "Patek Philippe",
  "Richard Mille",
  "Vacheron Constantin",
] as const;

export type MarketItem = {
  source: "ebay" | "chrono24_dataset";
  title: string;
  brand: string;
  model: string | null;
  reference: string | null;
  price: number;
  currency: string;
  condition: string | null;
  year: string | null;
  seller_feedback_pct: number | null;
  item_url: string | null;
  image_url: string | null;
  observed_on: string;
  observed_at: string | null;
};

export type PriceStats = { count: number; min: number; median: number; max: number };

export type EbaySection = {
  status: "ok" | "not_configured" | "error";
  fetched_at: string | null;
  total_reported?: number;
  stats: PriceStats | null;
  items: MarketItem[];
  cached: boolean;
};

export type HistogramBin = { from: number; to: number; count: number };
export type MarketSort = "near_median" | "lowest" | "highest";

export type Chrono24Section = {
  status: "ok" | "no_match";
  snapshot_date: string | null;
  note: string;
  stats: PriceStats | null;
  sort?: MarketSort;
  histogram?: HistogramBin[];
  items: MarketItem[];
};

export type MarketSearch = {
  query: string;
  brand: string | null;
  generated_at: string;
  currency: string;
  sources: { ebay: EbaySection; chrono24_dataset: Chrono24Section };
};

export type MarketHistory = {
  query: string;
  brand: string | null;
  currency: string;
  ebay_daily_median: Array<{ date: string; median: number; n: number }>;
  chrono24_reference: (PriceStats & { snapshot_date: string | null }) | null;
};

export type MarketStatus = {
  ebay: { configured: boolean; environment: string };
  sources: Record<string, { rows: number; first_observed_on: string | null; last_observed_on: string | null; last_observed_at: string | null }>;
};

export async function searchMarket(
  q: string,
  brand?: string | null,
  opts?: { sort?: MarketSort; limit?: number }
): Promise<MarketSearch> {
  const { data } = await api.get<MarketSearch>("/market/search", {
    params: { q, ...(brand ? { brand } : {}), ...(opts?.sort ? { sort: opts.sort } : {}), ...(opts?.limit ? { limit: opts.limit } : {}) },
  });
  return data;
}

export async function fetchListingMarket(listingId: string): Promise<MarketSearch> {
  const { data } = await api.get<MarketSearch>(`/listings/${listingId}/comparison`);
  return data;
}

export type Chrono24Listing = {
  listing_id: string;
  title: string;
  listing_url: string;
  price: number | null;
  currency: string | null;
  brand: string | null;
  model: string | null;
  reference: string | null;
  condition: string | null;
  year: string | null;
  location: string | null;
  seller_type: string | null;
  seller_name: string | null;
  seller_rating: string | null;
  image_url: string | null;
  image_urls: string[];
};

export type Chrono24Search = {
  status: "ok" | "no_match";
  provider: "apify" | "playwright" | "dataset";
  query: string;
  brand: string | null;
  fetched_at: string;
  count: number;
  price_range: { min: number; max: number; currency: string | null } | null;
  items: Chrono24Listing[];
  note?: string;
  cached?: boolean;
};

/** Chrono24 listings for a live (screened) listing. */
export async function fetchListingChrono24(listingId: string, limit = 12): Promise<Chrono24Search> {
  const { data } = await api.get<Chrono24Search>(`/listings/${listingId}/chrono24`, { params: { limit } });
  return data;
}

/** Chrono24 listings for free text, e.g. "Patek Philippe Nautilus 5711". */
export async function searchChrono24(q: string, brand?: string | null, limit = 12): Promise<Chrono24Search> {
  const { data } = await api.get<Chrono24Search>("/market/chrono24", { params: { q, limit, ...(brand ? { brand } : {}) } });
  return data;
}

export async function fetchMarketHistory(q: string, brand?: string | null): Promise<MarketHistory> {
  const { data } = await api.get<MarketHistory>("/market/history", { params: { q, ...(brand ? { brand } : {}) } });
  return data;
}

export async function fetchMarketStatus(): Promise<MarketStatus> {
  const { data } = await api.get<MarketStatus>("/market/status");
  return data;
}

export type BenchmarkModelRow = {
  model: string;
  coverage: number | null;
  accuracy: number | null;
  balanced_accuracy: number | null;
  false_authentic: { k: number; n: number; rate: number | null; upper95: number } | null;
  false_fake: { k: number; n: number; rate: number | null; upper95: number } | null;
  roc_auc: number | null;
  review: { generated: number; real: number } | null;
};

export type BenchmarkSummary = {
  protocol: {
    name: string | null;
    status: string | null;
    written_on: string | null;
    question: string | null;
    scope_limit: string | null;
    classes: { REAL_PHOTO: string | null; GENERATED: Record<string, string> };
    normalisation: string[] | null;
    sample_size: Record<string, unknown> | null;
    frontier_rule: Record<string, string> | null;
    models: Record<string, { checkpoint_sha256: string; policy: string }>;
  };
  confirmatory: null | {
    run_id: string;
    frontier: { outcome: string; frontier_model: string | null; statement: string };
    primary: null | {
      set: string;
      n: number;
      models: BenchmarkModelRow[];
      paired: { n_both_binary: number; discordant: number; dinov3_only_correct: number; dinov2_only_correct: number; mcnemar_p: number };
    };
  };
  pipeline_check: null | { run_id: string; passed: boolean; images_scored: number; note: string };
  previous_evaluation: null | { name: string; data_readiness: string; summary: string };
};

export async function fetchBenchmarkSummary(): Promise<BenchmarkSummary> {
  const { data } = await api.get<BenchmarkSummary>("/benchmark/summary");
  return data;
}

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const compactUsd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });

export function formatPrice(value: number | null | undefined, currency = "USD", compact = false): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (currency !== "USD") return `${currency} ${Math.round(value).toLocaleString("en-US")}`;
  return compact && Math.abs(value) >= 1000 ? compactUsd.format(value) : usd.format(value);
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}
