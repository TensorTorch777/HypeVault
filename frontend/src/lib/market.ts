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
  /** The query that produced these results; differs from `query` when the shorter fallback was used. */
  query_used?: string;
  tried?: string[];
  brand: string | null;
  generated_at: string;
  currency: string;
  /** true: has listings · false: searched, none · null: eBay not connected or failed */
  found?: { ebay: boolean | null; chrono24_dataset: boolean };
  sources: { ebay: EbaySection; chrono24_dataset: Chrono24Section };
};

export type IdentifyCandidate = {
  brand: string | null;
  model: string | null;
  reference: string | null;
  label: string;
  query: string;
  short_query: string;
  /** Share of the clue points (0–1); how strongly the clues agree, not a probability. Null for free text. */
  confidence: number | null;
  evidence: string[];
};

export type IdentifyResult = {
  status: "recognised" | "unsure" | "free_text" | "not_recognised" | "not_a_watch" | "unsupported_brand";
  candidates: IdentifyCandidate[];
  looks_like?: string | null;
  other_brand?: string;
  dial_text: string[];
  lookalikes: Array<{ title: string; similarity: number; item_url: string | null }>;
  /** eBay's own visual search over live listings (five brands only), best match first. */
  ebay_photo_matches?: EbayPhotoMatch[];
  used?: { photo: boolean; text: boolean; dial_text: boolean; lookalikes: boolean; ebay_photo_search?: boolean };
  gallery?: { ready: boolean; photos: number; built_on: string | null };
  elapsed_ms: number;
};

export type EbayPhotoMatch = {
  title: string;
  brand: string;
  price: number;
  currency: string;
  condition: string | null;
  item_url: string | null;
  image_url: string | null;
};

/** Name a watch from a photo, typed text, or both. */
export async function identifyWatch(photo: File | null, text: string): Promise<IdentifyResult> {
  const body = new FormData();
  if (photo) body.append("image", photo);
  if (text.trim()) body.append("q", text.trim());
  const { data } = await api.post<IdentifyResult>("/market/identify", body);
  return data;
}

/** A Chrono24 search page for the name. Chrono24 is linked to, never scraped. */
export function chrono24SearchUrl(query: string, brand?: string | null): string {
  const q = [brand, query].filter(Boolean).join(" ");
  return `https://www.chrono24.com/search/index.htm?query=${encodeURIComponent(q)}`;
}

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
  opts?: { sort?: MarketSort; limit?: number; fallback?: string | null }
): Promise<MarketSearch> {
  const { data } = await api.get<MarketSearch>("/market/search", {
    params: {
      q,
      ...(brand ? { brand } : {}),
      ...(opts?.sort ? { sort: opts.sort } : {}),
      ...(opts?.limit ? { limit: opts.limit } : {}),
      ...(opts?.fallback ? { fallback: opts.fallback } : {}),
    },
  });
  return data;
}

export async function fetchListingMarket(listingId: string): Promise<MarketSearch> {
  const { data } = await api.get<MarketSearch>(`/listings/${listingId}/comparison`);
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

export type KN = { k: number; n: number } | null;

export type LiveModelEvidence = {
  id: string;
  name: string;
  phase: number | null;
  features: string;
  benchmark: {
    name: string | null;
    written_on: string | null;
    scored_at: string | null;
    question: string | null;
    rule: string | null;
    met: boolean | null;
    systems: Record<string, string>;
    groups: { id: string; label: string; kind: "real" | "generated"; systems: Record<string, KN> }[];
    caveat: string;
  };
  unseen_generators: null | { name: string | null; caught: KN; real_ebay_flagged: KN; generators: string; note: string };
  edited_copy_hold: null | {
    rows: { edit: string; phase76: string | null; phase78: string | null }[];
    real_flagged: { phase76: string; phase78: string } | null;
    role: string;
  };
  known_ai: null | { cut: number | null; edited_matched: number | null; real_matched: number | null; real_n: number | null };
};

export type BenchmarkSummary = {
  live_model: LiveModelEvidence | null;
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
