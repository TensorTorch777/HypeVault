import axios from "axios";

import { clearTokens, getRefreshToken } from "./auth";

const baseURL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL, withCredentials: true });

/** FastAPI `detail`: string or validation error list; also connection/API-down hints */
export function getApiErrorMessage(err: unknown, fallback: string): string {
  if (!axios.isAxiosError(err)) return fallback;

  if (!err.response) {
    const code = err.code;
    if (code === "ERR_NETWORK" || err.message === "Network Error" || code === "ECONNREFUSED") {
      return `Cannot reach API at ${baseURL}. Start it from the repo: cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000 (and ensure Postgres + Redis are running).`;
    }
    return fallback;
  }

  const d = err.response.data as { detail?: unknown } | undefined;
  if (!d) {
    if (err.response.status === 401) return "Invalid credentials.";
    return fallback;
  }
  const detail = d.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: string } | undefined;
    if (first && typeof first.msg === "string") return first.msg;
  }
  if (err.response.status === 401) return "Invalid credentials.";
  return fallback;
}

export type AuthUser = {
  id: string;
  email: string;
  role: "buyer" | "seller";
};

export async function fetchMe(): Promise<AuthUser> {
  const { data } = await api.get<AuthUser>("/auth/me");
  return data;
}

export async function googleLogin(idToken: string, role: "buyer" | "seller" = "buyer"): Promise<void> {
  await api.post<{ access_token: string; refresh_token: string }>("/auth/google", {
    id_token: idToken,
    role,
  });
}

export async function logoutRemote(): Promise<void> {
  const refresh = typeof window !== "undefined" ? getRefreshToken() : null;
  try {
    await api.post("/auth/logout", refresh ? { refresh_token: refresh } : {});
  } catch {
    /* still clear local session */
  }
  clearTokens();
}

let isRefreshing = false;
let refreshWaiters: Array<(ok: boolean) => void> = [];
let lastRefreshFailureAt = 0;

function resolveRefreshWaiters(ok: boolean) {
  refreshWaiters.forEach((fn) => fn(ok));
  refreshWaiters = [];
}

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const now = Date.now();
    const statusCode = err.response?.status;
    const requestUrl = String(err.config?.url ?? "");
    const cfg = err.config as (typeof err.config & { _retry?: boolean }) | undefined;
    const isAuthMeRequest = requestUrl.includes("/auth/me");
    const refreshCooldownActive = now - lastRefreshFailureAt < 5000;
    const shouldAttemptRefresh =
      typeof window !== "undefined" &&
      statusCode === 401 &&
      !cfg?._retry &&
      !isAuthMeRequest &&
      !refreshCooldownActive &&
      !requestUrl.includes("/auth/login") &&
      !requestUrl.includes("/auth/google") &&
      !requestUrl.includes("/auth/refresh") &&
      !requestUrl.includes("/auth/logout");

    if (shouldAttemptRefresh && cfg) {
      cfg._retry = true;
      if (isRefreshing) {
        const ok = await new Promise<boolean>((resolve) => refreshWaiters.push(resolve));
        if (!ok) return Promise.reject(err);
        return api(cfg);
      }
      isRefreshing = true;
      try {
        await api.post("/auth/refresh", {});
        resolveRefreshWaiters(true);
        return api(cfg);
      } catch {
        resolveRefreshWaiters(false);
        lastRefreshFailureAt = Date.now();
        clearTokens();
        if (typeof window !== "undefined") window.location.href = "/login";
      } finally {
        isRefreshing = false;
      }
    }
    if (statusCode === 401 && typeof window !== "undefined") {
      clearTokens();
    }
    return Promise.reject(err);
  }
);

export type Listing = {
  id: string;
  seller_id: string;
  product_name: string;
  category: string;
  brand: string | null;
  condition: string | null;
  size: string | null;
  s3_url: string | null;
  verdict: string | null;
  confidence: number | null;
  status: string;
  created_at: string;
};

export async function fetchRecentListings(
  limit = 6,
  opts?: { category?: "sneaker" | "watch"; brand?: string }
): Promise<Listing[]> {
  try {
    const params: Record<string, string | number> = { limit };
    if (opts?.category) params.category = opts.category;
    if (opts?.brand?.trim()) params.brand = opts.brand.trim();
    const { data } = await api.get<Listing[]>(`/listings/recent`, { params });
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

export async function fetchListing(id: string): Promise<Listing> {
  const { data } = await api.get<Listing>(`/listings/${id}`);
  return data;
}

export type ComparisonPayload = {
  stockx: Array<Record<string, unknown>>;
  chrono24: Array<Record<string, unknown>>;
  ebay: Array<Record<string, unknown>>;
  scraped_at: string;
  cache_ttl_sec?: number;
  cache_remaining_sec?: number;
  error?: string;
};

export async function fetchComparison(listingId: string): Promise<ComparisonPayload> {
  try {
    const { data } = await api.get<ComparisonPayload>(`/listings/${listingId}/comparison`);
    return data;
  } catch {
    return {
      stockx: [],
      chrono24: [],
      ebay: [],
      scraped_at: new Date().toISOString(),
      cache_ttl_sec: 30 * 60,
      cache_remaining_sec: 0,
      error: "unavailable",
    };
  }
}

export async function fetchComparisonByQuery(productName: string): Promise<ComparisonPayload> {
  try {
    const { data } = await api.get<ComparisonPayload>(`/listings/compare`, {
      params: { q: productName },
    });
    return data;
  } catch {
    return {
      stockx: [],
      chrono24: [],
      ebay: [],
      scraped_at: new Date().toISOString(),
      cache_ttl_sec: 30 * 60,
      cache_remaining_sec: 0,
      error: "unavailable",
    };
  }
}
