import axios, { AxiosHeaders } from "axios";

import { coerceLoopbackHttpsToHttp } from "../../loopbackHttp.mjs";
import { clearTokens, getRefreshToken } from "./auth";

const configuredApiUrl = coerceLoopbackHttpsToHttp(process.env.NEXT_PUBLIC_API_URL?.trim() ?? "");

/** Browser tab is clearly local dev (same machine). */
function isLocalDevPage(): boolean {
  if (typeof window === "undefined") return false;
  const h = window.location.hostname.toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "[::1]" || h === "::1";
}

/**
 * Non-localhost: prefer `/api/upstream` unless NEXT_PUBLIC_API_URL is a *different* origin (e.g. second API tunnel).
 * Same host as the page (single ngrok → Next) still uses the BFF so redirects/cookies stay consistent.
 */
function needsSameOriginProxy(): boolean {
  if (typeof window === "undefined") return false;
  if (isLocalDevPage()) return false;
  if (!configuredApiUrl) return true;
  if (pointsAtLocalLoopback(configuredApiUrl)) return true;
  try {
    const cfgOrigin = new URL(configuredApiUrl).origin;
    if (cfgOrigin === new URL(window.location.href).origin) return true;
  } catch {
    return true;
  }
  return false;
}

/** True when the URL clearly targets this machine’s loopback (useless from a tunnel or phone). */
function pointsAtLocalLoopback(url: string): boolean {
  try {
    const u = new URL(url);
    const h = u.hostname.toLowerCase();
    return h === "localhost" || h === "127.0.0.1" || h === "::1";
  } catch {
    return /localhost|127\.0\.0\.1/i.test(url);
  }
}

function resolveBaseURL(): string {
  if (typeof window === "undefined") {
    return configuredApiUrl.length > 0 ? configuredApiUrl : "http://127.0.0.1:8000";
  }
  /* Never attach loopback baseURL on a non-local page — Firefox may force https://127.0.0.1:8000 and fail. */
  if (!isLocalDevPage()) {
    return "";
  }
  if (configuredApiUrl.length > 0) return configuredApiUrl;
  return "";
}

const baseURL = resolveBaseURL();

export const api = axios.create({
  baseURL,
  withCredentials: true,
  /* Torch first load + CPU inference can exceed 2m through tunnel; keep aligned with upstream proxy. */
  timeout: 300_000,
});

/**
 * Non-localhost: same-origin `/api/upstream` BFF → FastAPI (avoids loopback + bad redirects from the browser).
 */
api.interceptors.request.use((config) => {
  if (typeof window === "undefined") return config;

  const h = AxiosHeaders.from(config.headers);
  if (needsSameOriginProxy()) {
    const u = config.url ?? "";
    if (u.startsWith("/") && !u.startsWith("/api/upstream")) {
      config.url = `/api/upstream${u}`;
    }
    config.baseURL = "";
    /* Avoid free-ngrok HTML interstitial; harmless on other hosts. */
    h.set("ngrok-skip-browser-warning", "true");
  }
  config.headers = h;
  return config;
});

/** FastAPI `detail`: string or validation error list; also connection/API-down hints */
export function getApiErrorMessage(err: unknown, fallback: string): string {
  if (!axios.isAxiosError(err)) return fallback;

  if (!err.response) {
    const code = err.code;
    const netLike =
      code === "ERR_NETWORK" ||
      code === "ECONNREFUSED" ||
      code === "ECONNRESET" ||
      code === "ETIMEDOUT" ||
      code === "ECONNABORTED" ||
      code === "ERR_EMPTY_RESPONSE" ||
      err.message === "Network Error";
    if (netLike) {
      const where =
        baseURL ||
        (typeof window !== "undefined"
          ? `${window.location.origin} → /api/upstream → FastAPI (:8000)`
          : "http://127.0.0.1:8000");
      return `Cannot reach API (${where}). Keep ngrok→:3000, Next, and uvicorn running; try http://localhost:3000 on this PC.`;
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
  opts?: { category?: "watch"; brand?: string }
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
