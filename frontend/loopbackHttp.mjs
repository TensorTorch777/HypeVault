/**
 * Uvicorn serves plain HTTP on loopback. If INTERNAL_API_URL or NEXT_PUBLIC_API_URL
 * mistakenly uses https://127.0.0.1 or https://localhost, fetches get ERR_SSL_PROTOCOL_ERROR.
 */

const LOOPBACK = new Set(["localhost", "127.0.0.1", "::1"]);

function normHost(hostname) {
  return hostname.toLowerCase().replace(/^\[|\]$/g, "");
}

/** INTERNAL_API_URL (and similar): default FastAPI origin for rewrites + server fetch */
export function internalApiBaseUrl(envValue) {
  const fallback = "http://127.0.0.1:8000";
  const raw = String(envValue ?? "").trim() || fallback;
  try {
    const candidate = /^[\w+.-]+:\/\//.test(raw) ? raw : `http://${raw}`;
    const u = new URL(candidate);
    if (LOOPBACK.has(normHost(u.hostname))) {
      u.protocol = "http:";
    }
    return u.origin;
  } catch {
    return fallback;
  }
}

/** NEXT_PUBLIC_API_URL: optional; only fixes https→http on loopback */
export function coerceLoopbackHttpsToHttp(url) {
  const raw = String(url ?? "").trim();
  if (!raw) return "";
  try {
    const candidate = /^[\w+.-]+:\/\//.test(raw) ? raw : `http://${raw}`;
    const u = new URL(candidate);
    if (LOOPBACK.has(normHost(u.hostname)) && u.protocol === "https:") {
      u.protocol = "http:";
    }
    return u.href.replace(/\/$/, "");
  } catch {
    return raw;
  }
}
