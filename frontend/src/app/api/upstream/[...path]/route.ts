import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

import { internalApiBaseUrl } from "../../../../../loopbackHttp.mjs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const UPSTREAM = internalApiBaseUrl(process.env.INTERNAL_API_URL);

/** Browser-visible origin (ngrok, etc.). Dev often sees :3000 as 127.0.0.1 while Host is the tunnel hostname. */
function clientFacingOrigin(req: NextRequest): string {
  const xfHost = req.headers.get("x-forwarded-host");
  const rawHost = xfHost?.split(",")[0].trim() || req.headers.get("host")?.trim() || "";
  const xfProto = req.headers.get("x-forwarded-proto");
  if (rawHost) {
    const proto =
      xfProto?.split(",")[0].trim() ||
      (looksLikeLoopbackHost(rawHost.split(":")[0] || rawHost) ? "http" : "https");
    try {
      return new URL(`${proto}://${rawHost}`).origin;
    } catch {
      /* fall through */
    }
  }
  return req.nextUrl.origin;
}

function looksLikeLoopbackHost(hostname: string): boolean {
  const h = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "::1";
}

/** Treat localhost / 127.0.0.1 / ::1 as the same host for comparing FastAPI redirects to INTERNAL_API_URL. */
function canonicalLoopbackHostname(hostname: string): string {
  const h = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (h === "localhost" || h === "127.0.0.1" || h === "::1") return "127.0.0.1";
  return h;
}

/** FastAPI may use https vs http, or localhost vs 127.0.0.1 in Location — still same uvicorn process. */
function sameUpstreamHostPort(absolute: URL, base: URL): boolean {
  if (canonicalLoopbackHostname(absolute.hostname) !== canonicalLoopbackHostname(base.hostname)) {
    return false;
  }
  const aPort = absolute.port || (absolute.protocol === "https:" ? "443" : "80");
  const bPort = base.port || (base.protocol === "https:" ? "443" : "80");
  return aPort === bPort;
}

const REQ_HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
]);

const RES_SKIP = new Set(["transfer-encoding", "connection"]);

const UPSTREAM_FETCH_TIMEOUT_MS = 300_000;

/**
 * Use the real URL path (not `params.path`): Next catch-all segments drop a trailing slash, so
 * `/api/upstream/listings/` became `listings` → uvicorn saw `POST /listings` → 307 to `/listings/` →
 * we rewrote that to the browser; the next hop still forwarded `POST /listings` → infinite redirects
 * (`curl -L` “Maximum redirects”).
 */
function upstreamRelPath(req: NextRequest, pathSegments: string[] | undefined): string {
  const pathname = req.nextUrl.pathname;
  const prefix = "/api/upstream";
  if (pathname === prefix || pathname === `${prefix}/`) {
    return "";
  }
  if (pathname.startsWith(`${prefix}/`)) {
    return pathname.slice(prefix.length + 1);
  }
  return pathSegments?.join("/") ?? "";
}

async function proxy(req: NextRequest, pathSegments: string[] | undefined) {
  const path = upstreamRelPath(req, pathSegments);
  const basePath = path ? `${UPSTREAM}/${path}` : UPSTREAM;
  const targetUrl = `${basePath}${req.nextUrl.search}`;

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (REQ_HOP_BY_HOP.has(key.toLowerCase())) return;
    headers.set(key, value);
  });

  const signal =
    typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function"
      ? AbortSignal.timeout(UPSTREAM_FETCH_TIMEOUT_MS)
      : undefined;

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: "manual",
    signal,
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    const body = await req.arrayBuffer();
    if (body.byteLength > 0) {
      init.body = body;
    }
  }

  let upstream: Response;
  try {
    upstream = await fetch(targetUrl, init);
  } catch (e) {
    const isAbort = e instanceof Error && e.name === "AbortError";
    return NextResponse.json(
      {
        detail: isAbort
          ? "AI service timed out (model load or inference). Try again or use a GPU machine."
          : "Cannot reach API backend. Start uvicorn on port 8000.",
      },
      { status: isAbort ? 504 : 502 },
    );
  }

  /*
   * FastAPI often 307-redirects to absolute http://127.0.0.1:8000/... (trailing slash).
   * If we forward that Location to the browser, axios follows to loopback from an https ngrok
   * page → mixed content / blocked → ERR_NETWORK ("Cannot reach API").
   */
  if (upstream.status >= 300 && upstream.status < 400) {
    const loc = upstream.headers.get("location");
    if (loc) {
      try {
        const absolute = new URL(loc, UPSTREAM);
        const base = new URL(UPSTREAM);
        if (sameUpstreamHostPort(absolute, base)) {
          const rel = `${absolute.pathname}${absolute.search}${absolute.hash}`;
          const pathPart = rel.startsWith("/") ? rel : `/${rel}`;
          const publicLoc = new URL(`/api/upstream${pathPart}`, clientFacingOrigin(req)).href;
          const outRedirect = new NextResponse(null, { status: upstream.status });
          outRedirect.headers.set("Location", publicLoc);
          const setCookiesRedir =
            typeof upstream.headers.getSetCookie === "function" ? upstream.headers.getSetCookie() : [];
          for (const c of setCookiesRedir) {
            outRedirect.headers.append("Set-Cookie", c);
          }
          upstream.headers.forEach((value, key) => {
            const l = key.toLowerCase();
            if (l === "set-cookie" || l === "location") return;
            if (RES_SKIP.has(l)) return;
            outRedirect.headers.append(key, value);
          });
          return outRedirect;
        }
      } catch {
        /* fall through and forward redirect as-is (e.g. OAuth → accounts.google.com) */
      }
    }
  }

  /*
   * Undici/NextResponse rejects a body with 204/205/304 ("Invalid response status code 204").
   * Logout returns 204 + Set-Cookie clears — must use null body or the BFF 500s and cookies stick.
   */
  const nullBody = upstream.status === 204 || upstream.status === 205 || upstream.status === 304;
  const body = nullBody ? null : await upstream.arrayBuffer();

  const out = new NextResponse(body, { status: upstream.status });

  const setCookies =
    typeof upstream.headers.getSetCookie === "function" ? upstream.headers.getSetCookie() : [];
  for (const c of setCookies) {
    out.headers.append("Set-Cookie", c);
  }

  upstream.headers.forEach((value, key) => {
    const l = key.toLowerCase();
    if (l === "set-cookie") return;
    if (RES_SKIP.has(l)) return;
    /* content-length from upstream can disagree when we drop the body for 204 */
    if (nullBody && (l === "content-length" || l === "content-type")) return;
    out.headers.append(key, value);
  });

  return out;
}

export async function GET(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params.path);
}

export async function POST(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params.path);
}

export async function PUT(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params.path);
}

export async function PATCH(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params.path);
}

export async function DELETE(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params.path);
}

/** Avoid proxying OPTIONS to uvicorn (many routes 405) if a client ever preflights. */
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: { Allow: "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS" },
  });
}
