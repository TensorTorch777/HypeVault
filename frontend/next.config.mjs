import { internalApiBaseUrl } from "./loopbackHttp.mjs";

/** @type {import('next').NextConfig} */
const apiOrigin = internalApiBaseUrl(process.env.INTERNAL_API_URL);

const nextConfig = {
  reactStrictMode: true,
  /*
   * Proxy selected paths to uvicorn (plain HTTP on loopback). Never use https://127.0.0.1 here.
   * Do not add `source: "/api/:path*"` — it would steal `/api/upstream/*` from the Route Handler.
   */
  async rewrites() {
    return [
      { source: "/auth/:path*", destination: `${apiOrigin}/auth/:path*` },
      { source: "/verify/:path*", destination: `${apiOrigin}/verify/:path*` },
      { source: "/listings/:path*", destination: `${apiOrigin}/listings/:path*` },
      { source: "/static/:path*", destination: `${apiOrigin}/static/:path*` },
      { source: "/health", destination: `${apiOrigin}/health` },
      { source: "/health/ready", destination: `${apiOrigin}/health/ready` },
      { source: "/docs", destination: `${apiOrigin}/docs` },
      { source: "/openapi.json", destination: `${apiOrigin}/openapi.json` },
      { source: "/metrics", destination: `${apiOrigin}/metrics` },
    ];
  },
  /* Avoid corrupted filesystem cache (missing chunks / 404 on /_next/static) when .next is shared or wiped mid-run */
  webpack: (config, { dev }) => {
    if (dev) {
      config.cache = false;
    }
    return config;
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "**" },
      { protocol: "http", hostname: "localhost", port: "8000" },
    ],
  },
};

export default nextConfig;
