/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
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
