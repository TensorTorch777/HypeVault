"use client";

import Link from "next/link";

import { WobbleText } from "@/components/motion/BouncyText";

const COLUMNS = [
  {
    heading: "Shop",
    links: [
      { label: "Sneakers",             href: "/product/Sneakers" },
      { label: "Luxury watches",       href: "/product/Luxury%20watches" },
      { label: "Recent verifications", href: "/#recent" },
    ],
  },
  {
    heading: "Sellers",
    links: [
      { label: "List an item",       href: "/seller/upload" },
      { label: "Seller dashboard",   href: "/seller/dashboard" },
      { label: "Pricing",            href: "/register" },
    ],
  },
  {
    heading: "Trust",
    links: [
      { label: "How AI verification works", href: "/#how" },
      { label: "Market comparison",         href: "/product/Sneakers" },
      { label: "Authenticity promise",      href: "/" },
    ],
  },
  {
    heading: "Company",
    links: [
      { label: "Mission",    href: "/" },
      { label: "Research",   href: "/#how" },
      { label: "Contact",    href: "/" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer className="relative overflow-hidden border-t border-[rgba(255,31,164,0.18)] bg-[#0B0118] pt-20 pb-12">
      {/* sunset orb glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-32"
        style={{
          background:
            "radial-gradient(ellipse 90% 100% at 50% 0%, rgba(255,31,164,0.18), transparent 60%)",
        }}
      />

      <div className="relative mx-auto max-w-[1080px] px-5 text-[13px] text-[#FFEDF6]/60">
        {/* Massive wordmark */}
        <p
          className="cursor-default font-[family-name:var(--font-display)] text-[clamp(3rem,11vw,9rem)] font-black italic leading-none tracking-tighter"
          style={{
            background:
              "linear-gradient(95deg, #FFD24A, #FF7A1A 25%, #FF1FA4 60%, #9B5BFF)",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            filter: "drop-shadow(0 0 32px rgba(255,31,164,0.4))",
          }}
        >
          <WobbleText text="HYPE/VAULT" />
        </p>

        <hr className="my-10 border-[rgba(255,31,164,0.14)]" />

        <div className="grid grid-cols-2 gap-10 md:grid-cols-4 md:gap-6">
          {COLUMNS.map((c) => (
            <div key={c.heading}>
              <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#FF1FA4]">
                {c.heading}
              </p>
              <ul className="mt-3 space-y-2.5">
                {c.links.map((l) => (
                  <li key={`${c.heading}-${l.label}`}>
                    <Link
                      href={l.href}
                      className="hv-link-underline inline-block text-[12px] text-[#FFEDF6]/65 transition-colors hover:text-[#00E1FF]"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <hr className="my-10 border-[rgba(255,31,164,0.14)]" />

        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <p className="text-[11px] leading-[1.5] text-[#FFEDF6]/40">
            © {new Date().getFullYear()} HypeVault · Built on DINOv2-Giant ·
            Capstone — not a live marketplace.
          </p>
          <div className="flex gap-4 text-[11px] text-[#FFEDF6]/55">
            <Link href="/" className="hv-link-underline hover:text-[#00E1FF]">Privacy</Link>
            <Link href="/" className="hv-link-underline hover:text-[#00E1FF]">Terms</Link>
            <Link href="/" className="hv-link-underline hover:text-[#00E1FF]">Legal</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
