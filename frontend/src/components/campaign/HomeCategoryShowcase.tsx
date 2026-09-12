"use client";

import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

function WatchDialBg({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="100" cy="100" r="88" stroke="#1D1D1F" strokeOpacity="0.06" strokeWidth="1.5" />
      <circle cx="100" cy="100" r="72" stroke="#1D1D1F" strokeOpacity="0.05" strokeWidth="1" />
      {Array.from({ length: 12 }).map((_, i) => {
        const a = (i * Math.PI) / 6 - Math.PI / 2;
        const x1 = 100 + Math.cos(a) * 78;
        const y1 = 100 + Math.sin(a) * 78;
        const x2 = 100 + Math.cos(a) * 84;
        const y2 = 100 + Math.sin(a) * 84;
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#1D1D1F" strokeOpacity="0.06" strokeWidth="1" />;
      })}
      <circle cx="100" cy="100" r="4" fill="#1D1D1F" fillOpacity="0.1" />
    </svg>
  );
}

export function HomeCategoryShowcase() {
  return (
    <div className="grid gap-5">
      <Link href="/product/Luxury%20watches" className="group block">
        <div className="hv-card relative min-h-[340px] cursor-pointer overflow-hidden bg-gradient-to-br from-[#FBF8F2] via-[#FBFBFD] to-white">
          <WatchDialBg className="pointer-events-none absolute left-1/2 top-1/2 h-[120%] w-[120%] -translate-x-1/2 -translate-y-1/2" />
          <div className="relative flex h-full min-h-[340px] flex-col justify-between p-8 md:p-10">
            <div>
              <p className="text-[13px] font-semibold text-[#C9A84C]">Watches</p>
              <p className="mt-4 font-[family-name:var(--font-display)] text-[32px] font-bold leading-[1.05] tracking-tight text-[#1D1D1F] md:text-[40px]">
                Quiet luxury. Verified.
              </p>
              <p className="mt-3 max-w-[32ch] text-[15px] leading-[1.5] text-[#6E6E73]">
                Ultra-luxury timepieces with AI authenticity checks and market comparables in one view.
              </p>
            </div>
            <span className="inline-flex w-fit items-center gap-1 text-[15px] font-medium text-[#0066CC]">
              Browse watches
              <ArrowUpRight className="h-4 w-4 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </span>
          </div>
        </div>
      </Link>
    </div>
  );
}
