"use client";

import Link from "next/link";

function VaultIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path
        d="M12 20V14a4 4 0 014-4h16a4 4 0 014 4v6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <rect x="8" y="20" width="32" height="22" rx="3" stroke="currentColor" strokeWidth="2" />
      <circle cx="24" cy="31" r="3" stroke="currentColor" strokeWidth="2" />
      <path d="M21 31h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function ListingsEmptyState() {
  return (
    <div className="mx-auto mt-10 flex min-h-[220px] max-w-xl flex-col items-center justify-center rounded-2xl border-2 border-dashed border-black/[0.10] bg-white p-12 text-center">
      <VaultIcon className="h-12 w-12 text-[#FF3B00]" />
      <h3 className="mt-6 font-[family-name:var(--font-display)] text-[22px] font-semibold text-[#1D1D1F]">
        The vault is empty.
      </h3>
      <p className="mt-2 text-[15px] text-[#6E6E73]">
        Be the first to list a verified sneaker or watch.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-4">
        <Link href="/seller/upload" className="hv-btn-pill hv-btn-primary">
          List an item
        </Link>
        <Link href="/product/Sneakers" className="hv-chevron">
          Browse market prices
        </Link>
      </div>
    </div>
  );
}
