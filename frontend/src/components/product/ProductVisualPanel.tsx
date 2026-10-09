"use client";

import Image from "next/image";
import { Watch } from "lucide-react";

import type { Listing } from "@/lib/api";
import { customerStatusCopy } from "@/lib/customerLabel";
import { cn } from "@/lib/utils";

function PulseDots() {
  return (
    <span className="inline-flex gap-1.5 pl-1" aria-hidden>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block h-1.5 w-1.5 rounded-full bg-[#FF3B00]/80 animate-pulse"
          style={{ animationDelay: `${i * 200}ms` }}
        />
      ))}
    </span>
  );
}

export function ProductVisualPanel({
  title,
  category,
  listing,
  loading,
  isUuid,
}: {
  title: string;
  category: string;
  listing?: Listing | null;
  loading: boolean;
  isUuid: boolean;
}) {
  const Icon = Watch;
  const hasImage = Boolean(listing?.s3_url);
  const label = listing ? customerStatusCopy(listing).title : null;
  const showHero = hasImage && Boolean(label);
  const showPlaceholder = !showHero;

  return (
    <div
      className={cn(
        "relative flex h-full min-h-[min(420px,50vh)] flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0D0D0D]",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
      )}
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.35'/%3E%3C/svg%3E")`,
        backgroundSize: "120px 120px",
      }}
    >
      <div className="relative flex flex-1 flex-col px-6 pb-6 pt-10 md:px-8">
        {showHero ? (
          <div className="relative mx-auto mt-2 aspect-square w-full max-w-[min(100%,380px)] flex-1 overflow-hidden rounded-xl border border-white/10">
            <Image src={listing!.s3_url!} alt="" fill className="object-cover" sizes="(max-width: 1024px) 100vw, 50vw" />
            <span className="absolute left-3 top-3 max-w-[16rem] rounded-md bg-black/80 px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-[0.12em] text-white shadow-lg">
              {label}
            </span>
          </div>
        ) : null}

        {showPlaceholder ? (
          <div className="relative flex flex-1 flex-col items-center justify-center text-center">
            <div className="pointer-events-none absolute h-36 w-36 rounded-full bg-[#FF3B00]/10 blur-3xl" aria-hidden />
            <Icon className="relative z-[1] h-20 w-20 text-[#FF3B00]" strokeWidth={1.25} aria-hidden />
            <h2 className="relative z-[1] mt-6 max-w-md px-2 font-[family-name:var(--font-display)] text-[32px] font-extrabold leading-tight tracking-tight text-white">
              {title}
            </h2>
            <div className="relative z-[1] mt-4 flex flex-wrap justify-center gap-2">
              <span className="rounded-full bg-white/[0.08] px-3 py-1 text-[11px] font-medium uppercase tracking-[0.08em] text-[#a1a1aa]">
                {label ?? "Not a stored listing"}
              </span>
              <span className="rounded-full bg-white/[0.08] px-3 py-1 text-[11px] font-medium uppercase tracking-[0.08em] text-[#a1a1aa]">
                Watches
              </span>
              <span className="rounded-full bg-white/[0.08] px-3 py-1 text-[11px] font-medium uppercase tracking-[0.08em] text-[#a1a1aa]">
                Market compared
              </span>
            </div>
          </div>
        ) : null}

        <div
          className={cn(
            "mt-auto w-full border-t border-white/[0.06] pt-5",
            showHero ? "sr-only" : ""
          )}
        >
          {loading ? (
            <p className="text-sm font-normal text-[#a1a1aa]">
              Listing status
              <PulseDots />
            </p>
          ) : label ? (
            <div>
              <p className="text-sm text-white">{label}</p>
              <p className="mt-1 text-xs text-[#a1a1aa]">
                Published means the listing is visible. It is not an authenticity verification.
              </p>
            </div>
          ) : (
            <p className="text-sm font-normal text-[#a1a1aa]">Not a stored listing</p>
          )}
        </div>
      </div>
    </div>
  );
}
