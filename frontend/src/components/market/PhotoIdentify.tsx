"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Camera, Check, Minus, X } from "lucide-react";

import { type EbayPhotoMatch, type IdentifyCandidate, type IdentifyResult, type MarketSearch, chrono24SearchUrl, formatPrice } from "@/lib/market";
import { cn } from "@/lib/utils";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;
const MIN_SIDE = 200;

/** Client-side checks only; the API re-validates type, size and resolution. */
async function checkPhoto(file: File): Promise<string | null> {
  if (!ACCEPT.includes(file.type)) return "Use a JPEG, PNG or WebP photo.";
  if (file.size > MAX_BYTES) return "The photo is larger than 10 MB.";
  const url = URL.createObjectURL(file);
  try {
    const img = new window.Image();
    img.src = url;
    await img.decode();
    if (Math.min(img.naturalWidth, img.naturalHeight) < MIN_SIDE) {
      return `The photo is too small (${img.naturalWidth}×${img.naturalHeight}). Use one at least ${MIN_SIDE}px on each side.`;
    }
    return null;
  } catch {
    return "This file is not a readable image.";
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Camera button plus a preview chip for the chosen photo. Drop handling lives on the search form. */
export function PhotoPicker({
  photo,
  onPhoto,
  onError,
}: {
  photo: File | null;
  onPhoto: (file: File | null) => void;
  onError: (message: string | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => {
    if (!photo) return setPreview(null);
    const url = URL.createObjectURL(photo);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  return (
    <>
      <input
        ref={input}
        type="file"
        accept={ACCEPT.join(",")}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={async (e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) await pickPhoto(file, onPhoto, onError);
        }}
      />
      {photo && preview ? (
        <span className="flex shrink-0 items-center gap-1 self-center rounded-full bg-white/10 py-0.5 pl-0.5 pr-1">
          {/* eslint-disable-next-line @next/next/no-img-element -- local blob preview */}
          <img src={preview} alt="Your watch photo" className="h-9 w-9 rounded-full object-cover" />
          <button
            type="button"
            onClick={() => onPhoto(null)}
            className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full text-primary/70 hover:bg-white/10 hover:text-primary"
            aria-label="Remove photo"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </span>
      ) : (
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center self-center rounded-full text-primary/60 transition-colors hover:bg-white/10 hover:text-accent"
          aria-label="Search with a photo"
          title="Search with a photo"
        >
          <Camera className="h-5 w-5" aria-hidden="true" />
        </button>
      )}
    </>
  );
}

export async function pickPhoto(file: File, onPhoto: (f: File) => void, onError: (m: string | null) => void) {
  const problem = await checkPhoto(file);
  onError(problem);
  if (!problem) onPhoto(file);
}

const STATUS_TEXT: Record<IdentifyResult["status"], string> = {
  recognised: "We think this is",
  unsure: "We're not sure. Is it one of these?",
  free_text: "Search as typed",
  not_recognised: "We couldn't recognise this watch",
  not_a_watch: "This doesn't look like a watch",
  unsupported_brand: "This brand isn't covered",
};

function Confidence({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <span className="flex items-center gap-2 text-[0.6875rem] font-bold text-primary/60" title="How strongly the clues agree on this name">
      <span className="h-1 w-14 overflow-hidden rounded-full bg-white/10">
        <span className="block h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
      </span>
      {pct}%
    </span>
  );
}

/** The guesses for the user to confirm. Picking one runs the market search. */
export function IdentifyResultCard({
  result,
  onPick,
  onTypeInstead,
  onDismiss,
}: {
  result: IdentifyResult;
  onPick: (c: IdentifyCandidate) => void;
  onTypeInstead: () => void;
  onDismiss: () => void;
}) {
  const [best, ...rest] = result.candidates;
  return (
    <section className="hv-hud relative mt-4 max-w-2xl p-5" aria-live="polite" aria-label="Watch identification">
      <button
        type="button"
        onClick={onDismiss}
        className="absolute right-3 top-3 flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-primary/50 hover:bg-white/10 hover:text-primary"
        aria-label="Close"
      >
        <X className="h-4 w-4" aria-hidden="true" />
      </button>
      <p className="hv-eyebrow">{STATUS_TEXT[result.status]}</p>

      {best ? (
        <>
          <button
            type="button"
            onClick={() => onPick(best)}
            className="group mt-2 flex w-full cursor-pointer flex-wrap items-center justify-between gap-3 rounded-[0.75rem] border border-accent/40 bg-accent/10 p-4 text-left transition-colors hover:border-accent"
          >
            <span>
              <span className="block font-display text-[1.6rem] font-extrabold leading-tight text-primary">{best.label}</span>
              {best.evidence.length ? (
                <span className="mt-1 block text-[0.75rem] leading-5 text-primary/60">{best.evidence.slice(0, 2).join(" · ")}</span>
              ) : null}
            </span>
            <span className="flex items-center gap-3">
              {best.confidence != null ? <Confidence value={best.confidence} /> : null}
              <span className="hv-btn-pill hv-btn-primary">Search this</span>
            </span>
          </button>
          {rest.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {rest.map((c) => (
                <button
                  key={c.label}
                  type="button"
                  onClick={() => onPick(c)}
                  className="flex min-h-[2.75rem] cursor-pointer items-center gap-3 rounded-full border border-white/15 px-4 text-[0.8125rem] font-bold text-primary/80 transition-colors hover:border-accent/60 hover:text-primary"
                >
                  {c.label}
                  {c.confidence != null ? <span className="text-primary/45">{Math.round(c.confidence * 100)}%</span> : null}
                </button>
              ))}
            </div>
          ) : null}
        </>
      ) : (
        <p className="mt-2 text-[0.875rem] leading-6 text-primary/75">
          {result.status === "not_a_watch"
            ? `The photo looks like ${result.looks_like ?? "something else"}. Try a clear photo of the watch dial.`
            : result.status === "unsupported_brand"
              ? `It looks like a ${result.other_brand ?? "different"} watch. We only cover A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille and Vacheron Constantin.`
              : "Try a sharper, straight-on photo of the dial, or type what you can read on it."}
        </p>
      )}

      <EbayPhotoMatches matches={result.ebay_photo_matches} className="mt-4" />

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-white/10 pt-3 text-[0.75rem] text-primary/55">
        {result.dial_text.length ? <span>Dial text read: “{result.dial_text.slice(0, 4).join(" · ")}”</span> : null}
        <button type="button" onClick={onTypeInstead} className="cursor-pointer font-bold text-primary/75 underline-offset-4 hover:text-accent hover:underline">
          {best ? "None of these? Type the name" : "Type the name instead"}
        </button>
      </div>
    </section>
  );
}

/** eBay's closest photos to the user's, like a visual-search strip. Each opens the listing. */
export function EbayPhotoMatches({ matches, className }: { matches?: EbayPhotoMatch[]; className?: string }) {
  if (!matches?.length) return null;
  return (
    <div className={className}>
      <p className="hv-eyebrow">Closest photos on eBay</p>
      <div className="-mx-1 mt-2 flex gap-2 overflow-x-auto px-1 pb-2">
        {matches.slice(0, 8).map((m, i) => (
          <a
            key={`${m.item_url ?? m.title}-${i}`}
            href={m.item_url ?? undefined}
            target="_blank"
            rel="noreferrer"
            className="group w-32 shrink-0 overflow-hidden rounded-[0.75rem] border border-white/10 bg-card transition-colors hover:border-accent/50"
          >
            <span className="relative block aspect-square bg-white">
              {m.image_url ? <Image src={m.image_url} alt={m.title} fill sizes="128px" className="object-contain" /> : null}
            </span>
            <span className="block p-2">
              <span className="block font-display text-[1rem] font-extrabold leading-none text-primary hv-num">{formatPrice(m.price, m.currency, true)}</span>
              <span className="mt-1 line-clamp-2 block text-[0.6875rem] leading-4 text-primary/65">{m.title}</span>
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}

function SourceChip({ label, state, count }: { label: string; state: boolean | null; count?: number }) {
  const look =
    state === true
      ? { cls: "bg-success/15 text-success", Icon: Check, text: count != null ? `Found · ${count}` : "Found" }
      : state === false
        ? { cls: "bg-danger/15 text-danger", Icon: X, text: "Not found" }
        : { cls: "bg-white/10 text-primary/60", Icon: Minus, text: "Not connected" };
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[0.75rem] font-bold", look.cls)}>
      <look.Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}: {look.text}
    </span>
  );
}

/** Where the watch was found: both, one, or neither source. */
export function FoundBanner({ data }: { data: MarketSearch }) {
  if (!data.found) return null;
  const { ebay, chrono24_dataset: c24 } = data.found;
  const nowhere = ebay !== true && c24 === false;
  const usedFallback = data.query_used && data.query_used !== data.query;
  const shown = data.query_used ?? data.query;
  return (
    <div className={cn("mb-5 rounded-[1.125rem] border p-5", nowhere ? "border-danger/40 bg-danger/10" : "border-white/10 bg-card")} role="status">
      {nowhere ? (
        <p className="font-display text-[1.6rem] font-extrabold uppercase leading-tight text-primary">
          Not found on {ebay === null ? "Chrono24 (eBay is not connected)" : "eBay or Chrono24"}
        </p>
      ) : null}
      {usedFallback ? (
        <p className={cn("text-[0.8125rem] text-primary/70", nowhere && "mt-1")}>
          Nothing matched “{data.query}”, so we also searched “{data.query_used}”.
        </p>
      ) : null}
      <div className={cn("flex flex-wrap items-center gap-2", (nowhere || usedFallback) && "mt-3")}>
        <SourceChip label="eBay" state={ebay} count={data.sources.ebay.stats?.count ?? data.sources.ebay.items.length} />
        <SourceChip label="Chrono24 dataset" state={c24} count={data.sources.chrono24_dataset.stats?.count} />
        <a
          href={chrono24SearchUrl(shown, data.brand)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-[0.75rem] font-bold text-primary/70 underline-offset-4 hover:text-accent hover:underline"
        >
          Search Chrono24 live <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
        </a>
      </div>
    </div>
  );
}
