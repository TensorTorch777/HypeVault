import Image from "next/image";
import Link from "next/link";

import { Lockup } from "@/components/brand/Lockup";
import { CinematicBand } from "@/components/home/CinematicBand";
import { KeyArtHero, type KeyArtPanel } from "@/components/home/KeyArtHero";
import { Showdown } from "@/components/home/Showdown";
import { Turnaround } from "@/components/home/Turnaround";
import { BrandIndex } from "@/components/market/BrandIndex";
import { STUDIO_WATCHES, backdropUrl, getRenderAssets, heroUrl } from "@/lib/renderAssets";

// Painted skies per panel; a generated backdrop replaces each one once it exists.
const SKIES = {
  lavender: "linear-gradient(180deg,#2B2350 0%,#6B4A97 45%,#C77BB0 78%,#F6B3B0 100%)",
  peach: "linear-gradient(180deg,#3A2552 0%,#9A4F86 40%,#F08AA8 70%,#FFC08A 100%)",
  rose: "linear-gradient(180deg,#24204A 0%,#5A3F8C 35%,#E0789A 75%,#FFD9A0 100%)",
  sea: "linear-gradient(180deg,#1C2550 0%,#3B4E9A 40%,#9C86C8 75%,#F6B3B0 100%)",
  night: "linear-gradient(180deg,#0F1230 0%,#2A2160 50%,#7A3C78 85%,#E0789A 100%)",
};

const METHOD = [
  {
    n: "01",
    title: "Declare and shoot",
    body: "The seller picks the manufacture from the five we cover and uploads a photo. It must be at least 400 px on its short side.",
    watch: "lange_one",
  },
  {
    n: "02",
    title: "Screened, or sent back",
    body: "A signed AI label or a copy of a known AI image is stopped. CLIP checks there is a wristwatch, then the DINOv3 + Perception Encoder model looks for the traces image generators leave behind. A borderline or edited photo is held until the seller sends a proof photo. Pass and the watch is listed: screened, not certified.",
    watch: "ap_royal_oak",
  },
  {
    n: "03",
    title: "Set against the street",
    body: "Every listing sits beside real asking prices: live from the eBay Browse API and from a dated Chrono24 dataset, each labelled with where and when it was seen.",
    watch: "vc_overseas",
  },
];

export default function HomePage() {
  const assets = getRenderAssets();
  const hero = (id: string) => heroUrl(assets, id, "hero");
  const flat = (id: string) => heroUrl(assets, id, "flat");

  const panels: KeyArtPanel[] = [
    { area: "a", render: hero("patek_nautilus"), backdrop: backdropUrl(assets, "marina"), label: "Patek Philippe", tint: SKIES.lavender },
    { area: "b", render: flat("ap_royal_oak"), backdrop: backdropUrl(assets, "ocean_drive"), label: "Audemars Piguet", tint: SKIES.peach, fit: "dial" },
    { area: "c", render: hero("rm_rm011"), backdrop: backdropUrl(assets, "neon_rain"), label: "Richard Mille", tint: SKIES.night },
    { area: "d", render: hero("lange_one"), backdrop: backdropUrl(assets, "penthouse"), label: "A. Lange & Söhne", tint: SKIES.rose },
    { area: "f", render: hero("vc_overseas"), backdrop: backdropUrl(assets, "rooftop_pool"), label: "Vacheron Constantin", tint: SKIES.sea },
    { area: "g", render: flat("patek_nautilus"), backdrop: backdropUrl(assets, "bay_dusk"), label: "Five houses", tint: SKIES.peach, fit: "dial" },
  ];

  const turnaround = STUDIO_WATCHES.map((s) => ({
    id: s.id,
    brand: s.brand,
    piece: s.piece,
    frames: assets.watches[s.id]?.turntable_frames ?? 0,
    hero: hero(s.id),
  }));

  const tints = [SKIES.lavender, SKIES.peach, SKIES.rose, SKIES.sea, SKIES.night];
  const houses = STUDIO_WATCHES.map((s, i) => ({
    brand: s.brand,
    piece: s.piece,
    city: s.city,
    since: s.since,
    render: hero(s.id),
    backdrop: backdropUrl(assets, s.backdrop),
    tint: tints[i],
  }));

  return (
    <>
      <KeyArtHero panels={panels} quality={assets.quality} />

      <Turnaround watches={turnaround} />

      <CinematicBand backdrop={backdropUrl(assets, "bay_dusk")} render={hero("patek_nautilus")} />

      {/* The five houses */}
      <section aria-labelledby="houses" className="relative py-28">
        <div className="mx-auto max-w-[66.25rem] px-4 sm:px-6">
          <p className="hv-eyebrow">People &amp; places</p>
          <h2 id="houses" className="mt-1 font-display text-[clamp(2.6rem,5vw,4.2rem)] font-extrabold uppercase leading-[0.9] text-primary">
            Only in the vault
          </h2>
          <p className="mt-4 max-w-xl text-[0.9375rem] leading-7 text-primary/70">
            Five manufactures, each with its live median asking price. Anything else is outside scope and gets no result at all.
          </p>
        </div>
        <div className="mt-12 px-4 sm:px-6">
          <BrandIndex houses={houses} />
        </div>
      </section>

      {/* Method */}
      <section id="method" aria-labelledby="method-title" className="scroll-mt-20 py-24">
        <div className="mx-auto max-w-[66.25rem] px-4 sm:px-6">
          <div className="flex items-end justify-between gap-6">
            <h2 id="method-title" className="font-display text-[clamp(2.2rem,4vw,3.4rem)] font-extrabold uppercase leading-[0.9] text-primary">
              How it works
            </h2>
            <Link href="/research" className="hv-chevron shrink-0">
              Research console
            </Link>
          </div>
          <ol className="mt-10 grid gap-5 md:grid-cols-3">
            {METHOD.map((m) => {
              const img = flat(m.watch);
              return (
                <li key={m.n} className="hv-card overflow-hidden hover:-translate-y-1">
                  <div className="hv-sky relative aspect-[16/10] overflow-hidden">
                    {img ? <Image src={img} alt="" fill sizes="(min-width: 768px) 33vw, 100vw" className="scale-[1.6] object-contain" /> : null}
                    <span className="absolute left-4 top-3 font-display text-[2.2rem] font-black text-white/90">{m.n}</span>
                  </div>
                  <div className="p-6">
                    <h3 className="font-display text-[1.7rem] font-extrabold uppercase leading-none text-primary">{m.title}</h3>
                    <p className="mt-3 text-[0.875rem] leading-6 text-primary/70">{m.body}</p>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      </section>

      <Showdown dials={[flat("ap_royal_oak"), flat("patek_nautilus"), flat("vc_overseas")]} />

      {/* Closing */}
      <section aria-labelledby="closing" className="hv-sky relative isolate overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-background via-transparent to-[#14112A]/40" aria-hidden="true" />
        <div className="mx-auto flex min-h-[80svh] max-w-[66.25rem] flex-col items-center justify-center px-4 py-28 text-center sm:px-6">
          <Lockup />
          <h2 id="closing" className="hv-wide mt-10 text-[clamp(1.2rem,2.6vw,2rem)] text-white">
            List a watch. Bring the evidence.
          </h2>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link href="/seller/upload" className="hv-btn-pill hv-btn-primary">
              Start a listing <span aria-hidden="true">→</span>
            </Link>
            <Link href="/market" className="hv-btn-pill hv-btn-secondary">
              Browse the market
            </Link>
          </div>
          <p className="mt-10 max-w-md text-[0.75rem] font-medium leading-5 text-white/85">
            A university research prototype. Model classifications are screening evidence, never a certificate of authenticity.
            Studio watches are procedural models in the style of each house, without logos.
          </p>
        </div>
      </section>
    </>
  );
}
