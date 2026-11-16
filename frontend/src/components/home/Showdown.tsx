"use client";

import Image from "next/image";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { motion, useReducedMotion } from "framer-motion";

import { type KN, type LiveModelEvidence, fetchBenchmarkSummary } from "@/lib/market";

const POSTERS = [
  {
    name: "DINOv2",
    role: "The original",
    status: "Research comparison",
    sky: "linear-gradient(180deg,#1C2550 0%,#3B4E9A 45%,#7FA6D8 80%,#C9E0F2 100%)",
    stats: ["ViT-B/14", "504 px", "36 × 36 patches", "Meta, 2023"],
    grid: 36,
    live: false,
  },
  {
    name: "DINOv3",
    role: "The challenger",
    status: "Research comparison",
    sky: "linear-gradient(180deg,#2B1840 0%,#7A3C78 40%,#E0789A 72%,#FFC08A 100%)",
    stats: ["ViT-B/16", "512 px", "32 × 32 patches", "Meta, 2025"],
    grid: 32,
    live: false,
  },
  {
    name: "DINOv3 + PE",
    role: "The champion",
    status: "Live on every listing",
    sky: "linear-gradient(180deg,#3A1A2E 0%,#9A3F6E 38%,#F08AA8 68%,#FFD27A 100%)",
    stats: ["DINOv3 ViT-B/16", "Perception Encoder L/14", "One frozen head", "Phase 76, 2026"],
    grid: 32,
    live: true,
  },
];

const frac = (x: KN) => (x ? `${x.k}/${x.n}` : "—");

/** Every number in the verdict is read from the API; nothing here is hard-coded. */
function verdict(live: LiveModelEvidence): string {
  const g = Object.fromEntries(live.benchmark.groups.map((row) => [row.id, row.systems]));
  const ours = "phase76_head";
  const base = "dinov2_live";
  const real = g.real_ebay_v6?.[ours];
  let text =
    `On ${live.benchmark.name}, written before any of its images existed and scored once, it caught ${frac(g.chatgpt_v6?.[ours])} ChatGPT ` +
    `and ${frac(g.gemini_v6?.[ours])} Gemini watch images, against ${frac(g.chatgpt_v6?.[base])} and ${frac(g.gemini_v6?.[base])} for DINOv2, ` +
    `and wrongly flagged ${real ? `${real.k} of ${real.n}` : "—"} real eBay photos.`;
  const unseen = live.unseen_generators;
  if (unseen?.caught) text += ` On generators it never trained on (${unseen.generators}) it caught ${frac(unseen.caught)}.`;
  return `${text} It now screens every listing; DINOv3 and DINOv2 stay in the research console for comparison.`;
}

/** DINOv2, DINOv3 and the live DINOv3 + Perception Encoder hybrid as three trailer posters. */
export function Showdown({ dials }: { dials: (string | null)[] }) {
  const reduced = useReducedMotion();
  const q = useQuery({ queryKey: ["benchmark-summary"], queryFn: fetchBenchmarkSummary });
  const live = q.data?.live_model;

  return (
    <section aria-labelledby="showdown" className="relative py-28">
      <div className="mx-auto max-w-[66.25rem] px-4 sm:px-6">
        <p className="text-center font-display text-[0.8125rem] font-bold uppercase tracking-[0.08em] text-primary/70">Watch</p>
        <h2 id="showdown" className="hv-wide mt-1 text-center text-[clamp(1.6rem,3.6vw,3rem)] text-primary">
          The showdown
        </h2>

        <div className="relative mt-12 grid gap-5 sm:grid-cols-3">
          {POSTERS.map((p, i) => (
            <motion.article
              key={p.name}
              initial={reduced ? false : { opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: i * 0.12, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
              className={`relative flex aspect-[4/5] flex-col justify-end overflow-hidden rounded-[1.375rem] border p-6 ${p.live ? "border-[#FFD27A]/70 shadow-[0_0_0_1px_rgba(255,210,122,0.35),0_30px_80px_rgba(240,138,168,0.35)]" : "border-white/10"}`}
              style={{ background: p.sky }}
            >
              {dials[i] ? (
                <Image src={dials[i] as string} alt="" fill sizes="(min-width: 640px) 33vw, 100vw" className="scale-[1.25] object-contain opacity-90" />
              ) : null}
              {/* the patch grid this model actually reads the photo through */}
              <div
                className="absolute inset-[12%] rounded-[0.375rem] border border-white/30"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px)",
                  backgroundSize: `calc(100% / ${p.grid}) calc(100% / ${p.grid})`,
                }}
                aria-hidden="true"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#14112A] via-[#14112A]/35 to-transparent" />
              {p.live ? (
                <span className="absolute right-4 top-4 rounded-full bg-[#FFD27A] px-3 py-1 font-display text-[0.6875rem] font-black uppercase tracking-[0.12em] text-[#2B1840]">
                  Live
                </span>
              ) : null}
              <div className="relative">
                <p className="hv-eyebrow">
                  {p.role} · {p.status}
                </p>
                <p className={`mt-1 font-display font-black uppercase leading-[0.85] text-white ${p.live ? "text-[clamp(2rem,3.6vw,3.1rem)]" : "text-[clamp(2.6rem,5.2vw,4.4rem)]"}`}>{p.name}</p>
                <ul className="mt-4 flex flex-wrap gap-2">
                  {p.stats.map((s) => (
                    <li key={s} className="rounded-full bg-white/10 px-2.5 py-1 text-[0.6875rem] font-semibold text-white/90 backdrop-blur">
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.article>
          ))}
        </div>

        <div className="mx-auto mt-12 grid max-w-3xl gap-6 text-center">
          <p className="hv-headline text-[clamp(1.6rem,3vw,2.4rem)] leading-tight">
            {q.isLoading ? "Loading the verdict…" : live ? live.name : "Verdict unavailable."}
          </p>
          <p className="text-[0.9375rem] leading-7 text-primary/70">
            {live
              ? verdict(live)
              : "Each model is judged by a benchmark written before its images existed and scored once. The result could not be loaded."}
          </p>
          <Link href="/benchmark" className="hv-btn-pill hv-btn-primary justify-self-center">
            Read the rulebook <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </section>
  );
}
