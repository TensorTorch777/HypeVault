"use client";

import Image from "next/image";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { motion, useReducedMotion } from "framer-motion";

import { fetchBenchmarkSummary } from "@/lib/market";

const POSTERS = [
  {
    name: "DINOv2",
    role: "The champion",
    status: "Live on every listing",
    sky: "linear-gradient(180deg,#1C2550 0%,#3B4E9A 45%,#7FA6D8 80%,#C9E0F2 100%)",
    stats: ["ViT-B/14", "504 px", "36 × 36 patches", "Meta, 2023"],
    grid: 36,
  },
  {
    name: "DINOv3",
    role: "The challenger",
    status: "On trial",
    sky: "linear-gradient(180deg,#2B1840 0%,#7A3C78 40%,#E0789A 72%,#FFC08A 100%)",
    stats: ["ViT-B/16", "512 px", "32 × 32 patches", "Meta, 2025"],
    grid: 32,
  },
];

/** DINOv2 vs DINOv3 as two trailer posters. The verdict comes from the API, never from this file. */
export function Showdown({ dials }: { dials: [string | null, string | null] }) {
  const reduced = useReducedMotion();
  const q = useQuery({ queryKey: ["benchmark-summary"], queryFn: fetchBenchmarkSummary });
  const frontier = q.data?.confirmatory?.frontier;
  const required = q.data?.protocol.sample_size?.["required_discordant"];

  return (
    <section aria-labelledby="showdown" className="relative py-28">
      <div className="mx-auto max-w-[1060px] px-4 sm:px-6">
        <p className="text-center font-display text-[13px] font-bold uppercase tracking-[0.08em] text-primary/70">Watch</p>
        <h2 id="showdown" className="hv-wide mt-1 text-center text-[clamp(1.6rem,3.6vw,3rem)] text-primary">
          The showdown
        </h2>

        <div className="relative mt-12 grid gap-5 sm:grid-cols-2">
          {POSTERS.map((p, i) => (
            <motion.article
              key={p.name}
              initial={reduced ? false : { opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: i * 0.12, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
              className="relative flex aspect-[4/5] flex-col justify-end overflow-hidden rounded-[22px] border border-white/10 p-7"
              style={{ background: p.sky }}
            >
              {dials[i] ? (
                <Image src={dials[i] as string} alt="" fill sizes="(min-width: 640px) 50vw, 100vw" className="scale-[1.25] object-contain opacity-90" />
              ) : null}
              {/* the patch grid this model actually reads the photo through */}
              <div
                className="absolute inset-[12%] rounded-[6px] border border-white/30"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px)",
                  backgroundSize: `calc(100% / ${p.grid}) calc(100% / ${p.grid})`,
                }}
                aria-hidden="true"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#14112A] via-[#14112A]/35 to-transparent" />
              <div className="relative">
                <p className="hv-eyebrow">
                  {p.role} · {p.status}
                </p>
                <p className="mt-1 font-display text-[clamp(3.6rem,8vw,6rem)] font-black uppercase leading-[0.85] text-white">{p.name}</p>
                <ul className="mt-4 flex flex-wrap gap-2">
                  {p.stats.map((s) => (
                    <li key={s} className="rounded-full bg-white/10 px-3 py-1 text-[12px] font-semibold text-white/90 backdrop-blur">
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.article>
          ))}
          <span
            className="pointer-events-none absolute left-1/2 top-1/2 hidden h-20 w-20 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/20 bg-background font-display text-[1.8rem] font-black text-primary shadow-[0_20px_40px_rgba(0,0,0,0.5)] sm:grid"
            aria-label="versus"
          >
            VS
          </span>
        </div>

        <div className="mx-auto mt-12 grid max-w-3xl gap-6 text-center">
          <p className="hv-headline text-[clamp(1.6rem,3vw,2.4rem)] leading-tight">
            {q.isLoading ? "Loading the verdict…" : frontier ? (frontier.frontier_model ?? frontier.outcome.replaceAll("_", " ").toLowerCase()) : "Fight not yet scored."}
          </p>
          <p className="text-[15px] leading-7 text-primary/70">
            {frontier?.statement ??
              `The rules were written before any data: at least ${typeof required === "number" ? required : "the pre-registered number of"} photos where the two disagree, an exact McNemar test below 0.05, and the winner's false-authentic upper bound at or below 10%. Until then DINOv2 stays live.`}
          </p>
          <Link href="/benchmark" className="hv-btn-pill hv-btn-primary justify-self-center">
            Read the rulebook <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </section>
  );
}
