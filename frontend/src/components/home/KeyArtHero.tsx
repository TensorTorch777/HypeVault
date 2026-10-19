"use client";

import Image from "next/image";
import Link from "next/link";
import { useRef } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";

import { Lockup } from "@/components/brand/Lockup";
import { cn } from "@/lib/utils";

export type KeyArtPanel = {
  area: string;
  render: string | null;
  backdrop: string | null;
  label: string;
  tint: string; // CSS gradient used when no backdrop has been generated yet
  fit?: "hero" | "dial";
};

function Panel({ p, i }: { p: KeyArtPanel; i: number }) {
  const reduced = useReducedMotion();
  return (
    <motion.figure
      initial={reduced ? false : { opacity: 0, scale: 1.06, filter: "blur(8px)" }}
      animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
      transition={{ delay: 0.12 + i * 0.09, duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
      className="group relative overflow-hidden rounded-[4px]"
      style={{ gridArea: p.area, background: p.tint }}
    >
      {p.backdrop ? (
        <Image src={p.backdrop} alt="" fill sizes="(min-width: 1024px) 40vw, 50vw" className="object-cover transition-transform duration-[1.6s] ease-out group-hover:scale-105" />
      ) : null}
      <div className="absolute inset-0 bg-gradient-to-t from-[#14112A]/45 via-transparent to-transparent" />
      {p.render ? (
        <Image
          src={p.render}
          alt={p.label}
          fill
          sizes="(min-width: 1024px) 30vw, 50vw"
          className={cn(
            "object-contain drop-shadow-[0_24px_30px_rgba(20,17,42,0.55)] transition-transform duration-[1.6s] ease-out group-hover:scale-[1.04]",
            p.fit === "dial" ? "scale-[1.35]" : "scale-[1.08]"
          )}
        />
      ) : null}
      <figcaption className="hv-mono absolute bottom-2 left-3 text-[10px] text-white/80">{p.label}</figcaption>
    </motion.figure>
  );
}

/**
 * Key-art cover: a tight collage of studio renders around the HypeVault lockup,
 * in the manner of a GTA VI cover. Scrolling eases the cover forward and away.
 */
export function KeyArtHero({ panels, quality }: { panels: KeyArtPanel[]; quality: string | null }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const scale = useTransform(scrollYProgress, [0, 1], reduced ? [1, 1] : [1, 1.12]);
  const y = useTransform(scrollYProgress, [0, 1], reduced ? ["0%", "0%"] : ["0%", "10%"]);
  const fade = useTransform(scrollYProgress, [0.35, 0.95], [1, 0]);

  return (
    <section ref={ref} aria-label="HypeVault" className="relative min-h-[100svh] overflow-hidden pt-20">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_45%_at_50%_40%,rgba(185,163,240,0.16),transparent_70%)]" aria-hidden="true" />
      <motion.div style={{ scale, y, opacity: fade }} className="relative mx-auto w-full max-w-[1180px] px-4 sm:px-6">
        <div
          className="relative grid aspect-[4/5] gap-[5px] rounded-[8px] bg-[#0A0B17] p-[5px] shadow-[0_60px_120px_-40px_rgba(0,0,0,0.9)] sm:aspect-[16/10]
            [grid-template-areas:'e_e''a_b''d_c''f_g'] [grid-template-columns:1fr_1fr] [grid-template-rows:1.4fr_1fr_1fr_1fr]
            sm:[grid-template-areas:'a_a_b_c''d_e_e_c''d_f_g_g'] sm:[grid-template-columns:1.1fr_1fr_1fr_1.1fr] sm:[grid-template-rows:1fr_1.15fr_1fr]"
        >
          {panels.map((p, i) => (
            <Panel key={p.area} p={p} i={i} />
          ))}
          <motion.div
            initial={reduced ? false : { opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.75, duration: 1, ease: [0.22, 1, 0.36, 1] }}
            className="hv-sky relative grid place-items-center overflow-hidden rounded-[4px]"
            style={{ gridArea: "e" }}
          >
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(20,17,42,0.35))]" />
            <Lockup className="relative" />
          </motion.div>
        </div>

        <div className="mt-6 grid grid-cols-[1fr_auto] items-center gap-4 pb-10 sm:grid-cols-[1fr_auto_1fr]">
          <p className="font-display text-[15px] font-bold uppercase leading-tight tracking-[0.04em] text-primary">
            Five houses
            <br />
            One market
          </p>
          <Link href="/market" className="hv-btn-pill hv-btn-primary">
            Enter the market
          </Link>
          <p className="hidden text-right font-display text-[15px] font-bold uppercase leading-tight tracking-[0.04em] text-primary/80 sm:block">
            DINOv2 · DINOv3
            <br />
            <span className="text-primary/55">Screening every photo</span>
          </p>
        </div>
        {quality === "preview" ? (
          <p className="hv-mono -mt-6 pb-8 text-center text-[10px] text-primary/40">Preview renders · final RTX renders pending</p>
        ) : null}
      </motion.div>
    </section>
  );
}
