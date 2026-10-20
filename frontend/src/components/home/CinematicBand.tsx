"use client";

import Image from "next/image";
import { useRef } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";

import { HomeFigures } from "@/components/home/HomeFigures";

/**
 * Full-bleed dusk panorama with a salmon deco headline set straight onto the image,
 * like the "Vice City, USA." band on the GTA VI site.
 */
export function CinematicBand({ backdrop, render }: { backdrop: string | null; render: string | null }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const imgY = useTransform(scrollYProgress, [0, 1], reduced ? ["0%", "0%"] : ["-8%", "8%"]);
  const watchY = useTransform(scrollYProgress, [0, 1], reduced ? ["0%", "0%"] : ["14%", "-14%"]);

  return (
    <section ref={ref} aria-labelledby="band-title" className="relative isolate min-h-[110svh] overflow-hidden">
      <motion.div style={{ y: imgY }} className="absolute -inset-y-[10%] inset-x-0 -z-20">
        {backdrop ? (
          <Image src={backdrop} alt="" fill sizes="100vw" className="hv-kenburns object-cover" />
        ) : (
          <div className="hv-sky h-full w-full" />
        )}
      </motion.div>
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-background via-background/10 to-background" aria-hidden="true" />
      <div className="absolute inset-0 -z-10 bg-gradient-to-r from-background/85 via-background/30 to-transparent" aria-hidden="true" />

      {render ? (
        <motion.div style={{ y: watchY }} className="absolute right-[-6%] top-[12%] -z-10 hidden aspect-square w-[min(52vw,47.5rem)] md:block">
          <Image src={render} alt="" fill sizes="52vw" className="hv-fade-y object-contain drop-shadow-[0_40px_60px_rgba(10,11,23,0.7)]" />
        </motion.div>
      ) : null}

      <div className="mx-auto flex min-h-[110svh] max-w-[77.5rem] flex-col justify-end px-4 pb-24 sm:px-6">
        <motion.div
          initial={reduced ? false : { opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl"
        >
          <h2 id="band-title" className="hv-headline text-[clamp(2.8rem,6vw,5rem)] leading-[0.98]">
            Five houses. One market.
          </h2>
          <p className="mt-5 text-[clamp(1.1rem,1.8vw,1.45rem)] font-semibold leading-[1.4] text-salmon/95">
            Patek Philippe, Audemars Piguet, A. Lange &amp; Söhne, Vacheron Constantin and Richard Mille. Every listing photo is
            screened before a buyer sees it, turned away with a reason when it fails, and set beside what the market is really
            asking.
          </p>
        </motion.div>
        <div className="mt-12 max-w-3xl">
          <HomeFigures />
        </div>
      </div>
    </section>
  );
}
