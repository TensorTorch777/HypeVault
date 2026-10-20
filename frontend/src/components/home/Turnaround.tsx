"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useMotionValueEvent, useReducedMotion, useScroll, useTransform } from "framer-motion";

import { cn } from "@/lib/utils";

export type TurnaroundWatch = { id: string; brand: string; piece: string; frames: number; hero: string | null };

const CHAPTERS = [
  {
    title: "Is it a watch?",
    body: "Every upload is checked for a wristwatch first. A selfie, a sneaker or a screenshot is sent back to the seller with the reason.",
  },
  {
    title: "Is it a real photograph?",
    body: "A self-supervised vision transformer looks for the traces image generators leave behind. Generated or heavily edited photos are turned away.",
  },
  {
    title: "What is it worth?",
    body: "Listed watches sit beside real asking prices from the eBay Browse API and a dated Chrono24 dataset, each labelled with where and when it was seen.",
  },
];

/** Loads a rendered turntable and draws the frame matching scroll progress onto a canvas. */
function useSequence(watch: TurnaroundWatch, enabled: boolean) {
  const frames = useRef<HTMLImageElement[]>([]);
  const [loaded, setLoaded] = useState(0);
  useEffect(() => {
    if (!enabled || watch.frames === 0) return;
    let cancelled = false;
    setLoaded(0);
    frames.current = Array.from({ length: watch.frames }, (_, i) => {
      const img = new window.Image();
      img.decoding = "async";
      img.src = `/renders/${watch.id}/turntable/f_${String(i).padStart(3, "0")}.webp`;
      img.onload = () => !cancelled && setLoaded((n) => n + 1);
      return img;
    });
    return () => {
      cancelled = true;
    };
  }, [watch.id, watch.frames, enabled]);
  return { frames, loaded };
}

export function Turnaround({ watches }: { watches: TurnaroundWatch[] }) {
  const ref = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();
  const [selected, setSelected] = useState(0);
  const [near, setNear] = useState(false);
  const watch = watches[selected];
  const { frames, loaded } = useSequence(watch, near);
  const { scrollYProgress: p } = useScroll({ target: ref, offset: ["start start", "end end"] });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setNear(true), { rootMargin: "800px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const draw = (progress: number) => {
    const c = canvas.current;
    const list = frames.current;
    if (!c || list.length === 0) return;
    const target = Math.min(list.length - 1, Math.round((reduced ? 0.08 : progress) * (list.length - 1) * 1.0));
    // nearest frame that has finished loading
    let img: HTMLImageElement | null = null;
    for (let d = 0; d < list.length && !img; d++) {
      for (const k of [target - d, target + d]) {
        const f = list[k];
        if (f && f.complete && f.naturalWidth) {
          img = f;
          break;
        }
      }
    }
    if (!img) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth * dpr;
    const h = c.clientHeight * dpr;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    const s = Math.min(w / img.naturalWidth, h / img.naturalHeight);
    const dw = img.naturalWidth * s;
    const dh = img.naturalHeight * s;
    ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  };

  useMotionValueEvent(p, "change", (v) => draw(v));
  useEffect(() => {
    draw(p.get());
    const onResize = () => draw(p.get());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, selected]);

  const [chapter, setChapter] = useState(0);
  useMotionValueEvent(p, "change", (v) => setChapter(Math.min(CHAPTERS.length - 1, Math.floor(v * CHAPTERS.length))));
  const glow = useTransform(p, [0, 0.5, 1], [0.35, 0.6, 0.35]);
  const hasSequence = watch.frames > 0;

  return (
    <section ref={ref} aria-labelledby="turnaround-title" className="relative h-[420vh]">
      <div className="sticky top-0 flex h-[100svh] flex-col overflow-hidden">
        <motion.div
          style={{ opacity: glow }}
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_45%_55%_at_50%_55%,rgba(240,138,168,0.35),rgba(185,163,240,0.12)_55%,transparent_75%)]"
          aria-hidden="true"
        />
        <div className="relative z-10 pt-24 text-center">
          <p className="font-display text-[0.8125rem] font-bold uppercase tracking-[0.08em] text-primary/70">Watch</p>
          <h2 id="turnaround-title" className="hv-wide mt-1 text-[clamp(1.6rem,3.6vw,3rem)] text-primary">
            The turnaround
          </h2>
          <div className="mt-5 flex flex-wrap justify-center gap-2 px-4" role="tablist" aria-label="Choose a watch">
            {watches.map((w, i) => (
              <button
                key={w.id}
                type="button"
                role="tab"
                aria-selected={i === selected}
                onClick={() => setSelected(i)}
                className={cn(
                  "min-h-[2.25rem] cursor-pointer rounded-full border px-3.5 text-[0.75rem] font-semibold transition-colors",
                  i === selected ? "border-accent bg-accent text-ink" : "border-primary/20 text-primary/75 hover:border-primary/50 hover:text-primary"
                )}
              >
                {w.brand}
              </button>
            ))}
          </div>
        </div>

        <div className="relative flex-1">
          {hasSequence ? (
            <canvas ref={canvas} className="hv-fade-y absolute inset-0 h-full w-full" aria-label={`${watch.brand} ${watch.piece}, studio render turning`} role="img" />
          ) : watch.hero ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={watch.hero} alt={`${watch.brand} ${watch.piece}, studio render`} className="hv-fade-y absolute inset-0 m-auto h-full max-h-full w-auto object-contain" />
          ) : null}
          {hasSequence && loaded < watch.frames ? (
            <p className="hv-mono absolute bottom-3 left-1/2 -translate-x-1/2 text-[0.625rem] text-primary/45">
              Loading frames {loaded}/{watch.frames}
            </p>
          ) : null}
        </div>

        <div className="relative z-10 mx-auto grid w-full max-w-[77.5rem] gap-6 px-4 pb-10 sm:px-6 md:grid-cols-[1fr_auto] md:items-end">
          <div className="min-h-[9.375rem] max-w-xl">
            <motion.div key={chapter} initial={reduced ? false : { opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
              <p className="hv-headline text-[clamp(1.8rem,3.4vw,2.8rem)] leading-[1.05]">{CHAPTERS[chapter].title}</p>
              <p className="mt-3 text-[1rem] leading-7 text-salmon/90 md:text-[1.125rem]">{CHAPTERS[chapter].body}</p>
            </motion.div>
          </div>
          <div className="flex items-center gap-3 md:flex-col md:items-end" aria-hidden="true">
            {CHAPTERS.map((c, i) => (
              <span key={c.title} className={cn("h-1.5 rounded-full transition-all duration-500", i === chapter ? "w-10 bg-accent" : "w-4 bg-primary/25")} />
            ))}
          </div>
        </div>
        <p className="hv-mono absolute bottom-3 right-4 text-[0.5625rem] text-primary/35">
          {watch.piece} · studio model, no logos · rendered in Blender Cycles
        </p>
      </div>
    </section>
  );
}
