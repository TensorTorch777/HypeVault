"use client";

import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  className?: string;
  title: string;
  subtitle: string;
  hint?: string;
};

const BRUSH = 44;

/**
 * Scratch silver layer to reveal message (campaign-style interaction).
 */
export function ScratchReveal({ className = "", title, subtitle, hint = "Scratch to reveal" }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const [pct, setPct] = useState(0);
  const [done, setDone] = useState(false);

  const paintSilver = useCallback(() => {
    const canvas = ref.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const rect = wrap.getBoundingClientRect();
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const w = rect.width;
    const h = rect.height;
    if (w < 8 || h < 8) return;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    /* Darker metallic foil — light silver + semi-transparency read as “white on white” with the message */
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, "#4a4a4a");
    g.addColorStop(0.4, "#7a7a7a");
    g.addColorStop(0.55, "#9a9a9a");
    g.addColorStop(1, "#3d3d3d");
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    for (let i = 0; i < 120; i++) {
      ctx.fillRect(Math.random() * w, Math.random() * h, 1.2, 1.2);
    }
    ctx.strokeStyle = "rgba(0,0,0,0.22)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 24; i++) {
      ctx.beginPath();
      ctx.moveTo(Math.random() * w, Math.random() * h);
      ctx.lineTo(Math.random() * w, Math.random() * h);
      ctx.stroke();
    }
  }, []);

  useEffect(() => {
    paintSilver();
    const ro = new ResizeObserver(() => paintSilver());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [paintSilver]);

  const scratch = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = ref.current;
      const wrap = wrapRef.current;
      if (!canvas || !wrap) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const sx = (clientX - rect.left) * dpr;
      const sy = (clientY - rect.top) * dpr;
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.arc(sx, sy, BRUSH * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      try {
        if (canvas.width < 1 || canvas.height < 1) return;
        const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
        if (data.length === 0) return;
        let transparent = 0;
        for (let i = 3; i < data.length; i += 4) {
          if (data[i] < 32) transparent++;
        }
        const p = transparent / (data.length / 4);
        setPct(p);
        if (p > 0.38 && !done) setDone(true);
      } catch {
        /* canvas tainted or zero-sized in some browsers */
      }
    },
    [done]
  );

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.buttons !== 1 && e.pointerType !== "touch") return;
    scratch(e.clientX, e.clientY);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    scratch(e.clientX, e.clientY);
  };

  return (
    <motion.div
      initial={{ opacity: 1, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className={`relative overflow-hidden rounded-2xl border border-white/[0.14] bg-[#0c0c0c] shadow-[0_40px_100px_rgba(0,0,0,0.45),inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-sm ${className}`}
    >
      <div
        ref={wrapRef}
        className="relative w-full min-h-[min(72vw,22rem)] py-6 sm:min-h-[19rem] sm:py-8 md:min-h-[20.5rem] md:py-9 lg:min-h-[22rem]"
      >
        <div
          className="absolute inset-0 flex flex-col items-center justify-center overflow-y-auto bg-[#060606] px-4 text-center sm:px-8 md:px-12"
          style={{
            backgroundImage:
              "radial-gradient(ellipse 90% 70% at 50% -10%, rgba(255,59,0,0.2), transparent 55%), radial-gradient(ellipse 50% 40% at 100% 100%, rgba(255,255,255,0.06), transparent 55%)",
          }}
        >
          <div className="mx-auto max-w-xl rounded-2xl border border-white/12 bg-black/70 px-5 py-5 shadow-[0_12px_40px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-md sm:px-7 sm:py-6 md:max-w-2xl">
            <p className="text-[10px] font-bold uppercase tracking-[0.32em] text-accent sm:tracking-[0.38em]">
              The market spoke
            </p>
            <h3 className="mt-3 font-[family-name:var(--font-display)] text-xl font-extrabold leading-[1.12] tracking-tight text-white sm:mt-4 sm:text-2xl md:text-[1.85rem] lg:text-3xl">
              {title}
            </h3>
            <p className="mt-3 text-sm leading-relaxed text-white/92 sm:mt-4 md:text-base">
              {subtitle}
            </p>
          </div>
        </div>

        <canvas
          ref={ref}
          className="absolute inset-0 z-10 h-full w-full cursor-grab touch-none active:cursor-grabbing"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            try {
              e.currentTarget.releasePointerCapture(e.pointerId);
            } catch {
              /* ignore */
            }
          }}
        />

        <div className="pointer-events-none absolute bottom-2 left-0 right-0 z-20 flex justify-center px-3 sm:bottom-4 md:bottom-5">
          {!done ? (
            <span
              className="max-w-[min(100%,24rem)] rounded-full border-2 border-accent bg-[#030303] px-4 py-2.5 text-center text-[10px] font-bold uppercase leading-snug tracking-[0.18em] text-white shadow-[0_4px_24px_rgba(255,59,0,0.35),0_8px_32px_rgba(0,0,0,0.75)] sm:px-6 sm:text-[11px] sm:tracking-[0.22em] md:text-xs md:tracking-[0.24em]"
              style={{ textShadow: "0 1px 3px rgba(0,0,0,1)" }}
            >
              {hint}
            </span>
          ) : (
            <motion.span
              initial={{ opacity: 1, scale: 1 }}
              animate={{ opacity: 1, scale: 1 }}
              className="rounded-full border border-white/20 bg-accent/95 px-4 py-2 text-[10px] font-bold uppercase tracking-[0.22em] text-white shadow-[0_8px_32px_rgba(255,77,0,0.35)] backdrop-blur-md"
            >
              {Math.min(100, Math.round(pct * 100))}% revealed
            </motion.span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
