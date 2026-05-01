"use client";

import { useEffect, useRef } from "react";
import Matter from "matter-js";

import { GobblyText } from "@/components/motion/BouncyText";
import { LUXURY_SNEAKER_BRANDS, LUXURY_WATCH_BRANDS } from "@/lib/catalog";

const BRANDS = [
  ...LUXURY_SNEAKER_BRANDS.map((b) => ({ name: b, kind: "sneaker" as const })),
  ...LUXURY_WATCH_BRANDS.map((b)  => ({ name: b, kind: "watch"   as const })),
];

const COLOR_BG     = "#0B0118";
const COLOR_INK    = "#FFEDF6";
const COLOR_PINK   = "#FF1FA4";
const COLOR_ORANGE = "#FF7A1A";
const COLOR_CYAN   = "#00E1FF";
const COLOR_VIOLET = "#9B5BFF";
const COLOR_YELLOW = "#FFD24A";

/**
 * A full-width physics playground: brand pills drop from above, collide,
 * pile up, and can be dragged with the cursor. Intentionally chaotic.
 */
export function PhysicsBrandsCanvas({ height = 520 }: { height?: number }) {
  const wrapRef   = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const measureRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!wrapRef.current || !canvasRef.current || !measureRef.current) return;
    if (typeof window === "undefined") return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const wrapEl   = wrapRef.current;
    const canvasEl = canvasRef.current;
    const measure  = measureRef.current;

    const w = wrapEl.clientWidth;
    const h = height;

    canvasEl.width  = w * window.devicePixelRatio;
    canvasEl.height = h * window.devicePixelRatio;
    canvasEl.style.width  = `${w}px`;
    canvasEl.style.height = `${h}px`;

    /* ── Engine ─────────────────────────────────────────────── */
    const engine = Matter.Engine.create({
      gravity: { x: 0, y: 1.2 },
      enableSleeping: true,
    });
    engine.timing.timeScale = 1.0;

    /* ── Renderer ───────────────────────────────────────────── */
    const render = Matter.Render.create({
      canvas: canvasEl,
      engine,
      options: {
        width:  w,
        height: h,
        pixelRatio: window.devicePixelRatio,
        wireframes: false,
        background: COLOR_BG,
        showSleeping: false,
      },
    });

    /* ── Walls (invisible) ──────────────────────────────────── */
    const wallStyle = { isStatic: true, render: { visible: false } };
    Matter.Composite.add(engine.world, [
      Matter.Bodies.rectangle(w / 2,  h + 30, w + 100, 60, wallStyle), // floor
      Matter.Bodies.rectangle(-30,    h / 2, 60,        h * 2, wallStyle), // left
      Matter.Bodies.rectangle(w + 30, h / 2, 60,        h * 2, wallStyle), // right
    ]);

    /* ── Helper: measure pill width ──────────────────────────── */
    const measurePill = (text: string) => {
      measure.textContent = text;
      const rect = measure.getBoundingClientRect();
      const pad = 36;
      return { w: Math.ceil(rect.width + pad), h: 44 };
    };

    /* ── Brand pill bodies ───────────────────────────────────── */
    type Pill = { body: Matter.Body; text: string; bg: string; fg: string; };
    const pills: Pill[] = [];

    const palettes = [
      { bg: COLOR_PINK,   fg: "#FFFFFF" },
      { bg: COLOR_CYAN,   fg: "#0B0118" },
      { bg: COLOR_ORANGE, fg: "#0B0118" },
      { bg: COLOR_VIOLET, fg: "#FFFFFF" },
      { bg: COLOR_YELLOW, fg: "#0B0118" },
      { bg: "#1B0533",    fg: COLOR_PINK, stroke: true },
    ];

    // Dropping schedule — brands rain down one every ~120ms
    let dropIdx = 0;
    const dropTimer = window.setInterval(() => {
      if (dropIdx >= BRANDS.length) {
        window.clearInterval(dropTimer);
        return;
      }
      const b = BRANDS[dropIdx++];
      const palette = palettes[dropIdx % palettes.length];
      const dim = measurePill(b.name);
      const x   = 60 + Math.random() * (w - 120);

      const body = Matter.Bodies.rectangle(x, -50, dim.w, dim.h, {
        chamfer: { radius: dim.h / 2 },
        restitution: 0.45,
        friction:    0.05,
        frictionAir: 0.01,
        density:     0.0014,
        angle:       (Math.random() - 0.5) * 0.6,
        render: {
          fillStyle:    palette.bg,
          strokeStyle:  palette.stroke ? COLOR_INK : palette.bg,
          lineWidth:    palette.stroke ? 1.5 : 0,
        },
      });
      Matter.Composite.add(engine.world, body);
      pills.push({ body, text: b.name, bg: palette.bg, fg: palette.fg });
    }, 120);

    /* ── Mouse drag ──────────────────────────────────────────── */
    const mouse = Matter.Mouse.create(canvasEl);
    mouse.pixelRatio = window.devicePixelRatio;
    const mouseConstraint = Matter.MouseConstraint.create(engine, {
      mouse,
      constraint: {
        stiffness: 0.18,
        damping:   0.2,
        render: { visible: false },
      },
    });
    Matter.Composite.add(engine.world, mouseConstraint);
    render.mouse = mouse;

    // Allow page scroll over the canvas
    const wheelHandler = (e: WheelEvent) => {
      // forward wheel events to the page
      window.scrollBy({ top: e.deltaY, behavior: "auto" });
    };
    canvasEl.addEventListener("wheel", wheelHandler, { passive: true });

    /* ── Custom text overlay rendered after Matter ───────────── */
    Matter.Events.on(render, "afterRender", () => {
      const ctx = canvasEl.getContext("2d");
      if (!ctx) return;
      ctx.save();
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      ctx.font = "600 14px var(--font-display), system-ui, sans-serif";
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";

      pills.forEach(({ body, text, fg }) => {
        ctx.save();
        ctx.translate(body.position.x, body.position.y);
        ctx.rotate(body.angle);
        ctx.fillStyle = fg;
        ctx.fillText(text, 0, 1);
        ctx.restore();
      });
      ctx.restore();
    });

    /* ── Resize handling ─────────────────────────────────────── */
    const onResize = () => {
      const newW = wrapEl.clientWidth;
      canvasEl.width  = newW * window.devicePixelRatio;
      canvasEl.style.width = `${newW}px`;
      render.options.width = newW;
      render.canvas.width  = newW * window.devicePixelRatio;
    };
    window.addEventListener("resize", onResize);

    /* ── Start ───────────────────────────────────────────────── */
    Matter.Render.run(render);
    const runner = Matter.Runner.create();
    Matter.Runner.run(runner, engine);

    return () => {
      window.clearInterval(dropTimer);
      window.removeEventListener("resize", onResize);
      canvasEl.removeEventListener("wheel", wheelHandler);
      Matter.Render.stop(render);
      Matter.Runner.stop(runner);
      Matter.World.clear(engine.world, false);
      Matter.Engine.clear(engine);
      render.canvas.remove();
    };
  }, [height]);

  return (
    <div ref={wrapRef} className="relative w-full overflow-hidden" style={{ height }}>
      {/* Hidden span used to measure pill widths in canvas pixels */}
      <span
        ref={measureRef}
        aria-hidden
        className="invisible absolute font-[family-name:var(--font-display)] text-[14px] font-semibold"
        style={{ whiteSpace: "nowrap" }}
      />

      <canvas ref={canvasRef} className="block h-full w-full select-none" />

      {/* Top fade so pills appear to enter from "off-stage" */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-12 bg-gradient-to-b from-[#0B0118] to-transparent"
      />

      {/* Floor caption */}
      <div className="pointer-events-none absolute inset-x-0 bottom-3 flex items-center justify-center text-[10px] font-bold uppercase tracking-[0.32em] text-[#FF1FA4]">
        <span className="rounded-full border border-[rgba(255,31,164,0.4)] bg-[#0B0118]/80 px-4 py-1.5 backdrop-blur">
          <GobblyText text="DRAG · THEY HAVE WEIGHT" amount={2} duration={3} />
        </span>
      </div>
    </div>
  );
}
