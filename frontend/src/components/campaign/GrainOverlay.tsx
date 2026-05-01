"use client";

/**
 * Subtle animated film grain + paper texture (campaign sites).
 */
export function GrainOverlay() {
  return (
    <div className="pointer-events-none fixed inset-0 z-[5] overflow-hidden" aria-hidden>
      <div
        className="hv-grain absolute inset-[-100%] h-[300%] w-[300%] animate-hvGrain opacity-[0.22]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.78' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
          backgroundRepeat: "repeat",
          backgroundSize: "180px 180px",
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-br from-accent/[0.04] via-transparent to-white/[0.03]" />
    </div>
  );
}
