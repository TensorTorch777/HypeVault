import { useId } from "react";

import { cn } from "@/lib/utils";

/**
 * HypeVault lockup: a painted-sunset "HV" monogram with the name stacked over its corner.
 * Drawn as SVG so the dark keyline sits behind the letters (paint-order) in every browser.
 */
export function Lockup({ className, compact = false }: { className?: string; compact?: boolean }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg
      viewBox="0 0 330 210"
      className={cn("h-auto select-none", compact ? "w-[9.375rem]" : "w-[clamp(13.75rem,30vw,25rem)]", className)}
      role="img"
      aria-label="HypeVault"
    >
      <defs>
        <linearGradient id={`dusk-${id}`} x1="0" y1="0" x2="0.25" y2="1">
          <stop offset="0" stopColor="#FFD9A0" />
          <stop offset="0.3" stopColor="#FF9E8A" />
          <stop offset="0.58" stopColor="#F27AA6" />
          <stop offset="0.85" stopColor="#A46BD6" />
          <stop offset="1" stopColor="#6E58C8" />
        </linearGradient>
        <filter id={`shadow-${id}`} x="-10%" y="-10%" width="120%" height="130%">
          <feDropShadow dx="0" dy="8" stdDeviation="10" floodColor="#14112A" floodOpacity="0.55" />
        </filter>
      </defs>
      <text
        x="4"
        y="196"
        fill={`url(#dusk-${id})`}
        filter={`url(#shadow-${id})`}
        style={{ fontFamily: "var(--font-display), 'Big Shoulders Display', sans-serif", fontWeight: 900, fontSize: 236, letterSpacing: "-6px" }}
      >
        HV
      </text>
      <g
        fill="#FFFFFF"
        stroke="#14112A"
        strokeWidth="11"
        strokeLinejoin="round"
        paintOrder="stroke"
        style={{ fontFamily: "var(--font-wide), 'Josefin Sans', sans-serif", fontWeight: 700, fontSize: 58, letterSpacing: "-1.5px" }}
      >
        <text x="126" y="140">hype</text>
        <text x="126" y="190">vault</text>
      </g>
    </svg>
  );
}

/** Small monogram for the header, like the corner "VI". */
export function Monogram({ className }: { className?: string }) {
  return (
    <span className={cn("hv-vi font-display text-[1.875rem] font-black leading-none tracking-[-0.03em]", className)} aria-hidden="true">
      HV
    </span>
  );
}
