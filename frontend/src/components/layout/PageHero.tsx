import Image from "next/image";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Cinematic page header: a generated dusk backdrop fading into the navy page, a condensed
 * deco title and an optional studio render. Every inner page opens with one.
 */
export function PageHero({
  eyebrow,
  title,
  subtitle,
  scene = "bay_dusk",
  render,
  actions,
  compact = false,
}: {
  eyebrow: string;
  title: ReactNode;
  subtitle?: ReactNode;
  scene?: string;
  render?: string;
  actions?: ReactNode;
  compact?: boolean;
}) {
  return (
    <header className={cn("relative isolate overflow-hidden", compact ? "min-h-[300px]" : "min-h-[380px] md:min-h-[440px]")}>
      <div className="hv-sky absolute inset-0 -z-20" aria-hidden="true" />
      <Image src={`/renders/backdrops/${scene}_0.webp`} alt="" fill priority sizes="100vw" className="hv-kenburns -z-20 object-cover" />
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-background/80 via-background/35 to-background" aria-hidden="true" />
      <div className="absolute inset-0 -z-10 bg-gradient-to-r from-background/80 via-background/20 to-transparent" aria-hidden="true" />
      {render ? (
        <div
          className={cn(
            "absolute right-[3%] -z-10 hidden aspect-square md:block",
            compact ? "bottom-[-14%] w-[min(24vw,300px)]" : "bottom-[-12%] w-[min(34vw,430px)]"
          )}
          aria-hidden="true"
        >
          <Image src={render} alt="" fill sizes="36vw" className="hv-fade-y object-contain drop-shadow-[0_30px_50px_rgba(10,11,23,0.7)]" />
        </div>
      ) : null}
      <div className={cn("mx-auto flex max-w-[1180px] flex-col justify-end px-4 pb-10 pt-28 sm:px-6", compact ? "min-h-[300px]" : "min-h-[380px] md:min-h-[440px]")}>
        <p className="hv-eyebrow">{eyebrow}</p>
        <h1 className="mt-1 font-display text-[clamp(2.8rem,6vw,5rem)] font-extrabold uppercase leading-[0.88] text-white">{title}</h1>
        {subtitle ? <div className="mt-4 max-w-xl text-[15px] leading-7 text-white/80 md:text-[16px]">{subtitle}</div> : null}
        {actions ? <div className="mt-6 flex flex-wrap gap-3">{actions}</div> : null}
      </div>
    </header>
  );
}
