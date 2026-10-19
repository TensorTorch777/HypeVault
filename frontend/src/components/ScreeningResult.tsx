"use client";

import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

export type ScreeningOutcome = {
  listing_status: "live" | "rejected" | "pending";
  screening_code: string | null;
  screening_reason: string | null;
};

const CODE_TITLE: Record<string, string> = {
  NOT_A_WATCH: "No watch found",
  IMAGE_TOO_SMALL: "Photo too small",
  LIKELY_GENERATED: "Looks AI-generated",
  NOT_CONFIDENTLY_ORIGINAL: "Couldn't confirm an original photo",
};

/** Seller-facing result of listing screening: listed, or rejected with the reason. */
export function ScreeningResult({ outcome }: { outcome: ScreeningOutcome }) {
  const reduced = useReducedMotion();
  const listed = outcome.listing_status === "live";
  const Icon = listed ? CheckCircle2 : XCircle;
  const tone = listed ? "#3EF2A6" : "#FF5A5F";
  const title = listed ? "Listed" : outcome.listing_status === "pending" ? "Held — not listed" : "Not listed";
  const subtitle = listed ? "Passed photo screening" : (outcome.screening_code && CODE_TITLE[outcome.screening_code]) || "Screening stopped this photo";

  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 24 }}
      className="hv-hud p-6"
      style={{ borderColor: `${tone}66` }}
      role="status"
    >
      <div className="flex items-start gap-4">
        <Icon className="mt-1 h-8 w-8 shrink-0" style={{ color: tone }} strokeWidth={1.75} aria-hidden="true" />
        <div className="min-w-0">
          <p className="hv-mono text-[11px] uppercase tracking-[0.18em]" style={{ color: tone }}>
            {subtitle}
          </p>
          <p className="hv-display mt-1 text-[2.2rem] uppercase leading-none text-primary">{title}</p>
          {outcome.screening_reason ? <p className="mt-4 text-[15px] leading-7 text-primary/85">{outcome.screening_reason}</p> : null}
          <p className="mt-4 text-[12px] leading-5 text-primary/55">
            {listed
              ? "Buyers can now see this listing. Screening checks the photo, not the watch: it is not a certificate of authenticity."
              : "Buyers cannot see this listing. Upload a new photo to screen it again; the same listing is reused."}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
