"use client";

import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2, Clock3, XCircle } from "lucide-react";

export type ScreeningOutcome = {
  listing_status: "live" | "rejected" | "pending";
  screening_code: string | null;
  screening_reason: string | null;
  /** Set when dial OCR could not read the declared brand's name. Informational only. */
  seller_hint?: string | null;
};

const CODE_TITLE: Record<string, string> = {
  NOT_A_WATCH: "No watch found",
  IMAGE_TOO_SMALL: "Photo too small",
  LIKELY_GENERATED: "Looks AI-generated",
  AI_LABEL: "Labelled as AI-made",
  NOT_CONFIDENTLY_ORIGINAL: "Couldn't confirm an original photo",
  KNOWN_AI_MATCH: "Copy of a known AI image",
  NEEDS_PROOF: "Proof photo needed",
  PROOF_SUBMITTED: "Waiting for an admin check",
  REVIEW_REJECTED: "Not approved by an admin",
  REVIEW_APPROVED: "Approved after an admin check",
};

/** Seller-facing result of listing screening: listed, or rejected with the reason. */
export function ScreeningResult({ outcome }: { outcome: ScreeningOutcome }) {
  const reduced = useReducedMotion();
  const listed = outcome.listing_status === "live";
  const held = outcome.listing_status === "pending";
  const Icon = listed ? CheckCircle2 : held ? Clock3 : XCircle;
  const tone = listed ? "#3EF2A6" : held ? "#F5C451" : "#FF5A5F";
  const title = listed ? "Listed" : held ? "Held — not listed yet" : "Not listed";
  const subtitle =
    (outcome.screening_code && CODE_TITLE[outcome.screening_code]) ||
    (listed ? "Passed photo screening" : "Screening stopped this photo");

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
          <p className="hv-mono text-[0.6875rem] uppercase tracking-[0.18em]" style={{ color: tone }}>
            {subtitle}
          </p>
          <p className="hv-display mt-1 text-[2.2rem] uppercase leading-none text-primary">{title}</p>
          {outcome.screening_reason ? <p className="mt-4 text-[0.9375rem] leading-7 text-primary/85">{outcome.screening_reason}</p> : null}
          {outcome.seller_hint ? (
            <p className="mt-4 rounded-lg border border-[#F5C451]/40 bg-[#F5C451]/10 px-3 py-2 text-[0.8125rem] leading-6 text-primary/85">
              {outcome.seller_hint}
            </p>
          ) : null}
          <p className="mt-4 text-[0.75rem] leading-5 text-primary/55">
            {listed
              ? "Buyers can now see this listing. Screening checks the photo, not the watch: it is not a certificate of authenticity."
              : held
                ? "Buyers cannot see this listing until an admin approves it."
                : "Buyers cannot see this listing. Upload a new photo to screen it again; the same listing is reused."}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
