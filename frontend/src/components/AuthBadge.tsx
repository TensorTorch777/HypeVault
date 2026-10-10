"use client";

import { motion } from "framer-motion";

export function AuthBadge({
  verdict,
  confidence,
  declaredBrand,
  lane = "legacy",
}: {
  verdict: "AUTHENTIC" | "FAKE" | "REVIEW" | null;
  confidence: number | null;
  declaredBrand?: string | null;
  lane?: "legacy" | "research";
}) {
  const pct = confidence == null ? null : Math.min(100, Math.max(0, confidence * 100));
  const workflow =
    lane === "research"
      ? "No listing was published. This is not a marketplace status."
      : verdict === "FAKE"
        ? "Listing workflow: Rejected by the legacy screening workflow. That is not independent proof of counterfeit status."
        : "Listing workflow: pending. The model result does not publish the listing or verify authenticity.";

  if (!verdict) {
    return (
      <div className="rounded-card border border-primary/10 bg-card p-6 text-primary/60">
        <p className="text-sm font-semibold">{lane === "research" ? "Research demo pending" : "Pending"}</p>
        <p className="mt-2 text-sm text-primary/55">
          {lane === "research"
            ? "DINOv3 research prototype. Not production. The brand is user-declared."
            : "Legacy DINOv2 listing check. Not the DINOv3 research candidate."}
        </p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 22 }}
      className="rounded-card border border-primary/15 bg-card p-6 text-primary"
    >
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-wider text-primary/55">
          {lane === "research" ? "Research result — not verified" : "Legacy DINOv2 screening result — not verified"}
        </p>
        <p className="mt-2 text-2xl font-semibold tracking-tight">Model classification: {verdict}</p>
        <p className="mt-2 text-sm text-primary/70">{workflow}</p>
        <p className="mt-2 text-sm text-primary/70">
          {lane === "research"
            ? "Five-brand research prototype. This classification cannot publish a listing. DINOv3 is not approved for production."
            : "A model classification is stored as evidence. It does not authorize marketplace publication."}
        </p>
        <p className="mt-1 text-sm text-primary/70">
          Declared brand: {declaredBrand?.trim() ? declaredBrand : "not supplied"}
        </p>
        <p className="mt-1 text-sm text-primary/70">Brand verification: not independently verified</p>
        {pct != null && (
          <p className="mt-2 text-sm text-primary/65">
            <span className="font-semibold text-primary">{pct.toFixed(1)}%</span> historical model score
          </p>
        )}
        <p className="mt-2 text-xs text-primary/50">
          The selected brand is user-declared and is not independently verified from the image. A model result is not marketplace approval and is not a guarantee of authenticity.
        </p>
      </div>
    </motion.div>
  );
}
