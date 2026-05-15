"use client";

import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

export function AuthBadge({
  verdict,
  confidence,
}: {
  verdict: "AUTHENTIC" | "FAKE" | null;
  confidence: number | null;
}) {
  const pct = confidence == null ? null : Math.min(100, Math.max(0, confidence * 100));
  const isAuthentic = verdict === "AUTHENTIC";

  if (!verdict) {
    return (
      <div className="rounded-card border border-primary/10 bg-card p-6 text-primary/60">
        <p className="text-sm font-semibold">Verification pending</p>
        <p className="mt-2 text-sm text-primary/55">Upload images to run AI verification.</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 22 }}
      className={
        isAuthentic
          ? "rounded-card border border-success/25 bg-success/10 p-6 text-primary"
          : "rounded-card border border-danger/25 bg-danger/10 p-6 text-primary"
      }
    >
      <div className="flex items-start gap-4">
        {isAuthentic ? (
          <CheckCircle2 className="h-12 w-12 text-success" aria-hidden />
        ) : (
          <XCircle className="h-12 w-12 text-danger" aria-hidden />
        )}
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-primary/55">AI verdict</p>
          <p className={`mt-2 text-2xl font-semibold tracking-tight ${isAuthentic ? "text-success" : "text-danger"}`}>
            {isAuthentic ? "AI Verified Authentic" : "AI detected — fake / rejected"}
          </p>
          {pct != null && (
            <p className="mt-2 text-sm text-primary/65">
              <span className="font-semibold text-primary">{pct.toFixed(1)}%</span>{" "}
              {isAuthentic ? "authentic confidence" : "confidence (fake side)"}
            </p>
          )}
          {!isAuthentic ? (
            <p className="mt-2 text-xs text-primary/50">
              Listings below the server&apos;s minimum authentic-confidence threshold are classified as fake and not
              published.
            </p>
          ) : null}
        </div>
      </div>
    </motion.div>
  );
}
