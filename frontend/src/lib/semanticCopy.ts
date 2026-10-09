/** User-facing distinctions. These strings do not change model or publication decisions. */

export type FailureKind = "access" | "scope" | "invalid" | "unavailable" | "network" | "inference";

export function classifyHttpFailure(
  status: number | undefined,
  bodyStatus?: string | null,
): FailureKind {
  if (status === 401 || status === 403) return "access";
  if (bodyStatus === "UNSUPPORTED_SCOPE") return "scope";
  if (bodyStatus === "INVALID_INPUT" || status === 400) return "invalid";
  if (bodyStatus === "MODEL_ERROR" || bodyStatus === "POLICY_ERROR" || status === 503) return "unavailable";
  if (status == null) return "network";
  return "inference";
}

export function researchFailureMessage(kind: FailureKind): string {
  switch (kind) {
    case "access":
      return "Research access is not enabled for this account or session. No screening result was produced.";
    case "scope":
      return "Outside current research scope — no result produced. This is not a fake or counterfeit verdict.";
    case "invalid":
      return "The image could not be processed. Use a JPEG, PNG, or WebP file. No authenticity result was produced.";
    case "unavailable":
      return "Screening is temporarily unavailable. No result was produced, and no other model was used.";
    case "network":
      return "Could not reach the screening service. No result was produced.";
    default:
      return "The selected model did not return a classification. No authenticity result was produced.";
  }
}

export type ComparisonView = "missing" | "error" | "empty" | "data";

export function comparisonView(data: {
  error?: string;
  stockx?: unknown[];
  chrono24?: unknown[];
  ebay?: unknown[];
} | null | undefined): ComparisonView {
  if (!data) return "missing";
  if (typeof data.error === "string" && data.error.length > 0) return "error";
  const count = (data.stockx?.length ?? 0) + (data.chrono24?.length ?? 0) + (data.ebay?.length ?? 0);
  return count === 0 ? "empty" : "data";
}

export const SERVING_STATUS_LABEL = "Serving status";
export const VALIDITY_NOT_ESTABLISHED = "Real-world authenticity validity: not established.";
export const ILLUSTRATIVE_PRICE_LABEL = "Illustrative sample data — not live market prices.";
export const ILLUSTRATIVE_SCAN_LABEL = "Illustration only — not a live scan or model-generated localization.";
export const DEMO_INVENTORY_LABEL = "Demo examples — not live inventory";
export const DECLARED_BRAND_NOTE =
  "Declared brand — selected by you, not verified from the image.";
