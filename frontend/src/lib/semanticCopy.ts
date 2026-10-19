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

export const SERVING_STATUS_LABEL = "Serving status";
export const VALIDITY_NOT_ESTABLISHED = "Real-world authenticity validity: not established.";
export const DECLARED_BRAND_NOTE =
  "Declared brand — selected by you, not verified from the image.";
