import type { ResearchExplanationPayload } from "@/components/ResearchExplanation";

export type ExplanationAttachment = ResearchExplanationPayload;

export const INCONSISTENT_EXPLANATION_REASON =
  "A consistent explanation could not be produced.";

export function explanationForVerifiedResult(
  verified: { model: string; decision: string },
  explanation: ResearchExplanationPayload,
): ResearchExplanationPayload {
  if (explanation.status === "unavailable") {
    return explanation;
  }
  const matches =
    explanation.status === "ok" &&
    explanation.classification?.model === verified.model &&
    explanation.classification?.decision === verified.decision &&
    explanation.sensitivity?.model_id === verified.model &&
    explanation.sensitivity?.baseline_decision === verified.decision;
  if (matches) {
    return explanation;
  }
  return {
    status: "unavailable",
    reason: INCONSISTENT_EXPLANATION_REASON,
    publication_decision: "BLOCKED",
  };
}
