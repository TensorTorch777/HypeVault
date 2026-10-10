import type { ResearchExplanationPayload } from "@/components/ResearchExplanation";

/** Schema-valid success payload for the consistency tests. Not used in production. */
export const matchingExplanation: ResearchExplanationPayload = {
  status: "ok",
  research_only: true,
  publication_decision: "BLOCKED",
  independent_authentication: false,
  classification: {
    decision: "FAKE",
    model: "LEGACY_DINOV2",
    model_version: "1",
  },
  sensitivity: {
    method: {
      name: "region_occlusion_v1",
      version: "1",
      weak_abs_delta: 0.001,
      weak_abs_delta_role: "provisional_heuristic",
    },
    model_id: "LEGACY_DINOV2",
    model_version: "1",
    preprocessing_id: "legacy_square_resize_504_imagenet",
    baseline_logit: 1.25,
    baseline_decision: "FAKE",
    max_abs_delta_logit: 0.2,
    evidence_supports_visual_summary: true,
    patches: [
      {
        row: 0,
        col: 0,
        x: 0,
        y: 0,
        width: 8,
        height: 8,
        delta_logit: 0.2,
      },
    ],
  },
  explanation: {
    observation: "The existing model classified this image as FAKE.",
    hypothesis: "No separate hypothesis is offered. Only the measured score changes are reported.",
    limitations: "A sensitivity map is not proof of authenticity.",
    source: "deterministic_fallback",
  },
};
