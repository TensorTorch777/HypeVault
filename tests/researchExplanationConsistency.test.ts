import assert from "node:assert/strict";
import test from "node:test";

import {
  INCONSISTENT_EXPLANATION_REASON,
  explanationForVerifiedResult,
} from "../frontend/src/lib/researchExplanationConsistency.ts";

const verified = { model: "LEGACY_DINOV2", decision: "FAKE" };

const matching = {
  status: "ok" as const,
  classification: { decision: "FAKE", model: "LEGACY_DINOV2", model_version: "1" },
  sensitivity: { patches: [{ row: 0, col: 0, delta_logit: 0.2 }] },
  explanation: { observation: "The existing model classified this image as FAKE." },
  publication_decision: "BLOCKED",
};

test("a matching explanation is attached", () => {
  const attached = explanationForVerifiedResult(verified, matching);
  assert.equal(attached, matching);
  assert.equal(attached.status, "ok");
  assert.equal(attached.classification?.decision, "FAKE");
});

test("a model mismatch discards the explanation payload", () => {
  const attached = explanationForVerifiedResult(verified, {
    ...matching,
    classification: { decision: "FAKE", model: "DINOV3_RESEARCH_PROTOTYPE", model_version: "1" },
  });
  assert.equal(attached.status, "unavailable");
  assert.equal(attached.reason, INCONSISTENT_EXPLANATION_REASON);
  assert.equal(attached.sensitivity, undefined);
  assert.equal(attached.explanation, undefined);
  assert.equal(attached.publication_decision, "BLOCKED");
});

test("a decision mismatch discards the explanation payload", () => {
  const attached = explanationForVerifiedResult(verified, {
    ...matching,
    classification: { decision: "AUTHENTIC", model: "LEGACY_DINOV2", model_version: "1" },
  });
  assert.equal(attached.status, "unavailable");
  assert.equal(attached.reason, INCONSISTENT_EXPLANATION_REASON);
  assert.equal(attached.sensitivity, undefined);
  assert.equal(attached.explanation, undefined);
});

test("an unavailable explanation keeps its own reason", () => {
  const unavailable = {
    status: "unavailable" as const,
    reason: "The explanation timed out before every region was measured.",
    publication_decision: "BLOCKED",
    sensitivity: null,
    explanation: null,
  };
  const attached = explanationForVerifiedResult(verified, unavailable);
  assert.equal(attached, unavailable);
  assert.equal(attached.reason, "The explanation timed out before every region was measured.");
});
