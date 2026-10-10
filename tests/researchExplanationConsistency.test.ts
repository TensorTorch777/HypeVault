import assert from "node:assert/strict";
import test from "node:test";

import { matchingExplanation } from "../frontend/src/lib/researchExplanationFixture.ts";
import {
  INCONSISTENT_EXPLANATION_REASON,
  explanationForVerifiedResult,
} from "../frontend/src/lib/researchExplanationConsistency.ts";

const verified = { model: "LEGACY_DINOV2", decision: "FAKE" };

function discarded(explanation: typeof matchingExplanation) {
  const attached = explanationForVerifiedResult(verified, explanation);
  assert.equal(attached.status, "unavailable");
  assert.equal(attached.reason, INCONSISTENT_EXPLANATION_REASON);
  assert.equal(attached.sensitivity, undefined);
  assert.equal(attached.explanation, undefined);
  assert.equal(attached.publication_decision, "BLOCKED");
  assert.notEqual(attached, explanation);
}

test("a matching explanation is attached", () => {
  const attached = explanationForVerifiedResult(verified, matchingExplanation);
  assert.equal(attached, matchingExplanation);
  assert.equal(attached.status, "ok");
  assert.equal(attached.classification?.model, "LEGACY_DINOV2");
  assert.equal(attached.classification?.decision, "FAKE");
  assert.equal(attached.sensitivity?.model_id, "LEGACY_DINOV2");
  assert.equal(attached.sensitivity?.baseline_decision, "FAKE");
});

test("a model mismatch discards the explanation payload", () => {
  discarded({
    ...matchingExplanation,
    classification: {
      decision: "FAKE",
      model: "DINOV3_RESEARCH_PROTOTYPE",
      model_version: "1",
    },
  });
});

test("a decision mismatch discards the explanation payload", () => {
  discarded({
    ...matchingExplanation,
    classification: {
      decision: "AUTHENTIC",
      model: "LEGACY_DINOV2",
      model_version: "1",
    },
  });
});

test("a nested sensitivity model mismatch discards the explanation payload", () => {
  discarded({
    ...matchingExplanation,
    sensitivity: {
      ...matchingExplanation.sensitivity!,
      model_id: "DINOV3_RESEARCH_PROTOTYPE",
    },
  });
});

test("a nested sensitivity decision mismatch discards the explanation payload", () => {
  discarded({
    ...matchingExplanation,
    sensitivity: {
      ...matchingExplanation.sensitivity!,
      baseline_decision: "AUTHENTIC",
    },
  });
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
  assert.equal(attached.sensitivity, null);
});
