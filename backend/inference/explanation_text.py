"""Language layer over structured occlusion evidence.

The narrator may only restate measured logit changes and supplied metadata.
It does not see the customer image, change the classifier result, or decide publication.
"""

from __future__ import annotations

import os
import re
from typing import Any

FORBIDDEN_CLAIM = re.compile(
    r"serial|bezel|dial|movement|logo|provenance|craftsmanship|material|manufactur|"
    r"counterfeit defect|genuinely authentic|probability that the watch|"
    r"proves authentic|proves fake|publish",
    re.IGNORECASE,
)
LIMITATIONS = (
    "A sensitivity map shows which masked regions changed this model's raw logit. "
    "It is not proof of authenticity, a causal guarantee, or a validated counterfeit indicator. "
    "It does not identify physical watch components. "
    "The model score is not, by itself, the probability that the watch is genuinely authentic. "
    "This research result cannot decide publication."
)


def _score_note(model_id: str) -> str:
    if model_id == "DINOV3_RESEARCH_PROTOTYPE":
        return (
            "The stored DINOv3 score is a calibrated P(fake) from sigmoid(logit / frozen temperature). "
            "It is not the probability that the watch is genuinely authentic."
        )
    return (
        "The legacy DINOv2 class comes from the existing logit policy. "
        "The raw logit change below is not a calibrated probability that the watch is authentic."
    )


def deterministic_explanation(evidence: dict[str, Any]) -> dict[str, Any]:
    """Fallback that quotes only fields present in the sensitivity record."""
    decision = str(evidence["baseline_decision"])
    model_id = str(evidence["model_id"])
    if not evidence.get("evidence_supports_visual_summary"):
        observation = (
            f"The existing model classified this image as {decision}. "
            "Masking the measured regions did not change the raw logit enough to support a specific visual explanation."
        )
    else:
        ranked = sorted(evidence["patches"], key=lambda item: abs(float(item["delta_logit"])), reverse=True)[:3]
        parts = [
            f"row {item['row']}, column {item['col']} changed the raw logit by {float(item['delta_logit']):+.4f}"
            for item in ranked
        ]
        observation = (
            f"The existing model classified this image as {decision}. "
            "The largest measured raw-logit changes were: " + "; ".join(parts) + ". "
            "These are regions where masking changed the model score. They are not identified watch parts."
        )
    accepted = validate_explanation(
        {
            "observation": observation,
            "hypothesis": "No separate hypothesis is offered. Only the measured score changes are reported.",
            "source": "deterministic_fallback",
            "model_id": model_id,
            "baseline_decision": decision,
        },
        evidence,
    )
    accepted["limitations"] = LIMITATIONS + " " + _score_note(model_id)
    return accepted


def validate_explanation(payload: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Reject text that overrides the class, names parts, or leaves the schema."""
    required = ("observation", "hypothesis", "source", "model_id", "baseline_decision")
    if any(key not in payload or not str(payload[key]).strip() for key in required):
        raise ValueError("Explanation payload is missing required text")
    if payload["baseline_decision"] != evidence["baseline_decision"]:
        raise ValueError("Explanation changed the classifier result")
    if payload["model_id"] != evidence["model_id"]:
        raise ValueError("Explanation model id does not match the evidence")
    blob = f"{payload['observation']} {payload['hypothesis']}"
    if FORBIDDEN_CLAIM.search(blob):
        raise ValueError("Explanation contains an unsupported claim")
    if evidence["baseline_decision"] not in payload["observation"]:
        raise ValueError("Explanation omitted the model class")
    accepted = {key: str(payload[key]) for key in required}
    accepted["limitations"] = LIMITATIONS + " " + _score_note(str(evidence["model_id"]))
    return accepted


def explanation_llm_configured() -> bool:
    return bool(os.environ.get("HYPEVAULT_EXPLANATION_LLM_URL", "").strip())


async def narrate(evidence: dict[str, Any], generator=None) -> dict[str, Any]:
    """Use a supplied generator when configured. Invalid or failed output falls back."""
    if generator is None or not explanation_llm_configured():
        text = deterministic_explanation(evidence)
        text["llm_status"] = "not_configured" if generator is None else "skipped"
        return text
    try:
        candidate = await generator(evidence)
        accepted = validate_explanation(candidate, evidence)
        accepted["llm_status"] = "accepted"
        return accepted
    except Exception as exc:
        text = deterministic_explanation(evidence)
        text["llm_status"] = "rejected"
        text["llm_error"] = exc.__class__.__name__
        return text
