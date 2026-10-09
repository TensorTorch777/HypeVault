"""Pure DINOv3 access decisions. This module does not load a model."""

from __future__ import annotations

from inference.research_guard import dinov3_research_allowed
from inference.scope_gate import CHECKPOINT_SHA256, POLICY_VERSION, SCOPE_VERSION, TEMPERATURE, failure_body

FROZEN_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
FROZEN_TEMPERATURE = TEMPERATURE


def mode_block_body() -> dict[str, str | None] | None:
    """Block every deployment state that is not research or shadow."""
    if dinov3_research_allowed():
        return None
    return failure_body("POLICY_ERROR", "DINOv3 candidate is blocked outside research mode.")


def checkpoint_block_body(digest: str | None) -> dict[str, str | None] | None:
    if digest != FROZEN_CHECKPOINT_SHA256:
        return failure_body("POLICY_ERROR", "DINOv3 checkpoint does not match the frozen artifact.")
    return None


def temperature_block_body(temperature: float | None) -> dict[str, str | None] | None:
    if temperature != FROZEN_TEMPERATURE:
        return failure_body("POLICY_ERROR", "DINOv3 temperature does not match the frozen value.")
    return None


def research_success_body(
    *,
    decision: str,
    declared_brand: str,
    checkpoint_sha: str,
    policy_version: str,
    model: str = "DINOV3_RESEARCH_PROTOTYPE",
    model_version: str = "1",
    model_scope: str = "FIVE_BRAND_RESEARCH_PROTOTYPE",
) -> dict[str, str | bool]:
    if decision not in {"AUTHENTIC", "FAKE", "REVIEW"}:
        raise ValueError("research success requires an in-scope decision")
    if model == "DINOV3_RESEARCH_PROTOTYPE":
        blocked = checkpoint_block_body(checkpoint_sha)
        if blocked is not None:
            raise ValueError("refusing a success body for a mismatched checkpoint")
    return {
        "status": decision,
        "decision": decision,
        "declared_brand": declared_brand,
        "brand_verification": "NOT_PERFORMED",
        "model_scope": model_scope,
        "research_only": True,
        "production_ready": False,
        "publication_decision": "BLOCKED",
        "checkpoint_sha": checkpoint_sha,
        "policy_version": policy_version,
        "model": model,
        "model_version": model_version,
    }


def log_research_event(
    logger,
    *,
    declared_brand: str | None,
    decision: str | None,
    status: str,
    checkpoint_sha: str | None,
    error: bool,
) -> None:
    """Metadata only. Image bytes are not logged."""
    from inference.research_guard import deployment_state

    logger.info(
        "dinov3_research_request mode=%s model=dinov3 checkpoint_sha=%s scope_version=%s "
        "policy_version=%s declared_brand=%s status=%s decision=%s error=%s",
        deployment_state(),
        checkpoint_sha,
        SCOPE_VERSION,
        POLICY_VERSION,
        declared_brand,
        status,
        decision,
        error,
    )
