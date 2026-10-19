"""Authenticity governance. No model result is an authenticity certificate.

AUTHENTICITY_MODEL_PRODUCTION_APPROVED is an explicit governance flag. It is not
inferred from HTTP availability, deployment mode, Triton, or scores, and while it is
False nothing may be presented as authenticity-verified.

Whether a listing goes live is a separate, narrower question answered by photo
screening (inference.screening): is it a watch, and does the photo look original?
A screened listing is labelled as screened, never as authentic.
"""

from __future__ import annotations

AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False
PUBLICATION_BLOCKED = "BLOCKED"


def can_publish_listing(
    *,
    model: str | None,
    model_status: str | None,
    production_validation: str | None,
    model_decision: str | None,
    policy_status: str | None,
    listing_state: str | None,
) -> bool:
    """Return whether this request may set a listing live.

    Missing metadata, a research model, an unvalidated legacy model, a fake
    or error decision, and the explicit production-approval flag all refuse.
    """
    if not AUTHENTICITY_MODEL_PRODUCTION_APPROVED:
        return False
    if model_status != "PRODUCTION_APPROVED":
        return False
    if production_validation != "ESTABLISHED":
        return False
    if policy_status != "OK":
        return False
    if model_decision != "AUTHENTIC":
        return False
    if model in {None, "", "LEGACY_DINOV2", "DINOV3_RESEARCH_PROTOTYPE"}:
        return False
    if listing_state == "rejected":
        return False
    return False


def publication_decision(**kwargs: str | None) -> str:
    """A publication result is never an authenticity class."""
    allowed = can_publish_listing(**kwargs)
    if allowed:
        raise RuntimeError("automatic publication is not implemented")
    return PUBLICATION_BLOCKED

