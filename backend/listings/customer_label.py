"""Customer-facing listing labels.

`status` is publication state. `verdict` is stored evidence.
Neither one means the listing is authenticity-verified.
"""

from __future__ import annotations

LEGACY_SCREENING_LABEL = "Legacy screening — not verified"
DEMO_LISTING_LABEL = "Demo listing — not verified"
PENDING_LABEL = "Pending"
REJECTED_LABEL = "Rejected"
PUBLISHED_UNVERIFIED_LABEL = "Published — not verified"

SEED_CONFIDENCE = 0.965
CERTIFICATION_BADGES = (
    "Verified Authentic",
    "Authenticated",
    "Certified",
    "Guaranteed Authentic",
)


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value)


def is_seed_constant(
    *,
    verdict: str | None,
    confidence: float | None,
    s3_url: str | None,
) -> bool:
    """Historical seed rows stored a constant, not a model prediction."""
    if s3_url:
        return False
    if verdict != "AUTHENTIC":
        return False
    if confidence is None:
        return False
    return abs(float(confidence) - SEED_CONFIDENCE) <= 1e-9


def evidence_role(
    *,
    verdict: str | None,
    confidence: float | None,
    s3_url: str | None,
) -> str:
    if is_seed_constant(verdict=verdict, confidence=confidence, s3_url=s3_url):
        return "SEED_CONSTANT"
    if verdict in {"AUTHENTIC", "FAKE"}:
        return "HISTORICAL_MODEL_EVIDENCE"
    return "NONE"


def customer_label(
    *,
    status: object,
    verdict: str | None,
    confidence: float | None,
    s3_url: str | None,
) -> str:
    state = _status_value(status)
    if state == "pending":
        label = PENDING_LABEL
    elif state == "rejected":
        label = REJECTED_LABEL
    elif is_seed_constant(verdict=verdict, confidence=confidence, s3_url=s3_url):
        label = DEMO_LISTING_LABEL
    elif state == "live" and s3_url:
        label = LEGACY_SCREENING_LABEL
    elif state == "live":
        label = PUBLISHED_UNVERIFIED_LABEL
    else:
        label = PENDING_LABEL
    if any(phrase.casefold() in label.casefold() for phrase in CERTIFICATION_BADGES):
        raise RuntimeError("customer label must not certify authenticity")
    if label.casefold() == "authentic":
        raise RuntimeError("customer label must not certify authenticity")
    return label
