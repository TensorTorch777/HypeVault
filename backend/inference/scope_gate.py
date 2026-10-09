"""Product-scope gate for the five-brand research prototype.

This module does not load a model, does not read an image, and does not
infer a brand from an authenticity score. A declared brand is either one of
the five supported names or it is outside the validated scope.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
TEMPERATURE = 0.24038200410185356
POLICY_VERSION = "shadow_v1"
SCOPE_VERSION = "scope_lock_v1"

SUPPORTED_BRANDS = (
    "A. Lange & Söhne",
    "Audemars Piguet",
    "Patek Philippe",
    "Richard Mille",
    "Vacheron Constantin",
)
UNSUPPORTED_REASON = "Brand is outside the validated five-brand research scope."

ScopeStatus = Literal["AUTHENTIC", "FAKE", "REVIEW", "UNSUPPORTED_SCOPE", "POLICY_ERROR"]


class ResearchDecision(BaseModel):
    """Documented response union.

    AUTHENTIC and FAKE come only from the frozen decision path after the
    gate has returned SUPPORTED. REVIEW and POLICY_ERROR belong to the
    shadow policy. This gate emits only UNSUPPORTED_SCOPE, with decision null.
    """

    status: ScopeStatus
    decision: Literal["AUTHENTIC", "FAKE", "REVIEW"] | None = None
    reason: str | None = None


class ScopeGateResult(BaseModel):
    scope_status: Literal["SUPPORTED", "UNSUPPORTED_SCOPE"]
    decision: None = None
    reason: str | None = None
    canonical_brand: str | None = None


def normalize_brand(value: str) -> str:
    text = value.strip().casefold().replace("ö", "o").replace("ß", "ss")
    text = " ".join(text.replace("&", " and ").split())
    return text


_SUPPORTED_KEYS = {normalize_brand(name): name for name in SUPPORTED_BRANDS}


def evaluate_declared_brand(brand: str | None) -> ScopeGateResult:
    """Fail closed unless the caller explicitly named a supported brand."""
    canonical = _SUPPORTED_KEYS.get(normalize_brand(brand or ""))
    if canonical is None:
        return ScopeGateResult(
            scope_status="UNSUPPORTED_SCOPE",
            decision=None,
            reason=UNSUPPORTED_REASON,
        )
    return ScopeGateResult(
        scope_status="SUPPORTED",
        decision=None,
        reason=None,
        canonical_brand=canonical,
    )


def unsupported_scope_body() -> dict[str, str | None]:
    result = ResearchDecision(
        status="UNSUPPORTED_SCOPE",
        decision=None,
        reason=UNSUPPORTED_REASON,
    )
    return result.model_dump() | {"publication_decision": "BLOCKED"}


def failure_body(status: str, reason: str) -> dict[str, str | None]:
    """Fail closed. Infrastructure and input errors are not authenticity verdicts."""
    if status not in {"INVALID_INPUT", "MODEL_ERROR", "POLICY_ERROR", "UNSUPPORTED_SCOPE"}:
        raise ValueError(f"unknown failure status {status}")
    return {"status": status, "decision": None, "reason": reason, "publication_decision": "BLOCKED"}


BRAND_DECLARATION_WARNING = (
    "The supplied brand is a user-declared scope parameter and is not independently verified from the image."
)


def log_scope_request(logger, *, declared_brand: str | None, decision: str | None, mode: str, status: str) -> None:
    """Legacy DINOv2 audit line. The DINOv3 checkpoint is not attached."""
    logger.info(
        "legacy_dinov2_request mode=%s model=LEGACY_DINOV2 declared_brand=%s "
        "brand_verification=NOT_PERFORMED research_candidate=false status=%s decision=%s",
        mode,
        declared_brand,
        status,
        decision,
    )
