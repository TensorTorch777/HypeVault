"""Inference request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthenticateResponse(BaseModel):
    """Listing screening result (watch check + legacy DINOv2 origin check).

    `listing_status` and `screening_reason` say whether the listing went live or was
    rejected, and why. `verdict` is null when the photo was stopped before DINOv2 ran
    (too small, or no wristwatch found). Going live is a screening result, never an
    authenticity certificate.

    This is not the frozen DINOv3 research candidate. `declared_brand` is the
    caller's scope parameter. This schema has no `verified_brand` field and
    does not carry the DINOv3 checkpoint or temperature.
    """

    model_config = ConfigDict(protected_namespaces=("settings_",))

    verdict: Literal["AUTHENTIC", "FAKE"] | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    s3_url: str
    listing_id: str
    listing_status: Literal["live", "rejected", "pending"]
    status: Literal["AUTHENTIC", "FAKE"] | None
    model_decision: Literal["AUTHENTIC", "FAKE"] | None
    screening_code: str | None = None
    screening_reason: str | None = None
    publication_decision: Literal["PUBLISHED", "REJECTED", "BLOCKED"] = "BLOCKED"
    declared_brand: str
    brand_verification: Literal["NOT_PERFORMED"] = "NOT_PERFORMED"
    model: Literal["LEGACY_DINOV2"] = "LEGACY_DINOV2"
    model_status: Literal["LEGACY"] = "LEGACY"
    research_candidate: Literal[False] = False
    production_validation: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"


class ResearchVerifyResponse(BaseModel):
    """Research result inside a user-declared five-brand scope.

    `checkpoint_sha` is the frozen DINOv3 checkpoint for DINOv3 results and null for DINOv2.
    """

    model_config = ConfigDict(protected_namespaces=("settings_",))

    status: Literal["AUTHENTIC", "FAKE", "REVIEW"]
    decision: Literal["AUTHENTIC", "FAKE", "REVIEW"]
    declared_brand: str
    brand_verification: Literal["NOT_PERFORMED"] = "NOT_PERFORMED"
    research_only: Literal[True] = True
    production_ready: Literal[False] = False
    publication_decision: Literal["BLOCKED"] = "BLOCKED"
    checkpoint_sha: str | None
    policy_version: str
    model: Literal["DINOV3_RESEARCH_PROTOTYPE", "LEGACY_DINOV2"] = "DINOV3_RESEARCH_PROTOTYPE"
    model_version: str = "1"
    model_scope: Literal["FIVE_BRAND_RESEARCH_PROTOTYPE", "LEGACY_DINOV2"] = "FIVE_BRAND_RESEARCH_PROTOTYPE"
