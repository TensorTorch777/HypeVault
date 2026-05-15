"""Inference request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AuthenticateResponse(BaseModel):
    verdict: Literal["AUTHENTIC", "FAKE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    s3_url: str
    listing_id: str
    # Mirrors `Listing.status` after inference (live if model clears the confidence gate).
    listing_status: Literal["live", "rejected", "pending"]
