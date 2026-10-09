"""Pydantic schemas for listings."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from database import ListingCategory, ListingStatus
from listings.customer_label import customer_label as label_for_listing
from listings.customer_label import evidence_role as role_for_listing


class ListingCreate(BaseModel):
    product_name: str = Field(..., max_length=512)
    category: ListingCategory
    brand: str | None = Field(default=None, max_length=256)
    condition: str | None = Field(default=None, max_length=128)
    size: str | None = Field(default=None, max_length=64)


class ListingUpdate(BaseModel):
    product_name: str | None = Field(default=None, max_length=512)
    category: ListingCategory | None = None
    brand: str | None = Field(default=None, max_length=256)
    condition: str | None = Field(default=None, max_length=128)
    size: str | None = Field(default=None, max_length=64)


class ListingRead(BaseModel):
    id: UUID
    seller_id: UUID
    product_name: str
    category: ListingCategory
    brand: str | None
    condition: str | None
    size: str | None
    s3_url: str | None
    verdict: str | None
    confidence: float | None
    status: ListingStatus
    created_at: datetime

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def customer_label(self) -> str:
        return label_for_listing(
            status=self.status,
            verdict=self.verdict,
            confidence=self.confidence,
            s3_url=self.s3_url,
        )

    @computed_field
    @property
    def evidence_role(self) -> str:
        return role_for_listing(
            verdict=self.verdict,
            confidence=self.confidence,
            s3_url=self.s3_url,
        )


class PresignRequest(BaseModel):
    listing_id: UUID
    filename: str = Field(..., max_length=255)
    content_type: str = Field(..., max_length=128)


class PresignResponse(BaseModel):
    url: str
    key: str
    dev: bool | None = None
