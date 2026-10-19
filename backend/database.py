"""Async SQLAlchemy engine, session factory, and ORM models."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum as PyEnum
from typing import AsyncGenerator

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import settings


class Base(DeclarativeBase):
    pass


class UserRole(str, PyEnum):
    buyer = "buyer"
    seller = "seller"
    admin = "admin"  # granted only from the server-side admin_emails list, never self-assigned


class ListingCategory(str, PyEnum):
    # Legacy DB value — product is watches-only; API rejects new sneaker creates.
    sneaker = "sneaker"
    watch = "watch"


class ListingStatus(str, PyEnum):
    pending = "pending"
    live = "live"
    rejected = "rejected"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    listings: Mapped[list[Listing]] = relationship("Listing", back_populates="seller")


class Listing(Base):
    __tablename__ = "listings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    seller_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    product_name: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[ListingCategory] = mapped_column(Enum(ListingCategory), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(256), nullable=True)
    condition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size: Mapped[str | None] = mapped_column(String(64), nullable=True)
    s3_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[ListingStatus] = mapped_column(
        Enum(ListingStatus), nullable=False, default=ListingStatus.pending
    )
    # Why the listing is live, rejected or held (inference.screening). The reason is seller-facing.
    screening_code: Mapped[str | None] = mapped_column(String(48), nullable=True)
    screening_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    screening_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    seller: Mapped[User] = relationship("User", back_populates="listings")
    price_snapshots: Mapped[list[PriceSnapshot]] = relationship(
        "PriceSnapshot", back_populates="listing", cascade="all, delete-orphan"
    )


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    listing_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("listings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform: Mapped[str] = mapped_column(String(64), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    delivery_estimate: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seller_rating: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    listing: Mapped[Listing] = relationship("Listing", back_populates="price_snapshots")


class MarketListing(Base):
    """One observed market price for a watch, kept as history (a new row per observation)."""

    __tablename__ = "market_listings"
    __table_args__ = (
        UniqueConstraint("source", "external_id", "observed_on", name="uq_market_listing_observation"),
        Index("ix_market_listings_brand_reference", "brand", "reference"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    condition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    year: Mapped[str | None] = mapped_column(String(16), nullable=True)
    seller_ref: Mapped[str | None] = mapped_column(String(32), nullable=True)
    seller_feedback_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    item_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # When the price was seen at the source. NULL when the source does not record it (dataset rows).
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Day bucket used to keep one observation per item per day.
    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
