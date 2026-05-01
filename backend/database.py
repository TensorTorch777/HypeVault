"""Async SQLAlchemy engine, session factory, and ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import AsyncGenerator

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import settings


class Base(DeclarativeBase):
    pass


class UserRole(str, PyEnum):
    buyer = "buyer"
    seller = "seller"


class ListingCategory(str, PyEnum):
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
