"""Seed Postgres with demo users + luxury sneaker / watch listings (run with DATABASE_URL set).

These rows are **curated demo inventory**, not live web scraping. For production, ingest via
licensed feeds, seller uploads, or partner APIs — respect robots.txt and site terms.
"""

from __future__ import annotations

import asyncio
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from config import settings  # noqa: E402
from database import Listing, ListingCategory, ListingStatus, User, UserRole  # noqa: E402

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__default_rounds=12)

# (product_name, category, brand, condition, size)
DEMO_LISTINGS: list[tuple[str, ListingCategory, str, str, str]] = [
    # Sneakers — luxury designers
    ("Alexander McQueen Oversized Sneaker White", ListingCategory.sneaker, "Alexander McQueen", "New", "EU 42"),
    ("Alexander McQueen Tread Slick Boot", ListingCategory.sneaker, "Alexander McQueen", "Like new", "EU 41"),
    ("Balenciaga Triple S Grey", ListingCategory.sneaker, "Balenciaga", "New", "EU 43"),
    ("Balenciaga Track Sneaker Black", ListingCategory.sneaker, "Balenciaga", "New", "EU 42"),
    ("Dior B23 High-Top Oblique", ListingCategory.sneaker, "Dior", "New", "EU 42"),
    ("Dior B30 Mesh Runner", ListingCategory.sneaker, "Dior", "Excellent", "EU 41"),
    ("Gucci Rhyton Logo Leather", ListingCategory.sneaker, "Gucci", "New", "EU 42.5"),
    ("Gucci Ace Embroidered Bee", ListingCategory.sneaker, "Gucci", "Excellent", "EU 41"),
    ("Louis Vuitton Trainer Monogram", ListingCategory.sneaker, "Louis Vuitton", "New", "EU 9 US"),
    ("Louis Vuitton Skate Sneaker", ListingCategory.sneaker, "Louis Vuitton", "Like new", "EU 8 US"),
    # Watches — haute horlogerie
    ("A. Lange & Söhne Zeitwerk", ListingCategory.watch, "A. Lange & Söhne", "Unworn", "41.9mm"),
    ("A. Lange & Söhne Lange 1", ListingCategory.watch, "A. Lange & Söhne", "Excellent", "38.5mm"),
    ("Audemars Piguet Royal Oak 15500ST", ListingCategory.watch, "Audemars Piguet", "Excellent", "41mm"),
    ("Audemars Piguet Royal Oak Offshore", ListingCategory.watch, "Audemars Piguet", "New", "42mm"),
    ("Patek Philippe Nautilus 5711", ListingCategory.watch, "Patek Philippe", "Excellent", "40mm"),
    ("Patek Philippe Aquanaut 5167", ListingCategory.watch, "Patek Philippe", "New", "40.8mm"),
    ("Richard Mille RM 011", ListingCategory.watch, "Richard Mille", "Excellent", "40mm"),
    ("Richard Mille RM 35-02", ListingCategory.watch, "Richard Mille", "Unworn", "49.9mm"),
    ("Vacheron Constantin Overseas 4500V", ListingCategory.watch, "Vacheron Constantin", "New", "41mm"),
    ("Vacheron Constantin Patrimony", ListingCategory.watch, "Vacheron Constantin", "Excellent", "40mm"),
]


async def main() -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        email = "seller@hypevault.demo"
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none() is None:
            session.add(
                User(
                    email=email,
                    password_hash=pwd.hash("Seller123"),
                    role=UserRole.seller,
                )
            )
        email_b = "buyer@hypevault.demo"
        existing_b = await session.execute(select(User).where(User.email == email_b))
        if existing_b.scalar_one_or_none() is None:
            session.add(
                User(
                    email=email_b,
                    password_hash=pwd.hash("Buyer123"),
                    role=UserRole.buyer,
                )
            )
        await session.commit()

        q = await session.execute(select(User).where(User.email == "seller@hypevault.demo"))
        seller = q.scalar_one()

        for name, cat, brand, condition, size in DEMO_LISTINGS:
            exists = await session.execute(select(Listing).where(Listing.product_name == name))
            if exists.scalar_one_or_none() is not None:
                continue
            session.add(
                Listing(
                    seller_id=seller.id,
                    product_name=name,
                    category=cat,
                    brand=brand,
                    condition=condition,
                    size=size,
                    verdict="AUTHENTIC",
                    confidence=0.965,
                    status=ListingStatus.live,
                    s3_url=None,
                )
            )
        await session.commit()
    await engine.dispose()
    print("Seed complete: seller@hypevault.demo / Seller123, buyer@hypevault.demo / Buyer123")
    print(f"Ensured up to {len(DEMO_LISTINGS)} demo listings (skipped if product_name already exists).")


if __name__ == "__main__":
    asyncio.run(main())
