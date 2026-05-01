"""Import watch listings from a Chrono24-style CSV into Postgres (demo seller).

Bulk-scraping https://www.chrono24.in (or .com) for catalog data typically violates their
Terms of Service and breaks when markup changes. This script ingests **your own export**
(e.g. ``chrono24_unique_brand_plus_ref.csv``) so watches appear on HypeVault under the demo
seller account.

Usage (from repo root, DATABASE_URL set)::

    uv run python scripts/import_chrono24_csv.py \\
        --csv ~/Downloads/archive/chrono24_unique_brand_plus_ref.csv \\
        --limit 500

Optional: ``--dry-run`` to print counts only.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from config import settings  # noqa: E402
from database import Listing, ListingCategory, ListingStatus, User  # noqa: E402


def _truncate(s: str | None, max_len: int) -> str | None:
    if s is None:
        return None
    t = " ".join(str(s).split()).strip()
    if not t:
        return None
    return t[:max_len]


async def run(csv_path: Path, limit: int, seller_email: str, dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append({k: (v or "") for k, v in row.items()})

    if dry_run:
        print(f"Would import up to {len(rows)} rows from {csv_path}")
        await engine.dispose()
        return

    async with factory() as session:
        q = await session.execute(select(User).where(User.email == seller_email))
        seller = q.scalar_one_or_none()
        if seller is None:
            raise SystemExit(
                f"No user with email {seller_email!r}. Run scripts/seed_database.py first."
            )

        added = 0
        skipped = 0
        batch = 0
        for row in rows:
            title = _truncate(row.get("listing_title") or row.get("name"), 512)
            if not title:
                brand = (row.get("brand") or "").strip()
                model = (row.get("model") or "").strip()
                ref = (row.get("reference") or row.get("ref") or "").strip()
                title = _truncate(" ".join(p for p in (brand, model, ref) if p).strip(), 512)
            if not title:
                skipped += 1
                continue

            exists = await session.execute(select(Listing).where(Listing.product_name == title))
            if exists.scalar_one_or_none() is not None:
                skipped += 1
                continue

            brand = _truncate(row.get("brand"), 256)
            cond = _truncate(row.get("condition_chrono") or row.get("cond"), 128)
            size = _truncate(row.get("size"), 64)

            session.add(
                Listing(
                    seller_id=seller.id,
                    product_name=title,
                    category=ListingCategory.watch,
                    brand=brand,
                    condition=cond,
                    size=size,
                    verdict="AUTHENTIC",
                    confidence=0.94,
                    status=ListingStatus.live,
                    s3_url=None,
                )
            )
            added += 1
            batch += 1
            if batch >= 100:
                await session.commit()
                batch = 0

        if batch:
            await session.commit()

    await engine.dispose()
    print(f"Imported {added} watch listings (skipped {skipped} empty or duplicate titles).")


def main() -> None:
    default_csv = Path.home() / "Downloads" / "archive" / "chrono24_unique_brand_plus_ref.csv"
    p = argparse.ArgumentParser(description="Import Chrono24-format CSV watches into HypeVault.")
    p.add_argument("--csv", type=Path, default=default_csv, help="Path to CSV export")
    p.add_argument("--limit", type=int, default=400, help="Max rows to read from CSV")
    p.add_argument("--seller-email", default="seller@hypevault.demo", help="Seller user email")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    asyncio.run(run(args.csv, args.limit, args.seller_email, args.dry_run))


if __name__ == "__main__":
    main()
