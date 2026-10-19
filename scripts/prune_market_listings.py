"""Delete stored market observations that the current filters would no longer import.

    python scripts/prune_market_listings.py          # show what would go
    python scripts/prune_market_listings.py --apply  # delete it
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import delete, select  # noqa: E402

from database import AsyncSessionLocal, MarketListing  # noqa: E402
from market.types import excluded_listing  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(MarketListing.id, MarketListing.source, MarketListing.brand, MarketListing.title))).all()
        doomed = [r for r in rows if excluded_listing(r.title)]
        print(f"{len(doomed)} of {len(rows)} rows fail the current filters")
        print(Counter((r.source, r.brand) for r in doomed).most_common())
        if args.apply and doomed:
            await db.execute(delete(MarketListing).where(MarketListing.id.in_([r.id for r in doomed])))
            await db.commit()
            print("deleted")


if __name__ == "__main__":
    asyncio.run(main())
