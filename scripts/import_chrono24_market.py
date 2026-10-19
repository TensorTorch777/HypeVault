"""Import a Chrono24 dataset CSV into market_listings (five supported brands only).

Chrono24 is not scraped; this loads an existing dataset file. Re-running is safe: rows are
de-duplicated per (source, external id, snapshot date).

    .venv/bin/python scripts/import_chrono24_market.py \
        --csv ~/Downloads/archive/chrono24_unique_brand_plus_ref.csv --snapshot-date 2023-09-16
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from database import AsyncSessionLocal, engine  # noqa: E402
from market.chrono24_dataset import file_sha256, read_items  # noqa: E402
from market.repository import save_observations  # noqa: E402


async def run(csv_path: Path, snapshot: date, dry_run: bool) -> None:
    items = list(read_items(csv_path, snapshot, dataset_sha256=file_sha256(csv_path)))
    by_brand = collections.Counter(i.brand for i in items)
    print(f"{len(items)} usable rows from {csv_path.name} (snapshot {snapshot}): {dict(by_brand)}")
    if dry_run:
        return
    written = 0
    async with AsyncSessionLocal() as session:
        for start in range(0, len(items), 500):
            written += await save_observations(session, items[start : start + 500])
    await engine.dispose()
    print(f"wrote {written} new rows (existing rows were left unchanged)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--snapshot-date", type=date.fromisoformat, required=True,
                        help="Date the dataset represents (file date if unknown)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.csv.expanduser(), args.snapshot_date, args.dry_run))


if __name__ == "__main__":
    main()
