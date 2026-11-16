"""Read a Chrono24 dataset export (CSV) into market items.

The dataset is a third-party snapshot of public Chrono24 listings. It records no capture date,
so `observed_at` stays empty and `observed_on` is the snapshot date given by the caller.
Chrono24's terms do not allow automated scraping, so this importer is the only Chrono24 path.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path
from typing import Iterator

from market.types import SOURCE_CHRONO24_DATASET, MarketItem, canonical_brand, excluded_listing, parse_price


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_items(path: Path, snapshot_date: date, *, dataset_sha256: str | None = None) -> Iterator[MarketItem]:
    """Yield one item per usable row for the five supported brands."""
    provenance = {
        "dataset_file": path.name,
        "dataset_sha256": dataset_sha256 or file_sha256(path),
        "snapshot_date_basis": "dataset file date; listing capture dates are not recorded",
        "terms": "imported from an existing dataset; Chrono24 is not scraped",
    }
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            brand = canonical_brand(row.get("brand"))
            price = parse_price(row.get("price_example") or row.get("price"))
            title = (row.get("listing_title") or row.get("name") or "").strip()
            if brand is None or price is None or not title or excluded_listing(title):
                continue
            reference = (row.get("reference") or row.get("ref") or "").strip() or None
            key = "|".join([brand, reference or "", title, f"{price:.2f}"])
            yield MarketItem(
                source=SOURCE_CHRONO24_DATASET,
                external_id=hashlib.sha1(key.encode("utf-8")).hexdigest()[:32],
                brand=brand,
                model=(row.get("model") or "").strip()[:256] or None,
                reference=reference[:128] if reference else None,
                title=title[:512],
                price=price,
                currency="USD",
                condition=(row.get("condition_chrono") or row.get("cond") or "").strip()[:128] or None,
                year=(row.get("year") or "").strip()[:16] or None,
                observed_on=snapshot_date,
                provenance=provenance,
            )
