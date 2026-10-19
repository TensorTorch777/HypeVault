"""Re-run listing screening on stored listing photos.

Listings screened before the wristwatch check existed (for example a selfie that sat in
"pending") are re-decided with the same pipeline as POST /verify/authenticate: size,
wristwatch present, original photograph. Each ends live or rejected with a reason.

Usage:
    python scripts/rescreen_listings.py                 # pending listings only
    python scripts/rescreen_listings.py --all           # every listing with a stored photo
    python scripts/rescreen_listings.py --all --dry-run # show decisions, change nothing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from sqlalchemy import select  # noqa: E402

from config import settings  # noqa: E402
from database import AsyncSessionLocal, Listing, ListingStatus  # noqa: E402
from inference.content_gate import score_watch  # noqa: E402
from inference.engine import classify_image  # noqa: E402
from inference.screening import check_content, check_origin, check_size  # noqa: E402
from inference.triton_client import preprocess_chw  # noqa: E402
from inference.verdict import apply_min_authentic_confidence  # noqa: E402
from s3_client import LOCAL_UPLOAD_DIR  # noqa: E402


def local_path(s3_url: str | None) -> Path | None:
    if not s3_url or "/static/local/" not in s3_url:
        return None
    path = LOCAL_UPLOAD_DIR / s3_url.rsplit("/static/local/", 1)[1]
    return path if path.is_file() else None


async def screen(listing: Listing, path: Path) -> None:
    pil = Image.open(path).convert("RGB")
    outcome = check_size(pil.width, pil.height, settings.listing_min_short_side_px)
    verdict = confidence = None
    if outcome is None:
        content = await asyncio.to_thread(score_watch, pil)
        outcome = check_content(content)
        if outcome is None:
            raw_verdict, raw_confidence = await classify_image(preprocess_chw(np.array(pil)))
            verdict, confidence = apply_min_authentic_confidence(
                raw_verdict, raw_confidence, settings.inference_min_authentic_confidence
            )
            outcome = check_origin(
                raw_verdict=raw_verdict,
                raw_confidence=float(raw_confidence),
                min_authentic_confidence=settings.inference_min_authentic_confidence,
                content=content,
            )
    listing.status = ListingStatus(outcome.status)
    listing.verdict = verdict
    listing.confidence = None if confidence is None else float(confidence)
    listing.screening_code = outcome.code
    listing.screening_reason = outcome.reason
    listing.screening_details = outcome.details


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="re-screen every listing with a stored photo, not only pending ones")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    async with AsyncSessionLocal() as db:
        q = select(Listing).order_by(Listing.created_at)
        if not args.all:
            q = q.where(Listing.status == ListingStatus.pending)
        listings = list((await db.execute(q)).scalars().all())
        changed = skipped = 0
        for listing in listings:
            path = local_path(listing.s3_url)
            if path is None:
                skipped += 1
                continue
            before = listing.status.value
            await screen(listing, path)
            changed += 1
            print(f"{listing.id}  {before:>8} → {listing.status.value:<8} {listing.screening_code:<24} {listing.product_name[:40]}")
            if listing.status != ListingStatus.live:
                print(f"    reason: {listing.screening_reason}")
        if args.dry_run:
            await db.rollback()
            print(f"\nDry run: {changed} re-screened, {skipped} without a stored photo. Nothing saved.")
        else:
            await db.commit()
            print(f"\n{changed} re-screened and saved, {skipped} without a stored photo left unchanged.")


if __name__ == "__main__":
    asyncio.run(main())
